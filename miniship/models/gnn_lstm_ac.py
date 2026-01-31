# miniship/models/gnn_lstm_ac.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


class MPNNLayer(nn.Module):
    """
    单层 Message-Passing GNN：
      h_i^(l+1) = φ_v( [ h_i^(l),  Σ_j φ_e( h_i^(l), h_j^(l), e_ij ) ] )
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # 消息函数 φ_e: [h_i, h_j, e_ij] -> m_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
        )

        # 节点更新 φ_v: [h_i, m_i] -> h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h_self: torch.Tensor,      # [B*, D]
        h_nei: torch.Tensor,       # [B*, K, D]
        e_ij: torch.Tensor,        # [B*, K, E]
        mask: torch.Tensor,        # [B*, K]  (1 有效邻居, 0 padding)
    ) -> torch.Tensor:
        B, K, D = h_nei.shape
        E = e_ij.shape[-1]

        # 扩展自船节点到每个邻居 slot: [B,K,D]
        h_i_exp = h_self.unsqueeze(1).expand(B, K, D)

        # 拼接 [h_i, h_j, e_ij] -> [B,K,2D+E]
        m_in = torch.cat([h_i_exp, h_nei, e_ij], dim=-1)

        # 消息 MLP: [B,K,D]
        m_ij = self.edge_mlp(m_in)

        # mask 掉 padding 邻居
        m_ij = m_ij * mask.unsqueeze(-1)

        # sum 聚合（置换不变）：[B,D]
        m_i = m_ij.sum(dim=1)

        # 节点更新: [h_i, m_i] -> [B,D]
        h_out = self.node_mlp(torch.cat([h_self, m_i], dim=-1))
        return h_out


class MiniShipGNNLSTMActorCritic(TorchModelV2, nn.Module):
    """
    MiniShip 专用：MPNN + LSTM Actor-Critic 模型（多船 & 序列）

    输入 obs_flat 结构（与 builder.py 对齐）：
      - self 6 维: [ v/Vm, cos(psi), sin(psi), r_goal/Rn, cos(theta_goal), sin(theta_goal) ]
      - neighbor K*8 维: 每个邻居 8 维节点特征
      - edge     K*8 维: 每个邻居 8 维边特征
      - id 1 维: 归一化自 ID

    总长度 = 6 + K*8 + K*8 + 1
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # RLlib passes custom_model_config inside model_config, not as a separate kwarg
        custom_cfg: Dict[str, Any] = model_config.get("custom_model_config", {})
        if not custom_cfg:
            # Fallback to kwargs for backward compatibility
            custom_cfg = kwargs.get("custom_model_config", {})

        # 先确定 K / 邻居 / 边 / id 维度（这些我们在 builder 里是固定设计好的）
        # Default num_neighbors=4 matches env default (miniship_core_env.py:135 numNeighbors)
        self.num_neighbors: int = int(custom_cfg.get("num_neighbors", 4))
        self.neighbor_dim: int = int(custom_cfg.get("neighbor_dim", 11))
        self.edge_dim: int = int(custom_cfg.get("edge_dim", 8))
        self.id_dim: int = int(custom_cfg.get("id_dim", 1))
        self.self_dim = int(custom_cfg.get("self_dim", -1))

        # 序列长度（RLlib 会据此打包 RNN batch）
        self.max_seq_len: int = int(
            model_config.get("max_seq_len", custom_cfg.get("max_seq_len", 64))
        )

        # GNN 嵌入维度（节点表示维度）
        self.gnn_hidden_size: int = int(custom_cfg.get("gnn_hidden_size", 128))

        # ---------- 自动推断 self_dim ----------
        obs_dim = int(obs_space.shape[0])
        base_dim = (
            self.num_neighbors * self.neighbor_dim
            + self.num_neighbors * self.edge_dim
            + self.id_dim
        )
        auto_self_dim = obs_dim - base_dim



        # Hard contract check (fail-fast).
        if auto_self_dim <= 0:
            raise ValueError(
                f"[GNNLSTM] invalid dims: obs_dim={obs_dim}, base_dim={base_dim}, "
                f"K={self.num_neighbors}, nei_dim={self.neighbor_dim}, edge_dim={self.edge_dim}, id_dim={self.id_dim}"
            )

        # If self_dim explicitly provided, enforce exact match.
        if self.self_dim > 0 and self.self_dim != auto_self_dim:
            raise ValueError(
                f"[GNNLSTM] self_dim mismatch: cfg self_dim={self.self_dim} but inferred={auto_self_dim}. "
                f"obs_dim={obs_dim}, K={self.num_neighbors}, nei_dim={self.neighbor_dim}, edge_dim={self.edge_dim}, id_dim={self.id_dim}"
            )

        cfg_self_dim = custom_cfg.get("self_dim", None)
        if cfg_self_dim is None:
            # 完全由 obs_space 推断
            self.self_dim = auto_self_dim
        else:
            cfg_self_dim = int(cfg_self_dim)
            if cfg_self_dim != auto_self_dim:
                print(
                    "[MiniShipGNNLSTMActorCritic] WARNING: self_dim from config "
                    f"({cfg_self_dim}) != inferred ({auto_self_dim}) from obs_space; "
                    "using inferred value."
                )
                self.self_dim = auto_self_dim
            else:
                self.self_dim = cfg_self_dim

        # ---------- obs 维度 sanity check ----------
        flat_dim = obs_dim
        expected_dim = (
            self.self_dim
            + self.num_neighbors * self.neighbor_dim
            + self.num_neighbors * self.edge_dim
            + self.id_dim
        )
        if flat_dim != expected_dim:
            raise ValueError(
                f"[MiniShipGNNLSTMActorCritic] obs_dim={flat_dim} != expected_dim={expected_dim} "
                f"(self={self.self_dim}, K={self.num_neighbors}, "
                f"F_nei={self.neighbor_dim}, F_edge={self.edge_dim}, id={self.id_dim})"
            )


        # ---------- 节点初始化 ----------
        # 自船节点: self(6) + id(1) -> gnn_hidden_size
        self.self_init = nn.Sequential(
            nn.Linear(self.self_dim + self.id_dim, self.gnn_hidden_size),
            nn.ReLU(),
            nn.Linear(self.gnn_hidden_size, self.gnn_hidden_size),
            nn.ReLU(),
        )

        # 邻船节点: neighbor_dim(8) -> gnn_hidden_size
        self.neigh_init = nn.Sequential(
            nn.Linear(self.neighbor_dim, self.gnn_hidden_size),
            nn.ReLU(),
            nn.Linear(self.gnn_hidden_size, self.gnn_hidden_size),
            nn.ReLU(),
        )

        # Skip connection projection: self_feat -> gnn_hidden_size
        # This preserves goal direction signal which is critical for learning
        self.self_skip = nn.Linear(self.self_dim, self.gnn_hidden_size)

        # ---------- 两层 MPNN ----------
        self.gnn1 = MPNNLayer(
            node_dim=self.gnn_hidden_size,
            edge_dim=self.edge_dim,
        )
        self.gnn2 = MPNNLayer(
            node_dim=self.gnn_hidden_size,
            edge_dim=self.edge_dim,
        )

        self.gnn_out_dim: int = self.gnn_hidden_size

        # ---------- LSTM 序列建模 ----------
        self.lstm_hidden_size: int = int(custom_cfg.get("lstm_hidden_size", 128))
        self.lstm = nn.LSTM(
            input_size=self.gnn_out_dim,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,  # [B, T, H]
        )

        # ---------- Actor-Critic 头 ----------
        num_actions = int(action_space.n) if hasattr(action_space, "n") else num_outputs

        self.policy_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._value_out: torch.Tensor | None = None

    # --------------------------------------------------
    # RLlib RNN 接口：初始 state
    # --------------------------------------------------
    def get_initial_state(self) -> List[torch.Tensor]:
        h = torch.zeros(self.lstm_hidden_size)
        c = torch.zeros(self.lstm_hidden_size)
        return [h, c]

    # --------------------------------------------------
    # 解析 obs_flat -> self + neighbors + edges + id
    # --------------------------------------------------
    def _split_obs(
        self, obs_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        严格按照:
        [ self(self_dim) |
            K * neighbor(neighbor_dim) |
            K * edge(edge_dim) |
            id(id_dim) ]
        来切割。
        """
        B_T, D = obs_flat.shape
        expected_dim = (
            self.self_dim
            + self.num_neighbors * self.neighbor_dim
            + self.num_neighbors * self.edge_dim
            + self.id_dim
        )
        if D != expected_dim:
            raise RuntimeError(
                f"[MiniShipGNNLSTMActorCritic._split_obs] got D={D}, expected={expected_dim}"
            )

        idx = 0
        self_feat = obs_flat[:, idx: idx + self.self_dim]
        idx += self.self_dim

        neigh_size = self.num_neighbors * self.neighbor_dim
        neigh_flat = obs_flat[:, idx: idx + neigh_size]
        idx += neigh_size

        edge_size = self.num_neighbors * self.edge_dim
        edge_flat = obs_flat[:, idx: idx + edge_size]
        idx += edge_size

        id_feat = obs_flat[:, idx: idx + self.id_dim]

        neigh_feat = neigh_flat.view(B_T, self.num_neighbors, self.neighbor_dim)
        edge_feat = edge_flat.view(B_T, self.num_neighbors, self.edge_dim)

        return self_feat, neigh_feat, edge_feat, id_feat

    # --------------------------------------------------
    # [B*T, H] -> [B, T, H]
    # --------------------------------------------------
    @staticmethod
    def _add_time_dim(
        x: torch.Tensor,
        seq_lens: torch.Tensor | None,
        max_seq_len: int,
    ) -> torch.Tensor:
        B_T, H = x.shape

        if seq_lens is not None and len(seq_lens) > 0:
            B = int(seq_lens.shape[0])
            T = int(B_T // B) if B > 0 else max_seq_len
        else:
            B = B_T
            T = 1

        if B * T != B_T:
            return x.unsqueeze(0)

        return x.view(B, T, H)

    # --------------------------------------------------
    # forward: obs_flat -> logits, new_state
    # --------------------------------------------------
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        input_dict["obs_flat"]: [B*T, D]
        state: [h, c]，每个 [B, H]
        seq_lens: [B]
        """
        obs: torch.Tensor = input_dict["obs_flat"].float()  # [B*T, D]

        # ---------- GNN: 解析观测 ----------
        self_feat, neigh_feat, edge_feat, id_feat = self._split_obs(obs)
        B_T = self_feat.shape[0]

        # 自船节点初始化: [B*T, D_gnn]
        self_in = torch.cat([self_feat, id_feat], dim=-1)
        h_self = self.self_init(self_in)

        # 邻船节点初始化: [B*T, K, D_gnn]
        h_nei = self.neigh_init(neigh_feat)

        # mask：使用邻居的 valid 标志（neighbor_dim 的最后一个元素）
        # neighbor feature layout: [n8(8), u_stale(1), u_silence(1), valid(1)] = 11 dims
        # valid=1 表示有效邻居，valid=0 表示无效/padding
        # 旧的实现 `neigh_feat.abs().sum() > 0` 有 bug：
        #   当 ais_valid=False 时，n8=0 但 u_stale=1, u_silence=1，sum > 0 导致 mask=1
        #   这会让无效邻居参与消息传递，影响学习
        mask = neigh_feat[:, :, -1]  # [B*T, K]，直接使用 valid 标志

        # 第一层 MPNN
        h1 = self.gnn1(h_self, h_nei, edge_feat, mask)

        # 第二层 MPNN（此处简单地仍使用初始邻居表示 h_nei；若要更完整的图更新可以后续扩展）
        h2 = self.gnn2(h1, h_nei, edge_feat, mask)

        # Skip connection: preserve self_feat (especially goal direction g_fwd_norm, g_lat_norm)
        # Without this, goal signal gets diluted through MPNN layers and model can't learn goal-seeking
        skip_feat = self.self_skip(self_feat)  # [B*T, D_gnn]
        gnn_emb = h2 + skip_feat  # Residual connection

        # ---------- LSTM：时间维 ----------
        lstm_in = self._add_time_dim(gnn_emb, seq_lens, self.max_seq_len)  # [B, T, D_gnn]
        B = lstm_in.shape[0]

        if state and len(state) == 2 and state[0] is not None:
            h_in = state[0].view(1, B, self.lstm_hidden_size)
            c_in = state[1].view(1, B, self.lstm_hidden_size)
        else:
            h_in = torch.zeros(1, B, self.lstm_hidden_size, device=lstm_in.device)
            c_in = torch.zeros(1, B, self.lstm_hidden_size, device=lstm_in.device)

        lstm_out, (h_n, c_n) = self.lstm(lstm_in, (h_in, c_in))  # [B,T,H]

        # 展平回 [B*T, H]，与 obs_flat 对齐
        lstm_out_flat = lstm_out.reshape(-1, self.lstm_hidden_size)  # [B*T,H]

        # ---------- Actor / Critic ----------
        logits = self.policy_head(lstm_out_flat)            # [B*T, num_actions]
        value = self.value_head(lstm_out_flat).squeeze(-1)  # [B*T]

        self._value_out = value

        new_state = [
            h_n.view(B, self.lstm_hidden_size).detach(),
            c_n.view(B, self.lstm_hidden_size).detach(),
        ]
        return logits, new_state

    # --------------------------------------------------
    # RLlib 要求的 value_function 接口
    # --------------------------------------------------
    def value_function(self) -> torch.Tensor:
        assert self._value_out is not None, "value_function called before forward()"
        return self._value_out


# ==================================================
# 注册函数：供训练脚本调用
# ==================================================
def register_miniship_gnn_lstm_model():
    """
    在 Ray RLlib 的 ModelCatalog 中注册自定义模型。
    训练脚本中使用:
        model = {"custom_model": "miniship_gnn_lstm_ac", ...}
    """
    ModelCatalog.register_custom_model(
        "miniship_gnn_lstm_ac",
        MiniShipGNNLSTMActorCritic,
    )
