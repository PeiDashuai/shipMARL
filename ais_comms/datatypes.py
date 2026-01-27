from __future__ import annotations

"""
================================================================================
ais_comms.datatypes — SEMANTICS CONTRACT (DO NOT CHANGE LIGHTLY)
================================================================================

This module defines the *shared* datatypes used by:
  - AISCommsSim (ais_comms.py)
  - AISTrackManagerPF (track_manager_pf.py)
  - MiniShipAISCommsEnv wrapper (miniship/envs/_ais_comms_env_impl.py)
  - Staging writers (stage3/4)

Key angle conventions used in this project:

(A1) reported_cog / yaw_east_ccw_rad:
    - Meaning: direction of velocity vector in ENU/world frame.
    - 0 rad at +x (East), positive counter-clockwise.
    - Range: (-pi, pi], Unit: radians.
    - In code: RxMsg.reported_cog, TrueState.yaw_east_ccw_rad (alias: yaw_sim_rad)

This is NOT the nautical COG convention (North=0, clockwise, degrees).

Time conventions:
  - t: simulation time in seconds (float), consistent across env, AISCommsSim, and PF.
================================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict

ShipId = int
AgentId = str
Ts = float


@dataclass(frozen=True, slots=True)
class RawTxMsg:
    """
    Raw message emitted by a ship's AIS transmitter (before channel effects).
    """
    ship_id: ShipId
    t_tx: Ts
    x: float
    y: float
    sog: float
    cog: float  # yaw_east_ccw_rad (ENU, rad)
    meta: Dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RxMsg:
    """
    Received AIS message after channel effects (drop/delay/noise).

    reported_* fields are the *received* values (may be noisy).
    """
    ship_id: ShipId
    arrival_t: Ts
    reported_t: Ts
    reported_x: float
    reported_y: float
    reported_sog: float
    reported_cog: float  # yaw_east_ccw_rad (ENU, rad)
    meta: Dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrueState:
    """
    Ground-truth kinematic state used by AISCommsSim and PF for evaluation.

    Required fields:
      - x, y, vx, vy are ENU/world coordinates.
      - yaw_east_ccw_rad is atan2(vy, vx) in ENU (East=0, CCW+), unit rad.
      - t is simulation time (seconds).
    """
    ship_id: ShipId
    t: Ts
    x: float
    y: float
    vx: float
    vy: float
    yaw_east_ccw_rad: float

    @property
    def yaw_sim_rad(self) -> float:
        return self.yaw_east_ccw_rad


@dataclass(frozen=True, slots=True)
class ShipState:
    """
    Convenience state container used by environment wrapper.

    Note: this is not required by AISCommsSim; it exists to keep env code explicit.
    """
    ship_id: ShipId
    x: float
    y: float
    vx: float
    vy: float
    yaw_east_ccw_rad: float

    @property
    def yaw_sim_rad(self) -> float:
        return self.yaw_east_ccw_rad


@dataclass(frozen=True, slots=True)
class WorldState:
    """
    Convenience world snapshot: mapping ship_id -> TrueState.
    """
    t: Ts
    ships: Dict[ShipId, TrueState]
