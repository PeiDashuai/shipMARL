import numpy as np

def tcpa_dcpa_matrix(ships):
    """计算 TCPA/DCPA 矩阵与速度向量表。"""
    N = len(ships)
    tc = np.full((N, N), np.inf, dtype=np.float64)
    dc = np.full((N, N), np.inf, dtype=np.float64)
    v_list = [np.array([np.cos(s.psi), np.sin(s.psi)]) * s.v for s in ships]
    for i in range(N-1):
        for j in range(i+1, N):
            pij = ships[j].pos - ships[i].pos
            vij = v_list[j] - v_list[i]
            dv2 = float(np.dot(vij, vij))
            if dv2 < 1e-12:
                tstar = 0.0
                dmin = float(np.linalg.norm(pij))
            else:
                tstar = - float(np.dot(pij, vij)) / dv2
                if tstar < 0: tstar = 0.0
                dmin = float(np.linalg.norm(pij + tstar * vij))
            tc[i,j] = tc[j,i] = tstar
            dc[i,j] = dc[j,i] = dmin
    return tc, dc, v_list
