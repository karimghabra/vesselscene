"""stub"""
import numpy as np
THROUGH_COS = 0.8
def type_from_arms(dirs, through_cos=THROUGH_COS):
    dirs = np.asarray(dirs, float).reshape(-1, 2)
    n = len(dirs)
    if n == 3:
        return "pseudo-T"
    if n == 4:
        u = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        best = min(max(u[a] @ u[b], u[c] @ u[d]) for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)))
        return "crossing" if best <= -through_cos else "compound"
    return "compound"
