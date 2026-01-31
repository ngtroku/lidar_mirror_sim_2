
import numpy as np

def eigen_value_decomposition_2d(points):
    points_xy = points[:, :2]
    Sigma = np.cov(points_xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    idx = eigenvalues.argsort()[::-1]
    lam = eigenvalues[idx]      # λ1, λ2
    v = eigenvectors[:, idx]    # v1, v2

    anisotropy_2d = (lam[0] - lam[1]) / lam[0]
    angle_rad = np.arctan2(v[1, 0], v[0, 0])
    angle_deg = np.degrees(angle_rad)

    return anisotropy_2d, angle_deg