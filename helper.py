import numpy as np


def hat(x):
    """
    skew-symmetric matrix of x.
    """
    return np.array([
        [    0.0, -x[2],  x[1]],
        [ x[2],     0.0, -x[0]],
        [-x[1],  x[0],    0.0]
    ], dtype=float)


def rotation_matrix(theta):
    """
    Constructs a rotation matrix around the z-axis by angle theta.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ], dtype=float)

def Rtheta_dot(theta):
    """
    Computes the partial derivative of the rotation matrix Rtheta with respect to theta.

    Parameters:
    -----------
    theta : float
        Rotation angle about the z-axis.

    Returns:
    --------
    dRtheta_dtheta : ndarray
        3x3 partial derivative matrix.
    """
    return np.array([
        [-np.sin(theta), -np.cos(theta), 0],
        [ np.cos(theta), -np.sin(theta), 0],
        [0,              0,             0]
    ], dtype=float)

