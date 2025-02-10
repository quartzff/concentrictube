from get_tube_info_i import define_breakpoints, get_current_segment, get_tube_info_i
import numpy as np
from helper import hat, rotation_matrix, Rtheta_dot
def deriv(segment_index, y, params):
    """
    Computes the derivative dy/ds for the Cosserat rod.

    Parameters:
    -----------
    s : float
        Current arc length.
    y : ndarray
        Current state vector.
    params : dict
        Dictionary containing all necessary parameters:
            - 'tubes': List of tube dictionaries with properties
            - 'F_tip': External force at the tip
            - 'Break_Points': List of breakpoints
            - Other parameters as needed

    Returns:
    --------
    dy_ds : ndarray
        Derivative of the state vector.
    """
    # Extract tubes and breakpoints
    tubes = params['tubes']
    Break_Points = params['Break_Points']

    # Retrieve tube properties for the current segment
    K_list, u_star_list, u_star_dot_list = get_tube_info_i(segment_index, tubes, Break_Points)
    n = len(tubes)  # Number of tubes

    # Extract position
    p = y[0:3]  # [x, y, z]

    # Extract and reshape orientation matrix
    R_flat = y[3:12]
    R = R_flat.reshape((3, 3), order='F')  # 3x3 rotation matrix

    # Extract reference tube curvature
    u1x, u1y, u1z = y[12:15]
    u1 = np.array([u1x, u1y, u1z])


    # Extract additional tubes' torsions and angles
    if n > 1:
        uz_list = y[15:15 + (n - 1)]  # [u2z, u3z, ..., unz]
        theta_list = y[15 + (n - 1):15 + 2 * (n - 1)]  # [theta2, theta3, ..., thetan]
    else:
        uz_list = []
        theta_list = []

    # Compute rotation matrices for additional tubes
    Rtheta_list = [rotation_matrix(theta) for theta in theta_list]
    ParRtheta_list = [Rtheta_dot(theta) for theta in theta_list]

    e3 = np.array([0.0, 0.0, 1.0])

    # Compute combined stiffness matrix
    K = sum(K_list)
    invK = np.linalg.inv(K)

    # Compute theta_dot for additional tubes
    theta_dot_list = [uz - u1z for uz in uz_list]

    # Compute local curvatures for additional tubes
    u_list = []
    for i in range(n - 1):
        Rtheta_i_T = Rtheta_list[i].T
        u_i = Rtheta_i_T @ u1 + theta_dot_list[i] * e3
        u_list.append(u_i)

    # Compute hat matrices
    u1hat = hat(u1)
    u_hat_list = [hat(u_i) for u_i in u_list]

    # Compute p_dot and R_dot
    p_dot = R @ e3
    R_dot = R @ u1hat

    # Initialize K_diff_list with K1 * (u1 - u1_star)
    K_diff_list = [K_list[0] @ (u1 - u_star_list[0])]

    # Compute K_i * (u_i - u_i_star) for tubes 2 to n and append to K_diff_list
    for i in range(1, n):
        K_diff = K_list[i] @ (u_list[i - 1] - u_star_list[i])
        K_diff_list.append(K_diff)

    # Compute partial derivatives of rotation matrices
    PR_Tu1_list = [ParRtheta_list[i].T @ u1 for i in range(n - 1)]

    # Compute the moment contributions
    moment_contributions = K_list[0] @ (-u_star_dot_list[0]) + u1hat @ K_diff_list[0]

    for i in range(n - 1):
        term = (K_list[i + 1] @ (theta_dot_list[i] * PR_Tu1_list[i] - u_star_dot_list[i + 1]) +
                u_hat_list[i] @ K_diff_list[i + 1])
        moment_contributions += Rtheta_list[i] @ term

    # Add external tip force contribution
    moment_contributions += hat(e3) @ (R.T @ params['F_tip'])

    # Compute u1xy_dot
    u1xy_dot = (np.array([[1, 0, 0],
                          [0, 1, 0]]) @ (-invK) @ moment_contributions).flatten()

    # Compute u_z_dot for reference tube
    E1I1, G1J1 = tubes[0]['E_I'], tubes[0]['G_J']
    u1z_dot = u_star_dot_list[0][2] + (E1I1 / G1J1) * (u1[0] * u_star_list[0][1] - u1[1] * u_star_list[0][0])
    u_star_dot = u_star_dot_list[0][2]

    # Compute u_z_dot for additional tubes using u1_star_dot_z (All use Tube 1's derivative)
    uz_dot_list = []
    if n > 1:
        for i in range(n - 1):
            EiIi = tubes[i + 1]['E_I']
            GiJi = tubes[i + 1]['G_J']
            u_i = u_list[i]
            u_star_i = u_star_list[i + 1]
            # Use u1_star_dot_z for all tubes
            u_i_dot = u_star_dot + (EiIi / GiJi) * (u_i[0] * u_star_i[1] - u_i[1] * u_star_i[0])
            uz_dot_list.append(u_i_dot)

    # Compute s_dot
    s_dot = 1.0

    # Assemble dy/ds
    dy_ds = np.zeros_like(y)

    # Position derivatives
    dy_ds[0:3] = p_dot

    # Orientation derivatives (flattened)
    dy_ds[3:12] = R_dot.reshape(9, order='F')  # MATLAB column-major equivalent

    # Curvature derivatives for reference tube
    dy_ds[12:14] = u1xy_dot  # [du1x/ds, du1y/ds]
    dy_ds[14] = u1z_dot       # du1z/ds

    # Curvature derivatives for additional tubes
    if n > 1:
        dy_ds[15:15 + (n - 1)] = uz_dot_list  # [du2z/ds, ..., dunz/ds]

        # Relative angle derivatives
        dy_ds[15 + (n - 1):15 + 2 * (n - 1)] = theta_dot_list  # [dtheta2/ds, ..., dthetan/ds]

    # Arc length derivative
    dy_ds[-1] = s_dot



    return dy_ds