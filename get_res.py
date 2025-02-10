import numpy as np
from get_tube_info_i import define_breakpoints, get_tube_info_i
from run_ode import run_ode

def get_res(u_init_guess, params):
    """
    Computes the residual error of the boundary conditions for a given guess
    of the unknown initial conditions.

    Parameters:
    -----------
    u_init_guess : ndarray
        Initial guess for the unknown initial conditions.
    params : dict
        Dictionary containing all necessary parameters, including:
            - tubes: List of tube properties
            - F_tip: Tip force
            - L_tip: Tip moment

    Returns:
    --------
    res : ndarray
        Residual errors for the boundary conditions.
    """
    # Extract parameters
    tubes = params['tubes']
    F_tip = params['F_tip']
    L_tip = params['L_tip']

    # Dynamically define breakpoints from the tube properties
    Break_Points = define_breakpoints(tubes)

    # Run ODE to solve for the states
    s, y = run_ode(u_init_guess, {**params, 'Break_Points': Break_Points})
    # Extract state variables at the tip
    u1_tip = y[-1, 12:15]  # u1x, u1y, u1z at the tip
    R_tip = y[-1, 3:12].reshape((3, 3))  # Rotation matrix at the tip

    # Get tube stiffness and curvature at the last segment
    end_index = len(Break_Points) - 1
    K_list, u_star_list, _ = get_tube_info_i(end_index-1, tubes, Break_Points)

    # Only the innermost tube (Tube 1) affects boundary condition moments
    K1 = K_list[0]
    u1_star_tip = u_star_list[0]
    try:
        invK1 = np.linalg.inv(K1)
    except np.linalg.LinAlgError:
        print("Singular K1 matrix at tip, using pseudo-inverse.")
        invK1 = np.linalg.pinv(K1)

    # Compute moment residual for the innermost tube
    moment_res = u1_tip - u1_star_tip - invK1 @ R_tip.T @ L_tip

    # Compute torsional residuals for all additional tubes
    torsion_res = []
    for i in range(1, len(tubes)):  # Skip tube 1 since torsional residual is not computed for it
        uz_tip = y[-1, 14 + i]  # Extract torsional component at the tip
        uz_star_tip = u_star_list[i][2]  # Extract precurvature's z-component
        torsion_res.append(uz_tip - uz_star_tip)

    # Combine all residuals
    res = np.hstack([moment_res, torsion_res])
    return res

