import numpy as np
from scipy.integrate import solve_ivp
from deriv import deriv
from get_tube_info_i import define_breakpoints, get_tube_info_i


def run_ode(u_init_guess, params):
    """
    Integrates the Cosserat rod ODE over successive segments defined by Break_Points.
    Should work with any number of tubes.

    State vector layout:
      [ p (3),
        R (flattened, 9),
        u1x, u1y, u1z (3) for reference tube,
        uz_2 ... uz_n (n-1),
        theta_2 ... theta_n (n-1),
        s_arc (1) ]
    """
    # Extract parameters
    tubes = params['tubes']
    Break_Points = np.unique(params['Break_Points'])  # sorted unique breakpoints
    num_tubes = len(tubes)
    alphas = params.get('alphas', [0.0] * num_tubes)

    # Unpack initial guess
    u1x_init, u1y_init, u1z_init = u_init_guess[0:3]
    uz_inits = u_init_guess[3:3 + (num_tubes - 1)]  # additional tubes' twist values

    # Use each tube's insertion depth from its dict:
    Ds = [tube['D'] for tube in tubes]

    # Initial position and rotation:
    p_init = np.array([0.0, 0.0, 0.0])
    alpha0 = alphas[0]
    D0 = Ds[0]
    R_init = np.array([
        [np.cos(alpha0 + D0 * u1z_init), -np.sin(alpha0 + D0 * u1z_init), 0],
        [np.sin(alpha0 + D0 * u1z_init), np.cos(alpha0 + D0 * u1z_init), 0],
        [0, 0, 1]
    ])

    # Initial relative angles for additional tubes (tube i: i=2..n)
    thetas_init = []
    for i in range(1, num_tubes):
        thetas_init.append(alphas[i] + Ds[i] * uz_inits[i - 1] - (alpha0 + D0 * u1z_init))

    # Construct initial state vector as 1D array
    y_init = np.hstack([
        p_init,  # 3
        R_init.flatten(order='F'),  # 9 (column-major flattening)
        [u1x_init, u1y_init, u1z_init],  # 3
        uz_inits,  # (num_tubes-1)
        thetas_init,  # (num_tubes-1)
        [0.0]  # arc length
    ])

    # Integration settings
    s = [0.0]
    y_list = [y_init]
    delta = 1e-5
    options = {'atol': 1e-2, 'rtol': 1e-1, 'max_step': 0.005}

    # Main loop: integrate over each segment defined by Break_Points
    for i_seg in range(len(Break_Points) - 1):
        s_start, s_end = Break_Points[i_seg], Break_Points[i_seg + 1]
        if abs(s_end - s_start) < 2 * delta:
            print(f"Segment {i_seg} is too small. Skipping.")
            continue

        # Solve ODE for this segment; pass i_seg as the segment index to deriv
        sol = solve_ivp(
            lambda s_val, y_val: deriv(i_seg, y_val, params),
            [s_start, s_end],
            y_init,
            t_eval=np.linspace(s_start, s_end, 11),
            **options
        )

        if not sol.success:
            print(f"ODE solver failed at segment {i_seg}: {sol.message}")
            break

        # Append the current segment's solution
        s.extend(sol.t)
        y_list.extend(sol.y.T.tolist())  # Convert to list for dynamic handling

        # --- Transition Conditions ---
        # Let y_old be the final state from this segment:
        y_old = y_list[-1]
        # Extract position and rotation:
        p = y_old[0:3]
        R = np.array(y_old[3:12]).reshape(3, 3, order='F')
        # Extract reference tube curvature (u1) from state:
        u1_bef = np.array(y_old[12:15])  # [u1x, u1y, u1z]
        u1z_bef = u1_bef[2]
        # Extract additional tubes' twist:
        uz_bef = y_old[15: 15 + (num_tubes - 1)]
        # Extract additional tubes' angles:
        theta_bef = y_old[15 + (num_tubes - 1): 15 + 2 * (num_tubes - 1)]


        # Get tube info for before and after segments
        K_list_bef, star_bef_list, _ = get_tube_info_i(i_seg, tubes, Break_Points)
        K_bef = sum(K_list_bef)
        K_list_aft, star_aft_list, _ = get_tube_info_i(i_seg + 1, tubes, Break_Points)
        K_aft = sum(K_list_aft)
        # Invert K_aft:
        try:
            invK_aft = np.linalg.inv(K_aft)
        except np.linalg.LinAlgError:
            invK_aft = np.linalg.pinv(K_aft)

        # Compute new z twist for the reference tube:
        u1z_aft = u1z_bef + (star_aft_list[0][2] - star_bef_list[0][2])
        # For additional tubes:
        uz_aft = []
        for j in range(1, num_tubes):
            uz_new = uz_bef[j - 1] + (star_aft_list[j][2] - star_bef_list[j][2])
            uz_aft.append(uz_new)
        # Compute theta_dot for additional tubes:
        theta_dot_aft = [uz_aft[j - 1] - u1z_aft for j in range(1, num_tubes)]
        # Also, for transition, we can use the old angles (theta_bef) to form rotation matrices:
        Rtheta_list = []
        for j in range(1, num_tubes):
            Rtheta = np.array([
                [np.cos(theta_bef[j - 1]), -np.sin(theta_bef[j - 1]), 0],
                [np.sin(theta_bef[j - 1]), np.cos(theta_bef[j - 1]), 0],
                [0, 0, 1]
            ])
            Rtheta_list.append(Rtheta)


        # followed matlab equation
        big_term = K_bef @ u1_bef + (K_list_aft[0] @ star_aft_list[0] - K_list_bef[0] @ star_bef_list[0])
        theta_dot_bef = [uz_bef[j - 1] - u1z_bef for j in range(1, num_tubes)]
        for j in range(1, num_tubes):
            big_term += (Rtheta_list[j - 1] @ (
                    K_list_aft[j] @ (star_aft_list[j] - theta_dot_aft[j - 1] * np.array([0, 0, 1])) -
                    K_list_bef[j] @ (star_bef_list[j] - theta_dot_bef[j - 1] * np.array([0, 0, 1]))
            ))
        # Multiply by invK_aft and then by [1 0; 0 1] to get a 2-element vector.
        Eye_xy = np.array([[1, 0, 0], [0, 1, 0]])
        u1xy_aft = Eye_xy @ (invK_aft @ big_term)  # This yields a 2-element vector

        # Assemble the new y_init for the next segment.
        # New state: [p, R.flatten(order='F'), u1xy_aft (2 elements),
        #             u1z_aft, (reference tube twist)
        #             uz_aft (list of length num_tubes-1),
        #             theta_bef (we keep the same angles),
        #             arc length (last element unchanged)]
        new_y = np.hstack([
            p,
            R.flatten(order='F'),
            u1xy_aft,
            [u1z_aft],
            uz_aft,
            theta_bef,
            [y_old[-1]]
        ])
        y_init = new_y

    return np.array(s), np.array(y_list)
