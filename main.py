import numpy as np
from scipy.optimize import fsolve
from get_tube_info_i import define_breakpoints
from get_res import  get_res
from run_ode import run_ode
from robot_visualization import plot_robot_pyvista


def main():
    # Define Material properties
    E = 60e9  # Young's modulus in Pa
    nu = 0.3
    G = E / (2 * (1 + nu))  # Shear modulus in Pa

    # Tube dimensions and properties
    # Tube 1 (Innermost Tube)
    L1s = 34 * 0.001  # Straight length in meters
    L1c = 0 * 0.001   # Curved length in meters
    ro1 = 0.8 / 2 * 0.001  # Outer radius in meters
    ri1 = 0.00 / 2 * 0.001  # Inner radius in meters (solid tube)

    # Tube 2 (Middle Tube)
    L2s = 28 * 0.001
    L2c = 0 * 0.001
    ro2 = 1.04 / 2 * 0.001
    ri2 = 0.82 / 2 * 0.001

    # Tube 3 (Outer Tube)
    L3c = 10 * 0.001
    L3s2 = 15 * 0.001
    L3s = 5 * 0.001  # Straight length
    ro3 = 1.56 / 2 * 0.001
    ri3 = 1.14 / 2 * 0.001

    # Moments of inertia and stiffnesses
    # Tube 1
    I1 = (np.pi / 4) * (ro1**4 - ri1**4)
    J1 = 2 * I1
    E1I1 = E * I1
    G1J1 = G * J1

    # Tube 2
    I2 = (np.pi / 4) * (ro2**4 - ri2**4)
    J2 = 2 * I2
    E2I2 = E * I2
    G2J2 = G * J2

    # Tube 3
    I3 = (np.pi / 4) * (ro3**4 - ri3**4)
    J3 = 2 * I3
    E3I3 = E * I3
    G3J3 = G * J3

    # Define stiffnesses
    Stiff = np.array([E1I1, E2I2, E3I3, G1J1, G2J2, G3J3])

    # Pre-curvatures
    u1s = np.array([0.0, 0.0, 0.0])
    u2s = np.array([0.0, 0.0, 0.0])
    u3s = np.array([60.0, 0.0, 0.0])

    # Actuator inputs
    alpha1 = 0
    alpha2 = 0
    alpha3 = 3.5
    D1 = 0.0 * 0.001
    D2 = 0.0 * 0.001
    D3 = 0.0 * 0.001
    Q = np.array([alpha1, alpha2, alpha3, D1, D2, D3])

    # Lengths
    Lengths = np.array([L1s, L2s, L3s, L1c, L2c, L3c])

    # Initial guess for curvature and twist rates
    u_init_guess = np.array([60.0, 0.0, 0.0, 0.0, 0.0])

    # Tip force and moment
    F_tip = np.array([1.0, 2.0, 3.0])  # Applied tip force in Newtons
    L_tip = np.array([0.0, 0.0, 0.0])  # No external tip moment

    # Define tubes
    tubes = [
        {
            'D': D1,
            'sections': [
                {'type': 'straight', 'length': L1s},
                {'type': 'curved', 'length': L1c}
            ],
            'u_star_sections': [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0])
                #u1s
            ],
            'E_I': E1I1,  # From calculated stiffness
            'G_J': G1J1   # From calculated torsional stiffness
        },
        {
            'D': D2,
            'sections': [
                {'type': 'straight', 'length': L2s},
                {'type': 'curved', 'length': L2c}
            ],
            'u_star_sections': [
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0])
                #u2s
            ],
            'E_I': E2I2,  # From calculated stiffness
            'G_J': G2J2   # From calculated torsional stiffness
        },
        {
            'D': D3,
            'sections': [
                {'type': 'straight', 'length': L3s},
                {'type': 'curved', 'length': L3c}
                #{'type': 'straight', 'length': L3s2}
            ],
            'u_star_sections': [
                np.array([0.0, 0.0, 0.0]),
                u3s
                #np.array([0.0, 0.0, 0.0])
            ],
            'E_I': E3I3,  # From calculated stiffness
            'G_J': G3J3   # From calculated torsional stiffness
        }
    ]

    # Define parameters for the ODE
    Break_Points = define_breakpoints(tubes)
    print(Break_Points)
    params = {
        'tubes': tubes,
        'Break_Points': Break_Points,
        'F_tip': F_tip,
        'L_tip': L_tip,
        'alphas': [alpha1, alpha2, alpha3]
    }

    # Solve for initial conditions using fsolve

    options = {'xtol': 1e-2}  # solver tolerance
    u_init, info, exitflag, msg = fsolve(get_res, u_init_guess, args=(params,), full_output=True, xtol=1e-2)

    print("Optimal initial conditions:", u_init)

    # Compute backbone
    s, y = run_ode(u_init, params)
    backbone = np.column_stack((y[:, :3], s))  # Combine position and arc length
    print(backbone)

    # Extract True Tip State
    y_tip = y[-1, :]
    true_tip_pose = y_tip[0:3]
    R_tip = y_tip[3:12].reshape((3, 3), order='F')

    # Compute base force.
    # F_tip is defined in world coordinates
    F_tip_world = params['F_tip']
    true_F_tip_local = R_tip.T @ F_tip_world
    true_base_force = true_F_tip_local[2]  # axial component along local z
    print("\nTrue Tip Pose (world):", true_tip_pose)
    print("True Transmitted Base Force: {:.3f} N".format(true_base_force))

# For plotting
    # Define tube positions and dimensions
    tube_positions = {
        'tube_1': {'start': D1, 'end': D1 + L1s + L1c},
        'tube_2': {'start': D2, 'end': D2 + L2s + L2c},
        'tube_3': {'start': D3, 'end': D3 + L3s + L3c}
    }

    tube_dimensions = {
        'tube_1': {'outer_diameter': ro1 * 2},
        'tube_2': {'outer_diameter': ro2 * 2},
        'tube_3': {'outer_diameter': ro3 * 2}
    }

    # Plotting
    plot_robot_pyvista(
        backbone,
        tube_positions,
        tube_dimensions,
        F_tip=F_tip,
        first_person=False
    )

    plot_robot_pyvista(
        backbone,
        tube_positions,
        tube_dimensions,
        F_tip=F_tip,
        first_person=True
    )

if __name__ == "__main__":
    main()
