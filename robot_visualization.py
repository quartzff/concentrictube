# robot_visualization.py

import numpy as np
import pyvista as pv

def plot_robot_pyvista(backbone, tube_positions, tube_dimensions, F_tip=None, first_person=False):
    """Plot the robot backbone and tubes using PyVista."""
    # Extract s and coordinates
    s = backbone[:, 3]
    x = backbone[:, 0]
    y = backbone[:, 1]
    z = backbone[:, 2]
    points = np.column_stack((x, y, z))

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Define tube plotting order and styles
    tube_order = ['tube_1', 'tube_2', 'tube_3']  # Plot from innermost to outermost
    tube_styles = {
        'tube_1': {'color': 'red'},
        'tube_2': {'color': 'blue'},
        'tube_3': {'color': 'black'}
    }

    # Transformation to position the tube base at (1.525 mm, -1.62 mm, 0)
    translation_vector = np.array([-1.525 * 0.001, -3.24 * 0.001, 0])  # in meters

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation_vector

    # Apply the transformation to the backbone points
    points_homogeneous = np.c_[points, np.ones(len(points))]
    points_transformed = (translation_matrix @ points_homogeneous.T).T[:, :3]

    # Plot each tube in the correct order
    for tube_name in tube_order:
        positions = tube_positions[tube_name]
        start_s = positions['start']
        end_s = positions['end']
        indices = np.where((s >= start_s) & (s <= end_s))[0]

        if len(indices) > 1:
            tube_points = points_transformed[indices]

            # Create a line for the tube
            line = pv.lines_from_points(tube_points)

            # Create a tube mesh with the specified outer diameter
            outer_radius = tube_dimensions[tube_name]['outer_diameter'] / 2  # Convert diameter to radius
            tube_mesh = line.tube(radius=outer_radius)

            # Add the tube mesh to the plotter
            plotter.add_mesh(
                tube_mesh,
                color=tube_styles[tube_name]['color'],
                label=tube_name.replace('_', ' ').capitalize()
            )

    # Camera settings
    if first_person:
        # Camera position in front of the tubes at z = +7 mm
        #camera_position = np.array([0.0, 1.62 * 0.001, 7 * 0.001])  # in meters
        camera_position = np.array([0.0, 0.0, 7 * 0.001])

        # Compute camera direction along positive z-axis
        camera_direction = np.array([0, 0, 1])  # Along positive z-axis

        # Set the focus point ahead along the camera direction
        focus_distance = 0.3  # 100 mm ahead
        angle_degrees = -0
        angle_radians = np.deg2rad(angle_degrees)  # Convert to radians

        focus_z= np.tan(angle_radians) * focus_distance * np.array([0, 1, 0])

        focus_point = camera_position + camera_direction * focus_distance + focus_z

        # Set the 'up' vector
        up_vector = np.array([0, 1, 0])  # Up is along positive y-axis

        # Set the camera position and orientation
        plotter.camera.position = camera_position
        plotter.camera.focal_point = focus_point
        plotter.camera.up = up_vector

        # Use perspective projection
        plotter.camera.parallel_projection = False

        # Set the view angle (field of view)
        plotter.camera.view_angle = 80  # 80 degrees field of view


        # Adjust clipping range
        plotter.camera.clipping_range = (0.001, 0.2)

    else:
        # Default view
        plotter.view_isometric()
        # Add grid and axes
        plotter.show_grid(color='lightgray')
        plotter.add_axes()

    # Add legend
    plotter.add_legend()

    # Show the plot
    plotter.show()