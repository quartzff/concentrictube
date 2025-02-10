import numpy as np

def define_breakpoints(tubes):
    """
    Define the breakpoints for integration based on tube sections,
    and computing each breakpoint as (-D + cumulative_length).

    Parameters:
    -----------
    tubes : list of dict
        Each dict represents a tube with translation (D) and sections.

    Returns:
    --------
    Break_Points : ndarray
        Sorted unique breakpoints, preserving the 0 reference.
    """
    breakpoints = [0.0]  # Include 0 explicitly, as in MATLAB


    # For each tube, compute breakpoints as (-D + cumulative_length)
    for tube in tubes:
        D = tube['D']                # Insertion depth
        sections = tube['sections']  # Straight/curved segments

        cumulative_length = 0.0
        for section in sections:
            cumulative_length += section['length']
            breakpoint = -D + cumulative_length
            breakpoints.append(breakpoint)

    breakpoints_array = np.array(breakpoints)
    breakpoints_sorted =  np.sort(breakpoints_array)

    return breakpoints_sorted


def get_current_segment(s, Break_Points):
    """
    Determine the current segment index based on arc length s.

    Parameters:
    -----------
    s : float
        Current arc length.
    Break_Points : ndarray
        Sorted list of unique breakpoints.

    Returns:
    --------
    segment_index : int
        Index of the current segment.
    """
    for idx, bp in enumerate(Break_Points):
        if s <= bp:
            return max(idx - 1, 0)
    return len(Break_Points) - 1


def get_tube_info_i(i_segment, tubes, Break_Points):
    """
    Determines stiffness matrices (K_list) and precurvatures (u_star_list)
    Parameters:
    -----------
    tubes : list of dict
        Each dict represents a tube with:
          - 'D': insertion depth
          - 'E_I': bending stiffness (EI)
          - 'G_J': torsional stiffness (GJ)
          - 'sections': a list of dict:
              [{'type': 'straight' or 'curved', 'length': L}, ...]
          - 'u_star_sections': list of np.array(3,) for each section
    Break_Points : ndarray
        Sorted array of unique breakpoints, e.g. [0, -D+..., -D+...], as in MATLAB.

    Returns:
    --------
    K_list : list of ndarray (3x3)
        Stiffness matrices for each tube (like [K1, K2, K3] in MATLAB).
    u_star_list : list of ndarray (3,)
        Precurvatures for each tube (like [u1_star, u2_star, u3_star] in MATLAB).
    u_star_dot_list : list of ndarray (3,)
        Curvature derivatives (all zero by default).
    """

    bp = Break_Points[i_segment]

    K_list = []
    u_star_list = []
    u_star_dot_list = []

    # Iterate over each tube with its index
    for tube_index, tube in enumerate(tubes):
        D     = tube['D']
        E_I   = tube['E_I']
        G_J   = tube['G_J']
        secs  = tube['sections']         # e.g. [{'type': 'straight', 'length': ...}, ...]
        u_secs = tube['u_star_sections'] # e.g. [array([0,0,0]), array([kappa,0,0])]

        # Defaults
        K = np.zeros((3, 3))
        u_star = np.array([0.0, 0.0, 0.0])
        u_star_dot = np.array([0.0, 0.0, 0.0])

        # (1) Decide if this tube is "present" or not
        #     if tube_index == 0, we force the tube to always be active,
        total_length = sum(sec['length'] for sec in secs)

        if tube_index == 0:
            # Always keep Tube 1 on
            K = np.diag([E_I, E_I, G_J])
        else:
            # For tubes #2, #3, etc., check presence
            if bp < -D + total_length:
                K = np.diag([E_I, E_I, G_J])
            else:
                K = np.zeros((3, 3))

        # (2) Decide which section is active (straight vs. curved)
        cumulative = 0.0
        for idx_sec, sec in enumerate(secs):
            cumulative += sec['length']
            if bp < -D + cumulative:  # use '<' instead of '<='
                if sec['type'] == 'straight':
                    u_star = np.array([0.0, 0.0, 0.0])
                elif sec['type'] == 'curved':
                    u_star = u_secs[idx_sec]
                break
        else:
            # Past the last section => use the last section's property
            last_sec = secs[-1]
            if last_sec['type'] == 'curved':
                u_star = u_secs[-1]
            else:
                u_star = np.array([0.0, 0.0, 0.0])

        # (3) Curvature derivative = 0
        u_star_dot = np.array([0.0, 0.0, 0.0])

        K_list.append(K)
        u_star_list.append(u_star)
        u_star_dot_list.append(u_star_dot)

    return K_list, u_star_list, u_star_dot_list
