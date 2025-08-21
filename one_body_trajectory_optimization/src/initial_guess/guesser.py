import numpy as np
from tqdm import tqdm
from src.optimize.two_point_boundary_value_problem import tpbvp_objective_and_jacobian


def generate_guess(
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
    ):
    """
    Generates a robust initial guess for the co-states: copos_vec, covel_vec

    [delta_time_of, pos_vec_o, vel_vec_o, copos_vec_o, covel_vec_o, mass_o, opt_ctrl_obj_o] : 1, 2, 2, 2, 2, 1, 1
    """
    print("\n\nINITIAL GUESS PROCESS")
    
    # Define list for dictionary structure
    ordered_variables  = ['time', 'pos_vec', 'vel_vec', 'copos_vec', 'covel_vec', 'ham']
    ordered_boundaries = ['o', 'f']
    ordered_sides      = ['pls', 'mns']

    # Unpack
    #   Some parameters will unpack as zero and be set by the guesser
    init_guess_steps = optimization_parameters['init_guess_steps']

    time_o_mns       = equality_parameters[     'time']['o']['mns']['value']
    pos_vec_o_mns    = equality_parameters[  'pos_vec']['o']['mns']['value']
    vel_vec_o_mns    = equality_parameters[  'vel_vec']['o']['mns']['value']
    copos_vec_o_mns  = equality_parameters['copos_vec']['o']['mns']['value']
    covel_vec_o_mns  = equality_parameters['covel_vec']['o']['mns']['value']
    ham_o_mns        = equality_parameters[      'ham']['o']['mns']['value']

    time_o_pls       = equality_parameters[     'time']['o']['pls']['value']
    pos_vec_o_pls    = equality_parameters[  'pos_vec']['o']['pls']['value']
    vel_vec_o_pls    = equality_parameters[  'vel_vec']['o']['pls']['value']
    copos_vec_o_pls  = equality_parameters['copos_vec']['o']['pls']['value']
    covel_vec_o_pls  = equality_parameters['covel_vec']['o']['pls']['value']
    ham_o_pls        = equality_parameters[      'ham']['o']['pls']['value']

    time_f_mns       = equality_parameters[     'time']['f']['mns']['value']
    pos_vec_f_mns    = equality_parameters[  'pos_vec']['f']['mns']['value']
    vel_vec_f_mns    = equality_parameters[  'vel_vec']['f']['mns']['value']
    copos_vec_f_mns  = equality_parameters['copos_vec']['f']['mns']['value']
    covel_vec_f_mns  = equality_parameters['covel_vec']['f']['mns']['value']
    ham_f_mns        = equality_parameters[      'ham']['f']['mns']['value']

    time_f_pls       = equality_parameters[     'time']['f']['pls']['value']
    pos_vec_f_pls    = equality_parameters[  'pos_vec']['f']['pls']['value']
    vel_vec_f_pls    = equality_parameters[  'vel_vec']['f']['pls']['value']
    copos_vec_f_pls  = equality_parameters['copos_vec']['f']['pls']['value']
    covel_vec_f_pls  = equality_parameters['covel_vec']['f']['pls']['value']
    ham_f_pls        = equality_parameters[      'ham']['f']['pls']['value']

    time_o_mode      = equality_parameters[     'time']['o']['mode'] # choice: not used
    pos_vec_o_mode   = equality_parameters[  'pos_vec']['o']['mode']
    vel_vec_o_mode   = equality_parameters[  'vel_vec']['o']['mode']
    copos_vec_o_mode = equality_parameters['copos_vec']['o']['mode']
    covel_vec_o_mode = equality_parameters['covel_vec']['o']['mode']
    ham_o_mode       = equality_parameters[      'ham']['o']['mode']

    time_f_mode      = equality_parameters[     'time']['f']['mode'] # choice: used
    pos_vec_f_mode   = equality_parameters[  'pos_vec']['f']['mode']
    vel_vec_f_mode   = equality_parameters[  'vel_vec']['f']['mode']
    copos_vec_f_mode = equality_parameters['copos_vec']['f']['mode']
    covel_vec_f_mode = equality_parameters['covel_vec']['f']['mode']
    ham_f_mode       = equality_parameters[      'ham']['f']['mode']
    
    # Print free and fixed variables
    free_vars = []
    fixed_vars = []
    free_vars_len = 0
    fixed_vars_len = 0
    for var_name, var_data in equality_parameters.items():
        if isinstance(var_data, list):
            continue
        for boundary_type, boundary_data in var_data.items():
            var_len = np.size(boundary_data['pls']['value'])
            full_var_name = f"{var_name}_{boundary_type}"
            if boundary_data['mode'] == 'free':
                free_vars.append(full_var_name)
                free_vars_len += var_len
            else:
                fixed_vars.append(full_var_name)
                fixed_vars_len += var_len

    print(f"  Unknowns (free): {free_vars_len}")
    print(f"    {', '.join(free_vars)}")
    print(f"  Knowns (fixed): {fixed_vars_len}")
    print(f"    {', '.join(fixed_vars)}")
    print(f"  Known State: {', '.join([str(t_or_f) for t_or_f in equality_parameters['known_states']])}")

    # Set initial guess for fixed variables
    if time_o_mode == 'fixed':
        time_o_pls = time_o_mns
    if pos_vec_o_mode == 'fixed':
        pos_vec_o_pls = pos_vec_o_mns
    if vel_vec_o_mode == 'fixed':
        vel_vec_o_pls = vel_vec_o_mns
    if copos_vec_o_mode == 'fixed':
        copos_vec_o_pls = copos_vec_o_mns
    if covel_vec_o_mode == 'fixed':
        covel_vec_o_pls = covel_vec_o_mns
    if ham_o_mode == 'fixed':
        ham_o_pls = ham_o_mns
    if time_f_mode == 'fixed':
        time_f_mns = time_f_pls
    if pos_vec_f_mode == 'fixed':
        pos_vec_f_mns = pos_vec_f_pls
    if vel_vec_f_mode == 'fixed':
        vel_vec_f_mns = vel_vec_f_pls
    if copos_vec_f_mode == 'fixed':
        copos_vec_f_mns = copos_vec_f_pls
    if covel_vec_f_mode == 'fixed':
        covel_vec_f_mns = covel_vec_f_pls
    if ham_f_mode == 'fixed':
        ham_f_mns = ham_f_pls

    # Initialize loop for random guesses for free variables
    optimization_parameters['include_jacobian'] = False
    inequality_parameters['k_steepness'] = inequality_parameters['k_idxinitguess']
    if inequality_parameters['use_thrust_acc_limits']:
        inequality_parameters['use_thrust_acc_smoothing'] = True
    if inequality_parameters['use_thrust_limits']:
        inequality_parameters['use_thrust_smoothing'] = True

    # Loop through random guesses for the costates
    print("  Random Initial Guess Generation")
    error_mag_min = np.Inf
    for idx in tqdm(range(init_guess_steps), desc="Processing", leave=False, total=init_guess_steps):
        decision_state_idx = np.array([])
        if time_o_mode == 'free':
            time_o_pls = np.random.uniform(low=-1, high=1, size=1)
        # decision_state_idx = np.hstack([decision_state_idx, time_o_pls])
        if pos_vec_o_mode == 'free':
            pos_vec_o_pls = np.random.uniform(low=-1, high=1, size=2)
        # decision_state_idx = np.hstack([decision_state_idx, pos_vec_o_pls])
        if vel_vec_o_mode == 'free':
            vel_vec_o_pls = np.random.uniform(low=-1, high=1, size=2)
        # decision_state_idx = np.hstack([decision_state_idx, vel_vec_o_pls])
        if copos_vec_o_mode == 'free':
            copos_vec_o_pls = np.random.uniform(low=-1, high=1, size=2)
        # decision_state_idx = np.hstack([decision_state_idx, copos_vec_o_pls])
        if covel_vec_o_mode == 'free':
            covel_vec_o_pls = np.random.uniform(low=-1, high=1, size=2)
        # decision_state_idx = np.hstack([decision_state_idx, covel_vec_o_pls])
        if ham_o_mode == 'free':
            ham_o_pls = np.random.uniform(low=-1, high=1, size=1)
        # decision_state_idx = np.hstack([decision_state_idx, ham_o_pls])

        if time_f_mode == 'free':
            time_f_mns = np.random.uniform(low=-1, high=1, size=1)
        # decision_state_idx = np.hstack([decision_state_idx, time_f_mns])
        if pos_vec_f_mode == 'free':
            pos_vec_f_mns = np.random.uniform(low=-1, high=1, size=2)
        # decision_state_idx = np.hstack([decision_state_idx, pos_vec_f_mns])
        if vel_vec_f_mode == 'free':
            vel_vec_f_mns = np.random.uniform(low=-1, high=1, size=2)
        # decision_state_idx = np.hstack([decision_state_idx, vel_vec_f_mns])
        if copos_vec_f_mode == 'free':
            copos_vec_f_mns = np.random.uniform(low=-1, high=1, size=2)
        # decision_state_idx = np.hstack([decision_state_idx, copos_vec_f_mns])
        if covel_vec_f_mode == 'free':
            covel_vec_f_mns = np.random.uniform(low=-1, high=1, size=2)
        # decision_state_idx = np.hstack([decision_state_idx, covel_vec_f_mns])
        if ham_f_mode == 'free':
            ham_f_mns = np.random.uniform(low=-1, high=1, size=1)
        # decision_state_idx = np.hstack([decision_state_idx, ham_f_mns])

        decision_state_idx = np.hstack([
            time_o_pls     ,
            pos_vec_o_pls  ,
            vel_vec_o_pls  ,
            copos_vec_o_pls,
            covel_vec_o_pls,
            ham_o_pls      ,
            time_f_mns     ,
            pos_vec_f_mns  ,
            vel_vec_f_mns  ,
            copos_vec_f_mns,
            covel_vec_f_mns,
            ham_f_mns      ,
        ])

        # # Print decision state
        # print(f"  Decision State")
        # idx_decision_state = 0
        # idx_parameter      = 0
        # for bnd in ordered_boundaries:
        #     for var in ordered_variables:
        #         for side in ordered_sides:
        #             if (
        #                 (bnd == 'o' and side == 'pls')
        #                 or (bnd == 'f' and side == 'mns')
        #             ):
        #                 number_elements = np.size(equality_parameters[var][bnd][side]['value'])
        #                 fixed_or_free   = str('Fixed' if equality_parameters['known_states'][idx_decision_state] else 'Free')
        #                 print("    " + f"{f'{var}_{bnd}_{side}':<15s}" + " : " + f"{fixed_or_free:<5s}" + " : " + f"{decision_state_idx[idx_decision_state:idx_decision_state+number_elements]}")
        #                 idx_decision_state += number_elements
        #                 idx_parameter      += 1

        error_idx = \
            tpbvp_objective_and_jacobian(
                decision_state_idx          ,
                optimization_parameters     ,
                integration_state_parameters,
                equality_parameters         ,
                inequality_parameters       ,
            )

        error_mag_idx = np.linalg.norm(error_idx)

        if error_mag_idx < error_mag_min:
            idx_min            = idx
            error_mag_min      = error_mag_idx
            decision_state_min = decision_state_idx

            pos_vec_o_pls   = decision_state_idx[ 1: 3]
            vel_vec_o_pls   = decision_state_idx[ 4: 6]
            copos_vec_o_pls = decision_state_idx[ 7: 9]
            covel_vec_o_pls = decision_state_idx[10:12]
            state_costate_o_min = np.hstack([pos_vec_o_pls, vel_vec_o_pls, copos_vec_o_pls, covel_vec_o_pls])

            # if idx==0:
            #     tqdm.write(f"                               {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s}")
            #     tqdm.write(f"          {'Step':>5s} {'Error-Mag':>14s} {'Pos-Xo':>14s} {'Pos-Yo':>14s} {'Vel-Xo':>14s} {'Vel-Yo':>14s} {'Co-Pos-Xo':>14s} {'Co-Pos-Yo':>14s} {'Co-Vel-Xo':>14s} {'Co-Vel-Yo':>14s}")
            # state_costate_o_min_str = ' '.join(f"{x:>14.6e}" for x in state_costate_o_min)
            # tqdm.write(f"     {idx_min:>5d}/{init_guess_steps:>4d} {error_mag_min:>14.6e} {state_costate_o_min_str}")

    # Pack up and print solution
    return decision_state_min