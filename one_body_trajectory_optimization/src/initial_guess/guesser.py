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
    """
    print("\n\nINITIAL GUESS PROCESS")

    # Unpack
    #   Some parameters will unpack as zero and be set by the guesser
    init_guess_steps = optimization_parameters['init_guess_steps']
    
    pos_vec_o_mns    = equality_parameters[  'pos_vec']['o']['mns']
    vel_vec_o_mns    = equality_parameters[  'vel_vec']['o']['mns']
    copos_vec_o_mns  = equality_parameters['copos_vec']['o']['mns']
    covel_vec_o_mns  = equality_parameters['covel_vec']['o']['mns']

    pos_vec_o_pls    = equality_parameters[  'pos_vec']['o']['pls']
    vel_vec_o_pls    = equality_parameters[  'vel_vec']['o']['pls']
    copos_vec_o_pls  = equality_parameters['copos_vec']['o']['pls']
    covel_vec_o_pls  = equality_parameters['covel_vec']['o']['pls']

    time_f_mns       = equality_parameters[     'time']['f']['mns']
    time_f_pls       = equality_parameters[     'time']['f']['pls']

    pos_vec_o_mode   = equality_parameters[  'pos_vec']['o']['mode']
    vel_vec_o_mode   = equality_parameters[  'vel_vec']['o']['mode']
    copos_vec_o_mode = equality_parameters['copos_vec']['o']['mode']
    covel_vec_o_mode = equality_parameters['covel_vec']['o']['mode']

    time_f_mode      = equality_parameters[     'time']['f']['mode']

    # Set initial guess for fixed variables
    if pos_vec_o_mode == 'fixed':
        pos_vec_o_pls = pos_vec_o_mns
    if vel_vec_o_mode == 'fixed':
        vel_vec_o_pls = vel_vec_o_mns
    if copos_vec_o_mode == 'fixed':
        copos_vec_o_pls = copos_vec_o_mns
    if covel_vec_o_mode == 'fixed':
        covel_vec_o_pls = covel_vec_o_mns
    if time_f_mode == 'fixed':
        time_f_mns = time_f_pls

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
        if pos_vec_o_mode == 'free':
            pos_vec_o_pls = np.random.uniform(low=-1, high=1, size=2)
        if vel_vec_o_mode == 'free':
            vel_vec_o_pls = np.random.uniform(low=-1, high=1, size=2)
        if copos_vec_o_mode == 'free':
            copos_vec_o_pls = np.random.uniform(low=-10, high=10, size=2)
        if covel_vec_o_mode == 'free':
            covel_vec_o_pls = np.random.uniform(low=-10, high=10, size=2)
        if time_f_mode == 'free':
            time_f_mns = np.random.uniform(low=1, high=300, size=1)

        decision_state_idx = np.hstack([
            pos_vec_o_pls  ,
            vel_vec_o_pls  ,
            copos_vec_o_pls,
            covel_vec_o_pls,
            time_f_mns     ,
        ])

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
            if idx==0:
                tqdm.write("    Minimum Error         Step")
            tqdm.write(f"    {error_mag_min:>13.6e}  {idx_min:>5d}/{init_guess_steps:>5d}")
            decision_state_min = decision_state_idx

    # Pack up and print solution
    return decision_state_min