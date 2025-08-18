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
    init_guess_steps = optimization_parameters['init_guess_steps']
    pos_vec_o_mns    =     equality_parameters['pos_vec_o_mns'   ]
    vel_vec_o_mns    =     equality_parameters['vel_vec_o_mns'   ]

    # Initialize loop for random guesses
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
        copos_vec_o        = np.random.uniform(low=-1, high=1, size=2)
        covel_vec_o        = np.random.uniform(low=-1, high=1, size=2)
        decision_state_idx = np.hstack([copos_vec_o, covel_vec_o])
        
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
            integ_state_min    = np.hstack([pos_vec_o_mns, vel_vec_o_mns, decision_state_min])
            if idx==0:
                tqdm.write(f"                               {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s}")
                tqdm.write(f"          {'Step':>5s} {'Error-Mag':>14s} {'Pos-Xo':>14s} {'Pos-Yo':>14s} {'Vel-Xo':>14s} {'Vel-Yo':>14s} {'Co-Pos-Xo':>14s} {'Co-Pos-Yo':>14s} {'Co-Vel-Xo':>14s} {'Co-Vel-Yo':>14s}")
            integ_state_min_str = ' '.join(f"{x:>14.6e}" for x in integ_state_min)
            tqdm.write(f"     {idx_min:>5d}/{init_guess_steps:>4d} {error_mag_min:>14.6e} {integ_state_min_str}")

    # Pack up and print solution
    costate_o_guess = decision_state_min
    return costate_o_guess