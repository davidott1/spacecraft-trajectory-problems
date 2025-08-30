import numpy as np
from tqdm                                          import tqdm
from src.initial_guess.guesser                     import generate_guess
from src.optimize.two_point_boundary_value_problem import solve_for_root_and_compute_progress, solve_ivp_func, compute_hamiltonian
from src.plot.final_results                        import plot_final_results
from src.dynamics                                  import control_thrust_acceleration


def optimal_trajectory_solve(
        files_folders_parameters    ,
        system_parameters           ,
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
    ):
    """
    Main solver that implements the two-stage continuation process
    using the unified smoothed dynamics.
    """

    # Generate initial guess for the costates
    decision_state_initguess = \
        generate_guess(
            optimization_parameters     ,
            integration_state_parameters,
            equality_parameters         ,
            inequality_parameters       ,
        )

    # Unpack files and folders parameters
    min_type              =      optimization_parameters['min_type'             ]
    # pos_vec_o_mns         =          equality_parameters['pos_vec']['o']['mns']['value']
    # vel_vec_o_mns         =          equality_parameters['vel_vec']['o']['mns']['value']
    # pos_vec_f_pls         =          equality_parameters['pos_vec']['f']['pls']['value']
    # vel_vec_f_pls         =          equality_parameters['vel_vec']['f']['pls']['value']
    use_thrust_acc_limits =        inequality_parameters['use_thrust_acc_limits']
    use_thrust_limits     =        inequality_parameters['use_thrust_limits'    ]
    k_idxinitguess        =        inequality_parameters['k_idxinitguess'       ]
    k_idxfinsoln          =        inequality_parameters['k_idxfinsoln'         ]
    k_idxdivs             =        inequality_parameters['k_idxdivs'            ]

    # Optimize and enforce thrust or thrust-acc constraints
    print("\n\nOPTIMIZATION PROCESS")

    # Solve for the optimal min-fuel or min-energy trajectory

    # Intermediate solution: thrust- or thrust-acc-steepness continuation process
    if use_thrust_acc_limits or use_thrust_limits:
        print("  Thrust- or Thrust-Acc-Steepness Continuation Process")

    # Intermediate solution: initialize loop
    results_k_idx                = {}
    k_idxinitguess_to_idxfinsoln = np.logspace(np.log10(k_idxinitguess), np.log10(k_idxfinsoln), k_idxdivs)
    options_root                 = {
        'maxiter' : 100 * len(decision_state_initguess), # 100 * n
        'ftol'    : 1.0e-8, # 1e-8
        'xtol'    : 1.0e-8, # 1e-8
        'gtol'    : 1.0e-8, # 1e-8
    }
    optimization_parameters['include_jacobian'] = False # should be True
    # integration_state_parameters['include_scstm'] = True
    if use_thrust_acc_limits:
        inequality_parameters['use_thrust_acc_smoothing'] = True
    if use_thrust_limits:
        inequality_parameters['use_thrust_smoothing'] = True

    # Intermediate solution: loop though k values
    for idx, k_idx in tqdm(enumerate(k_idxinitguess_to_idxfinsoln), desc="Processing", leave=False, total=len(k_idxinitguess_to_idxfinsoln)):

        # Set the k-idx
        inequality_parameters['k_steepness'] = k_idx
        
        # Root solve and compute progress of current root solve
        soln_root, soln_ivp = \
            solve_for_root_and_compute_progress(
                decision_state_initguess    ,
                optimization_parameters     ,
                integration_state_parameters,
                equality_parameters         ,
                inequality_parameters       ,
                options_root                ,
            )
        
        # Record the results of the current step and update the decision state initial guess
        results_k_idx[k_idx] = soln_ivp
        decision_state_initguess = soln_root.x

        # Print the results of the current step
        error_mag = np.linalg.norm(soln_root.fun)
        if min_type == 'energy' and not use_thrust_acc_limits and not use_thrust_limits:
            if idx==0:
                tqdm.write(f"       {'Step':>5s} {'Error-Mag':>14s}")
            tqdm.write(f"     {idx+1:>3d}/{len(k_idxinitguess_to_idxfinsoln):>3d} {error_mag:>14.6e}")
        else:
            if idx==0:
                tqdm.write(f"       {'Step':>5s} {'k':>14s} {'Error-Mag':>14s}")
            tqdm.write(f"     {idx+1:>3d}/{len(k_idxinitguess_to_idxfinsoln):>3d} {k_idx:>14.6e} {error_mag:>14.6e}")

    # Final solution: no thrust or thrust-acc smoothing
    print("\n\nFINAL SOLUTION PROCESS")
    print("  Root-Solve Results")
    
    # Final solution: root solve and compute progress of current root solve
    optimization_parameters['include_jacobian']       = False  # should be True
    integration_state_parameters['include_scstm']     = False  # should be True
    integration_state_parameters['post_process']      = False # should be False
    inequality_parameters['use_thrust_acc_smoothing'] = False # should be False
    inequality_parameters['use_thrust_smoothing']     = False # should be False
    soln_root, soln_ivp = \
        solve_for_root_and_compute_progress(
            decision_state_initguess    ,
            optimization_parameters     ,
            integration_state_parameters,
            equality_parameters         ,
            inequality_parameters       ,
            options_root                ,
        )
    for key, value in soln_root.items():
        if isinstance(value, np.ndarray):
            if len(value.shape) == 1:
                if len(value) <= 4:
                    value_construct = '  '.join([str(f"{val:>13.6e}") for val in value])
                else:
                    value_construct = '  '.join([str(f"{val:>13.6e}") for val in value[:4]]) + '  ...'
            elif len(value.shape) == 2:
                if value.shape[1] <= 4:
                    value_construct = ['  '.join( str(f"{val:>13.6e}") for val in row) for row in value]
                    value_construct = '\n            : '.join(value_construct)
                else:
                    value_construct = ['  '.join( str(f"{val:>13.6e}") for val in row[:4]) for row in value]
                    value_construct = '  ...\n            : '.join(value_construct) + '  ...'
        else:
            value_construct = value
        print(f"    {key:>7s} : {value_construct}")
    
    # Final solution: post-process step
    optimization_parameters['include_jacobian']       = False # should be False
    integration_state_parameters['include_scstm']     = False # should be False
    integration_state_parameters['post_process']      = True  # should be True
    inequality_parameters['use_thrust_acc_smoothing'] = False # should be False
    inequality_parameters['use_thrust_smoothing']     = False # should be False
    # decision_state_initguess = soln_root.x
    # pos_vec_o_mns   = equality_parameters[  'pos_vec']['o']['mns']
    # vel_vec_o_mns   = equality_parameters[  'vel_vec']['o']['mns']
    # time_f_pls      = equality_parameters[     'time']['f']['pls']
    # pos_vec_f_pls   = equality_parameters[  'pos_vec']['f']['pls']
    # vel_vec_f_pls   = equality_parameters[  'vel_vec']['f']['pls']
    # copos_vec_f_pls = equality_parameters['copos_vec']['f']['pls']
    # covel_vec_f_pls = equality_parameters['covel_vec']['f']['pls']
    # ham_f_pls       = equality_parameters[      'ham']['f']['pls']

    decision_state_initguess = soln_root.x
    time_o_pls      = decision_state_initguess[0]
    time_f_mns      = decision_state_initguess[10]
    time_span       = np.array([time_o_pls, time_f_mns])
    pos_vec_o_pls   = decision_state_initguess[1:3]
    vel_vec_o_pls   = decision_state_initguess[3:5]
    copos_vec_o_pls = decision_state_initguess[5:7]
    covel_vec_o_pls = decision_state_initguess[7:9]

    thrust_acc_x, thrust_acc_y = \
        control_thrust_acceleration(
            min_type, 
            covel_vec_o_pls[0], covel_vec_o_pls[1],
            mass_o_pls,
            use_thrust_acc_limits, use_thrust_acc_smoothing, thrust_acc_min, thrust_acc_max,
            use_thrust_limits, use_thrust_smoothing, thrust_min, thrust_max,
            k_steepness,
        )

    ham_o_pls       = compute_hamiltonian(
            min_type     = optimization_parameters['min_type'],
            vel_x        = vel_vec_o_pls[0]                   ,
            vel_y        = vel_vec_o_pls[1]                   ,
            copos_x      = copos_vec_o_pls[0]                 ,
            copos_y      = copos_vec_o_pls[1]                 ,
            covel_x      = covel_vec_o_pls[0]                 ,
            covel_y      = covel_vec_o_pls[1]                 ,
            thrust_acc_x = -covel_vec_o_pls[0]                ,
            thrust_acc_y = -covel_vec_o_pls[1]                ,
            acc_x        = -covel_vec_o_pls[0]                ,
            acc_y        = -covel_vec_o_pls[1]                ,
        )


    state_costate_o = np.hstack([pos_vec_o_pls, vel_vec_o_pls, copos_vec_o_pls, covel_vec_o_pls])

    soln_ivp = \
        solve_ivp_func(
            time_span                   ,
            state_costate_o             ,
            optimization_parameters     ,
            integration_state_parameters,
            inequality_parameters       ,
        )
    
    results_finalsoln = soln_ivp
    state_f_finalsoln = results_finalsoln.y[0:4, -1]

    # Update initial minus and final plus value if parameter is free

    # pos_vec_o_mns   = equality_parameters[  'pos_vec']['o']['mns']
    # vel_vec_o_mns   = equality_parameters[  'vel_vec']['o']['mns']
    # time_f_pls      = equality_parameters[     'time']['f']['pls']
    # pos_vec_f_pls   = equality_parameters[  'pos_vec']['f']['pls']
    # vel_vec_f_pls   = equality_parameters[  'vel_vec']['f']['pls']
    # copos_vec_f_pls = equality_parameters['copos_vec']['f']['pls']
    # covel_vec_f_pls = equality_parameters['covel_vec']['f']['pls']
    # ham_f_pls       = equality_parameters[      'ham']['f']['pls']

    if equality_parameters['time'     ]['o']['mode'] == 'free': equality_parameters['time'     ]['o']['mns'] = time_o_pls
    if equality_parameters['pos_vec'  ]['o']['mode'] == 'free': equality_parameters['pos_vec'  ]['o']['mns'] = pos_vec_o_pls
    if equality_parameters['vel_vec'  ]['o']['mode'] == 'free': equality_parameters['vel_vec'  ]['o']['mns'] = vel_vec_o_pls
    if equality_parameters['copos_vec']['o']['mode'] == 'free': equality_parameters['copos_vec']['o']['mns'] = copos_vec_o_pls
    if equality_parameters['covel_vec']['o']['mode'] == 'free': equality_parameters['covel_vec']['o']['mns'] = covel_vec_o_pls
    if equality_parameters['ham'      ]['o']['mode'] == 'free': equality_parameters['ham'      ]['o']['mns'] = ham_o_pls

    if equality_parameters['time'     ]['f']['mode'] == 'free': time_f_pls      = time_f_mns
    if equality_parameters['pos_vec'  ]['f']['mode'] == 'free': pos_vec_f_pls   = pos_vec_f_mns
    if equality_parameters['vel_vec'  ]['f']['mode'] == 'free': vel_vec_f_pls   = vel_vec_f_mns
    if equality_parameters['copos_vec']['f']['mode'] == 'free': copos_vec_f_pls = copos_vec_f_mns
    if equality_parameters['covel_vec']['f']['mode'] == 'free': covel_vec_f_pls = covel_vec_f_mns
    if equality_parameters['ham'      ]['f']['mode'] == 'free': ham_f_pls       = ham_f_mns

    equality_parameters['copos_vec_o_mns'] = decision_state_initguess[0:2]
    equality_parameters['covel_vec_o_mns'] = decision_state_initguess[2:4]
    equality_parameters['copos_vec_o_pls'] = decision_state_initguess[0:2]
    equality_parameters['covel_vec_o_pls'] = decision_state_initguess[2:4]
    equality_parameters['copos_vec_f_mns'] = results_finalsoln.y[4:6, -1]
    equality_parameters['covel_vec_f_mns'] = results_finalsoln.y[6:8, -1]
    equality_parameters['copos_vec_f_pls'] = results_finalsoln.y[4:6, -1]
    equality_parameters['covel_vec_f_pls'] = results_finalsoln.y[6:8, -1]

    # Final solution: approx and true
    results_approx_finalsoln = results_k_idx[k_idxinitguess_to_idxfinsoln[-1]]
    state_f_approx_finalsoln = results_approx_finalsoln.y[0:4, -1]

    # Final solution: check final state error
    error_approx_finalsoln_vec = np.hstack([pos_vec_f_pls, vel_vec_f_pls]) - state_f_approx_finalsoln
    error_finalsoln_vec        = np.hstack([pos_vec_f_pls, vel_vec_f_pls]) - state_f_finalsoln

    print("\n  State Error Check")
    print(f"             {'Time-f':>14s} {'Pos-Xf':>14s} {'Pos-Yf':>14s} {'Vel-Xf':>14s} {'Vel-Yf':>14s} {'Co-Pos-Xf':>14s} {'Co-Pos-Yf':>14s} {'Co-Vel-Xf':>14s} {'Co-Vel-Yf':>14s} {  'Ham-f':>14s}")
    if min_type == 'fuel':
        print(f"             {'s':>14s} {'m':>14s} {'m':>14s} {'m/s':>14s} {'m/s':>14s} {'1/s':>14s} {'1/s':>14s} {'1':>14s} {'1':>14s} {'m/s^2':>14s}")
    else: # assume min_type == 'energy'
        print(f"             {'s':>14s} {'m':>14s} {'m':>14s} {'m/s':>14s} {'m/s':>14s} {'m/s^3':>14s} {'m/s^3':>14s} {'m/s^2':>14s} {'m/s^2':>14s} {'m^2/s^4':>14s}")
    print(f"    Target : {time_f_pls:>14.6e} {pos_vec_f_pls[0]:>14.6e} {pos_vec_f_pls[1]:>14.6e} {vel_vec_f_pls[0]:>14.6e} {vel_vec_f_pls[1]:>14.6e} {copos_vec_f_pls[0]:>14.6e} {copos_vec_f_pls[1]:>14.6e} {covel_vec_f_pls[0]:>14.6e} {covel_vec_f_pls[1]:>14.6e} {ham_f_pls:>14.6e}")
    print(f"    Approx : {0.0:>14.6e} {  state_f_approx_finalsoln[0]:>14.6e} {  state_f_approx_finalsoln[1]:>14.6e} {  state_f_approx_finalsoln[2]:>14.6e} {  state_f_approx_finalsoln[3]:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e}")
    print(f"    Error  : {0.0:>14.6e} {error_approx_finalsoln_vec[0]:>14.6e} {error_approx_finalsoln_vec[1]:>14.3e} {error_approx_finalsoln_vec[2]:>14.6e} {error_approx_finalsoln_vec[3]:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e}")
    print(f"    Actual : {0.0:>14.6e} {         state_f_finalsoln[0]:>14.6e} {         state_f_finalsoln[1]:>14.6e} {         state_f_finalsoln[2]:>14.6e} {         state_f_finalsoln[3]:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e}")
    print(f"    Error  : {0.0:>14.6e} {       error_finalsoln_vec[0]:>14.6e} {       error_finalsoln_vec[1]:>14.3e} {       error_finalsoln_vec[2]:>14.6e} {       error_finalsoln_vec[3]:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e} {0.0:>14.6e}")

    # Final solution: plot the results
    print("\n  Plot Final Solution Trajectory")
    plot_final_results(
        results_finalsoln       ,
        files_folders_parameters,
        system_parameters       ,
        optimization_parameters ,
        equality_parameters     ,
        inequality_parameters   ,
    )