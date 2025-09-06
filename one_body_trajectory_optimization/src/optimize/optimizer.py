import numpy as np
from tqdm                                          import tqdm
from src.initial_guess.guesser                     import generate_guess
from src.optimize.two_point_boundary_value_problem import solve_for_root_and_compute_progress, solve_ivp_func, compute_hamiltonian
from src.plot.final_results                        import plot_final_results
from src.model.dynamics                            import control_thrust_acceleration


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
    min_type              = optimization_parameters['min_type'             ]
    use_thrust_acc_limits =   inequality_parameters['use_thrust_acc_limits']
    use_thrust_limits     =   inequality_parameters['use_thrust_limits'    ]
    k_idxinitguess        =   inequality_parameters['k_idxinitguess'       ]
    k_idxfinsoln          =   inequality_parameters['k_idxfinsoln'         ]
    k_idxdivs             =   inequality_parameters['k_idxdivs'            ]

    # Optimize and enforce thrust or thrust-acc constraints
    print("\n\nOPTIMIZATION PROCESS")

    # Solve for the optimal min-fuel or min-energy trajectory

    # Intermediate solution: thrust- or thrust-acc-steepness continuation process
    if use_thrust_acc_limits or use_thrust_limits:
        print("  Thrust- or Thrust-Acc Continuation Process: Smooth Inequality Enforcement Using k Steepness")

    # Intermediate solution: initialize loop
    results_store = {}
    k_idxinitguess_to_idxfinsoln = np.logspace(np.log10(k_idxinitguess), np.log10(k_idxfinsoln), k_idxdivs)
    options_root                 = {
        'maxiter' : 100 * len(decision_state_initguess), # 100 * len(decision_state_initguess)
        'ftol'    : 1.0e-8, # 1e-8
        'xtol'    : 1.0e-8, # 1e-8
        'gtol'    : 1.0e-8, # 1e-8
    }

    if min_type in ('energy', 'fuel'):

        optimization_parameters['include_jacobian'] = True # should be True
        if use_thrust_acc_limits:
            inequality_parameters['use_thrust_acc_smoothing'] = True
            inequality_parameters['use_thrust_smoothing'    ] = False
        if use_thrust_limits:
            inequality_parameters['use_thrust_acc_smoothing'] = False
            inequality_parameters['use_thrust_smoothing'    ] = True

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
            results_store[k_idx]     = soln_ivp
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

        # Get final solution
        results_approx_finalsoln = results_store[k_idxinitguess_to_idxfinsoln[-1]]

    elif min_type == 'energyfuel':

        optimization_parameters['include_jacobian'] = True # should be True
        if use_thrust_acc_limits:
            inequality_parameters['use_thrust_acc_smoothing'] = True
            inequality_parameters['use_thrust_smoothing'    ] = False
        if use_thrust_limits:
            inequality_parameters['use_thrust_acc_smoothing'] = False
            inequality_parameters['use_thrust_smoothing'    ] = True

        alphas_1 = np.linspace(1, 0.01, 100) ** 1
        alphas_2 = np.array([
            10e-3,  9e-3,  8e-3,  7e-3,  6e-3,  5e-3,  4e-3,  3e-3,  2e-3,
            19e-4, 18e-4, 17e-4, 16e-4, 15e-4, 14e-4, 13e-4, 12e-4, 11e-4,
            10e-4,  9e-4,  8e-4,  7e-4,  6e-4,  5e-4,  4e-4,  3e-4,  2e-4,
            19e-5, 18e-5, 17e-5, 16e-5, 15e-5, 14e-5, 13e-5, 12e-5, 11e-5,
            10e-5,  9e-5,  8e-5,  7e-5,  6e-5,  5e-5,  4e-5,  3e-5,  2e-5,
            19e-6, 18e-6, 17e-6, 16e-6, 15e-6, 14e-6, 13e-6, 12e-6, 11e-6,
            10e-6,  9e-6,  8e-6,  7e-6,  6e-6,  5e-6,  4e-6,  3e-6,  2e-6,
            19e-7, 18e-7, 17e-7, 16e-7, 15e-7, 14e-7, 13e-7, 12e-7, 11e-7,
            10e-7,  9e-7,  8e-7,  7e-7,  6e-7,  5e-7,  4e-7,  3e-7,  2e-7,
            19e-8, 18e-8, 17e-8, 16e-8, 15e-8, 14e-8, 13e-8, 12e-8, 11e-8,
            10e-8,  9e-8,  8e-8,  7e-8,  6e-8,  5e-8,  4e-8,  3e-8,  2e-8,
            10e-9
        ])
        alphas = np.hstack([alphas_1, alphas_2])
        alpha = alphas[0]
        inequality_parameters['alpha'] = alpha

        # Intermediate solution: loop though k values
        for idx_k, k_idx in tqdm(enumerate(k_idxinitguess_to_idxfinsoln), desc="Processing", leave=False, total=len(k_idxinitguess_to_idxfinsoln)):

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
            results_store[(k_idx, alpha)] = soln_ivp
            decision_state_initguess   = soln_root.x

            # Print the results of the current step
            error_mag = np.linalg.norm(soln_root.fun)
            if min_type == 'energy' and not use_thrust_acc_limits and not use_thrust_limits:
                if idx_k==0:
                    tqdm.write(f"       {'Step':>5s} {'Error-Mag':>14s}")
                tqdm.write(f"     {idx_k+1:>3d}/{len(k_idxinitguess_to_idxfinsoln):>3d} {error_mag:>14.6e}")
            else:
                if idx_k==0:
                    tqdm.write(f"       {'Step':>5s} {'k':>14s} {'alpha':>14s} {'Error-Mag':>14s}")
                tqdm.write(f"     {idx_k+1:>3d}/{len(k_idxinitguess_to_idxfinsoln):>3d} {k_idx:>14.6e} {alpha:>14.6e} {error_mag:>14.6e}")

        # Intermediate solution: loop though alpha values
        for idx_alpha, alpha in tqdm(enumerate(alphas[1:], start=1), desc="Processing", leave=False, total=len(alphas[1:])):

            inequality_parameters['alpha'] = alpha
            # inequality_parameters['k_steepness'] = k_idx
            
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
            results_store[(k_idx, alpha)] = soln_ivp
            decision_state_initguess   = soln_root.x

            # Print the results of the current step
            error_mag = np.linalg.norm(soln_root.fun)
            if min_type == 'energy' and not use_thrust_acc_limits and not use_thrust_limits:
                if idx_alpha==1:
                    tqdm.write("\n  Thrust- or Thrust-Acc Continuation Process: Smooth Energy to Fuel Transition Using alpha Weighting")
                    tqdm.write(f"       {'Step':>5s} {'Error-Mag':>14s}")
                tqdm.write(f"     {idx_alpha+1:>3d}/{len(alphas):>3d} {error_mag:>14.6e}")
            else:
                if idx_alpha==1:
                    tqdm.write("\n  Thrust- or Thrust-Acc Continuation Process: Smooth Energy to Fuel Transition Using alpha Weighting")
                    tqdm.write(f"       {'Step':>5s} {'k':>14s} {'alpha':>14s} {'Error-Mag':>14s}")
                tqdm.write(f"     {idx_alpha+1:>3d}/{len(alphas):>3d} {k_idx:>14.6e} {alpha:>14.6e} {error_mag:>14.6e}")

        # Get final solution
        results_approx_finalsoln = results_store[(k_idxinitguess_to_idxfinsoln[-1], alphas[-1])]

    # Final solution: no thrust or thrust-acc smoothing
    print("\n\nFINAL SOLUTION PROCESS")
    print("  Root-Solve Results")

    # Final solution: root solve and compute progress of current root solve

    # Set up
    optimization_parameters['include_jacobian']       = True # should be True
    integration_state_parameters['include_scstm']     = True # should be True
    integration_state_parameters['post_process']      = False # should be False
    inequality_parameters['use_thrust_acc_smoothing'] = False # should be False
    inequality_parameters['use_thrust_smoothing']     = False # should be False

    # Use min type fuel for finalized solution
    if optimization_parameters['min_type'] == 'energyfuel':
        optimization_parameters['min_type'] = 'fuel' 

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
    
    # Final solution: post-process step, no use smoothing
    optimization_parameters['include_jacobian']       = False # should be False
    integration_state_parameters['include_scstm']     = False # should be False
    integration_state_parameters['post_process']      = True  # should be True
    inequality_parameters['use_thrust_acc_smoothing'] = False # should be False
    inequality_parameters['use_thrust_smoothing']     = False # should be False

    decision_state_initguess = soln_root.x
    time_o_pls      = decision_state_initguess[0]
    time_f_mns      = decision_state_initguess[10]
    time_span       = np.array([time_o_pls, time_f_mns])
    pos_vec_o_pls   = decision_state_initguess[1:3]
    vel_vec_o_pls   = decision_state_initguess[3:5]
    copos_vec_o_pls = decision_state_initguess[5:7]
    covel_vec_o_pls = decision_state_initguess[7:9]
    thrust_acc_x_o_pls, thrust_acc_y_o_pls, _, _, _, _, _, _ = \
        control_thrust_acceleration(
            min_type                 = optimization_parameters['min_type'],
            covel_x                  = covel_vec_o_pls[0],
            covel_y                  = covel_vec_o_pls[1],
            use_thrust_acc_limits    = inequality_parameters['use_thrust_acc_limits'],
            use_thrust_acc_smoothing = inequality_parameters['use_thrust_acc_smoothing'],
            thrust_acc_min           = inequality_parameters['thrust_acc_min'],
            thrust_acc_max           = inequality_parameters['thrust_acc_max'],
            use_thrust_limits        = inequality_parameters['use_thrust_limits'],
            use_thrust_smoothing     = inequality_parameters['use_thrust_smoothing'],
            thrust_min               = inequality_parameters['thrust_min'],
            thrust_max               = inequality_parameters['thrust_max'],
            k_steepness              = inequality_parameters['k_steepness'],
            mass                     = integration_state_parameters['mass_o'],
        )
    ham_o_pls = \
        compute_hamiltonian(
            min_type     = optimization_parameters['min_type'],
            vel_x        = vel_vec_o_pls[0]                   ,
            vel_y        = vel_vec_o_pls[1]                   ,
            copos_x      = copos_vec_o_pls[0]                 ,
            copos_y      = copos_vec_o_pls[1]                 ,
            covel_x      = covel_vec_o_pls[0]                 ,
            covel_y      = covel_vec_o_pls[1]                 ,
            thrust_acc_x = thrust_acc_x_o_pls                 ,
            thrust_acc_y = thrust_acc_y_o_pls                 ,
            acc_x        = thrust_acc_x_o_pls                 ,
            acc_y        = thrust_acc_y_o_pls                 ,
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
    
    results_finalsoln   = soln_ivp
    state_o_finalsoln   = results_finalsoln.y[0:4,  0]
    costate_o_finalsoln = results_finalsoln.y[4:8,  0]
    state_f_finalsoln   = results_finalsoln.y[0:4, -1]
    costate_f_finalsoln = results_finalsoln.y[4:8, -1]

    pos_vec_f_mns   = results_finalsoln.y[0:2, -1]
    vel_vec_f_mns   = results_finalsoln.y[2:4, -1]
    copos_vec_f_mns = results_finalsoln.y[4:6, -1]
    covel_vec_f_mns = results_finalsoln.y[6:8, -1]
    mass_f_mns      = results_finalsoln.y[  8, -1]
    thrust_acc_x_f_mns, thrust_acc_y_f_mns, _, _, _, _, _, _ = \
        control_thrust_acceleration(
            min_type                 = optimization_parameters['min_type'],
            covel_x                  = covel_vec_f_mns[0],
            covel_y                  = covel_vec_f_mns[1],
            use_thrust_acc_limits    = inequality_parameters['use_thrust_acc_limits'],
            use_thrust_acc_smoothing = inequality_parameters['use_thrust_acc_smoothing'],
            thrust_acc_min           = inequality_parameters['thrust_acc_min'],
            thrust_acc_max           = inequality_parameters['thrust_acc_max'],
            use_thrust_limits        = inequality_parameters['use_thrust_limits'],
            use_thrust_smoothing     = inequality_parameters['use_thrust_smoothing'],
            thrust_min               = inequality_parameters['thrust_min'],
            thrust_max               = inequality_parameters['thrust_max'],
            k_steepness              = inequality_parameters['k_steepness'],
            mass                     = mass_f_mns,
        )
    ham_f_mns = \
        compute_hamiltonian(
            min_type     = optimization_parameters['min_type'],
            vel_x        = vel_vec_f_mns[0]                   ,
            vel_y        = vel_vec_f_mns[1]                   ,
            copos_x      = copos_vec_f_mns[0]                 ,
            copos_y      = copos_vec_f_mns[1]                 ,
            covel_x      = covel_vec_f_mns[0]                 ,
            covel_y      = covel_vec_f_mns[1]                 ,
            thrust_acc_x = thrust_acc_x_f_mns                 ,
            thrust_acc_y = thrust_acc_y_f_mns                 ,
            acc_x        = thrust_acc_x_f_mns                 ,
            acc_y        = thrust_acc_y_f_mns                 ,
        )
    
    if equality_parameters['time'     ]['o']['mode'] == 'free': equality_parameters['time'     ]['o']['mns'] =      time_o_pls
    if equality_parameters['pos_vec'  ]['o']['mode'] == 'free': equality_parameters['pos_vec'  ]['o']['mns'] =   pos_vec_o_pls
    if equality_parameters['vel_vec'  ]['o']['mode'] == 'free': equality_parameters['vel_vec'  ]['o']['mns'] =   vel_vec_o_pls
    if equality_parameters['copos_vec']['o']['mode'] == 'free': equality_parameters['copos_vec']['o']['mns'] = copos_vec_o_pls
    if equality_parameters['covel_vec']['o']['mode'] == 'free': equality_parameters['covel_vec']['o']['mns'] = covel_vec_o_pls
    if equality_parameters['ham'      ]['o']['mode'] == 'free': equality_parameters['ham'      ]['o']['mns'] =       ham_o_pls

    if equality_parameters['time'     ]['f']['mode'] == 'free': equality_parameters['time'     ]['f']['pls'] =      time_f_mns
    if equality_parameters['pos_vec'  ]['f']['mode'] == 'free': equality_parameters['pos_vec'  ]['f']['pls'] =   pos_vec_f_mns
    if equality_parameters['vel_vec'  ]['f']['mode'] == 'free': equality_parameters['vel_vec'  ]['f']['pls'] =   vel_vec_f_mns
    if equality_parameters['copos_vec']['f']['mode'] == 'free': equality_parameters['copos_vec']['f']['pls'] = copos_vec_f_mns
    if equality_parameters['covel_vec']['f']['mode'] == 'free': equality_parameters['covel_vec']['f']['pls'] = covel_vec_f_mns
    if equality_parameters['ham'      ]['f']['mode'] == 'free': equality_parameters['ham'      ]['f']['pls'] =       ham_f_mns

    time_o_mns      = equality_parameters['time'     ]['o']['mns']
    vel_vec_o_mns   = equality_parameters['vel_vec'  ]['o']['mns']
    pos_vec_o_mns   = equality_parameters['pos_vec'  ]['o']['mns']
    copos_vec_o_mns = equality_parameters['copos_vec']['o']['mns']
    covel_vec_o_mns = equality_parameters['covel_vec']['o']['mns']
    ham_o_mns       = equality_parameters['ham'      ]['o']['mns']

    time_f_pls      = equality_parameters['time'     ]['f']['pls']
    vel_vec_f_pls   = equality_parameters['vel_vec'  ]['f']['pls']
    pos_vec_f_pls   = equality_parameters['pos_vec'  ]['f']['pls']
    copos_vec_f_pls = equality_parameters['copos_vec']['f']['pls']
    covel_vec_f_pls = equality_parameters['covel_vec']['f']['pls']
    ham_f_pls       = equality_parameters['ham'      ]['f']['pls']

    error_o_finalsoln_vec = \
        np.hstack([time_o_mns, pos_vec_o_mns, vel_vec_o_mns, copos_vec_o_mns, covel_vec_o_mns, ham_o_mns]) \
        - np.hstack([time_o_pls, state_o_finalsoln, costate_o_finalsoln, ham_o_pls])
    error_f_finalsoln_vec = \
        np.hstack([time_f_pls, pos_vec_f_pls, vel_vec_f_pls, copos_vec_f_pls, covel_vec_f_pls, ham_f_pls]) \
        - np.hstack([time_f_mns, state_f_finalsoln, costate_f_finalsoln, ham_f_mns])

    # Final solution: approximate (use smoothing)
    # results_approx_finalsoln = results_store[k_idxinitguess_to_idxfinsoln[-1]]
    state_costate_o_approx_finalsoln = results_approx_finalsoln.y[0:8,  0]
    state_costate_f_approx_finalsoln = results_approx_finalsoln.y[0:8, -1]
    if inequality_parameters['use_thrust_limits']:
        mass_f_mns = results_approx_finalsoln.y[8, -1]
    else:
        mass_f_mns = 1.0 # dummy value

    pos_vec_o_pls   = state_costate_o_approx_finalsoln[0:2]
    vel_vec_o_pls   = state_costate_o_approx_finalsoln[2:4]
    copos_vec_o_pls = state_costate_o_approx_finalsoln[4:6]
    covel_vec_o_pls = state_costate_o_approx_finalsoln[6:8]
    thrust_acc_x_o_pls, thrust_acc_y_o_pls, _, _, _, _, _, _ = \
        control_thrust_acceleration(
            min_type                 = optimization_parameters['min_type']              ,
            covel_x                  = state_costate_o_approx_finalsoln[6]              ,
            covel_y                  = state_costate_o_approx_finalsoln[7]              ,
            use_thrust_acc_limits    = inequality_parameters['use_thrust_acc_limits']   ,
            use_thrust_acc_smoothing = inequality_parameters['use_thrust_acc_smoothing'],
            thrust_acc_min           = inequality_parameters['thrust_acc_min']          ,
            thrust_acc_max           = inequality_parameters['thrust_acc_max']          ,
            use_thrust_limits        = inequality_parameters['use_thrust_limits']       ,
            use_thrust_smoothing     = inequality_parameters['use_thrust_smoothing']    ,
            thrust_min               = inequality_parameters['thrust_min']              ,
            thrust_max               = inequality_parameters['thrust_max']              ,
            k_steepness              = inequality_parameters['k_steepness']             ,
            mass                     = mass_f_mns                                       ,
        )
    ham_o_pls = \
        compute_hamiltonian(
            min_type     = optimization_parameters['min_type'],
            vel_x        = state_costate_o_approx_finalsoln[2],
            vel_y        = state_costate_o_approx_finalsoln[3],
            copos_x      = state_costate_o_approx_finalsoln[4],
            copos_y      = state_costate_o_approx_finalsoln[5],
            covel_x      = state_costate_o_approx_finalsoln[6],
            covel_y      = state_costate_o_approx_finalsoln[7],
            thrust_acc_x = thrust_acc_x_o_pls                 ,
            thrust_acc_y = thrust_acc_y_o_pls                 ,
            acc_x        = thrust_acc_x_o_pls                 ,
            acc_y        = thrust_acc_y_o_pls                 ,
        )

    pos_vec_f_mns   = state_costate_f_approx_finalsoln[0:2]
    vel_vec_f_mns   = state_costate_f_approx_finalsoln[2:4]
    copos_vec_f_mns = state_costate_f_approx_finalsoln[4:6]
    covel_vec_f_mns = state_costate_f_approx_finalsoln[6:8]
    thrust_acc_x_f_mns, thrust_acc_y_f_mns, _, _, _, _, _, _ = \
        control_thrust_acceleration(
            min_type                 = optimization_parameters['min_type']              ,
            covel_x                  = state_costate_f_approx_finalsoln[6]              ,
            covel_y                  = state_costate_f_approx_finalsoln[7]              ,
            use_thrust_acc_limits    = inequality_parameters['use_thrust_acc_limits']   ,
            use_thrust_acc_smoothing = inequality_parameters['use_thrust_acc_smoothing'],
            thrust_acc_min           = inequality_parameters['thrust_acc_min']          ,
            thrust_acc_max           = inequality_parameters['thrust_acc_max']          ,
            use_thrust_limits        = inequality_parameters['use_thrust_limits']       ,
            use_thrust_smoothing     = inequality_parameters['use_thrust_smoothing']    ,
            thrust_min               = inequality_parameters['thrust_min']              ,
            thrust_max               = inequality_parameters['thrust_max']              ,
            k_steepness              = inequality_parameters['k_steepness']             ,
            mass                     = mass_f_mns                                       ,
        )
    ham_f_mns = \
        compute_hamiltonian(
            min_type     = optimization_parameters['min_type'],
            vel_x        = state_costate_f_approx_finalsoln[2],
            vel_y        = state_costate_f_approx_finalsoln[3],
            copos_x      = state_costate_f_approx_finalsoln[4],
            copos_y      = state_costate_f_approx_finalsoln[5],
            covel_x      = state_costate_f_approx_finalsoln[6],
            covel_y      = state_costate_f_approx_finalsoln[7],
            thrust_acc_x = thrust_acc_x_f_mns                 ,
            thrust_acc_y = thrust_acc_y_f_mns                 ,
            acc_x        = thrust_acc_x_f_mns                 ,
            acc_y        = thrust_acc_y_f_mns                 ,
        )

    if equality_parameters['time'     ]['o']['mode'] == 'free': equality_parameters['time'     ]['o']['mns'] =      time_o_pls
    if equality_parameters['pos_vec'  ]['o']['mode'] == 'free': equality_parameters['pos_vec'  ]['o']['mns'] =   pos_vec_o_pls
    if equality_parameters['vel_vec'  ]['o']['mode'] == 'free': equality_parameters['vel_vec'  ]['o']['mns'] =   vel_vec_o_pls
    if equality_parameters['copos_vec']['o']['mode'] == 'free': equality_parameters['copos_vec']['o']['mns'] = copos_vec_o_pls
    if equality_parameters['covel_vec']['o']['mode'] == 'free': equality_parameters['covel_vec']['o']['mns'] = covel_vec_o_pls
    if equality_parameters['ham'      ]['o']['mode'] == 'free': equality_parameters['ham'      ]['o']['mns'] =       ham_o_pls

    if equality_parameters['time'     ]['f']['mode'] == 'free': equality_parameters['time'     ]['f']['pls'] =      time_f_mns
    if equality_parameters['pos_vec'  ]['f']['mode'] == 'free': equality_parameters['pos_vec'  ]['f']['pls'] =   pos_vec_f_mns
    if equality_parameters['vel_vec'  ]['f']['mode'] == 'free': equality_parameters['vel_vec'  ]['f']['pls'] =   vel_vec_f_mns
    if equality_parameters['copos_vec']['f']['mode'] == 'free': equality_parameters['copos_vec']['f']['pls'] = copos_vec_f_mns
    if equality_parameters['covel_vec']['f']['mode'] == 'free': equality_parameters['covel_vec']['f']['pls'] = covel_vec_f_mns
    if equality_parameters['ham'      ]['f']['mode'] == 'free': equality_parameters['ham'      ]['f']['pls'] =       ham_f_mns

    time_o_mns      = equality_parameters['time'     ]['o']['mns']
    vel_vec_o_mns   = equality_parameters['vel_vec'  ]['o']['mns']
    pos_vec_o_mns   = equality_parameters['pos_vec'  ]['o']['mns']
    copos_vec_o_mns = equality_parameters['copos_vec']['o']['mns']
    covel_vec_o_mns = equality_parameters['covel_vec']['o']['mns']
    ham_o_mns       = equality_parameters['ham'      ]['o']['mns']

    time_f_pls      = equality_parameters['time'     ]['f']['pls']
    vel_vec_f_pls   = equality_parameters['vel_vec'  ]['f']['pls']
    pos_vec_f_pls   = equality_parameters['pos_vec'  ]['f']['pls']
    copos_vec_f_pls = equality_parameters['copos_vec']['f']['pls']
    covel_vec_f_pls = equality_parameters['covel_vec']['f']['pls']
    ham_f_pls       = equality_parameters['ham'      ]['f']['pls']

    error_o_approx_finalsoln_vec = \
        np.hstack([time_o_mns, pos_vec_o_mns, vel_vec_o_mns, copos_vec_o_mns, covel_vec_o_mns, ham_o_mns]) \
        - np.hstack([time_o_pls, state_costate_o_approx_finalsoln[0:8], ham_o_pls])
    error_f_approx_finalsoln_vec = \
        np.hstack([time_f_pls, pos_vec_f_pls, vel_vec_f_pls, copos_vec_f_pls, covel_vec_f_pls, ham_f_pls]) \
        - np.hstack([time_f_mns, state_costate_f_approx_finalsoln[0:8], ham_f_mns])

    # Print error check
    print("\n  Error Check: Time, State, Co-State, and Hamiltonian")
    print(
        "                      "
        f" {'Time-o':>14s}"
        f" {   'Pos-Xo':>14s} {   'Pos-Yo':>14s}"
        f" {   'Vel-Xo':>14s} {   'Vel-Yo':>14s}"
        f" {'Co-Pos-Xo':>14s} {'Co-Pos-Yo':>14s}"
        f" {'Co-Vel-Xo':>14s} {'Co-Vel-Yo':>14s}"
        f" {    'Ham-o':>14s}"
    )
    if min_type == 'fuel':
        print(f"                       {'s':>14s} {'m':>14s} {'m':>14s} {'m/s':>14s} {'m/s':>14s} {'1/s':>14s} {'1/s':>14s} {'1':>14s} {'1':>14s} {'m/s^2':>14s}")
    else: # assume min_type == 'energy'
        print(f"                       {'s':>14s} {'m':>14s} {'m':>14s} {'m/s':>14s} {'m/s':>14s} {'m/s^3':>14s} {'m/s^3':>14s} {'m/s^2':>14s} {'m/s^2':>14s} {'m^2/s^4':>14s}")
    print(
        f"                      "
        f" {equality_parameters[     'time']['o']['mode']:>14s}" 
        f" {equality_parameters[  'pos_vec']['o']['mode']:>14s} {equality_parameters[  'pos_vec']['o']['mode']:>14s}"
        f" {equality_parameters[  'vel_vec']['o']['mode']:>14s} {equality_parameters[  'vel_vec']['o']['mode']:>14s}"
        f" {equality_parameters['copos_vec']['o']['mode']:>14s} {equality_parameters['copos_vec']['o']['mode']:>14s}"
        f" {equality_parameters['covel_vec']['o']['mode']:>14s} {equality_parameters['covel_vec']['o']['mode']:>14s}"
        f" {equality_parameters[      'ham']['o']['mode']:>14s}"
    )
    print(
        "        Target Value :"
        f" {     time_o_mns   :>14.6e}"
        f" {  pos_vec_o_mns[0]:>14.6e} {  pos_vec_o_mns[1]:>14.6e}"
        f" {  vel_vec_o_mns[0]:>14.6e} {  vel_vec_o_mns[1]:>14.6e}"
        f" {copos_vec_o_mns[0]:>14.6e} {copos_vec_o_mns[1]:>14.6e}"
        f" {covel_vec_o_mns[0]:>14.6e} {covel_vec_o_mns[1]:>14.6e}"
        f" {      ham_o_mns   :>14.6e}"
    )
    print(
        "        Smooth Value :"
        f" {         time_o_pls                :>14.6e}"
        f" {state_costate_o_approx_finalsoln[0]:>14.6e} {state_costate_o_approx_finalsoln[1]:>14.6e}"
        f" {state_costate_o_approx_finalsoln[2]:>14.6e} {state_costate_o_approx_finalsoln[3]:>14.6e}"
        f" {state_costate_o_approx_finalsoln[4]:>14.6e} {state_costate_o_approx_finalsoln[5]:>14.6e}"
        f" {state_costate_o_approx_finalsoln[6]:>14.6e} {state_costate_o_approx_finalsoln[7]:>14.6e}"
        f" {          ham_o_pls                :>14.6e}"
    )
    print(
        "               Error :"
        f" {error_o_approx_finalsoln_vec[0]:>14.6e} {error_o_approx_finalsoln_vec[1]:>14.6e}"
        f" {error_o_approx_finalsoln_vec[2]:>14.6e} {error_o_approx_finalsoln_vec[3]:>14.6e}"
        f" {error_o_approx_finalsoln_vec[4]:>14.6e} {error_o_approx_finalsoln_vec[5]:>14.6e}"
        f" {error_o_approx_finalsoln_vec[6]:>14.6e} {error_o_approx_finalsoln_vec[7]:>14.6e}"
        f" {error_o_approx_finalsoln_vec[8]:>14.6e} {error_o_approx_finalsoln_vec[9]:>14.6e}"
    )
    print(
        "    Non-Smooth Value :"
        f" {   time_o_pls         :>14.6e}"
        f" {  state_o_finalsoln[0]:>14.6e} {  state_o_finalsoln[1]:>14.6e}"
        f" {  state_o_finalsoln[2]:>14.6e} {  state_o_finalsoln[3]:>14.6e}"
        f" {costate_o_finalsoln[0]:>14.6e} {costate_o_finalsoln[1]:>14.6e}"
        f" {costate_o_finalsoln[2]:>14.6e} {costate_o_finalsoln[3]:>14.6e}"
        f" {    ham_o_pls         :>14.6e}"
    )
    print(
        "               Error :"
        f" {error_o_finalsoln_vec[0]:>14.6e} {error_o_finalsoln_vec[1]:>14.6e}"
        f" {error_o_finalsoln_vec[2]:>14.6e} {error_o_finalsoln_vec[3]:>14.6e}"
        f" {error_o_finalsoln_vec[4]:>14.6e} {error_o_finalsoln_vec[5]:>14.6e}"
        f" {error_o_finalsoln_vec[6]:>14.6e} {error_o_finalsoln_vec[7]:>14.6e}"
        f" {error_o_finalsoln_vec[8]:>14.6e} {error_o_finalsoln_vec[9]:>14.6e}"
    )
    print()
    print(
        "                      "
        f" {'Time-f':>14s}"
        f" {   'Pos-Xf':>14s} {   'Pos-Yf':>14s}"
        f" {   'Vel-Xf':>14s} {   'Vel-Yf':>14s}"
        f" {'Co-Pos-Xf':>14s} {'Co-Pos-Yf':>14s}"
        f" {'Co-Vel-Xf':>14s} {'Co-Vel-Yf':>14s}"
        f" {    'Ham-f':>14s}"
    )
    if min_type == 'fuel':
        print(f"                       {'s':>14s} {'m':>14s} {'m':>14s} {'m/s':>14s} {'m/s':>14s} {'1/s':>14s} {'1/s':>14s} {'1':>14s} {'1':>14s} {'m/s^2':>14s}")
    else: # assume min_type == 'energy'
        print(f"                       {'s':>14s} {'m':>14s} {'m':>14s} {'m/s':>14s} {'m/s':>14s} {'m/s^3':>14s} {'m/s^3':>14s} {'m/s^2':>14s} {'m/s^2':>14s} {'m^2/s^4':>14s}")
    print(
        f"                      "
        f" {equality_parameters[     'time']['f']['mode']:>14s}" 
        f" {equality_parameters[  'pos_vec']['f']['mode']:>14s} {equality_parameters[  'pos_vec']['f']['mode']:>14s}"
        f" {equality_parameters[  'vel_vec']['f']['mode']:>14s} {equality_parameters[  'vel_vec']['f']['mode']:>14s}"
        f" {equality_parameters['copos_vec']['f']['mode']:>14s} {equality_parameters['copos_vec']['f']['mode']:>14s}"
        f" {equality_parameters['covel_vec']['f']['mode']:>14s} {equality_parameters['covel_vec']['f']['mode']:>14s}"
        f" {equality_parameters[      'ham']['f']['mode']:>14s}"
    )
    print(
        "        Target Value :"
        f" {     time_f_pls   :>14.6e}"
        f" {  pos_vec_f_pls[0]:>14.6e} {  pos_vec_f_pls[1]:>14.6e}"
        f" {  vel_vec_f_pls[0]:>14.6e} {  vel_vec_f_pls[1]:>14.6e}"
        f" {copos_vec_f_pls[0]:>14.6e} {copos_vec_f_pls[1]:>14.6e}"
        f" {covel_vec_f_pls[0]:>14.6e} {covel_vec_f_pls[1]:>14.6e}"
        f" {      ham_f_pls   :>14.6e}"
    )
    print(
        "        Smooth Value :"
        f" {time_f_mns:>14.6e}"
        f" {state_costate_f_approx_finalsoln[0]:>14.6e} {state_costate_f_approx_finalsoln[1]:>14.6e}"
        f" {state_costate_f_approx_finalsoln[2]:>14.6e} {state_costate_f_approx_finalsoln[3]:>14.6e}"
        f" {state_costate_f_approx_finalsoln[4]:>14.6e} {state_costate_f_approx_finalsoln[5]:>14.6e}"
        f" {state_costate_f_approx_finalsoln[6]:>14.6e} {state_costate_f_approx_finalsoln[7]:>14.6e}"
        f" {ham_f_mns:>14.6e}"
    )
    print(
        "               Error :"
        f" {error_f_approx_finalsoln_vec[0]:>14.6e} {error_f_approx_finalsoln_vec[1]:>14.6e}"
        f" {error_f_approx_finalsoln_vec[2]:>14.6e} {error_f_approx_finalsoln_vec[3]:>14.6e}"
        f" {error_f_approx_finalsoln_vec[4]:>14.6e} {error_f_approx_finalsoln_vec[5]:>14.6e}"
        f" {error_f_approx_finalsoln_vec[6]:>14.6e} {error_f_approx_finalsoln_vec[7]:>14.6e}"
        f" {error_f_approx_finalsoln_vec[8]:>14.6e} {error_f_approx_finalsoln_vec[9]:>14.6e}"
    )
    print(
        "    Non-Smooth Value :"
        f" {   time_f_mns         :>14.6e}"
        f" {  state_f_finalsoln[0]:>14.6e} {  state_f_finalsoln[1]:>14.6e}"
        f" {  state_f_finalsoln[2]:>14.6e} {  state_f_finalsoln[3]:>14.6e}"
        f" {costate_f_finalsoln[0]:>14.6e} {costate_f_finalsoln[1]:>14.6e}"
        f" {costate_f_finalsoln[2]:>14.6e} {costate_f_finalsoln[3]:>14.6e}"
        f" {ham_f_mns:>14.6e}"
    )
    print(
        "               Error :"
        f" {error_f_finalsoln_vec[0]:>14.6e} {error_f_finalsoln_vec[1]:>14.6e}"
        f" {error_f_finalsoln_vec[2]:>14.6e} {error_f_finalsoln_vec[3]:>14.6e}"
        f" {error_f_finalsoln_vec[4]:>14.6e} {error_f_finalsoln_vec[5]:>14.6e}"
        f" {error_f_finalsoln_vec[6]:>14.6e} {error_f_finalsoln_vec[7]:>14.6e}"
        f" {error_f_finalsoln_vec[8]:>14.6e} {error_f_finalsoln_vec[9]:>14.6e}"
    )

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