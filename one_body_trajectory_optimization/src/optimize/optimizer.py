import numpy as np
from tqdm                      import tqdm
from pathlib                   import Path
from scipy.integrate           import solve_ivp
from scipy.optimize            import root
from src.model.dynamics        import one_body_dynamics__indirect
# from src.initial_guess.guesser import generate_guess
from src.plot.final_results    import plot_final_results


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


def tpbvp_objective_and_jacobian(
        decision_state               : np.ndarray,
        optimization_parameters      : dict      ,
        integration_state_parameters : dict      ,
        equality_parameters          : dict      ,
        inequality_parameters        : dict      ,
    ):
    """
    Objective function that also returns the analytical Jacobian.
    """

    # Unpack
    include_jacobian  =      optimization_parameters['include_jacobian'        ]
    time_span         = integration_state_parameters['time_span'               ]
    pos_vec_o_mns     =          equality_parameters['pos_vec_o_mns'           ]
    vel_vec_o_mns     =          equality_parameters['vel_vec_o_mns'           ]
    pos_vec_f_pls     =          equality_parameters['pos_vec_f_pls'           ]
    vel_vec_f_pls     =          equality_parameters['vel_vec_f_pls'           ]

    # Initial state
    state_costate_o = np.hstack([pos_vec_o_mns, vel_vec_o_mns, decision_state])
    
    # Solve initial value problem
    soln_ivp = \
        _solve_ivp_func(
            time_span                   ,
            state_costate_o             ,
            optimization_parameters     ,
            integration_state_parameters,
            inequality_parameters       ,
        )

    # Extract final state and final STM
    state_costate_scstm_f = soln_ivp.sol(time_span[1])
    state_costate_f       = state_costate_scstm_f[:8]
    if include_jacobian:
        stm_of = state_costate_scstm_f[8:8+8**2].reshape((8,8))
    
    # Calculate the error vector and error vector Jacobian
    #   jacobian = d(state_final) / d(costate_initial)
    error = state_costate_f[:4] - np.hstack([pos_vec_f_pls, vel_vec_f_pls])
    if include_jacobian:
        error_jacobian = stm_of[0:4, 4:8]

    # Pack up: error and error-jacobian
    if include_jacobian:
        return error, error_jacobian
    else:
        return error
    

def _solve_for_root(
        decision_state_initguess,
        optimization_parameters,
        integration_state_parameters,
        equality_parameters,
        inequality_parameters,
        options_root
    ):
    """
    Helper function to call the root solver.
    """
    root_func = lambda decision_state: tpbvp_objective_and_jacobian(
        decision_state,
        optimization_parameters,
        integration_state_parameters,
        equality_parameters,
        inequality_parameters,
    )
    return root(
        root_func,
        decision_state_initguess,
        method  = 'lm',
        tol     = 1.0e-11,
        jac     = optimization_parameters['include_jacobian'],
        options = options_root,
    )


def _solve_ivp_func(
        time_span                     ,
        state_costate_o               ,
        optimization_parameters       ,
        integration_state_parameters  ,
        inequality_parameters         ,
    ):
    """
    Solve IVP function for the final solution.
    """

    min_type                 =      optimization_parameters['min_type'                ]
    include_jacobian         =      optimization_parameters['include_jacobian'        ]
    use_thrust_acc_limits    =        inequality_parameters['use_thrust_acc_limits'   ]
    use_thrust_acc_smoothing =        inequality_parameters['use_thrust_acc_smoothing']
    thrust_acc_min           =        inequality_parameters['thrust_acc_min'          ]
    thrust_acc_max           =        inequality_parameters['thrust_acc_max'          ]
    use_thrust_limits        =        inequality_parameters['use_thrust_limits'       ]
    use_thrust_smoothing     =        inequality_parameters['use_thrust_smoothing'    ]
    thrust_min               =        inequality_parameters['thrust_min'              ]
    thrust_max               =        inequality_parameters['thrust_max'              ]
    k_steepness              =        inequality_parameters['k_steepness'             ]
    time_span                = integration_state_parameters['time_span'               ]
    mass_o                   = integration_state_parameters['mass_o'                  ]
    include_scstm            = integration_state_parameters['include_scstm'           ]
    post_process             = integration_state_parameters['post_process'            ]

    if include_jacobian:
        integration_state_parameters['include_scstm'] = True
        stm_oo                         = np.identity(8).flatten()
        state_costate_scstm_mass_obj_o = np.hstack([state_costate_o, stm_oo])
    else:
        integration_state_parameters['include_scstm'] = False
        state_costate_scstm_mass_obj_o = state_costate_o
    include_scstm = integration_state_parameters['include_scstm']
    if use_thrust_limits:
        state_costate_scstm_mass_obj_o = np.hstack([state_costate_scstm_mass_obj_o, mass_o])

    time_eval_points = np.linspace(time_span[0], time_span[1], 401)

    solve_ivp_func = \
        lambda time, state_costate_scstm_mass_obj: \
            one_body_dynamics__indirect(
                time                                                   ,
                state_costate_scstm_mass_obj                           ,
                include_scstm                = include_scstm           ,
                min_type                     = min_type                ,
                use_thrust_acc_limits        = use_thrust_acc_limits   ,
                use_thrust_acc_smoothing     = use_thrust_acc_smoothing,
                thrust_acc_min               = thrust_acc_min          ,
                thrust_acc_max               = thrust_acc_max          ,
                use_thrust_limits            = use_thrust_limits       ,
                use_thrust_smoothing         = use_thrust_smoothing    ,
                thrust_min                   = thrust_min              ,
                thrust_max                   = thrust_max              ,
                post_process                 = post_process            ,
                k_steepness                  = k_steepness             ,
            )

    soln_ivp = \
        solve_ivp(
            solve_ivp_func                                   ,
            time_span                                        ,
            state_costate_scstm_mass_obj_o                   ,
            t_eval                         = time_eval_points,
            dense_output                   = True            ,
            method                         = 'RK45'          ,
            rtol                           = 1.0e-12         ,
            atol                           = 1.0e-12         ,
        )

    return soln_ivp


def _solve_for_root_and_compute_progress(
        decision_state_initguess    ,
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
        options_root                ,
    ):

    # Unpack
    pos_vec_o_mns =          equality_parameters['pos_vec_o_mns']
    vel_vec_o_mns =          equality_parameters['vel_vec_o_mns']
    time_span     = integration_state_parameters['time_span'    ]

    # Solve root problem
    soln_root = \
        _solve_for_root(
            decision_state_initguess    ,
            optimization_parameters     ,
            integration_state_parameters,
            equality_parameters         ,
            inequality_parameters       ,
            options_root                ,
        )

    # Update decision-state initial guess and the state-costate
    decision_state_initguess = soln_root.x
    state_costate_o = np.hstack([pos_vec_o_mns, vel_vec_o_mns, decision_state_initguess])

    # Solve initial value problem
    soln_ivp = \
        _solve_ivp_func(
            time_span                   ,
            state_costate_o             ,
            optimization_parameters     ,
            integration_state_parameters,
            inequality_parameters       ,
        )

    return soln_root, soln_ivp


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
    time_span             = integration_state_parameters['time_span'            ]
    mass_o                = integration_state_parameters['mass_o'               ] 
    pos_vec_o_mns         =          equality_parameters['pos_vec_o_mns'        ]
    vel_vec_o_mns         =          equality_parameters['vel_vec_o_mns'        ]
    pos_vec_f_pls         =          equality_parameters['pos_vec_f_pls'        ]
    vel_vec_f_pls         =          equality_parameters['vel_vec_f_pls'        ]
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
    optimization_parameters['include_jacobian']   = True
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
            _solve_for_root_and_compute_progress(
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
    optimization_parameters['include_jacobian']       = True  # should be True
    integration_state_parameters['include_scstm']     = True  # should be True
    integration_state_parameters['post_process']      = False # should be False
    inequality_parameters['use_thrust_acc_smoothing'] = False # should be False
    inequality_parameters['use_thrust_smoothing']     = False # should be False
    soln_root, soln_ivp = \
        _solve_for_root_and_compute_progress(
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
                value_construct = '  '.join([str(f"{val:>13.6e}") for val in value])
            elif len(value.shape) == 2:
                value_construct = ['  '.join( str(f"{val:>13.6e}") for val in row) for row in value]
                value_construct = '\n            : '.join(value_construct)
        else:
            value_construct = value
        print(f"    {key:>7s} : {value_construct}")
    
    # Final solution: post-process step
    optimization_parameters['include_jacobian']       = False # should be False
    integration_state_parameters['include_scstm']     = False # should be False
    integration_state_parameters['post_process']      = True  # should be True
    inequality_parameters['use_thrust_acc_smoothing'] = False # should be False
    inequality_parameters['use_thrust_smoothing']     = False # should be False
    decision_state_initguess       = soln_root.x
    state_costate_o                = np.hstack([pos_vec_o_mns, vel_vec_o_mns, decision_state_initguess])
    optimal_control_objective_o    = np.float64(0.0)
    state_costate_scstm_mass_obj_o = np.hstack([state_costate_o, mass_o, optimal_control_objective_o])
    soln_ivp = \
        _solve_ivp_func(
            time_span                     ,
            state_costate_scstm_mass_obj_o,
            optimization_parameters       ,
            integration_state_parameters  ,
            inequality_parameters         ,
        )
    
    results_finalsoln = soln_ivp
    state_f_finalsoln = results_finalsoln.y[0:4, -1]

    # Final solution: approx and true
    results_approx_finalsoln = results_k_idx[k_idxinitguess_to_idxfinsoln[-1]]
    state_f_approx_finalsoln = results_approx_finalsoln.y[0:4, -1]

    # Final solution: check final state error
    error_approx_finalsoln_vec = state_f_approx_finalsoln - np.hstack([pos_vec_f_pls, vel_vec_f_pls])
    error_finalsoln_vec        = state_f_finalsoln        - np.hstack([pos_vec_f_pls, vel_vec_f_pls])

    print("\n  State Error Check")
    print(f"             {'Pos-Xf':>14s} {'Pos-Yf':>14s} {'Vel-Xf':>14s} {'Vel-Yf':>14s}")
    print(f"             {     'm':>14s} {     'm':>14s} {   'm/s':>14s} {   'm/s':>14s}")
    print(f"    Target : {             pos_vec_f_pls[0]:>14.6e} {             pos_vec_f_pls[1]:>14.6e} {             vel_vec_f_pls[0]:>14.6e} {             vel_vec_f_pls[1]:>14.6e}")
    print(f"    Approx : {  state_f_approx_finalsoln[0]:>14.6e} {  state_f_approx_finalsoln[1]:>14.6e} {  state_f_approx_finalsoln[2]:>14.6e} {  state_f_approx_finalsoln[3]:>14.6e}")
    print(f"    Error  : {error_approx_finalsoln_vec[0]:>14.6e} {error_approx_finalsoln_vec[1]:>14.3e} {error_approx_finalsoln_vec[2]:>14.6e} {error_approx_finalsoln_vec[3]:>14.6e}")
    print(f"    Actual : {         state_f_finalsoln[0]:>14.6e} {         state_f_finalsoln[1]:>14.6e} {         state_f_finalsoln[2]:>14.6e} {         state_f_finalsoln[3]:>14.6e}")
    print(f"    Error  : {       error_finalsoln_vec[0]:>14.6e} {       error_finalsoln_vec[1]:>14.3e} {       error_finalsoln_vec[2]:>14.6e} {       error_finalsoln_vec[3]:>14.6e}")

    # Final solution: enforce initial and final co-state boundary conditions (trivial right now)
    equality_parameters['copos_vec_o_mns'] = decision_state_initguess[0:2]
    equality_parameters['covel_vec_o_mns'] = decision_state_initguess[2:4]
    equality_parameters['copos_vec_o_pls'] = decision_state_initguess[0:2]
    equality_parameters['covel_vec_o_pls'] = decision_state_initguess[2:4]
    equality_parameters['copos_vec_f_mns'] = results_finalsoln.y[4:6, -1]
    equality_parameters['covel_vec_f_mns'] = results_finalsoln.y[6:8, -1]
    equality_parameters['copos_vec_f_pls'] = results_finalsoln.y[4:6, -1]
    equality_parameters['covel_vec_f_pls'] = results_finalsoln.y[6:8, -1]

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