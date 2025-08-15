import numpy as np
from tqdm                      import tqdm
from pathlib                   import Path
from scipy.integrate           import solve_ivp
from scipy.optimize            import root
from src.model.dynamics        import free_body_dynamics__indirect
# from src.initial_guess.guesser import generate_guess
from src.plot.final_results    import plot_final_results

def tpbvp_objective_and_jacobian(
        decision_state               : np.ndarray                     ,
        time_span                    : np.ndarray                     ,
        boundary_condition_pos_vec_o : np.ndarray                     ,
        boundary_condition_vel_vec_o : np.ndarray                     ,
        boundary_condition_pos_vec_f : np.ndarray                     ,
        boundary_condition_vel_vec_f : np.ndarray                     ,
        min_type                     : str        = 'energy'          ,
        mass_o                       : np.float64 = np.float64(1.0e+3),
        use_thrust_acc_limits        : bool       = False             ,
        use_thrust_acc_smoothing     : bool       = False             ,
        thrust_acc_min               : np.float64 = np.float64(0.0e+0),
        thrust_acc_max               : np.float64 = np.float64(1.0e+1),
        use_thrust_limits            : bool       = False             ,
        use_thrust_smoothing         : bool       = False             ,
        thrust_min                   : np.float64 = np.float64(0.0e+0),
        thrust_max                   : np.float64 = np.float64(1.0e+1),
        k_steepness                  : np.float64 = np.float64(1.0e+0),
        include_jacobian             : bool       = False             ,
    ):
    """
    Objective function that also returns the analytical Jacobian.
    """

    # Initial state and stm
    state_costate_o = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decision_state])
    if include_jacobian:
        include_scstm         = True
        stm_oo                = np.identity(8).flatten()
        state_costate_scstm_o = np.hstack([state_costate_o, stm_oo])
    else:
        include_scstm         = False
        state_costate_scstm_o = state_costate_o
    if use_thrust_limits:
        state_costate_scstm_o = np.hstack([state_costate_scstm_o, mass_o])

    # Integrate
    solve_ivp_func = \
        lambda time, state_costate_scstm: \
            free_body_dynamics__indirect(
                time                                               ,
                state_costate_scstm                                ,
                include_scstm            = include_scstm           ,
                min_type                 = min_type                ,
                use_thrust_acc_limits    = use_thrust_acc_limits   ,
                use_thrust_acc_smoothing = use_thrust_acc_smoothing,
                thrust_acc_min           = thrust_acc_min          ,
                thrust_acc_max           = thrust_acc_max          ,
                use_thrust_limits        = use_thrust_limits       ,
                use_thrust_smoothing     = use_thrust_smoothing    ,
                thrust_min               = thrust_min              ,
                thrust_max               = thrust_max              ,
                k_steepness              = k_steepness             ,
            )
    soln = \
        solve_ivp(
            solve_ivp_func                 ,
            time_span                      ,
            state_costate_scstm_o          ,
            dense_output          = True   , 
            method                = 'RK45' ,
            rtol                  = 1.0e-12,
            atol                  = 1.0e-12,
        )

    # Extract final state and final STM
    state_costate_scstm_f = soln.sol(time_span[1])
    state_costate_f       = state_costate_scstm_f[:8]
    if include_jacobian:
        stm_of = state_costate_scstm_f[8:8+8**2].reshape((8,8))
    
    # Calculate the error vector and error vector Jacobian
    #   jacobian = d(state_final) / d(costate_initial)
    error = state_costate_f[:4] - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])
    if include_jacobian:
        error_jacobian = stm_of[0:4, 4:8]

    # Pack up: error and error-jacobian
    if include_jacobian:
        return error, error_jacobian
    else:
        return error
    

# def optimal_trajectory_solve(
#         time_span                    : np.ndarray                                         ,
#         boundary_condition_pos_vec_o : np.ndarray                                         ,
#         boundary_condition_vel_vec_o : np.ndarray                                         ,
#         boundary_condition_pos_vec_f : np.ndarray                                         ,
#         boundary_condition_vel_vec_f : np.ndarray                                         ,
#         min_type                     : str        = 'energy'                              ,
#         use_thrust_acc_limits        : bool       = True                                  ,
#         thrust_acc_min               : np.float64 = np.float64(0.0e+0)                    ,
#         thrust_acc_max               : np.float64 = np.float64(1.0e+1)                    ,
#         use_thrust_limits            : bool       = False                                 ,
#         thrust_min                   : np.float64 = np.float64(0.0e+0)                    ,
#         thrust_max                   : np.float64 = np.float64(1.0e+1)                    ,
#         k_idxinitguess               : np.float64 = np.float64(1.0e-1)                    ,
#         k_idxfinsoln                 : np.float64 = np.float64(1.0e+1)                    ,
#         k_idxdivs                    : int        = 100                                   ,
#         init_guess_steps             : int        = 3000                                  ,
#         mass_o                       : np.float64 = np.float64(1.0e+3)                    ,
#         input_filepath               : Path       = Path('input/examples/example_01.json'),
#         output_folderpath            : Path       = Path('output/')                       ,
#     ):
def optimal_trajectory_solve(
        files_folders_params : dict,
    ):
    """
    Main solver that implements the two-stage continuation process
    using the unified smoothed dynamics.
    """

    # Generate initial guess for the costates
    decision_state_initguess = \
        generate_guess(
            files_folders_params,
        )

    # Unpack files and folders parameters
    time_span                    = files_folders_params['time_span'                   ]
    boundary_condition_pos_vec_o = files_folders_params['boundary_condition_pos_vec_o']
    boundary_condition_vel_vec_o = files_folders_params['boundary_condition_vel_vec_o']
    boundary_condition_pos_vec_f = files_folders_params['boundary_condition_pos_vec_f']
    boundary_condition_vel_vec_f = files_folders_params['boundary_condition_vel_vec_f']
    input_filepath               = files_folders_params['input_filepath'       ]
    output_folderpath            = files_folders_params['output_folderpath'    ]
    min_type                     = files_folders_params.get('min_type'             , 'energy')
    mass_o                       = files_folders_params.get('mass_o'               , 1.0e+3  )
    use_thrust_acc_limits        = files_folders_params.get('use_thrust_acc_limits', True    )
    thrust_acc_min               = files_folders_params.get('thrust_acc_min'       , 0.0e+0  )
    thrust_acc_max               = files_folders_params.get('thrust_acc_max'       , 1.0e+1  )
    use_thrust_limits            = files_folders_params.get('use_thrust_limits'    , False   )
    thrust_min                   = files_folders_params.get('thrust_min'           , 0.0e+0  )
    thrust_max                   = files_folders_params.get('thrust_max'           , 1.0e+1  )
    k_idxinitguess               = files_folders_params.get('k_idxinitguess'       , 1.0e-1  )
    k_idxfinsoln                 = files_folders_params.get('k_idxfinsoln'         , 1.0e+1  )
    k_idxdivs                    = files_folders_params.get('k_idxdivs'            , 100     )
    init_guess_steps             = files_folders_params.get('init_guess_steps'     , 3000    )
    
    # Optimize and enforce thrust or thrust-acc constraints
    print("\n\nOPTIMIZATION PROCESS")

    # Solve for the optimal min-fuel or min-energy trajectory

    # Thrust- or Thrust-Acc-Steepness Continuation Process
    if use_thrust_acc_limits or use_thrust_limits:
        print("  Thrust- or Thrust-Acc-Steepness Continuation Process")

    # Loop initialization
    results_k_idx    = {}
    include_jacobian = True # temp
    options_root     = {
        'maxiter' : 100 * len(decision_state_initguess), # 100 * n
        'ftol'    : 1e-8, # 1e-8
        'xtol'    : 1e-8, # 1e-8
        'gtol'    : 1e-8, # 1e-8
    }
    k_idxinitguess_to_idxfinsoln = np.logspace(np.log(k_idxinitguess), np.log(k_idxfinsoln), k_idxdivs)

    # Loop
    for idx, k_idx in tqdm(enumerate(k_idxinitguess_to_idxfinsoln), desc="Processing", leave=False, total=len(k_idxinitguess_to_idxfinsoln)):
        
        # Define root function
        root_func = \
            lambda decision_state: \
                tpbvp_objective_and_jacobian(
                    decision_state                                      , 
                    time_span                                           ,
                    boundary_condition_pos_vec_o                        ,
                    boundary_condition_vel_vec_o                        ,
                    boundary_condition_pos_vec_f                        ,
                    boundary_condition_vel_vec_f                        ,
                    min_type                     = min_type             ,
                    mass_o                       = mass_o               ,
                    use_thrust_acc_limits        = use_thrust_acc_limits,
                    use_thrust_acc_smoothing     = True                 ,
                    thrust_acc_min               = thrust_acc_min       ,
                    thrust_acc_max               = thrust_acc_max       ,
                    use_thrust_limits            = use_thrust_limits    ,
                    use_thrust_smoothing         = True                 ,
                    thrust_min                   = thrust_min           ,
                    thrust_max                   = thrust_max           ,
                    k_steepness                  = k_idx                ,
                    include_jacobian             = include_jacobian     ,
                )

        # Root solve
        soln_root = \
            root(
                root_func                                 ,
                decision_state_initguess                  ,
                method                  = 'lm'            ,
                tol                     = 1e-11           ,
                jac                     = include_jacobian,
                options                 = options_root    ,
            )
        
        # Compute progress
        decision_state_initguess = soln_root.x
        state_costate_o          = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decision_state_initguess])
        if use_thrust_limits:
            state_costate_mass_o = np.hstack([state_costate_o, mass_o])
        else:
            state_costate_mass_o = state_costate_o
        time_eval_points = np.linspace(time_span[0], time_span[1], 201)
        solve_ivp_func = \
            lambda time, state_costate_scstm: \
                free_body_dynamics__indirect(
                    time                                            ,
                    state_costate_scstm                             ,
                    include_scstm            = False                ,
                    min_type                 = min_type             ,
                    use_thrust_acc_limits    = use_thrust_acc_limits,
                    use_thrust_acc_smoothing = True                 ,
                    thrust_acc_min           = thrust_acc_min       ,
                    thrust_acc_max           = thrust_acc_max       ,
                    use_thrust_limits        = use_thrust_limits    ,
                    use_thrust_smoothing     = True                 ,
                    thrust_min               = thrust_min           ,
                    thrust_max               = thrust_max           ,
                    k_steepness              = k_idx                ,
                )
        soln_ivp = \
            solve_ivp(
                solve_ivp_func                 ,
                time_span                      ,
                state_costate_mass_o           ,
                t_eval       = time_eval_points,
                dense_output = True            , 
                method       = 'RK45'          ,
                rtol         = 1e-12           ,
                atol         = 1e-12           ,
            )
        results_k_idx[k_idx] = soln_ivp
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
    root_func = \
        lambda decision_state: \
            tpbvp_objective_and_jacobian(
                decision_state                                      , 
                time_span                                           ,
                boundary_condition_pos_vec_o                        ,
                boundary_condition_vel_vec_o                        ,
                boundary_condition_pos_vec_f                        ,
                boundary_condition_vel_vec_f                        ,
                min_type                     = min_type             ,
                mass_o                       = mass_o               ,
                use_thrust_acc_limits        = use_thrust_acc_limits,
                use_thrust_acc_smoothing     = False                ,
                thrust_acc_min               = thrust_acc_min       ,
                thrust_acc_max               = thrust_acc_max       ,
                use_thrust_limits            = use_thrust_limits    ,
                use_thrust_smoothing         = False                ,
                thrust_min                   = thrust_min           ,
                thrust_max                   = thrust_max           ,
                k_steepness                  = k_idx                ,
                include_jacobian             = include_jacobian     ,
            )
    soln_root = \
        root(
            root_func                                  ,
            decision_state_initguess                   ,
            method                   = 'lm'            ,
            tol                      = 1e-11           ,
            jac                      = include_jacobian,
            options                  = options_root    ,
        )
    print("\n\nFINAL SOLUTION PROCESS")
    print("  Root-Solve Results")
    # print(soln_root)
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
    decision_state_initguess       = soln_root.x
    state_costate_o                = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decision_state_initguess])
    optimal_control_objective_o    = np.float64(0.0)
    state_costate_scstm_mass_obj_o = np.hstack([state_costate_o, mass_o, optimal_control_objective_o])
    time_eval_points               = np.linspace(time_span[0], time_span[1], 401)
    solve_ivp_func = \
        lambda time, state_costate_scstm_mass_obj: \
            free_body_dynamics__indirect(
                time                                                ,
                state_costate_scstm_mass_obj                        ,
                include_scstm                = False                ,
                min_type                     = min_type             ,
                use_thrust_acc_limits        = use_thrust_acc_limits,
                use_thrust_acc_smoothing     = False                ,
                thrust_acc_min               = thrust_acc_min       ,
                thrust_acc_max               = thrust_acc_max       ,
                use_thrust_limits            = use_thrust_limits    ,
                use_thrust_smoothing         = False                ,
                thrust_min                   = thrust_min           ,
                thrust_max                   = thrust_max           ,
                post_process                 = True                 ,
                k_steepness                  = k_idx                ,
            )
    soln_ivp = \
        solve_ivp(
            solve_ivp_func                                   ,
            time_span                                        ,
            state_costate_scstm_mass_obj_o                   ,
            t_eval                         = time_eval_points,
            dense_output                   = True            , 
            method                         = 'RK45'          ,
            rtol                           = 1e-12           ,
            atol                           = 1e-12           ,
        ) 
    results_finalsoln = soln_ivp
    state_f_finalsoln = results_finalsoln.y[0:4, -1]

    # Final solution: approx and true
    results_approx_finalsoln = results_k_idx[k_idxinitguess_to_idxfinsoln[-1]]
    state_f_approx_finalsoln = results_approx_finalsoln.y[0:4, -1]

    # Check final state error
    error_approx_finalsoln_vec = state_f_approx_finalsoln - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])
    error_finalsoln_vec        = state_f_finalsoln        - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])

    print("\n  State Error Check")
    print(f"             {'Pos-Xf':>14s} {'Pos-Yf':>14s} {'Vel-Xf':>14s} {'Vel-Yf':>14s}")
    print(f"             {    'm':>14s} {    'm':>14s} {  'm/s':>14s} {  'm/s':>14s}")
    print(f"    Target : {boundary_condition_pos_vec_f[0]:>14.6e} {boundary_condition_pos_vec_f[1]:>14.6e} {boundary_condition_vel_vec_f[0]:>14.6e} {boundary_condition_vel_vec_f[1]:>14.6e}")
    print(f"    Approx : {    state_f_approx_finalsoln[0]:>14.6e} {    state_f_approx_finalsoln[1]:>14.6e} {    state_f_approx_finalsoln[2]:>14.6e} {    state_f_approx_finalsoln[3]:>14.6e}")
    print(f"    Error  : {  error_approx_finalsoln_vec[0]:>14.6e} {  error_approx_finalsoln_vec[1]:>14.3e} {  error_approx_finalsoln_vec[2]:>14.6e} {  error_approx_finalsoln_vec[3]:>14.6e}")
    print(f"    Actual : {           state_f_finalsoln[0]:>14.6e} {           state_f_finalsoln[1]:>14.6e} {           state_f_finalsoln[2]:>14.6e} {           state_f_finalsoln[3]:>14.6e}")
    print(f"    Error  : {         error_finalsoln_vec[0]:>14.6e} {         error_finalsoln_vec[1]:>14.3e} {         error_finalsoln_vec[2]:>14.6e} {         error_finalsoln_vec[3]:>14.6e}")

    # Enforce initial and final co-state boundary conditions (trivial right now)
    boundary_condition_copos_vec_o = decision_state_initguess[0:2]
    boundary_condition_covel_vec_o = decision_state_initguess[2:4]
    boundary_condition_copos_vec_f = results_finalsoln.y[4:6, -1]
    boundary_condition_covel_vec_f = results_finalsoln.y[6:8, -1]

    # Plot the results
    print("\n  Plot Final Solution Trajectory")
    plot_final_results(
        results_finalsoln                                     ,
        boundary_condition_pos_vec_o                          ,
        boundary_condition_vel_vec_o                          ,
        boundary_condition_pos_vec_f                          ,
        boundary_condition_vel_vec_f                          ,
        boundary_condition_copos_vec_o                        ,
        boundary_condition_covel_vec_o                        ,
        boundary_condition_copos_vec_f                        ,
        boundary_condition_covel_vec_f                        ,
        min_type                       = min_type             ,
        use_thrust_acc_limits          = use_thrust_acc_limits,
        use_thrust_acc_smoothing       = False                ,
        thrust_acc_min                 = thrust_acc_min       ,
        thrust_acc_max                 = thrust_acc_max       ,
        use_thrust_limits              = use_thrust_limits    ,
        use_thrust_smoothing           = False                ,
        thrust_min                     = thrust_min           ,
        thrust_max                     = thrust_max           ,
        k_steepness                    = k_idx                ,
        plot_show                      = True                 ,
        plot_save                      = True                 ,
        input_filepath                 = input_filepath       ,
        output_folderpath              = output_folderpath    ,
    )


def generate_guess(
        files_folders_params,
    ):
    """
    Generates a robust initial guess for the co-states: copos_vec, covel_vec
    """
    print("\n\nINITIAL GUESS PROCESS")

    # Unpack
    time_span                    = files_folders_params['time_span'                   ]
    boundary_condition_pos_vec_o = files_folders_params['boundary_condition_pos_vec_o']
    boundary_condition_vel_vec_o = files_folders_params['boundary_condition_vel_vec_o']
    boundary_condition_pos_vec_f = files_folders_params['boundary_condition_pos_vec_f']
    boundary_condition_vel_vec_f = files_folders_params['boundary_condition_vel_vec_f']
    min_type                     = files_folders_params.get('min_type'             , 'energy')
    mass_o                       = files_folders_params.get('mass_o'               , 1.0e+3  )
    use_thrust_acc_limits        = files_folders_params.get('use_thrust_acc_limits', True    )
    thrust_acc_min               = files_folders_params.get('thrust_acc_min'       , 0.0e+0  )
    thrust_acc_max               = files_folders_params.get('thrust_acc_max'       , 1.0e+1  )
    use_thrust_limits            = files_folders_params.get('use_thrust_limits'    , False   )
    thrust_min                   = files_folders_params.get('thrust_min'           , 0.0e+0  )
    thrust_max                   = files_folders_params.get('thrust_max'           , 1.0e+1  )
    k_steepness                  = files_folders_params.get('k_steepness'          , 1.0e+0  )
    init_guess_steps             = files_folders_params.get('init_guess_steps'     , 3000    )

    # Loop through random guesses for the costates
    print("  Random Initial Guess Generation")
    error_mag_min = np.Inf
    for idx in tqdm(range(init_guess_steps), desc="Processing", leave=False, total=init_guess_steps):
        copos_vec_o        = np.random.uniform(low=-1, high=1, size=2)
        covel_vec_o        = np.random.uniform(low=-1, high=1, size=2)
        decision_state_idx = np.hstack([copos_vec_o, covel_vec_o])
        
        error_idx = \
            tpbvp_objective_and_jacobian(
                decision_state_idx                                  ,
                time_span                                           ,
                boundary_condition_pos_vec_o                        ,
                boundary_condition_vel_vec_o                        ,
                boundary_condition_pos_vec_f                        ,
                boundary_condition_vel_vec_f                        ,
                min_type                     = min_type             ,
                mass_o                       = mass_o               ,
                use_thrust_acc_limits        = use_thrust_acc_limits,
                use_thrust_acc_smoothing     = True                 ,
                thrust_acc_min               = thrust_acc_min       ,
                thrust_acc_max               = thrust_acc_max       ,
                use_thrust_limits            = use_thrust_limits    ,
                use_thrust_smoothing         = True                 ,
                thrust_min                   = thrust_min           ,
                thrust_max                   = thrust_max           ,
                k_steepness                  = k_steepness          ,
                include_jacobian             = False                ,
            )

        error_mag_idx = np.linalg.norm(error_idx)
        if error_mag_idx < error_mag_min:
            idx_min            = idx
            error_mag_min      = error_mag_idx
            decision_state_min = decision_state_idx
            integ_state_min    = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decision_state_min])
            if idx==0:
                tqdm.write(f"                               {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s}")
                tqdm.write(f"          {'Step':>5s} {'Error-Mag':>14s} {'Pos-Xo':>14s} {'Pos-Yo':>14s} {'Vel-Xo':>14s} {'Vel-Yo':>14s} {'Co-Pos-Xo':>14s} {'Co-Pos-Yo':>14s} {'Co-Vel-Xo':>14s} {'Co-Vel-Yo':>14s}")
            integ_state_min_str = ' '.join(f"{x:>14.6e}" for x in integ_state_min)
            tqdm.write(f"     {idx_min:>5d}/{init_guess_steps:>4d} {error_mag_min:>14.6e} {integ_state_min_str}")

    # Pack up and print solution
    costate_o_guess = decision_state_min
    return costate_o_guess

