import numpy as np
from scipy.integrate    import solve_ivp
from scipy.optimize     import root
from src.model.dynamics import one_body_dynamics__indirect, control_thrust_acceleration
# approx_derivative import with fallback for environments missing the public symbol
try:
    from scipy.optimize import approx_derivative  # type: ignore[attr-defined]
except Exception:
    try:
        from scipy.optimize._numdiff import approx_derivative  # type: ignore
    except Exception:
        def approx_derivative(fun, x0, method="3-point", rel_step=1e-6):
            x0 = np.asarray(x0, dtype=float)
            f0 = np.asarray(fun(x0), dtype=float)
            n = x0.size
            m = f0.size
            J = np.empty((m, n), dtype=float)
            for j in range(n):
                h = rel_step * max(1.0, abs(x0[j]))
                xp = x0.copy()
                xm = x0.copy()
                xp[j] += h
                xm[j] -= h
                fp = np.asarray(fun(xp), dtype=float)
                fm = np.asarray(fun(xm), dtype=float)
                J[:, j] = (fp - fm) / (2.0 * h)
            return J


def check_jacobian(
    decision_state,
    optimization_parameters,
    integration_state_parameters,
    equality_parameters,
    inequality_parameters,
    rel_step=1e-6,
    abs_tol=1e-6,
    rel_tol=1e-5,
    n_dir_checks=5,
):
    """
    Finite-difference check for the analytical Jacobian of tpbvp_objective_and_jacobian.
    Returns a dict with error metrics and prints a short report.
    """

    # Analytical error and Jacobian
    opt_params_analytic = dict(optimization_parameters)
    opt_params_analytic['include_jacobian'] = True
    err_analytic, jac_analytic = tpbvp_objective_and_jacobian(
        decision_state,
        opt_params_analytic,
        integration_state_parameters,
        equality_parameters,
        inequality_parameters,
    )

    # Numerical Jacobian by finite differences (vector-valued function)
    opt_params_fd = dict(optimization_parameters)
    opt_params_fd['include_jacobian'] = False

    def f_only(x):
        return tpbvp_objective_and_jacobian(
            x,
            opt_params_fd,
            integration_state_parameters,
            equality_parameters,
            inequality_parameters,
        )

    jac_numeric = approx_derivative(
        f_only,
        decision_state,
        method='3-point',
        rel_step=rel_step,
    )

    # Metrics
    diff = jac_analytic - jac_numeric
    max_abs_err = np.max(np.abs(diff))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(diff) / np.maximum(1.0, np.abs(jac_numeric))
    max_rel_err = np.nanmax(rel)

    i_max, j_max = np.unravel_index(np.abs(diff).argmax(), diff.shape)

    # Directional derivative spot checks
    rng = np.random.default_rng(123)
    dir_stats = []
    for _ in range(n_dir_checks):
        v = rng.standard_normal(decision_state.shape[0])
        v /= np.linalg.norm(v) + 1e-16
        eps = rel_step
        f_plus  = f_only(decision_state + eps * v)
        f_minus = f_only(decision_state - eps * v)
        dd_fd   = (f_plus - f_minus) / (2 * eps)
        dd_an   = jac_analytic @ v
        dd_err  = np.linalg.norm(dd_an - dd_fd, ord=np.inf)
        dir_stats.append(dd_err)
    max_dir_err = float(np.max(dir_stats)) if dir_stats else 0.0

    print("Jacobian check:")
    print(f"  max abs error    = {max_abs_err:.3e} at ({i_max},{j_max})")
    print(f"  max rel error    = {max_rel_err:.3e}")
    print(f"  max dir-der error= {max_dir_err:.3e}")

    return {
        "err_analytic": err_analytic,
        "jac_analytic": jac_analytic,
        "jac_numeric": jac_numeric,
        "max_abs_error": max_abs_err,
        "max_rel_error": max_rel_err,
        "max_dir_der_error": max_dir_err,
        "worst_index": (int(i_max), int(j_max)),
    }


def solve_ivp_func(
        time_span                   ,
        state_costate_o             ,
        optimization_parameters     ,
        integration_state_parameters,
        inequality_parameters       ,
    ):
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
    mass_o                   = integration_state_parameters['mass_o'                  ]
    include_scstm            = integration_state_parameters['include_scstm'           ]
    post_process             = integration_state_parameters['post_process'            ]

    # Form integration state
    if include_jacobian:
        integration_state_parameters['include_scstm'] = True
        stm_oo                                        = np.identity(8).flatten()
        integration_state_o = np.hstack([state_costate_o, stm_oo])
    else:
        integration_state_parameters['include_scstm'] = False
        integration_state_o                           = state_costate_o
    if use_thrust_limits:
        integration_state_o = np.hstack([integration_state_o, mass_o])
    include_scstm = integration_state_parameters['include_scstm']
    if integration_state_parameters['post_process']:
        # Post-processing overrides the form of the state
        mass_o                      = integration_state_parameters['mass_o']
        optimal_control_objective_o = np.float64(0.0)
        integration_state_o         = np.hstack([state_costate_o, mass_o, optimal_control_objective_o])

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
            solve_ivp_func                        ,
            time_span                             ,
            integration_state_o                   ,
            t_eval              = time_eval_points,
            dense_output        = True            ,
            method              = 'RK45'          ,
            rtol                = 1.0e-12         ,
            atol                = 1.0e-12         ,
        )

    return soln_ivp


def _solve_for_root(
        decision_state_initguess    ,
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
        options_root                ,
    ):
    """
    Helper function to call the root solver.
    """
    root_func = lambda decision_state: tpbvp_objective_and_jacobian(
        decision_state              ,
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
    )
    return root(
        root_func,
        decision_state_initguess,
        method  = 'lm',
        tol     = 1.0e-11,
        jac     = optimization_parameters['include_jacobian'],
        options = options_root,
    )


def solve_for_root_and_compute_progress(
        decision_state_initguess    ,
        optimization_parameters     ,
        integration_state_parameters,
        equality_parameters         ,
        inequality_parameters       ,
        options_root                ,
    ):

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

    # Unpack decision-state initial guess and update the state-costate
    decision_state_initguess = soln_root.x
    time_span                = np.array([decision_state_initguess[0], decision_state_initguess[10]])
    pos_vec_o_pls            = decision_state_initguess[1:3]
    vel_vec_o_pls            = decision_state_initguess[3:5]
    copos_vec_o_pls          = decision_state_initguess[5:7]
    covel_vec_o_pls          = decision_state_initguess[7:9]
    state_costate_o          = np.hstack([pos_vec_o_pls, vel_vec_o_pls, copos_vec_o_pls, covel_vec_o_pls])

    # Solve initial value problem
    soln_ivp = \
        solve_ivp_func(
            time_span                   ,
            state_costate_o             ,
            optimization_parameters     ,
            integration_state_parameters,
            inequality_parameters       ,
        )

    return soln_root, soln_ivp


def compute_hamiltonian(
        min_type    ,
        vel_x       ,
        vel_y       ,
        copos_x     ,
        copos_y     ,
        covel_x     ,
        covel_y     ,
        thrust_acc_x,
        thrust_acc_y,
        acc_x       ,
        acc_y       ,
    ):
    if min_type == 'fuel':
        # ham = 1/2 (Gamma_vec^T Gamma_vec)^(1/2) + lambda_pos_vec^T vel_vec + lambda_vel_vec^T Gamma_vec
        return (
            (thrust_acc_x**2 + thrust_acc_y**2)**(1/2)
            + copos_x * vel_x + copos_y * vel_y
            + covel_x * acc_x + covel_y * acc_y
        )
    else: # assume min_type == 'energy'
        # ham = 1/2 (Gamma_vec^T Gamma_vec) + lambda_pos_vec^T vel_vec + lambda_vel_vec^T Gamma_vec
        return (
            1/2 * (thrust_acc_x**2 + thrust_acc_y**2)
            + copos_x * vel_x + copos_y * vel_y
            + covel_x * acc_x + covel_y * acc_y
        )


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
    
    # Define list for dictionary structure
    ordered_variables  = ['time', 'pos_vec', 'vel_vec', 'copos_vec', 'covel_vec', 'ham']
    ordered_boundaries = ['o', 'f']
    ordered_sides      = ['pls', 'mns']

    # Override the initial state with the relevant decision state variables
    idx = 0
    for bnd in ordered_boundaries:
        for var in ordered_variables:
            for side in ordered_sides:
                var_bnd = equality_parameters[var][bnd]
                if ( (bnd == 'o' and side == 'mns') or (bnd == 'f' and side == 'pls') ):
                    continue

                number_elements = np.size(var_bnd[side])
                value_slice     = decision_state[idx:idx+number_elements]

                if number_elements == 1:
                    value_slice = value_slice[0]
                
                equality_parameters[var][bnd][side] = value_slice
                idx += number_elements

    # Unpack
    include_jacobian = optimization_parameters['include_jacobian']

    time_o_mns, pos_vec_o_mns, vel_vec_o_mns, copos_vec_o_mns, covel_vec_o_mns, ham_o_mns = (
        equality_parameters[     'time']['o']['mns'],
        equality_parameters[  'pos_vec']['o']['mns'], equality_parameters[  'vel_vec']['o']['mns'],
        equality_parameters['copos_vec']['o']['mns'], equality_parameters['covel_vec']['o']['mns'],
        equality_parameters[      'ham']['o']['mns']
    )
    time_o_pls, pos_vec_o_pls, vel_vec_o_pls, copos_vec_o_pls, covel_vec_o_pls, ham_o_pls = (
        equality_parameters[     'time']['o']['pls'],
        equality_parameters[  'pos_vec']['o']['pls'], equality_parameters[  'vel_vec']['o']['pls'],
        equality_parameters['copos_vec']['o']['pls'], equality_parameters['covel_vec']['o']['pls'],
        equality_parameters[      'ham']['o']['pls']
    )
    time_f_mns, pos_vec_f_mns, vel_vec_f_mns, copos_vec_f_mns, covel_vec_f_mns, ham_f_mns = (
        equality_parameters[     'time']['f']['mns'],
        equality_parameters[  'pos_vec']['f']['mns'], equality_parameters[  'vel_vec']['f']['mns'],
        equality_parameters['copos_vec']['f']['mns'], equality_parameters['covel_vec']['f']['mns'],
        equality_parameters[      'ham']['f']['mns']
    )
    time_f_pls, pos_vec_f_pls, vel_vec_f_pls, copos_vec_f_pls, covel_vec_f_pls, ham_f_pls = (
        equality_parameters[     'time']['f']['pls'],
        equality_parameters[  'pos_vec']['f']['pls'], equality_parameters[  'vel_vec']['f']['pls'],
        equality_parameters['copos_vec']['f']['pls'], equality_parameters['covel_vec']['f']['pls'],
        equality_parameters[      'ham']['f']['pls']
    )

    # Time span
    time_span = np.array([time_o_pls, time_f_mns])

    # Initial state
    # state_costate_o = np.hstack([pos_vec_o_pls, vel_vec_o_pls, decision_state])
    state_o         = np.hstack([  pos_vec_o_pls,   vel_vec_o_pls])
    costate_o       = np.hstack([copos_vec_o_pls, covel_vec_o_pls])
    state_costate_o = np.hstack([state_o, costate_o])

    # Solve initial value problem
    soln_ivp = \
        solve_ivp_func(
            time_span                   ,
            state_costate_o             ,
            optimization_parameters     ,
            integration_state_parameters,
            inequality_parameters       ,
        )
    
    # Extract final state and final STM
    state_costate_scstm_f = soln_ivp.sol(time_span[1])
    state_costate_f       = state_costate_scstm_f[:8]
    pos_vec_f_mns         = state_costate_f[0:2]
    vel_vec_f_mns         = state_costate_f[2:4]
    copos_vec_f_mns       = state_costate_f[4:6]
    covel_vec_f_mns       = state_costate_f[6:8]
    if include_jacobian:
        scstm_of = state_costate_scstm_f[8:8+8**2].reshape((8,8))
    if inequality_parameters['use_thrust_limits']:
        mass_f_mns = state_costate_scstm_f[-1]
    else:
        mass_f_mns = 1.0 # dummy value
    
    # Compute the hamiltonian at the initial and final time
    if optimization_parameters['min_type'] == 'fuel':
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
                vel_x        = vel_vec_o_pls[0]   ,
                vel_y        = vel_vec_o_pls[1]   ,
                copos_x      = copos_vec_o_pls[0] ,
                copos_y      = copos_vec_o_pls[1] ,
                covel_x      = covel_vec_o_pls[0] ,
                covel_y      = covel_vec_o_pls[1] ,
                thrust_acc_x = thrust_acc_x_o_pls,
                thrust_acc_y = thrust_acc_y_o_pls,
                acc_x        = thrust_acc_x_o_pls,
                acc_y        = thrust_acc_y_o_pls,
            )
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
                vel_x        = vel_vec_f_mns[0]   ,
                vel_y        = vel_vec_f_mns[1]   ,
                copos_x      = copos_vec_f_mns[0] ,
                copos_y      = copos_vec_f_mns[1] ,
                covel_x      = covel_vec_f_mns[0] ,
                covel_y      = covel_vec_f_mns[1] ,
                thrust_acc_x = thrust_acc_x_f_mns,
                thrust_acc_y = thrust_acc_y_f_mns,
                acc_x        = thrust_acc_x_f_mns,
                acc_y        = thrust_acc_y_f_mns,
            )
    else: # optimization_parameters['min_type'] == 'energy':
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
                vel_x        = vel_vec_o_pls[0]   ,
                vel_y        = vel_vec_o_pls[1]   ,
                copos_x      = copos_vec_o_pls[0] ,
                copos_y      = copos_vec_o_pls[1] ,
                covel_x      = covel_vec_o_pls[0] ,
                covel_y      = covel_vec_o_pls[1] ,
                thrust_acc_x = thrust_acc_x_o_pls,
                thrust_acc_y = thrust_acc_y_o_pls,
                acc_x        = thrust_acc_x_o_pls,
                acc_y        = thrust_acc_y_o_pls,
            )
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
                vel_x        = vel_vec_f_mns[0]   ,
                vel_y        = vel_vec_f_mns[1]   ,
                copos_x      = copos_vec_f_mns[0] ,
                copos_y      = copos_vec_f_mns[1] ,
                covel_x      = covel_vec_f_mns[0] ,
                covel_y      = covel_vec_f_mns[1] ,
                thrust_acc_x = thrust_acc_x_f_mns,
                thrust_acc_y = thrust_acc_y_f_mns,
                acc_x        = thrust_acc_x_f_mns,
                acc_y        = thrust_acc_y_f_mns,
            )
    # if include_jacobian:
    #     # Partials of the Hamiltonian at the initial time
    #     # xxx

    #     # Partials of the Hamiltonian at the final time
    #     d_ham_f_mns__d_time_o_pls      = 0.0
    #     d_ham_f_mns__d_pos_vec_o_pls   = 0.0
    #     d_ham_f_mns__d_vel_vec_o_pls   = 0.0
    #     d_ham_f_mns__d_copos_vec_o_pls = 0.0
    #     d_ham_f_mns__d_covel_vec_o_pls = 0.0
    #     d_ham_f_mns__d_ham_o_pls       = 0.0

    # Calculate the error vector and error vector Jacobian
    #   error = [ delta [time_o, pos_vec_o, vel_vec_o, copos_vec_o, covel_vel_o, ham_o] ]
    #           [ delta [time_f, pos_vec_f, vel_vec_f, copos_vec_f, covel_vel_f, ham_f] ]
    #   jacobian = d(state_costate_f) / d(state_costate_o)

    # Enforce overrides for free variables
    if equality_parameters['time'     ]['o']['mode'] == 'free': time_o_mns      = time_o_pls
    if equality_parameters['pos_vec'  ]['o']['mode'] == 'free': pos_vec_o_mns   = pos_vec_o_pls
    if equality_parameters['vel_vec'  ]['o']['mode'] == 'free': vel_vec_o_mns   = vel_vec_o_pls
    if equality_parameters['copos_vec']['o']['mode'] == 'free': copos_vec_o_mns = copos_vec_o_pls
    if equality_parameters['covel_vec']['o']['mode'] == 'free': covel_vec_o_mns = covel_vec_o_pls
    if equality_parameters['ham'      ]['o']['mode'] == 'free': ham_o_mns       = ham_o_pls

    if equality_parameters['time'     ]['f']['mode'] == 'free': time_f_pls      = time_f_mns
    if equality_parameters['pos_vec'  ]['f']['mode'] == 'free': pos_vec_f_pls   = pos_vec_f_mns
    if equality_parameters['vel_vec'  ]['f']['mode'] == 'free': vel_vec_f_pls   = vel_vec_f_mns
    if equality_parameters['copos_vec']['f']['mode'] == 'free': copos_vec_f_pls = copos_vec_f_mns
    if equality_parameters['covel_vec']['f']['mode'] == 'free': covel_vec_f_pls = covel_vec_f_mns
    if equality_parameters['ham'      ]['f']['mode'] == 'free': ham_f_pls       = ham_f_mns

    # Time error
    error_time_o = time_o_pls - time_o_mns
    error_time_f = time_f_pls - time_f_mns

    # Position, velocity, co-position, co-velocity error
    error_pos_vec_o = pos_vec_o_pls - pos_vec_o_mns
    error_pos_vec_f = pos_vec_f_pls - pos_vec_f_mns 
    error_vel_vec_o = vel_vec_o_pls - vel_vec_o_mns
    error_vel_vec_f = vel_vec_f_pls - vel_vec_f_mns

    # Co-position and co-velocity error
    error_copos_vec_o = copos_vec_o_pls - copos_vec_o_mns
    error_copos_vec_f = copos_vec_f_pls - copos_vec_f_mns
    error_covel_vec_o = covel_vec_o_pls - covel_vec_o_mns
    error_covel_vec_f = covel_vec_f_pls - covel_vec_f_mns

    # Hamiltonian error
    error_ham_o = ham_o_pls - ham_o_mns
    error_ham_f = ham_f_pls - ham_f_mns

    # Full error vector
    error_full = {
        'time' : {
            'o': error_time_o,
            'f': error_time_f 
        },
        'pos_vec' : {
            'o': error_pos_vec_o,
            'f': error_pos_vec_f
        },
        'vel_vec' : {
            'o': error_vel_vec_o,
            'f': error_vel_vec_f
        },
        'copos_vec' : {
            'o': error_copos_vec_o,
            'f': error_copos_vec_f
        },
        'covel_vec' : {
            'o': error_covel_vec_o,
            'f': error_covel_vec_f
        },
        'ham' : {
            'o': error_ham_o,
            'f': error_ham_f
        }
    }

    # Error vector
    error_components = []
    for bnd in ordered_boundaries:
        for var in ordered_variables:
            error_components.append(error_full[var][bnd])
    error = np.hstack(error_components)

    # Error jacobian
    if include_jacobian:
        #   error = [ delta [time_o, pos_vec_o, vel_vec_o, copos_vec_o, covel_vel_o, ham_o] ]
        #           [ delta [time_f, pos_vec_f, vel_vec_f, copos_vec_f, covel_vel_f, ham_f] ]
        #   jacobian = d(state_costate_f) / d(state_costate_o)
        #   error_jacobian = d(error) / d(decision_state)
        #                  = [ d(delta_time_o)      / d(time_o_pls)    d(delta_time_o)      / d(pos_vec_o_pls)    ...
        #                      d(delta_pos_vec_o)   / d(time_o_pls)    d(delta_pos_vec_o)   / d(pos_vec_o_pls)    ...
        #                      d(delta_vel_vec_o)   / d(time_o_pls)    d(delta_vel_vec_o)   / d(pos_vec_o_pls)    ...
        #                      d(delta_copos_vec_o) / d(time_o_pls)    d(delta_copos_vec_o) / d(pos_vec_o_pls)    ...
        #                      d(delta_covel_vel_o) / d(time_o_pls)    d(delta_covel_vel_o) / d(pos_vec_o_pls)    ...
        #                      d(delta_ham_o)       / d(time_o_pls)    d(delta_ham_o)       / d(pos_vec_o_pls)    ...
        #                      d(delta_time_f)      / d(time_f_pls)    d(delta_time_f)      / d(pos_vec_f_pls)    ...
        #                      d(delta_pos_vec_f)   / d(time_f_pls)    d(delta_pos_vec_f)   / d(pos_vec_f_pls)    ...
        #                      d(delta_vel_vec_f)   / d(time_f_pls)    d(delta_vel_vec_f)   / d(pos_vec_f_pls)    ...
        #                      d(delta_copos_vec_f) / d(time_f_pls)    d(delta_copos_vec_f) / d(pos_vec_f_pls)    ...
        #                      d(delta_covel_vel_f) / d(time_f_pls)    d(delta_covel_vel_f) / d(pos_vec_f_pls)    ...
        #                      d(delta_ham_f)       / d(time_f_pls)    d(delta_ham_f)       / d(pos_vec_f_pls)    ...
        
        # stm_of
        error_jacobian = np.zeros((len(error), len(error)))

        # d(error_time_o)/d(time_o_pls)
        error_jacobian[0,0] = 1.0

        # d(error_ham_o)/d(time_o_pls) = - d(ham_f_mns)/d(state_costate_f) d(state_costate_f)/d(state_costate_o) d(state_costate_o)/d(time_o_pls)

        # Build RHS state with mass if thrust limits are active
        if inequality_parameters['use_thrust_limits']:
            mass_o = integration_state_parameters['mass_o']
            integration_state_o = np.hstack([state_costate_o, mass_o])
        else:
            integration_state_o = state_costate_o

        # d(error_state_costate_f)/d(time_o_pls) = -d(state_costate_f)/d(time_o_pls) = scstm_of * (d(state_o)/dt)
        d_state_costate_o__d_t = \
            one_body_dynamics__indirect(
                time_o_pls,
                integration_state_o,
                include_scstm=False,
                min_type=optimization_parameters['min_type'],
                use_thrust_acc_limits=inequality_parameters['use_thrust_acc_limits'],
                use_thrust_acc_smoothing=inequality_parameters['use_thrust_acc_smoothing'],
                thrust_acc_min=inequality_parameters['thrust_acc_min'],
                thrust_acc_max=inequality_parameters['thrust_acc_max'],
                use_thrust_limits=inequality_parameters['use_thrust_limits'],
                use_thrust_smoothing=inequality_parameters['use_thrust_smoothing'],
                thrust_min=inequality_parameters['thrust_min'],
                thrust_max=inequality_parameters['thrust_max'],
                post_process=False,
                k_steepness=inequality_parameters['k_steepness'],
            )

        d_state_costate_o__d_t = d_state_costate_o__d_t[0:8]

        d_state_costate_f_mns__d_time_o_pls = -1 * scstm_of @ d_state_costate_o__d_t
        # d(error_pos_vec_f)/d(time_o_pls)
        error_jacobian[11:13, 0] = -1 * d_state_costate_f_mns__d_time_o_pls[0:2]
        # d(error_vel_vec_f)/d(time_o_pls)
        error_jacobian[13:15, 0] = -1 * d_state_costate_f_mns__d_time_o_pls[2:4]
        # d(error_copos_vec_f)/d(time_o_pls)
        error_jacobian[15:17, 0] = -1 * d_state_costate_f_mns__d_time_o_pls[4:6]
        # d(error_covel_vec_f)/d(time_o_pls)
        error_jacobian[17:19, 0] = -1 * d_state_costate_f_mns__d_time_o_pls[6:8]

        # d(error_ham_f)/d(time_o_pls) = - d(ham_f_mns)/d(state_costate_f) d(state_costate_f)/d(state_costate_o) d(state_costate_o)/d(time_o_pls)
        if inequality_parameters['use_thrust_limits']:
            integration_state_f = np.hstack([state_costate_f, mass_f_mns])
        else:
            integration_state_f = state_costate_f
        d_state_costate_f_mns__d_t = \
            one_body_dynamics__indirect(
                time_f_mns,
                integration_state_f,
                include_scstm=False,
                min_type=optimization_parameters['min_type'],
                use_thrust_acc_limits=inequality_parameters['use_thrust_acc_limits'],
                use_thrust_acc_smoothing=inequality_parameters['use_thrust_acc_smoothing'],
                thrust_acc_min=inequality_parameters['thrust_acc_min'],
                thrust_acc_max=inequality_parameters['thrust_acc_max'],
                use_thrust_limits=inequality_parameters['use_thrust_limits'],
                use_thrust_smoothing=inequality_parameters['use_thrust_smoothing'],
                thrust_min=inequality_parameters['thrust_min'],
                thrust_max=inequality_parameters['thrust_max'],
                post_process=False,
                k_steepness=inequality_parameters['k_steepness'],
            )
        d_state_costate_f_mns__d_t = d_state_costate_f_mns__d_t[0:8]
        d_ham_f_mns__d_state_costate_f = \
            np.hstack([
                -d_state_costate_f_mns__d_t[4:6], # -d(copos)/dt
                -d_state_costate_f_mns__d_t[6:8], # -d(covel)/dt
                 d_state_costate_f_mns__d_t[0:2], #  d(  pos)/dt
                 d_state_costate_f_mns__d_t[2:4]  #  d(  vel)/dt
            ])
        d_ham_f_mns__d_time_o_pls = d_ham_f_mns__d_state_costate_f @ d_state_costate_f_mns__d_time_o_pls
        error_jacobian[19, 0] = -1.0 * d_ham_f_mns__d_time_o_pls

        # d(error_time_f)/d(time_f_mns) = d(time_f_pls - time_f_mns)/d(time_f_mns)
        error_jacobian[10, 10] = -1.0

        # d(error_f)/d(time_f_mns) = -d(state_f)/d(time_f_mns) = -d(state_f)/dt
        d_state_costate_f_mns__d_time_f_mns = d_state_costate_f_mns__d_t
        # d(error_pos_vec_f)/d(time_f_mns)
        error_jacobian[11:13, 10] = -1 * d_state_costate_f_mns__d_time_f_mns[0:2]
        # d(error_vel_vec_f)/d(time_f_mns)
        error_jacobian[13:15, 10] = -1 * d_state_costate_f_mns__d_time_f_mns[2:4]
        # d(error_copos_vec_f)/d(time_f_mns)
        error_jacobian[15:17, 10] = -1 * d_state_costate_f_mns__d_time_f_mns[4:6]
        # d(error_covel_vec_f)/d(time_f_mns)
        error_jacobian[17:19, 10] = -1 * d_state_costate_f_mns__d_time_f_mns[6:8]

        # d(error_f)/d(time_f_mns) = -f(x_f) already set for state/costate blocks
        d_state_costate_f_mns__d_time_f_mns = d_state_costate_f_mns__d_t
        # d(error_ham_f)/d(time_f_mns) = - dH_f/dx_f · f(x_f)
        d_ham_f_mns__d_time_f_mns = d_ham_f_mns__d_state_costate_f @ d_state_costate_f_mns__d_time_f_mns
        error_jacobian[19, 10] = -1.0 * d_ham_f_mns__d_time_f_mns

        # d(error_pos_vec_o)/d(pos_vec_o_pls) = d(pos_vec_o_pls - pos_vec_o_mns)/d(pos_vec_o_pls) = I
        error_jacobian[1:3, 1:3] = np.identity(2)
        # d(error_vel_vec_o)/d(vel_vec_o_pls) = d(vel_vec_o_pls - vel_vec_o_mns)/d(vel_vec_o_pls) = I
        error_jacobian[3:5, 3:5] = np.identity(2)
        # d(error_copos_vec_o)/d(copos_vec_o_pls) = d(copos_vec_o_pls - copos_vec_o_mns)/d(copos_vec_o_pls) = I
        error_jacobian[5:7, 5:7] = np.identity(2)
        # d(error_covel_vec_o)/d(covel_vec_o_pls) = d(covel_vec_o_pls - covel_vec_o_mns)/d(covel_vec_o_pls) = I
        error_jacobian[7:9, 7:9] = np.identity(2)

        # Add d(error_ham_o)/d(state_costate_o) and d(error_ham_o)/d(time_o_pls)
        d_ham_o_pls__d_state_costate_o = np.hstack([
            -d_state_costate_o__d_t[4:6], # -d(copos)/dt at t0
            -d_state_costate_o__d_t[6:8], # -d(covel)/dt at t0
             d_state_costate_o__d_t[0:2], #  d(pos)/dt at t0
             d_state_costate_o__d_t[2:4]  #  d(vel)/dt at t0
        ])
        error_jacobian[9, 1:9] = d_ham_o_pls__d_state_costate_o
        # d_ham_o_pls__d_time_o_pls = d_ham_o_pls__d_state_costate_o @ d_state_costate_o__d_t
        # error_jacobian[9, 0] = d_ham_o_pls__d_time_o_pls

        # d(error_pos_vec_f)/d(state_costate_o) = -d(pos_vec_f_mns)/d(state_costate_o) = -scstm_of[0:2, 0:8]
        error_jacobian[11:13, 1:9] = -scstm_of[0:2, 0:8] # should be negative
        # d(error_vel_vec_f)/d(state_costate_o) = -d(vel_vec_f_mns)/d(state_costate_o) = -scstm_of[2:4, 0:8]
        error_jacobian[13:15, 1:9] = -scstm_of[2:4, 0:8] # should be negative
        # d(error_copos_vec_f)/d(state_costate_o) = -d(copos_vec_f_mns)/d(state_costate_o) = -scstm_of[4:6, 0:8]
        error_jacobian[15:17, 1:9] = -scstm_of[4:6, 0:8] # should be negative
        # d(error_covel_vec_f)/d(state_costate_o) = -d(covel_vec_f_mns)/d(state_costate_o) = -scstm_of[6:8, 0:8]
        error_jacobian[17:19, 1:9] = -scstm_of[6:8, 0:8] # should be negative

        # d(error_ham_f)/d(state_costate_o) = -d(ham_f_mns)/d(state_costate_o) = -d(ham_f_mns)/d(state_costate_f) * d(state_costate_f)/d(state_costate_o)
        d_ham_f_mns__d_state_costate_o = d_ham_f_mns__d_state_costate_f @ scstm_of
        error_jacobian[19, 1:9]        = -1 * d_ham_f_mns__d_state_costate_o

        # Zero out jacobian rows for free variables, as their error is always zero
        idx = 0
        for bnd in ordered_boundaries:
            for var in ordered_variables:
                num_elements = np.size(error_full[var][bnd])
                if equality_parameters[var][bnd]['mode'] == 'free':
                    error_jacobian[idx:idx+num_elements, :] = 0.0
                idx += num_elements
    


    # Return
    if include_jacobian:
        return error, error_jacobian
    else:
        return error