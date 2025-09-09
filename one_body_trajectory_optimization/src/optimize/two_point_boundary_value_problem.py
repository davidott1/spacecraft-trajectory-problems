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
    alpha                    =        inequality_parameters['alpha'                   ]
    mass_o                   = integration_state_parameters['mass_o'                  ]
    exhaust_velocity         = integration_state_parameters['exhaust_velocity'        ]
    constant_gravity         = integration_state_parameters['constant_gravity'        ]
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
                exhaust_velocity             = exhaust_velocity        ,
                post_process                 = post_process            ,
                k_steepness                  = k_steepness             ,
                alpha                        = alpha                   ,
                constant_gravity             = constant_gravity        ,
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
    #   decision_state = [  pos_vec_o  vel_vec_o  copos_vec_o  covel_vec_o  time_f  ]
    decision_state_initguess = soln_root.x
    time_span                = np.array([0.0, decision_state_initguess[8]])
    pos_vec_o_pls            = decision_state_initguess[1:3]
    vel_vec_o_pls            = decision_state_initguess[3:5]
    copos_vec_o_pls          = decision_state_initguess[5:7]
    covel_vec_o_pls          = decision_state_initguess[7:9]
    state_costate_o          = np.hstack([ pos_vec_o_pls, vel_vec_o_pls, copos_vec_o_pls, covel_vec_o_pls ])

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
        alpha = 0   ,
    ):
    if min_type == 'fuel':
        # ham = (Gamma_vec^T Gamma_vec)^(1/2) + lambda_pos_vec^T vel_vec + lambda_vel_vec^T Gamma_vec
        return (
            (thrust_acc_x**2 + thrust_acc_y**2)**(1/2)
            + copos_x * vel_x + copos_y * vel_y
            + covel_x * acc_x + covel_y * acc_y
        )
    elif min_type == 'energyfuel':
        # ham = (1 - alpha) (Gamma_vec^T Gamma_vec)^(1/2) + alpha 1/2 (Gamma_vec^T Gamma_vec)^(1/2) + lambda_pos_vec^T vel_vec + lambda_vel_vec^T Gamma_vec
        return (
            (1 - alpha) * (thrust_acc_x**2 + thrust_acc_y**2)**(1/2)
            + alpha * (thrust_acc_x**2 + thrust_acc_y**2)
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
    
    # Override the initial state with the relevant decision state variables
    #   decision_state = [  pos_vec_o  vel_vec_o  copos_vec_o  covel_vec_o  time_f  ]
    equality_parameters[  'pos_vec']['o']['pls'] = decision_state[0:2]
    equality_parameters[  'vel_vec']['o']['pls'] = decision_state[2:4]
    equality_parameters['copos_vec']['o']['pls'] = decision_state[4:6]
    equality_parameters['covel_vec']['o']['pls'] = decision_state[6:8]
    equality_parameters[     'time']['f']['mns'] = decision_state[8  ]

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

    constant_gravity = integration_state_parameters['constant_gravity']

    # Time span
    time_span = np.array([time_o_pls, time_f_mns])

    # Initial state
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
            min_type     = optimization_parameters['min_type']  ,
            vel_x        = vel_vec_o_pls[0]                     ,
            vel_y        = vel_vec_o_pls[1]                     ,
            copos_x      = copos_vec_o_pls[0]                   ,
            copos_y      = copos_vec_o_pls[1]                   ,
            covel_x      = covel_vec_o_pls[0]                   ,
            covel_y      = covel_vec_o_pls[1]                   ,
            thrust_acc_x = thrust_acc_x_o_pls                   ,
            thrust_acc_y = thrust_acc_y_o_pls                   ,
            acc_x        = thrust_acc_x_o_pls                   ,
            acc_y        = constant_gravity + thrust_acc_y_o_pls,
            alpha        = inequality_parameters['alpha']       ,
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
            min_type     = optimization_parameters['min_type']  ,
            vel_x        = vel_vec_f_mns[0]                     ,
            vel_y        = vel_vec_f_mns[1]                     ,
            copos_x      = copos_vec_f_mns[0]                   ,
            copos_y      = copos_vec_f_mns[1]                   ,
            covel_x      = covel_vec_f_mns[0]                   ,
            covel_y      = covel_vec_f_mns[1]                   ,
            thrust_acc_x = thrust_acc_x_f_mns                   ,
            thrust_acc_y = thrust_acc_y_f_mns                   ,
            acc_x        = thrust_acc_x_f_mns                   ,
            acc_y        = constant_gravity + thrust_acc_y_f_mns,
            alpha        = inequality_parameters['alpha']       ,
        )
    
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
    error_time_o = time_o_pls - time_o_mns # not needed
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
    error_ham_o = ham_o_pls - ham_o_mns # not needed
    error_ham_f = ham_f_pls - ham_f_mns

    # Error vector
    #   error = [  error_pos_vec_o  error_vel_vec_o  error_copos_vec_o  error_covel_vel_o
    #              error_pos_vec_f  error_vel_vec_f  error_copos_vec_f  error_covel_vel_f
    #              error_time_f  error_ham_f  ]
    error_components = []
    if equality_parameters['pos_vec']['o']['mode'] == 'fixed':
        error_components.append(error_pos_vec_o)
    if equality_parameters['vel_vec']['o']['mode'] == 'fixed':
        error_components.append(error_vel_vec_o)
    if equality_parameters['copos_vec']['o']['mode'] == 'fixed':
        error_components.append(error_copos_vec_o)
    if equality_parameters['covel_vec']['o']['mode'] == 'fixed':
        error_components.append(error_covel_vec_o)
    if equality_parameters['pos_vec']['f']['mode'] == 'fixed':
        error_components.append(error_pos_vec_f)
    if equality_parameters['vel_vec']['f']['mode'] == 'fixed':
        error_components.append(error_vel_vec_f)
    if equality_parameters['copos_vec']['f']['mode'] == 'fixed':
        error_components.append(error_copos_vec_f)
    if equality_parameters['covel_vec']['f']['mode'] == 'fixed':
        error_components.append(error_covel_vec_f)
    if equality_parameters['time']['f']['mode'] == 'fixed':
        error_components.append(error_time_f)
    if equality_parameters['ham']['f']['mode'] == 'fixed':
        error_components.append(error_ham_f)
    error = np.hstack(error_components)

    # Error jacobian
    if include_jacobian:
        #   decision_state = [  pos_vec_o_pls  vel_vec_o_pls  copos_vec_o_pls  covel_vec_o_pls  time_f_mns  ]
        #   error          = [  error_pos_vec_o  error_vel_vec_o  error_copos_vec_o  error_covel_vel_o
        #                       error_pos_vec_f  error_vel_vec_f  error_copos_vec_f  error_covel_vel_f
        #                       error_time_f     error_ham_f  ]
        #   error_jacobian = d(error)/d(decision_state)
        
        # Initialize error jacobian
        error_jacobian_components = []

        # Build error jacobian
        #   d(error_pos_vec_o)/d(decision_state)
        if equality_parameters['pos_vec']['o']['mode'] == 'fixed':
            d_error_pos_vec_o__d_pos_vec_o_pls   = np.eye(2)
            d_error_pos_vec_o__d_vel_vec_o_pls   = np.zeros((2,2))
            d_error_pos_vec_o__d_copos_vec_o_pls = np.zeros((2,2))
            d_error_pos_vec_o__d_covel_vec_o_pls = np.zeros((2,2))
            d_error_pos_vec_o__d_time_f_mns      = np.zeros((2,1))
            d_error_pos_vec_o__d_decision_state = \
                np.hstack([
                    d_error_pos_vec_o__d_pos_vec_o_pls,
                    d_error_pos_vec_o__d_vel_vec_o_pls,
                    d_error_pos_vec_o__d_copos_vec_o_pls,
                    d_error_pos_vec_o__d_covel_vec_o_pls,
                    d_error_pos_vec_o__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_pos_vec_o__d_decision_state)

        # Build error jacobian
        #   d(error_vel_vec_o)/d(decision_state)
        if equality_parameters['vel_vec']['o']['mode'] == 'fixed':
            d_error_vel_vec_o__d_pos_vec_o_pls   = np.zeros((2,2))
            d_error_vel_vec_o__d_vel_vec_o_pls   = np.eye(2)
            d_error_vel_vec_o__d_copos_vec_o_pls = np.zeros((2,2))
            d_error_vel_vec_o__d_covel_vec_o_pls = np.zeros((2,2))
            d_error_vel_vec_o__d_time_f_mns      = np.zeros((2,1))
            d_error_vel_vec_o__d_decision_state = \
                np.hstack([
                    d_error_vel_vec_o__d_pos_vec_o_pls,
                    d_error_vel_vec_o__d_vel_vec_o_pls,
                    d_error_vel_vec_o__d_copos_vec_o_pls,
                    d_error_vel_vec_o__d_covel_vec_o_pls,
                    d_error_vel_vec_o__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_vel_vec_o__d_decision_state)

        # Build error jacobian
        #   d(error_copos_vec_o)/d(decision_state)
        if equality_parameters['copos_vec']['o']['mode'] == 'fixed':
            d_error_copos_vec_o__d_pos_vec_o_pls   = np.zeros((2,2))
            d_error_copos_vec_o__d_vel_vec_o_pls   = np.zeros((2,2))
            d_error_copos_vec_o__d_copos_vec_o_pls = np.eye(2)
            d_error_copos_vec_o__d_covel_vec_o_pls = np.zeros((2,2))
            d_error_copos_vec_o__d_time_f_mns      = np.zeros((2,1))
            d_error_copos_vec_o__d_decision_state = \
                np.hstack([
                    d_error_copos_vec_o__d_pos_vec_o_pls,
                    d_error_copos_vec_o__d_vel_vec_o_pls,
                    d_error_copos_vec_o__d_copos_vec_o_pls,
                    d_error_copos_vec_o__d_covel_vec_o_pls,
                    d_error_copos_vec_o__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_copos_vec_o__d_decision_state)

        # Build error jacobian
        #   d(error_covel_vec_o)/d(decision_state)
        if equality_parameters['covel_vec']['o']['mode'] == 'fixed':
            d_error_covel_vec_o__d_pos_vec_o_pls   = np.zeros((2,2))
            d_error_covel_vec_o__d_vel_vec_o_pls   = np.zeros((2,2))
            d_error_covel_vec_o__d_copos_vec_o_pls = np.zeros((2,2))
            d_error_covel_vec_o__d_covel_vec_o_pls = np.eye(2)
            d_error_covel_vec_o__d_time_f_mns      = np.zeros((2,1))
            d_error_covel_vec_o__d_decision_state = \
                np.hstack([
                    d_error_covel_vec_o__d_pos_vec_o_pls,
                    d_error_covel_vec_o__d_vel_vec_o_pls,
                    d_error_covel_vec_o__d_copos_vec_o_pls,
                    d_error_covel_vec_o__d_covel_vec_o_pls,
                    d_error_covel_vec_o__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_covel_vec_o__d_decision_state)

        # Build error jacobian
        #   d(error_pos_vec_o)/d(decision_state)
        if equality_parameters['pos_vec']['f']['mode'] == 'fixed':
            if inequality_parameters['use_thrust_limits']:
                integration_state_f = np.hstack([state_costate_f, mass_f_mns])
            else:
                integration_state_f = state_costate_f
            d_state_costate_f_mns__d_time_f_mns = \
                one_body_dynamics__indirect(
                    time_f_mns,
                    integration_state_f,
                    include_scstm            = False,
                    min_type                 = optimization_parameters['min_type'],
                    use_thrust_acc_limits    = inequality_parameters['use_thrust_acc_limits'],
                    use_thrust_acc_smoothing = inequality_parameters['use_thrust_acc_smoothing'],
                    thrust_acc_min           = inequality_parameters['thrust_acc_min'],
                    thrust_acc_max           = inequality_parameters['thrust_acc_max'],
                    use_thrust_limits        = inequality_parameters['use_thrust_limits'],
                    use_thrust_smoothing     = inequality_parameters['use_thrust_smoothing'],
                    thrust_min               = inequality_parameters['thrust_min'],
                    thrust_max               = inequality_parameters['thrust_max'],
                    exhaust_velocity         = integration_state_parameters['exhaust_velocity'],
                    post_process             = False,
                    k_steepness              = inequality_parameters['k_steepness'],
                    alpha                    = inequality_parameters['alpha'],
                    constant_gravity         = integration_state_parameters['constant_gravity'],
                )
            d_state_costate_f_mns__d_time_f_mns = d_state_costate_f_mns__d_time_f_mns[0:8]

            d_error_pos_vec_f__d_pos_vec_o_pls   = -scstm_of[0:2, 0:2]
            d_error_pos_vec_f__d_vel_vec_o_pls   = -scstm_of[0:2, 2:4]
            d_error_pos_vec_f__d_copos_vec_o_pls = -scstm_of[0:2, 4:6]
            d_error_pos_vec_f__d_covel_vec_o_pls = -scstm_of[0:2, 6:8]
            d_error_pos_vec_f__d_time_f_mns      = -d_state_costate_f_mns__d_time_f_mns[0:2].reshape((2,1))
            d_error_pos_vec_f__d_decision_state = \
                np.hstack([
                    d_error_pos_vec_f__d_pos_vec_o_pls,
                    d_error_pos_vec_f__d_vel_vec_o_pls,
                    d_error_pos_vec_f__d_copos_vec_o_pls,
                    d_error_pos_vec_f__d_covel_vec_o_pls,
                    d_error_pos_vec_f__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_pos_vec_f__d_decision_state)

        # Build error jacobian
        #   d(error_vel_vec_f)/d(decision_state)
        if equality_parameters['vel_vec']['f']['mode'] == 'fixed':
            d_error_vel_vec_f__d_pos_vec_o_pls   = -scstm_of[2:4, 0:2]
            d_error_vel_vec_f__d_vel_vec_o_pls   = -scstm_of[2:4, 2:4]
            d_error_vel_vec_f__d_copos_vec_o_pls = -scstm_of[2:4, 4:6]
            d_error_vel_vec_f__d_covel_vec_o_pls = -scstm_of[2:4, 6:8]
            d_error_vel_vec_f__d_time_f_mns      = -d_state_costate_f_mns__d_time_f_mns[2:4].reshape((2,1))
            d_error_vel_vec_f__d_decision_state = \
                np.hstack([
                    d_error_vel_vec_f__d_pos_vec_o_pls,
                    d_error_vel_vec_f__d_vel_vec_o_pls,
                    d_error_vel_vec_f__d_copos_vec_o_pls,
                    d_error_vel_vec_f__d_covel_vec_o_pls,
                    d_error_vel_vec_f__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_vel_vec_f__d_decision_state)
        
        # Build error jacobian
        #   d(error_copos_vec_f)/d(decision_state)
        if equality_parameters['copos_vec']['f']['mode'] == 'fixed':
            d_error_copos_vec_f__d_pos_vec_o_pls   = -scstm_of[4:6, 0:2]
            d_error_copos_vec_f__d_vel_vec_o_pls   = -scstm_of[4:6, 2:4]
            d_error_copos_vec_f__d_copos_vec_o_pls = -scstm_of[4:6, 4:6]
            d_error_copos_vec_f__d_covel_vec_o_pls = -scstm_of[4:6, 6:8]
            d_error_copos_vec_f__d_time_f_mns      = -d_state_costate_f_mns__d_time_f_mns[4:6].reshape((2,1))
            d_error_copos_vec_f__d_decision_state = \
                np.hstack([
                    d_error_copos_vec_f__d_pos_vec_o_pls,
                    d_error_copos_vec_f__d_vel_vec_o_pls,
                    d_error_copos_vec_f__d_copos_vec_o_pls,
                    d_error_copos_vec_f__d_covel_vec_o_pls,
                    d_error_copos_vec_f__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_copos_vec_f__d_decision_state)

        # Build error jacobian
        #   d(error_covel_vec_f)/d(decision_state)
        if equality_parameters['covel_vec']['f']['mode'] == 'fixed':
            d_error_covel_vec_f__d_pos_vec_o_pls   = -scstm_of[6:8, 0:2]
            d_error_covel_vec_f__d_vel_vec_o_pls   = -scstm_of[6:8, 2:4]
            d_error_covel_vec_f__d_copos_vec_o_pls = -scstm_of[6:8, 4:6]
            d_error_covel_vec_f__d_covel_vec_o_pls = -scstm_of[6:8, 6:8]
            d_error_covel_vec_f__d_time_f_mns      = -d_state_costate_f_mns__d_time_f_mns[6:8].reshape((2,1))
            d_error_covel_vec_f__d_decision_state = \
                np.hstack([
                    d_error_covel_vec_f__d_pos_vec_o_pls,
                    d_error_covel_vec_f__d_vel_vec_o_pls,
                    d_error_covel_vec_f__d_copos_vec_o_pls,
                    d_error_covel_vec_f__d_covel_vec_o_pls,
                    d_error_covel_vec_f__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_covel_vec_f__d_decision_state)

        # Build error jacobian
        #   d(error_time_f)/d(decision_state)
        if equality_parameters['time']['f']['mode'] == 'fixed':
            d_error_time_f__d_pos_vec_o_pls   = np.zeros((1,2))
            d_error_time_f__d_vel_vec_o_pls   = np.zeros((1,2))
            d_error_time_f__d_copos_vec_o_pls = np.zeros((1,2))
            d_error_time_f__d_covel_vec_o_pls = np.zeros((1,2))
            d_error_time_f__d_time_f_mns      = np.array([[-1.0]])
            d_error_time_f__d_decision_state  = np.hstack([d_error_time_f__d_pos_vec_o_pls,d_error_time_f__d_vel_vec_o_pls,d_error_time_f__d_copos_vec_o_pls,d_error_time_f__d_covel_vec_o_pls,d_error_time_f__d_time_f_mns])
            error_jacobian_components.append(d_error_time_f__d_decision_state)

        # Build error jacobian
        #   d(error_ham_f)/d(decision_state)
        if equality_parameters['ham']['f']['mode'] == 'fixed':
            d_ham_f_mns__d_state_costate_f_mns = \
                np.hstack([
                    -d_state_costate_f_mns__d_time_f_mns[4:6], # -d(copos)/dt
                    -d_state_costate_f_mns__d_time_f_mns[6:8], # -d(covel)/dt
                     d_state_costate_f_mns__d_time_f_mns[0:2], #  d(  pos)/dt
                     d_state_costate_f_mns__d_time_f_mns[2:4]  #  d(  vel)/dt
                ])
            
            d_ham_f_mns__d_state_costate_o   = d_ham_f_mns__d_state_costate_f_mns @ scstm_of
            d_error_ham_f__d_pos_vec_o_pls   = -1.0 * d_ham_f_mns__d_state_costate_o[0:2].reshape((1,2))
            d_error_ham_f__d_vel_vec_o_pls   = -1.0 * d_ham_f_mns__d_state_costate_o[2:4].reshape((1,2))
            d_error_ham_f__d_copos_vec_o_pls = -1.0 * d_ham_f_mns__d_state_costate_o[4:6].reshape((1,2))
            d_error_ham_f__d_covel_vec_o_pls = -1.0 * d_ham_f_mns__d_state_costate_o[6:8].reshape((1,2))

            d_ham_f_mns__d_time_f_mns   = d_ham_f_mns__d_state_costate_f_mns @ d_state_costate_f_mns__d_time_f_mns
            d_error_ham_f__d_time_f_mns = -1.0 * d_ham_f_mns__d_time_f_mns.reshape((1,1))

            d_error_ham_f__d_decision_state = \
                np.hstack([
                    d_error_ham_f__d_pos_vec_o_pls,
                    d_error_ham_f__d_vel_vec_o_pls,
                    d_error_ham_f__d_copos_vec_o_pls,
                    d_error_ham_f__d_covel_vec_o_pls,
                    d_error_ham_f__d_time_f_mns
                ])
            error_jacobian_components.append(d_error_ham_f__d_decision_state)

        # Combine error jacobian components
        error_jacobian = np.vstack(error_jacobian_components)
    
    # Return
    if include_jacobian:
        return error, error_jacobian
    else:
        return error