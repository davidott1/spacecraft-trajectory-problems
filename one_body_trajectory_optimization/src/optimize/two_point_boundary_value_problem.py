import numpy as np
from scipy.integrate    import solve_ivp
from scipy.optimize     import root
from src.model.dynamics import one_body_dynamics__indirect


def solve_ivp_func(
        time_span                   ,
        state_costate_o             ,
        optimization_parameters     ,
        integration_state_parameters,
        inequality_parameters       ,
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


def solve_for_root_and_compute_progress(
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
        solve_ivp_func(
            time_span                   ,
            state_costate_o             ,
            optimization_parameters     ,
            integration_state_parameters,
            inequality_parameters       ,
        )

    return soln_root, soln_ivp


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