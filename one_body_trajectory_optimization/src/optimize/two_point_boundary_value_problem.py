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
    include_jacobian  =      optimization_parameters['include_jacobian']

    # Segment variables
    time_o_pls        =          equality_parameters[     'time']['o']['pls']['value']
    pos_vec_o_pls     =          equality_parameters[  'pos_vec']['o']['pls']['value']
    vel_vec_o_pls     =          equality_parameters[  'vel_vec']['o']['pls']['value']
    copos_vec_o_pls   =          equality_parameters['copos_vec']['o']['pls']['value']
    covel_vec_o_pls   =          equality_parameters['covel_vec']['o']['pls']['value']
    ham_o_pls         =          equality_parameters[      'ham']['o']['pls']['value']
    
    time_f_mns       =          equality_parameters[     'time']['f']['mns']['value']
    pos_vec_f_mns    =          equality_parameters[  'pos_vec']['f']['mns']['value']
    vel_vec_f_mns    =          equality_parameters[  'vel_vec']['f']['mns']['value']
    copos_vec_f_mns  =          equality_parameters['copos_vec']['f']['mns']['value']
    covel_vec_f_mns  =          equality_parameters['covel_vec']['f']['mns']['value']
    ham_f_mns        =          equality_parameters[      'ham']['f']['mns']['value']

    # Boundary variables
    time_o_mns        =          equality_parameters[     'time']['o']['mns']['value']
    pos_vec_o_mns     =          equality_parameters[  'pos_vec']['o']['mns']['value']
    vel_vec_o_mns     =          equality_parameters[  'vel_vec']['o']['mns']['value']
    copos_vec_o_mns   =          equality_parameters['copos_vec']['o']['mns']['value']
    covel_vec_o_mns   =          equality_parameters['covel_vec']['o']['mns']['value']
    ham_o_mns         =          equality_parameters[      'ham']['o']['mns']['value']

    time_f_pls        =          equality_parameters[     'time']['f']['pls']['value']
    pos_vec_f_pls     =          equality_parameters[  'pos_vec']['f']['pls']['value']
    vel_vec_f_pls     =          equality_parameters[  'vel_vec']['f']['pls']['value']
    copos_vec_f_pls   =          equality_parameters['copos_vec']['f']['pls']['value']
    covel_vec_f_pls   =          equality_parameters['covel_vec']['f']['pls']['value']
    ham_f_pls         =          equality_parameters[      'ham']['f']['pls']['value']

    # Override the initial state with the decision state
    #   The decision state is a vector of the form:
    #   [time_o, pos_vec_o, vel_vec_o, copos_vec_o, covel_vec_o, ham_o]
    ordered_variables = ['time', 'pos_vec', 'vel_vec', 'copos_vec', 'covel_vec', 'ham']
    idx = 0
    for boundary in ['o', 'f']:
        for variable in ordered_variables:
            var_bnd = equality_parameters[variable][boundary]
            is_known = var_bnd['mode'] == 'fixed'
            if not is_known:
                if variable == 'time' and boundary == 'o':
                    time_o_pls = decision_state[idx]
                    idx += 1
                elif variable == 'pos_vec' and boundary == 'o':
                    pos_vec_o_pls = decision_state[idx:idx+2]
                    idx += 2
                elif variable == 'vel_vec' and boundary == 'o':
                    vel_vec_o_pls = decision_state[idx:idx+2]
                    idx += 2
                elif variable == 'copos_vec' and boundary == 'o':
                    copos_vec_o_pls = decision_state[idx:idx+2]
                    idx += 2
                elif variable == 'covel_vec' and boundary == 'o':
                    covel_vec_o_pls = decision_state[idx:idx+2]
                    idx += 2
                elif variable == 'ham' and boundary == 'o':
                    ham_o_pls = decision_state[idx]
                    idx += 1
                elif variable == 'time' and boundary == 'f':
                    time_f_mns = decision_state[idx]
                    idx += 1
                elif variable == 'pos_vec' and boundary == 'f':
                    pos_vec_f_mns = decision_state[idx:idx+2]
                    idx += 2
                elif variable == 'vel_vec' and boundary == 'f':
                    vel_vec_f_mns = decision_state[idx:idx+2]
                    idx += 2
                elif variable == 'copos_vec' and boundary == 'f':
                    copos_vec_f_mns = decision_state[idx:idx+2]
                    idx += 2
                elif variable == 'covel_vec' and boundary == 'f':
                    covel_vec_f_mns = decision_state[idx:idx+2]
                    idx += 2
                elif variable == 'ham' and boundary == 'f':
                    ham_f_mns = decision_state[idx]
                    idx += 1

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
        stm_of = state_costate_scstm_f[8:8+8**2].reshape((8,8))

    # Compute the hamiltonian at the initial and final time
    ham_o_pls = 0.0
    ham_f_mns = 0.0
    
    # Calculate the error vector and error vector Jacobian
    #   error = [ delta [time_o, pos_vec_o, vel_vec_o, copos_vec_o, covel_vel_o, ham_o] ]
    #           [ delta [time_f, pos_vec_f, vel_vec_f, copos_vec_f, covel_vel_f, ham_f] ]
    #   jacobian = d(state_final) / d(costate_initial)

    # Enforce trivial overrides
    time_o_pls      = time_o_mns # trivial
    time_f_mns      = time_f_pls # trivial
    copos_vec_f_pls = copos_vec_f_mns # trivial
    covel_vec_f_pls = covel_vec_f_mns # trivial
    ham_o_mns       = ham_o_pls # trivial, might not be correct

    # Time error
    error_time_o = time_o_pls - time_o_mns # trivial
    error_time_f = time_f_pls - time_f_mns # trivial

    # Position, velocity, co-position, co-velocity error
    error_pos_vec_o = pos_vec_o_pls - pos_vec_o_mns # trivial
    error_pos_vec_f = pos_vec_f_pls - pos_vec_f_mns 
    error_vel_vec_o = vel_vec_o_pls - vel_vec_o_mns # trivial
    error_vel_vec_f = vel_vec_f_pls - vel_vec_f_mns

    # Co-position and co-velocity error
    error_copos_vec_o = copos_vec_o_pls - copos_vec_o_mns # trivial
    error_copos_vec_f = copos_vec_f_pls - copos_vec_f_mns # trivial
    error_covel_vec_o = covel_vec_o_pls - covel_vec_o_mns # trivial
    error_covel_vec_f = covel_vec_f_pls - covel_vec_f_mns # trivial

    # Hamiltonian error
    error_ham_o = ham_o_pls - ham_o_mns # trivial
    error_ham_f = ham_f_pls - ham_f_mns

    # Pack up the error vector
    error_full = np.hstack([
        error_time_o     , # trivial
        error_pos_vec_o  , # trivial
        error_vel_vec_o  , # trivial
        error_copos_vec_o, # trivial
        error_covel_vec_o, # trivial
        error_ham_o      , # trivial
        error_time_f     , # trivial
        error_pos_vec_f  ,
        error_vel_vec_f  ,
        error_copos_vec_f,
        error_covel_vec_f,
        error_ham_f      ,
    ])

    # Consolidate errors
    case_choice = 1 # 1 : fixed fin-time; fixed init-pos; fixed init-vel; fixed fin-pos; fixed fin-vel
                    # 2 :  free fin-time; fixed init-pos; fixed init-vel; fixed fin-pos; fixed fin-vel
    if case_choice == 1:
        error = error_full[11:15] # 4 constraints: error_pos_vec_f, error_vel_vec_f
    elif case_choice == 2:
        error = error_full[11:15 and 19] # 5 constraints: error_pos_vec_f, error_vel_vec_f, error_ham_f

    # error = state_costate_f[:4] - np.hstack([pos_vec_f_pls, vel_vec_f_pls])
    # if include_jacobian:
    #     error_jacobian = stm_of[0:4, 4:8]

    # Pack up: error and error-jacobian
    if include_jacobian:
        return error, error_jacobian
    else:
        return error