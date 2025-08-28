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
    mass_o                   = integration_state_parameters['mass_o'                  ]
    include_scstm            = integration_state_parameters['include_scstm'           ]
    post_process             = integration_state_parameters['post_process'            ]

    # Form integration state
    if include_jacobian: breakpoint()
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
    time_span       = np.array([decision_state_initguess[0], decision_state_initguess[10]])
    pos_vec_o_pls   = decision_state_initguess[1:3]
    vel_vec_o_pls   = decision_state_initguess[3:5]
    copos_vec_o_pls = decision_state_initguess[5:7]
    covel_vec_o_pls = decision_state_initguess[7:9]
    state_costate_o = np.hstack([pos_vec_o_pls, vel_vec_o_pls, copos_vec_o_pls, covel_vec_o_pls])

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
                if (
                    (bnd == 'o' and side == 'mns')
                    or (bnd == 'f' and side == 'pls')
                    # or var_bnd['mode'] == 'fixed' # include later, doing len_dec_state = 20 first
                ):
                    continue
                # elif var_bnd['mode'] == 'free':
                # print(f"(mode, var, bnd, side): {var_bnd['mode']}, {var}, {bnd}, {side}")

                number_elements = np.size(var_bnd[side]['value'])
                value_slice     = decision_state[idx:idx+number_elements]

                if number_elements == 1:
                    value_slice = value_slice[0]
                
                equality_parameters[var][bnd][side]['value'] = value_slice
                idx += number_elements

    # Unpack
    include_jacobian = optimization_parameters['include_jacobian']

    time_o_mns, pos_vec_o_mns, vel_vec_o_mns, copos_vec_o_mns, covel_vec_o_mns, ham_o_mns = (
        equality_parameters[     'time']['o']['mns']['value'],
        equality_parameters[  'pos_vec']['o']['mns']['value'], equality_parameters[  'vel_vec']['o']['mns']['value'],
        equality_parameters['copos_vec']['o']['mns']['value'], equality_parameters['covel_vec']['o']['mns']['value'],
        equality_parameters[      'ham']['o']['mns']['value']
    )
    time_o_pls, pos_vec_o_pls, vel_vec_o_pls, copos_vec_o_pls, covel_vec_o_pls, ham_o_pls = (
        equality_parameters[     'time']['o']['pls']['value'],
        equality_parameters[  'pos_vec']['o']['pls']['value'], equality_parameters[  'vel_vec']['o']['pls']['value'],
        equality_parameters['copos_vec']['o']['pls']['value'], equality_parameters['covel_vec']['o']['pls']['value'],
        equality_parameters[      'ham']['o']['pls']['value']
    )
    time_f_mns, pos_vec_f_mns, vel_vec_f_mns, copos_vec_f_mns, covel_vec_f_mns, ham_f_mns = (
        equality_parameters[     'time']['f']['mns']['value'],
        equality_parameters[  'pos_vec']['f']['mns']['value'], equality_parameters[  'vel_vec']['f']['mns']['value'],
        equality_parameters['copos_vec']['f']['mns']['value'], equality_parameters['covel_vec']['f']['mns']['value'],
        equality_parameters[      'ham']['f']['mns']['value']
    )
    time_f_pls, pos_vec_f_pls, vel_vec_f_pls, copos_vec_f_pls, covel_vec_f_pls, ham_f_pls = (
        equality_parameters[     'time']['f']['pls']['value'],
        equality_parameters[  'pos_vec']['f']['pls']['value'], equality_parameters[  'vel_vec']['f']['pls']['value'],
        equality_parameters['copos_vec']['f']['pls']['value'], equality_parameters['covel_vec']['f']['pls']['value'],
        equality_parameters[      'ham']['f']['pls']['value']
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
        stm_of = state_costate_scstm_f[8:8+8**2].reshape((8,8))
    
    # Compute the hamiltonian at the initial and final time
    if optimization_parameters['min_type'] == 'fuel':

        use_thrust_acc_limits    =        inequality_parameters['use_thrust_acc_limits'   ]
        use_thrust_acc_smoothing =        inequality_parameters['use_thrust_acc_smoothing']
        thrust_acc_min           =        inequality_parameters['thrust_acc_min'          ]
        thrust_acc_max           =        inequality_parameters['thrust_acc_max'          ]
        use_thrust_limits        =        inequality_parameters['use_thrust_limits'       ]
        use_thrust_smoothing     =        inequality_parameters['use_thrust_smoothing'    ]
        thrust_min               =        inequality_parameters['thrust_min'              ]
        thrust_max               =        inequality_parameters['thrust_max'              ]
        k_steepness              =        inequality_parameters['k_steepness'             ]

        # H = (Gamma_x^2 + Gamma_y^2)^(1/2) + lambda_r_x v_x + lambda_r_y v_y + lambda_v_x Gamma_x + lambda_v_y Gamma_y
        vel_x_o_pls   =   vel_vec_o_pls[0]
        vel_y_o_pls   =   vel_vec_o_pls[1]
        copos_x_o_pls = copos_vec_o_pls[0]
        copos_y_o_pls = copos_vec_o_pls[1]
        covel_x_o_pls = covel_vec_o_pls[0]
        covel_y_o_pls = covel_vec_o_pls[1]
        epsilon   = 1.0e-6
        covel_mag_o_pls = np.sqrt(covel_x_o_pls**2 + covel_y_o_pls**2 + epsilon**2)
        covel_mag_o_pls_inv = 1.0 / covel_mag_o_pls
        switching_func_o_pls = covel_mag_o_pls - 1.0
        if use_thrust_smoothing or use_thrust_acc_smoothing:
            heaviside_approx = 0.5 + 0.5 * np.tanh(k_steepness * switching_func_o_pls)
            thrust_acc_mag_o_pls   = thrust_acc_min + (thrust_acc_max - thrust_acc_min) * heaviside_approx
        else: # no use_thrust_smoothing and no use_thrust_acc_smoothing
            thrust_acc_mag_o_pls = np.where(switching_func_o_pls > 0.0, thrust_acc_max, thrust_acc_min)
        thrust_acc_x_dir_o_pls = -covel_x_o_pls * covel_mag_o_pls_inv
        thrust_acc_y_dir_o_pls = -covel_y_o_pls * covel_mag_o_pls_inv
        thrust_acc_x_o_pls = thrust_acc_mag_o_pls * thrust_acc_x_dir_o_pls
        thrust_acc_y_o_pls = thrust_acc_mag_o_pls * thrust_acc_y_dir_o_pls

        vel_x_f_mns   =   vel_vec_f_mns[0]
        vel_y_f_mns   =   vel_vec_f_mns[1]
        copos_x_f_mns = copos_vec_f_mns[0]
        copos_y_f_mns = copos_vec_f_mns[1]
        covel_x_f_mns = covel_vec_f_mns[0]
        covel_y_f_mns = covel_vec_f_mns[1]
        epsilon   = 1.0e-6
        covel_mag_f_mns = np.sqrt(covel_x_f_mns**2 + covel_y_f_mns**2 + epsilon**2)
        covel_mag_f_mns_inv = 1.0 / covel_mag_f_mns
        switching_func_f_mns = covel_mag_f_mns - 1.0
        if use_thrust_smoothing or use_thrust_acc_smoothing:
            heaviside_approx = 0.5 + 0.5 * np.tanh(k_steepness * switching_func_f_mns)
            thrust_acc_mag_f_mns   = thrust_acc_min + (thrust_acc_max - thrust_acc_min) * heaviside_approx
        else: # no use_thrust_smoothing and no use_thrust_acc_smoothing
            thrust_acc_mag_f_mns = np.where(switching_func_f_mns > 0.0, thrust_acc_max, thrust_acc_min)
        thrust_acc_x_dir_f_mns = -covel_x_f_mns * covel_mag_f_mns_inv
        thrust_acc_y_dir_f_mns = -covel_y_f_mns * covel_mag_f_mns_inv
        thrust_acc_x_f_mns = thrust_acc_mag_f_mns * thrust_acc_x_dir_f_mns
        thrust_acc_y_f_mns = thrust_acc_mag_f_mns * thrust_acc_y_dir_f_mns
    else: # optimization_parameters['min_type'] == 'energy':
        # H = 1/2 (Gamma_x^2 + Gamma_y^2) + lambda_r_x v_x + lambda_r_y v_y + lambda_v_x Gamma_x + lambda_v_y Gamma_y
        vel_x_o_pls        =      vel_vec_o_pls[0]
        vel_y_o_pls        =      vel_vec_o_pls[1]
        copos_x_o_pls      =    copos_vec_o_pls[0]
        copos_y_o_pls      =    copos_vec_o_pls[1]
        covel_x_o_pls      =    covel_vec_o_pls[0]
        covel_y_o_pls      =    covel_vec_o_pls[1]
        thrust_acc_x_o_pls =      covel_x_o_pls # should be negative
        thrust_acc_y_o_pls =      covel_y_o_pls # should be negative
        acc_x_o_pls        = thrust_acc_x_o_pls
        acc_y_o_pls        = thrust_acc_y_o_pls
        ham_o_pls          = 1/2 * (thrust_acc_x_o_pls**2 + thrust_acc_y_o_pls**2) \
            + copos_x_o_pls * vel_x_o_pls + copos_y_o_pls * vel_y_o_pls \
            + covel_x_o_pls * acc_x_o_pls + covel_y_o_pls * acc_y_o_pls

        vel_x_f_mns        =   vel_vec_f_mns[0]
        vel_y_f_mns        =   vel_vec_f_mns[1]
        copos_x_f_mns      = copos_vec_f_mns[0]
        copos_y_f_mns      = copos_vec_f_mns[1]
        covel_x_f_mns      = covel_vec_f_mns[0]
        covel_y_f_mns      = covel_vec_f_mns[1]
        thrust_acc_x_f_mns = covel_x_f_mns # should be negative
        thrust_acc_y_f_mns = covel_y_f_mns # should be negative
        acc_x_f_mns        = thrust_acc_x_f_mns
        acc_y_f_mns        = thrust_acc_y_f_mns
        ham_f_mns          = 1/2 * (thrust_acc_x_f_mns**2 + thrust_acc_y_f_mns**2) \
            + copos_x_f_mns * vel_x_f_mns + copos_y_f_mns * vel_y_f_mns \
            + covel_x_f_mns * acc_x_f_mns + covel_y_f_mns * acc_y_f_mns
    if include_jacobian:
        # Partials of the Hamiltonian at the initial time
        # xxx

        # Partials of the Hamiltonian at the final time
        d_ham_f_mns__d_time_o_pls      = 0.0
        d_ham_f_mns__d_pos_vec_o_pls   = 0.0
        d_ham_f_mns__d_vel_vec_o_pls   = 0.0
        d_ham_f_mns__d_copos_vec_o_pls = 0.0
        d_ham_f_mns__d_covel_vec_o_pls = 0.0
        d_ham_f_mns__d_ham_o_pls       = 0.0

    # Calculate the error vector and error vector Jacobian
    #   error = [ delta [time_o, pos_vec_o, vel_vec_o, copos_vec_o, covel_vel_o, ham_o] ]
    #           [ delta [time_f, pos_vec_f, vel_vec_f, copos_vec_f, covel_vel_f, ham_f] ]
    #   jacobian = d(state_costate_f) / d(state_costate_o)

    # Enforce overrides for free variables
    # ordered_variables  = ['time', 'pos_vec', 'vel_vec', 'copos_vec', 'covel_vec', 'ham']
    # ordered_boundaries = ['o', 'f']
    # ordered_sides      = ['pls', 'mns']
    for var in ordered_variables:
        for bound in ordered_boundaries:
            if equality_parameters[var][bound]['mode'] == 'free':
                if bound == 'o':
                    exec(f"{var}_{bound}_mns = {var}_{bound}_pls")
                else: # assume bound == 'f'
                    exec(f"{var}_{bound}_pls = {var}_{bound}_mns")

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
    error_ham_o = ham_o_pls - ham_o_mns # trivial
    error_ham_f = ham_f_pls - ham_f_mns

    # Full error vector
    error_full = {
        'time' : {
            'o': error_time_o, # trivial
            'f': error_time_f  # trivial
        },
        'pos_vec' : {
            'o': error_pos_vec_o, # trivial
            'f': error_pos_vec_f
        },
        'vel_vec' : {
            'o': error_vel_vec_o, # trivial
            'f': error_vel_vec_f
        },
        'copos_vec' : {
            'o': error_copos_vec_o, # trivial
            'f': error_copos_vec_f  # trivial
        },
        'covel_vec' : {
            'o': error_covel_vec_o, # trivial
            'f': error_covel_vec_f  # trivial
        },
        'ham' : {
            'o': error_ham_o, # trivial
            'f': error_ham_f  
        }
    }

    # Error vector (currently not consolidated)
    error_components = []
    for bnd in ordered_boundaries:
        for var in ordered_variables:
            # print(f"error {var}_{bnd}: {error_full[var][bnd]}")
            error_components.append(error_full[var][bnd])
    error = np.hstack(error_components)

    # start here: error is correct. need to get rid of trivial zero errors
    if include_jacobian:
        # 4x4 : -d(pos_vec_f_mns, vel_vec_f_mns) / d(copos_vec_o_pls, covel_vec_o_pls)
        # 5x5: -d(pos_vec_f_mns, vel_vec_f_mns, ham_f_mns) / d(time_f_mns, copos_vec_o_pls, covel_vec_o_pls)
        error_jacobian = np.zeros((len(error), len(error)))

    # # Consolidate errors
    # case_choice = 1 # 1 : fixed fin-time; fixed init-pos; fixed init-vel; fixed fin-pos; fixed fin-vel
    #                 # 2 :  free fin-time; fixed init-pos; fixed init-vel; fixed fin-pos; fixed fin-vel
    # if case_choice == 1:
    #     error = np.hstack([error_pos_vec_f, error_vel_vec_f]) # 4 constraints
    #     if include_jacobian:
    #         error_jacobian = -stm_of[0:4, 4:8] # 4x4 : -d(pos_vec_f_mns, vel_vec_f_mns) / d(copos_vec_o_pls, covel_vec_o_pls)
    # elif case_choice == 2:
    #     error = np.hstack([error_pos_vec_f, error_vel_vec_f, error_ham_f]) # 5 constraints
    #     if include_jacobian:
    #         ... # 5x5: -d(pos_vec_f_mns, vel_vec_f_mns, ham_f_mns) / d(time_f_mns, copos_vec_o_pls, covel_vec_o_pls)

    # error = state_costate_f[:4] - np.hstack([pos_vec_f_pls, vel_vec_f_pls])
    # if include_jacobian:
    #     error_jacobian = stm_of[0:4, 4:8]

    # Return
    if include_jacobian:
        return error, error_jacobian
    else:
        return error