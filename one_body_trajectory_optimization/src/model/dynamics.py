import numpy as np
from src.utility.bounding_functions import bounded_smooth_func, bounded_nonsmooth_func, derivative__bounded_smooth_func, derivative__bounded_nonsmooth_func


def control_thrust_acceleration(
        min_type, 
        covel_x, covel_y,
        use_thrust_acc_limits, use_thrust_acc_smoothing, thrust_acc_min, thrust_acc_max,
        use_thrust_limits, use_thrust_smoothing, thrust_min, thrust_max,
        k_steepness,
        mass  : float      = 1.0            ,
        alpha : np.float64 = np.float64(1.0),
    ):
    """
    Control: thrust_acceleration_vector = thrust_acceleration_magnitude * thrust_acceleration_direction
    """
    if min_type == 'fuel':
        epsilon   = 1.0e-6
        covel_mag = np.sqrt(covel_x**2 + covel_y**2 + epsilon**2)
    elif min_type == 'energyfuel':
        covel_mag = np.sqrt(covel_x**2 + covel_y**2)
    else: # assume 'energy'
        covel_mag = np.sqrt(covel_x**2 + covel_y**2)
    covel_mag_inv = 1.0 / covel_mag
    if use_thrust_limits:
        thrust_acc_min = thrust_min / mass
        thrust_acc_max = thrust_max / mass
    if min_type == 'fuel':
        switching_func = covel_mag - 1.0
        if use_thrust_smoothing or use_thrust_acc_smoothing:
            heaviside_approx = 0.5 + 0.5 * np.tanh(k_steepness * switching_func)
            thrust_acc_mag   = thrust_acc_min + (thrust_acc_max - thrust_acc_min) * heaviside_approx
        else: # no use_thrust_smoothing and no use_thrust_acc_smoothing
            thrust_acc_mag = np.where(switching_func > 0.0, thrust_acc_max, thrust_acc_min)
    elif min_type == 'energyfuel':
        switching_func = covel_mag - (1.0 - alpha)
        if switching_func > 0:
            # Thrust on
            thrust_acc_mag = (covel_mag - (1 - alpha)) / alpha
            thrust_acc_mag = bounded_smooth_func(thrust_acc_mag, thrust_acc_min, thrust_acc_max, k_steepness)
        else:
            # Thrust off
            thrust_acc_mag = 0.0
    else: # assume 'energy'
        switching_func = np.zeros_like(covel_mag)
        thrust_acc_mag = covel_mag
        if use_thrust_limits or use_thrust_acc_limits:
            if use_thrust_smoothing or use_thrust_acc_smoothing:
                thrust_acc_mag = bounded_smooth_func(thrust_acc_mag, thrust_acc_min, thrust_acc_max, k_steepness)
            else: # no use_thrust_smoothing and no use_thrust_acc_smoothing
                thrust_acc_mag = bounded_nonsmooth_func(thrust_acc_mag, thrust_acc_min, thrust_acc_max)
    thrust_acc_x_dir = -covel_x * covel_mag_inv
    thrust_acc_y_dir = -covel_y * covel_mag_inv
    thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
    thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir

    return thrust_acc_x, thrust_acc_y, thrust_acc_mag, thrust_acc_x_dir, thrust_acc_y_dir, covel_mag, covel_mag_inv, switching_func


def one_body_dynamics__indirect(
        time                     : np.float64                     ,
        state_costate_scstm      : np.ndarray                     ,
        include_scstm            : bool       = False             ,
        min_type                 : str        = 'energy'          ,
        use_thrust_acc_limits    : bool       = False             ,
        use_thrust_acc_smoothing : bool       = False             ,
        thrust_acc_min           : np.float64 = np.float64(0.0e+0),
        thrust_acc_max           : np.float64 = np.float64(1.0e+0),
        use_thrust_limits        : bool       = False             ,
        use_thrust_smoothing     : bool       = False             ,
        thrust_min               : np.float64 = np.float64(0.0e+0),
        thrust_max               : np.float64 = np.float64(0.0e+0),
        exhaust_velocity         : np.float64 = np.float64(3.0e+3),
        k_steepness              : np.float64 = np.float64(0.0e+0),
        alpha                    : np.float64 = np.float64(1.0)   ,
        post_process             : bool       = False             ,
        constant_gravity         : np.float64 = np.float64(-9.81) ,
    ) -> np.ndarray:
    """
    Calculates the time derivatives of state variables for a free-body system
    with a minimum-fuel thrust control approximated by a smooth Heaviside function.
    
    This function represents the right-hand side of the ordinary differential
    equations (ODEs) for integration.

    Input
    -----
    time : float
        Current time (t).
    state_costate_scstm : np.ndarray
        Current integration state:
        [ pos_x, pos_y, vel_x, vel_y, copos_x, copos_y, covel_x, covel_y ]
        -   pos_x,   pos_y : position in x and y
        -   vel_x,   vel_y : velocity in x and y
        - copos_x, copos_y : co-position in x and y
        - covel_x, covel_y : co-velocity in x and y
    thrust_acc_min : float, optional
        Minimum thrust acceleration magnitude.
        Defaults to 1.0e-1.
    thrust_acc_max : float, optional
        Maximum thrust acceleration magnitude.
        Defaults to 1.0e1.
    k_steepness : float, optional
        Coefficient for the tanh approximation of the Heaviside function.
        Higher values lead to a sharper transition.
        Defaults to 1.0.

    Output
    ------
    np.ndarray
        Array of time derivatives of the state variables:
        [   dpos_x__dtime,   dpos_y__dtime,   dvel_x__dtime,   dvel_y__dtime,
          dcopos_x__dtime, dcopos_y__dtime, dcovel_x__dtime, dcovel_y__dtime  ]
    """

    # Validate input
    if use_thrust_acc_limits and use_thrust_limits:
        use_thrust_limits = False
    if (
            min_type=='fuel' 
            and use_thrust_acc_limits is False and use_thrust_limits is False
        ):
        use_thrust_acc_limits = True
        use_thrust_limits     = False

    # Unpack: full state into state and variational-state-transition matrix
    n_state_costate = 8
    state_costate   = state_costate_scstm[:n_state_costate]
    if include_scstm:
        scstm = state_costate_scstm[n_state_costate:n_state_costate+n_state_costate**2].reshape((n_state_costate, n_state_costate))

    # Unpack: state
    pos_x, pos_y, vel_x, vel_y, copos_x, copos_y, covel_x, covel_y = state_costate

    # Unpack: post-process states mass and objective
    mass = 1.0 # dummy value
    if post_process:
        mass                      = state_costate_scstm[-2]
        optimal_control_objective = state_costate_scstm[-1]
    if use_thrust_limits and not post_process:
        mass = state_costate_scstm[-1]

    # Control: thrust acceleration
    thrust_acc_x, thrust_acc_y, thrust_acc_mag, thrust_acc_x_dir, thrust_acc_y_dir, covel_mag, covel_mag_inv, switching_func = \
        control_thrust_acceleration(
            min_type, 
            covel_x, covel_y,
            use_thrust_acc_limits, use_thrust_acc_smoothing, thrust_acc_min, thrust_acc_max,
            use_thrust_limits, use_thrust_smoothing, thrust_min, thrust_max,
            k_steepness  ,
            mass  = mass ,
            alpha = alpha,
        )

    # Dynamics: free-body
    #   dstate/dtime = dynamics(time,state)
    dpos_x__dtime   = vel_x
    dpos_y__dtime   = vel_y
    dvel_x__dtime   =                    thrust_acc_x
    dvel_y__dtime   = constant_gravity + thrust_acc_y
    dcopos_x__dtime = 0.0
    dcopos_y__dtime = 0.0
    dcovel_x__dtime = -copos_x
    dcovel_y__dtime = -copos_y
    if post_process or use_thrust_limits:
        dmass__dtime = -thrust_acc_mag * mass / exhaust_velocity # type: ignore
    if post_process:
        if min_type == 'fuel':
            doptimal_control_objective__dtime =       thrust_acc_mag
        elif min_type == 'energyfuel':
            doptimal_control_objective__dtime = (1 - alpha) * thrust_acc_mag + alpha * 1/2 * thrust_acc_mag**2
        else: # assume 'energy'
            doptimal_control_objective__dtime = 1/2 * thrust_acc_mag**2
    
    dstate_costate__dtime = \
        np.array([
            dpos_x__dtime,
            dpos_y__dtime,
            dvel_x__dtime,
            dvel_y__dtime,
            dcopos_x__dtime,
            dcopos_y__dtime,
            dcovel_x__dtime,
            dcovel_y__dtime,
        ])

    # Variational Dynamics: free-body
    #   stm_dot = jacobian * stm
    if include_scstm:

        # Jacobian: free-body
        #   d(dstate/dtime)/dstate
        #     = [ d(  dpos_x/dtime)/dpos_x, d(  dpos_x/dtime)/dpos_y, d(  dpos_x/dtime)/dvel_x, d(  dpos_x/dtime)/dvel_y, d(  dpos_x/dtime)/dcopos_x, d(  dpos_x/dtime)/dcopos_y, d(  dpos_x/dtime)/dcovel_x, d(  dpos_x/dtime)/dcovel_y ]
        #       [ d(  dpos_y/dtime)/dpos_x, d(  dpos_y/dtime)/dpos_y, d(  dpos_y/dtime)/dvel_x, d(  dpos_y/dtime)/dvel_y, d(  dpos_y/dtime)/dcopos_x, d(  dpos_y/dtime)/dcopos_y, d(  dpos_y/dtime)/dcovel_x, d(  dpos_y/dtime)/dcovel_y ]
        #       [ d(  dvel_x/dtime)/dpos_x, d(  dvel_x/dtime)/dpos_y, d(  dvel_x/dtime)/dvel_x, d(  dvel_x/dtime)/dvel_y, d(  dvel_x/dtime)/dcopos_x, d(  dvel_x/dtime)/dcopos_y, d(  dvel_x/dtime)/dcovel_x, d(  dvel_x/dtime)/dcovel_y ]
        #       [ d(  dvel_y/dtime)/dpos_x, d(  dvel_y/dtime)/dpos_y, d(  dvel_y/dtime)/dvel_x, d(  dvel_y/dtime)/dvel_y, d(  dvel_y/dtime)/dcopos_x, d(  dvel_y/dtime)/dcopos_y, d(  dvel_y/dtime)/dcovel_x, d(  dvel_y/dtime)/dcovel_y ]
        #       [ d(dcopos_x/dtime)/dpos_x, d(dcopos_x/dtime)/dpos_y, d(dcopos_x/dtime)/dvel_x, d(dcopos_x/dtime)/dvel_y, d(dcopos_x/dtime)/dcopos_x, d(dcopos_x/dtime)/dcopos_y, d(dcopos_x/dtime)/dcovel_x, d(dcopos_x/dtime)/dcovel_y ]
        #       [ d(dcopos_y/dtime)/dpos_x, d(dcopos_y/dtime)/dpos_y, d(dcopos_y/dtime)/dvel_x, d(dcopos_y/dtime)/dvel_y, d(dcopos_y/dtime)/dcopos_x, d(dcopos_y/dtime)/dcopos_y, d(dcopos_y/dtime)/dcovel_x, d(dcopos_y/dtime)/dcovel_y ]
        #       [ d(dcovel_x/dtime)/dpos_x, d(dcovel_x/dtime)/dpos_y, d(dcovel_x/dtime)/dvel_x, d(dcovel_x/dtime)/dvel_y, d(dcovel_x/dtime)/dcopos_x, d(dcovel_x/dtime)/dcopos_y, d(dcovel_x/dtime)/dcovel_x, d(dcovel_x/dtime)/dcovel_y ]
        #       [ d(dcovel_y/dtime)/dpos_x, d(dcovel_y/dtime)/dpos_y, d(dcovel_y/dtime)/dvel_x, d(dcovel_y/dtime)/dvel_y, d(dcovel_y/dtime)/dcopos_x, d(dcovel_y/dtime)/dcopos_y, d(dcovel_y/dtime)/dcovel_x, d(dcovel_y/dtime)/dcovel_y ]
        #     = [                      0.0,                      0.0, d(  dpos_x/dtime)/dvel_x,                      0.0,                        0.0,                        0.0,                        0.0,                        0.0 ]
        #       [                      0.0,                      0.0,                      0.0, d(  dpos_y/dtime)/dvel_y,                        0.0,                        0.0,                        0.0,                        0.0 ]
        #       [                      0.0,                      0.0,                      0.0,                      0.0,                        0.0,                        0.0, d(  dvel_x/dtime)/dcovel_x, d(  dvel_x/dtime)/dcovel_y ]
        #       [                      0.0,                      0.0,                      0.0,                      0.0,                        0.0,                        0.0, d(  dvel_y/dtime)/dcovel_x, d(  dvel_y/dtime)/dcovel_y ]
        #       [                      0.0,                      0.0,                      0.0,                      0.0,                        0.0,                        0.0,                        0.0,                        0.0 ]
        #       [                      0.0,                      0.0,                      0.0,                      0.0,                        0.0,                        0.0,                        0.0,                        0.0 ]
        #       [                      0.0,                      0.0,                      0.0,                      0.0, d(dcovel_x/dtime)/dcopos_x,                        0.0,                        0.0,                        0.0 ]
        #       [                      0.0,                      0.0,                      0.0,                      0.0,                        0.0, d(dcovel_y/dtime)/dcopos_y,                        0.0,                        0.0 ]
        ddstatedtime__dstate = np.zeros((n_state_costate, n_state_costate))

        # Row 0 and 1
        #   d(dpos_x/dtime)/dvel_x
        #   d(dpos_y/dtime)/dvel_y
        ddstatedtime__dstate[0,2] = 1.0
        ddstatedtime__dstate[1,3] = 1.0

        # Row 2 and 3
        #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
        #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
        if use_thrust_limits:
            
            thrust_acc_min = thrust_min / mass
            thrust_acc_max = thrust_max / mass

        if use_thrust_limits or use_thrust_acc_limits:
            
            dcovel_mag__dcovel_x       = covel_x * covel_mag_inv
            dcovel_mag__dcovel_y       = covel_y * covel_mag_inv
            dcovel_x__covel_x          = 1.0
            dcovel_y__covel_y          = 1.0
            dcovel_mag_inv__dcovel_mag = -1.0 / covel_mag**2

        if min_type == 'fuel':

            if use_thrust_limits or use_thrust_acc_limits:

                dthrust_acc_x_dir__dcovel_x = -1.0 * dcovel_x__covel_x * covel_mag_inv - covel_x * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_x
                dthrust_acc_x_dir__dcovel_y =                                          - covel_x * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_y
                dthrust_acc_y_dir__dcovel_y = -1.0 * dcovel_y__covel_y * covel_mag_inv - covel_y * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_y
                dthrust_acc_y_dir__dcovel_x =                                          - covel_y * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_x

                if use_thrust_smoothing or use_thrust_acc_smoothing:

                    one_mns_tanhsq                 = 1.0 - np.tanh(k_steepness * switching_func)**2
                    onehalf_times_k_one_mns_tanhsq = 0.5 * k_steepness * one_mns_tanhsq
                    dheaviside_approx__dcovel_x    = onehalf_times_k_one_mns_tanhsq * dcovel_mag__dcovel_x
                    dheaviside_approx__dcovel_y    = onehalf_times_k_one_mns_tanhsq * dcovel_mag__dcovel_y
                    delta_thrust_acc_max2min       = thrust_acc_max - thrust_acc_min
                    dthrust_acc_mag__dcovel_x      = delta_thrust_acc_max2min * dheaviside_approx__dcovel_x
                    dthrust_acc_mag__dcovel_y      = delta_thrust_acc_max2min * dheaviside_approx__dcovel_y

                    # Row 2 and 3
                    #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                    #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                    ddstatedtime__dstate[2,6] = dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                    ddstatedtime__dstate[2,7] = dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y
                    ddstatedtime__dstate[3,6] = dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                    ddstatedtime__dstate[3,7] = dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

                else: # no use_thrust_smoothing and no use_thrust_acc_smoothing
                    
                    # Row 2 and 3
                    #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                    #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                    ddstatedtime__dstate[2,6] = thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                    ddstatedtime__dstate[2,7] = thrust_acc_mag * dthrust_acc_x_dir__dcovel_y
                    ddstatedtime__dstate[3,6] = thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                    ddstatedtime__dstate[3,7] = thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

        elif min_type == 'energyfuel':

            thrust_acc_mag_uncon              = (covel_mag - (1 - alpha)) / alpha
            dthrust_acc_mag_uncon__dcovel_mag = 1.0 / alpha
            dthrust_acc_mag__dcovel_mag       = derivative__bounded_smooth_func(thrust_acc_mag_uncon, thrust_acc_min, thrust_acc_max, k_steepness) * dthrust_acc_mag_uncon__dcovel_mag

            dthrust_acc_mag__dcovel_x = dthrust_acc_mag__dcovel_mag * dcovel_mag__dcovel_x
            dthrust_acc_mag__dcovel_y = dthrust_acc_mag__dcovel_mag * dcovel_mag__dcovel_y

            dthrust_acc_x_dir__dcovel_x = -1 * dcovel_x__covel_x * covel_mag_inv + -1 * covel_x * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_x
            dthrust_acc_x_dir__dcovel_y =                                          -1 * covel_x * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_y
            dthrust_acc_y_dir__dcovel_y = -1 * dcovel_y__covel_y * covel_mag_inv + -1 * covel_y * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_y
            dthrust_acc_y_dir__dcovel_x =                                          -1 * covel_y * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_x

            # Row 2 and 3
            #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
            #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
            ddstatedtime__dstate[2,6] = dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
            ddstatedtime__dstate[2,7] = dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y
            ddstatedtime__dstate[3,6] = dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
            ddstatedtime__dstate[3,7] = dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

        else: # assume 'energy'

            if use_thrust_limits or use_thrust_acc_limits:

                if use_thrust_smoothing or use_thrust_acc_smoothing:
                    dthrust_acc_mag__dcovel_mag = derivative__bounded_smooth_func(covel_mag, thrust_acc_min, thrust_acc_max, k_steepness)
                else: # no use_thrust_smoothing
                    dthrust_acc_mag__dcovel_mag = derivative__bounded_nonsmooth_func(covel_mag, thrust_acc_min, thrust_acc_max)

                dthrust_acc_mag__dcovel_x = dthrust_acc_mag__dcovel_mag * dcovel_mag__dcovel_x
                dthrust_acc_mag__dcovel_y = dthrust_acc_mag__dcovel_mag * dcovel_mag__dcovel_y

                dthrust_acc_x_dir__dcovel_x = -1 * dcovel_x__covel_x * covel_mag_inv + -1 * covel_x * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_x
                dthrust_acc_x_dir__dcovel_y =                                          -1 * covel_x * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_y
                dthrust_acc_y_dir__dcovel_y = -1 * dcovel_y__covel_y * covel_mag_inv + -1 * covel_y * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_y
                dthrust_acc_y_dir__dcovel_x =                                          -1 * covel_y * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_x

                # Row 2 and 3
                #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                ddstatedtime__dstate[2,6] = dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                ddstatedtime__dstate[2,7] = dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y
                ddstatedtime__dstate[3,6] = dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                ddstatedtime__dstate[3,7] = dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

            else: # no_limits

                # Row 2 and 3
                #   d(dvel_x/dtime)/dcovel_x
                #   d(dvel_y/dtime)/dcovel_y
                ddstatedtime__dstate[2,6] = -1.0
                ddstatedtime__dstate[3,7] = -1.0

        # Row 6 and 7
        #   d(dcovel_x__dtime)/dcopos_x
        #   d(dcovel_y__dtime)/dcopos_y
        ddstatedtime__dstate[6,4] = -1.0
        ddstatedtime__dstate[7,5] = -1.0

        # Combine: time-derivative of state-transition matrix
        dscstm__dtime = ddstatedtime__dstate @ scstm

    # Pack up: time-derivative of state or state+stm
    if include_scstm:
        # State + STM
        if post_process:
            return np.hstack([dstate_costate__dtime, dscstm__dtime.flatten(), dmass__dtime, doptimal_control_objective__dtime])
        elif use_thrust_limits and not post_process:
            return np.hstack([dstate_costate__dtime, dscstm__dtime.flatten(), dmass__dtime])
        else:
            return np.hstack([dstate_costate__dtime, dscstm__dtime.flatten()])
    else:
        # State
        if post_process:
            return np.hstack([dstate_costate__dtime, dmass__dtime, doptimal_control_objective__dtime])
        elif use_thrust_limits and not post_process:
            return np.hstack([dstate_costate__dtime, dmass__dtime])
        else:
            return dstate_costate__dtime

