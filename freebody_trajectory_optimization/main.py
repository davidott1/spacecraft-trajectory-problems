# Imports
# import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
# import matplotlib.pyplot as mplt
# import matplotlib.colors as mcolors
# import matplotlib.ticker as mticker
# import matplotlib.axes   as maxes
# from matplotlib.widgets import Button
# import json
import random
np.random.seed(42)
import astropy.units as u
from typing import Optional
from tqdm import tqdm
from src.loader.readers import read_general
from src.loader.process import process_input
from src.plotters.final_results import plot_final_results

# Free-body dynamics
def free_body_dynamics__indirect(
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
        post_process             : bool       = False             ,
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
    if post_process:
        mass                      = state_costate_scstm[-2]
        optimal_control_objective = state_costate_scstm[-1]
    if use_thrust_limits and not post_process:
        mass = state_costate_scstm[-1]

    # Control: thrust acceleration
    #   fuel   : thrust_acc_vec = -covel_vec / cvel_mag
    #   energy : thrust_acc_vec =  covel_vec
    if min_type == 'fuel':
        epsilon   = 1.0e-6
        covel_mag = np.sqrt(covel_x**2 + covel_y**2 + epsilon**2)
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
        thrust_acc_x_dir = -covel_x * covel_mag_inv
        thrust_acc_y_dir = -covel_y * covel_mag_inv
    else: # assume 'energy'
        thrust_acc_mag = covel_mag
        if use_thrust_limits or use_thrust_acc_limits:
            if use_thrust_smoothing or use_thrust_acc_smoothing:
                thrust_acc_mag = bounded_smooth_func(thrust_acc_mag, thrust_acc_min, thrust_acc_max, k_steepness)
            else: # no use_thrust_smoothing and no use_thrust_acc_smoothing
                thrust_acc_mag = bounded_nonsmooth_func(thrust_acc_mag, thrust_acc_min, thrust_acc_max)
        thrust_acc_x_dir = covel_x * covel_mag_inv
        thrust_acc_y_dir = covel_y * covel_mag_inv
    thrust_acc_x = thrust_acc_mag * thrust_acc_x_dir
    thrust_acc_y = thrust_acc_mag * thrust_acc_y_dir

    # Dynamics: free-body
    #   dstate/dtime = dynamics(time,state)
    dpos_x__dtime   = vel_x
    dpos_y__dtime   = vel_y
    dvel_x__dtime   = thrust_acc_x
    dvel_y__dtime   = thrust_acc_y
    dcopos_x__dtime = 0.0
    dcopos_y__dtime = 0.0
    dcovel_x__dtime = -copos_x
    dcovel_y__dtime = -copos_y
    if post_process or use_thrust_limits:
        dmass__dtime = -thrust_acc_mag * mass / exhaust_velocity # type: ignore
    if post_process:
        if min_type == 'fuel':
            doptimal_control_objective__dtime =       thrust_acc_mag
        else: # assume 'energy'
            doptimal_control_objective__dtime = 0.5 * thrust_acc_mag**2
    
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

        # Row 1 and 2
        #   d(dpos_x/dtime)/dvel_x
        #   d(dpos_y/dtime)/dvel_y
        ddstatedtime__dstate[0,2] = 1.0
        ddstatedtime__dstate[1,3] = 1.0

        # Row 3 and 4
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

                    # Row 3 and 4
                    #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                    #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                    ddstatedtime__dstate[2,6] = dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                    ddstatedtime__dstate[2,7] = dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y
                    ddstatedtime__dstate[3,6] = dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                    ddstatedtime__dstate[3,7] = dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

                else: # no use_thrust_smoothing and no use_thrust_acc_smoothing
                    
                    # Row 3 and 4
                    #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                    #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                    ddstatedtime__dstate[2,6] = thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                    ddstatedtime__dstate[2,7] = thrust_acc_mag * dthrust_acc_x_dir__dcovel_y
                    ddstatedtime__dstate[3,6] = thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                    ddstatedtime__dstate[3,7] = thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

        else: # assume 'energy'

            if use_thrust_limits or use_thrust_acc_limits:

                if use_thrust_smoothing or use_thrust_acc_smoothing:
                    dthrust_acc_mag__dcovel_mag = derivative__bounded_smooth_func(covel_mag, thrust_acc_min, thrust_acc_max, k_steepness)
                else: # no use_thrust_smoothing
                    dthrust_acc_mag__dcovel_mag = derivative__bounded_nonsmooth_func(covel_mag, thrust_acc_min, thrust_acc_max)

                dthrust_acc_mag__dcovel_x = dthrust_acc_mag__dcovel_mag * dcovel_mag__dcovel_x
                dthrust_acc_mag__dcovel_y = dthrust_acc_mag__dcovel_mag * dcovel_mag__dcovel_y

                dthrust_acc_x_dir__dcovel_x = dcovel_x__covel_x * covel_mag_inv + covel_x * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_x
                dthrust_acc_x_dir__dcovel_y =                                   + covel_x * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_y
                dthrust_acc_y_dir__dcovel_y = dcovel_y__covel_y * covel_mag_inv + covel_y * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_y
                dthrust_acc_y_dir__dcovel_x =                                   + covel_y * dcovel_mag_inv__dcovel_mag * dcovel_mag__dcovel_x

                # Row 3 and 4
                #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                ddstatedtime__dstate[2,6] = dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                ddstatedtime__dstate[2,7] = dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y
                ddstatedtime__dstate[3,6] = dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                ddstatedtime__dstate[3,7] = dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

            else: # no_limits

                # Row 3 and 4
                #   d(dvel_x/dtime)/dcovel_x
                #   d(dvel_y/dtime)/dcovel_y
                ddstatedtime__dstate[2,6] = 1.0
                ddstatedtime__dstate[3,7] = 1.0

        # Row 7 and 8
        #   d(dcovel_x_dtime)/dcopos_x
        #   d(dcovel_y__dttime)/dcopos_y
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

# Constant-gravity dynamics
def constantgravitydynamics__minfuel__indirect_thrustaccmax_heaviside_stm(
        time             : float              ,
        state            : np.ndarray         ,
        thrust_acc_min   : float      = 1.0e-1,
        thrust_acc_max   : float      = 1.0e+1,
        k_steepness      : float      = 1.0   ,
        constant_gravity : float      = -9.81 , 
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
    state : np.ndarray
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
    constant_gravity : float, optional
        Gravity constant in the y-direction.
        Defaults to -9.81 m/s^2.

    Output
    ------
    np.ndarray
        Array of time derivatives of the state variables:
        [   dpos_x__dtime,   dpos_y__dtime,   dvel_x__dtime,   dvel_y__dtime,
          dcopos_x__dtime, dcopos_y__dtime, dcovel_x__dtime, dcovel_y__dtime  ]
    """

    # Unpack: state into scalar components
    pos_x, pos_y, vel_x, vel_y, copos_x, copos_y, covel_x, covel_y = state

    # Control: thrust acceleration
    #   thrust_acc_vec = -covel_vec / cvel_mag
    epsilon          = np.float64(1.0e-6)
    covel_mag        = np.sqrt(covel_x**2 + covel_y**2 + epsilon**2)
    switching_func   = covel_mag - 1.0
    heaviside_approx = 0.5 + 0.5 * np.tanh(k_steepness * switching_func)
    thrust_acc_mag   = thrust_acc_min + (thrust_acc_max - thrust_acc_min) * heaviside_approx
    thrust_acc_x_dir = -covel_x / covel_mag
    thrust_acc_y_dir = -covel_y / covel_mag
    thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
    thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir

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
    
    # Pack up: scalar components into state
    return np.array([
        dpos_x__dtime,
        dpos_y__dtime,
        dvel_x__dtime,
        dvel_y__dtime,
        dcopos_x__dtime,
        dcopos_y__dtime,
        dcovel_x__dtime,
        dcovel_y__dtime,
    ])

# Two-point boundary-value-problem objective and the associated jacobian
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

# Generate guess
def generate_guess(
        time_span                    : np.ndarray                     ,
        boundary_condition_pos_vec_o : np.ndarray                     ,
        boundary_condition_vel_vec_o : np.ndarray                     ,
        boundary_condition_pos_vec_f : np.ndarray                     ,
        boundary_condition_vel_vec_f : np.ndarray                     ,
        min_type                     : str        = 'energy'          ,
        mass_o                       : np.float64 = np.float64(1.0e+3),
        use_thrust_acc_limits        : bool       = True              ,
        thrust_acc_min               : np.float64 = np.float64(0.0e+0),
        thrust_acc_max               : np.float64 = np.float64(1.0e+1),
        use_thrust_limits            : bool       = False             ,
        thrust_min                   : np.float64 = np.float64(0.0e+0),
        thrust_max                   : np.float64 = np.float64(1.0e+1),
        k_steepness                  : np.float64 = np.float64(0.0e+0),
        init_guess_steps             : int        = 3000              ,
    ):
    """
    Generates a robust initial guess for the co-states: copos_vec, covel_vec
    """
    print("\nInitial Guess Process")

    # Loop through random guesses for the costates
    print("\nRandom Initial Guess Generation")
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
                tqdm.write(f"                                     {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Fixed':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s} {'Free':>14s}")
                tqdm.write(f"                {'Step':>5s} {'Error-Mag':>14s} {'Pos-Xo':>14s} {'Pos-Yo':>14s} {'Vel-Xo':>14s} {'Vel-Yo':>14s} {'Co-Pos-Xo':>14s} {'Co-Pos-Yo':>14s} {'Co-Vel-Xo':>14s} {'Co-Vel-Yo':>14s}")
            integ_state_min_str = ' '.join(f"{x:>14.6e}" for x in integ_state_min)
            tqdm.write(f"           {idx_min:>5d}/{init_guess_steps:>4d} {error_mag_min:>14.6e} {integ_state_min_str}")

    # Pack up and print solution
    costate_o_guess = decision_state_min
    print(f"  MIN: *** {idx_min:>5d}/{init_guess_steps:>4d} {error_mag_min:>14.6e} {integ_state_min_str} ***")
    return costate_o_guess

# Optimal trajectory solver
def optimal_trajectory_solve(
        time_span                    : np.ndarray                     ,
        boundary_condition_pos_vec_o : np.ndarray                     ,
        boundary_condition_vel_vec_o : np.ndarray                     ,
        boundary_condition_pos_vec_f : np.ndarray                     ,
        boundary_condition_vel_vec_f : np.ndarray                     ,
        min_type                     : str        = 'energy'          ,
        use_thrust_acc_limits        : bool       = True              ,
        thrust_acc_min               : np.float64 = np.float64(0.0e+0),
        thrust_acc_max               : np.float64 = np.float64(1.0e+1),
        use_thrust_limits            : bool       = False             ,
        thrust_min                   : np.float64 = np.float64(0.0e+0),
        thrust_max                   : np.float64 = np.float64(1.0e+1),
        k_idxinitguess               : np.float64 = np.float64(1.0e-1),
        k_idxfinsoln                 : np.float64 = np.float64(1.0e+1),
        k_idxdivs                    : int        = 100               ,
        init_guess_steps             : int        = 3000              ,
        mass_o                       : np.float64 = np.float64(1.0e+3),
        input_filename               : str        = 'blank.json'      ,
    ):
    """
    Main solver that implements the two-stage continuation process
    using the unified smoothed dynamics.
    """

    # Generate initial guess for the costates
    decision_state_initguess = \
        generate_guess(
            time_span                                           ,
            boundary_condition_pos_vec_o                        ,
            boundary_condition_vel_vec_o                        ,
            boundary_condition_pos_vec_f                        ,
            boundary_condition_vel_vec_f                        ,
            min_type                     = min_type             ,
            mass_o                       = mass_o               ,
            use_thrust_acc_limits        = use_thrust_acc_limits,
            thrust_acc_min               = thrust_acc_min       ,
            thrust_acc_max               = thrust_acc_max       ,
            use_thrust_limits            = use_thrust_limits    ,
            thrust_min                   = thrust_min           ,
            thrust_max                   = thrust_max           ,
            k_steepness                  = k_idxinitguess       ,
            init_guess_steps             = init_guess_steps     ,
        )

    # Optimize and enforce thrust or thrust-acc constraints
    print("\nOptimizing Process")

    # Solve for the optimal min-fuel or min-energy trajectory

    # Thrust- or Thrust-Acc-Steepness Continuation Process
    if use_thrust_acc_limits or use_thrust_limits:
        print("\nThrust- or Thrust-Acc-Steepness Continuation Process")

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
                use_thrust_acc_smoothing     = False                 , # temp
                thrust_acc_min               = thrust_acc_min       ,
                thrust_acc_max               = thrust_acc_max       ,
                use_thrust_limits            = use_thrust_limits    ,
                use_thrust_smoothing         = False                 , # temp
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
    print()
    print("Final Solution")
    print("\nRoot-Solve Results")
    print(soln_root)
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
                use_thrust_acc_smoothing     = False                 , # temp
                thrust_acc_min               = thrust_acc_min       ,
                thrust_acc_max               = thrust_acc_max       ,
                use_thrust_limits            = use_thrust_limits    ,
                use_thrust_smoothing         = False                 , # temp
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

    print("\nState Error Check")
    print(f"           {'Pos-Xf':>14s} {'Pos-Yf':>14s} {'Vel-Xf':>14s} {'Vel-Yf':>14s}")
    print(f"           {    'm':>14s} {    'm':>14s} {  'm/s':>14s} {  'm/s':>14s}")
    print(f"  Target : {boundary_condition_pos_vec_f[0]:>14.6e} {boundary_condition_pos_vec_f[1]:>14.6e} {boundary_condition_vel_vec_f[0]:>14.6e} {boundary_condition_vel_vec_f[1]:>14.6e}")
    print(f"  Approx : {    state_f_approx_finalsoln[0]:>14.6e} {    state_f_approx_finalsoln[1]:>14.6e} {    state_f_approx_finalsoln[2]:>14.6e} {    state_f_approx_finalsoln[3]:>14.6e}")
    print(f"  Error  : {  error_approx_finalsoln_vec[0]:>14.6e} {  error_approx_finalsoln_vec[1]:>14.3e} {  error_approx_finalsoln_vec[2]:>14.6e} {  error_approx_finalsoln_vec[3]:>14.6e}")
    print(f"  Actual : {           state_f_finalsoln[0]:>14.6e} {           state_f_finalsoln[1]:>14.6e} {           state_f_finalsoln[2]:>14.6e} {           state_f_finalsoln[3]:>14.6e}")
    print(f"  Error  : {         error_finalsoln_vec[0]:>14.6e} {         error_finalsoln_vec[1]:>14.3e} {         error_finalsoln_vec[2]:>14.6e} {         error_finalsoln_vec[3]:>14.6e}")

    # Enforce initial and final co-state boundary conditions (trivial right now)
    boundary_condition_copos_vec_o = decision_state_initguess[0:2]
    boundary_condition_covel_vec_o = decision_state_initguess[2:4]
    boundary_condition_copos_vec_f = results_finalsoln.y[4:6, -1]
    boundary_condition_covel_vec_f = results_finalsoln.y[6:8, -1]

    # Plot the results
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
        input_filename                 = input_filename       ,
    )

    # End
    print()


# Optimal trajectory input
def optimal_trajectory_input():

    # Read input
    input_files_params = read_general()

    # Process input
    input_processed = process_input(input_files_params)

    return input_processed

# Main
if __name__ == '__main__':

    # Start optimization trajectory program
    print(f"\nOPTIMAL TRAJECTORY PROGRAM")

    # Optimal trajectory input
    (
        min_type                    ,
        time_span                   ,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        use_thrust_acc_limits       ,
        thrust_acc_min              ,
        thrust_acc_max              ,
        use_thrust_limits           ,
        thrust_min                  ,
        thrust_max                  ,
        k_idxinitguess              ,
        k_idxfinsoln                ,
        k_idxdivs                   ,
        init_guess_steps            ,
        mass_o                      ,
        input_filename              ,
    ) = \
        optimal_trajectory_input()

    # Optimal trajectory solve
    optimal_trajectory_solve(
        time_span                                           ,
        boundary_condition_pos_vec_o                        ,
        boundary_condition_vel_vec_o                        ,
        boundary_condition_pos_vec_f                        ,
        boundary_condition_vel_vec_f                        ,
        min_type                     = min_type             ,
        use_thrust_acc_limits        = use_thrust_acc_limits,
        thrust_acc_min               = thrust_acc_min       ,
        thrust_acc_max               = thrust_acc_max       ,
        use_thrust_limits            = use_thrust_limits    ,
        thrust_min                   = thrust_min           ,
        thrust_max                   = thrust_max           ,
        k_idxinitguess               = k_idxinitguess       ,
        k_idxfinsoln                 = k_idxfinsoln         , 
        k_idxdivs                    = k_idxdivs            ,
        init_guess_steps             = init_guess_steps     ,
        mass_o                       = mass_o               ,
        input_filename               = input_filename       ,
    )

    # End optimization trajectory program
    print()

