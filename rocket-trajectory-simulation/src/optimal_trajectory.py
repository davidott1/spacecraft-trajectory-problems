# Imports
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as mplt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.axes   as maxes
from matplotlib.widgets import Button
import json
import random
np.random.seed(42)
import astropy.units as u
from functools import partial
from typing import Optional
from tqdm import tqdm

def smax(val1, val2, k):
    """
    Smooth maximum using Log-Sum-Exp. This expression is mumerically stable and produces a value 
    slightly larger than max(val1,val2), depending on k.
    """
    m = np.maximum(k * val1, k * val2)
    return (1.0 / k) * (m + np.log(np.exp(k * val1 - m) + np.exp(k * val2 - m)))
def dsmax__dval1(val1, val2, k):
    """
    Calculates the partial derivative of smax(val1, val2, k) with respect to val1.
    """
    return np.exp(k * (val1 - smax(val1, val2, k)))
def dsmax__dval2(val1, val2, k):
    """
    Calculates the partial derivative of smax(val1, val2, k) with respect to val2.
    """
    return dsmax_dval1(val2, val1, k)

def smin(val1, val2, k):
    """
    Smooth minimum using Log-Sum-Exp. This expression is mumerically stable and produces a value 
    slightly smaller than min(val1,val2), depending on k.
    """
    m = np.maximum(-k * val1, -k * val2)
    return (-1.0 / k) * (m + np.log(np.exp(-k * val1 - m) + np.exp(-k * val2 - m)))
def dsmin__dval1(val1, val2, k):
    """
    Calculates the partial derivative of smin with respect to val1.
    """
    return np.exp(-k * (val1 - smin(val1, val2, k)))
def dsmin__dval2(val1, val2, k):
    """
    Calculates the partial derivative of smin with respect to val2.
    """
    return dsmin_dval1(val2, val1, k)

# Free-body dynamics
def freebodydynamics__indirect(
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
        k_heaviside              : np.float64 = np.float64(0.0e+0),
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
    k_heaviside : float, optional
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

    # fuel
    #   thrust_limits
    #     no_thrust_smoothing
    #     thrust_smoothing
    #   thrust_acc_limits
    #     no_thrust_acc_smoothing
    #     thrust_acc_smoothing
    # energy
    #   thrust_limits
    #     no_thrust_smoothing
    #     thrust_smoothing
    #   thrust_acc_limits
    #     no_thrust_acc_smoothing
    #     thrust_acc_smoothing
    #   no_limits

    # if min_type == 'fuel':
    #     if use_thrust_limits:
    #         if use_thrust_smoothing:
    #             ...
    #         else: # use_no_thrust_smoothing
    #             ...
    #     elif use_thrust_acc_limits:
    #         if use_thrust_acc_smoothing:
    #             ...
    #         else: # use_no_thrust_acc_smoothing
    #             ...
    # else: # assume 'energy'
    #     if use_thrust_limits:
    #         if use_thrust_smoothing:
    #             ...
    #         else: # use_no_thrust_smoothing
    #             ...
    #     elif use_thrust_acc_limits:
    #         if use_thrust_acc_smoothing:
    #             ...
    #         else: # use_no_thrust_acc_smoothing
    #             ...
    #     else: # no_limits
    #         ...

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
        scstm = state_costate_scstm[n_state_costate:].reshape((n_state_costate, n_state_costate))

    # Unpack: state
    pos_x, pos_y, vel_x, vel_y, copos_x, copos_y, covel_x, covel_y = state_costate

    # Unpack: post-process states mass and objective
    if post_process:
        mass                      = state_costate_scstm[-2]
        optimal_control_objective = state_costate_scstm[-1]

    # Control: thrust acceleration
    #   fuel   : thrust_acc_vec = -covel_vec / cvel_mag
    #   energy : thrust_acc_vec =  covel_vec
    if min_type == 'fuel':
        epsilon        = np.float64(1.0e-6)
        covel_mag      = np.sqrt(covel_x**2 + covel_y**2 + epsilon**2)
        switching_func = covel_mag - np.float64(1.0)
        if use_thrust_limits:
            # thrust_mag       = np.float64(0.0e+0)
            # thrust_acc_mag   = np.float64(0.0e+0) # thrust_mag / mass
            # thrust_acc_x_dir = -covel_x / covel_mag
            # thrust_acc_y_dir = -covel_y / covel_mag
            # thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
            # thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir
            if use_thrust_smoothing:
                ...
            else: # use_no_thrust_smoothing
                ...
        elif use_thrust_acc_limits:
            if use_thrust_acc_smoothing:
                heaviside_approx = np.float64(0.5) + np.float64(0.5) * np.tanh(k_heaviside * switching_func)
                thrust_acc_mag   = thrust_acc_min + (thrust_acc_max - thrust_acc_min) * heaviside_approx
            else: # use_no_thrust_acc_smoothing
                if switching_func > np.float64(0.0):
                    thrust_acc_mag = thrust_acc_max
                elif switching_func < np.float64(0.0):
                    thrust_acc_mag = thrust_acc_min
                else:
                    thrust_acc_mag = thrust_acc_min # undetermined, thrust_acc_min chosen
        thrust_acc_x_dir = -covel_x / covel_mag
        thrust_acc_y_dir = -covel_y / covel_mag
        thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
        thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir
    else: # assume 'energy'
        covel_mag = np.sqrt(covel_x**2 + covel_y**2)
        if use_thrust_limits:
            # thrust_mag       = np.float64(0.0e+0)
            # thrust_acc_mag   = np.float64(0.0e+0) # thrust_mag / mass
            # thrust_acc_x_dir = covel_x / covel_mag
            # thrust_acc_y_dir = covel_y / covel_mag
            # thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
            # thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir
            if use_thrust_smoothing:
                ...
            else: # use_no_thrust_smoothing
                ...
        elif use_thrust_acc_limits:
            if use_thrust_acc_smoothing:
                k_steepness    = k_heaviside
                thrust_acc_mag = covel_mag
                thrust_acc_mag = smin( thrust_acc_mag, thrust_acc_max, k_steepness ) # max thrust-acc constraint
                thrust_acc_mag = smax( thrust_acc_mag, thrust_acc_min, k_steepness ) # min thrust-acc constraint
            else: # use_no_thrust_acc_smoothing
                thrust_acc_mag = covel_mag
                thrust_acc_mag = min( thrust_acc_mag, thrust_acc_max ) # max thrust-acc constraint
                thrust_acc_mag = max( thrust_acc_mag, thrust_acc_min ) # min thrust-acc constraint
            thrust_acc_x_dir = covel_x / covel_mag
            thrust_acc_y_dir = covel_y / covel_mag
            thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
            thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir
        else: # no_limits
            thrust_acc_mag   = covel_mag
            thrust_acc_x_dir = covel_x / covel_mag
            thrust_acc_y_dir = covel_y / covel_mag
            thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
            thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir

    # Dynamics: free-body
    #   dstate/dtime = dynamics(time,state)
    dpos_x__dtime   = vel_x
    dpos_y__dtime   = vel_y
    dvel_x__dtime   = thrust_acc_x
    dvel_y__dtime   = thrust_acc_y
    dcopos_x__dtime = np.float64(0.0)
    dcopos_y__dtime = np.float64(0.0)
    dcovel_x__dtime = -copos_x
    dcovel_y__dtime = -copos_y
    if post_process:
        dmass__dtime = -thrust_acc_mag * mass / exhaust_velocity # type: ignore
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

        # Row 1
        #   d(dpos_x/dtime)/dvel_x
        ddstatedtime__dstate[0,2] = np.float64(1.0)

        # Row 2
        #   d(dpos_y/dtime)/dvel_y
        ddstatedtime__dstate[1,3] = np.float64(1.0)

        if min_type == 'fuel':
            if use_thrust_limits:
                if use_thrust_smoothing:
                    ...
                else: # use_no_thrust_smoothing
                    ...
            elif use_thrust_acc_limits:

                # Common terms
                dcovel_mag__dcovel_x = covel_x / covel_mag
                dcovel_mag__dcovel_y = covel_y / covel_mag

                dthrust_acc_x_dir__dcovel_x = np.float64(-1.0) / covel_mag - covel_x * (np.float64(-1.0) / covel_mag**2) * dcovel_mag__dcovel_x
                dthrust_acc_x_dir__dcovel_y =                              - covel_x * (np.float64(-1.0) / covel_mag**2) * dcovel_mag__dcovel_y
                dthrust_acc_y_dir__dcovel_y = np.float64(-1.0) / covel_mag - covel_y * (np.float64(-1.0) / covel_mag**2) * dcovel_mag__dcovel_y
                dthrust_acc_y_dir__dcovel_x =                              - covel_y * (np.float64(-1.0) / covel_mag**2) * dcovel_mag__dcovel_x

                if use_thrust_acc_smoothing:
                    
                    # Common terms
                    one_mns_tanhsq              = np.float64(1.0) - np.tanh(k_heaviside * switching_func)**2
                    dheaviside_approx__dcovel_x = np.float64(0.5) * k_heaviside * one_mns_tanhsq * dcovel_mag__dcovel_x
                    dheaviside_approx__dcovel_y = np.float64(0.5) * k_heaviside * one_mns_tanhsq * dcovel_mag__dcovel_y
                    dthrust_acc_mag__dcovel_x   = (thrust_acc_max - thrust_acc_min) * dheaviside_approx__dcovel_x
                    dthrust_acc_mag__dcovel_y   = (thrust_acc_max - thrust_acc_min) * dheaviside_approx__dcovel_y

                    # Row 3
                    #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                    ddstatedtime__dstate[2,6] = dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                    ddstatedtime__dstate[2,7] = dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y

                    # Row 4
                    #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                    ddstatedtime__dstate[3,6] = dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                    ddstatedtime__dstate[3,7] = dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

                else: # use_no_thrust_acc_smoothing
                    
                    # Row 3
                    #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                    ddstatedtime__dstate[2,6] = thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                    ddstatedtime__dstate[2,7] = thrust_acc_mag * dthrust_acc_x_dir__dcovel_y

                    # Row 4
                    #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                    ddstatedtime__dstate[3,6] = thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                    ddstatedtime__dstate[3,7] = thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

        else: # assume 'energy'
            if use_thrust_limits:
                if use_thrust_smoothing:
                    ...
                else: # use_no_thrust_smoothing
                    ...
            elif use_thrust_acc_limits:
                if use_thrust_acc_smoothing:

                    # Common terms
                    dcovel_mag__dcovel_x = covel_x / covel_mag
                    dcovel_mag__dcovel_y = covel_y / covel_mag

                    thrust_acc_mag = covel_mag
                    thrust_acc_mag = smin( thrust_acc_mag, thrust_acc_max, k_steepness ) # max thrust-acc constraint
                    thrust_acc_mag = smax( thrust_acc_mag, thrust_acc_min, k_steepness ) # min thrust-acc constraint

                    dthrust_acc_mag__dcovel_x = dsmax__dval1( smin(covel_mag, thrust_acc_max, k_steepness), thrust_acc_min, k_steepness ) * dsmin__dval1(covel_mag, thrust_acc_max, k_steepness) * dcovel_mag__dcovel_x
                    dthrust_acc_mag__dcovel_y = dsmax__dval1( smin(covel_mag, thrust_acc_max, k_steepness), thrust_acc_min, k_steepness ) * dsmin__dval1(covel_mag, thrust_acc_max, k_steepness) * dcovel_mag__dcovel_y

                    thrust_acc_x_dir = covel_x / covel_mag
                    thrust_acc_y_dir = covel_y / covel_mag

                    dthrust_acc_x_dir__dcovel_x = np.float64(1.0) / covel_mag - covel_x * (np.float64(1.0) / covel_mag**2) * dcovel_mag__dcovel_x
                    dthrust_acc_x_dir__dcovel_y =                             - covel_x * (np.float64(1.0) / covel_mag**2) * dcovel_mag__dcovel_y
                    dthrust_acc_y_dir__dcovel_y = np.float64(1.0) / covel_mag - covel_y * (np.float64(1.0) / covel_mag**2) * dcovel_mag__dcovel_y
                    dthrust_acc_y_dir__dcovel_x =                             - covel_y * (np.float64(1.0) / covel_mag**2) * dcovel_mag__dcovel_x

                    # Row 3
                    #   d(dvel_x__dtime)/dcovel_x, d(dvel_x__dtime)/dcovel_y
                    ddstatedtime__dstate[2,6] = dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x
                    ddstatedtime__dstate[2,7] = dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y

                    # Row 4
                    #   d(dvel_y__dtime)/dcovel_x, d(dvel_y__dtime)/dcovel_y
                    ddstatedtime__dstate[3,6] = dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x
                    ddstatedtime__dstate[3,7] = dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

                else: # use_no_thrust_acc_smoothing
                    ...
            else: # no_limits

                # Row 3
                #   d(dvel_x/dtime)/dcovel_x
                ddstatedtime__dstate[2,6] = np.float64(+1.0)

                # Row 4
                #   d(dvel_y/dtime)/dcovel_y
                ddstatedtime__dstate[3,7] = np.float64(+1.0)

        # Row 7
        #   d(dcovel_x_dtime)/dcopos_x
        ddstatedtime__dstate[6,4] = np.float64(-1.0)

        # Row 8
        #   d(dcovel_y__dttime)/dcopos_y
        ddstatedtime__dstate[7,5] = np.float64(-1.0)

        # Combine: time-derivative of state-transition matrix
        dscstm__dtime = np.dot(ddstatedtime__dstate, scstm)

    # Pack up: time-derivative of state or state+stm
    if include_scstm:
        # State + STM
        if post_process:
            return np.hstack([dstate_costate__dtime, dscstm__dtime.flatten(), dmass__dtime, doptimal_control_objective__dtime])
        else:
            return np.hstack([dstate_costate__dtime, dscstm__dtime.flatten()])
    else:
        # State
        if post_process:
            return np.hstack([dstate_costate__dtime, dmass__dtime, doptimal_control_objective__dtime])
        else:
            return dstate_costate__dtime

# Constant-gravity dynamics
def constantgravitydynamics__minfuel__indirect_thrustaccmax_heaviside_stm(
        time             : float              ,
        state            : np.ndarray         ,
        thrust_acc_min   : float      = 1.0e-1,
        thrust_acc_max   : float      = 1.0e+1,
        k_heaviside      : float      = 1.0   ,
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
    k_heaviside : float, optional
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
    heaviside_approx = 0.5 + 0.5 * np.tanh(k_heaviside * switching_func)
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
        use_thrust_acc_limits        : bool       = True              ,
        use_thrust_acc_smoothing     : bool       = False             ,
        thrust_acc_min               : np.float64 = np.float64(0.0e+0),
        thrust_acc_max               : np.float64 = np.float64(1.0e+1),
        use_thrust_limits            : bool       = False             ,
        use_thrust_smoothing         : bool       = False             ,
        thrust_min                   : np.float64 = np.float64(0.0e+0),
        thrust_max                   : np.float64 = np.float64(1.0e+1),
        k_heaviside                  : np.float64 = np.float64(0.0e+0),
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

    # Integrate
    if min_type == 'fuel':
        solve_ivp_func = \
            lambda time, state_costate_scstm: \
                freebodydynamics__indirect(
                    time                                               ,
                    state_costate_scstm                                ,
                    include_scstm            = include_scstm           ,
                    min_type                 = min_type                ,
                    use_thrust_acc_limits    = use_thrust_acc_limits   ,
                    use_thrust_acc_smoothing = use_thrust_acc_smoothing,
                    thrust_acc_min           = thrust_acc_min          ,
                    thrust_acc_max           = thrust_acc_max          ,
                    k_heaviside              = k_heaviside             ,
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
    else: # assume energy
        solve_ivp_func = \
            lambda time, state_costate_scstm: \
                freebodydynamics__indirect(
                    time                                               ,
                    state_costate_scstm                                ,
                    include_scstm            = include_scstm           ,
                    min_type                 = min_type                ,
                    use_thrust_acc_limits    = use_thrust_acc_limits   ,
                    use_thrust_acc_smoothing = use_thrust_acc_smoothing,
                    thrust_acc_min           = thrust_acc_min          ,
                    thrust_acc_max           = thrust_acc_max          ,
                    k_heaviside              = k_heaviside             ,
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
        stm_of = state_costate_scstm_f[8:].reshape((8,8))
    
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
        use_thrust_acc_limits        : bool       = True              ,
        thrust_acc_min               : np.float64 = np.float64(0.0e+0),
        thrust_acc_max               : np.float64 = np.float64(1.0e+1),
        use_thrust_limits            : bool       = False             ,
        thrust_min                   : np.float64 = np.float64(0.0e+0),
        thrust_max                   : np.float64 = np.float64(1.0e+1),
        k_heaviside                  : np.float64 = np.float64(0.0e+0),
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
                use_thrust_acc_limits        = use_thrust_acc_limits,
                use_thrust_acc_smoothing     = True                 ,
                thrust_acc_min               = thrust_acc_min       ,
                thrust_acc_max               = thrust_acc_max       ,
                k_heaviside                  = k_heaviside          ,
                min_type                     = min_type             ,
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
    ):
    """
    Main solver that implements the two-stage continuation process
    using the unified smoothed dynamics.
    """

    # Generate initial guess for the costates
    decisionstate_initguess = \
        generate_guess(
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
            k_heaviside                  = k_idxinitguess       ,
            init_guess_steps             = init_guess_steps     ,
        )

    # Optimize and enforce thrust or thrust-acc constraints
    print("\nOptimizing Process")
    print("\nThrust- or Thrust-Acc-Steepness Continuation Process")

    # Select minimization type
    if min_type == "fuel":
        # Solve for the optimal min-fuel trajectory

        # Thrust- or Thrust-Acc-Steepness Continuation Process

        # Loop initialization
        results_k_idx = {}

        # Loop
        k_idxinitguess_to_idxfinsoln = np.logspace(np.log(k_idxinitguess), np.log(k_idxfinsoln), k_idxdivs)
        for idx, k_idx in tqdm(enumerate(k_idxinitguess_to_idxfinsoln), desc="Processing", leave=False, total=len(k_idxinitguess_to_idxfinsoln)):
            root_func = \
                lambda decisionstate_initguess: \
                    tpbvp_objective_and_jacobian(
                        decisionstate_initguess                             , 
                        time_span                                           ,
                        boundary_condition_pos_vec_o                        ,
                        boundary_condition_vel_vec_o                        ,
                        boundary_condition_pos_vec_f                        ,
                        boundary_condition_vel_vec_f                        ,
                        min_type                     = min_type             ,
                        use_thrust_acc_limits        = use_thrust_acc_limits,
                        use_thrust_acc_smoothing     = True                 ,
                        thrust_acc_min               = thrust_acc_min       ,
                        thrust_acc_max               = thrust_acc_max       ,
                        k_heaviside                  = k_idx                ,
                        include_jacobian             = True                 ,
                    )
            soln_root = \
                root(
                    root_func                      ,
                    decisionstate_initguess        ,
                    method                  = 'lm' ,
                    tol                     = 1e-11,
                    jac                     = True ,
                )
            if soln_root.success:
                decisionstate_initguess = soln_root.x
                statecostate_o          = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decisionstate_initguess])
                include_scstm           = True
                stm_oo                  = np.identity(8).flatten()
                state_costate_scstm_o   = np.concatenate([statecostate_o, stm_oo])
                time_eval_points        = np.linspace(time_span[0], time_span[1], 201)
                solve_ivp_func = \
                    lambda time, state_costate_scstm: \
                        freebodydynamics__indirect(
                            time                                            ,
                            state_costate_scstm                             ,
                            include_scstm            = include_scstm        ,
                            min_type                 = min_type             ,
                            use_thrust_acc_limits    = use_thrust_acc_limits,
                            use_thrust_acc_smoothing = True                 ,
                            thrust_acc_min           = thrust_acc_min       ,
                            thrust_acc_max           = thrust_acc_max       ,
                            k_heaviside              = k_idx                ,
                        )
                soln_ivp = \
                    solve_ivp(
                        solve_ivp_func                 ,
                        time_span                      ,
                        state_costate_scstm_o          ,
                        t_eval       = time_eval_points,
                        dense_output = True            , 
                        method       = 'RK45'          ,
                        rtol         = 1e-12           ,
                        atol         = 1e-12           ,
                    )
                results_k_idx[k_idx] = soln_ivp
                error_mag = np.linalg.norm(soln_root.fun)
                if idx==0:
                    tqdm.write(f"       {'Step':>5s} {'k':>14s} {'Error-Mag':>14s}")
                tqdm.write(f"     {idx+1:>3d}/{len(k_idxinitguess_to_idxfinsoln):>3d} {k_idx:>14.6e} {error_mag:>14.6e}")

            else:
                print(f"Convergence Failed for k={k_idx:>14.6e}. Stopping.")
                break

        # Final solution: no thrust or thrust-acc smoothing
        root_func = \
            lambda decisionstate: \
                tpbvp_objective_and_jacobian(
                    decisionstate                                       , 
                    time_span                                           ,
                    boundary_condition_pos_vec_o                        ,
                    boundary_condition_vel_vec_o                        ,
                    boundary_condition_pos_vec_f                        ,
                    boundary_condition_vel_vec_f                        ,
                    use_thrust_acc_limits        = use_thrust_acc_limits,
                    use_thrust_acc_smoothing     = False                ,
                    thrust_acc_min               = thrust_acc_min       ,
                    thrust_acc_max               = thrust_acc_max       ,
                    min_type                     = min_type             ,
                    include_jacobian             = True                 ,
                )
        soln_root = \
            root(
                root_func                      ,
                decisionstate_initguess        ,
                method                  = 'lm' ,
                tol                     = 1e-11,
                jac                     = True ,
            )
        print()
        print("Final Solution")
        print("\nRoot-Solve Results")
        print(soln_root)
        decisionstate_initguess     = soln_root.x
        state_costate_o             = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decisionstate_initguess])
        optimal_control_objective_o = np.float64(0.0)
        state_costate_scstm_o       = np.hstack([state_costate_o, mass_o, optimal_control_objective_o])
        time_eval_points            = np.linspace(time_span[0], time_span[1], 401)
        solve_ivp_func = \
            lambda time, state_costate_scstm: \
                freebodydynamics__indirect(
                    time                                            ,
                    state_costate_scstm                             ,
                    include_scstm            = False                ,
                    min_type                 = min_type             ,
                    use_thrust_acc_limits    = use_thrust_acc_limits,
                    use_thrust_acc_smoothing = False                ,
                    thrust_acc_min           = thrust_acc_min       ,
                    thrust_acc_max           = thrust_acc_max       ,
                    k_heaviside              = np.float64(0.0)      ,
                    post_process             = True                 ,
                )
        soln_ivp = \
            solve_ivp(
                solve_ivp_func                          ,
                time_span                               ,
                state_costate_scstm_o                   ,
                t_eval                = time_eval_points,
                dense_output          = True            , 
                method                = 'RK45'          ,
                rtol                  = 1e-12           ,
                atol                  = 1e-12           ,
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
        print(f"           {'Pos-X':>14s} {'Pos-Y':>14s} {'Vel-X':>14s} {'Vel-Y':>14s}")
        print(f"           {    'm':>14s} {    'm':>14s} {  'm/s':>14s} {  'm/s':>14s}")
        print(f"  Target : {boundary_condition_pos_vec_f[0]:>14.6e} {boundary_condition_pos_vec_f[1]:>14.6e} {boundary_condition_vel_vec_f[0]:>14.6e} {boundary_condition_vel_vec_f[1]:>14.6e}")
        print(f"  Approx : {    state_f_approx_finalsoln[0]:>14.6e} {    state_f_approx_finalsoln[1]:>14.6e} {    state_f_approx_finalsoln[2]:>14.6e} {    state_f_approx_finalsoln[3]:>14.6e}")
        print(f"  Error  : {  error_approx_finalsoln_vec[0]:>14.6e} {  error_approx_finalsoln_vec[1]:>14.3e} {  error_approx_finalsoln_vec[2]:>14.6e} {  error_approx_finalsoln_vec[3]:>14.6e}")
        print(f"  Actual : {           state_f_finalsoln[0]:>14.6e} {           state_f_finalsoln[1]:>14.6e} {           state_f_finalsoln[2]:>14.6e} {           state_f_finalsoln[3]:>14.6e}")
        print(f"  Error  : {         error_finalsoln_vec[0]:>14.6e} {         error_finalsoln_vec[1]:>14.3e} {         error_finalsoln_vec[2]:>14.6e} {         error_finalsoln_vec[3]:>14.6e}")

    else: # assume energy
        # Solve for the optimal min-energy trajectory
        # Thrust- or Thrust-Acc-Steepness Continuation Process

        # Loop initialization
        results_k_idx = {}

        # Loop through k's
        k_idxinitguess_to_idxfinsoln = np.logspace( np.log(k_idxinitguess), np.log(k_idxfinsoln), k_idxdivs )
        for idx, k_idx in tqdm(enumerate(k_idxinitguess_to_idxfinsoln), desc="Processing", leave=False, total=len(k_idxinitguess_to_idxfinsoln)):
        
            root_func = \
                lambda decisionstate: \
                    tpbvp_objective_and_jacobian(
                        decisionstate                                       , 
                        time_span                                           ,
                        boundary_condition_pos_vec_o                        ,
                        boundary_condition_vel_vec_o                        ,
                        boundary_condition_pos_vec_f                        ,
                        boundary_condition_vel_vec_f                        ,
                        min_type                     = min_type             ,
                        include_jacobian             = True                 ,
                        use_thrust_acc_limits        = use_thrust_acc_limits,
                        use_thrust_acc_smoothing     = True                 ,
                        thrust_acc_min               = thrust_acc_min       ,
                        thrust_acc_max               = thrust_acc_max       ,
                        k_heaviside                  = k_idx                ,
                    )
            soln_root = \
                root(
                    root_func                                   ,
                    decisionstate_initguess                     ,
                    method                  = 'lm'              ,
                    tol                     = 1e-11             ,
                    jac                     = True              ,
                    options                 = {'maxiter': 10000},
                )
            if soln_root.success or True:
                decisionstate_initguess     = soln_root.x
                state_costate_o             = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decisionstate_initguess])
                optimal_control_objective_o = np.float64(0.0)
                state_costate_scstm_o       = np.hstack([state_costate_o, mass_o, optimal_control_objective_o])
                time_eval_points            = np.linspace(time_span[0], time_span[1], 201)
                solve_ivp_func = \
                    lambda time, state_costate_scstm: \
                        freebodydynamics__indirect(
                            time                                            ,
                            state_costate_scstm                             ,
                            min_type                 = min_type             ,
                            use_thrust_acc_limits    = use_thrust_acc_limits,
                            use_thrust_acc_smoothing = True                 ,
                            thrust_acc_min           = thrust_acc_min       ,
                            thrust_acc_max           = thrust_acc_max       ,
                            k_heaviside              = k_idx                ,
                            post_process             = True                 ,
                        )
                soln_ivp = \
                    solve_ivp(
                        solve_ivp_func                          ,
                        time_span                               ,
                        state_costate_scstm_o                   ,
                        t_eval                = time_eval_points,
                        dense_output          = True            , 
                        method                = 'RK45'          ,
                        rtol                  = 1e-12           ,
                        atol                  = 1e-12           ,
                    )

                results_k_idx[k_idx] = soln_ivp
                error_mag = np.linalg.norm(soln_root.fun)
                if idx==0:
                    tqdm.write(f"       {'Step':>5s} {'k':>14s} {'Error Mag':>14s}")
                tqdm.write(f"     {idx+1:>3d}/{len(k_idxinitguess_to_idxfinsoln):>3d} {k_idx:>14.6e} {error_mag:>14.6e}")

        # Final solution: no thrust or thrust-acc smoothing
        print()
        print("Final Solution")
        print("\nRoot-Solve Results")
        print(soln_root)
        results_finalsoln = soln_ivp
        state_f_finalsoln = results_finalsoln.y[0:4, -1]

        # Check final state error
        error_finalsoln_vec = state_f_finalsoln - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])
        print("\nState Error Check")
        print(f"           {'Pos-X':>14s} {'Pos-Y':>14s} {'Vel-X':>14s} {'Vel-Y':>14s}")
        print(f"           {    'm':>14s} {    'm':>14s} {  'm/s':>14s} {  'm/s':>14s}")
        print(f"  Target : {boundary_condition_pos_vec_f[0]:>14.6e} {boundary_condition_pos_vec_f[1]:>14.6e} {boundary_condition_vel_vec_f[0]:>14.6e} {boundary_condition_vel_vec_f[1]:>14.6e}")
        print(f"  Actual : {           state_f_finalsoln[0]:>14.6e} {           state_f_finalsoln[1]:>14.6e} {           state_f_finalsoln[2]:>14.6e} {           state_f_finalsoln[3]:>14.6e}")
        print(f"  Error  : {         error_finalsoln_vec[0]:>14.6e} {         error_finalsoln_vec[1]:>14.3e} {         error_finalsoln_vec[2]:>14.6e} {         error_finalsoln_vec[3]:>14.6e}")

    # Plot the results
    plot_final_results(
        results_finalsoln                                   ,
        boundary_condition_pos_vec_o                        ,
        boundary_condition_vel_vec_o                        ,
        boundary_condition_pos_vec_f                        ,
        boundary_condition_vel_vec_f                        ,
        min_type                     = min_type             ,
        use_thrust_acc_limits        = use_thrust_acc_limits,
        thrust_acc_min               = thrust_acc_min       ,
        thrust_acc_max               = thrust_acc_max       ,
        k_steepness                  = k_idx                ,
    )

    # End
    print()

# Plot final results
def plot_final_results(
        results_finsoln                                               ,
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
        k_steepness                  : np.float64 = np.float64(0.0e+0),
    ):
    """
    Calculates and plots all relevant results for the final trajectory solution.
    """
    
    # Unpack state and costate histories
    time_t                                     = results_finsoln.t
    state_t                                    = results_finsoln.y
    pos_x_t, pos_y_t, vel_x_t, vel_y_t         = state_t[0:4]
    copos_x_t, copos_y_t, covel_x_t, covel_y_t = state_t[4:8]
    mass_t                                     = state_t[-2]
    opt_ctrl_obj_t                             = state_t[-1]
    
    # Recalculate the thrust profile to match the dynamics function
    if min_type == 'fuel':
        epsilon          = np.float64(1.0e-6)
        covel_mag_t      = np.sqrt(covel_x_t**2 + covel_y_t**2 + epsilon**2)
        switching_func_t = covel_mag_t - 1.0
        thrust_acc_mag_t = np.zeros_like(switching_func_t)
        for idx, switching_func_t_value in enumerate(switching_func_t):
            if switching_func_t_value > 0:
                thrust_acc_mag_t[idx] = thrust_acc_max
            elif switching_func_t_value < 0:
                thrust_acc_mag_t[idx] = thrust_acc_min
            else:
                thrust_acc_mag_t[idx] = thrust_acc_min
        thrust_acc_dir_t = np.array([ -covel_x_t/covel_mag_t, -covel_y_t/covel_mag_t ])
        thrust_acc_vec_t = thrust_acc_mag_t * thrust_acc_dir_t
    else: # assumes energy
        if use_thrust_limits:
            ...
        elif use_thrust_acc_limits:
            covel_mag_t      = np.sqrt(covel_x_t**2 + covel_y_t**2)
            if k_steepness == np.float64(0.0e+0):
                # No thrust or thrust-acc constraint smoothing
                thrust_acc_mag_t = covel_mag_t
                thrust_acc_mag_t = min( thrust_acc_mag_t, thrust_acc_max ) # max thrust-acc constraint
                thrust_acc_mag_t = max( thrust_acc_mag_t, thrust_acc_min ) # min thrust-acc constraint
            else:
                # Thrust or thrust-acc constraint smoothing
                thrust_acc_mag_t = covel_mag_t
                thrust_acc_mag_t = smin( thrust_acc_mag_t, thrust_acc_max, k_steepness ) # max thrust-acc constraint
                thrust_acc_mag_t = smax( thrust_acc_mag_t, thrust_acc_min, k_steepness ) # min thrust-acc constraint
            thrust_acc_x_dir_t = covel_x_t / covel_mag_t
            thrust_acc_y_dir_t = covel_y_t / covel_mag_t
            thrust_acc_dir_t   = np.vstack([thrust_acc_x_dir_t, thrust_acc_y_dir_t])
            thrust_acc_x_t     = thrust_acc_mag_t * thrust_acc_x_dir_t
            thrust_acc_y_t     = thrust_acc_mag_t * thrust_acc_y_dir_t
            thrust_acc_vec_t   = np.vstack([thrust_acc_x_t, thrust_acc_y_t])
        else:
            # Thrust or thrust-acc constraints
            thrust_acc_vec_t = np.array([ covel_x_t, covel_y_t ])
            thrust_acc_dir_t = thrust_acc_vec_t / np.linalg.norm(thrust_acc_vec_t, axis=0, keepdims=True)
            thrust_acc_mag_t = np.sqrt( thrust_acc_vec_t[0]**2 + thrust_acc_vec_t[1]**2 )
    thrust_mag_t = mass_t * thrust_acc_mag_t
    thrust_dir_t = thrust_acc_dir_t # same
    thrust_vec_t = thrust_mag_t * thrust_dir_t

    # Create trajectory figure
    mplt.style.use('seaborn-v0_8-whitegrid')
    fig = mplt.figure(figsize=(15,8))
    gs = fig.add_gridspec(5, 2, width_ratios=[8, 7])

    # Configure figure
    if min_type == 'fuel':
        title_min_type = "Minimize Fuel"
    elif min_type == 'energy':
        title_min_type = "Minimize Energy"
    else: # assume energy
        title_min_type = "Minimize Energy"
    fig.suptitle(
        f"OPTIMAL TRAJECTORY: {title_min_type}"
        + "\nFree-Body Dynamics"
        + "\nFixed Time-of-Flight | Fixed-Initial-Position, Fixed-Initial-Velocity to Fixed-Final-Position, Fixed-Final-Velocity"
        + "\nThrust Acceleration Max",
        fontsize   = 16              ,
        fontweight = 'normal'        ,
    )

    # 2D position path: pos-x vs. pos-y
    fig_w, fig_h  = fig.get_size_inches()
    ax_height     = 0.75
    ax_width      = ax_height * (fig_h / fig_w)
    square_coords = [0.07, 0.1, ax_width, ax_height]
    ax1           = fig.add_axes(square_coords) # type: ignore
    ax1.plot(                     pos_x_t    ,                      pos_y_t    , color=mcolors.CSS4_COLORS['black'], linewidth=2.0,                                                                                                                                         label='Trajectory' )
    ax1.plot(boundary_condition_pos_vec_o[ 0], boundary_condition_pos_vec_o[ 1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='>', markersize=20, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1.plot(boundary_condition_pos_vec_f[ 0], boundary_condition_pos_vec_f[ 1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='s', markersize=16, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1.plot(                     pos_x_t[ 0],                      pos_y_t[ 0], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='>', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='Start'      )
    ax1.plot(                     pos_x_t[-1],                      pos_y_t[-1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='s', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='End'        )
    def _plot_thrust_on_position_space(
            ax                    : maxes.Axes                     ,
            pos_x_t               : np.ndarray                     ,
            pos_y_t               : np.ndarray                     ,
            thrust_acc_vec_t      : np.ndarray                     ,
            thrust_acc_mag_t      : np.ndarray                     ,
            use_thrust_acc_limits : bool       = True              ,
            thrust_acc_min        : np.float64 = np.float64(0.0e+0),
            thrust_acc_max        : np.float64 = np.float64(1.0e+1),
            use_thrust_limits     : bool       = False             ,
            thrust_min            : np.float64 = np.float64(0.0e+0),
            thrust_max            : np.float64 = np.float64(1.0e+1),
        ):

        min_pos = min(min(pos_x_t), min(pos_y_t))
        max_pos = max(max(pos_x_t), max(pos_y_t))
        if use_thrust_limits:
            plot_thrust_max  = max(thrust_mag_t)
            thrust_vec_scale = 0.2 * (max_pos - min_pos) / plot_thrust_max
            end_x            = pos_x_t + thrust_vec_t[0] * thrust_vec_scale
            end_y            = pos_y_t + thrust_vec_t[1] * thrust_vec_scale
        else: # assume min-fuel or min-energy is using thrust-acc and possibley thrust-acc limits:
            plot_thrust_acc_max  = max(thrust_acc_mag_t)
            thrust_acc_vec_scale = 0.2 * (max_pos - min_pos) / plot_thrust_acc_max
            end_x                = pos_x_t + thrust_acc_vec_t[0] * thrust_acc_vec_scale
            end_y                = pos_y_t + thrust_acc_vec_t[1] * thrust_acc_vec_scale

        # Find contiguous segments where thrust is active
        is_thrust_on = thrust_acc_mag_t > 1.0e-9

        # Find the start and end indices of each 'True' block
        padded = np.concatenate(([False], is_thrust_on, [False]))
        diffs  = np.diff(padded.astype(int))
        starts = np.where(diffs == +1)[0]
        stops  = np.where(diffs == -1)[0]

        # Loop through each segment and draw a separate polygon
        for idx, (start_idx, stop_idx) in enumerate(zip(starts, stops)):

            # Slice the data arrays to get just the points for this segment
            segment_pos_x = pos_x_t[start_idx:stop_idx]
            segment_pos_y = pos_y_t[start_idx:stop_idx]
            segment_end_x = end_x  [start_idx:stop_idx]
            segment_end_y = end_y  [start_idx:stop_idx]

            # Construct the polygon for this segment
            poly_x = np.concatenate([segment_pos_x, segment_end_x[::-1]])
            poly_y = np.concatenate([segment_pos_y, segment_end_y[::-1]])

            # Draw the polygon for the current segment
            ax.fill(
                poly_x,
                poly_y, 
                facecolor = mcolors.CSS4_COLORS['red'],
                alpha     = 0.5,
                edgecolor = 'none',
            )

        # Draw some thrust or thrust-acc vectors
        for idx in np.linspace(0,len(pos_x_t)-1,20).astype(int):
            ax.plot(
                [pos_x_t[idx], end_x[idx]]            ,
                [pos_y_t[idx], end_y[idx]]            ,
                color     = mcolors.CSS4_COLORS['red'],
                linewidth = 2.0                       ,
            )
    _plot_thrust_on_position_space(
        ax                    = ax1                  ,
        pos_x_t               = pos_x_t              ,
        pos_y_t               = pos_y_t              ,
        thrust_acc_vec_t      = thrust_acc_vec_t     ,
        thrust_acc_mag_t      = thrust_acc_mag_t     ,
        use_thrust_acc_limits = use_thrust_acc_limits,
        thrust_acc_min        = thrust_acc_min       ,
        thrust_acc_max        = thrust_acc_max       ,
    )
    ax1.set_xlabel('Position X [m]')
    ax1.set_ylabel('Position Y [m]')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend(loc='upper left')

    # 2D velocity path: vel-x vs. vel-y
    ax1_vel = fig.add_axes(square_coords) # type: ignore
    ax1_vel.set_visible(False)
    ax1_vel.plot(                     vel_x_t    ,                      vel_y_t    , color=mcolors.CSS4_COLORS['black'], linewidth=2.0,                                                                                                                                          label='Trajectory' )
    ax1_vel.plot(boundary_condition_vel_vec_o[ 0], boundary_condition_vel_vec_o[ 1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='>', markersize=20, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1_vel.plot(boundary_condition_vel_vec_f[ 0], boundary_condition_vel_vec_f[ 1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='s', markersize=16, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1_vel.plot(                     vel_x_t[ 0],                      vel_y_t[ 0], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='>', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='Start'      )
    ax1_vel.plot(                     vel_x_t[-1],                      vel_y_t[-1], color=mcolors.CSS4_COLORS['black'], linewidth=1.0, marker='s', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='End'        )
    def _plot_thrust_on_velocity_space(
            ax                    : maxes.Axes                     ,
            vel_x_t               : np.ndarray                     ,
            vel_y_t               : np.ndarray                     ,
            thrust_acc_vec_t      : np.ndarray                     ,
            thrust_acc_mag_t      : np.ndarray                     ,
            use_thrust_acc_limits : bool       = True              ,
            thrust_acc_min        : np.float64 = np.float64(0.0e+0),
            thrust_acc_max        : np.float64 = np.float64(1.0e+1),
            use_thrust_limits     : bool       = False             ,
            thrust_min            : np.float64 = np.float64(0.0e+0),
            thrust_max            : np.float64 = np.float64(1.0e+1),
        ):

        min_vel = min(min(vel_x_t), min(vel_y_t))
        max_vel = max(max(vel_x_t), max(vel_y_t))
        if use_thrust_limits:
            plot_thrust_max  = max(thrust_mag_t)
            thrust_vec_scale = 1.0 * (max_vel - min_vel) / plot_thrust_max
            end_x            = vel_x_t + thrust_vec_t[0] * thrust_vec_scale
            end_y            = vel_y_t + thrust_vec_t[1] * thrust_vec_scale
        else: # assume min-fuel or min-energy is using thrust-acc and possibley thrust-acc limits:
            plot_thrust_acc_max  = max(thrust_acc_mag_t)
            thrust_acc_vec_scale = 1.0 * (max_vel - min_vel) / plot_thrust_acc_max
            end_x                = vel_x_t + thrust_acc_vec_t[0] * thrust_acc_vec_scale
            end_y                = vel_y_t + thrust_acc_vec_t[1] * thrust_acc_vec_scale

        # Find contiguous segments where thrust is active
        is_thrust_on = thrust_acc_mag_t > 1.0e-9

        # Find the start and end indices of each 'True' block
        padded = np.concatenate(([False], is_thrust_on, [False]))
        diffs  = np.diff(padded.astype(int))
        starts = np.where(diffs == +1)[0]
        stops  = np.where(diffs == -1)[0]

        # Loop through each segment and draw a separate polygon
        for idx, (start_idx, stop_idx) in enumerate(zip(starts, stops)):

            # Slice the data arrays to get just the points for this segment
            segment_vel_x = vel_x_t[start_idx:stop_idx]
            segment_vel_y = vel_y_t[start_idx:stop_idx]
            segment_end_x = end_x  [start_idx:stop_idx]
            segment_end_y = end_y  [start_idx:stop_idx]

            # Construct the polygon for this segment
            poly_x = np.concatenate([segment_vel_x, segment_end_x[::-1]])
            poly_y = np.concatenate([segment_vel_y, segment_end_y[::-1]])

            # Draw the polygon for the current segment
            ax.fill(
                poly_x,
                poly_y, 
                facecolor = mcolors.CSS4_COLORS['red'],
                alpha     = 0.5,
                edgecolor = 'none',
            )

        # Draw some thrust or thrust-acc vectors
        for idx in np.linspace(0,len(vel_x_t)-1,20).astype(int):
            ax.plot(
                [vel_x_t[idx], end_x[idx]]            ,
                [vel_y_t[idx], end_y[idx]]            ,
                color     = mcolors.CSS4_COLORS['red'],
                linewidth = 2.0                       ,
            )
    _plot_thrust_on_velocity_space(
        ax                    = ax1_vel              ,
        vel_x_t               = vel_x_t              ,
        vel_y_t               = vel_y_t              ,
        thrust_acc_vec_t      = thrust_acc_vec_t     ,
        thrust_acc_mag_t      = thrust_acc_mag_t     ,
        use_thrust_acc_limits = use_thrust_acc_limits,
        thrust_acc_min        = thrust_acc_min       ,
        thrust_acc_max        = thrust_acc_max       ,
    )
    ax1_vel.set_xlabel("Velocity X [m/s]", labelpad=2)
    ax1_vel.set_ylabel("Velocity Y [m/s]", labelpad=10)
    ax1_vel.grid(True)
    ax1_vel.axis('equal')
    ax1_vel.legend(loc='upper left')
    
    # Create a button to swap pos and vel plots
    ax_button = fig.add_axes([0.015, 0.02, 0.04, 0.04]) # [left, bottom, width, height] # type: ignore
    button = Button(ax_button, "Swap", color=mcolors.CSS4_COLORS['darkgrey'], hovercolor='0.975')
    def _swap_plots(event):
        """
        Toggles visibility of the position and velocity plots.
        """
        # Switch visibility of the position and velocity axes
        is_pos_visible = ax1.get_visible()
        if ax1.get_visible():
            ax1.set_visible(False)
            ax1_vel.set_visible(True)
        else:
            ax1.set_visible(True)
            ax1_vel.set_visible(False)

        # Redraw the canvas to show the changes
        fig.canvas.draw_idle()
    button.on_clicked(_swap_plots)
    fig._button = button # type: ignore

    # Optimal Control Objective vs. Time
    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(time_t, opt_ctrl_obj_t, color=mcolors.CSS4_COLORS['black'], linewidth=2.0, label='Mass')
    ax2.set_xticklabels([])
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    if min_type == 'fuel':
        obj_label_unit = '[m/s]'
    elif min_type == 'energy':
        obj_label_unit = '[m$^2$/$s^3$]'
    ax2.set_ylabel(f'Objective\n{obj_label_unit}')

    # Thrust-Acc or Thrust Profile
    ax3 = fig.add_subplot(gs[1,1])
    if min_type=='fuel':
        ax3.axhline(y=float(thrust_acc_min), color=mcolors.CSS4_COLORS['black'], linestyle=':', linewidth=2.0, label=f'Thrust Acc Min')
        ax3.axhline(y=float(thrust_acc_max), color=mcolors.CSS4_COLORS['black'], linestyle=':', linewidth=2.0, label=f'Thrust Acc Max')
    if use_thrust_limits:
        ax3.plot(time_t, thrust_mag_t    , color=mcolors.CSS4_COLORS['blue'], linewidth=2.0, label='Thrust Mag'    )
    else: # assume use_thrust_acc_limits
        ax3.plot(time_t, thrust_acc_mag_t, color=mcolors.CSS4_COLORS['red' ], linewidth=2.0, label='Thrust Acc Mag')
    ax3.fill_between(
        time_t,
        thrust_acc_mag_t,
        where     = (thrust_acc_mag_t > 0.0), # type: ignore
        facecolor = mcolors.CSS4_COLORS['red'],
        edgecolor = 'none',
        alpha     = 0.5
    )
    ax3.set_xticklabels([])
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    if use_thrust_limits:
        ax3.set_ylabel('Thrust Mag'     + '\n' + '[kg$\cdot$m/s$^2$]')
    else: # assume use_thrust_acc_limits
        ax3.set_ylabel('Thrust Acc Mag' + '\n' + '[m/s$^2$]'         )
    if use_thrust_limits:
        plot_thrust_min = 0.0
        plot_thrust_max = max(thrust_mag_t)
        ax3.set_ylim(
            plot_thrust_min - (plot_thrust_max - plot_thrust_min) * 0.1,
            plot_thrust_max + (plot_thrust_max - plot_thrust_min) * 0.1,
        )
    else: # assume use_thrust_acc_limits
        plot_thrust_acc_min = 0.0
        plot_thrust_acc_max = max(thrust_acc_mag_t)
        ax3.set_ylim(
            plot_thrust_acc_min - (plot_thrust_acc_max - plot_thrust_acc_min) * 0.1,
            plot_thrust_acc_max + (plot_thrust_acc_max - plot_thrust_acc_min) * 0.1,
        )

    # Mass vs. Time
    ax4 = fig.add_subplot(gs[2,1])
    ax4.plot(time_t, mass_t, color=mcolors.CSS4_COLORS['black'], linewidth=2.0, label='Mass')
    ax4.set_xticklabels([])
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    ax4.set_ylabel('Mass\n[kg]')

    # Position vs. Time
    ax5 = fig.add_subplot(gs[3,1])
    ax5.plot(time_t[ 0], boundary_condition_pos_vec_o[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5.plot(time_t[-1], boundary_condition_pos_vec_f[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5.plot(time_t    ,                      pos_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax5.plot(time_t[ 0],                      pos_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5.plot(time_t[-1],                      pos_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax5.plot(time_t[ 0], boundary_condition_pos_vec_o[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5.plot(time_t[-1], boundary_condition_pos_vec_f[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5.plot(time_t    ,                      pos_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax5.plot(time_t[ 0],                      pos_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5.plot(time_t[-1],                      pos_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax5.set_xticklabels([])
    ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    ax5.set_ylabel('Position\n[m]')
    ax5.legend()
    ax5.grid(True)
    min_ylim = min(min(pos_x_t), min(pos_y_t))
    max_ylim = max(max(pos_x_t), max(pos_y_t))
    ax5.set_ylim(
        min_ylim - (max_ylim - min_ylim) * 0.2,
        max_ylim + (max_ylim - min_ylim) * 0.2,
    )

    # Velocity vs. Time
    ax6 = fig.add_subplot(gs[4,1])
    ax6.plot(time_t[ 0], boundary_condition_vel_vec_o[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6.plot(time_t[-1], boundary_condition_vel_vec_f[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6.plot(time_t    ,                      vel_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax6.plot(time_t[ 0],                      vel_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6.plot(time_t[-1],                      vel_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax6.plot(time_t[ 0], boundary_condition_vel_vec_o[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6.plot(time_t[-1], boundary_condition_vel_vec_f[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 16, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6.plot(time_t    ,                      vel_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax6.plot(time_t[ 0],                      vel_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6.plot(time_t[-1],                      vel_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax6.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True, useOffset=False)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Velocity\n[m/s]')
    ax6.legend()
    ax6.grid(True)
    min_ylim = min(min(vel_x_t), min(vel_y_t))
    max_ylim = max(max(vel_x_t), max(vel_y_t))
    ax6.set_ylim(
        min_ylim - (max_ylim - min_ylim) * 0.2,
        max_ylim + (max_ylim - min_ylim) * 0.2,
    )

    # Configure figure
    fig.subplots_adjust(
        left   = 0.05,
        right  = 0.95,
        top    = 0.83,
        hspace = 0.25, # 0.25
        wspace = 0.15, # 0.15
    )
    fig.align_ylabels()
    mplt.show()

# Read input
def read_input():
    print("\nReading Input")

    # Grab command line input
    example_name = sys.argv[1]

     # Check if the filename already has the .json extension
    if example_name.endswith('.json'):
        filename = example_name
    else:
        filename = f"{example_name}.json"

    # Read input
    with open(filename, "r") as file:
       parameters_input = json.load(file)
    print(f"  Successfully read input: {filename}")

    return parameters_input

# Process input: units, valid checks
def process_input(
        parameters_input,
    ):

    print("\nInput Parameters")

    # Create parameters dictionary and print to screen
    max_parameter_length = max([len(parameter) for parameter in parameters_input.keys()])
    max_value_length     = 14
    print(f"  {'Variable':<{max_parameter_length}s} : {'Value':>{2*max_value_length+2}s} {'Unit':<{max_value_length}s}")
    parameters_with_units = {}
    for parameter, value_unit in parameters_input.items():
        if isinstance(value_unit, dict):
            # Handle value_unit as dictionary
            
            # Unpack
            value    = value_unit['value']
            unit_str = value_unit['unit']

            # Handle parameters with unit
            if unit_str not in ("None", None):
                parameters_with_units[parameter] = value * u.Unit(unit_str)
                if np.asarray(value_unit['value']).ndim == 0: # type: ignore
                    value_str = str(f"{value_unit['value']:>{max_value_length}.6e}") # type: ignore
                else:
                    value_str = ', '.join([str(f"{val:>{max_value_length}.6e}") for val in value_unit['value']])

            # Handle parameters with None unit
            else:
                if isinstance(value, (str, bool)):
                    parameters_with_units[parameter] = value
                    value_str = str(value)
                    unit_str  = ""
                elif isinstance(value, (int, float)): 
                    parameters_with_units[parameter] = value * u.one
                    value_str = str(value)
                    unit_str  = ""
            
            # Print row: variable, value, and unit
            print(f"  {parameter:<{max_parameter_length}s} : {value_str:>{2*max_value_length+2}s} {unit_str:<{max_value_length}s}") # type: ignore

        elif isinstance(value_unit, str):
            # Handle value_unit as string

            # Assign parameter value
            parameters_with_units[parameter] = value_unit

            # Print row: variable, value
            print(f"  {parameter:<14s} : {value_unit:>{2*max_value_length+2}s}")

    # Convert to standard units: seconds, meters, kilograms, one
    min_type              = parameters_with_units.get(             'min_type', 'energy'                            )
    time_span             = parameters_with_units.get(            'time_span', [ 0.0e+0, 1.0e+1 ] * u.s            ).to_value(u.s            ) # type: ignore
    pos_vec_o             = parameters_with_units.get(            'pos_vec_o', [ 0.0e+0, 0.0e+0 ] * u.m            ).to_value(u.m            ) # type: ignore
    vel_vec_o             = parameters_with_units.get(            'vel_vec_o', [ 0.0e+0, 0.0e+0 ] * u.m/u.s        ).to_value(u.m/u.s        ) # type: ignore
    pos_vec_f             = parameters_with_units.get(            'pos_vec_f', [ 1.0e+1, 1.0e+1 ] * u.m            ).to_value(u.m            ) # type: ignore
    vel_vec_f             = parameters_with_units.get(            'vel_vec_f', [ 1.0e+0, 1.0e+0 ] * u.m/u.s        ).to_value(u.m/u.s        ) # type: ignore
    mass_o                = parameters_with_units.get(               'mass_o', 1.0e+3             * u.kg           ).to_value(u.kg           ) # type: ignore
    use_thrust_acc_limits = parameters_with_units.get('use_thrust_acc_limits', False                               )
    thrust_acc_min        = parameters_with_units.get(       'thrust_acc_min', 0.0e+0             * u.m/u.s**2     ).to_value(u.m/u.s**2     ) # type: ignore
    thrust_acc_max        = parameters_with_units.get(       'thrust_acc_max', 1.0e+0             * u.m/u.s**2     ).to_value(u.m/u.s**2     ) # type: ignore
    use_thrust_limits     = parameters_with_units.get(    'use_thrust_limits', False                               )
    thrust_min            = parameters_with_units.get(           'thrust_min', 0.0e+0             * u.kg*u.m/u.s**2).to_value(u.kg*u.m/u.s**2) # type: ignore
    thrust_max            = parameters_with_units.get(           'thrust_max', 1.0e+0             * u.kg*u.m/u.s**2).to_value(u.kg*u.m/u.s**2) # type: ignore
    k_idxinitguess        = parameters_with_units.get(       'k_idxinitguess', None                                )
    k_idxfinsoln          = parameters_with_units.get(         'k_idxfinsoln', None                                )
    k_idxdivs             = parameters_with_units.get(            'k_idxdivs', 10                 * u.one          ).to_value(u.one          ) # type: ignore
    init_guess_steps      = parameters_with_units.get(     'init_guess_steps', 3000               * u.one          ).to_value(u.one          ) # type: ignore

    # Create boundary conditions
    boundary_condition_pos_vec_o = pos_vec_o
    boundary_condition_vel_vec_o = vel_vec_o
    boundary_condition_pos_vec_f = pos_vec_f
    boundary_condition_vel_vec_f = vel_vec_f

    # Validate input
    print("\nValidate Input")

    # Enforce types
    k_idxdivs        = int(k_idxdivs)
    init_guess_steps = int(init_guess_steps)

    # Check if both thrust and thrust-acc constraints are set
    print("  Check thrust and thrust-acc constraints")
    if use_thrust_acc_limits and use_thrust_limits:
        use_thrust_acc_limits = True
        use_thrust_limits     = False
        print("\n    Warning: Cannot use both thrust acceleration limits and thrust limits."
              + f"   Choosing use_thrust_acc_limits = {use_thrust_acc_limits} and use_thrust_limits = {use_thrust_limits}.")
    
    # Check if min-type fuel is set but no thrust or thrust-acc constraint
    if (
            min_type=='fuel' 
            and use_thrust_acc_limits is False and use_thrust_limits is False
        ):
        use_thrust_acc_limits = True
        use_thrust_limits     = False
        print("\n    Warning: Min type is fuel, but no thrust or thrust-acc constraint is set."
              + f"   Choosing use_thrust_acc_limits = {use_thrust_acc_limits} and use_thrust_limits = {use_thrust_limits}.")

    # Determine the first k value based on thrust or thrust-acc constraints if not an input
    print("  Compute k-continuation parameters")
    if k_idxinitguess is None:
        if use_thrust_limits:
            k_idxinitguess = np.float64(4.0 / (thrust_max - thrust_min + 1.0e-9))
        elif use_thrust_acc_limits:
            k_idxinitguess = np.float64(4.0 / (thrust_acc_max - thrust_acc_min + 1.0e-9))
        else:
            k_idxinitguess = np.float64(1.0e+0)
        print(f"    Initial k-steepness : {k_idxinitguess:<{max_value_length}.6e}")
    else:
        k_idxinitguess = np.float64(k_idxinitguess) # type: ignore

    # Determine last k value
    if k_idxfinsoln is None:
        k_idxfinsoln = np.float64(10.0 * k_idxinitguess)
        print(f"    Final k-steepness   : {k_idxfinsoln:<{max_value_length}.6e}")
    else:
        k_idxfinsoln = np.float64(k_idxfinsoln) # type: ignore

    # Pack up variable input
    return (
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
    )

# Optimal trajectory input
def optimal_trajectory_input():
    # Read input
    parameters_input = read_input()

    # Process input
    return process_input(parameters_input)

# Main
if __name__ == '__main__':

    # Start optimization
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
    )



