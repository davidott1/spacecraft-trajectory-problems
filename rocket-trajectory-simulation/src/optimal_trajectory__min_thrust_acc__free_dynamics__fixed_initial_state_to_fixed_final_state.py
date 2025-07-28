# Imports
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as mplt
import matplotlib.colors as mcolors
import json
import random
np.random.seed(42)
import astropy.units as u


# Dynamics functions
def freebodydynamics__minfuel__indirect_thrustaccmax(
        time           : float              ,
        state          : np.ndarray         ,
        thrust_acc_min : float      = 1.0e-1, 
        thrust_acc_max : float      = 1.0e+1,
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
    epsilon        = 1.0e-6
    covel_mag      = np.sqrt(covel_x**2 + covel_y**2 + epsilon**2)
    switching_func = covel_mag - 1.0
    if switching_func > 0.0:
        thrust_acc_mag = thrust_acc_max
    elif switching_func < 0.0:
        thrust_acc_mag = thrust_acc_min
    else:
        thrust_acc_mag = thrust_acc_min # undetermined. choose thrust_acc_min.
    thrust_acc_x_dir = -covel_x / covel_mag
    thrust_acc_y_dir = -covel_y / covel_mag
    thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
    thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir

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


def freebodydynamics__minfuel__indirect_thrustaccmax_stm(
        time           : float              ,
        state__stm     : np.ndarray         ,
        thrust_acc_min : float      = 1.0e-1, 
        thrust_acc_max : float      = 1.0e+1,
    ) -> np.ndarray:
    """
    xxx
    """

    # Unpack: full state into state and variational-state-transition matrix
    n_states = 8
    state    = state__stm[:n_states]
    stm      = state__stm[n_states:].reshape((n_states, n_states))

    # Unpack: state into scalar components
    pos_x, pos_y, vel_x, vel_y, copos_x, copos_y, covel_x, covel_y = state

    # Control: thrust acceleration
    #   thrust_acc_vec = -covel_vec / cvel_mag
    epsilon        = 1.0e-6
    covel_mag      = np.sqrt(covel_x**2 + covel_y**2 + epsilon**2)
    switching_func = covel_mag - 1.0
    if switching_func > 0.0:
        thrust_acc_mag = thrust_acc_max
    elif switching_func < 0.0:
        thrust_acc_mag = thrust_acc_min
    else:
        thrust_acc_mag = thrust_acc_min # undetermined. choose thrust_acc_min.
    thrust_acc_x_dir = -covel_x / covel_mag
    thrust_acc_y_dir = -covel_y / covel_mag
    thrust_acc_x     = thrust_acc_mag * thrust_acc_x_dir
    thrust_acc_y     = thrust_acc_mag * thrust_acc_y_dir

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
    
    # Pack up: scalar components into state
    dstate__dtime = \
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
    ddstatedtime__dstate = np.zeros((n_states, n_states))

    # Row 1
    #   d(dpos_x/dtime)/dvel_x
    ddstatedtime__dstate[0,2] = 1.0

    # Row 2
    #   d(dpos_y/dtime)/dvel_y
    ddstatedtime__dstate[1,3] = 1.0

    # Row 3
    #   d(dvel_x/dtime)/dcovel_x, d(dvel_x/dtime)/dcovel_y
    # Row 4
    #   d(dvel_y/dtime)/dcovel_x, d(dvel_y/dtime)/dcovel_y

    # Common terms for derivatives
    dcovel_mag__dcovel_x = covel_x / covel_mag
    dcovel_mag__dcovel_y = covel_y / covel_mag

    k_heaviside = 0.0
    one_mns_tanhsq              = 1.0 - np.tanh(k_heaviside * switching_func)**2
    dheaviside_approx__dcovel_x = 0.5 * k_heaviside * one_mns_tanhsq * dcovel_mag__dcovel_x
    dheaviside_approx__dcovel_y = 0.5 * k_heaviside * one_mns_tanhsq * dcovel_mag__dcovel_y

    dthrust_acc_mag__dcovel_x = (thrust_acc_max - thrust_acc_min) * dheaviside_approx__dcovel_x
    dthrust_acc_mag__dcovel_y = (thrust_acc_max - thrust_acc_min) * dheaviside_approx__dcovel_y

    dthrust_acc_x_dir__dcovel_x = -1.0 / covel_mag - covel_x * (-1.0 / covel_mag**2) * dcovel_mag__dcovel_x
    dthrust_acc_x_dir__dcovel_y =                  - covel_x * (-1.0 / covel_mag**2) * dcovel_mag__dcovel_y
    dthrust_acc_y_dir__dcovel_y = -1.0 / covel_mag - covel_y * (-1.0 / covel_mag**2) * dcovel_mag__dcovel_y
    dthrust_acc_y_dir__dcovel_x =                  - covel_y * (-1.0 / covel_mag**2) * dcovel_mag__dcovel_x

    # d(dvel_x__dtime)/dcovel_x
    ddstatedtime__dstate[2,6] = 0.0 * dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x

    # d(dvel_x__dtime)/dcovel_y
    ddstatedtime__dstate[2,7] = 0.0 * dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y

    # d(dvel_y__dtime)/dcovel_x
    ddstatedtime__dstate[3,6] = 0.0 * dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x

    # d(dvel_y__dtime)/dcovel_y
    ddstatedtime__dstate[3,7] = 0.0 * dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

    # Row 5
    #   dcopos_x__dtime = 0
    # Do nothing. All zeros.

    # Row 6
    #   dcopos_y__dtime = 0
    # Do nothing. All zeros.

    # Row 7
    #   d(dcovel_x_dtime)/dcopos_x
    ddstatedtime__dstate[6,4] = -1.0 

    # Row 8
    #   d(dcovel_y__dttime)/dcopos_y
    ddstatedtime__dstate[7,5] = -1.0

    # Combine: time-derivative of state-transition matrix
    dstm__dtime = np.dot(ddstatedtime__dstate, stm)

    # Pack up: time-derivative of state and stm
    return np.concatenate((dstate__dtime, dstm__dtime.flatten()))


def freebodydynamics__minfuel__indirect_thrustaccmax_heaviside(
        time           : float              ,
        state          : np.ndarray         ,
        thrust_acc_min : float      = 1.0e-1, 
        thrust_acc_max : float      = 1.0e+1,
        k_heaviside    : float      = 1.0   ,
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
    epsilon          = 1.0e-6
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
    dvel_x__dtime   = thrust_acc_x
    dvel_y__dtime   = thrust_acc_y
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


def freebodydynamics__minfuel__indirect_thrustaccmax_heaviside_stm(
        time           : float              ,
        state__stm     : np.ndarray         ,
        thrust_acc_min : float      = 1.0e-1, 
        thrust_acc_max : float      = 1.0e1 ,
        k_heaviside    : float      = 1.0   ,
    ) -> np.ndarray:

    # Unpack: full state into state and variational-state-transition matrix
    n_states = 8
    state    = state__stm[:n_states]
    stm      = state__stm[n_states:].reshape((n_states, n_states))

    # Unpack: state into scalar components
    pos_x, pos_y, vel_x, vel_y, copos_x, copos_y, covel_x, covel_y = state

    # Control: thrust acceleration
    #   thrust_acc_vec = -covel_vec / cvel_mag
    epsilon          = 1.0e-6
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
    dvel_x__dtime   = thrust_acc_x
    dvel_y__dtime   = thrust_acc_y
    dcopos_x__dtime = 0.0
    dcopos_y__dtime = 0.0
    dcovel_x__dtime = -copos_x
    dcovel_y__dtime = -copos_y
    
    dstate__dtime = \
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
    ddstatedtime__dstate = np.zeros((n_states, n_states))

    # Row 1
    #   d(dpos_x/dtime)/dvel_x
    ddstatedtime__dstate[0,2] = 1.0

    # Row 2
    #   d(dpos_y/dtime)/dvel_y
    ddstatedtime__dstate[1,3] = 1.0

    # Row 3
    #   d(dvel_x/dtime)/dcovel_x, d(dvel_x/dtime)/dcovel_y
    # Row 4
    #   d(dvel_y/dtime)/dcovel_x, d(dvel_y/dtime)/dcovel_y

    # Common terms for derivatives
    dcovel_mag__dcovel_x = covel_x / covel_mag
    dcovel_mag__dcovel_y = covel_y / covel_mag

    one_mns_tanhsq              = 1.0 - np.tanh(k_heaviside * switching_func)**2
    dheaviside_approx__dcovel_x = 0.5 * k_heaviside * one_mns_tanhsq * dcovel_mag__dcovel_x
    dheaviside_approx__dcovel_y = 0.5 * k_heaviside * one_mns_tanhsq * dcovel_mag__dcovel_y

    dthrust_acc_mag__dcovel_x = (thrust_acc_max - thrust_acc_min) * dheaviside_approx__dcovel_x
    dthrust_acc_mag__dcovel_y = (thrust_acc_max - thrust_acc_min) * dheaviside_approx__dcovel_y

    dthrust_acc_x_dir__dcovel_x = -1.0 / covel_mag - covel_x * (-1.0 / covel_mag**2) * dcovel_mag__dcovel_x
    dthrust_acc_x_dir__dcovel_y =                  - covel_x * (-1.0 / covel_mag**2) * dcovel_mag__dcovel_y
    dthrust_acc_y_dir__dcovel_y = -1.0 / covel_mag - covel_y * (-1.0 / covel_mag**2) * dcovel_mag__dcovel_y
    dthrust_acc_y_dir__dcovel_x =                  - covel_y * (-1.0 / covel_mag**2) * dcovel_mag__dcovel_x

    # d(dvel_x__dtime)/dcovel_x
    ddstatedtime__dstate[2,6] = dthrust_acc_mag__dcovel_x * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_x

    # d(dvel_x__dtime)/dcovel_y
    ddstatedtime__dstate[2,7] = dthrust_acc_mag__dcovel_y * thrust_acc_x_dir + thrust_acc_mag * dthrust_acc_x_dir__dcovel_y

    # d(dvel_y__dtime)/dcovel_x
    ddstatedtime__dstate[3,6] = dthrust_acc_mag__dcovel_x * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_x

    # d(dvel_y__dtime)/dcovel_y
    ddstatedtime__dstate[3,7] = dthrust_acc_mag__dcovel_y * thrust_acc_y_dir + thrust_acc_mag * dthrust_acc_y_dir__dcovel_y

    # Row 5
    #   dcopos_x__dtime = 0
    # Do nothing. All zeros.

    # Row 6
    #   dcopos_y__dtime = 0
    # Do nothing. All zeros.

    # Row 7
    #   d(dcovel_x_dtime)/dcopos_x
    ddstatedtime__dstate[6,4] = -1.0 

    # Row 8
    #   d(dcovel_y__dttime)/dcopos_y
    ddstatedtime__dstate[7,5] = -1.0

    # Combine: time-derivative of state-transition matrix
    dstm__dtime = np.dot(ddstatedtime__dstate, stm)

    # Pack up: time-derivative of state and stm
    return np.concatenate((dstate__dtime, dstm__dtime.flatten()))


def constantgravitydynamics__minfuel__indirect_thrustaccmax_heaviside(
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
    epsilon          = 1.0e-6
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
    epsilon          = 1.0e-6
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


def objective_function(
        decision_state,
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
        k_heaviside,
    ):
    """
    Objective function for the root-finder that calls the unified dynamics.
    """
    statecostate_o = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decision_state])
    sol = \
        solve_ivp(
            freebodydynamics__minfuel__indirect_thrustaccmax_heaviside,
            time_span,
            statecostate_o,
            dense_output=True, 
            args=(thrust_acc_min, thrust_acc_max, k_heaviside), 
            method='DOP853',
            rtol=1e-12,
            atol=1e-12,
        )
    state_f = sol.sol(time_span[1])[:4]
    return state_f - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])


def objective_and_jacobian(
        decision_state,
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
        k_heaviside,
    ):
    """
    Objective function that also returns the analytical Jacobian.
    """

    # Initial state and stm
    statecostate_o    = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decision_state])
    stm_oo            = np.identity(8).flatten()
    statecostatestm_o = np.hstack([statecostate_o, stm_oo])
    
    # Integrate the augmented system
    soln = \
        solve_ivp(
            freebodydynamics__minfuel__indirect_thrustaccmax_heaviside_stm,
            time_span,
            statecostatestm_o,
            dense_output = True, 
            args         = (thrust_acc_min, thrust_acc_max, k_heaviside), 
            method       = 'RK45',
            rtol         = 1.0e-12,
            atol         = 1.0e-12,
        )
    
    # Extract final state and final STM
    statecostatestm_f = soln.sol(time_span[1])
    statecostate_f    = statecostatestm_f[:8]
    stm_of            = statecostatestm_f[8:].reshape((8,8))
    
    # Calculate the error vector
    error = statecostate_f[:4] - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])
    
    # Extract 4x4 Jacobian from the final STM
    #   jacobian = d(state_final) / d(costate_initial)
    error_jacobian = stm_of[0:4, 4:8]
    
    return error, error_jacobian


def objective_and_jacobian_2(
        decision_state,
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
    ):
    """
    Objective function that also returns the analytical Jacobian.
    """

    # Initial state and stm
    statecostate_o    = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decision_state])
    stm_oo            = np.identity(8).flatten()
    statecostatestm_o = np.hstack([statecostate_o, stm_oo])
    
    # Integrate the augmented system
    soln = \
        solve_ivp(
            freebodydynamics__minfuel__indirect_thrustaccmax_stm,
            time_span,
            statecostatestm_o,
            dense_output = True, 
            args         = (thrust_acc_min, thrust_acc_max), 
            method       = 'RK45',
            rtol         = 1.0e-12,
            atol         = 1.0e-12,
        )
    
    # Extract final state and final STM
    statecostatestm_f = soln.sol(time_span[1])
    statecostate_f    = statecostatestm_f[:8]
    stm_of            = statecostatestm_f[8:].reshape((8,8))
    
    # Calculate the error vector
    error = statecostate_f[:4] - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])
    
    # Extract 4x4 Jacobian from the final STM
    #   jacobian = d(state_final) / d(costate_initial)
    error_jacobian = stm_of[0:4, 4:8]
    
    return error, error_jacobian


def generate_guess(
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
        k_heaviside,
    ):
    """
    Generates a robust initial guess for the co-states: copos_vec, covel_vec
    """
    print("\nHeuristic Initial Guess Process")
    error_mag_min = np.Inf
    for idx in range(1000):
        copos_vec_o        = np.random.uniform(low=-1, high=1, size=2)
        covel_vec_o        = np.random.uniform(low=-1, high=1, size=2)
        decision_state_idx = np.hstack([copos_vec_o, covel_vec_o])
        
        error_idx = \
            objective_function(
                decision_state_idx,
                time_span,
                boundary_condition_pos_vec_o,
                boundary_condition_vel_vec_o,
                boundary_condition_pos_vec_f,
                boundary_condition_vel_vec_f,
                thrust_acc_min,
                thrust_acc_max,
                k_heaviside,
            )

        error_mag_idx = np.linalg.norm(error_idx)
        if error_mag_idx < error_mag_min:
            idx_min            = idx
            error_mag_min      = error_mag_idx
            decision_state_min = decision_state_idx
            if idx==0:
                print(f"                              {'decision_state':>{4*14+3}s}")
                print(f"         {'idx':>5s} {'error_mag':>14s} {'copos_x':>14s} {'copos_y':>14s} {'covel_x':>14s} {'covel_y':>14s}")
            decision_state_min_str = ' '.join(f"{x:>14.6e}" for x in decision_state_idx)
            print(f"         {idx_min:>5d} {error_mag_min:>14.6e} {decision_state_min_str}")

    # Pack up and print solution
    costate_o_guess = decision_state_min
    print(f"MIN: *** {idx_min:>5d} {error_mag_min:>14.6e} {decision_state_min_str} ***")
    return costate_o_guess


def optimal_trajectory_solve(
        min_type,
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
        k_idxinitguess,
        k_idxfinsoln, 
        k_idxdivs,
    ):
    """
    Main solver that implements the two-stage continuation process
    using the unified smoothed dynamics.
    """

    # Generate initial guess for the costates
    decisionstate_initguess = generate_guess(
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
        k_idxinitguess,
    )

    # Select minimization type
    if min_type == "fuel":

        # K-Continuation Process
        print(f"\nK-Continuation Process")
        k_idxinitguess_to_idxfinsoln = np.logspace( np.log(k_idxinitguess), np.log(k_idxfinsoln), k_idxdivs )
        results_k_idx = {}
        for idx, k_idx in enumerate(k_idxinitguess_to_idxfinsoln):
            soln_root = \
                root(
                    objective_and_jacobian,
                    decisionstate_initguess,
                    args   = (time_span, boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, boundary_condition_pos_vec_f, boundary_condition_vel_vec_f, thrust_acc_min, thrust_acc_max, k_idx),
                    method = 'lm',
                    tol    = 1e-7,
                    jac    = True,
                )
            if soln_root.success:
                decisionstate_initguess = soln_root.x
                statecostate_o          = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decisionstate_initguess])
                stm_oo                  = np.identity(8).flatten()
                statecostatestm_o       = np.concatenate([statecostate_o, stm_oo])
                time_eval_points        = np.linspace(time_span[0], time_span[1], 201)
                soln_ivp = \
                    solve_ivp(
                        freebodydynamics__minfuel__indirect_thrustaccmax_heaviside_stm,
                        time_span,
                        statecostatestm_o,
                        t_eval       = time_eval_points,
                        dense_output = True, 
                        args         = (thrust_acc_min, thrust_acc_max, k_idx), 
                        method       = 'RK45', # DOP853 | RK45
                        rtol         = 1e-12,
                        atol         = 1e-12,
                    )
                results_k_idx[k_idx] = soln_ivp
                error_mag = np.linalg.norm(soln_root.fun)
                if idx==0:
                    print(f"       {'Step':>5s} {'k':>14s} {'Error Mag':>14s}")
                print(f"     {idx+1:>3d}/{len(k_idxinitguess_to_idxfinsoln):>3d} {k_idx:>14.6e} {error_mag:>14.6e}")

            else:
                print(f"Convergence Failed for k={k_idx:>14.6e}. Stopping.")
                break

        # Final solution: no heaviside approximation
        soln_root = \
            root(
                objective_and_jacobian_2,
                decisionstate_initguess,
                args   = (time_span, boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, boundary_condition_pos_vec_f, boundary_condition_vel_vec_f, thrust_acc_min, thrust_acc_max),
                method = 'lm',
                tol    = 1e-12,
                jac    = True,
            )
        decisionstate_initguess = soln_root.x
        statecostate_o          = np.hstack([boundary_condition_pos_vec_o, boundary_condition_vel_vec_o, decisionstate_initguess])
        stm_oo                  = np.identity(8).flatten()
        statecostatestm_o       = np.concatenate([statecostate_o, stm_oo])
        time_eval_points        = np.linspace(time_span[0], time_span[1], 401)
        soln_ivp = \
            solve_ivp(
                freebodydynamics__minfuel__indirect_thrustaccmax_stm,
                time_span,
                statecostatestm_o,
                t_eval       = time_eval_points,
                dense_output = True, 
                args         = (thrust_acc_min, thrust_acc_max), 
                method       = 'RK45', # DOP853 | RK45
                rtol         = 1e-12,
                atol         = 1e-12,
            )
        results_finalsoln = soln_ivp
        state_f_finalsoln = results_finalsoln.y[0:4, -1]

        # Final solution: approx and true
        results_approx_finalsoln = results_k_idx[k_idxinitguess_to_idxfinsoln[-1]]
        state_f_approx_finalsoln = results_approx_finalsoln.y[0:4, -1]
        k_finsoln                = k_idxinitguess_to_idxfinsoln[-1]

        # Check final state error
        error_approx_finalsoln_vec = state_f_approx_finalsoln - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])
        error_finalsoln_vec        = state_f_finalsoln        - np.hstack([boundary_condition_pos_vec_f, boundary_condition_vel_vec_f])
        print("\nFinal State Error Check")
        print(f"           {'pos_x':>14s} {'pos_y':>14s} {'vel_x':>14s} {'vel_y':>14s}")
        print(f"           {    'm':>14s} {    'm':>14s} {  'm/s':>14s} {  'm/s':>14s}")
        print(f"  Target : {boundary_condition_pos_vec_f[0]:>14.6e} {boundary_condition_pos_vec_f[1]:>14.6e} {boundary_condition_vel_vec_f[0]:>14.6e} {boundary_condition_vel_vec_f[1]:>14.6e}")
        print(f"  Approx : {  state_f_approx_finalsoln[0]:>14.6e} {  state_f_approx_finalsoln[1]:>14.6e} {  state_f_approx_finalsoln[2]:>14.6e} {  state_f_approx_finalsoln[3]:>14.6e}")
        print(f"  Error  : {error_approx_finalsoln_vec[0]:>14.6e} {error_approx_finalsoln_vec[1]:>14.3e} {error_approx_finalsoln_vec[2]:>14.6e} {error_approx_finalsoln_vec[3]:>14.6e}")
        print(f"  Actual : {         state_f_finalsoln[0]:>14.6e} {         state_f_finalsoln[1]:>14.6e} {         state_f_finalsoln[2]:>14.6e} {         state_f_finalsoln[3]:>14.6e}")
        print(f"  Error  : {       error_finalsoln_vec[0]:>14.6e} {       error_finalsoln_vec[1]:>14.3e} {       error_finalsoln_vec[2]:>14.6e} {       error_finalsoln_vec[3]:>14.6e}")
    
        # Plot the results
        plot_final_results(
            results_finalsoln,
            boundary_condition_pos_vec_o,
            boundary_condition_vel_vec_o,
            boundary_condition_pos_vec_f,
            boundary_condition_vel_vec_f,
            thrust_acc_min,
            thrust_acc_max,
        )

    # End
    print()


def plot_final_results(
        results_finsoln,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min, 
        thrust_acc_max, 
    ):
    """
    Calculates and plots all relevant results for the final trajectory solution.
    """
    # Unpack state and costate histories
    time_t                                     = results_finsoln.t
    states_t                                   = results_finsoln.y
    pos_x_t, pos_y_t, vel_x_t, vel_y_t         = states_t[0:4]
    copos_x_t, copos_y_t, covel_x_t, covel_y_t = states_t[4:8]
    
    # Recalculate the thrust profile to match the dynamics function
    epsilon          = 1.0e-6
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

    # Create trajectory figure
    mplt.style.use('seaborn-v0_8-whitegrid')
    fig = mplt.figure(figsize=(15, 8))
    gs  = fig.add_gridspec(3, 2)

    # Configure figure
    fig.suptitle(
        "OPTIMAL TRAJECTORY: Minimize Fuel"
        + "\nFree-Body Dynamics"
        + "\nFixed Time-of-Flight | Fixed-Initial-Position, Fixed-Initial-Velocity to Fixed-Final-Position, Fixed-Final-Velocity"
        + "\nThrust Acceleration Max",
        fontsize=16,
        fontweight='normal',
    )

    # 2D Trajectory Path
    ax1 = fig.add_subplot(gs[0:3, 0])
    
    ax1.plot(                     pos_x_t    ,                      pos_y_t    , color=mcolors.CSS4_COLORS['black'],                                                                                                                                          label='Trajectory' )
    ax1.plot(boundary_condition_pos_vec_o[ 0], boundary_condition_pos_vec_o[ 1], color=mcolors.CSS4_COLORS['black'], marker='>', markersize=20, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1.plot(boundary_condition_pos_vec_f[ 0], boundary_condition_pos_vec_f[ 1], color=mcolors.CSS4_COLORS['black'], marker='s', markersize=20, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1.plot(                     pos_x_t[ 0],                      pos_y_t[ 0], color=mcolors.CSS4_COLORS['black'], marker='>', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='Start'      )
    ax1.plot(                     pos_x_t[-1],                      pos_y_t[-1], color=mcolors.CSS4_COLORS['black'], marker='s', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='End'        )

    min_pos = min(min(pos_x_t), min(pos_y_t))
    max_pos = max(max(pos_x_t), max(pos_y_t))
    thrust_acc_vec_scale = 0.1 * (max_pos - min_pos) / thrust_acc_max
    for idx in range(len(time_t)):
        start_x = pos_x_t[idx]
        start_y = pos_y_t[idx]
        end_x   = pos_x_t[idx] + thrust_acc_vec_t[0][idx] * thrust_acc_vec_scale
        end_y   = pos_y_t[idx] + thrust_acc_vec_t[1][idx] * thrust_acc_vec_scale
        if idx == 0:
            ax1.plot([start_x, end_x], [start_y, end_y], color=mcolors.CSS4_COLORS['red'], linewidth=5.0, alpha=0.2, label='Thrust Acc Vec' )
        else:
            ax1.plot([start_x, end_x], [start_y, end_y], color=mcolors.CSS4_COLORS['red'], linewidth=5.0, alpha=0.2 )
    ax1.set_xlabel('Position X [m]')
    ax1.set_ylabel('Position Y [m]')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()

    # Thrust Profile
    ax2 = fig.add_subplot(gs[0,1])
    ax2.axhline(y=thrust_acc_max, color=mcolors.CSS4_COLORS['black'], linestyle=':', linewidth=2.0, label=f'Thrust Acc Min')
    ax2.axhline(y=thrust_acc_min, color=mcolors.CSS4_COLORS['black'], linestyle=':' , linewidth=2.0, label=f'Thrust Acc Max')
    ax2.plot(time_t, thrust_acc_mag_t, color=mcolors.CSS4_COLORS['red'], linewidth=2.0, label='Thrust Acc Mag')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Thrust Acc Mag [m/s2]')
    ax2.grid(True)
    ax2.set_ylim(
        thrust_acc_min - (thrust_acc_max - thrust_acc_min) * 0.1,
        thrust_acc_max + (thrust_acc_max - thrust_acc_min) * 0.1,
    )

    # Position vs. Time
    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(time_t[ 0], boundary_condition_pos_vec_o[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax3.plot(time_t[-1], boundary_condition_pos_vec_f[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax3.plot(time_t    ,                      pos_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax3.plot(time_t[ 0],                      pos_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax3.plot(time_t[-1],                      pos_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax3.plot(time_t[ 0], boundary_condition_pos_vec_o[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax3.plot(time_t[-1], boundary_condition_pos_vec_f[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax3.plot(time_t    ,                      pos_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax3.plot(time_t[ 0],                      pos_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax3.plot(time_t[-1],                      pos_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax3.set_ylabel('Position [m]')
    ax3.legend()
    ax3.grid(True)
    min_ylim = min(min(pos_x_t), min(pos_y_t))
    max_ylim = max(max(pos_x_t), max(pos_y_t))
    ax3.set_ylim(
        min_ylim - (max_ylim - min_ylim) * 0.2,
        max_ylim + (max_ylim - min_ylim) * 0.2,
    )

    # Velocity vs. Time
    ax4 = fig.add_subplot(gs[2,1])
    ax4.plot(time_t[ 0], boundary_condition_vel_vec_o[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax4.plot(time_t[-1], boundary_condition_vel_vec_f[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax4.plot(time_t    ,                      vel_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax4.plot(time_t[ 0],                      vel_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax4.plot(time_t[-1],                      vel_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax4.plot(time_t[ 0], boundary_condition_vel_vec_o[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax4.plot(time_t[-1], boundary_condition_vel_vec_f[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax4.plot(time_t    ,                      vel_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax4.plot(time_t[ 0],                      vel_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax4.plot(time_t[-1],                      vel_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Velocity [m/s]')
    ax4.legend()
    ax4.grid(True)
    min_ylim = min(min(vel_x_t), min(vel_y_t))
    max_ylim = max(max(vel_x_t), max(vel_y_t))
    ax4.set_ylim(
        min_ylim - (max_ylim - min_ylim) * 0.2,
        max_ylim + (max_ylim - min_ylim) * 0.2,
    )

    # Configure figure
    mplt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0]) # type: ignore
    mplt.show()


def read_input():
    with open("optimal_trajectory_input.json", "r") as file:
       parameters_input = json.load(file)
    return parameters_input


def proceess_input(
        parameters_input,
    ):

    # Create parameters dictionary and print to screen
    print("\nInput Parameters")
    print(f"  {'Variable':<14s} : {'Value':>{2*14+2}s} {'Unit':<14s}")
    parameters_with_units = {}
    for variable, value_unit in parameters_input.items():
        if isinstance(value_unit, dict):

            # Unpack
            value    = value_unit['value']
            unit_str = value_unit['unit']

            # Handle parameters with unit
            if unit_str not in ("None", None):
                parameters_with_units[variable] = value * u.Unit(unit_str)
                if np.asarray(value_unit['value']).ndim == 0: # type: ignore
                    value_str = str(f"{value_unit['value']:>14.6e}") # type: ignore
                else:
                    value_str = ', '.join([str(f"{val:>14.6e}") for val in value_unit['value']])

            # Handle parameters without unit
            else:
                parameters_with_units[variable] = value * u.one
                unit_str = ""
            
            # Print row: variable, value, and unit
            print(f"  {variable:<14s} : {value_str:>{2*14+2}s} {unit_str:<14s}") # type: ignore

        elif isinstance(value_unit, str):
            parameters_with_units[variable] = value_unit

            # Print row: variable, value
            print(f"  {variable:<14s} : {value_unit:>{2*14+2}s}")

    # Convert to standard units: seconds, meters, kilograms, one
    min_type       = parameters_with_units[      'min_type']
    time_span      = parameters_with_units[     'time_span'].to_value(u.s       ) # type: ignore
    pos_vec_o      = parameters_with_units[     'pos_vec_o'].to_value(u.m       ) # type: ignore
    vel_vec_o      = parameters_with_units[     'vel_vec_o'].to_value(u.m/u.s   ) # type: ignore
    pos_vec_f      = parameters_with_units[     'pos_vec_f'].to_value(u.m       ) # type: ignore
    vel_vec_f      = parameters_with_units[     'vel_vec_f'].to_value(u.m/u.s   ) # type: ignore
    thrust_acc_min = parameters_with_units['thrust_acc_min'].to_value(u.m/u.s**2) # type: ignore
    thrust_acc_max = parameters_with_units['thrust_acc_max'].to_value(u.m/u.s**2) # type: ignore
    k_idxinitguess = parameters_with_units['k_idxinitguess'].to_value(u.one     ) # type: ignore
    k_idxfinsoln   = parameters_with_units[  'k_idxfinsoln'].to_value(u.one     ) # type: ignore
    k_idxdivs      = parameters_with_units[     'k_idxdivs'].to_value(u.one     ) # type: ignore

    # Enforce types
    k_idxdivs = int(k_idxdivs)

    # Create boundary conditions
    boundary_condition_pos_vec_o = pos_vec_o
    boundary_condition_vel_vec_o = vel_vec_o
    boundary_condition_pos_vec_f = pos_vec_f
    boundary_condition_vel_vec_f = vel_vec_f

    # Pack up variable input
    return (
        min_type,
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
        k_idxinitguess,
        k_idxfinsoln, 
        k_idxdivs,
    )


def optimal_trajectory_input():
    # Read input
    parameters_input = read_input()

    # Process input
    return proceess_input( parameters_input )


if __name__ == '__main__':

    # Optimal trajectory input
    (
        min_type,
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
        k_idxinitguess,
        k_idxfinsoln, 
        k_idxdivs,
    ) = \
        optimal_trajectory_input()

    # Optimal trajectory solve
    optimal_trajectory_solve(
        min_type,
        time_span,
        boundary_condition_pos_vec_o,
        boundary_condition_vel_vec_o,
        boundary_condition_pos_vec_f,
        boundary_condition_vel_vec_f,
        thrust_acc_min,
        thrust_acc_max,
        k_idxinitguess,
        k_idxfinsoln, 
        k_idxdivs,
    )


# {
#     "_commenet_0"    : "fuel eneergy || { 'value': [  0.0, 400.0 ], 'unit': 's'     } free || { 'value': [  0.0,   0.0 ], 'unit': 'm'     } free || { 'value': [ -0.3,  -0.5 ], 'unit': 'm/s'   } free || { 'value':          8.0e-3, 'unit': 'm/s^2' }",
#     "min_type"       : "fuel", 
#     "time_span"      : { "value": [  0.0, 400.0 ], "unit": "s"     },
#     "pos_vec_o"      : { "value": [  0.0,   0.0 ], "unit": "m"     },
#     "vel_vec_o"      : { "value": [ -0.3,  -0.5 ], "unit": "m/s"   },
#     "pos_vec_f"      : { "value": [ 10.0,   5.0 ], "unit": "m"     },
#     "vel_vec_f"      : { "value": [  0.3,  -0.5 ], "unit": "m/s"   },
#     "thrust_acc_min" : { "value":          0.0e+0, "unit": "m/s^2" },
#     "thrust_acc_max" : { "value":          8.0e-3, "unit": "m/s^2" },
#     "k_idxinitguess" : { "value":          1.0e-1, "unit": "None"  },
#     "k_idxfinsoln"   : { "value":          1.0e+2, "unit": "None"  },
#     "k_idxdivs"      : { "value":             100, "unit": "None"  }
# }