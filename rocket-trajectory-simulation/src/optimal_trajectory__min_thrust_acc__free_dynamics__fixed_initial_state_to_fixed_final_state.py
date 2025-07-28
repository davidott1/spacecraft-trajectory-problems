# Imports
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as mplt
import matplotlib.colors as mcolors
import random
np.random.seed(42)
import astropy.units as u



# Dynamical functions
def freebodydynamics__minfuel__indirect_thrustacc_heaviside(
        time           : float              ,
        state          : np.ndarray         ,
        thrust_acc_min : float      = 1.0e-1, 
        thrust_acc_max : float      = 1.0e1 ,
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
    switching_func   = covel_mag - 1
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


def freebodydynamics__minfuel__indirect_thrustacc_heaviside_stm(
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
    switching_func   = covel_mag - 1
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


def objective_function(
        decision_state,
        time_span,
        boundary_condition_state_o,
        boundary_condition_state_f,
        thrust_acc_min,
        thrust_acc_max,
        k_heaviside,
    ):
    """
    Objective function for the root-finder that calls the unified dynamics.
    """
    statecostate_o = np.hstack((boundary_condition_state_o, decision_state))
    sol = \
        solve_ivp(
            freebodydynamics__minfuel__indirect_thrustacc_heaviside,
            time_span,
            statecostate_o,
            dense_output=True, 
            args=(thrust_acc_min, thrust_acc_max, k_heaviside), 
            method='DOP853',
            rtol=1e-12,
            atol=1e-12,
        )
    state_f = sol.sol(time_span[1])[:4]
    return state_f - boundary_condition_state_f


def objective_and_jacobian(initial_costates, time_span, initial_states, final_states, T_min, T_max, k_heaviside):
    """
    Objective function that also returns the analytical Jacobian.
    """

    # Initial augmented state
    z0      = np.concatenate((initial_states, initial_costates))
    phi0    = np.identity(8).flatten()
    z_aug_0 = np.concatenate([z0, phi0])
    
    # Integrate the augmented system
    sol = solve_ivp(freebodydynamics__minfuel__indirect_thrustacc_heaviside_stm, time_span, z_aug_0, dense_output=True, 
                    args=(T_min, T_max, k_heaviside), 
                    method='RK45', # DOP853 | RK45
                    rtol=1e-12,
                    atol=1e-12)
    
    # Extract final state and final STM
    final_aug_state = sol.sol(time_span[1])
    final_z         = final_aug_state[:8]
    final_phi       = final_aug_state[8:].reshape((8, 8))
    
    # Calculate the error vector
    error = final_z[:4] - final_states
    
    # Extract the required 4x4 Jacobian from the final STM
    #   jacobian = d(x_final) / d(l_initial)
    jacobian = final_phi[0:4, 4:8]
    
    return error, jacobian


def generate_guess(time_span, boundary_condition_state_o, boundary_condition_state_f, thrust_min, thrust_max, k_heaviside):
    """
    Generates a robust initial guess for the co-states: copos_vec, covel_vec
    """
    print("\nHeuristic Initial Guess Process")
    error_mag_min = np.Inf
    for idx in range(1000):
        copos_vec_o        = np.random.uniform(low=-1, high=1, size=2)
        covel_vec_o        = np.random.uniform(low=-1, high=1, size=2)
        decision_state_idx = np.hstack([copos_vec_o, covel_vec_o])
        
        error_idx = objective_function(
            decision_state_idx,
            time_span,
            boundary_condition_state_o,
            boundary_condition_state_f,
            thrust_min,
            thrust_max,
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


def solve_trajectory():
    """
    Main solver that implements the two-stage continuation process
    using the unified smoothed dynamics.
    """
    # Problem input
    units_time, units_distance, units_mass, units_force = "s", "m", "kg", "N" # "s", "m", "kg", "N" | ???
    time_o, time_f                          = 0, 400
    pos_vec_o                               = [  0.0,  0.0 ]
    vel_vec_o                               = [ -0.3, -0.5 ]
    pos_vec_f                               = [ 10.0,  5.0 ]
    vel_vec_f                               = [  0.3, -0.5 ] 
    thrust_acc_min, thrust_acc_max          = 0.0e+0, 8.0e-3
    k_idxinitguess, k_idxfinsoln, k_idxdivs = 1.0e-1, 1.0e+2, 10

    # Process input
    time_span                  = np.hstack([time_o, time_f])
    state_o                    = np.hstack([pos_vec_o, vel_vec_o])
    state_f                    = np.hstack([pos_vec_f, vel_vec_f])
    boundary_condition_state_o = state_o
    boundary_condition_state_f = state_f

    # Generate initial guess for the costates
    decisionstate_initguess = generate_guess(time_span, boundary_condition_state_o, boundary_condition_state_f, thrust_acc_min, thrust_acc_max, k_idxinitguess)

    # K-Continuation Process
    print(f"\nK-Continuation Process")
    k_idxinitguess_to_idxfinsoln = np.logspace(np.log(k_idxinitguess), np.log(k_idxfinsoln), k_idxdivs)
    results_k_idx = {}
    for idx, k_idx in enumerate(k_idxinitguess_to_idxfinsoln):
        soln_root = \
            root(
                objective_and_jacobian,
                decisionstate_initguess,
                args=(time_span, state_o, state_f, thrust_acc_min, thrust_acc_max, k_idx),
                method='lm',
                tol=1e-7,
                jac=True,
            )
        if soln_root.success:
            decisionstate_initguess = soln_root.x
            statecostate_o          = np.hstack((pos_vec_o, vel_vec_o, decisionstate_initguess))
            stm_oo                  = np.identity(8).flatten()
            statecostatestm_o       = np.concatenate([statecostate_o, stm_oo])
            time_eval_points        = np.linspace(time_span[0], time_span[1], 201)
            soln_ivp = \
                solve_ivp(
                    freebodydynamics__minfuel__indirect_thrustacc_heaviside_stm,
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
    
    # Final solution
    results_finsoln = results_k_idx[k_idxinitguess_to_idxfinsoln[-1]]
    state_f_finsoln = results_finsoln.y[0:4, -1]
    k_finsoln       = k_idxinitguess_to_idxfinsoln[-1]

    # Check final state error
    error_finsoln_vec = state_f_finsoln - boundary_condition_state_f
    print("\nFinal State Error Check")
    print(f"           {'pos_x':>14s} {'pos_y':>14s} {'vel_x':>14s} {'vel_y':>14s}")
    print(f"           {    'm':>14s} {    'm':>14s} {  'm/s':>14s} {  'm/s':>14s}")
    print(f"  Target : {boundary_condition_state_f[0]:>14.6e} {boundary_condition_state_f[1]:>14.6e} {boundary_condition_state_f[2]:>14.6e} {boundary_condition_state_f[3]:>14.6e}")
    print(f"  Actual : {           state_f_finsoln[0]:>14.6e} {           state_f_finsoln[1]:>14.6e} {           state_f_finsoln[2]:>14.6e} {           state_f_finsoln[3]:>14.6e}")
    print(f"  Error  : {         error_finsoln_vec[0]:>14.6e} {         error_finsoln_vec[1]:>14.3e} {         error_finsoln_vec[2]:>14.6e} {         error_finsoln_vec[3]:>14.6e}")
   
    # Plot the results
    plot_final_results(
        results_finsoln, 
        boundary_condition_state_o,
        boundary_condition_state_f,
        thrust_acc_min,
        thrust_acc_max,
        k_finsoln,
    )


def plot_final_results(
        results_finsoln,
        boundary_condition_state_o, 
        boundary_condition_state_f, 
        thrust_acc_min, 
        thrust_acc_max, 
        k,
    ):
    """
    Calculates and plots all relevant results for the final trajectory solution.
    """
    # Unpack state and costate histories
    time_t, states_t                           = results_finsoln.t, results_finsoln.y
    pos_x_t, pos_y_t, vel_x_t, vel_y_t         = states_t[0:4]
    copos_x_t, copos_y_t, covel_x_t, covel_y_t = states_t[4:8]
    
    # Recalculate the thrust profile to match the dynamics function
    epsilon          = 1.0e-6
    covel_mag_t      = np.sqrt(covel_x_t**2 + covel_y_t**2 + epsilon**2)
    switching_func_t = covel_mag_t - 1
    heaviside_approx = 0.5 + 0.5 * np.tanh(k * switching_func_t)
    thrust_acc_mag_t = thrust_acc_min + (thrust_acc_max - thrust_acc_min) * heaviside_approx
    thrust_acc_dir_t = np.array([ -covel_x_t/covel_mag_t, -covel_y_t/covel_mag_t ])
    thrust_acc_vec_t = thrust_acc_mag_t * thrust_acc_dir_t

    # Create trajectory figure
    mplt.style.use('seaborn-v0_8-whitegrid')
    fig = mplt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 2)

    # Configure figure
    fig.suptitle(
        "OPTIMAL TRAJECTORY: Minimize Fuel"
        + "\nFree Dynamics"
        + "\nFixed Time-of-Flight | Fixed-Initial-Position, Fixed-Initial-Velocity to Fixed-Final-Position, Fixed-Final-Velocity"
        + "\nThrust Acceleration Max",
        fontsize=16,
        fontweight='normal',
    )

    # Panel 1: 2D Trajectory Path
    ax1 = fig.add_subplot(gs[0:3, 0])
    
    ax1.plot(                   pos_x_t    ,                    pos_y_t    , color=mcolors.CSS4_COLORS['black'],                                                                                                                                           label='Trajectory' )
    ax1.plot(boundary_condition_state_o[ 0], boundary_condition_state_o[ 1], color=mcolors.CSS4_COLORS['black'], marker='>', markersize=20, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1.plot(boundary_condition_state_f[ 0], boundary_condition_state_f[ 1], color=mcolors.CSS4_COLORS['black'], marker='s', markersize=20, markerfacecolor=mcolors.CSS4_COLORS['white'], markeredgecolor=mcolors.CSS4_COLORS['black']                                       )
    ax1.plot(                   pos_x_t[ 0],                    pos_y_t[ 0], color=mcolors.CSS4_COLORS['black'], marker='>', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='Start'      )
    ax1.plot(                   pos_x_t[-1],                    pos_y_t[-1], color=mcolors.CSS4_COLORS['black'], marker='s', markersize=10, markerfacecolor=mcolors.CSS4_COLORS['black'], markeredgecolor=mcolors.CSS4_COLORS['black'], linestyle='None', label='End'        )

    min_pos = min(min(pos_x_t), min(pos_y_t))
    max_pos = max(max(pos_x_t), max(pos_y_t))
    thrust_acc_vec_scale = 0.1 * (max_pos - min_pos) / thrust_acc_max
    for idx in range(len(time_t)):
        start_x = pos_x_t[idx]
        start_y = pos_y_t[idx]
        end_x   = pos_x_t[idx] + thrust_acc_vec_t[0][idx] * thrust_acc_vec_scale
        end_y   = pos_y_t[idx] + thrust_acc_vec_t[1][idx] * thrust_acc_vec_scale
        if idx == 0:
            ax1.plot([start_x, end_x], [start_y, end_y], color=mcolors.CSS4_COLORS['red'], linewidth=5.0, alpha=0.5, label='Thrust Acc Vec' )
        else:
            ax1.plot([start_x, end_x], [start_y, end_y], color=mcolors.CSS4_COLORS['red'], linewidth=5.0, alpha=0.5 )
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
    ax3.plot(time_t[ 0], boundary_condition_state_o[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax3.plot(time_t[-1], boundary_condition_state_f[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax3.plot(time_t    ,                    pos_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax3.plot(time_t[ 0],                    pos_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax3.plot(time_t[-1],                    pos_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax3.plot(time_t[ 0], boundary_condition_state_o[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax3.plot(time_t[-1], boundary_condition_state_f[ 1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax3.plot(time_t    ,                    pos_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax3.plot(time_t[ 0],                    pos_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax3.plot(time_t[-1],                    pos_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
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
    ax4.plot(time_t[ 0], boundary_condition_state_o[ 2], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax4.plot(time_t[-1], boundary_condition_state_f[ 2], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax4.plot(time_t    ,                    vel_x_t    , color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, label='X' )
    ax4.plot(time_t[ 0],                    vel_x_t[ 0], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax4.plot(time_t[-1],                    vel_x_t[-1], color=mcolors.CSS4_COLORS[  'indianred'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS[  'indianred'], markeredgecolor=mcolors.CSS4_COLORS[  'indianred'], linestyle='None' )
    ax4.plot(time_t[ 0], boundary_condition_state_o[ 3], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax4.plot(time_t[-1], boundary_condition_state_f[ 3], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 20, markerfacecolor=mcolors.CSS4_COLORS[      'white'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax4.plot(time_t    ,                    vel_y_t    , color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, label='Y' )
    ax4.plot(time_t[ 0],                    vel_y_t[ 0], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='>', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
    ax4.plot(time_t[-1],                    vel_y_t[-1], color=mcolors.CSS4_COLORS['forestgreen'], linewidth=2.0, marker='s', markersize= 10, markerfacecolor=mcolors.CSS4_COLORS['forestgreen'], markeredgecolor=mcolors.CSS4_COLORS['forestgreen'], linestyle='None' )
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


if __name__ == '__main__':
    solve_trajectory()

