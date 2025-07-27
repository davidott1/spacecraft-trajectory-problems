# Imports
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
np.random.seed(42)


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
    statecostate_o = np.concatenate((boundary_condition_state_o, decision_state))
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


def generate_heuristic_guess(time_o, time_f, posvel_o, posvel_f, thrust_min, thrust_max, k_heaviside):
    """
    Generates a robust, physics-based initial guess for the costates.
    """
    time_span    = [time_o, time_f]
    pos_o, vel_o = posvel_o[:2], posvel_o[2:]
    pos_f, vel_f = posvel_f[:2], posvel_f[2:]
    # l_r0_guess = (pos_o - pos_f) / (t_f - t_o)
    # l_v0_guess =  vel_o - vel_f
    # costates_o_guess = np.concatenate([l_r0_guess, l_v0_guess])
    # l_r0_guess = np.array([1.0e-9, 1.0e-9])
    # l_v0_guess = np.array([1.0e-9, 1.0e-9])
    # costates_o_guess = np.concatenate([l_r0_guess, l_v0_guess])

    error_mag_min = np.Inf
    for idx in range(1000):
        l_r0_guess     = np.random.uniform(low=-1, high=1, size=2)
        l_v0_guess     = np.random.uniform(low=-1, high=1, size=2)
        costates_o_idx = np.concatenate([l_r0_guess, l_v0_guess])

        error_idx = objective_function(costates_o_idx, time_span, posvel_o, posvel_f, thrust_min, thrust_max, k_heaviside)

        error_mag_idx = np.linalg.norm(error_idx)
        if error_mag_idx < error_mag_min:
            error_mag_min  = error_mag_idx
            costates_o_min = costates_o_idx
            print(f"idx error_mag costate_o : {idx}, {error_mag_min}, {costates_o_min}")
    costates_o_guess = costates_o_min

    
    print("--- Generated Heuristic Initial Guess ---")
    print(f"  l_r(0) guess: [{costates_o_guess[0]:.2f}, {costates_o_guess[1]:.2f}]")
    print(f"  l_v(0) guess: [{costates_o_guess[2]:.2f}, {costates_o_guess[3]:.2f}]")
    return costates_o_min


def solve_trajectory():
    """
    Main solver that implements the two-stage continuation process
    using the unified smoothed dynamics.
    """
    # Problem input
    time_o, time_f         = 0, 50
    posvel_o               = [ 0,  0, 0, 0]
    posvel_f               = [10, 15, 5, 0]
    thrust_min, thrust_max = 0.0e-0, 0.5e+0
    k_idx0, k_idxn, k_divs = 1.0e-1, 1.0e+2, 50

    # Process input
    time_span = [time_o, time_f]
    posvel_o = np.array(posvel_o)
    posvel_f = np.array(posvel_f)
    
    # Generate an initial guess for the costates
    decisionstate = generate_heuristic_guess(time_o, time_f, posvel_o, posvel_f, thrust_min, thrust_max, k_idx0)

    # --- K-Continuation Process ---
    print(f"\n--- K-Continuation Process ---")
    k_idx0ton = np.logspace(np.log(k_idx0), np.log(k_idxn), k_divs)
    k_results = {}
    for idx, k_idx in enumerate(k_idx0ton):
        print(f"Step {idx+1}/{len(k_idx0ton)}: Solving for k = {k_idx:.2f}...", end="")
        sol_root = \
            root(
                objective_and_jacobian,
                decisionstate,
                args=(time_span, posvel_o, posvel_f, thrust_min, thrust_max, k_idx),
                method='lm',
                tol=1e-7,
                jac=True,
            )
        if sol_root.success:
            print("Success!")
            decisionstate     = sol_root.x
            statecostate_o    = np.concatenate((posvel_o, decisionstate))
            stm_oo            = np.identity(8).flatten()
            statecostatestm_o = np.concatenate([statecostate_o, stm_oo])
            time_eval_points  = np.linspace(time_span[0], time_span[1], 201)
            sol_optimal = \
                solve_ivp(
                    freebodydynamics__minfuel__indirect_thrustacc_heaviside_stm,
                    time_span,
                    statecostatestm_o,
                    t_eval=time_eval_points,
                    dense_output=True, 
                    args=(thrust_min, thrust_max, k_idx), 
                    method='RK45', # DOP853 | RK45
                    rtol=1e-12,
                    atol=1e-12,
                )
            k_results[k_idx] = sol_optimal
        else:
            print(f"Convergence Failed for k={k_idx:.2f}. Stopping.")
            break
    
    # The final solution is the one with the highest k
    final_solution = k_results[k_idx0ton[-1]]
    final_k        = k_idx0ton[-1]

    # Check final state error
    final_state_vector = final_solution.y[0:4, -1]
    error_vector = final_state_vector - posvel_f
    print("\n--- Final State Error Check ---")
    print(f"               {'rx':>12s} {'ry':>12s} {'vx':>12s} {'vy':>12s}")
    print(f"Target:      {posvel_f[0]:12.6f} {posvel_f[1]:12.6f} {posvel_f[2]:12.6f} {posvel_f[3]:12.6f}")
    print(f"Actual:      {final_state_vector[0]:12.6f} {final_state_vector[1]:12.6f} {final_state_vector[2]:12.6f} {final_state_vector[3]:12.6f}")
    print(f"Error:       {error_vector[0]:12.3e} {error_vector[1]:12.3e} {error_vector[2]:12.3e} {error_vector[3]:12.3e}")

    # Plot the results
    plot_final_results(final_solution, posvel_o, posvel_f, thrust_min, thrust_max, final_k)


def plot_final_results(solution, initial_states, final_states, thrust_acc_min, thrust_acc_max, k):
    """
    Calculates and plots all relevant results for the final trajectory solution.
    """
    # Unpack state and costate histories
    time_t, states_t = solution.t, solution.y
    pos_x_t, pos_y_t, vel_x_t, vel_y_t = states_t[0], states_t[1], states_t[2], states_t[3]
    covel_x_t, covel_y_t = states_t[6], states_t[7]
    
    # Recalculate the thrust profile to match the dynamics function
    epsilon          = 1e-6
    covel_t_mag      = np.sqrt(covel_x_t**2 + covel_y_t**2 + epsilon**2)
    switching_func   = covel_t_mag - 1
    heaviside_approx = 0.5 + 0.5 * np.tanh(k * switching_func)
    thrust_acc_mag_t = thrust_acc_min + (thrust_acc_max - thrust_acc_min) * heaviside_approx
    thrust_acc_dir_t = np.array([ -covel_x_t/covel_t_mag, -covel_y_t/covel_t_mag ])
    thrust_acc_vec_t = thrust_acc_mag_t * thrust_acc_dir_t

    # --- Create the plots ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 2)

    # Panel 1: 2D Trajectory Path
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(pos_x_t, pos_y_t, label='Trajectory')
    ax1.scatter(initial_states[0], initial_states[1], color='g', s=100, zorder=5, label='Start')
    ax1.scatter(  final_states[0],   final_states[1], color='r', s=100, zorder=5, label='End'  )
    ax1.set_title('2D Trajectory Path'); ax1.set_xlabel('$r_x$'); ax1.set_ylabel('$r_y$')
    ax1.legend(); ax1.grid(True); ax1.axis('equal')

    # Define a scale for the tangent lines (like quiver's scale)
    line_scale = 1.0e0

    # 5. Plot each tangent line using a loop
    for idx in range(len(time_t)):
        # Define the start and end points of the line segment
        start_x = pos_x_t[idx]
        start_y = pos_y_t[idx]
        end_x   = pos_x_t[idx] + thrust_acc_vec_t[0][idx] * line_scale
        end_y   = pos_y_t[idx] + thrust_acc_vec_t[1][idx] * line_scale

        # Add a label only to the first line to avoid a messy legend
        if idx == 0:
            ax1.plot([start_x, end_x], [start_y, end_y], color='red', linewidth=3.5, label='Velocity $\\vec{v}(t)$' )
        else:
            ax1.plot([start_x, end_x], [start_y, end_y], color='red', linewidth=3.5 )

    # Panel 2: Thrust Profile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_t, thrust_acc_mag_t, 'r-', label='Thrust Acc Mag $||\Gamma||$')
    ax2.axhline(y=thrust_acc_max, color='k', linestyle='--', label=f'thrust_acc_max = {thrust_acc_max}')
    ax2.axhline(y=thrust_acc_min, color='k', linestyle=':' , label=f'thrust_acc_min = {thrust_acc_min}')
    ax2.set_title('Thrust Acc Profile')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Thrust Acc Mag')
    ax2.legend(); ax2.grid(True)
    ax2.set_ylim(0, thrust_acc_max * 1.1)

    # Panel 3: Position vs. Time
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time_t, pos_x_t, label='$r_x(t)$')
    ax3.plot(time_t, pos_y_t, label='$r_y(t)$', linestyle='--')
    ax3.set_title('Position vs. Time'); ax3.set_ylabel('Position'); ax3.legend(); ax3.grid(True)

    # Panel 4: Velocity vs. Time
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time_t, vel_x_t, label='$v_x(t)$')
    ax4.plot(time_t, vel_y_t, label='$v_y(t)$', linestyle='--')
    ax4.set_title('Velocity vs. Time'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Velocity')
    ax4.legend(); ax4.grid(True)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96]) # type: ignore


if __name__ == '__main__':
    solve_trajectory()
    plt.show()