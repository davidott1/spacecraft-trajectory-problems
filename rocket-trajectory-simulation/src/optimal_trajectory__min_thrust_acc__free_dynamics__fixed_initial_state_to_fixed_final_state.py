import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

np.random.seed(42)


def freedynamics__minfuel__heaviside(t, z, T_min=1.0e-1, T_max=1.0e1, k_heaviside=1.0):

    rx, ry, vx, vy, l_rx, l_ry, l_vx, l_vy = z

    epsilon          = 1.0e-10
    lambda_v_norm    = np.sqrt(l_vx**2 + l_vy**2 + epsilon**2)
    switching_func   = lambda_v_norm - 1
    heaviside_approx = 0.5 + 0.5 * np.tanh(k_heaviside * switching_func)
    thrust_mag       = T_min + (T_max - T_min) * heaviside_approx
    thrust_acc_x     = - (l_vx / lambda_v_norm) * thrust_mag
    thrust_acc_y     = - (l_vy / lambda_v_norm) * thrust_mag

    drx_dt, dry_dt     = vx, vy
    dvx_dt, dvy_dt     = thrust_acc_x, thrust_acc_y
    dl_rx_dt, dl_ry_dt = 0, 0
    dl_vx_dt, dl_vy_dt = -l_rx, -l_ry
    
    return [drx_dt, dry_dt, dvx_dt, dvy_dt, dl_rx_dt, dl_ry_dt, dl_vx_dt, dl_vy_dt]

def freedynamics__minfuel__heaviside__stm(t, z, T_min=1.0e-1, T_max=1.0e1, k_heaviside=1.0):
    # State vector: [rx, ry, vx, vy, l_rx, l_ry, l_vx, l_vy]
    # Augment with STM: Phi_11, Phi_12, ..., Phi_88 (8x8 = 64 elements)

    n_states = 8 # Number of primary states
    primary_state = z[:n_states]
    stm_flat      = z[n_states:]

    rx, ry, vx, vy, l_rx, l_ry, l_vx, l_vy = primary_state

    epsilon          = 1.0e-10
    lambda_v_norm    = np.sqrt(l_vx**2 + l_vy**2 + epsilon**2)
    switching_func   = lambda_v_norm - 1
    heaviside_approx = 0.5 + 0.5 * np.tanh(k_heaviside * switching_func)
    thrust_mag       = T_min + (T_max - T_min) * heaviside_approx

    # --- Primary State Dynamics ---
    drx_dt, dry_dt     = vx, vy
    thrust_acc_x       = - (l_vx / lambda_v_norm) * thrust_mag
    thrust_acc_y       = - (l_vy / lambda_v_norm) * thrust_mag
    dvx_dt, dvy_dt     = thrust_acc_x, thrust_acc_y
    dl_rx_dt, dl_ry_dt = 0, 0
    dl_vx_dt, dl_vy_dt = -l_rx, -l_ry
    
    primary_state_dot = [drx_dt, dry_dt, dvx_dt, dvy_dt, 
                         dl_rx_dt, dl_ry_dt, dl_vx_dt, dl_vy_dt]

    # --- Jacobian Matrix A(t) = d(f)/d(x) ---
    A = np.zeros((n_states, n_states))

    # Fill in partial derivatives (non-zero terms only)
    # wrt rx, ry, vx, vy, l_rx, l_ry, l_vx, l_vy

    # Row 1 (drx_dt = vx)
    A[0, 2] = 1.0 # d(drx_dt)/dvx

    # Row 2 (dry_dt = vy)
    A[1, 3] = 1.0 # d(dry_dt)/dvy

    # Row 3 (dvx_dt = thrust_acc_x) and Row 4 (dvy_dt = thrust_acc_y)
    # These are the most complex.
    # We need partials of thrust_acc_x and thrust_acc_y with respect to l_vx and l_vy

    # Common terms for derivatives
    inv_lambda_v_norm     = 1.0 / lambda_v_norm
    d_lambda_v_norm_dl_vx = l_vx * inv_lambda_v_norm
    d_lambda_v_norm_dl_vy = l_vy * inv_lambda_v_norm

    sech_squared_val = (1.0 - np.tanh(k_heaviside * switching_func)**2)
    d_heaviside_approx_dl_vx = 0.5 * k_heaviside * sech_squared_val * d_lambda_v_norm_dl_vx
    d_heaviside_approx_dl_vy = 0.5 * k_heaviside * sech_squared_val * d_lambda_v_norm_dl_vy

    d_thrust_mag_dl_vx = (T_max - T_min) * d_heaviside_approx_dl_vx
    d_thrust_mag_dl_vy = (T_max - T_min) * d_heaviside_approx_dl_vy

    # d(thrust_acc_x)/dl_vx
    term1 = - (1.0 / lambda_v_norm) * thrust_mag
    term2 = - l_vx * (-1.0 / lambda_v_norm**2) * d_lambda_v_norm_dl_vx * thrust_mag
    term3 = - (l_vx / lambda_v_norm) * d_thrust_mag_dl_vx
    A[2, 6] = term1 + term2 + term3 # d(dvx_dt)/dl_vx

    # d(thrust_acc_x)/dl_vy
    term1 = - l_vx * (-1.0 / lambda_v_norm**2) * d_lambda_v_norm_dl_vy * thrust_mag
    term2 = - (l_vx / lambda_v_norm) * d_thrust_mag_dl_vy
    A[2, 7] = term1 + term2 # d(dvx_dt)/dl_vy

    # d(thrust_acc_y)/dl_vx
    term1 = - l_vy * (-1.0 / lambda_v_norm**2) * d_lambda_v_norm_dl_vx * thrust_mag
    term2 = - (l_vy / lambda_v_norm) * d_thrust_mag_dl_vx
    A[3, 6] = term1 + term2 # d(dvy_dt)/dl_vx

    # d(thrust_acc_y)/dl_vy
    term1 = - (1.0 / lambda_v_norm) * thrust_mag
    term2 = - l_vy * (-1.0 / lambda_v_norm**2) * d_lambda_v_norm_dl_vy * thrust_mag
    term3 = - (l_vy / lambda_v_norm) * d_thrust_mag_dl_vy
    A[3, 7] = term1 + term2 + term3 # d(dvy_dt)/dl_vy

    # Row 5 (dl_rx_dt = 0) - all zeros

    # Row 6 (dl_ry_dt = 0) - all zeros

    # Row 7 (dl_vx_dt = -l_rx)
    A[6, 4] = -1.0 # d(dl_vx_dt)/dl_rx

    # Row 8 (dl_vy_dt = -l_ry)
    A[7, 5] = -1.0 # d(dl_vy_dt)/dl_ry

    # --- STM Dynamics ---
    # Reshape the flat STM vector into a matrix
    Phi = stm_flat.reshape((n_states, n_states))

    # Compute dPhi_dt = A * Phi
    dPhi_dt = np.dot(A, Phi)

    # Flatten dPhi_dt back into a vector
    dPhi_dt_flat = dPhi_dt.flatten()

    # Concatenate and return
    return np.concatenate((primary_state_dot, dPhi_dt_flat))

def objective_function(initial_costates, t_span, initial_states, final_states, T_min, T_max, k_heaviside):
    """
    Objective function for the root-finder that calls the unified dynamics.
    """
    z0 = np.concatenate((initial_states, initial_costates))
    sol = solve_ivp(freedynamics__minfuel__heaviside, t_span, z0, dense_output=True, 
                    args=(T_min, T_max, k_heaviside), 
                    method='DOP853', rtol=1e-12, atol=1e-12)
    final_states_from_integration = sol.sol(t_span[1])[:4]
    error = final_states_from_integration - final_states
    return error

def objective_and_jacobian(initial_costates, t_span, initial_states, final_states, T_min, T_max, k_heaviside):
    """
    Objective function that also returns the analytical Jacobian.
    """

    # Initial augmented state
    z0      = np.concatenate((initial_states, initial_costates))
    phi0    = np.identity(8).flatten()
    z_aug_0 = np.concatenate([z0, phi0])
    
    # Integrate the augmented system
    sol = solve_ivp(freedynamics__minfuel__heaviside__stm, t_span, z_aug_0, dense_output=True, 
                    args=(T_min, T_max, k_heaviside), 
                    method='RK45', # DOP853 | RK45
                    rtol=1e-12, atol=1e-12)
    
    # Extract final state and final STM
    final_aug_state = sol.sol(t_span[1])
    final_z         = final_aug_state[:8]
    final_phi       = final_aug_state[8:].reshape((8, 8))
    
    # Calculate the error vector
    error = final_z[:4] - final_states
    
    # Extract the required 4x4 Jacobian from the final STM
    #   jacobian = d(x_final) / d(l_initial)
    jacobian = final_phi[0:4, 4:8]
    
    return error, jacobian


def generate_heuristic_guess(t_o, t_f, posvel_o, posvel_f, thrust_min, thrust_max, k_heaviside):
    """
    Generates a robust, physics-based initial guess for the costates.
    """
    time_span = [t_o, t_f]
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
    # --- Problem Definition ---
    t_o, t_f               = 0, 30
    posvel_o               = np.array([ 0,  0, 0, 2])
    posvel_f               = np.array([10, 15, 1, 0])
    thrust_min, thrust_max = 0.0e-0, 0.5e+0
    k_idx0, k_idxn, k_divs = 1.0e-1, 1.0e+2, 50
    # k_divs = int(10*(np.log10(k_idxn)-np.log10(k_idx0)+1) - 1)
    # k_divs = 50
    # k_idx0, k_idxn, k_divs = 1.0e-1, 5.0e1, 10
    
    # Generate an initial guess for the costates
    initial_costates_guess = generate_heuristic_guess(t_o, t_f, posvel_o, posvel_f, thrust_min, thrust_max, k_idx0)

    # --- PART 1: K-Continuation Loop ---
    k_idx0ton = np.logspace(np.log(k_idx0), np.log(k_idxn), k_divs)
    k_results = {}
    print(f"\n--- K-Continuation ---")

    for idx, k_idx in enumerate(k_idx0ton):
        print(f"Step {idx+1}/{len(k_idx0ton)}: Solving for k = {k_idx:.2f}...", end="")
        # sol_root = \
        #     root(
        #         objective, initial_costates_guess,
        #         args=([t_o, t_f], posvel_o, posvel_f, thrust_min, thrust_max, k_idx),
        #         method='lm',
        #         tol=1e-7,
        #     )
        # print(f"initial_costates_guess: {initial_costates_guess}")
        sol_root = \
            root(
                objective_and_jacobian, initial_costates_guess,
                args=([t_o, t_f], posvel_o, posvel_f, thrust_min, thrust_max, k_idx),
                method='lm',
                tol=1e-7,
                jac=True,
            )
        if sol_root.success:
            # initial_costates_guess = sol_root.x
            # print("Success!")
            # z0_optimal = np.concatenate((posvel_o, initial_costates_guess, ))
            # sol_optimal = solve_ivp(freedynamics__minfuel__heaviside__stm, [t_o, t_f], z0_optimal, dense_output=True,
            #                         t_eval=np.linspace(t_o, t_f, 500), args=(thrust_min, thrust_max, k_idx))
            # k_results[k_idx] = sol_optimal

            print("Success!")
            initial_costates_guess = sol_root.x
            z0      = np.concatenate((posvel_o, initial_costates_guess))
            phi0    = np.identity(8).flatten()
            z_aug_0 = np.concatenate([z0, phi0])
    
            # Integrate the augmented system
            sol_optimal = solve_ivp(freedynamics__minfuel__heaviside__stm, [t_o, t_f], z_aug_0, dense_output=True, 
                            args=(thrust_min, thrust_max, k_idx), 
                            method='RK45', # DOP853 | RK45
                            rtol=1e-12, atol=1e-12)
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


def plot_final_results(solution, initial_states, final_states, thrust_min, thrust_max, k):
    """
    Calculates and plots all relevant results for the final trajectory solution.
    """
    # Unpack state and costate histories
    time, states = solution.t, solution.y
    rx, ry, vx, vy = states[0], states[1], states[2], states[3]
    l_vx, l_vy = states[6], states[7]
    
    # Recalculate the thrust profile to match the dynamics function
    epsilon          = 1e-10
    lambda_v_norm    = np.sqrt(l_vx**2 + l_vy**2 + epsilon**2)
    switching_func   = lambda_v_norm - 1
    heaviside_approx = 0.5 + 0.5 * np.tanh(k * switching_func)
    thrust_mag_hist  = thrust_min + (thrust_max - thrust_min) * heaviside_approx

    # --- Create the plots ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle(f'Final Trajectory (k={k:.1f}, T_max={thrust_max})', fontsize=18)

    # Panel 1: 2D Trajectory Path
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rx, ry, label='Trajectory')
    ax1.scatter(initial_states[0], initial_states[1], color='g', s=100, zorder=5, label='Start')
    ax1.scatter(final_states[0], final_states[1], color='r', s=100, zorder=5, label='End')
    ax1.set_title('2D Trajectory Path'); ax1.set_xlabel('$r_x$'); ax1.set_ylabel('$r_y$')
    ax1.legend(); ax1.grid(True); ax1.axis('equal')

    # Panel 2: Thrust Profile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, thrust_mag_hist, 'r-', label='Thrust Magnitude $||u||$')
    ax2.axhline(y=thrust_max, color='k', linestyle='--', label=f'T_max = {thrust_max}')
    ax2.axhline(y=thrust_min, color='k', linestyle=':', label=f'T_min = {thrust_min}')
    ax2.set_title('Thrust Profile'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Thrust Magnitude')
    ax2.legend(); ax2.grid(True); ax2.set_ylim(0, thrust_max * 1.1)

    # Panel 3: Position vs. Time
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time, rx, label='$r_x(t)$')
    ax3.plot(time, ry, label='$r_y(t)$', linestyle='--')
    ax3.set_title('Position vs. Time'); ax3.set_ylabel('Position'); ax3.legend(); ax3.grid(True)

    # Panel 4: Velocity vs. Time
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time, vx, label='$v_x(t)$')
    ax4.plot(time, vy, label='$v_y(t)$', linestyle='--')
    ax4.set_title('Velocity vs. Time'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Velocity')
    ax4.legend(); ax4.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])


if __name__ == '__main__':
    solve_trajectory()
    plt.show()