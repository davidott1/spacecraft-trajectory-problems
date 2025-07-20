import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def dynamics(t, z, T_max, alpha, k_heaviside):
    """
    A single, unified dynamics function that handles both alpha and k continuation.
    
    - If alpha is not at the minimum threshold, it uses the standard continuation law.
    - If alpha IS at the minimum threshold, it switches to the smoothed Heaviside
      control law, allowing k to be varied to "sharpen" the switch.
    """
    rx, ry, vx, vy, l_rx, l_ry, l_vx, l_vy = z
    epsilon = 1e-10
    lambda_v_norm = np.sqrt(l_vx**2 + l_vy**2) + epsilon
    
    # This is the threshold where the dynamics law changes
    alpha_min_threshold = 1.0e-7 + epsilon 

    if alpha > alpha_min_threshold:
        # --- REGIME 1: Alpha-Continuation (from min-energy to ~min-fuel) ---
        # Note: k_heaviside has no effect in this regime
        unconstrained_thrust = (lambda_v_norm - (1 - alpha)) / alpha
        # Use max(0,...) to ensure thrust is non-negative
        thrust_mag = min(T_max, max(0, unconstrained_thrust))
    else:
        # --- REGIME 2: K-Continuation (sharpening the control switch) ---
        # Note: alpha is fixed at its minimum value in this regime
        switching_function = lambda_v_norm - 1.0
        heaviside_approx = 0.5 + 0.5 * np.tanh(k_heaviside * switching_function)
        thrust_mag = T_max * heaviside_approx
            
    thrust_acc_x = - (l_vx / lambda_v_norm) * thrust_mag
    thrust_acc_y = - (l_vy / lambda_v_norm) * thrust_mag

    # State and Costate Derivatives
    drx_dt, dry_dt = vx, vy
    dvx_dt, dvy_dt = thrust_acc_x, thrust_acc_y
    dl_rx_dt, dl_ry_dt = 0, 0
    dl_vx_dt, dl_vy_dt = -l_rx, -l_ry
    
    return [drx_dt, dry_dt, dvx_dt, dvy_dt, dl_rx_dt, dl_ry_dt, dl_vx_dt, dl_vy_dt]


def dynamics_true_min_fuel(t, z, T_max):
    """
    Dynamics for the ideal minimum-fuel problem with a discontinuous
    bang-bang control law. This is used for the final propagation.
    """
    rx, ry, vx, vy, l_rx, l_ry, l_vx, l_vy = z
    epsilon = 1e-9
    lambda_v_norm = np.sqrt(l_vx**2 + l_vy**2) + epsilon

    # Ideal bang-bang control law
    thrust_mag = T_max if lambda_v_norm > 1.0 else 0.0
            
    thrust_acc_x = - (l_vx / lambda_v_norm) * thrust_mag
    thrust_acc_y = - (l_vy / lambda_v_norm) * thrust_mag

    drx_dt, dry_dt = vx, vy
    dvx_dt, dvy_dt = thrust_acc_x, thrust_acc_y
    dl_rx_dt, dl_ry_dt = 0, 0
    dl_vx_dt, dl_vy_dt = -l_rx, -l_ry
    
    return [drx_dt, dry_dt, dvx_dt, dvy_dt, dl_rx_dt, dl_ry_dt, dl_vx_dt, dl_vy_dt]



def objective_function(initial_costates, t_span, initial_states, final_states, T_max, alpha, k_heaviside):
    """
    Objective function for the root-finder that calls the unified dynamics.
    """
    z0 = np.concatenate((initial_states, initial_costates))
    sol = solve_ivp(dynamics, t_span, z0, dense_output=True, 
                    args=(T_max, alpha, k_heaviside), 
                    method='DOP853', rtol=1e-8, atol=1e-8)
    final_states_from_integration = sol.sol(t_span[1])[:4]
    error = final_states_from_integration - final_states
    return error

def solve_trajectory(T_max):
    """
    Main solver that implements the two-stage continuation process.
    1. Alpha-Continuation: Varies alpha from 1 to 1e-7 with k=1.
    2. K-Continuation: Varies k from 1 to 10 with alpha=1e-7.
    """
    t0, tf         = 0, 10
    initial_states = np.array([ 0,  0, 1, 2])
    final_states   = np.array([10, 15, 0, 0])
    alpha_final    = 1.0e-7
    initial_costates_guess = np.zeros(4)
    
    # --- PART 1: Alpha-Continuation Loop ---
    k_for_alpha_loop = 1.0
    alphas = np.logspace(0, np.log10(alpha_final), 100) # From 1.0 down to 1e-7
    alpha_results = {}
    print(f"--- PART 1: Alpha-Continuation (k = {k_for_alpha_loop}) ---")

    for i, alpha in enumerate(alphas):
        print(f"Step {i+1}/{len(alphas)}: Solving for alpha = {alpha:.1e}...", end="")
        sol_root = root(objective_function, initial_costates_guess, 
                        args=([t0, tf], initial_states, final_states, T_max, alpha, k_for_alpha_loop),
                        method='lm', # hybr | lm
                        tol=1e-7)
        if sol_root.success:
            initial_costates_guess = sol_root.x
            print("Success!")
        else:
            print(f"Convergence Failed for alpha={alpha:.1e}. Stopping.")
            return

    # --- PART 2: K-Continuation Loop ---
    k_initial, k_final = 1.0, 100000.0
    ks = np.linspace(k_initial, k_final, 1000) # From 1 to 10
    k_results = {}
    # alpha_final = 1.0e-6 # override alpha to 0 for k-continuation
    print(f"\n--- PART 2: K-Continuation (alpha = {alpha_final:.1e}) ---")

    for i, k in enumerate(ks):
        print(f"Step {i+1}/{len(ks)}: Solving for k = {k:.2f}...", end="")
        sol_root = root(objective_function, initial_costates_guess,
                        args=([t0, tf], initial_states, final_states, T_max, alpha_final, k),
                        method='hybr', # hybr | lm
                        tol=1e-7)
        if sol_root.success:
            initial_costates_guess = sol_root.x
            print("Success!")
            # Store result for plotting the k-continuation
            z0_optimal = np.concatenate((initial_states, initial_costates_guess))
            t_eval = np.linspace(t0, tf, 500)
            sol_optimal = solve_ivp(dynamics, [t0, tf], z0_optimal, dense_output=True,
                                    t_eval=t_eval, args=(T_max, alpha_final, k))
            k_results[k] = sol_optimal
        else:
            print(f"Convergence Failed for k={k:.2f}. Stopping.")
            break
            
    print("\n--- Two-Stage Continuation Complete ---")
    final_solution = k_results[ks[-1]]
    
    # --- PART 3: Final Propagation with True Bang-Bang Dynamics ---
    print("\n--- PART 3: Propagating final trajectory with true bang-bang control ---")
    z0_final = np.concatenate((initial_states, initial_costates_guess))
    t_eval = np.linspace(t0, tf, 500)
    final_solution = solve_ivp(dynamics_true_min_fuel, [t0, tf], z0_final,
                             dense_output=True, t_eval=t_eval, args=(T_max,))

    # Check final state error
    final_state_vector = final_solution.y[0:4, -1]
    error_vector = final_state_vector - final_states
    print("\n--- Final State Error Check ---")
    print(f"               {'rx':>12s} {'ry':>12s} {'vx':>12s} {'vy':>12s}")
    print(f"Target:      {final_states[0]:12.6f} {final_states[1]:12.6f} {final_states[2]:12.6f} {final_states[3]:12.6f}")
    print(f"Actual:      {final_state_vector[0]:12.6f} {final_state_vector[1]:12.6f} {final_state_vector[2]:12.6f} {final_state_vector[3]:12.6f}")
    print(f"Error:       {error_vector[0]:12.3e} {error_vector[1]:12.3e} {error_vector[2]:12.3e} {error_vector[3]:12.3e}")

    # Plot the final results
    plot_true_min_fuel_results(final_solution, initial_states, final_states, T_max)



def plot_k_continuation(results, T_max, alpha):
    """
    MODIFIED: Plots the comparison of thrust profiles for different k values,
    but limits the legend to a maximum of 10 entries for clarity.
    """
    ks = sorted(results.keys())
    colors = cm.plasma(np.linspace(0, 1, len(ks)))
    
    # Determine which k-values to label in the legend
    max_legend_entries = 10
    num_ks = len(ks)
    if num_ks > max_legend_entries:
        indices_to_label = set(np.linspace(0, num_ks - 1, max_legend_entries, dtype=int))
    else:
        indices_to_label = set(range(num_ks))

    plt.figure(figsize=(10, 6))
    plt.title(f'Thrust Profile Sharpening via k-Continuation ($\\alpha={alpha:.1e}$)', fontsize=16)
    
    for i, (k, color) in enumerate(zip(ks, colors)):
        sol = results[k]
        l_vx, l_vy = sol.y[6], sol.y[7]
        
        lambda_v_norm = np.sqrt(l_vx**2 + l_vy**2)
        switching_function = lambda_v_norm - 1.0
        heaviside_approx = 0.5 + 0.5 * np.tanh(k * switching_function)
        thrust_mag_hist = T_max * heaviside_approx
        
        # Only add a label for the selected indices
        label = f'k = {k:.1f}' if i in indices_to_label else None
        plt.plot(sol.t, thrust_mag_hist, color=color, label=label)

    plt.xlabel('Time (s)'); plt.ylabel('Thrust Magnitude $||u||$'); plt.legend(title="k Param"); plt.grid(True)
    plt.tight_layout()


def plot_final_results(solution, initial_states, final_states, T_max, k, alpha):
    """Calculates and plots the results for the final solution."""
    time, states = solution.t, solution.y
    rx, ry, vx, vy = states[0], states[1], states[2], states[3]
    l_vx, l_vy = states[6], states[7]
    
    # Re-calculate thrust history for plotting
    lambda_v_norm = np.sqrt(l_vx**2 + l_vy**2)
    switching_function = lambda_v_norm - 1.0
    heaviside_approx = 0.5 + 0.5 * np.tanh(k * switching_function)
    thrust_mag_hist = T_max * heaviside_approx

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle(f'Final Trajectory ($\\alpha={alpha:.1e}$, k={k:.1f}, T_max={T_max})', fontsize=18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rx, ry, label='Trajectory')
    ax1.scatter(initial_states[0], initial_states[1], color='g', s=100, zorder=5, label='Start')
    ax1.scatter(final_states[0], final_states[1], color='r', s=100, zorder=5, label='End')
    ax1.set_title('2D Trajectory Path'); ax1.set_xlabel('$r_x$'); ax1.set_ylabel('$r_y$'); ax1.legend(); ax1.grid(True); ax1.axis('equal')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, thrust_mag_hist, 'r-', label='Thrust Magnitude $||u||$')
    ax2.axhline(y=T_max, color='k', linestyle='--', label=f'T_max = {T_max}')
    ax2.set_title('Thrust Profile'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Thrust Magnitude'); ax2.legend(); ax2.grid(True); ax2.set_ylim(-0.1, T_max + 0.1)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time, rx, label='$r_x(t)$'); ax3.plot(time, ry, label='$r_y(t)$', linestyle='--')
    ax3.set_title('Position vs. Time'); ax3.set_ylabel('Position'); ax3.legend(); ax3.grid(True)

    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time, vx, label='$v_x(t)$'); ax4.plot(time, vy, label='$v_y(t)$', linestyle='--')
    ax4.set_title('Velocity vs. Time'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Velocity'); ax4.legend(); ax4.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])


def plot_true_min_fuel_results(solution, initial_states, final_states, T_max):
    """Calculates and plots the results for the true min-fuel propagation."""
    time, states = solution.t, solution.y
    rx, ry, vx, vy = states[0], states[1], states[2], states[3]
    l_vx, l_vy = states[6], states[7]
    
    # Re-calculate thrust history using the ideal bang-bang law
    thrust_mag_hist = np.array([T_max if np.sqrt(lvx**2 + lvy**2) > 1.0 else 0.0 for lvx, lvy in zip(l_vx, l_vy)])

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle(f'True Minimum Fuel Trajectory (T_max={T_max})', fontsize=18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rx, ry, label='Trajectory')
    ax1.scatter(initial_states[0], initial_states[1], color='g', s=100, zorder=5, label='Start')
    ax1.scatter(final_states[0], final_states[1], color='r', s=100, zorder=5, label='End')
    ax1.set_title('2D Trajectory Path'); ax1.set_xlabel('$r_x$'); ax1.set_ylabel('$r_y$'); ax1.legend(); ax1.grid(True); ax1.axis('equal')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, thrust_mag_hist, 'r-', label='Thrust Magnitude $||u||$')
    ax2.axhline(y=T_max, color='k', linestyle='--', label=f'T_max = {T_max}')
    ax2.set_title('Thrust Profile (Bang-Bang)'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Thrust Magnitude'); ax2.legend(); ax2.grid(True); ax2.set_ylim(-0.1, T_max + 0.1)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time, rx, label='$r_x(t)$'); ax3.plot(time, ry, label='$r_y(t)$', linestyle='--')
    ax3.set_title('Position vs. Time'); ax3.set_ylabel('Position'); ax3.legend(); ax3.grid(True)

    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(time, vx, label='$v_x(t)$'); ax4.plot(time, vy, label='$v_y(t)$', linestyle='--')
    ax4.set_title('Velocity vs. Time'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Velocity'); ax4.legend(); ax4.grid(True)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96]) # type: ignore


if __name__ == '__main__':
    max_thrust = 2.0
    solve_trajectory(T_max=max_thrust)
    plt.show()



# --- Continuation Complete ---

# --- Best Costate Guess from Continuation (for alpha=0.000) ---
#   Initial l_rx(0) = -0.071891
#   Initial l_ry(0) = 0.016474
#   Initial l_vx(0) = -0.122375
#   Initial l_vy(0) = 0.997848

# --- PART 2: Refining for pure Minimum-Fuel Solution (alpha = 0) ---
# Using result from alpha_continuation as initial guess...
# Success! Found refined minimum-fuel solution. ✅

# --- Exact Final State vs. Target ---
#                          rx           ry           vx           vy
# Target:         10.000000    15.000000     0.000000     0.000000
# Actual:         10.002447    14.999810     0.000227     0.001531
# Error:          2.447e-03   -1.898e-04    2.268e-04    1.531e-03

# --- Terminal Costate Solution (at t=tf) ---
#   l_rx(10) = -0.071908
#   l_ry(10) = 0.016467
#   l_vx(10) = 0.596706
#   l_vy(10) = 0.833176

####

# --- Continuation Complete ---

# --- Best Costate Guess from Continuation (for alpha=0.000) ---
#   Initial l_rx(0) = -0.071814
#   Initial l_ry(0) = 0.016395
#   Initial l_vx(0) = -0.122421
#   Initial l_vy(0) = 0.997900

# --- PART 2: Refining for pure Minimum-Fuel Solution (alpha = 0) ---
# Using result from alpha_continuation as initial guess...
# Success! Found refined minimum-fuel solution. ✅

# --- Exact Final State vs. Target ---
#                          rx           ry           vx           vy
# Target:         10.000000    15.000000     0.000000     0.000000
# Actual:          9.998953    15.000812    -0.002724    -0.002885
# Error:         -1.047e-03    8.122e-04   -2.724e-03   -2.885e-03

# --- Terminal Costate Solution (at t=tf) ---
#   l_rx(10) = -0.071900
#   l_ry(10) = 0.016454
#   l_vx(10) = 0.596653
#   l_vy(10) = 0.833298