import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def dynamics(t, z, T_max, alpha, k_heaviside):
    """
    A single, unified dynamics function that handles both alpha and k continuation.
    """
    rx, ry, vx, vy, l_rx, l_ry, l_vx, l_vy = z
    epsilon = 1e-10
    lambda_v_norm = np.sqrt(l_vx**2 + l_vy**2) + epsilon
    
    alpha_min_threshold = 1.0e-6 + epsilon 

    if alpha > alpha_min_threshold:
        # REGIME 1: Alpha-Continuation (from min-energy to ~min-fuel)
        unconstrained_thrust = (lambda_v_norm - (1 - alpha)) / alpha
        thrust_mag = min(T_max, max(0, unconstrained_thrust))
    else:
        # REGIME 2: K-Continuation (sharpening the control switch)
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
    epsilon = 1e-10
    lambda_v_norm = np.sqrt(l_vx**2 + l_vy**2) + epsilon

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

def generate_heuristic_guess(initial_states, final_states, tf):
    """
    Generates a robust, physics-based initial guess for the costates.
    """
    r0, v0 = initial_states[:2], initial_states[2:]
    rf, vf = final_states[:2], final_states[2:]
    l_r0_guess = (r0 - rf) / tf
    l_v0_guess = v0 - vf
    initial_costates_guess = np.concatenate([l_r0_guess, l_v0_guess])
    print("--- Generated Heuristic Initial Guess ---")
    print(f"  l_r(0) guess: [{l_r0_guess[0]:.2f}, {l_r0_guess[1]:.2f}]")
    print(f"  l_v(0) guess: [{l_v0_guess[0]:.2f}, {l_v0_guess[1]:.2f}]")
    return initial_costates_guess

def solve_trajectory(T_max):
    """
    Main solver that implements the three-stage solution process.
    """
    # --- Problem Definition ---
    t0, tf = 0, 10
    initial_states = np.array([0, 0, 1, 2])
    final_states = np.array([10, 15, 2, 0])
    alpha_final = 1.0e-6
    
    # Generate an initial guess for the costates
    initial_costates_guess = generate_heuristic_guess(initial_states, final_states, tf)

    # --- PART 1: Alpha-Continuation Loop ---
    k_for_alpha_loop = 1.0
    alphas = np.logspace(0, np.log10(alpha_final), 20)
    alpha_results = {}
    
    # Select which alphas to store for plotting
    num_alpha_plots = 6
    indices_to_plot = set(np.linspace(0, len(alphas) - 1, num_alpha_plots, dtype=int))

    print(f"\n--- PART 1: Alpha-Continuation (k = {k_for_alpha_loop}) ---")
    for i, alpha in enumerate(alphas):
        print(f"Step {i+1}/{len(alphas)}: Solving for alpha = {alpha:.1e}...", end="")
        sol_root = root(objective_function, initial_costates_guess, 
                        args=([t0, tf], initial_states, final_states, T_max, alpha, k_for_alpha_loop),
                        method='lm', tol=1e-8)
        if sol_root.success:
            initial_costates_guess = sol_root.x
            print("Success!")
            # If this index is one we want to plot, store the full solution
            if i in indices_to_plot:
                z0 = np.concatenate((initial_states, initial_costates_guess))
                sol = solve_ivp(dynamics, [t0, tf], z0, dense_output=True,
                                args=(T_max, alpha, k_for_alpha_loop), t_eval=np.linspace(t0, tf, 300))
                alpha_results[alpha] = sol
        else:
            print(f"Convergence Failed for alpha={alpha:.1e}. Stopping.")
            return

    plot_alpha_continuation(alpha_results, initial_states, final_states)

    # --- PART 2: K-Continuation Loop ---
    k_initial, k_final = 1.0, 5000.0
    ks = np.linspace(k_initial, k_final, 100)
    k_results = {}
    print(f"\n--- PART 2: K-Continuation (alpha = {alpha_final:.1e}) ---")

    for i, k in enumerate(ks):
        print(f"Step {i+1}/{len(ks)}: Solving for k = {k:.2f}...", end="")
        sol_root = root(objective_function, initial_costates_guess,
                        args=([t0, tf], initial_states, final_states, T_max, alpha_final, k),
                        method='lm', tol=1e-7)
        if sol_root.success:
            initial_costates_guess = sol_root.x
            print("Success!")
            # Store result for plotting the k-continuation
            z0_optimal = np.concatenate((initial_states, initial_costates_guess))
            sol_optimal = solve_ivp(dynamics, [t0, tf], z0_optimal, dense_output=True,
                                    t_eval=np.linspace(t0, tf, 500), args=(T_max, alpha_final, k))
            k_results[k] = sol_optimal
        else:
            print(f"Convergence Failed for k={k:.2f}. Stopping.")
            break
            
    print("\n--- Two-Stage Continuation Complete ---")
    plot_k_continuation(k_results, T_max, alpha_final)

    # --- PART 3: Final Propagation with True Bang-Bang Dynamics ---
    print("\n--- PART 3: Propagating final trajectory with true bang-bang control ---")
    z0_final = np.concatenate((initial_states, initial_costates_guess))
    final_solution = solve_ivp(dynamics_true_min_fuel, [t0, tf], z0_final,
                             dense_output=True, t_eval=np.linspace(t0, tf, 500), args=(T_max,))

    print("\n--- Final State Error Check ---")
    final_state_vector = final_solution.y[0:4, -1]
    error_vector = final_state_vector - final_states
    print(f"               {'rx':>12s} {'ry':>12s} {'vx':>12s} {'vy':>12s}")
    print(f"Target:      {final_states[0]:12.6f} {final_states[1]:12.6f} {final_states[2]:12.6f} {final_states[3]:12.6f}")
    print(f"Actual:      {final_state_vector[0]:12.6f} {final_state_vector[1]:12.6f} {final_state_vector[2]:12.6f} {final_state_vector[3]:12.6f}")
    print(f"Error:       {error_vector[0]:12.3e} {error_vector[1]:12.3e} {error_vector[2]:12.3e} {error_vector[3]:12.3e}")

    plot_true_min_fuel_results(final_solution, initial_states, final_states, T_max)

def plot_alpha_continuation(results, initial_states, final_states):
    """NEW: Plots the progress of the alpha-continuation process."""
    if not results: return
    alphas = sorted(results.keys(), reverse=True)
    colors = cm.cividis(np.linspace(0.1, 1.0, len(alphas)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Part 1: Alpha-Continuation Progress', fontsize=18)

    # Plot 1: 2D Trajectory Path vs. Alpha
    ax1.set_title('Trajectory Path vs. $\\alpha$')
    for alpha, color in zip(alphas, colors):
        rx, ry = results[alpha].y[0], results[alpha].y[1]
        ax1.plot(rx, ry, color=color, label=f'$\\alpha$ = {alpha:.1e}')
    ax1.scatter(initial_states[0], initial_states[1], color='lime', s=100, zorder=5, ec='black', label='Start')
    ax1.scatter(final_states[0], final_states[1], color='red', s=100, zorder=5, ec='black', label='End')
    ax1.set_xlabel('$r_x$'); ax1.set_ylabel('$r_y$'); ax1.legend(title="$\\alpha$ Param"); ax1.grid(True); ax1.axis('equal')

    # Plot 2: Thrust Magnitude vs. Alpha
    ax2.set_title('Thrust Magnitude vs. $\\alpha$')
    for alpha, color in zip(alphas, colors):
        sol = results[alpha]
        l_vx, l_vy = sol.y[6], sol.y[7]
        lambda_v_norm = np.sqrt(l_vx**2 + l_vy**2)
        unconstrained_thrust = (lambda_v_norm - (1 - alpha)) / alpha
        thrust_mag_hist = np.minimum(T_max, np.maximum(0, unconstrained_thrust))
        ax2.plot(sol.t, thrust_mag_hist, color=color)
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Thrust Magnitude $||u||$'); ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

def plot_k_continuation(results, T_max, alpha):
    """Plots the comparison of thrust profiles for different k values."""
    if not results: return
    ks = sorted(results.keys())
    colors = cm.plasma(np.linspace(0.1, 1.0, len(ks)))
    max_legend_entries = 10
    num_ks = len(ks)
    indices_to_label = set(np.linspace(0, num_ks - 1, min(num_ks, max_legend_entries), dtype=int))

    plt.figure(figsize=(10, 6))
    plt.title(f'Part 2: Thrust Profile Sharpening via k-Continuation ($\\alpha={alpha:.1e}$)', fontsize=16)
    
    for i, (k, color) in enumerate(zip(ks, colors)):
        sol = results[k]
        l_vx, l_vy = sol.y[6], sol.y[7]
        lambda_v_norm = np.sqrt(l_vx**2 + l_vy**2)
        switching_function = lambda_v_norm -