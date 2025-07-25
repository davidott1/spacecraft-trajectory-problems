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


def dynamics_with_stm(t, z_aug, T_max, alpha, k_heaviside):
    """
    Computes the derivatives for the augmented state (state + STM).
    z_aug = [rx, ry, ..., l_vy, phi_11, phi_12, ..., phi_88]
    """
    # 1. Unpack the state and the STM
    z = z_aug[:8]
    phi = z_aug[8:].reshape((8, 8))
    
    # 2. Calculate the state derivatives (same as before)
    dz_dt = dynamics(t, z, T_max, alpha, k_heaviside) # Re-use your existing dynamics!
    
    # Unpack for clarity
    l_rx, l_ry, l_vx, l_vy = z[4], z[5], z[6], z[7]
    # Constant parts of the Jacobian
    # State and Costate Derivatives
    # thrust_acc_x = -l_vx
    # thrust_acc_y = -l_vy
    # drx_dt, dry_dt = vx, vy
    # dvx_dt, dvy_dt = thrust_acc_x, thrust_acc_y
    # dl_rx_dt, dl_ry_dt = 0, 0
    # dl_vx_dt, dl_vy_dt = -l_rx, -l_ry
    # z =  [ rx ry vx vy lrx lry lvx lvy ]
    # A = x[  0  0  1  0   0   0   0   0 ] # d( rx_dot) / d(z)
    #     x[  0  0  0  1   0   0   0   0 ] # d( ry_dot) / d(z)
    #      [  0  0  0  0   0   0   x   x ] # d( vx_dot) / d(z)
    #      [  0  0  0  0   0   0   x   x ] # d( vy_dot) / d(z)
    #     x[  0  0  0  0   0   0   0   0 ] # d(lrx_dot) / d(z)
    #     x[  0  0  0  0   0   0   0   0 ] # d(lry_dot) / d(z)
    #     x[  0  0  0  0  -1   0   0   0 ] # d(lvx_dot) / d(z)
    #     x[  0  0  0  0   0  -1   0   0 ] # d(lvy_dot) / d(z)

    epsilon = 1e-10
    L = np.sqrt(l_vx**2 + l_vy**2)
    lambda_v_norm = L + epsilon
    unconstrained_thrust = (lambda_v_norm - (1 - alpha)) / alpha
    thrust_acc_mag = min(T_max, max(0, unconstrained_thrust))
    thrust_acc_x = - (l_vx / lambda_v_norm) * thrust_acc_mag
    thrust_acc_y = - (l_vy / lambda_v_norm) * thrust_acc_mag
    
    # d(vx_dot) / d(lvx), d(vx_dot) / d(lvy)
    # d(vy_dot) / d(lvx), d(vy_dot) / d(lvy)
    dvxdot__dlvx = 0.0 # d(vx_dot) / d(lvx)
    dvxdot__dlvy = 0.0 # d(vx_dot) / d(lvy)
    dvydot__dlvx = 0.0 # d(vy_dot) / d(lvx)
    dvydot__dlvy = 0.0 # d(vy_dot) / d(lvy)
    if L <= 1 - alpha:
        dvxdot__dlvx = 0.0 # d(vx_dot) / d(lvx)
        dvxdot__dlvy = 0.0 # d(vx_dot) / d(lvy)
        dvydot__dlvx = 0.0 # d(vy_dot) / d(lvx)
        dvydot__dlvy = 0.0 # d(vy_dot) / d(lvy)
    elif L >= alpha*T_max + 1 - alpha:
        dvxdot__dlvx = -T_max*l_vy**2  /L**3 # d(vx_dot) / d(lvx)
        dvxdot__dlvy =  T_max*l_vx*l_vy/L**3 # d(vx_dot) / d(lvy)
        dvydot__dlvx =  T_max*l_vx*l_vy/L**3 # d(vy_dot) / d(lvx)
        dvydot__dlvy = -T_max*l_vx**2  /L**3 # d(vy_dot) / d(lvy)
    elif (1 - alpha < L) and (L < alpha*T_max + 1 - alpha):
        dvxdot__dlvx = -1/alpha * (1 - (1 - alpha)       * l_vy**2  /L**3) # d(vx_dot) / d(lvx)
        dvxdot__dlvy = -1       *      (1 - alpha)/alpha * l_vx*l_vy/L**3 # d(vx_dot) / d(lvy)
        dvydot__dlvx = -1       *      (1 - alpha)/alpha * l_vx*l_vy/L**3 # d(vy_dot) / d(lvx)
        dvydot__dlvy = -1/alpha * (1 - (1 - alpha)       * l_vx**2  /L**3) # d(vy_dot) / d(lvy)

    A = np.zeros((8, 8))
    A[0, 2] =  1.0         # d( rx_dot)/d( vx)
    A[1, 3] =  1.0         # d( ry_dot)/d( vy)
    A[2, 6] = dvxdot__dlvx # d( vx_dot)/d(lrx)
    A[2, 7] = dvxdot__dlvy # d( vx_dot)/d(lry)
    A[3, 6] = dvydot__dlvx # d( vy_dot)/d(lrx)
    A[3, 7] = dvydot__dlvy # d( vy_dot)/d(lry)
    A[6, 4] = -1.0         # d(lvx_dot)/d(lrx)
    A[7, 5] = -1.0         # d(lvy_dot)/d(lry)

    # 4. Calculate the STM derivative
    dphi_dt = A @ phi
    
    # 5. Flatten and combine for the final return
    return np.concatenate([dz_dt, dphi_dt.flatten()])


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


def objective_and_jacobian(initial_costates, t_span, initial_states, final_states, T_max, alpha, k_heaviside):
    """
    Objective function that also returns the analytical Jacobian.
    """
    # Initial STM is the 8x8 identity matrix
    phi0 = np.identity(8).flatten()
    
    # Initial augmented state
    z0 = np.concatenate((initial_states, initial_costates))
    z_aug_0 = np.concatenate([z0, phi0])
    
    # Integrate the augmented system
    sol = solve_ivp(dynamics_with_stm, t_span, z_aug_0, dense_output=True, 
                    args=(T_max, alpha, k_heaviside), 
                    method='RK45', # DOP853 | RK45
                    rtol=1e-12, atol=1e-12)
    
    # Extract final state and final STM
    final_aug_state = sol.sol(t_span[1])
    final_z = final_aug_state[:8]
    final_phi = final_aug_state[8:].reshape((8, 8))
    
    # Calculate the error vector
    error = final_z[:4] - final_states
    
    # Extract the required 4x4 Jacobian from the final STM
    #   jacobian = d(x_final) / d(l_initial)
    jacobian = final_phi[0:4, 4:8]
    
    return error, jacobian


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
    final_states = np.array([10, 15, 5, 0])
    alpha_final = 1.0e-1
    
    # Generate an initial guess for the costates
    initial_costates_guess = generate_heuristic_guess(initial_states, final_states, tf)

    # --- PART 1: Alpha-Continuation Loop ---
    k_for_alpha_loop = 1.0
    alphas = np.logspace(0, np.log10(alpha_final), 10)
    alpha_results = {}
    
    # Select which alphas to store for plotting
    num_alpha_plots = 6
    indices_to_plot = set(np.linspace(0, len(alphas) - 1, num_alpha_plots, dtype=int))

    print(f"\n--- PART 1: Alpha-Continuation (k = {k_for_alpha_loop}) ---")
    for i, alpha in enumerate(alphas):
        print(f"Step {i+1}/{len(alphas)}: Solving for alpha = {alpha:.1e}...", end="")
        # sol_root = \
        #     root(
        #         objective_function,
        #         initial_costates_guess, 
        #         args=([t0, tf], initial_states, final_states, T_max, alpha, k_for_alpha_loop),
        #         method='lm',
        #         tol=1e-8,
        #     )
        sol_root = \
            root(
                objective_and_jacobian,
                initial_costates_guess, 
                args=([t0, tf], initial_states, final_states, T_max, alpha, k_for_alpha_loop),
                method='lm', # hybr | lm
                tol=1e-10,
                jac=True,
            )

        if sol_root.success:
            initial_costates_guess = sol_root.x
            print("Success!")
            if i in indices_to_plot:
                z0 = np.concatenate((initial_states, initial_costates_guess))
                sol = solve_ivp(dynamics, [t0, tf], z0, dense_output=True,
                                args=(T_max, alpha, k_for_alpha_loop), t_eval=np.linspace(t0, tf, 300))
                alpha_results[alpha] = sol
        else:
            print(f"Convergence Failed for alpha={alpha:.1e}. Stopping.")
            return

    plot_alpha_continuation(alpha_results, initial_states, final_states, T_max)


def plot_alpha_continuation(results, initial_states, final_states, T_max):
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


if __name__ == '__main__':
    max_thrust = 2.0e1
    solve_trajectory(T_max=max_thrust)
    plt.show()