import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar

# --- CONSTANTS ---
MU = 398600.44  # km^3/s^2

# --- INPUTS ---
sma1 = 6778.0    # km (LEO)
sma2 = 42164.0   # km (GEO)
inc1 = 28.5      # deg (Cape Canaveral latitude)
inc2 = 0.0       # deg (equatorial GEO)

# Convert to radians
inc1_rad = np.radians(inc1)
inc2_rad = np.radians(inc2)
delta_inc_total = inc2_rad - inc1_rad  # Total inclination change (radians)


def get_transfer_time(r1, r2):
    """Calculate time of flight for Hohmann transfer (half period of transfer ellipse)."""
    a_transfer = (r1 + r2) / 2
    T_transfer = np.pi * np.sqrt(a_transfer**3 / MU)
    return T_transfer


def get_deltav(r1, r2, delta_inc1, delta_inc2):
    """
    Calculate ΔV1 and ΔV2 for Hohmann transfer with plane change.
    
    Combined maneuver ΔV = sqrt(v_before^2 + v_after^2 - 2*v_before*v_after*cos(Δi))
    
    Args:
        r1: Initial circular orbit radius
        r2: Final circular orbit radius  
        delta_inc1: Inclination change at burn 1 (radians)
        delta_inc2: Inclination change at burn 2 (radians)
    
    Returns:
        dv1, dv2: Delta-V magnitudes for each burn
    """
    a_transfer = (r1 + r2) / 2
    
    # Velocities at periapsis (r1)
    v_circ1 = np.sqrt(MU / r1)  # Circular velocity at r1
    v_transfer_peri = np.sqrt(MU * (2/r1 - 1/a_transfer))  # Transfer orbit velocity at periapsis
    
    # Velocities at apoapsis (r2)
    v_transfer_apo = np.sqrt(MU * (2/r2 - 1/a_transfer))  # Transfer orbit velocity at apoapsis
    v_circ2 = np.sqrt(MU / r2)  # Circular velocity at r2
    
    # Combined plane change + orbit maneuver (law of cosines)
    dv1 = np.sqrt(v_circ1**2 + v_transfer_peri**2 - 2*v_circ1*v_transfer_peri*np.cos(delta_inc1))
    dv2 = np.sqrt(v_transfer_apo**2 + v_circ2**2 - 2*v_transfer_apo*v_circ2*np.cos(delta_inc2))
    
    return dv1, dv2


def kepler_E(M, e, tol=1e-10):
    """Solve Kepler's equation M = E - e*sin(E) for E using Newton-Raphson."""
    E = M if e < 0.8 else np.pi
    for _ in range(100):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        E_new = E - f / fp
        if np.abs(E_new - E) < tol:
            return E_new
        E = E_new
    return E


def propagate_kepler(r0, v0, dt):
    """
    Propagate state (r0, v0) forward by time dt using Kepler's equation.
    
    Args:
        r0: Initial position vector (3,)
        v0: Initial velocity vector (3,)
        dt: Time to propagate
    
    Returns:
        r, v: Final position and velocity vectors
    """
    # Orbital elements from state
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    
    # Specific angular momentum
    h = np.cross(r0, v0)
    h_mag = np.linalg.norm(h)
    
    # Eccentricity vector
    e_vec = np.cross(v0, h) / MU - r0 / r0_mag
    e = np.linalg.norm(e_vec)
    
    # Semi-major axis
    energy = v0_mag**2 / 2 - MU / r0_mag
    a = -MU / (2 * energy)
    
    # Mean motion
    n = np.sqrt(MU / a**3)
    
    # Initial true anomaly
    if e > 1e-10:
        cos_nu0 = np.dot(e_vec, r0) / (e * r0_mag)
        cos_nu0 = np.clip(cos_nu0, -1, 1)
        sin_nu0 = np.dot(np.cross(e_vec, r0), h) / (e * r0_mag * h_mag)
        nu0 = np.arctan2(sin_nu0, cos_nu0)
    else:
        # Circular orbit - use position angle
        nu0 = np.arctan2(r0[1], r0[0])
    
    # Initial eccentric anomaly
    E0 = 2 * np.arctan2(np.sqrt(1-e) * np.sin(nu0/2), np.sqrt(1+e) * np.cos(nu0/2))
    
    # Initial mean anomaly
    M0 = E0 - e * np.sin(E0)
    
    # Propagate mean anomaly
    M = M0 + n * dt
    
    # Solve for eccentric anomaly
    E = kepler_E(M, e)
    
    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1+e) * np.sin(E/2), np.sqrt(1-e) * np.cos(E/2))
    
    # Radius
    r_mag = a * (1 - e * np.cos(E))
    
    # Perifocal frame unit vectors
    if e > 1e-10:
        p_hat = e_vec / e
    else:
        p_hat = r0 / r0_mag
    
    w_hat = h / h_mag
    q_hat = np.cross(w_hat, p_hat)
    
    # Position and velocity in perifocal frame, then rotate to inertial
    r_pf = r_mag * np.array([np.cos(nu), np.sin(nu), 0])
    v_pf = np.sqrt(MU / (a * (1 - e**2))) * np.array([-np.sin(nu), e + np.cos(nu), 0])
    
    # Rotation matrix from perifocal to inertial
    R = np.column_stack([p_hat, q_hat, w_hat])
    
    r = R @ r_pf
    v = R @ v_pf
    
    return r, v


def generate_trajectory(r1, r2, inc1_rad, delta_inc1, delta_inc2, n_points=200):
    """
    Generate the full transfer trajectory.
    
    Args:
        r1, r2: Orbital radii
        inc1_rad: Initial inclination (radians)
        delta_inc1: Inclination change at first burn (radians)
        delta_inc2: Inclination change at second burn (radians)
        n_points: Number of points for plotting
    
    Returns:
        trajectory: dict with initial orbit, transfer, and final orbit points
    """
    tof = get_transfer_time(r1, r2)
    dv1, dv2 = get_deltav(r1, r2, delta_inc1, delta_inc2)
    
    # Initial state: circular orbit at r1 with inclination inc1
    # Position at ascending node (x-axis), velocity in y-z plane
    r0 = np.array([r1, 0, 0])
    v_circ1 = np.sqrt(MU / r1)
    v0 = v_circ1 * np.array([0, np.cos(inc1_rad), np.sin(inc1_rad)])
    
    # After burn 1: new velocity direction with partial plane change
    inc_after_burn1 = inc1_rad + delta_inc1
    a_transfer = (r1 + r2) / 2
    v_transfer_peri = np.sqrt(MU * (2/r1 - 1/a_transfer))
    v1_after = v_transfer_peri * np.array([0, np.cos(inc_after_burn1), np.sin(inc_after_burn1)])
    
    # Generate initial circular orbit (one full orbit for display)
    t_orbit1 = 2 * np.pi * np.sqrt(r1**3 / MU)
    times_init = np.linspace(0, t_orbit1, n_points)
    init_orbit = []
    for t in times_init:
        r, _ = propagate_kepler(r0, v0, t)
        init_orbit.append(r)
    init_orbit = np.array(init_orbit)
    
    # Generate transfer trajectory
    times_transfer = np.linspace(0, tof, n_points)
    transfer_orbit = []
    for t in times_transfer:
        r, _ = propagate_kepler(r0, v1_after, t)
        transfer_orbit.append(r)
    transfer_orbit = np.array(transfer_orbit)
    
    # Final state after transfer
    r_final, v_transfer_end = propagate_kepler(r0, v1_after, tof)
    
    # After burn 2: circular orbit at r2 with final inclination
    inc_final = inc1_rad + delta_inc1 + delta_inc2
    v_circ2 = np.sqrt(MU / r2)
    v2_after = v_circ2 * np.array([0, -np.cos(inc_final), -np.sin(inc_final)])  # Opposite direction at apoapsis
    
    # Generate final circular orbit
    t_orbit2 = 2 * np.pi * np.sqrt(r2**3 / MU)
    times_final = np.linspace(0, t_orbit2, n_points)
    final_orbit = []
    for t in times_final:
        r, _ = propagate_kepler(r_final, v2_after, t)
        final_orbit.append(r)
    final_orbit = np.array(final_orbit)
    
    # Calculate delta-V vectors
    dv1_vec = v1_after - v0
    dv2_vec = v2_after - v_transfer_end
    
    return {
        'initial': init_orbit,
        'transfer': transfer_orbit,
        'final': final_orbit,
        'tof': tof,
        'dv1': dv1,
        'dv2': dv2,
        'dv_total': dv1 + dv2,
        # Burn 1 vectors
        'r_burn1': r0,
        'v_before_burn1': v0,
        'v_after_burn1': v1_after,
        'dv1_vec': dv1_vec,
        # Burn 2 vectors
        'r_burn2': r_final,
        'v_before_burn2': v_transfer_end,
        'v_after_burn2': v2_after,
        'dv2_vec': dv2_vec,
    }


# --- MAIN ---
if __name__ == "__main__":
    # Calculate transfer time
    tof = get_transfer_time(sma1, sma2)
    print(f"=== Hohmann Transfer with Plane Change ===")
    print(f"\nOrbits:")
    print(f"  Initial: SMA = {sma1:.1f} km, Inc = {inc1:.1f}°")
    print(f"  Final:   SMA = {sma2:.1f} km, Inc = {inc2:.1f}°")
    print(f"\nTransfer time: {tof:.1f} s ({tof/3600:.2f} hours)")
    print(f"Total inclination change: {np.degrees(abs(delta_inc_total)):.1f}°")
    
    # Compare different plane change splits
    print(f"\n--- ΔV vs Plane Change Split ---")
    splits = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    best_split = None
    best_dv = float('inf')
    
    for split in splits:
        delta_inc1 = split * delta_inc_total
        delta_inc2 = (1 - split) * delta_inc_total
        dv1, dv2 = get_deltav(sma1, sma2, delta_inc1, delta_inc2)
        dv_total = dv1 + dv2
        
        print(f"  Split {split*100:5.1f}% at burn 1: ΔV1={dv1:.4f}, ΔV2={dv2:.4f}, Total={dv_total:.4f} km/s")
        
        if dv_total < best_dv:
            best_dv = dv_total
            best_split = split
    
    # Find optimal split numerically
    def total_dv(split):
        delta_inc1 = split * delta_inc_total
        delta_inc2 = (1 - split) * delta_inc_total
        dv1, dv2 = get_deltav(sma1, sma2, delta_inc1, delta_inc2)
        return dv1 + dv2
    result = minimize_scalar(total_dv, bounds=(0, 1), method='bounded',
                             options={'xatol': 1e-12})
    optimal_split = result.x
    optimal_dv = result.fun
    
    print(f"\n  Optimal split: {optimal_split*100:.1f}% at burn 1")
    print(f"  Optimal total ΔV: {optimal_dv:.4f} km/s")
    
    # Use optimal split for trajectory
    delta_inc1_opt = optimal_split * delta_inc_total
    delta_inc2_opt = (1 - optimal_split) * delta_inc_total
    
    # Generate and plot trajectory
    traj = generate_trajectory(sma1, sma2, inc1_rad, delta_inc1_opt, delta_inc2_opt)
    
    print(f"\n--- Trajectory Details ---")
    print(f"  ΔV1: {traj['dv1']:.4f} km/s (inc change: {np.degrees(delta_inc1_opt):.2f}°)")
    print(f"  ΔV2: {traj['dv2']:.4f} km/s (inc change: {np.degrees(delta_inc2_opt):.2f}°)")
    print(f"  Total ΔV: {traj['dv_total']:.4f} km/s")
    
    # 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot orbits
    ax.plot(traj['initial'][:, 0], traj['initial'][:, 1], traj['initial'][:, 2], 
            'b-', linewidth=1.5, label=f'Initial Orbit (i={inc1}°)')
    ax.plot(traj['transfer'][:, 0], traj['transfer'][:, 1], traj['transfer'][:, 2], 
            'g-', linewidth=2, label='Transfer')
    ax.plot(traj['final'][:, 0], traj['final'][:, 1], traj['final'][:, 2], 
            'r-', linewidth=1.5, label=f'Final Orbit (i={inc2}°)')
    
    # Plot Earth
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R_earth = 6371
    x = R_earth * np.outer(np.cos(u), np.sin(v))
    y = R_earth * np.outer(np.sin(u), np.sin(v))
    z = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.6)
    
    # Scale factor for velocity vectors (to make them visible on the plot)
    v_scale = 2000  # km per (km/s) - adjust for visibility
    
    # --- Burn 1 ΔV vector ---
    r1_pos = traj['r_burn1']
    dv1_vec = traj['dv1_vec'] * v_scale
    
    ax.quiver(*r1_pos, *dv1_vec, color='black', arrow_length_ratio=0, 
              linewidth=2.5, label=f'ΔV1 = {traj["dv1"]:.3f} km/s')
    
    # --- Burn 2 ΔV vector ---
    r2_pos = traj['r_burn2']
    dv2_vec = traj['dv2_vec'] * v_scale
    
    ax.quiver(*r2_pos, *dv2_vec, color='black', arrow_length_ratio=0, 
              linewidth=2.5, label=f'ΔV2 = {traj["dv2"]:.3f} km/s')
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(f'Hohmann Transfer with Plane Change\nΔV_total = {traj["dv_total"]:.3f} km/s')
    ax.legend()
    
    # Equal aspect ratio - cube space
    max_range = sma2 * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig('hohmann_transfer.png', dpi=150)
    plt.show()
