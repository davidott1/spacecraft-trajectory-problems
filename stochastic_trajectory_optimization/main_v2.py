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


def lambert_solver(r1, r2, tof, mu=MU, tm=1):
    """
    Solve Lambert's problem using the universal variable formulation.
    
    Args:
        r1: Initial position vector (3,)
        r2: Final position vector (3,)
        tof: Time of flight
        mu: Gravitational parameter
        tm: Transfer type: +1 for short way, -1 for long way
    
    Returns:
        v1, v2: Velocity vectors at r1 and r2
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    # Compute cos(delta_nu) 
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = np.clip(cos_dnu, -1, 1)
    
    # A parameter depends on transfer direction
    A = tm * np.sqrt(r1_mag * r2_mag * (1 + cos_dnu))
    
    if abs(A) < 1e-10:
        raise ValueError("Lambert solver: A ≈ 0, trajectory is degenerate (180° transfer)")
    
    # Stumpff functions
    def stumpff_C(z):
        if z > 1e-6:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < -1e-6:
            return (1 - np.cosh(np.sqrt(-z))) / z
        else:
            return 0.5 - z/24 + z**2/720 - z**3/40320
    
    def stumpff_S(z):
        if z > 1e-6:
            sz = np.sqrt(z)
            return (sz - np.sin(sz)) / (sz**3)
        elif z < -1e-6:
            sz = np.sqrt(-z)
            return (np.sinh(sz) - sz) / (sz**3)
        else:
            return 1/6 - z/120 + z**2/5040 - z**3/362880
    
    # Function to compute time of flight for given z
    def tof_from_z(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        
        y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)
        
        if y < 0:
            return float('inf')
        
        x = np.sqrt(y / C)
        t = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        return t
    
    # Use bisection to find z (more robust than Newton)
    # For elliptic orbits: 0 < z < (2*pi)^2
    # Initial bounds
    z_low = -4 * np.pi**2  # Hyperbolic
    z_high = 4 * np.pi**2  # Elliptic
    
    # Adjust bounds
    while tof_from_z(z_low) < tof:
        z_low -= 4 * np.pi**2
    while tof_from_z(z_high) > tof:
        z_high += 4 * np.pi**2
    
    # Bisection
    for _ in range(100):
        z = (z_low + z_high) / 2
        t = tof_from_z(z)
        
        if abs(t - tof) < 1e-10:
            break
        
        if t < tof:
            z_low = z
        else:
            z_high = z
    
    # Compute final values
    C = stumpff_C(z)
    S = stumpff_S(z)
    y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)
    
    # Lagrange coefficients
    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_mag
    
    # Velocities
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2


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
    
    # --- Transfer positions at EA = 0°, 90°, 180° ---
    # Transfer orbit parameters
    e_transfer = (r2 - r1) / (r2 + r1)  # Eccentricity of transfer ellipse
    
    # EA = 0° (periapsis) - initial position
    EA_0 = 0.0
    r_EA0 = r0.copy()  # Already at periapsis
    _, v_EA0 = propagate_kepler(r0, v1_after, 0)
    
    # EA = 90° (middle of transfer)
    EA_90 = np.pi / 2
    # Time from Kepler's equation: M = E - e*sin(E), t = M/n
    M_90 = EA_90 - e_transfer * np.sin(EA_90)
    n_transfer = np.sqrt(MU / a_transfer**3)
    t_90 = M_90 / n_transfer
    r_EA90, v_EA90 = propagate_kepler(r0, v1_after, t_90)
    
    # EA = 180° (apoapsis) - final position
    EA_180 = np.pi
    r_EA180 = r_final.copy()  # Already at apoapsis
    v_EA180 = v_transfer_end.copy()
    
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
        # Transfer positions at EA = 0°, 90°, 180°
        'transfer_EA0': {'r': r_EA0, 'v': v_EA0, 'EA_deg': 0.0, 't': 0.0},
        'transfer_EA90': {'r': r_EA90, 'v': v_EA90, 'EA_deg': 90.0, 't': t_90},
        'transfer_EA180': {'r': r_EA180, 'v': v_EA180, 'EA_deg': 180.0, 't': tof},
        'transfer_params': {'a': a_transfer, 'e': e_transfer},
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
    
    print(f"\n--- Transfer Positions (EA = 0°, 90°, 180°) ---")
    print(f"  Transfer orbit: a = {traj['transfer_params']['a']:.1f} km, e = {traj['transfer_params']['e']:.4f}")
    for key, label in [('transfer_EA0', 'EA=0° (periapsis)'), 
                       ('transfer_EA90', 'EA=90° (middle)'), 
                       ('transfer_EA180', 'EA=180° (apoapsis)')]:
        pos = traj[key]
        r_mag = np.linalg.norm(pos['r'])
        v_mag = np.linalg.norm(pos['v'])
        print(f"  {label}:")
        print(f"    t = {pos['t']:.1f} s ({pos['t']/60:.1f} min)")
        print(f"    r = [{pos['r'][0]:10.1f}, {pos['r'][1]:10.1f}, {pos['r'][2]:10.1f}] km  (|r| = {r_mag:.1f} km)")
        print(f"    v = [{pos['v'][0]:10.4f}, {pos['v'][1]:10.4f}, {pos['v'][2]:10.4f}] km/s  (|v| = {v_mag:.4f} km/s)")
    
    # --- LAMBERT PROBLEM SOLUTION ---
    print(f"\n--- Lambert Problem Solution (Nominal) ---")
    
    # Get positions and times
    r_init = traj['transfer_EA0']['r']
    r_mid = traj['transfer_EA90']['r']
    r_final = traj['transfer_EA180']['r']
    
    t_init = traj['transfer_EA0']['t']
    t_mid = traj['transfer_EA90']['t']
    t_final = traj['transfer_EA180']['t']
    
    tof_1 = t_mid - t_init    # Time of flight: init -> mid
    tof_2 = t_final - t_mid   # Time of flight: mid -> final
    
    # Solve Lambert problem 1: r_init -> r_mid (perturbed)
    # tm=+1 for short way (prograde direction based on angular momentum)
    v1_dep_L1, v1_arr_L1 = lambert_solver(r_init, r_mid, tof_1, tm=1)
    
    # Solve Lambert problem 2: r_mid (perturbed) -> r_final
    v2_dep_L2, v2_arr_L2 = lambert_solver(r_mid, r_final, tof_2, tm=1)
    
    # Get reference velocities from the direct transfer
    v_init_orbit = traj['v_before_burn1']  # Initial orbit velocity at node
    v_init_transfer = traj['transfer_EA0']['v']  # Transfer velocity at init
    v_mid_transfer = traj['transfer_EA90']['v']  # Transfer velocity at mid
    v_final_transfer = traj['transfer_EA180']['v']  # Transfer velocity at final
    v_final_orbit = traj['v_after_burn2']  # Final orbit velocity at node
    
    # Compute delta-Vs at each node (Lambert solution)
    dv_init_lambert = v1_dep_L1 - v_init_orbit  # From initial orbit to Lambert arc 1
    dv_mid_lambert = v2_dep_L2 - v1_arr_L1      # From Lambert arc 1 arrival to Lambert arc 2 departure
    dv_final_lambert = v_final_orbit - v2_arr_L2  # From Lambert arc 2 to final orbit
    
    dv_init_mag = np.linalg.norm(dv_init_lambert)
    dv_mid_mag = np.linalg.norm(dv_mid_lambert)
    dv_final_mag = np.linalg.norm(dv_final_lambert)
    dv_total_lambert = dv_init_mag + dv_mid_mag + dv_final_mag
    
    # Reference values from direct transfer
    dv_init_direct = np.linalg.norm(traj['dv1_vec'])
    dv_mid_direct = 0.0  # No mid-course maneuver in direct transfer
    dv_final_direct = np.linalg.norm(traj['dv2_vec'])
    dv_total_direct = dv_init_direct + dv_mid_direct + dv_final_direct
    
    # Print comparison table
    print(f"\n  {'Node':<12} {'Lambert ΔV':<14} {'Direct ΔV':<14} {'Difference':<14}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'Initial':<12} {dv_init_mag:>10.6f} km/s {dv_init_direct:>10.6f} km/s {abs(dv_init_mag-dv_init_direct):>10.6f} km/s")
    print(f"  {'Middle':<12} {dv_mid_mag:>10.6f} km/s {dv_mid_direct:>10.6f} km/s {abs(dv_mid_mag-dv_mid_direct):>10.6f} km/s")
    print(f"  {'Final':<12} {dv_final_mag:>10.6f} km/s {dv_final_direct:>10.6f} km/s {abs(dv_final_mag-dv_final_direct):>10.6f} km/s")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'TOTAL':<12} {dv_total_lambert:>10.6f} km/s {dv_total_direct:>10.6f} km/s {abs(dv_total_lambert-dv_total_direct):>10.6f} km/s")
    
    print(f"\n  Lambert arc velocities:")
    print(f"    Arc 1 departure (at init):  [{v1_dep_L1[0]:10.4f}, {v1_dep_L1[1]:10.4f}, {v1_dep_L1[2]:10.4f}] km/s")
    print(f"    Arc 1 arrival (at mid):     [{v1_arr_L1[0]:10.4f}, {v1_arr_L1[1]:10.4f}, {v1_arr_L1[2]:10.4f}] km/s")
    print(f"    Arc 2 departure (at mid):   [{v2_dep_L2[0]:10.4f}, {v2_dep_L2[1]:10.4f}, {v2_dep_L2[2]:10.4f}] km/s")
    print(f"    Arc 2 arrival (at final):   [{v2_arr_L2[0]:10.4f}, {v2_arr_L2[1]:10.4f}, {v2_arr_L2[2]:10.4f}] km/s")
    
    # --- UNOPTIMIZED MCC SOLUTION WITH UNCERTAINTY (3 Samples) ---
    print(f"\n--- Unoptimized MCC + Fixed Final State, Fixed Flight Times (3 Samples) ---")
    
    # Uncertainty parameters
    np.random.seed(42)
    N_samples = 3
    
    # Nominal ΔV1 vector and magnitude
    dv1_nominal_vec = traj['dv1_vec']
    dv1_nominal_mag = np.linalg.norm(dv1_nominal_vec)
    dv1_direction = dv1_nominal_vec / dv1_nominal_mag  # Unit vector
    
    # Compute 1σ as 10% of nominal in absolute units (km/s)
    sigma_dv1_abs = 0.10 * dv1_nominal_mag  # km/s
    
    print(f"  Nominal ΔV1: {dv1_nominal_mag:.4f} km/s")
    print(f"  Uncertainty (1σ): {sigma_dv1_abs*1000:.1f} m/s ({sigma_dv1_abs/dv1_nominal_mag*100:.1f}% of nominal)")
    
    # Storage for MCC solutions
    mcc_solutions = []
    
    # Colors for each sample
    mcc_colors = ['orange', 'magenta', 'cyan']
    
    for i in range(N_samples):
        print(f"\n  --- Sample {i+1} ---")
        
        # 1. Generate random magnitude error (additive, in km/s)
        dv1_error = np.random.normal(0, sigma_dv1_abs)  # Error in km/s
        dv1_actual_mag = dv1_nominal_mag + dv1_error
        dv1_actual_vec = dv1_actual_mag * dv1_direction
        
        print(f"    Error: {dv1_error*1000:+.1f} m/s ({dv1_error/dv1_nominal_mag*100:+.2f}%)")
        print(f"    Actual ΔV1: {dv1_actual_mag:.4f} km/s")
        
        # New velocity after burn 1
        v1_actual = v_init_orbit + dv1_actual_vec
        
        # 2. Propagate to t_mid with perturbed velocity
        r_mid_new, v_mid_arrival = propagate_kepler(r_init, v1_actual, tof_1)
        pos_error = np.linalg.norm(r_mid_new - r_mid)
        print(f"    Position error at mid: {pos_error:.1f} km")
        
        # 3. Use Lambert to solve r_mid_new -> r_final with fixed tof_2
        v_mid_dep_L, v_final_arr_L = lambert_solver(r_mid_new, r_final, tof_2, tm=1)
        
        # 4. Calculate ΔVs
        dv_init_mcc = dv1_actual_mag
        dv_mid_mcc_vec = v_mid_dep_L - v_mid_arrival
        dv_mid_mcc = np.linalg.norm(dv_mid_mcc_vec)
        dv_final_mcc_vec = v_final_orbit - v_final_arr_L
        dv_final_mcc = np.linalg.norm(dv_final_mcc_vec)
        dv_total_mcc = dv_init_mcc + dv_mid_mcc + dv_final_mcc
        
        print(f"    ΔV_mid: {dv_mid_mcc:.4f} km/s, ΔV_final: {dv_final_mcc:.4f} km/s, TOTAL: {dv_total_mcc:.4f} km/s")
        
        # 5. Generate trajectory arcs for plotting
        n_mcc_pts = 100
        
        # Arc 1: Perturbed trajectory from init to r_mid_new
        mcc_arc1 = []
        times_arc1 = np.linspace(0, tof_1, n_mcc_pts)
        for t in times_arc1:
            r_t, _ = propagate_kepler(r_init, v1_actual, t)
            mcc_arc1.append(r_t)
        mcc_arc1 = np.array(mcc_arc1)
        
        # Arc 2: Corrected trajectory from r_mid_new to r_final
        mcc_arc2 = []
        times_arc2 = np.linspace(0, tof_2, n_mcc_pts)
        for t in times_arc2:
            r_t, _ = propagate_kepler(r_mid_new, v_mid_dep_L, t)
            mcc_arc2.append(r_t)
        mcc_arc2 = np.array(mcc_arc2)
        
        # Store solution
        mcc_solutions.append({
            'sample': i + 1,
            'error_ms': dv1_error * 1000,  # Error in m/s
            'error_pct': dv1_error / dv1_nominal_mag * 100,
            'dv_init': dv_init_mcc,
            'dv_mid': dv_mid_mcc,
            'dv_final': dv_final_mcc,
            'dv_total': dv_total_mcc,
            'pos_error': pos_error,
            'r_mid_new': r_mid_new,
            'arc1': mcc_arc1,
            'arc2': mcc_arc2,
            'color': mcc_colors[i]
        })
    
    # Summary table
    print(f"\n  --- Summary Table ---")
    print(f"\n  {'Sample':<8} {'Error':<20} {'ΔV_init':<12} {'ΔV_mid':<12} {'ΔV_final':<12} {'TOTAL':<12} {'Δ from nom':<12}")
    print(f"  {'-'*8} {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Nominal':<8} {'0.0000 km/s (0.00%)':<20} {dv_init_direct:>8.4f} km/s {dv_mid_direct:>8.4f} km/s {dv_final_direct:>8.4f} km/s {dv_total_direct:>8.4f} km/s {0.0:>+8.4f} km/s")
    for sol in mcc_solutions:
        error_str = f"{sol['error_ms']/1000:+.4f} km/s ({sol['error_pct']:+.2f}%)"
        print(f"  {sol['sample']:<8} {error_str:<20} {sol['dv_init']:>8.4f} km/s {sol['dv_mid']:>8.4f} km/s {sol['dv_final']:>8.4f} km/s {sol['dv_total']:>8.4f} km/s {sol['dv_total']-dv_total_direct:>+8.4f} km/s")

    # 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot orbits
    ax.plot(traj['initial'][:, 0], traj['initial'][:, 1], traj['initial'][:, 2], 
            'b-', linewidth=1.5, label=f'Initial Orbit (i={inc1}°)')
    ax.plot(traj['transfer'][:, 0], traj['transfer'][:, 1], traj['transfer'][:, 2], 
            'g-', linewidth=2, label='Nominal Transfer')
    ax.plot(traj['final'][:, 0], traj['final'][:, 1], traj['final'][:, 2], 
            'r-', linewidth=1.5, label=f'Final Orbit (i={inc2}°)')
    
    # Plot all MCC trajectories
    for sol in mcc_solutions:
        color = sol['color']
        label_arc1 = f"Sample {sol['sample']} ({sol['error_pct']:+.1f}%)"
        ax.plot(sol['arc1'][:, 0], sol['arc1'][:, 1], sol['arc1'][:, 2], 
                color=color, linestyle='--', linewidth=1.5, label=label_arc1)
        ax.plot(sol['arc2'][:, 0], sol['arc2'][:, 1], sol['arc2'][:, 2], 
                color=color, linestyle='--', linewidth=1.5)
        # Mark the MCC point
        ax.scatter(*sol['r_mid_new'], color=color, s=100, marker='*', zorder=6, edgecolor='black')
    
    # Plot transfer nodes: initial (EA=0°), middle (EA=90°), final (EA=180°)
    r_EA0 = traj['transfer_EA0']['r']
    r_EA90 = traj['transfer_EA90']['r']
    r_EA180 = traj['transfer_EA180']['r']
    
    ax.scatter(*r_EA0, color='green', s=80, marker='o', zorder=5, label='EA=0° (periapsis)')
    ax.scatter(*r_EA90, color='green', s=80, marker='s', zorder=5, label='EA=90° (nominal mid)')
    ax.scatter(*r_EA180, color='green', s=80, marker='o', zorder=5, label='EA=180° (apoapsis)')
    
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
