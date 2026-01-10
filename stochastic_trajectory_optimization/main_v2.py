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


def compute_stm(r0, v0, dt, eps=1e-6):
    """
    Compute the State Transition Matrix (STM) numerically using finite differences.
    
    The STM Φ(t, t₀) maps perturbations in initial state to final state:
    [δr(t)]   [Φ_rr  Φ_rv] [δr₀]
    [δv(t)] = [Φ_vr  Φ_vv] [δv₀]
    
    Args:
        r0: Initial position vector (3,)
        v0: Initial velocity vector (3,)
        dt: Time of flight
        eps: Finite difference step size
    
    Returns:
        STM: 6x6 State Transition Matrix
        Phi_rr, Phi_rv, Phi_vr, Phi_vv: 3x3 submatrices
    """
    STM = np.zeros((6, 6))
    
    # Nominal final state
    r_nom, v_nom = propagate_kepler(r0, v0, dt)
    
    # Perturb each initial state component
    for i in range(6):
        # Create perturbed initial state
        if i < 3:
            r0_pert = r0.copy()
            r0_pert[i] += eps
            v0_pert = v0.copy()
        else:
            r0_pert = r0.copy()
            v0_pert = v0.copy()
            v0_pert[i-3] += eps
        
        # Propagate perturbed state
        r_pert, v_pert = propagate_kepler(r0_pert, v0_pert, dt)
        
        # Finite difference
        STM[0:3, i] = (r_pert - r_nom) / eps
        STM[3:6, i] = (v_pert - v_nom) / eps
    
    # Extract submatrices
    Phi_rr = STM[0:3, 0:3]  # ∂r/∂r₀
    Phi_rv = STM[0:3, 3:6]  # ∂r/∂v₀  (position sensitivity to velocity)
    Phi_vr = STM[3:6, 0:3]  # ∂v/∂r₀
    Phi_vv = STM[3:6, 3:6]  # ∂v/∂v₀
    
    return STM, Phi_rr, Phi_rv, Phi_vr, Phi_vv


def stochastic_stm_analysis(r0, v0, v_nominal, dt, sigma_mag, sigma_dir_deg):
    """
    Perform stochastic analysis using the STM.
    
    Computes:
    1. Position covariance at MCC due to magnitude and direction errors
    2. Sensitivity of position error to each error source
    3. Expected ΔV_mid based on position dispersion
    
    Args:
        r0: Initial position (at burn 1)
        v0: Initial orbit velocity (before burn 1)
        v_nominal: Nominal post-burn velocity
        dt: Time of flight to MCC
        sigma_mag: 1σ magnitude error (km/s)
        sigma_dir_deg: 1σ direction error per axis (degrees)
    
    Returns:
        dict with covariance analysis results
    """
    # Compute STM
    _, Phi_rr, Phi_rv, Phi_vr, Phi_vv = compute_stm(r0, v_nominal, dt)
    
    # --- Velocity covariance due to MAGNITUDE error ---
    # δv = δ|v| * v_hat, so Σ_v = σ²_mag * v_hat ⊗ v_hat
    v_hat = v_nominal / np.linalg.norm(v_nominal)
    Sigma_v_mag = sigma_mag**2 * np.outer(v_hat, v_hat)
    
    # --- Velocity covariance due to DIRECTION error ---
    # Direction errors are perpendicular to v_hat
    # Build orthonormal basis
    if abs(v_hat[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])
    e1 = temp - np.dot(temp, v_hat) * v_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    
    # Direction error in velocity: δv = |v| * (tan(δθ₁)*e1 + tan(δθ₂)*e2)
    # For small angles: δv ≈ |v| * (δθ₁*e1 + δθ₂*e2) where δθ in radians
    v_mag = np.linalg.norm(v_nominal)
    sigma_dir_rad = np.radians(sigma_dir_deg)
    
    # Covariance: Σ_v = |v|² * σ²_dir * (e1⊗e1 + e2⊗e2)
    Sigma_v_dir = v_mag**2 * sigma_dir_rad**2 * (np.outer(e1, e1) + np.outer(e2, e2))
    
    # --- Propagate covariance to MCC position ---
    # P_r(t) = Φ_rv * Σ_v * Φ_rv^T  (for velocity-only initial uncertainty)
    P_r_mag = Phi_rv @ Sigma_v_mag @ Phi_rv.T
    P_r_dir = Phi_rv @ Sigma_v_dir @ Phi_rv.T
    P_r_total = P_r_mag + P_r_dir
    
    # Position standard deviations (sqrt of diagonal)
    sigma_r_mag = np.sqrt(np.diag(P_r_mag))
    sigma_r_dir = np.sqrt(np.diag(P_r_dir))
    sigma_r_total = np.sqrt(np.diag(P_r_total))
    
    # RMS position error (trace)
    rms_r_mag = np.sqrt(np.trace(P_r_mag))
    rms_r_dir = np.sqrt(np.trace(P_r_dir))
    rms_r_total = np.sqrt(np.trace(P_r_total))
    
    # --- Velocity covariance at MCC ---
    P_v_mag = Phi_vv @ Sigma_v_mag @ Phi_vv.T
    P_v_dir = Phi_vv @ Sigma_v_dir @ Phi_vv.T
    
    # --- Estimate ΔV_mid from position dispersion ---
    # ΔV_mid ≈ ||δr|| / t_remaining * correction_factor
    # More precisely: solve Lambert from perturbed r_mid to r_final
    # For linear estimate: ΔV ∝ position_error / time_remaining
    
    return {
        'STM_Phi_rv': Phi_rv,
        'STM_Phi_vv': Phi_vv,
        'Sigma_v_mag': Sigma_v_mag,
        'Sigma_v_dir': Sigma_v_dir,
        'P_r_mag': P_r_mag,
        'P_r_dir': P_r_dir,
        'P_r_total': P_r_total,
        'sigma_r_mag': sigma_r_mag,
        'sigma_r_dir': sigma_r_dir,
        'sigma_r_total': sigma_r_total,
        'rms_r_mag': rms_r_mag,
        'rms_r_dir': rms_r_dir,
        'rms_r_total': rms_r_total,
        'P_v_mag': P_v_mag,
        'P_v_dir': P_v_dir,
    }


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
    
    # --- Monte Carlo Objective Function (Magnitude Variation) ---
    def mc_objective(dv1_mag, dv1_dir, sigma_dv1, N_mc, seed=None):
        """
        Run Monte Carlo simulation and return expected total ΔV.
        MAGNITUDE VARIATION ONLY.
        
        Args:
            dv1_mag: Nominal ΔV1 magnitude (decision variable)
            dv1_dir: ΔV1 direction unit vector (fixed)
            sigma_dv1: Standard deviation of ΔV1 error (km/s)
            N_mc: Number of Monte Carlo samples
            seed: Random seed for reproducibility
        
        Returns:
            mean_total_dv: Expected total ΔV
            stats: Dictionary with detailed statistics
        """
        if seed is not None:
            np.random.seed(seed)
        
        dv_init_list = []
        dv_mid_list = []
        dv_final_list = []
        dv_total_list = []
        r_mid_list = []
        mag_error_list = []  # Track magnitude errors
        
        for _ in range(N_mc):
            # 1. Generate random magnitude error (additive)
            dv1_error = np.random.normal(0, sigma_dv1)
            dv1_actual_mag = dv1_mag + dv1_error
            dv1_actual_vec = dv1_actual_mag * dv1_dir
            
            # New velocity after burn 1
            v1_actual = v_init_orbit + dv1_actual_vec
            
            # Check if orbit is elliptic
            v_mag_sq = np.dot(v1_actual, v1_actual)
            r_mag = np.linalg.norm(r_init)
            energy = v_mag_sq / 2 - MU / r_mag
            if energy >= 0:
                continue
            
            # 2. Propagate to t_mid
            r_mid_new, v_mid_arrival = propagate_kepler(r_init, v1_actual, tof_1)
            if np.any(np.isnan(r_mid_new)):
                continue
            
            # 3. Lambert solve r_mid_new -> r_final
            v_mid_dep_L, v_final_arr_L = lambert_solver(r_mid_new, r_final, tof_2, tm=1)
            if np.any(np.isnan(v_mid_dep_L)):
                continue
            
            # 4. Calculate ΔVs
            dv_init = dv1_actual_mag
            dv_mid = np.linalg.norm(v_mid_dep_L - v_mid_arrival)
            dv_final = np.linalg.norm(v_final_orbit - v_final_arr_L)
            dv_total = dv_init + dv_mid + dv_final
            
            dv_init_list.append(dv_init)
            dv_mid_list.append(dv_mid)
            dv_final_list.append(dv_final)
            dv_total_list.append(dv_total)
            r_mid_list.append(r_mid_new)
            mag_error_list.append(dv1_error)  # Store magnitude error
        
        dv_init_arr = np.array(dv_init_list)
        dv_mid_arr = np.array(dv_mid_list)
        dv_final_arr = np.array(dv_final_list)
        dv_total_arr = np.array(dv_total_list)
        mag_error_arr = np.array(mag_error_list)
        
        stats = {
            'n_valid': len(dv_total_arr),
            'dv_init_mean': np.mean(dv_init_arr),
            'dv_init_std': np.std(dv_init_arr),
            'dv_mid_mean': np.mean(dv_mid_arr),
            'dv_mid_std': np.std(dv_mid_arr),
            'dv_final_mean': np.mean(dv_final_arr),
            'dv_final_std': np.std(dv_final_arr),
            'dv_total_mean': np.mean(dv_total_arr),
            'dv_total_std': np.std(dv_total_arr),
            'dv_total_min': np.min(dv_total_arr),
            'dv_total_max': np.max(dv_total_arr),
            'r_mid_list': r_mid_list,
            'mag_errors': mag_error_arr  # Add magnitude errors to stats
        }
        
        return np.mean(dv_total_arr), stats

    # --- Monte Carlo Objective Function (Magnitude Variation) with FREE MCC timing ---
    def mc_objective_free_time(dv1_mag, dv1_dir, sigma_dv1, delta_t_mid, N_mc, seed=None):
        """
        Run Monte Carlo simulation with FREE MCC timing.
        Total flight time is fixed, but MCC timing can vary.
        
        Args:
            dv1_mag: Nominal ΔV1 magnitude (decision variable)
            dv1_dir: ΔV1 direction unit vector (fixed)
            sigma_dv1: Standard deviation of ΔV1 error (km/s)
            delta_t_mid: Offset to nominal MCC time (seconds). 
                         tof_1_new = tof_1 + delta_t_mid
                         tof_2_new = tof_2 - delta_t_mid (total time conserved)
            N_mc: Number of Monte Carlo samples
            seed: Random seed for reproducibility
        
        Returns:
            mean_total_dv: Expected total ΔV
            stats: Dictionary with detailed statistics
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Adjusted flight times
        tof_1_adj = tof_1 + delta_t_mid
        tof_2_adj = tof_2 - delta_t_mid
        
        # Check validity
        if tof_1_adj <= 0 or tof_2_adj <= 0:
            return float('inf'), {}
        
        dv_init_list = []
        dv_mid_list = []
        dv_final_list = []
        dv_total_list = []
        r_mid_list = []
        
        for _ in range(N_mc):
            # 1. Generate random magnitude error (additive)
            dv1_error = np.random.normal(0, sigma_dv1)
            dv1_actual_mag = dv1_mag + dv1_error
            dv1_actual_vec = dv1_actual_mag * dv1_dir
            
            # New velocity after burn 1
            v1_actual = v_init_orbit + dv1_actual_vec
            
            # Check if orbit is elliptic
            v_mag_sq = np.dot(v1_actual, v1_actual)
            r_mag = np.linalg.norm(r_init)
            energy = v_mag_sq / 2 - MU / r_mag
            if energy >= 0:
                continue
            
            # 2. Propagate to adjusted t_mid
            try:
                r_mid_new, v_mid_arrival = propagate_kepler(r_init, v1_actual, tof_1_adj)
                if np.any(np.isnan(r_mid_new)):
                    continue
            except:
                continue
            
            # 3. Lambert solve r_mid_new -> r_final with adjusted tof_2
            try:
                v_mid_dep_L, v_final_arr_L = lambert_solver(r_mid_new, r_final, tof_2_adj, tm=1)
                if np.any(np.isnan(v_mid_dep_L)):
                    continue
            except:
                continue
            
            # 4. Calculate ΔVs
            dv_init = dv1_actual_mag
            dv_mid = np.linalg.norm(v_mid_dep_L - v_mid_arrival)
            dv_final = np.linalg.norm(v_final_orbit - v_final_arr_L)
            dv_total = dv_init + dv_mid + dv_final
            
            dv_init_list.append(dv_init)
            dv_mid_list.append(dv_mid)
            dv_final_list.append(dv_final)
            dv_total_list.append(dv_total)
            r_mid_list.append(r_mid_new)
        
        if len(dv_total_list) == 0:
            return float('inf'), {}
        
        dv_init_arr = np.array(dv_init_list)
        dv_mid_arr = np.array(dv_mid_list)
        dv_final_arr = np.array(dv_final_list)
        dv_total_arr = np.array(dv_total_list)
        
        stats = {
            'n_valid': len(dv_total_arr),
            'dv_init_mean': np.mean(dv_init_arr),
            'dv_init_std': np.std(dv_init_arr),
            'dv_mid_mean': np.mean(dv_mid_arr),
            'dv_mid_std': np.std(dv_mid_arr),
            'dv_final_mean': np.mean(dv_final_arr),
            'dv_final_std': np.std(dv_final_arr),
            'dv_total_mean': np.mean(dv_total_arr),
            'dv_total_std': np.std(dv_total_arr),
            'dv_total_min': np.min(dv_total_arr),
            'dv_total_max': np.max(dv_total_arr),
            'r_mid_list': r_mid_list,
            'tof_1_adj': tof_1_adj,
            'tof_2_adj': tof_2_adj
        }
        
        return np.mean(dv_total_arr), stats

    # --- Helper: Sample Direction with 2D Gaussian Perturbation ---
    def sample_gaussian_direction(nominal_dir, sigma_deg, N_samples, seed=None):
        """
        Sample directions by adding 2D Gaussian perturbations to the nominal direction.
        
        Model: v_perturbed = normalize(v_nominal + ε₁*e₁ + ε₂*e₂)
        where ε₁, ε₂ ~ N(0, σ²) are independent Gaussian errors in two 
        perpendicular directions.
        
        The resulting cone angle approximately follows a Rayleigh distribution.
        
        Args:
            nominal_dir: Unit vector for nominal direction (3,)
            sigma_deg: Standard deviation of each axis perturbation (degrees)
                       Converted to radians internally as perturbation magnitude
            N_samples: Number of samples to generate
            seed: Random seed
            
        Returns:
            directions: Array of unit direction vectors (N_samples, 3)
            cone_angles: Array of cone angles in degrees (N_samples,)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Convert sigma from degrees to radians (as a length scale on unit sphere)
        sigma_rad = np.radians(sigma_deg)
        
        # Normalize nominal direction
        nominal_dir = nominal_dir / np.linalg.norm(nominal_dir)
        
        # Build orthonormal basis: e1, e2 perpendicular to nominal_dir
        if abs(nominal_dir[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        # Gram-Schmidt
        e1 = temp - np.dot(temp, nominal_dir) * nominal_dir
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(nominal_dir, e1)
        
        # Sample 2D Gaussian perturbations
        eps1 = np.random.normal(0, sigma_rad, N_samples)
        eps2 = np.random.normal(0, sigma_rad, N_samples)
        
        directions = np.zeros((N_samples, 3))
        cone_angles_deg = np.zeros(N_samples)
        
        for i in range(N_samples):
            # Perturbed direction (not normalized yet)
            v_perturbed = nominal_dir + eps1[i] * e1 + eps2[i] * e2
            
            # Normalize to unit sphere
            v_perturbed = v_perturbed / np.linalg.norm(v_perturbed)
            directions[i] = v_perturbed
            
            # Compute cone angle from nominal
            cos_angle = np.clip(np.dot(v_perturbed, nominal_dir), -1, 1)
            cone_angles_deg[i] = np.degrees(np.arccos(cos_angle))
        
        return directions, cone_angles_deg

    # --- Monte Carlo Objective Function (Direction Variation Only) ---
    def mc_objective_direction(dv1_mag, dv1_dir_nominal, sigma_dir_deg, N_mc, seed=None):
        """
        Run Monte Carlo simulation with DIRECTION VARIATION ONLY.
        Magnitude is fixed at dv1_mag.
        
        Direction is perturbed using 2D Gaussian in two perpendicular axes.
        The resulting cone angle follows a Rayleigh distribution.
        
        Args:
            dv1_mag: Fixed ΔV1 magnitude (km/s)
            dv1_dir_nominal: Nominal ΔV1 direction unit vector
            sigma_dir_deg: 1σ pointing error per axis (degrees)
            N_mc: Number of Monte Carlo samples
            seed: Random seed for reproducibility
        
        Returns:
            mean_total_dv: Expected total ΔV
            stats: Dictionary with detailed statistics
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample directions with 2D Gaussian perturbations
        directions, cone_angles = sample_gaussian_direction(
            dv1_dir_nominal, sigma_dir_deg, N_mc, seed=None  # Already seeded
        )
        
        dv_init_list = []
        dv_mid_list = []
        dv_final_list = []
        dv_total_list = []
        r_mid_list = []
        cone_angle_list = []
        
        for i in range(N_mc):
            # 1. Apply direction perturbation (magnitude fixed)
            dv1_actual_dir = directions[i]
            dv1_actual_vec = dv1_mag * dv1_actual_dir
            
            # New velocity after burn 1
            v1_actual = v_init_orbit + dv1_actual_vec
            
            # Check if orbit is elliptic
            v_mag_sq = np.dot(v1_actual, v1_actual)
            r_mag = np.linalg.norm(r_init)
            energy = v_mag_sq / 2 - MU / r_mag
            if energy >= 0:
                continue
            
            # 2. Propagate to t_mid
            try:
                r_mid_new, v_mid_arrival = propagate_kepler(r_init, v1_actual, tof_1)
                if np.any(np.isnan(r_mid_new)):
                    continue
            except:
                continue
            
            # 3. Lambert solve r_mid_new -> r_final
            try:
                v_mid_dep_L, v_final_arr_L = lambert_solver(r_mid_new, r_final, tof_2, tm=1)
                if np.any(np.isnan(v_mid_dep_L)):
                    continue
            except:
                continue
            
            # 4. Calculate ΔVs
            dv_init = dv1_mag  # Fixed magnitude
            dv_mid = np.linalg.norm(v_mid_dep_L - v_mid_arrival)
            dv_final = np.linalg.norm(v_final_orbit - v_final_arr_L)
            dv_total = dv_init + dv_mid + dv_final
            
            dv_init_list.append(dv_init)
            dv_mid_list.append(dv_mid)
            dv_final_list.append(dv_final)
            dv_total_list.append(dv_total)
            r_mid_list.append(r_mid_new)
            cone_angle_list.append(cone_angles[i])
        
        dv_init_arr = np.array(dv_init_list)
        dv_mid_arr = np.array(dv_mid_list)
        dv_final_arr = np.array(dv_final_list)
        dv_total_arr = np.array(dv_total_list)
        cone_angle_arr = np.array(cone_angle_list)
        
        stats = {
            'n_valid': len(dv_total_arr),
            'dv_init_mean': np.mean(dv_init_arr),
            'dv_init_std': np.std(dv_init_arr),
            'dv_mid_mean': np.mean(dv_mid_arr),
            'dv_mid_std': np.std(dv_mid_arr),
            'dv_final_mean': np.mean(dv_final_arr),
            'dv_final_std': np.std(dv_final_arr),
            'dv_total_mean': np.mean(dv_total_arr),
            'dv_total_std': np.std(dv_total_arr),
            'dv_total_min': np.min(dv_total_arr),
            'dv_total_max': np.max(dv_total_arr),
            'cone_angle_mean': np.mean(cone_angle_arr),
            'cone_angle_std': np.std(cone_angle_arr),
            'cone_angle_max': np.max(cone_angle_arr),
            'cone_angles': cone_angle_arr,  # Full array of cone angles
            'r_mid_list': r_mid_list
        }
        
        return np.mean(dv_total_arr), stats

    # --- Monte Carlo Objective Function (Direction Variation) with FREE MCC timing ---
    def mc_objective_direction_free_time(dv1_mag, dv1_dir_nominal, sigma_dir_deg, delta_t_mid, N_mc, seed=None):
        """
        Run Monte Carlo simulation with DIRECTION VARIATION and FREE MCC timing.
        Total flight time is fixed, but MCC timing can vary.
        
        Args:
            dv1_mag: Fixed ΔV1 magnitude (km/s)
            dv1_dir_nominal: Nominal ΔV1 direction unit vector
            sigma_dir_deg: 1σ pointing error per axis (degrees)
            delta_t_mid: Offset to nominal MCC time (seconds)
            N_mc: Number of Monte Carlo samples
            seed: Random seed for reproducibility
        
        Returns:
            mean_total_dv: Expected total ΔV
            stats: Dictionary with detailed statistics
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Adjusted flight times
        tof_1_adj = tof_1 + delta_t_mid
        tof_2_adj = tof_2 - delta_t_mid
        
        # Check validity
        if tof_1_adj <= 0 or tof_2_adj <= 0:
            return float('inf'), {}
        
        # Sample directions with 2D Gaussian perturbations
        directions, cone_angles = sample_gaussian_direction(
            dv1_dir_nominal, sigma_dir_deg, N_mc, seed=None  # Already seeded
        )
        
        dv_init_list = []
        dv_mid_list = []
        dv_final_list = []
        dv_total_list = []
        r_mid_list = []
        cone_angle_list = []
        
        for i in range(N_mc):
            # 1. Apply direction perturbation (magnitude fixed)
            dv1_actual_dir = directions[i]
            dv1_actual_vec = dv1_mag * dv1_actual_dir
            
            # New velocity after burn 1
            v1_actual = v_init_orbit + dv1_actual_vec
            
            # Check if orbit is elliptic
            v_mag_sq = np.dot(v1_actual, v1_actual)
            r_mag = np.linalg.norm(r_init)
            energy = v_mag_sq / 2 - MU / r_mag
            if energy >= 0:
                continue
            
            # 2. Propagate to adjusted t_mid
            try:
                r_mid_new, v_mid_arrival = propagate_kepler(r_init, v1_actual, tof_1_adj)
                if np.any(np.isnan(r_mid_new)):
                    continue
            except:
                continue
            
            # 3. Lambert solve r_mid_new -> r_final with adjusted tof_2
            try:
                v_mid_dep_L, v_final_arr_L = lambert_solver(r_mid_new, r_final, tof_2_adj, tm=1)
                if np.any(np.isnan(v_mid_dep_L)):
                    continue
            except:
                continue
            
            # 4. Calculate ΔVs
            dv_init = dv1_mag  # Fixed magnitude
            dv_mid = np.linalg.norm(v_mid_dep_L - v_mid_arrival)
            dv_final = np.linalg.norm(v_final_orbit - v_final_arr_L)
            dv_total = dv_init + dv_mid + dv_final
            
            dv_init_list.append(dv_init)
            dv_mid_list.append(dv_mid)
            dv_final_list.append(dv_final)
            dv_total_list.append(dv_total)
            r_mid_list.append(r_mid_new)
            cone_angle_list.append(cone_angles[i])
        
        if len(dv_total_list) == 0:
            return float('inf'), {}
        
        dv_init_arr = np.array(dv_init_list)
        dv_mid_arr = np.array(dv_mid_list)
        dv_final_arr = np.array(dv_final_list)
        dv_total_arr = np.array(dv_total_list)
        cone_angle_arr = np.array(cone_angle_list)
        
        stats = {
            'n_valid': len(dv_total_arr),
            'dv_init_mean': np.mean(dv_init_arr),
            'dv_init_std': np.std(dv_init_arr),
            'dv_mid_mean': np.mean(dv_mid_arr),
            'dv_mid_std': np.std(dv_mid_arr),
            'dv_final_mean': np.mean(dv_final_arr),
            'dv_final_std': np.std(dv_final_arr),
            'dv_total_mean': np.mean(dv_total_arr),
            'dv_total_std': np.std(dv_total_arr),
            'dv_total_min': np.min(dv_total_arr),
            'dv_total_max': np.max(dv_total_arr),
            'cone_angle_mean': np.mean(cone_angle_arr),
            'cone_angle_std': np.std(cone_angle_arr),
            'cone_angle_max': np.max(cone_angle_arr),
            'cone_angles': cone_angle_arr,
            'r_mid_list': r_mid_list,
            'tof_1_adj': tof_1_adj,
            'tof_2_adj': tof_2_adj
        }
        
        return np.mean(dv_total_arr), stats
    
    # --- UNOPTIMIZED MCC SOLUTION (Initial Guess) ---
    print(f"\n--- Unoptimized MCC + Fixed Final State, Fixed Flight Times (Monte Carlo) ---")
    
    # Uncertainty parameters
    N_samples = 100
    
    # Nominal ΔV1 vector and magnitude (Hohmann solution = initial guess)
    dv1_nominal_vec = traj['dv1_vec']
    dv1_nominal_mag = np.linalg.norm(dv1_nominal_vec)
    dv1_direction = dv1_nominal_vec / dv1_nominal_mag  # Unit vector (fixed)
    
    # Compute 1σ as 10% of nominal in absolute units (km/s)
    sigma_dv1_abs = 0.10 * dv1_nominal_mag  # km/s
    
    print(f"  Initial Guess (Hohmann): ΔV1 = {dv1_nominal_mag:.4f} km/s")
    print(f"  Uncertainty (1σ): {sigma_dv1_abs*1000:.1f} m/s")
    print(f"  Number of MC samples: {N_samples}")
    
    # Evaluate initial guess
    mean_dv_init_guess, stats_init = mc_objective(dv1_nominal_mag, dv1_direction, sigma_dv1_abs, N_samples, seed=42)
    print(f"\n  Initial Guess Results:")
    print(f"    E[ΔV_total] = {mean_dv_init_guess:.4f} km/s")
    print(f"    Valid samples: {stats_init['n_valid']}/{N_samples}")
    
    # --- OPTIMIZE nominal ΔV1 to minimize E[ΔV_total] ---
    print(f"\n--- Optimizing Nominal ΔV1 to Minimize E[ΔV_total] ---")
    
    def objective_wrapper(x):
        """Wrapper for optimizer - returns only the mean total ΔV."""
        dv1_mag = x[0]
        mean_total, _ = mc_objective(dv1_mag, dv1_direction, sigma_dv1_abs, N_samples, seed=42)
        return mean_total
    
    # Optimization bounds: allow ±50% variation from Hohmann
    bounds = [(dv1_nominal_mag * 0.5, dv1_nominal_mag * 1.5)]
    
    from scipy.optimize import minimize
    
    result = minimize(
        objective_wrapper,
        x0=[dv1_nominal_mag],
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 50}
    )
    
    dv1_optimal_mag = result.x[0]
    print(f"\n  Optimization Result:")
    print(f"    Optimal ΔV1 = {dv1_optimal_mag:.4f} km/s")
    print(f"    Change from Hohmann: {(dv1_optimal_mag - dv1_nominal_mag)*1000:+.1f} m/s ({(dv1_optimal_mag/dv1_nominal_mag - 1)*100:+.2f}%)")
    
    # Evaluate optimal solution
    mean_dv_optimal, stats_opt = mc_objective(dv1_optimal_mag, dv1_direction, sigma_dv1_abs, N_samples, seed=42)
    
    # --- Comparison Table ---
    print(f"\n  --- Comparison: Initial Guess vs Optimized ---")
    print(f"\n  {'Metric':<20} {'Hohmann (Init)':<18} {'Optimized':<18} {'Improvement':<18}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18}")
    print(f"  {'Nominal ΔV1':<20} {dv1_nominal_mag:>14.4f} km/s {dv1_optimal_mag:>14.4f} km/s {(dv1_optimal_mag-dv1_nominal_mag)*1000:>+14.1f} m/s")
    print(f"  {'E[ΔV_init]':<20} {stats_init['dv_init_mean']:>14.4f} km/s {stats_opt['dv_init_mean']:>14.4f} km/s {(stats_opt['dv_init_mean']-stats_init['dv_init_mean'])*1000:>+14.1f} m/s")
    print(f"  {'E[ΔV_mid]':<20} {stats_init['dv_mid_mean']:>14.4f} km/s {stats_opt['dv_mid_mean']:>14.4f} km/s {(stats_opt['dv_mid_mean']-stats_init['dv_mid_mean'])*1000:>+14.1f} m/s")
    print(f"  {'E[ΔV_final]':<20} {stats_init['dv_final_mean']:>14.4f} km/s {stats_opt['dv_final_mean']:>14.4f} km/s {(stats_opt['dv_final_mean']-stats_init['dv_final_mean'])*1000:>+14.1f} m/s")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18}")
    print(f"  {'E[ΔV_total]':<20} {stats_init['dv_total_mean']:>14.4f} km/s {stats_opt['dv_total_mean']:>14.4f} km/s {(stats_opt['dv_total_mean']-stats_init['dv_total_mean'])*1000:>+14.1f} m/s")
    print(f"  {'Std[ΔV_total]':<20} {stats_init['dv_total_std']:>14.4f} km/s {stats_opt['dv_total_std']:>14.4f} km/s {(stats_opt['dv_total_std']-stats_init['dv_total_std'])*1000:>+14.1f} m/s")
    
    # Store for plotting
    mc_r_mid = np.array(stats_opt['r_mid_list'])
    avg_r_mid = np.mean(mc_r_mid, axis=0)

    # ===== DIRECTION VARIATION STUDY (Separate from Magnitude) =====
    print(f"\n" + "="*70)
    print(f"=== DIRECTION VARIATION STUDY (Magnitude Fixed) ===")
    print(f"="*70)
    
    # Parameters - 1σ pointing error per axis
    sigma_dir_deg = 10.0  # degrees (1σ per axis)
    N_dir_samples = 1000
    
    print(f"\n  Direction Error Model:")
    print(f"    Distribution: 2D Gaussian on direction vector")
    print(f"    1σ per axis: {sigma_dir_deg}°")
    print(f"    Cone angle follows Rayleigh distribution (mode ≈ σ)")
    print(f"    Magnitude: FIXED at Hohmann value ({dv1_nominal_mag:.4f} km/s)")
    print(f"    Number of MC samples: {N_dir_samples}")
    
    # Run direction variation study
    mean_dv_dir, stats_dir = mc_objective_direction(
        dv1_nominal_mag, dv1_direction, sigma_dir_deg, N_dir_samples, seed=42
    )
    
    print(f"\n  Direction Variation Results:")
    print(f"    Valid samples: {stats_dir['n_valid']}/{N_dir_samples}")
    print(f"    Cone angle stats: mean = {stats_dir['cone_angle_mean']:.2f}°, "
          f"std = {stats_dir['cone_angle_std']:.2f}°, max = {stats_dir['cone_angle_max']:.2f}°")
    
    print(f"\n  --- Comparison: Nominal vs Direction Variation ---")
    print(f"\n  {'Metric':<20} {'Nominal (no error)':<18} {'Direction Var':<18} {'Change':<18}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18}")
    print(f"  {'ΔV_init':<20} {dv_init_direct:>14.4f} km/s {stats_dir['dv_init_mean']:>14.4f} km/s {(stats_dir['dv_init_mean']-dv_init_direct)*1000:>+14.1f} m/s")
    print(f"  {'E[ΔV_mid]':<20} {dv_mid_direct:>14.4f} km/s {stats_dir['dv_mid_mean']:>14.4f} km/s {(stats_dir['dv_mid_mean']-dv_mid_direct)*1000:>+14.1f} m/s")
    print(f"  {'E[ΔV_final]':<20} {dv_final_direct:>14.4f} km/s {stats_dir['dv_final_mean']:>14.4f} km/s {(stats_dir['dv_final_mean']-dv_final_direct)*1000:>+14.1f} m/s")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18}")
    print(f"  {'E[ΔV_total]':<20} {dv_total_direct:>14.4f} km/s {stats_dir['dv_total_mean']:>14.4f} km/s {(stats_dir['dv_total_mean']-dv_total_direct)*1000:>+14.1f} m/s")
    print(f"  {'Std[ΔV_total]':<20} {'N/A':>14}      {stats_dir['dv_total_std']:>14.4f} km/s")
    print(f"  {'Min[ΔV_total]':<20} {dv_total_direct:>14.4f} km/s {stats_dir['dv_total_min']:>14.4f} km/s")
    print(f"  {'Max[ΔV_total]':<20} {dv_total_direct:>14.4f} km/s {stats_dir['dv_total_max']:>14.4f} km/s")
    
    # Compare magnitude vs direction variation
    print(f"\n  --- Comparison: Magnitude vs Direction Variation ---")
    print(f"\n  {'Variation Type':<20} {'E[ΔV_mid]':<18} {'E[ΔV_final]':<18} {'E[ΔV_total]':<18}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*18}")
    print(f"  {'None (Nominal)':<20} {dv_mid_direct:>14.4f} km/s {dv_final_direct:>14.4f} km/s {dv_total_direct:>14.4f} km/s")
    print(f"  {'Magnitude (10%)':<20} {stats_init['dv_mid_mean']:>14.4f} km/s {stats_init['dv_final_mean']:>14.4f} km/s {stats_init['dv_total_mean']:>14.4f} km/s")
    print(f"  {f'Direction (σ={sigma_dir_deg}°)':<20} {stats_dir['dv_mid_mean']:>14.4f} km/s {stats_dir['dv_final_mean']:>14.4f} km/s {stats_dir['dv_total_mean']:>14.4f} km/s")
    
    # --- OPTIMIZE nominal ΔV1 DIRECTION to minimize E[ΔV_total] under DIRECTION errors ---
    print(f"\n--- Optimizing Nominal ΔV1 Direction to Minimize E[ΔV_total] (Direction Errors) ---")
    print(f"    Decision variable: Nominal direction (2 perturbations in perpendicular axes)")
    print(f"    Fixed: Magnitude = {dv1_nominal_mag:.4f} km/s (Hohmann)")
    
    # Build orthonormal basis for direction parameterization
    # dv1_direction is the nominal Hohmann direction
    nominal_dir = dv1_direction / np.linalg.norm(dv1_direction)
    if abs(nominal_dir[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])
    e1_opt = temp - np.dot(temp, nominal_dir) * nominal_dir
    e1_opt = e1_opt / np.linalg.norm(e1_opt)
    e2_opt = np.cross(nominal_dir, e1_opt)
    
    def perturb_direction_from_nominal(nom_dir, delta1_deg, delta2_deg):
        """
        Perturb any nominal direction by adding offsets in perpendicular directions.
        Builds local orthonormal basis around nom_dir and applies perturbations.
        """
        nom = nom_dir / np.linalg.norm(nom_dir)
        if abs(nom[0]) < 0.9:
            temp_vec = np.array([1, 0, 0])
        else:
            temp_vec = np.array([0, 1, 0])
        e1_local = temp_vec - np.dot(temp_vec, nom) * nom
        e1_local = e1_local / np.linalg.norm(e1_local)
        e2_local = np.cross(nom, e1_local)
        
        delta1_rad = np.radians(delta1_deg)
        delta2_rad = np.radians(delta2_deg)
        
        d_perturbed = nom + np.tan(delta1_rad) * e1_local + np.tan(delta2_rad) * e2_local
        d_perturbed = d_perturbed / np.linalg.norm(d_perturbed)
        return d_perturbed
    
    print(f"    Nominal direction: [{nominal_dir[0]:.6f}, {nominal_dir[1]:.6f}, {nominal_dir[2]:.6f}]")
    print(f"    e1 (perpendicular): [{e1_opt[0]:.6f}, {e1_opt[1]:.6f}, {e1_opt[2]:.6f}]")
    print(f"    e2 (perpendicular): [{e2_opt[0]:.6f}, {e2_opt[1]:.6f}, {e2_opt[2]:.6f}]")
    
    def perturb_to_direction(delta1_deg, delta2_deg):
        """
        Perturb nominal direction by adding offsets in perpendicular directions.
        
        d_new = normalize(nominal + tan(delta1)*e1 + tan(delta2)*e2)
        
        For small angles, tan(delta) ≈ delta in radians.
        """
        delta1_rad = np.radians(delta1_deg)
        delta2_rad = np.radians(delta2_deg)
        
        # Add perturbations and renormalize
        d_perturbed = nominal_dir + np.tan(delta1_rad) * e1_opt + np.tan(delta2_rad) * e2_opt
        d_perturbed = d_perturbed / np.linalg.norm(d_perturbed)
        return d_perturbed
    
    def objective_wrapper_dir(x):
        """Wrapper for optimizer - optimizes nominal direction, magnitude fixed."""
        delta1_deg, delta2_deg = x
        dv1_dir_opt = perturb_to_direction(delta1_deg, delta2_deg)
        mean_total, _ = mc_objective_direction(dv1_nominal_mag, dv1_dir_opt, sigma_dir_deg, N_dir_samples, seed=42)
        return mean_total
    
    # Test at nominal first
    mean_at_nominal, _ = mc_objective_direction(dv1_nominal_mag, nominal_dir, sigma_dir_deg, N_dir_samples, seed=42)
    print(f"\n    E[ΔV_total] at nominal (0,0): {mean_at_nominal:.4f} km/s")
    
    # Test a few points to verify function works
    for d1, d2 in [(5, 0), (0, 5), (-5, 0), (0, -5)]:
        test_dir = perturb_to_direction(d1, d2)
        mean_test, _ = mc_objective_direction(dv1_nominal_mag, test_dir, sigma_dir_deg, N_dir_samples, seed=42)
        angle_from_nom = np.degrees(np.arccos(np.clip(np.dot(test_dir, nominal_dir), -1, 1)))
        print(f"    E[ΔV_total] at ({d1:+.0f}°, {d2:+.0f}°): {mean_test:.4f} km/s  (angle from nom: {angle_from_nom:.1f}°)")
    
    # Optimization bounds: allow ±20° deviation from nominal direction
    bounds_dir = [(-20.0, 20.0), (-20.0, 20.0)]  # [delta1, delta2] in degrees
    
    result_dir = minimize(
        objective_wrapper_dir,
        x0=[0.0, 0.0],  # Start at nominal Hohmann direction
        method='L-BFGS-B',
        bounds=bounds_dir,
        options={'disp': True, 'maxiter': 50}
    )
    
    opt_delta1, opt_delta2 = result_dir.x
    dv1_optimal_dir = perturb_to_direction(opt_delta1, opt_delta2)
    
    # Compute angular offset from nominal
    cos_offset = np.clip(np.dot(dv1_optimal_dir, nominal_dir), -1, 1)
    optimal_offset_deg = np.degrees(np.arccos(cos_offset))
    
    print(f"\n  Optimization Result (Direction Variation):")
    print(f"    Optimal perturbation: δ1 = {opt_delta1:.2f}°, δ2 = {opt_delta2:.2f}°")
    print(f"    Total angular offset from Hohmann: {optimal_offset_deg:.2f}°")
    print(f"    Optimal direction: [{dv1_optimal_dir[0]:.6f}, {dv1_optimal_dir[1]:.6f}, {dv1_optimal_dir[2]:.6f}]")
    
    # Evaluate optimal solution for direction variation
    mean_dv_opt_dir, stats_opt_dir = mc_objective_direction(dv1_nominal_mag, dv1_optimal_dir, sigma_dir_deg, N_dir_samples, seed=42)
    
    print(f"    E[ΔV_total] at optimal: {mean_dv_opt_dir:.4f} km/s")
    print(f"    Improvement over nominal: {(mean_dv_opt_dir - mean_at_nominal)*1000:.1f} m/s")
    
    # Store direction variation r_mid for plotting
    mc_r_mid_dir = np.array(stats_dir['r_mid_list'])

    # ===== CASES 6-9: FREE MCC TIMING (Total flight time fixed) =====
    print(f"\n" + "="*90)
    print(f"=== CASES 6-9: FREE MCC TIMING (Total flight time = {tof:.1f} s fixed) ===")
    print(f"="*90)
    
    # Time bounds: MCC can occur between 10% and 90% of total flight time
    delta_t_min = -tof_1 * 0.8  # Can't go below 20% of nominal tof_1
    delta_t_max = tof_2 * 0.8   # Can't exceed 80% of tof_2
    
    print(f"\n  Nominal MCC timing: tof_1 = {tof_1:.1f} s, tof_2 = {tof_2:.1f} s")
    print(f"  Allowed delta_t range: [{delta_t_min:.1f}, {delta_t_max:.1f}] s")
    
    # --- Case 6: Hohmann + MCC (mag var) + free time ---
    print(f"\n--- Case 6: Hohmann + MCC (mag var) + Free MCC Timing ---")
    
    def objective_case6(x):
        delta_t = x[0]
        mean_dv, _ = mc_objective_free_time(dv1_nominal_mag, dv1_direction, sigma_dv1_abs, delta_t, N_samples, seed=42)
        return mean_dv
    
    # Evaluate at nominal timing first
    mean_case6_nom, _ = mc_objective_free_time(dv1_nominal_mag, dv1_direction, sigma_dv1_abs, 0.0, N_samples, seed=42)
    print(f"    E[ΔV_total] at Δt=0: {mean_case6_nom:.4f} km/s")
    
    result_case6 = minimize(
        objective_case6,
        x0=[0.0],
        method='L-BFGS-B',
        bounds=[(delta_t_min, delta_t_max)],
        options={'disp': True, 'maxiter': 50}
    )
    
    delta_t_opt_case6 = result_case6.x[0]
    mean_case6, stats_case6 = mc_objective_free_time(dv1_nominal_mag, dv1_direction, sigma_dv1_abs, delta_t_opt_case6, N_samples, seed=42)
    print(f"    Optimal Δt = {delta_t_opt_case6:.1f} s")
    print(f"    New tof_1 = {tof_1 + delta_t_opt_case6:.1f} s, tof_2 = {tof_2 - delta_t_opt_case6:.1f} s")
    print(f"    E[ΔV_total] = {mean_case6:.4f} km/s")
    
    # --- Case 7: Optimized mag + MCC (mag var) + free time ---
    print(f"\n--- Case 7: Optimized Mag + MCC (mag var) + Free MCC Timing ---")
    
    def objective_case7(x):
        dv1_mag, delta_t = x
        mean_dv, _ = mc_objective_free_time(dv1_mag, dv1_direction, sigma_dv1_abs, delta_t, N_samples, seed=42)
        return mean_dv
    
    # Use Case 6's optimal timing as initial guess (not 0!)
    result_case7 = minimize(
        objective_case7,
        x0=[dv1_nominal_mag, delta_t_opt_case6],  # Start from Case 6's optimal timing
        method='L-BFGS-B',
        bounds=[(dv1_nominal_mag * 0.5, dv1_nominal_mag * 1.5), (delta_t_min, delta_t_max)],
        options={'disp': True, 'maxiter': 50}
    )
    
    dv1_opt_case7, delta_t_opt_case7 = result_case7.x
    mean_case7, stats_case7 = mc_objective_free_time(dv1_opt_case7, dv1_direction, sigma_dv1_abs, delta_t_opt_case7, N_samples, seed=42)
    print(f"    Optimal ΔV1 = {dv1_opt_case7:.4f} km/s, Δt = {delta_t_opt_case7:.1f} s")
    print(f"    New tof_1 = {tof_1 + delta_t_opt_case7:.1f} s, tof_2 = {tof_2 - delta_t_opt_case7:.1f} s")
    print(f"    E[ΔV_total] = {mean_case7:.4f} km/s")
    
    # --- Case 8: Hohmann + MCC (dir var) + free time ---
    print(f"\n--- Case 8: Hohmann + MCC (dir var) + Free MCC Timing ---")
    
    def objective_case8(x):
        delta_t = x[0]
        mean_dv, _ = mc_objective_direction_free_time(dv1_nominal_mag, dv1_direction, sigma_dir_deg, delta_t, N_dir_samples, seed=42)
        return mean_dv
    
    # Evaluate at nominal timing first
    mean_case8_nom, _ = mc_objective_direction_free_time(dv1_nominal_mag, dv1_direction, sigma_dir_deg, 0.0, N_dir_samples, seed=42)
    print(f"    E[ΔV_total] at Δt=0: {mean_case8_nom:.4f} km/s")
    
    # Test a few timing values to find a good starting point
    test_dt_vals = [0.0, delta_t_min/2, delta_t_min, delta_t_max/4]
    best_dt_init = 0.0
    best_val = mean_case8_nom
    for dt_test in test_dt_vals:
        val, _ = mc_objective_direction_free_time(dv1_nominal_mag, dv1_direction, sigma_dir_deg, dt_test, N_dir_samples, seed=42)
        print(f"    E[ΔV_total] at Δt={dt_test:.0f}: {val:.4f} km/s")
        if val < best_val:
            best_val = val
            best_dt_init = dt_test
    
    result_case8 = minimize(
        objective_case8,
        x0=[best_dt_init],  # Start from best test point
        method='L-BFGS-B',
        bounds=[(delta_t_min, delta_t_max)],
        options={'disp': True, 'maxiter': 50}
    )
    
    delta_t_opt_case8 = result_case8.x[0]
    mean_case8, stats_case8 = mc_objective_direction_free_time(dv1_nominal_mag, dv1_direction, sigma_dir_deg, delta_t_opt_case8, N_dir_samples, seed=42)
    print(f"    Optimal Δt = {delta_t_opt_case8:.1f} s")
    print(f"    New tof_1 = {tof_1 + delta_t_opt_case8:.1f} s, tof_2 = {tof_2 - delta_t_opt_case8:.1f} s")
    print(f"    E[ΔV_total] = {mean_case8:.4f} km/s")
    
    # --- Case 9: Optimized dir + MCC (dir var) + free time ---
    print(f"\n--- Case 9: Optimized Dir + MCC (dir var) + Free MCC Timing ---")
    
    def objective_case9(x):
        delta1_deg, delta2_deg, delta_t = x
        dv1_dir_opt = perturb_to_direction(delta1_deg, delta2_deg)
        mean_dv, _ = mc_objective_direction_free_time(dv1_nominal_mag, dv1_dir_opt, sigma_dir_deg, delta_t, N_dir_samples, seed=42)
        return mean_dv
    
    # Use Case 5's optimal direction and Case 8's optimal timing as starting point
    result_case9 = minimize(
        objective_case9,
        x0=[opt_delta1, opt_delta2, delta_t_opt_case8],  # Start from Case 5 direction + Case 8 timing
        method='L-BFGS-B',
        bounds=[(-20.0, 20.0), (-20.0, 20.0), (delta_t_min, delta_t_max)],
        options={'disp': True, 'maxiter': 50}
    )
    
    opt_d1_case9, opt_d2_case9, delta_t_opt_case9 = result_case9.x
    dv1_optimal_dir_case9 = perturb_to_direction(opt_d1_case9, opt_d2_case9)
    cos_offset_case9 = np.clip(np.dot(dv1_optimal_dir_case9, nominal_dir), -1, 1)
    optimal_offset_deg_case9 = np.degrees(np.arccos(cos_offset_case9))
    
    mean_case9, stats_case9 = mc_objective_direction_free_time(dv1_nominal_mag, dv1_optimal_dir_case9, sigma_dir_deg, delta_t_opt_case9, N_dir_samples, seed=42)
    print(f"    Optimal dir offset: δ1={opt_d1_case9:.2f}°, δ2={opt_d2_case9:.2f}° ({optimal_offset_deg_case9:.2f}° total)")
    print(f"    Optimal Δt = {delta_t_opt_case9:.1f} s")
    print(f"    New tof_1 = {tof_1 + delta_t_opt_case9:.1f} s, tof_2 = {tof_2 - delta_t_opt_case9:.1f} s")
    print(f"    E[ΔV_total] = {mean_case9:.4f} km/s")

    # ===== SUMMARY TABLE: ALL 9 CASES =====
    print(f"\n" + "="*110)
    print(f"=== SUMMARY: ALL 9 CASES ===")
    print(f"="*110)
    
    # Case 1: Hohmann (no randomness)
    case1_init = dv_init_direct
    case1_mid = dv_mid_direct
    case1_final = dv_final_direct
    case1_total = dv_total_direct
    
    # Case 2: Hohmann + MCC due to mag var
    case2_init = stats_init['dv_init_mean']
    case2_mid = stats_init['dv_mid_mean']
    case2_final = stats_init['dv_final_mean']
    case2_total = stats_init['dv_total_mean']
    
    # Case 3: Hohmann + MCC due to mag var + optim
    case3_init = stats_opt['dv_init_mean']
    case3_mid = stats_opt['dv_mid_mean']
    case3_final = stats_opt['dv_final_mean']
    case3_total = stats_opt['dv_total_mean']
    
    # Case 4: Hohmann + MCC due to dir var
    case4_init = stats_dir['dv_init_mean']
    case4_mid = stats_dir['dv_mid_mean']
    case4_final = stats_dir['dv_final_mean']
    case4_total = stats_dir['dv_total_mean']
    
    # Case 5: Hohmann + MCC due to dir var + optim
    case5_init = stats_opt_dir['dv_init_mean']
    case5_mid = stats_opt_dir['dv_mid_mean']
    case5_final = stats_opt_dir['dv_final_mean']
    case5_total = stats_opt_dir['dv_total_mean']
    
    # Case 6: Hohmann + MCC (mag var) + free time
    case6_init = stats_case6['dv_init_mean']
    case6_mid = stats_case6['dv_mid_mean']
    case6_final = stats_case6['dv_final_mean']
    case6_total = stats_case6['dv_total_mean']
    
    # Case 7: Optimized + MCC (mag var) + free time
    case7_init = stats_case7['dv_init_mean']
    case7_mid = stats_case7['dv_mid_mean']
    case7_final = stats_case7['dv_final_mean']
    case7_total = stats_case7['dv_total_mean']
    
    # Case 8: Hohmann + MCC (dir var) + free time
    case8_init = stats_case8['dv_init_mean']
    case8_mid = stats_case8['dv_mid_mean']
    case8_final = stats_case8['dv_final_mean']
    case8_total = stats_case8['dv_total_mean']
    
    # Case 9: Optimized + MCC (dir var) + free time
    case9_init = stats_case9['dv_init_mean']
    case9_mid = stats_case9['dv_mid_mean']
    case9_final = stats_case9['dv_final_mean']
    case9_total = stats_case9['dv_total_mean']
    
    print(f"\n  {'Case':<55} {'ΔV_init':>10} {'ΔV_mid':>10} {'ΔV_final':>10} {'ΔV_total':>10} {'Δt_mid':>12}")
    print(f"  {'-'*55} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    print(f"  {'1. Hohmann (deterministic)':<55} {case1_init:>8.4f}   {case1_mid:>8.4f}   {case1_final:>8.4f}   {case1_total:>8.4f}   {'N/A':>10}")
    print(f"  {'--- Fixed MCC timing (tof_1, tof_2 fixed) ---':<55}")
    print(f"  {'2. Hohmann + MCC (mag var σ=10%)':<55} {case2_init:>8.4f}   {case2_mid:>8.4f}   {case2_final:>8.4f}   {case2_total:>8.4f}   {0:>10.1f} s")
    print(f"  {'3. Opt mag + MCC (mag var σ=10%)':<55} {case3_init:>8.4f}   {case3_mid:>8.4f}   {case3_final:>8.4f}   {case3_total:>8.4f}   {0:>10.1f} s")
    print(f"  {f'4. Hohmann + MCC (dir var σ={sigma_dir_deg}°)':<55} {case4_init:>8.4f}   {case4_mid:>8.4f}   {case4_final:>8.4f}   {case4_total:>8.4f}   {0:>10.1f} s")
    print(f"  {f'5. Opt dir + MCC (dir var σ={sigma_dir_deg}°)':<55} {case5_init:>8.4f}   {case5_mid:>8.4f}   {case5_final:>8.4f}   {case5_total:>8.4f}   {0:>10.1f} s")
    print(f"  {'--- Free MCC timing (total tof fixed) ---':<55}")
    print(f"  {'6. Hohmann + MCC (mag var) + free time':<55} {case6_init:>8.4f}   {case6_mid:>8.4f}   {case6_final:>8.4f}   {case6_total:>8.4f}   {delta_t_opt_case6:>+10.1f} s")
    print(f"  {'7. Opt mag + MCC (mag var) + free time':<55} {case7_init:>8.4f}   {case7_mid:>8.4f}   {case7_final:>8.4f}   {case7_total:>8.4f}   {delta_t_opt_case7:>+10.1f} s")
    print(f"  {f'8. Hohmann + MCC (dir var σ={sigma_dir_deg}°) + free time':<55} {case8_init:>8.4f}   {case8_mid:>8.4f}   {case8_final:>8.4f}   {case8_total:>8.4f}   {delta_t_opt_case8:>+10.1f} s")
    print(f"  {f'9. Opt dir + MCC (dir var σ={sigma_dir_deg}°) + free time':<55} {case9_init:>8.4f}   {case9_mid:>8.4f}   {case9_final:>8.4f}   {case9_total:>8.4f}   {delta_t_opt_case9:>+10.1f} s")
    print(f"  {'-'*55} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    
    # Savings summary
    print(f"\n  Optimization Savings (ΔV_total):")
    print(f"    Fixed timing:")
    print(f"      Mag var: Case 3 vs Case 2 = {(case3_total - case2_total)*1000:+.1f} m/s ({(case3_total/case2_total - 1)*100:+.2f}%)")
    print(f"      Dir var: Case 5 vs Case 4 = {(case5_total - case4_total)*1000:+.1f} m/s ({(case5_total/case4_total - 1)*100:+.2f}%)")
    print(f"    Free timing:")
    print(f"      Mag var: Case 7 vs Case 6 = {(case7_total - case6_total)*1000:+.1f} m/s ({(case7_total/case6_total - 1)*100:+.2f}%)")
    print(f"      Dir var: Case 9 vs Case 8 = {(case9_total - case8_total)*1000:+.1f} m/s ({(case9_total/case8_total - 1)*100:+.2f}%)")
    print(f"    Free time benefit (at Hohmann):")
    print(f"      Mag var: Case 6 vs Case 2 = {(case6_total - case2_total)*1000:+.1f} m/s ({(case6_total/case2_total - 1)*100:+.2f}%)")
    print(f"      Dir var: Case 8 vs Case 4 = {(case8_total - case4_total)*1000:+.1f} m/s ({(case8_total/case4_total - 1)*100:+.2f}%)")
    
    print(f"\n  Optimal Decision Variables:")
    print(f"    Cases 3,7 (Mag var): ΔV1 = {dv1_optimal_mag:.4f} km/s (fixed), {dv1_opt_case7:.4f} km/s (free time)")
    print(f"    Cases 5,9 (Dir var): offset = {optimal_offset_deg:.2f}° (fixed), {optimal_offset_deg_case9:.2f}° (free time)")
    print(f"    Optimal Δt_mid: Case 6={delta_t_opt_case6:.1f}s, Case 7={delta_t_opt_case7:.1f}s, Case 8={delta_t_opt_case8:.1f}s, Case 9={delta_t_opt_case9:.1f}s")

    # ==========================================================================
    # STOCHASTIC STM ANALYSIS
    # ==========================================================================
    print("\n" + "="*100)
    print("=== STOCHASTIC STM ANALYSIS: Sensitivity of Position Error to ΔV1 Errors ===")
    print("="*100)
    
    # Compute STM analysis for different MCC timings
    v_after_burn1 = v_init_orbit + dv1_nominal_mag * dv1_direction
    
    print(f"\n  Initial state (after burn 1):")
    print(f"    r₀ = {r_init} km")
    print(f"    v₀ = {v_after_burn1} km/s")
    print(f"    |v₀| = {np.linalg.norm(v_after_burn1):.4f} km/s")
    
    # Analyze at three different MCC timings
    timing_cases = [
        ("Nominal (EA=90°)", tof_1, 0.0),
        ("Case 6/7/8 (early)", tof_1 + delta_t_opt_case6, delta_t_opt_case6),
        ("Case 9 (intermediate)", tof_1 + delta_t_opt_case9, delta_t_opt_case9),
    ]
    
    print(f"\n  Covariance Analysis at Different MCC Timings:")
    print(f"  (σ_mag = {sigma_dv1_abs*1000:.1f} m/s, σ_dir = {sigma_dir_deg}°/axis)")
    print(f"\n  {'Timing':<25} {'tof_1 (s)':<12} {'σ_r_mag (km)':<14} {'σ_r_dir (km)':<14} {'σ_r_total (km)':<14}")
    print(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    
    stm_results = {}
    for name, dt, delta_t in timing_cases:
        result = stochastic_stm_analysis(r_init, v_init_orbit, v_after_burn1, dt, 
                                         sigma_dv1_abs, sigma_dir_deg)
        stm_results[name] = result
        print(f"  {name:<25} {dt:<12.1f} {result['rms_r_mag']:<14.2f} {result['rms_r_dir']:<14.2f} {result['rms_r_total']:<14.2f}")
    
    # Detailed analysis for nominal timing
    print(f"\n  --- Detailed STM Analysis at Nominal Timing (tof_1 = {tof_1:.1f} s) ---")
    nom_result = stm_results["Nominal (EA=90°)"]
    
    print(f"\n  Φ_rv (Position sensitivity to velocity):")
    print(f"    Maps δv₀ to δr(t)")
    for i in range(3):
        print(f"    [{nom_result['STM_Phi_rv'][i,0]:12.4f} {nom_result['STM_Phi_rv'][i,1]:12.4f} {nom_result['STM_Phi_rv'][i,2]:12.4f}]")
    
    print(f"\n  Position Covariance P_r (km²):")
    print(f"    From magnitude error:")
    for i in range(3):
        print(f"    [{nom_result['P_r_mag'][i,0]:12.2f} {nom_result['P_r_mag'][i,1]:12.2f} {nom_result['P_r_mag'][i,2]:12.2f}]")
    print(f"    From direction error:")
    for i in range(3):
        print(f"    [{nom_result['P_r_dir'][i,0]:12.2f} {nom_result['P_r_dir'][i,1]:12.2f} {nom_result['P_r_dir'][i,2]:12.2f}]")
    
    print(f"\n  Position 1σ Dispersion (km):")
    print(f"    {'Component':<10} {'Mag Error':<14} {'Dir Error':<14} {'Total':<14}")
    print(f"    {'-'*10} {'-'*14} {'-'*14} {'-'*14}")
    print(f"    {'X':<10} {nom_result['sigma_r_mag'][0]:<14.2f} {nom_result['sigma_r_dir'][0]:<14.2f} {nom_result['sigma_r_total'][0]:<14.2f}")
    print(f"    {'Y':<10} {nom_result['sigma_r_mag'][1]:<14.2f} {nom_result['sigma_r_dir'][1]:<14.2f} {nom_result['sigma_r_total'][1]:<14.2f}")
    print(f"    {'Z':<10} {nom_result['sigma_r_mag'][2]:<14.2f} {nom_result['sigma_r_dir'][2]:<14.2f} {nom_result['sigma_r_total'][2]:<14.2f}")
    print(f"    {'RMS':<10} {nom_result['rms_r_mag']:<14.2f} {nom_result['rms_r_dir']:<14.2f} {nom_result['rms_r_total']:<14.2f}")
    
    # Compare variance contributions
    var_mag = nom_result['rms_r_mag']**2
    var_dir = nom_result['rms_r_dir']**2
    var_total = var_mag + var_dir
    
    print(f"\n  Variance Contribution:")
    print(f"    Magnitude error: {var_mag:.1f} km² ({var_mag/var_total*100:.1f}%)")
    print(f"    Direction error: {var_dir:.1f} km² ({var_dir/var_total*100:.1f}%)")
    
    # --- Timing sensitivity ---
    print(f"\n  --- Sensitivity to MCC Timing ---")
    dt_test = np.linspace(500, tof_1 + 2000, 20)
    rms_r_vs_dt = []
    for dt in dt_test:
        res = stochastic_stm_analysis(r_init, v_init_orbit, v_after_burn1, dt, 
                                      sigma_dv1_abs, sigma_dir_deg)
        rms_r_vs_dt.append(res['rms_r_total'])
    
    # Find minimum (though Lambert cost isn't just position error)
    min_idx = np.argmin(rms_r_vs_dt)
    print(f"    Minimum RMS position error at tof_1 = {dt_test[min_idx]:.0f} s (σ_r = {rms_r_vs_dt[min_idx]:.1f} km)")
    print(f"    Note: This differs from optimal ΔV timing because Lambert cost ≠ position error")
    
    # --- Bar Plot: ΔV Total Comparison (All 9 cases) ---
    fig_bar, ax_bar = plt.subplots(figsize=(14, 6))
    
    # Total ΔV for all 9 cases
    cases = ['1\nHohmann', '2\nMag', '3\nMag+Opt', '4\nDir', '5\nDir+Opt', 
             '6\nMag+FT', '7\nMag+Opt+FT', '8\nDir+FT', '9\nDir+Opt+FT']
    dv_totals = [case1_total, case2_total, case3_total, case4_total, case5_total,
                 case6_total, case7_total, case8_total, case9_total]
    
    colors = ['green', 'orange', 'darkorange', 'purple', 'darkviolet',
              'lightsalmon', 'coral', 'plum', 'mediumorchid']
    
    bars = ax_bar.bar(cases, dv_totals, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, dv in zip(bars, dv_totals):
        height = bar.get_height()
        ax_bar.annotate(f'{height:.3f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add horizontal line for Hohmann reference
    ax_bar.axhline(y=case1_total, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Hohmann')
    
    ax_bar.set_xlabel('Case')
    ax_bar.set_ylabel('E[ΔV_total] (km/s)')
    ax_bar.set_title(f'Total ΔV Comparison: All 9 Cases\n(Mag var σ=10%, Dir var σ={sigma_dir_deg}°/axis, FT=Free MCC Timing)')
    ax_bar.set_ylim(0, max(dv_totals) * 1.15)
    ax_bar.grid(axis='y', alpha=0.3)
    ax_bar.legend(loc='upper right')
    
    plt.tight_layout()
    
    # --- STM Position Error vs Timing Plot ---
    fig_stm, (ax_stm1, ax_stm2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute for full range
    dt_range = np.linspace(200, tof, 50)
    rms_mag_vs_dt = []
    rms_dir_vs_dt = []
    for dt in dt_range:
        res = stochastic_stm_analysis(r_init, v_init_orbit, v_after_burn1, dt, 
                                      sigma_dv1_abs, sigma_dir_deg)
        rms_mag_vs_dt.append(res['rms_r_mag'])
        rms_dir_vs_dt.append(res['rms_r_dir'])
    
    # Plot 1: Position dispersion vs timing
    ax_stm1.plot(dt_range/60, rms_mag_vs_dt, 'orange', linewidth=2, label=f'Magnitude (σ={sigma_dv1_abs*1000:.0f} m/s)')
    ax_stm1.plot(dt_range/60, rms_dir_vs_dt, 'purple', linewidth=2, label=f'Direction (σ={sigma_dir_deg}°/axis)')
    ax_stm1.plot(dt_range/60, np.sqrt(np.array(rms_mag_vs_dt)**2 + np.array(rms_dir_vs_dt)**2), 
                 'k--', linewidth=2, label='Total')
    
    # Mark key timings
    ax_stm1.axvline(tof_1/60, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Nominal MCC ({tof_1/60:.1f} min)')
    ax_stm1.axvline((tof_1 + delta_t_opt_case6)/60, color='red', linestyle=':', linewidth=1.5, alpha=0.7, 
                   label=f'Case 6/7/8 MCC ({(tof_1+delta_t_opt_case6)/60:.1f} min)')
    ax_stm1.axvline((tof_1 + delta_t_opt_case9)/60, color='blue', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Case 9 MCC ({(tof_1+delta_t_opt_case9)/60:.1f} min)')
    
    ax_stm1.set_xlabel('Time to MCC (min)')
    ax_stm1.set_ylabel('RMS Position Dispersion (km)')
    ax_stm1.set_title('Position Dispersion at MCC vs Timing\n(STM Covariance Propagation)')
    ax_stm1.legend(fontsize=8)
    ax_stm1.grid(alpha=0.3)
    ax_stm1.set_xlim([0, tof/60])
    
    # Plot 2: Variance contribution pie chart at nominal timing
    labels = ['Magnitude Error', 'Direction Error']
    sizes = [var_mag, var_dir]
    colors_pie = ['orange', 'purple']
    explode = (0.02, 0.02)
    
    ax_stm2.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                shadow=True, startangle=90)
    ax_stm2.set_title(f'Position Variance Contribution at Nominal MCC\n(tof_1 = {tof_1:.0f} s)')
    
    plt.tight_layout()
    plt.savefig('stm_analysis.png', dpi=150)
    plt.savefig('dv_comparison.png', dpi=150)

    # --- Histogram Plot: Error Distributions ---
    fig_hist, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Magnitude Error Distribution
    ax1 = axes[0]
    mag_errors_ms = stats_init['mag_errors'] * 1000  # Convert to m/s
    n_bins = 30
    
    # Plot histogram
    n, bins, patches = ax1.hist(mag_errors_ms, bins=n_bins, density=True, alpha=0.7, 
                                 color='orange', edgecolor='black', label='Sampled')
    
    # Overlay theoretical Gaussian
    x_gauss = np.linspace(mag_errors_ms.min(), mag_errors_ms.max(), 100)
    sigma_ms = sigma_dv1_abs * 1000
    gaussian_pdf = (1 / (sigma_ms * np.sqrt(2*np.pi))) * np.exp(-x_gauss**2 / (2*sigma_ms**2))
    ax1.plot(x_gauss, gaussian_pdf, 'k-', linewidth=2, label=f'Gaussian (σ={sigma_ms:.1f} m/s)')
    
    ax1.axvline(0, color='green', linestyle='--', linewidth=1.5, label='Nominal')
    ax1.axvline(np.mean(mag_errors_ms), color='red', linestyle='-', linewidth=1.5, 
                label=f'Mean = {np.mean(mag_errors_ms):.1f} m/s')
    
    ax1.set_xlabel('Magnitude Error (m/s)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'ΔV₁ Magnitude Error Distribution\n(Gaussian, σ = {sigma_ms:.1f} m/s)')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # Right: Direction Angle Distribution (Cone Angle) - Rayleigh
    ax2 = axes[1]
    cone_angles = stats_dir['cone_angles']
    
    # Plot histogram
    n2, bins2, patches2 = ax2.hist(cone_angles, bins=n_bins, density=True, alpha=0.7,
                                    color='purple', edgecolor='black', label='Sampled')
    
    # Overlay theoretical Rayleigh distribution
    # For 2D Gaussian with sigma_deg per axis, cone angle follows Rayleigh(sigma_deg)
    # PDF: p(θ) = (θ/σ²) * exp(-θ²/(2σ²))
    x_theta = np.linspace(0, np.max(cone_angles) * 1.1, 100)
    rayleigh_pdf = (x_theta / sigma_dir_deg**2) * np.exp(-x_theta**2 / (2 * sigma_dir_deg**2))
    ax2.plot(x_theta, rayleigh_pdf, 'k-', linewidth=2, 
             label=f'Rayleigh (σ={sigma_dir_deg}°)')
    
    ax2.axvline(np.mean(cone_angles), color='red', linestyle='-', linewidth=1.5,
                label=f'Mean = {np.mean(cone_angles):.1f}°')
    ax2.axvline(sigma_dir_deg, color='green', linestyle='--', linewidth=1.5,
                label=f'Mode = σ = {sigma_dir_deg}°')
    
    ax2.set_xlabel('Cone Angle (degrees)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title(f'ΔV₁ Direction Error Distribution\n(2D Gaussian → Rayleigh cone angle, σ = {sigma_dir_deg}°/axis)')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, np.max(cone_angles) * 1.1)
    
    plt.tight_layout()
    plt.savefig('error_distributions.png', dpi=150)

    # --- Unit Sphere Plot: Direction Variation ---
    fig_sphere = plt.figure(figsize=(10, 10))
    ax_sphere = fig_sphere.add_subplot(111, projection='3d')
    
    # Get sampled directions from direction variation study
    # Re-sample to get the actual direction vectors
    np.random.seed(42)
    sampled_directions, _ = sample_gaussian_direction(dv1_direction, sigma_dir_deg, N_dir_samples)
    
    # Plot unit sphere (wireframe)
    u_sphere = np.linspace(0, 2 * np.pi, 30)
    v_sphere = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    ax_sphere.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.3, linewidth=0.5)
    
    # Plot sampled directions as scatter points on unit sphere
    ax_sphere.scatter(sampled_directions[:, 0], sampled_directions[:, 1], sampled_directions[:, 2],
                      c='purple', s=10, alpha=0.5, label='Sampled directions')
    
    # Plot nominal Hohmann direction
    ax_sphere.scatter(*dv1_direction, color='green', s=200, marker='*', zorder=10,
                      edgecolor='black', linewidth=1.5, label='Nominal Hohmann')
    
    # Plot optimal direction (from direction optimization)
    ax_sphere.scatter(*dv1_optimal_dir, color='red', s=200, marker='*', zorder=10,
                      edgecolor='black', linewidth=1.5, label=f'Optimal ({optimal_offset_deg:.1f}° from nom)')
    
    # Draw line connecting nominal to optimal
    ax_sphere.plot([dv1_direction[0], dv1_optimal_dir[0]], 
                   [dv1_direction[1], dv1_optimal_dir[1]], 
                   [dv1_direction[2], dv1_optimal_dir[2]], 
                   'k--', linewidth=2, alpha=0.7)
    
    # Build orthonormal basis for drawing circles
    nominal_dir = dv1_direction / np.linalg.norm(dv1_direction)
    if abs(nominal_dir[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])
    e1 = temp - np.dot(temp, nominal_dir) * nominal_dir
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(nominal_dir, e1)
    
    # Draw circles at 1σ, 2σ, 3σ cone angles
    phi_circle = np.linspace(0, 2*np.pi, 100)
    sigma_levels = [1, 2, 3]
    colors_sigma = ['blue', 'orange', 'red']
    
    for sigma_mult, color in zip(sigma_levels, colors_sigma):
        theta_rad = np.radians(sigma_mult * sigma_dir_deg)
        sin_theta = np.sin(theta_rad)
        cos_theta = np.cos(theta_rad)
        
        # Circle on unit sphere at cone angle theta from nominal direction
        circle_x = sin_theta * np.cos(phi_circle)
        circle_y = sin_theta * np.sin(phi_circle)
        circle_z = cos_theta * np.ones_like(phi_circle)
        
        # Transform to global coordinates
        circle_points = np.zeros((len(phi_circle), 3))
        for i in range(len(phi_circle)):
            circle_points[i] = circle_x[i] * e1 + circle_y[i] * e2 + circle_z[i] * nominal_dir
        
        ax_sphere.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2],
                       color=color, linewidth=2, label=f'{sigma_mult}σ = {sigma_mult * sigma_dir_deg}°')
    
    ax_sphere.set_xlabel('X')
    ax_sphere.set_ylabel('Y')
    ax_sphere.set_zlabel('Z')
    ax_sphere.set_title(f'Direction Variation on Unit Sphere\n(σ = {sigma_dir_deg}°/axis, Optimal offset = {optimal_offset_deg:.1f}°)')
    ax_sphere.legend(loc='upper left')
    
    # Equal aspect ratio
    ax_sphere.set_xlim([-1.1, 1.1])
    ax_sphere.set_ylim([-1.1, 1.1])
    ax_sphere.set_zlim([-1.1, 1.1])
    ax_sphere.set_box_aspect([1, 1, 1])
    
    # Set view angle to see the distribution well
    ax_sphere.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('direction_sphere.png', dpi=150)

    # 3D Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot orbits
    ax.plot(traj['initial'][:, 0], traj['initial'][:, 1], traj['initial'][:, 2], 
            'b-', linewidth=1.5, label=f'Initial Orbit (i={inc1}°)')
    ax.plot(traj['transfer'][:, 0], traj['transfer'][:, 1], traj['transfer'][:, 2], 
            'g-', linewidth=2, label='Nominal Transfer')
    ax.plot(traj['final'][:, 0], traj['final'][:, 1], traj['final'][:, 2], 
            'r-', linewidth=1.5, label=f'Final Orbit (i={inc2}°)')
    
    # Plot MC mid-point scatter - Magnitude variation (orange)
    ax.scatter(mc_r_mid[:, 0], mc_r_mid[:, 1], mc_r_mid[:, 2], 
               color='orange', s=15, alpha=0.4, label=f'Mag Var r_mid')
    
    # Plot MC mid-point scatter - Direction variation (purple)
    ax.scatter(mc_r_mid_dir[:, 0], mc_r_mid_dir[:, 1], mc_r_mid_dir[:, 2], 
               color='purple', s=15, alpha=0.4, label=f'Dir Var r_mid')
    
    # Plot average r_mid
    ax.scatter(*avg_r_mid, color='red', s=150, marker='X', zorder=7, edgecolor='black',
               label=f'Avg r_mid (Mag Var)')
    
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
    
    # ==========================================================================
    # NEW: Cases 6-9 Free-time trajectory visualization
    # ==========================================================================
    print("\n" + "="*80)
    print("=== PLOTTING CASES 6-9: FREE MCC TIMING ===")
    print("="*80)
    
    # Compute nominal MCC positions for each case
    v1_nominal = dv1_nominal_mag * dv1_direction
    v_after_burn1 = v_init_orbit + v1_nominal  # Velocity after burn 1
    
    # Case 6 & 7: Same timing (Δt = delta_t_opt_case6), magnitude variation
    tof_1_case6 = tof_1 + delta_t_opt_case6
    r_mid_case6, _ = propagate_kepler(r_init, v_after_burn1, tof_1_case6)
    
    # Case 7: Optimized magnitude
    dv1_opt_case7 = result_case7.x[0]
    v1_case7 = dv1_opt_case7 * dv1_direction
    v_after_burn1_case7 = v_init_orbit + v1_case7
    tof_1_case7 = tof_1 + delta_t_opt_case7
    r_mid_case7, _ = propagate_kepler(r_init, v_after_burn1_case7, tof_1_case7)
    
    # Case 8: Same timing as Case 6, direction variation (Hohmann direction)
    tof_1_case8 = tof_1 + delta_t_opt_case8
    r_mid_case8, _ = propagate_kepler(r_init, v_after_burn1, tof_1_case8)
    
    # Case 9: Optimized direction and timing
    dv1_dir_opt_case9 = perturb_to_direction(opt_d1_case9, opt_d2_case9)
    v1_case9 = dv1_nominal_mag * dv1_dir_opt_case9
    v_after_burn1_case9 = v_init_orbit + v1_case9
    tof_1_case9 = tof_1 + delta_t_opt_case9
    r_mid_case9, _ = propagate_kepler(r_init, v_after_burn1_case9, tof_1_case9)
    
    print(f"  Case 6: tof_1 = {tof_1_case6:.1f} s, r_mid = {r_mid_case6}")
    print(f"  Case 7: tof_1 = {tof_1_case7:.1f} s, r_mid = {r_mid_case7}")
    print(f"  Case 8: tof_1 = {tof_1_case8:.1f} s, r_mid = {r_mid_case8}")
    print(f"  Case 9: tof_1 = {tof_1_case9:.1f} s, r_mid = {r_mid_case9}")
    
    # Generate partial transfer arcs for Cases 6-9
    # These show the trajectory from LEO to the new MCC position
    n_arc_pts = 100
    
    # Case 6 arc (Hohmann ΔV1, early MCC)
    arc_case6 = []
    for t in np.linspace(0, tof_1_case6, n_arc_pts):
        r_t, _ = propagate_kepler(r_init, v_after_burn1, t)
        arc_case6.append(r_t)
    arc_case6 = np.array(arc_case6)
    
    # Case 7 arc (Optimized ΔV1, early MCC)
    arc_case7 = []
    for t in np.linspace(0, tof_1_case7, n_arc_pts):
        r_t, _ = propagate_kepler(r_init, v_after_burn1_case7, t)
        arc_case7.append(r_t)
    arc_case7 = np.array(arc_case7)
    
    # Case 9 arc (Optimized direction, intermediate MCC)
    arc_case9 = []
    for t in np.linspace(0, tof_1_case9, n_arc_pts):
        r_t, _ = propagate_kepler(r_init, v_after_burn1_case9, t)
        arc_case9.append(r_t)
    arc_case9 = np.array(arc_case9)
    
    # ==========================================================================
    # 3D Plot: Cases 6-9 comparison
    # ==========================================================================
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot orbits
    ax.plot(traj['initial'][:, 0], traj['initial'][:, 1], traj['initial'][:, 2], 
            'b-', linewidth=1, alpha=0.5, label=f'LEO (i={inc1}°)')
    ax.plot(traj['transfer'][:, 0], traj['transfer'][:, 1], traj['transfer'][:, 2], 
            'gray', linewidth=1.5, linestyle='--', alpha=0.5, label='Nominal Transfer (EA=90°)')
    ax.plot(traj['final'][:, 0], traj['final'][:, 1], traj['final'][:, 2], 
            'r-', linewidth=1, alpha=0.5, label=f'GEO (i={inc2}°)')
    
    # Plot arcs to new MCC positions
    ax.plot(arc_case6[:, 0], arc_case6[:, 1], arc_case6[:, 2], 
            'orange', linewidth=2.5, label=f'Case 6: Δt={delta_t_opt_case6:.0f}s')
    ax.plot(arc_case7[:, 0], arc_case7[:, 1], arc_case7[:, 2], 
            'red', linewidth=2.5, linestyle='--', label=f'Case 7: Δt={delta_t_opt_case7:.0f}s, ΔV1={dv1_opt_case7:.3f}')
    ax.plot(arc_case9[:, 0], arc_case9[:, 1], arc_case9[:, 2], 
            'purple', linewidth=2.5, label=f'Case 9: Δt={delta_t_opt_case9:.0f}s, dir offset={np.sqrt(opt_d1_case9**2+opt_d2_case9**2):.1f}°')
    
    # Plot MCC positions
    ax.scatter(*r_EA90, color='green', s=150, marker='s', zorder=10, edgecolor='black', linewidth=2,
               label=f'Nominal MCC (EA=90°, tof_1={tof_1:.0f}s)')
    ax.scatter(*r_mid_case6, color='orange', s=150, marker='^', zorder=10, edgecolor='black', linewidth=2,
               label=f'Case 6/7/8 MCC (tof_1={tof_1_case6:.0f}s)')
    ax.scatter(*r_mid_case9, color='purple', s=150, marker='D', zorder=10, edgecolor='black', linewidth=2,
               label=f'Case 9 MCC (tof_1={tof_1_case9:.0f}s)')
    
    # Plot key positions
    ax.scatter(*r_EA0, color='blue', s=100, marker='o', zorder=5, edgecolor='black',
               label='Departure (periapsis)')
    ax.scatter(*r_EA180, color='red', s=100, marker='o', zorder=5, edgecolor='black',
               label='Arrival (apoapsis)')
    
    # Plot Earth
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R_earth = 6371
    x = R_earth * np.outer(np.cos(u), np.sin(v))
    y = R_earth * np.outer(np.sin(u), np.sin(v))
    z = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.6)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Cases 6-9: Free MCC Timing Optimization\n'
                 f'Nominal tof_1={tof_1:.0f}s → Optimized: Case 6/7/8={tof_1_case6:.0f}s, Case 9={tof_1_case9:.0f}s')
    ax.legend(loc='upper left', fontsize=9)
    
    # Equal aspect ratio
    max_range = sma2 * 0.7
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range/2, max_range/2])
    ax.set_box_aspect([1, 1, 0.5])
    
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    plt.savefig('cases_6_9_free_time.png', dpi=150)
    
    # ==========================================================================
    # Summary: MCC position comparison
    # ==========================================================================
    print("\n  MCC Position Comparison:")
    print(f"  {'Case':<12} {'tof_1 (s)':<12} {'|r_mid| (km)':<15} {'r_mid (km)':<45}")
    print(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*45}")
    print(f"  {'Nominal':<12} {tof_1:<12.1f} {np.linalg.norm(r_EA90):<15.1f} {str(np.round(r_EA90, 1)):<45}")
    print(f"  {'Case 6/7/8':<12} {tof_1_case6:<12.1f} {np.linalg.norm(r_mid_case6):<15.1f} {str(np.round(r_mid_case6, 1)):<45}")
    print(f"  {'Case 9':<12} {tof_1_case9:<12.1f} {np.linalg.norm(r_mid_case9):<15.1f} {str(np.round(r_mid_case9, 1)):<45}")
    
    print(f"\n  Key Insight: Earlier MCC (smaller tof_1) means:")
    print(f"    - Spacecraft closer to Earth (smaller |r_mid|)")
    print(f"    - Higher velocity → corrections cost less ΔV per km of position error")
    print(f"    - Case 9 finds intermediate optimum balancing timing and direction bias")
    
    plt.show()
