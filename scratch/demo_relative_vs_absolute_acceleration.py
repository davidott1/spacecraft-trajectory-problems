"""
Comparison of Relative (Indirect) vs Absolute Acceleration Formulations

Computes satellite acceleration RELATIVE TO EARTH using three methods:

1. INDIRECT: Earth-centered, perturbations in indirect form
2. DIRECT (EMB): Compute acc of sat and Earth relative to EMB, subtract
3. DIRECT (SSB): Compute acc of sat and Earth relative to SSB, subtract
"""

import numpy as np
import spiceypy as spice
from pathlib import Path
from scipy.integrate import solve_ivp

# =============================================================================
# Constants
# =============================================================================

GP_SUN = 1.32712440018e20    # m³/s²
GP_EARTH = 3.986004418e14    # m³/s²
GP_MOON = 4.9048695e12       # m³/s²

NAIF_SUN = 10
NAIF_EARTH = 399
NAIF_MOON = 301
NAIF_EMB = 3   # Earth-Moon Barycenter
NAIF_SSB = 0   # Solar System Barycenter

KM_TO_M = 1000.0


def load_spice_kernels():
    """Load required SPICE kernels."""
    project_root = Path(__file__).parent.parent
    spice_folder = project_root / 'data' / 'spice_kernels'
    
    for f in spice_folder.glob('naif*.tls'):
        spice.furnsh(str(f))
    for f in spice_folder.glob('de*.bsp'):
        spice.furnsh(str(f))
    for f in spice_folder.glob('pck*.tpc'):
        spice.furnsh(str(f))
    
    print(f"Loaded SPICE kernels from {spice_folder}")


def get_body_position(target_id: int, observer_id: int, et: float) -> np.ndarray:
    """Get position of target relative to observer [m]."""
    state, _ = spice.spkez(target_id, et, 'J2000', 'NONE', observer_id)
    return np.array(state[0:3]) * KM_TO_M


def dynamics_indirect(t, state, et0):
    """Dynamics using INDIRECT formulation."""
    pos = state[0:3]
    vel = state[3:6]
    
    # Get current ephemeris time
    et = et0 + t
    
    # Central body term
    r = np.linalg.norm(pos)
    acc = -GP_EARTH * pos / r**3
    
    # Sun perturbation (indirect form)
    pos_earth_to_sun = get_body_position(NAIF_SUN, NAIF_EARTH, et)
    pos_sat_to_sun = pos_earth_to_sun - pos
    acc += GP_SUN * (pos_sat_to_sun / np.linalg.norm(pos_sat_to_sun)**3 
                      - pos_earth_to_sun / np.linalg.norm(pos_earth_to_sun)**3)
    
    # Moon perturbation (indirect form)
    pos_earth_to_moon = get_body_position(NAIF_MOON, NAIF_EARTH, et)
    pos_sat_to_moon = pos_earth_to_moon - pos
    acc += GP_MOON * (pos_sat_to_moon / np.linalg.norm(pos_sat_to_moon)**3 
                       - pos_earth_to_moon / np.linalg.norm(pos_earth_to_moon)**3)
    
    return np.concatenate([vel, acc])


def dynamics_direct_ssb(t, state, et0):
    """Dynamics using DIRECT (SSB) formulation."""
    pos = state[0:3]  # Position relative to Earth
    vel = state[3:6]
    
    # Get current ephemeris time
    et = et0 + t
    
    # Get positions relative to SSB
    pos_ssb_to_sun = get_body_position(NAIF_SUN, NAIF_SSB, et)
    pos_ssb_to_earth = get_body_position(NAIF_EARTH, NAIF_SSB, et)
    pos_ssb_to_moon = get_body_position(NAIF_MOON, NAIF_SSB, et)
    
    # Satellite position relative to SSB
    pos_ssb_to_sat = pos_ssb_to_earth + pos
    
    # Absolute acceleration of satellite
    pos_sat_to_sun_ssb = pos_ssb_to_sun - pos_ssb_to_sat
    pos_sat_to_earth_ssb = pos_ssb_to_earth - pos_ssb_to_sat
    pos_sat_to_moon_ssb = pos_ssb_to_moon - pos_ssb_to_sat
    
    acc_sat_abs = (
        GP_SUN * pos_sat_to_sun_ssb / np.linalg.norm(pos_sat_to_sun_ssb)**3
        + GP_EARTH * pos_sat_to_earth_ssb / np.linalg.norm(pos_sat_to_earth_ssb)**3
        + GP_MOON * pos_sat_to_moon_ssb / np.linalg.norm(pos_sat_to_moon_ssb)**3
    )
    
    # Absolute acceleration of Earth
    pos_earth_to_sun_ssb = pos_ssb_to_sun - pos_ssb_to_earth
    pos_earth_to_moon_ssb = pos_ssb_to_moon - pos_ssb_to_earth
    
    acc_earth_abs = (
        GP_SUN * pos_earth_to_sun_ssb / np.linalg.norm(pos_earth_to_sun_ssb)**3
        + GP_MOON * pos_earth_to_moon_ssb / np.linalg.norm(pos_earth_to_moon_ssb)**3
    )
    
    # Relative acceleration
    acc = acc_sat_abs - acc_earth_abs
    
    return np.concatenate([vel, acc])


def main():
    print("=" * 80)
    print("INDIRECT vs DIRECT ACCELERATION FORMULATIONS")
    print("=" * 80)
    
    load_spice_kernels()
    
    et0 = spice.str2et('2025-01-01T12:00:00')
    
    # Satellite position relative to Earth (GEO)
    pos_earth_to_sat = np.array([42164e3, 0.0, 0.0])
    
    print(f"\nSatellite at GEO: [{pos_earth_to_sat[0]/1e6:.3f}, {pos_earth_to_sat[1]/1e6:.3f}, {pos_earth_to_sat[2]/1e6:.3f}] Mm")
    
    # =========================================================================
    # Get all body positions in J2000
    # =========================================================================
    
    # Positions relative to Earth
    pos_earth_to_sun  = get_body_position(NAIF_SUN , NAIF_EARTH, et0)
    pos_earth_to_moon = get_body_position(NAIF_MOON, NAIF_EARTH, et0)
    
    # Positions relative to SSB
    pos_ssb_to_sun   = get_body_position(NAIF_SUN  , NAIF_SSB, et0)
    pos_ssb_to_earth = get_body_position(NAIF_EARTH, NAIF_SSB, et0)
    pos_ssb_to_moon  = get_body_position(NAIF_MOON , NAIF_SSB, et0)
    
    # =========================================================================
    # METHOD 1: INDIRECT (Earth-centered)
    # =========================================================================
    
    # Central body term
    r_sat = np.linalg.norm(pos_earth_to_sat)
    acc_indirect = -GP_EARTH * pos_earth_to_sat / r_sat**3
    
    # Sun perturbation (indirect form)
    pos_sat_to_sun = pos_earth_to_sun - pos_earth_to_sat
    acc_indirect += GP_SUN * (pos_sat_to_sun / np.linalg.norm(pos_sat_to_sun)**3 
                              - pos_earth_to_sun / np.linalg.norm(pos_earth_to_sun)**3)
    
    # Moon perturbation (indirect form)
    pos_sat_to_moon = pos_earth_to_moon - pos_earth_to_sat
    acc_indirect += GP_MOON * (pos_sat_to_moon / np.linalg.norm(pos_sat_to_moon)**3 
                               - pos_earth_to_moon / np.linalg.norm(pos_earth_to_moon)**3)
    
    # =========================================================================
    # METHOD 2: DIRECT (SSB-centered)
    # =========================================================================
    
    # Satellite position relative to SSB
    pos_ssb_to_sat = pos_ssb_to_earth + pos_earth_to_sat
    
    # acc_sat_abs: absolute acceleration of satellite
    # Forces on satellite: Sun, Earth, Moon
    pos_sat_to_sun_ssb   = pos_ssb_to_sun   - pos_ssb_to_sat
    pos_sat_to_earth_ssb = pos_ssb_to_earth - pos_ssb_to_sat    
    pos_sat_to_moon_ssb  = pos_ssb_to_moon  - pos_ssb_to_sat
    acc_ssb_to_sat = (
          GP_SUN   * pos_sat_to_sun_ssb   / np.linalg.norm(pos_sat_to_sun_ssb  )**3
        + GP_EARTH * pos_sat_to_earth_ssb / np.linalg.norm(pos_sat_to_earth_ssb)**3
        + GP_MOON  * pos_sat_to_moon_ssb  / np.linalg.norm(pos_sat_to_moon_ssb )**3
    )
    
    # acc_earth_abs: absolute acceleration of Earth
    # Forces on Earth: Sun, Moon
    pos_earth_to_sun_ssb  = pos_ssb_to_sun  - pos_ssb_to_earth
    pos_earth_to_moon_ssb = pos_ssb_to_moon - pos_ssb_to_earth
    acc_ssb_to_earth = (
          GP_SUN  * pos_earth_to_sun_ssb  / np.linalg.norm(pos_earth_to_sun_ssb )**3
        + GP_MOON * pos_earth_to_moon_ssb / np.linalg.norm(pos_earth_to_moon_ssb)**3
    )
    
    # Final: acc_sat_rel_earth via SSB
    acc_direct_ssb = acc_ssb_to_sat - acc_ssb_to_earth
    
    # =========================================================================
    # PROPAGATION COMPARISON
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("PROPAGATION COMPARISON")
    print("=" * 80)
    
    # Initial state for NEARLY CIRCULAR orbit (e = 0.001)
    r_earth = 6378137.0  # Earth radius in meters
    
    # For a given eccentricity and perigee altitude, compute orbital parameters
    e = 0.500  # Eccentricity (nearly circular)
    perigee_altitude = 400e3  # 400 km altitude
    r_perigee = r_earth + perigee_altitude
    
    # Semi-major axis from perigee and eccentricity
    a = r_perigee / (1 - e)
    
    # Apogee radius
    r_apogee = a * (1 + e)
    apogee_altitude = r_apogee - r_earth
    
    # Start at perigee (where velocity is highest)
    r_initial = r_perigee
    v_initial = np.sqrt(GP_EARTH * (2/r_initial - 1/a))  # vis-viva equation
    
    # Initial state vector (at perigee, perpendicular velocity)
    # Ensure float64 precision
    state0 = np.array([r_initial, 0.0, 0.0, 0.0, v_initial, 0.0], dtype=np.float64)
    
    print(f"\nInitial state (NEARLY CIRCULAR ORBIT):")
    print(f"  Semi-major axis: {a/1e6:.3f} Mm")
    print(f"  Eccentricity: {e:.4f}")
    print(f"  Perigee altitude: {perigee_altitude/1e3:.1f} km")
    print(f"  Apogee altitude: {apogee_altitude/1e3:.1f} km")
    print(f"  Position: [{state0[0]/1e3:.3f}, {state0[1]/1e3:.3f}, {state0[2]/1e3:.3f}] km")
    print(f"  Velocity: [{state0[3]:.3f}, {state0[4]:.3f}, {state0[5]:.3f}] m/s")
    print(f"  Orbital period: {2*np.pi*np.sqrt(a**3/GP_EARTH)/60:.1f} minutes")
    
    # Time span (1 day)
    t_span = (0, 86400)  # seconds (24 hours)
    t_eval = np.linspace(0, 86400, 1441)  # Every minute for 24 hours
    
    # Integration tolerances (relax slightly for speed)
    rtol = 1e-10
    atol = 1e-12
    
    print(f"\nIntegration settings:")
    print(f"  Duration: {t_span[1]/3600:.1f} hours")
    print(f"  rtol: {rtol:.0e}, atol: {atol:.0e}")
    
    # Propagate using INDIRECT formulation
    print("\nPropagating with INDIRECT formulation...", end=" ", flush=True)
    sol_indirect = solve_ivp(
        lambda t, y: dynamics_indirect(t, y, et0),
        t_span, state0, method='DOP853', 
        rtol=rtol, atol=atol, t_eval=t_eval
    )
    print(f"Done. ({len(sol_indirect.t)} points, {sol_indirect.nfev} function evals)")
    
    # Propagate using DIRECT (SSB) formulation
    print("Propagating with DIRECT (SSB) formulation...", end=" ", flush=True)
    sol_direct = solve_ivp(
        lambda t, y: dynamics_direct_ssb(t, y, et0),
        t_span, state0, method='DOP853',
        rtol=rtol, atol=atol, t_eval=t_eval
    )
    print(f"Done. ({len(sol_direct.t)} points, {sol_direct.nfev} function evals)")
    
    # Compare final states
    final_pos_indirect = sol_indirect.y[0:3, -1]
    final_pos_direct = sol_direct.y[0:3, -1]
    pos_diff = final_pos_direct - final_pos_indirect
    
    print(f"\nFinal state after {t_span[1]/3600:.1f} hours:")
    print(f"  INDIRECT position: [{final_pos_indirect[0]/1e3:.6f}, {final_pos_indirect[1]/1e3:.6f}, {final_pos_indirect[2]/1e3:.6f}] km")
    print(f"  DIRECT position:   [{final_pos_direct[0]/1e3:.6f}, {final_pos_direct[1]/1e3:.6f}, {final_pos_direct[2]/1e3:.6f}] km")
    print(f"  Position difference: [{pos_diff[0]:.3e}, {pos_diff[1]:.3e}, {pos_diff[2]:.3e}] m")
    print(f"  |Position difference|: {np.linalg.norm(pos_diff):.3e} m")
    
    # Compute position differences over time
    pos_diffs = np.linalg.norm(sol_direct.y[0:3, :] - sol_indirect.y[0:3, :], axis=0)
    
    # Find when maximum difference occurs
    max_diff_idx = np.argmax(pos_diffs)
    max_diff_time = t_eval[max_diff_idx]
    max_diff_pos = sol_indirect.y[0:3, max_diff_idx]
    max_diff_radius = np.linalg.norm(max_diff_pos)
    
    print(f"\nPosition difference statistics:")
    print(f"  Maximum: {np.max(pos_diffs):.3e} m (at t={max_diff_time/3600:.1f} hours, r={max_diff_radius/1e6:.1f} Mm)")
    print(f"  Mean:    {np.mean(pos_diffs):.3e} m")
    print(f"  Final:   {pos_diffs[-1]:.3e} m")
    
    # Check altitude at different points to see orbit evolution
    altitudes = np.linalg.norm(sol_indirect.y[0:3, :], axis=0) - r_earth
    print(f"\nOrbit altitude range during propagation:")
    print(f"  Minimum: {np.min(altitudes)/1e3:.1f} km")
    print(f"  Maximum: {np.max(altitudes)/1e3:.1f} km")
    
    # =========================================================================
    # DEMONSTRATE WITH FLOAT32
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("COMPARISON: FLOAT64 vs FLOAT32")
    print("=" * 80)
    
    # Recompute with float32
    pos_earth_to_sat_32 = pos_earth_to_sat.astype(np.float32)
    pos_earth_to_sun_32 = pos_earth_to_sun.astype(np.float32)
    pos_earth_to_moon_32 = pos_earth_to_moon.astype(np.float32)
    pos_ssb_to_sun_32 = pos_ssb_to_sun.astype(np.float32)
    pos_ssb_to_earth_32 = pos_ssb_to_earth.astype(np.float32)
    pos_ssb_to_moon_32 = pos_ssb_to_moon.astype(np.float32)
    
    # INDIRECT with float32
    r_sat_32 = np.linalg.norm(pos_earth_to_sat_32)
    acc_indirect_32 = -np.float32(GP_EARTH) * pos_earth_to_sat_32 / r_sat_32**3
    
    pos_sat_to_sun_32 = pos_earth_to_sun_32 - pos_earth_to_sat_32
    acc_indirect_32 += np.float32(GP_SUN) * (
        pos_sat_to_sun_32 / np.linalg.norm(pos_sat_to_sun_32)**3 
        - pos_earth_to_sun_32 / np.linalg.norm(pos_earth_to_sun_32)**3
    )
    
    pos_sat_to_moon_32 = pos_earth_to_moon_32 - pos_earth_to_sat_32
    acc_indirect_32 += np.float32(GP_MOON) * (
        pos_sat_to_moon_32 / np.linalg.norm(pos_sat_to_moon_32)**3 
        - pos_earth_to_moon_32 / np.linalg.norm(pos_earth_to_moon_32)**3
    )
    
    # DIRECT with float32
    pos_ssb_to_sat_32 = pos_ssb_to_earth_32 + pos_earth_to_sat_32
    
    pos_sat_to_sun_ssb_32 = pos_ssb_to_sun_32 - pos_ssb_to_sat_32
    pos_sat_to_earth_ssb_32 = pos_ssb_to_earth_32 - pos_ssb_to_sat_32
    pos_sat_to_moon_ssb_32 = pos_ssb_to_moon_32 - pos_ssb_to_sat_32
    
    acc_sat_abs_ssb_32 = (
        np.float32(GP_SUN) * pos_sat_to_sun_ssb_32 / np.linalg.norm(pos_sat_to_sun_ssb_32)**3
        + np.float32(GP_EARTH) * pos_sat_to_earth_ssb_32 / np.linalg.norm(pos_sat_to_earth_ssb_32)**3
        + np.float32(GP_MOON) * pos_sat_to_moon_ssb_32 / np.linalg.norm(pos_sat_to_moon_ssb_32)**3
    )
    
    pos_earth_to_sun_ssb_32 = pos_ssb_to_sun_32 - pos_ssb_to_earth_32
    pos_earth_to_moon_ssb_32 = pos_ssb_to_moon_32 - pos_ssb_to_earth_32
    
    acc_earth_abs_ssb_32 = (
        np.float32(GP_SUN) * pos_earth_to_sun_ssb_32 / np.linalg.norm(pos_earth_to_sun_ssb_32)**3
        + np.float32(GP_MOON) * pos_earth_to_moon_ssb_32 / np.linalg.norm(pos_earth_to_moon_ssb_32)**3
    )
    
    acc_direct_ssb_32 = acc_sat_abs_ssb_32 - acc_earth_abs_ssb_32
    
    # Compare float32 results
    print("\nFloat32 Results:")
    print(f"  INDIRECT (float32)     : {np.linalg.norm(acc_indirect_32):.10e} m/s²")
    print(f"  DIRECT SSB (float32)   : {np.linalg.norm(acc_direct_ssb_32):.10e} m/s²")
    print(f"  Difference (float32)   : {np.linalg.norm(acc_direct_ssb_32 - acc_indirect_32):.10e} m/s²")
    print(f"  Relative error         : {np.linalg.norm(acc_direct_ssb_32 - acc_indirect_32) / np.linalg.norm(acc_indirect_32):.2e}")
    
    print("\nFloat64 Results (for comparison):")
    print(f"  INDIRECT (float64)     : {np.linalg.norm(acc_indirect):.15e} m/s²")
    print(f"  DIRECT SSB (float64)   : {np.linalg.norm(acc_direct_ssb):.15e} m/s²")
    print(f"  Difference (float64)   : {np.linalg.norm(acc_direct_ssb - acc_indirect):.2e} m/s²")
    print(f"  Relative error         : {np.linalg.norm(acc_direct_ssb - acc_indirect) / np.linalg.norm(acc_indirect):.2e}")
    
    # =========================================================================
    # INTEGRATOR STATISTICS
    # =========================================================================
    
    print(f"\nIntegrator statistics:")
    print(f"  INDIRECT:")
    print(f"    Output points: {len(sol_indirect.t)}")
    print(f"    Function evaluations: {sol_indirect.nfev}")
    print(f"    Evals per output point: {sol_indirect.nfev/len(sol_indirect.t):.1f}")
    print(f"    Success: {sol_indirect.success}")
    print(f"    Message: {sol_indirect.message}")
    print(f"  DIRECT (SSB):")
    print(f"    Output points: {len(sol_direct.t)}")
    print(f"    Function evaluations: {sol_direct.nfev}")
    print(f"    Evals per output point: {sol_direct.nfev/len(sol_direct.t):.1f}")
    print(f"    Success: {sol_direct.success}")
    print(f"    Message: {sol_direct.message}")
    
    # Computational efficiency comparison
    speedup = sol_direct.nfev / sol_indirect.nfev
    print(f"\nComputational Efficiency:")
    print(f"  DIRECT requires {speedup:.1f}x more function evaluations than INDIRECT")
    print(f"  Despite achieving position agreement to {np.linalg.norm(pos_diff):.3e} m")
    
    print("\nWhy the huge difference in function evaluations?")
    print("  INDIRECT formulation:")
    print("    - Accelerations are O(10^-3) m/s² for Earth gravity")
    print("    - Perturbations are O(10^-6) m/s² for Sun/Moon")
    print("    - Smooth variation, integrator can take large steps")
    print("  DIRECT (SSB) formulation:")
    print("    - Earth's acceleration around Sun is O(10^-3) m/s²")
    print("    - But position relative to SSB is O(10^11) m")
    print("    - Small numerical errors get amplified by subtraction")
    print("    - Integrator needs tiny steps to maintain accuracy")
    
    # # =========================================================================
    # # INTERMEDIATE VALUES
    # # =========================================================================
    
    # print("\n" + "=" * 80)
    # print("INTERMEDIATE VALUES")
    # print("=" * 80)
    
    # print(f"\nDIRECT (EMB) intermediates:")
    # print(f"  acc_sat_rel_EMB:   [{acc_sat_rel_emb[0]:.15e}, {acc_sat_rel_emb[1]:.15e}, {acc_sat_rel_emb[2]:.15e}]")
    # print(f"  acc_earth_rel_EMB: [{acc_earth_rel_emb[0]:.15e}, {acc_earth_rel_emb[1]:.15e}, {acc_earth_rel_emb[2]:.15e}]")
    
    # print(f"\nDIRECT (SSB) intermediates:")
    # print(f"  acc_sat_rel_SSB:   [{acc_sat_rel_ssb[0]:.15e}, {acc_sat_rel_ssb[1]:.15e}, {acc_sat_rel_ssb[2]:.15e}]")
    # print(f"  acc_earth_rel_SSB: [{acc_earth_rel_ssb[0]:.15e}, {acc_earth_rel_ssb[1]:.15e}, {acc_earth_rel_ssb[2]:.15e}]")
    
    spice.kclear()


if __name__ == "__main__":
    main()
