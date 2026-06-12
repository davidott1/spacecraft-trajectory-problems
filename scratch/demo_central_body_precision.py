"""
Demonstration of numerical issues when using wrong central body.

This script shows why switching central bodies matters for numerical integration
when a spacecraft is inside the Moon's sphere of influence.

Key insight: The physics is the same regardless of central body choice.
The issue is computational efficiency and numerical conditioning.
"""

import numpy as np

# Constants
GP_EARTH = 3.986004418e14  # Earth's GP [m³/s²]
GP_MOON  = 4.9048695e12    # Moon's GP [m³/s²]

# Moon's distance from Earth
MOON_DISTANCE = 384400e3   # [m]

# Moon's sphere of influence radius
SOI_MOON = MOON_DISTANCE * (GP_MOON / GP_EARTH)**(2/5)  # ~66,000 km


def compute_accelerations_earth_centered(sat_pos_from_earth, moon_pos_from_earth):
    """Compute accelerations with Earth as central body."""
    r_earth = np.linalg.norm(sat_pos_from_earth)
    a_earth = -GP_EARTH * sat_pos_from_earth / r_earth**3
    
    sat_to_moon = moon_pos_from_earth - sat_pos_from_earth
    r_sat_moon = np.linalg.norm(sat_to_moon)
    r_earth_moon = np.linalg.norm(moon_pos_from_earth)
    
    # Third-body formula includes indirect term
    a_moon = GP_MOON * (sat_to_moon / r_sat_moon**3 - moon_pos_from_earth / r_earth_moon**3)
    
    return a_earth, a_moon, a_earth + a_moon


def compute_accelerations_moon_centered(sat_pos_from_moon, earth_pos_from_moon):
    """Compute accelerations with Moon as central body."""
    r_moon = np.linalg.norm(sat_pos_from_moon)
    a_moon = -GP_MOON * sat_pos_from_moon / r_moon**3
    
    sat_to_earth = earth_pos_from_moon - sat_pos_from_moon
    r_sat_earth = np.linalg.norm(sat_to_earth)
    r_moon_earth = np.linalg.norm(earth_pos_from_moon)
    
    a_earth = GP_EARTH * (sat_to_earth / r_sat_earth**3 - earth_pos_from_moon / r_moon_earth**3)
    
    return a_moon, a_earth, a_moon + a_earth


def analyze_precision(dist_from_moon_km):
    """Analyze numerical conditioning at given distance from Moon's center."""
    
    dist_from_moon = dist_from_moon_km * 1000
    
    moon_pos_from_earth = np.array([MOON_DISTANCE, 0, 0])
    sat_pos_from_earth = np.array([MOON_DISTANCE - dist_from_moon, 0, 0])
    sat_pos_from_moon = sat_pos_from_earth - moon_pos_from_earth
    earth_pos_from_moon = -moon_pos_from_earth
    
    a_earth_central, a_moon_pert, a_total_earth = compute_accelerations_earth_centered(
        sat_pos_from_earth, moon_pos_from_earth
    )
    a_moon_central, a_earth_pert, a_total_moon = compute_accelerations_moon_centered(
        sat_pos_from_moon, earth_pos_from_moon
    )
    
    dist_from_earth = np.linalg.norm(sat_pos_from_earth)
    
    mag_a_earth_central = np.linalg.norm(a_earth_central)
    mag_a_moon_pert = np.linalg.norm(a_moon_pert)
    mag_a_moon_central = np.linalg.norm(a_moon_central)
    mag_a_earth_pert = np.linalg.norm(a_earth_pert)
    
    # Condition number proxy: ratio of perturbation to central body force
    # When > 1, the "perturbation" dominates - bad numerical conditioning
    condition_earth_centered = mag_a_moon_pert / mag_a_earth_central
    condition_moon_centered = mag_a_earth_pert / mag_a_moon_central
    
    print(f"\n{'='*70}")
    print(f"Satellite at {dist_from_moon_km:.0f} km from Moon, {dist_from_earth/1e3:.0f} km from Earth")
    print(f"Inside Moon SOI ({SOI_MOON/1e3:.0f} km): {'YES' if dist_from_moon < SOI_MOON else 'NO'}")
    print(f"{'='*70}")
    
    print(f"\n  Earth-Centered Frame:")
    print(f"    Central (Earth):     {mag_a_earth_central:12.6e} m/s²")
    print(f"    Perturbation (Moon): {mag_a_moon_pert:12.6e} m/s²")
    print(f"    Condition ratio:     {condition_earth_centered:12.2f}")
    
    print(f"\n  Moon-Centered Frame:")
    print(f"    Central (Moon):      {mag_a_moon_central:12.6e} m/s²")
    print(f"    Perturbation (Earth):{mag_a_earth_pert:12.6e} m/s²")
    print(f"    Condition ratio:     {condition_moon_centered:12.2f}")
    
    # The totals should match (physics is the same)
    print(f"\n  Total acceleration magnitude:")
    print(f"    Earth-centered: {np.linalg.norm(a_total_earth):.10e} m/s²")
    print(f"    Moon-centered:  {np.linalg.norm(a_total_moon):.10e} m/s²")
    
    if condition_earth_centered > 1:
        print(f"\n  ⚠️  Earth-centered: perturbation > central body!")
        print(f"      Integrator must track rapidly-changing dominant force as 'correction'.")
    if condition_moon_centered > 1:
        print(f"\n  ⚠️  Moon-centered: perturbation > central body!")
    
    return {
        'dist_from_moon_km': dist_from_moon_km,
        'inside_soi': dist_from_moon < SOI_MOON,
        'condition_earth': condition_earth_centered,
        'condition_moon': condition_moon_centered,
    }


def main():
    print("=" * 70)
    print("CENTRAL BODY SELECTION: NUMERICAL CONDITIONING ANALYSIS")
    print("=" * 70)
    print("""
The physics is IDENTICAL regardless of central body choice.
The issue is NUMERICAL CONDITIONING for the integrator.

When condition ratio >> 1:
  - The 'perturbation' is the dominant force
  - Integrator sees: small base + huge correction = hard to track accurately
  - Requires smaller step sizes, more function evaluations
  - Accumulated rounding errors grow faster

When condition ratio << 1:
  - Central body dominates (as intended)
  - Integrator sees: large base + small correction = easy to track
  - This is what numerical integrators are designed for
""")
    
    distances_km = [100000, 66000, 20000, 5000, 2000, 1000]
    
    results = []
    for dist in distances_km:
        results.append(analyze_precision(dist))
    
    print("\n" + "=" * 70)
    print("SUMMARY: Which frame is better conditioned?")
    print("=" * 70)
    print(f"{'Dist (km)':>10} | {'In SOI':>6} | {'Earth-ctr ratio':>15} | {'Moon-ctr ratio':>15} | {'Better Frame':>12}")
    print("-" * 70)
    for r in results:
        better = "Moon" if r['condition_earth'] > r['condition_moon'] else "Earth"
        print(f"{r['dist_from_moon_km']:>10.0f} | {'YES' if r['inside_soi'] else 'NO':>6} | {r['condition_earth']:>15.2f} | {r['condition_moon']:>15.2f} | {better:>12}")
    
    print("""
CONCLUSION: Use Moon-centered integration inside Moon's SOI (~66,000 km).
            Use Earth-centered integration outside.
            The physics is the same, but numerical conditioning differs dramatically.
""")


if __name__ == "__main__":
    main()
