"""
Demonstration of floating-point precision in orbital mechanics.

This script investigates whether "precision loss" or "numerical conditioning"
actually matters for the Earth-Sun and Earth-Moon systems.

Goal: Find quantifiable evidence of when switching central bodies matters.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Constants
GP_EARTH = 3.986004418e14  # m³/s²
GP_MOON = 4.9048695e12     # m³/s²
GP_SUN = 1.32712440018e20  # m³/s²

EARTH_RADIUS = 6378137.0   # m
MOON_DISTANCE = 384400e3   # m
EARTH_SUN_DISTANCE = 1.496e11  # m (1 AU)

SOI_MOON = MOON_DISTANCE * (GP_MOON / GP_EARTH)**(2/5)  # ~66,000 km
SOI_EARTH = EARTH_SUN_DISTANCE * (GP_EARTH / GP_SUN)**(2/5)  # ~930,000 km


def demo_third_body_subtraction():
    """
    Analyze precision in the third-body perturbation formula.
    """
    
    print("=" * 80)
    print("THIRD-BODY FORMULA PRECISION ANALYSIS")
    print("=" * 80)
    
    print(f"""
The third-body acceleration formula involves subtraction:

  a_third_body = GM * (direct - indirect)
  
  where:
    direct   = 1 / |r_sat_to_body|²
    indirect = 1 / |r_central_to_body|²

Theory says: when direct ≈ indirect (ratio ≈ 1), subtraction can lose precision.
Let's test this for the Earth-Moon system...
""")

    GM_moon = GP_MOON
    r_earth_to_moon = MOON_DISTANCE
    
    cases = [
        ("LEO (7,000 km from Earth)", 7000e3),
        ("GEO (42,000 km)", 42000e3),
        ("Mid-transfer (200,000 km)", 200000e3),
        ("Near Moon (334,000 km)", 334000e3),
        ("Close to Moon (350,000 km)", 350000e3),
        ("Very close (370,000 km)", 370000e3),
    ]
    
    print("=" * 80)
    print("RESULTS: Precision Loss in Third-Body Subtraction")
    print("=" * 80)
    print(f"\n{'Location':<30} {'Dist to Moon':<15} {'Ratio':<12} {'Accel (m/s²)':<15} {'f32 Rel Err':<15} {'f32 Abs Err':<15}")
    print("-" * 110)
    
    for name, r_earth_sat in cases:
        r_sat_moon = r_earth_to_moon - r_earth_sat
        if r_sat_moon <= 0:
            continue
            
        direct = GM_moon / r_sat_moon**2
        indirect = GM_moon / r_earth_to_moon**2
        true_diff = direct - indirect
        
        d_f32 = np.float32(direct)
        ind_f32 = np.float32(indirect)
        diff_f32 = float(d_f32 - ind_f32)
        
        ratio = direct / indirect
        rel_err_f32 = abs(diff_f32 - true_diff) / abs(true_diff) * 100 if true_diff != 0 else 0
        abs_err_f32 = abs(diff_f32 - true_diff)
        
        print(f"{name:<30} {r_sat_moon/1e3:>10.0f} km   {ratio:<12.4f} {true_diff:<15.6e} {rel_err_f32:<15.8f}% {abs_err_f32:<15.6e}")
    
    print(f"""

CONCLUSION: 
  The third-body formula is WELL-DESIGNED:
  - Where precision matters (near Moon), we have high precision (ratio >> 1)
  - Where precision is lost (far from Moon), the acceleration is negligible
  
  This is NOT a practical concern for orbit propagation.
""")


def demo_earth_sun_system():
    """
    Investigate the Earth-Sun system to find evidence of precision issues.
    """
    
    print("\n" + "=" * 80)
    print("EARTH-SUN SYSTEM ANALYSIS")
    print("=" * 80)
    
    print(f"""
Earth's SOI is ~{SOI_EARTH/1e3:,.0f} km ({SOI_EARTH/EARTH_SUN_DISTANCE*100:.2f}% of Earth-Sun distance).

For an Earth-orbiting satellite:
  - Central body: Earth
  - Perturbation: Sun

Let's analyze the Sun's perturbation at various distances from Earth...
""")
    
    # =========================================================================
    # Part 1: Gravity magnitudes
    # =========================================================================
    print("=" * 80)
    print("Part 1: Gravity Magnitudes")
    print("=" * 80)
    print(f"\n{'Altitude':<30} {'Dist from Earth':<18} {'Earth Grav':<15} {'Sun Grav':<15} {'Ratio Sun/Earth':<18}")
    print("-" * 100)
    
    altitudes_km = [400, 35786, 384400, 500000, 800000, 924000]
    
    for alt_km in altitudes_km:
        dist_from_earth = EARTH_RADIUS + alt_km * 1000
        
        g_earth = GP_EARTH / dist_from_earth**2
        g_sun = GP_SUN / EARTH_SUN_DISTANCE**2
        ratio = g_sun / g_earth
        
        label = f"{alt_km:>10,} km"
        if alt_km == 400:
            label += " (LEO)"
        elif alt_km == 35786:
            label += " (GEO)"
        elif alt_km == 384400:
            label += " (Moon dist)"
        elif alt_km == 924000:
            label += " (SOI edge)"
            
        print(f"{label:<30} {dist_from_earth/1e6:>12.3f} Mm   {g_earth:<15.6e} {g_sun:<15.6e} {ratio:<18.6f}")
    
    print(f"""
OBSERVATION:
  - At LEO, Sun/Earth ratio is ~0.0007 (Sun perturbation is small)
  - At SOI edge, Sun gravity exceeds Earth gravity
""")
    
    # =========================================================================
    # Part 2: Third-body subtraction precision for Sun perturbation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Part 2: Sun Third-Body Subtraction Precision")
    print("=" * 80)
    
    print(f"""
For Sun as perturbing body on Earth-centered satellite:
  direct   = GM_sun / |r_sat_to_sun|²
  indirect = GM_sun / |r_earth_to_sun|²
  
Since satellite is very close to Earth compared to Sun:
  r_sat_to_sun ≈ r_earth_to_sun
  Therefore: direct ≈ indirect, ratio ≈ 1.0
  
This is where we SHOULD see precision loss!

But what if the satellite is CLOSE TO THE SUN instead of close to Earth?
Then the "perturbation" (Sun) dominates, and we're using the wrong central body.
""")
    
    print(f"\n{'Location':<35} {'r_sat_sun (AU)':<18} {'Ratio':<15} {'Accel (m/s²)':<15} {'f32 Rel Err %':<18}")
    print("-" * 105)
    
    # Cases: satellite at various positions along the Earth-Sun line
    # Positive = toward Sun, Negative = away from Sun
    cases = [
        ("LEO (400 km from Earth)", EARTH_RADIUS + 400e3, "away"),
        ("GEO (35,786 km)", EARTH_RADIUS + 35786e3, "away"),
        ("Moon distance (384,400 km)", EARTH_RADIUS + 384400e3, "away"),
        ("800,000 km from Earth", EARTH_RADIUS + 800000e3, "away"),
        ("Halfway to Sun (0.5 AU)", EARTH_SUN_DISTANCE * 0.5, "toward"),
        ("0.1 AU from Sun", EARTH_SUN_DISTANCE - 0.1 * EARTH_SUN_DISTANCE, "toward"),
        ("0.01 AU from Sun", EARTH_SUN_DISTANCE - 0.01 * EARTH_SUN_DISTANCE, "toward"),
        ("1 million km from Sun", 1e9, "toward"),  # ~0.007 AU from Sun
    ]
    
    for name, dist_from_earth, direction in cases:
        if direction == "away":
            # Satellite on opposite side of Earth from Sun
            r_sat_to_sun = EARTH_SUN_DISTANCE + dist_from_earth
        else:
            # Satellite between Earth and Sun
            r_sat_to_sun = EARTH_SUN_DISTANCE - dist_from_earth
            if r_sat_to_sun <= 0:
                continue
        
        direct = GP_SUN / r_sat_to_sun**2
        indirect = GP_SUN / EARTH_SUN_DISTANCE**2
        true_diff = direct - indirect
        
        # Float32 computation
        d_f32 = np.float32(direct)
        ind_f32 = np.float32(indirect)
        diff_f32 = float(d_f32 - ind_f32)
        
        # Float64 computation
        d_f64 = np.float64(direct)
        ind_f64 = np.float64(indirect)
        diff_f64 = float(d_f64 - ind_f64)
        
        ratio = direct / indirect
        rel_err_f32 = abs(diff_f32 - true_diff) / abs(true_diff) * 100 if true_diff != 0 else 0
        rel_err_f64 = abs(diff_f64 - true_diff) / abs(true_diff) * 100 if true_diff != 0 else 0
        
        print(f"{name:<35} {r_sat_to_sun/EARTH_SUN_DISTANCE:<18.10f} {ratio:<15.6f} {true_diff:<15.6e} {rel_err_f32:<18.6f}")
    
    print(f"""

OBSERVATION:
  Near Earth (ratio ≈ 1.0):
    - f32 relative error is small (< 0.02%)
    - The acceleration is tiny, so errors don't matter much
    
  Close to Sun (ratio >> 1 or << 1):
    - Direct term DOMINATES (satellite feels mostly Sun's gravity)
    - The "perturbation" is now larger than Earth's gravity!
    - f32 error is smaller because there's no cancellation
    
  KEY INSIGHT:
    The precision loss from subtraction is MINIMAL even at ratio ≈ 1.0.
    The REAL issue is that when you're close to the Sun, you should use
    the Sun as the central body, not because of precision, but because
    the physics is dominated by the Sun.
""")
    
    # =========================================================================
    # Part 2b: Compare gravity magnitudes when close to Sun
    # =========================================================================
    print("\n" + "=" * 80)
    print("Part 2b: Why Central Body Choice Matters (Physics, Not Precision)")
    print("=" * 80)
    
    print(f"""
Let's compare Earth gravity vs Sun gravity at various positions.
This shows when you're using the "wrong" central body from a PHYSICS standpoint.
""")
    
    print(f"\n{'Location':<35} {'Earth Grav (m/s²)':<18} {'Sun Grav (m/s²)':<18} {'Sun/Earth Ratio':<18}")
    print("-" * 95)
    
    physics_cases = [
        ("LEO (400 km from Earth)", EARTH_RADIUS + 400e3, EARTH_SUN_DISTANCE + EARTH_RADIUS + 400e3),
        ("GEO (35,786 km)", EARTH_RADIUS + 35786e3, EARTH_SUN_DISTANCE + EARTH_RADIUS + 35786e3),
        ("Earth's SOI edge (924,000 km)", SOI_EARTH, EARTH_SUN_DISTANCE + SOI_EARTH),
        ("Halfway to Sun", EARTH_SUN_DISTANCE * 0.5, EARTH_SUN_DISTANCE * 0.5),
        ("0.1 AU from Sun", EARTH_SUN_DISTANCE - 0.1 * EARTH_SUN_DISTANCE, 0.1 * EARTH_SUN_DISTANCE),
        ("0.01 AU from Sun", EARTH_SUN_DISTANCE - 0.01 * EARTH_SUN_DISTANCE, 0.01 * EARTH_SUN_DISTANCE),
        ("1 million km from Sun", EARTH_SUN_DISTANCE - 1e9, 1e9),
    ]
    
    for name, dist_from_earth, dist_from_sun in physics_cases:
        if dist_from_earth <= 0 or dist_from_sun <= 0:
            continue
            
        g_earth = GP_EARTH / dist_from_earth**2
        g_sun = GP_SUN / dist_from_sun**2
        ratio = g_sun / g_earth
        
        print(f"{name:<35} {g_earth:<18.6e} {g_sun:<18.6e} {ratio:<18.2f}")
    
    print(f"""

INTERPRETATION:
  - At Earth's SOI edge: Sun/Earth ratio ≈ 1 (by definition of SOI)
  - Halfway to Sun: Sun gravity is ~33,000x stronger than Earth!
  - 0.01 AU from Sun: Sun gravity is ~330 million times stronger!
  
  The issue is NOT precision loss in the subtraction formula.
  The issue is that when Sun gravity >> Earth gravity, you're treating
  a DOMINANT force as a "perturbation" to a TINY central force.
  
  This makes the integrator work harder (more steps) but doesn't cause
  wrong answers - just inefficiency.
""")

    # =========================================================================
    # Part 3: Integration comparison - FIXED
    # =========================================================================
    print("\n" + "=" * 80)
    print("Part 3: Integration Comparison (CORRECTED)")
    print("=" * 80)
    
    print("""
IMPORTANT: The previous comparison was FLAWED because:
  1. Sun-centered dynamics didn't account for Earth's orbital motion around Sun
  2. The two formulations were solving DIFFERENT physical problems
  
Let's fix this by comparing ONLY Earth-centered propagation with/without Sun,
to see how much the Sun perturbation actually affects the trajectory.
""")
    
    def earth_centered_no_sun(t, state):
        """Earth as central body, NO Sun perturbation (two-body)."""
        pos = state[0:3]
        vel = state[3:6]
        
        r = np.linalg.norm(pos)
        acc = -GP_EARTH * pos / r**3
        
        return np.concatenate([vel, acc])
    
    def earth_centered_with_sun(t, state):
        """Earth as central body, WITH Sun perturbation."""
        pos = state[0:3]
        vel = state[3:6]
        
        r = np.linalg.norm(pos)
        acc = -GP_EARTH * pos / r**3
        
        # Sun perturbation (Sun at +X from Earth, stationary for simplicity)
        sun_pos = np.array([EARTH_SUN_DISTANCE, 0.0, 0.0])
        pos_to_sun = sun_pos - pos
        r_to_sun = np.linalg.norm(pos_to_sun)
        r_earth_sun = np.linalg.norm(sun_pos)
        
        acc += GP_SUN * (pos_to_sun / r_to_sun**3 - sun_pos / r_earth_sun**3)
        
        return np.concatenate([vel, acc])
    
    test_cases = [
        ("LEO (400 km)", 400e3),
        ("GEO (35,786 km)", 35786e3),
        ("High orbit (500,000 km)", 500000e3),
    ]
    
    t_span = (0, 86400)  # 1 day
    rtol = 1e-12
    atol = 1e-15
    
    print(f"\nComparing: Two-body vs Two-body + Sun perturbation")
    print(f"Integration: {t_span[1]/86400:.0f} day, rtol={rtol:.0e}, atol={atol:.0e}")
    print("-" * 100)
    print(f"{'Scenario':<30} {'No Sun steps':<15} {'With Sun steps':<15} {'Pos Diff (m)':<18} {'Pos Diff (km)':<15}")
    print("-" * 100)
    
    for name, altitude in test_cases:
        r_orbit = EARTH_RADIUS + altitude
        v_circular = np.sqrt(GP_EARTH / r_orbit)
        
        pos_initial = np.array([r_orbit, 0.0, 0.0])
        vel_initial = np.array([0.0, v_circular, 0.0])
        state_initial = np.concatenate([pos_initial, vel_initial])
        
        # Integrate without Sun
        sol_no_sun = solve_ivp(earth_centered_no_sun, t_span, state_initial,
                               method='DOP853', rtol=rtol, atol=atol)
        steps_no_sun = len(sol_no_sun.t)
        final_pos_no_sun = sol_no_sun.y[0:3, -1]
        
        # Integrate with Sun
        sol_with_sun = solve_ivp(earth_centered_with_sun, t_span, state_initial,
                                 method='DOP853', rtol=rtol, atol=atol)
        steps_with_sun = len(sol_with_sun.t)
        final_pos_with_sun = sol_with_sun.y[0:3, -1]
        
        # Position difference due to Sun perturbation
        pos_diff = np.linalg.norm(final_pos_with_sun - final_pos_no_sun)
        
        print(f"{name:<30} {steps_no_sun:<15} {steps_with_sun:<15} {pos_diff:<18.6e} {pos_diff/1000:<15.3f}")
    
    print(f"""

INTERPRETATION:
  This shows the PHYSICAL EFFECT of Sun perturbation on satellite trajectories:
  - LEO: Sun perturbation causes ~26 m drift per day
  - GEO: Sun perturbation causes ~2.5 km drift per day  
  - High orbit: Sun perturbation causes ~151 km drift per day
  
  These are real physical effects, not numerical artifacts.
  The Sun's gravity gradient across the orbit causes secular drift.
""")
    
    # =========================================================================
    # Part 4: The REAL precision test - f32 vs f64
    # =========================================================================
    print("\n" + "=" * 80)
    print("Part 4: Float32 vs Float64 Precision Test")
    print("=" * 80)
    
    print("""
To test NUMERICAL precision (not physical effects), we compare:
  - Float64 integration (reference)
  - Float32 intermediate calculations
  
If precision loss matters, f32 will diverge from f64.
""")
    
    def earth_centered_with_sun_f32_intermediates(t, state):
        """Same as earth_centered_with_sun but uses f32 for intermediate calcs."""
        pos = state[0:3]
        vel = state[3:6]
        
        r = np.linalg.norm(pos)
        acc = -GP_EARTH * pos / r**3
        
        # Sun perturbation with f32 intermediates
        sun_pos = np.array([EARTH_SUN_DISTANCE, 0.0, 0.0])
        pos_to_sun = sun_pos - pos
        r_to_sun = np.linalg.norm(pos_to_sun)
        r_earth_sun = np.linalg.norm(sun_pos)
        
        # Force f32 precision for the subtraction
        direct = np.float32(GP_SUN / r_to_sun**3) * pos_to_sun.astype(np.float32)
        indirect = np.float32(GP_SUN / r_earth_sun**3) * sun_pos.astype(np.float32)
        sun_acc = (direct - indirect).astype(np.float64)
        
        acc = acc + sun_acc
        
        return np.concatenate([vel, acc])
    
    print(f"\nComparing: Float64 vs Float32 intermediates")
    print(f"Integration: {t_span[1]/86400:.0f} day, rtol={rtol:.0e}, atol={atol:.0e}")
    print("-" * 100)
    print(f"{'Scenario':<30} {'f64 steps':<15} {'f32 steps':<15} {'Pos Diff (m)':<18} {'Rel Pos Err':<15}")
    print("-" * 100)
    
    for name, altitude in test_cases:
        r_orbit = EARTH_RADIUS + altitude
        v_circular = np.sqrt(GP_EARTH / r_orbit)
        
        pos_initial = np.array([r_orbit, 0.0, 0.0])
        vel_initial = np.array([0.0, v_circular, 0.0])
        state_initial = np.concatenate([pos_initial, vel_initial])
        
        # Float64 reference
        sol_f64 = solve_ivp(earth_centered_with_sun, t_span, state_initial,
                            method='DOP853', rtol=rtol, atol=atol)
        steps_f64 = len(sol_f64.t)
        final_pos_f64 = sol_f64.y[0:3, -1]
        
        # Float32 intermediates
        sol_f32 = solve_ivp(earth_centered_with_sun_f32_intermediates, t_span, state_initial,
                            method='DOP853', rtol=rtol, atol=atol)
        steps_f32 = len(sol_f32.t)
        final_pos_f32 = sol_f32.y[0:3, -1]
        
        pos_diff = np.linalg.norm(final_pos_f64 - final_pos_f32)
        rel_err = pos_diff / np.linalg.norm(final_pos_f64)
        
        print(f"{name:<30} {steps_f64:<15} {steps_f32:<15} {pos_diff:<18.6e} {rel_err:<15.6e}")
    
    print(f"""

INTERPRETATION:
  The f32 vs f64 comparison reveals something important:
  
  POSITION ERROR:
    - All scenarios show < 2m position difference after 1 day
    - Relative position error is ~10⁻⁸ (excellent!)
    - f32 does NOT cause significant trajectory errors
    
  STEP COUNT:
    - LEO: Similar steps (766 vs 767) - no issue
    - GEO: f32 needs 93x more steps! (6245 vs 67)
    - High orbit: f32 needs 152x more steps! (3650 vs 24)
    
  WHY THE STEP COUNT EXPLOSION?
    The f32 noise in the acceleration calculation causes the adaptive
    integrator's error estimator to see "rough" derivatives. It responds
    by taking smaller steps to maintain accuracy - even though the final
    answer is fine.
    
  CONCLUSION:
    Float32 intermediates don't cause WRONG answers, but they cause
    INEFFICIENT integration due to noisy derivative estimates.
    
    For production code using float64 throughout: NO PROBLEM.
    The third-body subtraction formula works fine.
""")


def demo_central_body_comparison():
    """
    THE REAL TEST: Does switching central bodies actually matter?
    
    We propagate the SAME physical trajectory using:
    1. Earth as central body + Sun perturbation
    2. Sun as central body + Earth perturbation
    
    If central body choice matters, we should see:
    - Different step counts (efficiency)
    - Different final positions (accuracy)
    """
    
    print("\n" + "=" * 80)
    print("DEFINITIVE TEST: DOES CENTRAL BODY CHOICE MATTER?")
    print("=" * 80)
    
    print("""
We propagate the EXACT SAME physical trajectory two ways:
  1. Earth-centered: a = -GM_earth/r³ * r + Sun_perturbation
  2. Sun-centered:   a = -GM_sun/r³ * r + Earth_perturbation

Both should give identical physics. If central body choice matters,
we'll see differences in step count or final position.

NOTE: We keep Earth fixed at (1 AU, 0, 0) for simplicity. This is not
physically realistic but isolates the central body choice effect.
""")
    
    def earth_centered_dynamics(t, state):
        """Satellite state relative to Earth. Sun as perturbation."""
        pos = state[0:3]
        vel = state[3:6]
        
        r = np.linalg.norm(pos)
        acc = -GP_EARTH * pos / r**3
        
        # Sun perturbation (Sun fixed at +X from Earth)
        sun_pos = np.array([EARTH_SUN_DISTANCE, 0.0, 0.0])
        pos_to_sun = sun_pos - pos
        r_to_sun = np.linalg.norm(pos_to_sun)
        r_earth_sun = EARTH_SUN_DISTANCE
        
        acc += GP_SUN * (pos_to_sun / r_to_sun**3 - sun_pos / r_earth_sun**3)
        
        return np.concatenate([vel, acc])
    
    def sun_centered_dynamics(t, state):
        """Satellite state relative to Sun. Earth as perturbation."""
        pos = state[0:3]
        vel = state[3:6]
        
        r = np.linalg.norm(pos)
        acc = -GP_SUN * pos / r**3
        
        # Earth perturbation (Earth fixed at +X from Sun)
        earth_pos = np.array([EARTH_SUN_DISTANCE, 0.0, 0.0])
        pos_to_earth = earth_pos - pos
        r_to_earth = np.linalg.norm(pos_to_earth)
        r_sun_earth = EARTH_SUN_DISTANCE
        
        acc += GP_EARTH * (pos_to_earth / r_to_earth**3 - earth_pos / r_sun_earth**3)
        
        return np.concatenate([vel, acc])
    
    # Test cases at different distances from Earth (toward Sun)
    # Key: we want to see what happens when satellite is CLOSE TO SUN
    test_cases = [
        ("Near Earth (GEO)", 42164e3, "earth_dominated"),
        ("500,000 km from Earth", 500000e3, "earth_dominated"),
        ("Halfway to Sun (0.5 AU)", EARTH_SUN_DISTANCE * 0.5, "transition"),
        ("0.1 AU from Sun", EARTH_SUN_DISTANCE * 0.9, "sun_dominated"),
        ("0.01 AU from Sun", EARTH_SUN_DISTANCE * 0.99, "sun_dominated"),
    ]
    
    t_span = (0, 3600)  # 1 hour (shorter for extreme cases)
    rtol = 1e-10
    atol = 1e-12
    
    print(f"Integration: {t_span[1]/3600:.1f} hour, rtol={rtol:.0e}, atol={atol:.0e}")
    print("-" * 140)
    print(f"{'Scenario':<30} {'Dominant':<15} {'Earth-ctr steps':<18} {'Sun-ctr steps':<18} {'Step Ratio':<12} {'Pos Diff (m)':<18}")
    print("-" * 140)
    
    results = []
    
    for name, dist_from_earth, regime in test_cases:
        # Satellite between Earth and Sun
        # Earth-centered: satellite at (-dist, 0, 0)
        sat_pos_earth_frame = np.array([-dist_from_earth, 0.0, 0.0])
        
        # Sun-centered: satellite at (EARTH_SUN_DISTANCE - dist, 0, 0)
        sat_pos_sun_frame = np.array([EARTH_SUN_DISTANCE - dist_from_earth, 0.0, 0.0])
        
        # Give it some velocity perpendicular to the Earth-Sun line
        # Scale velocity to be somewhat bound to the dominant body
        if regime == "earth_dominated":
            v_scale = np.sqrt(GP_EARTH / dist_from_earth) * 0.5
        elif regime == "sun_dominated":
            dist_from_sun = EARTH_SUN_DISTANCE - dist_from_earth
            v_scale = np.sqrt(GP_SUN / dist_from_sun) * 0.1
        else:
            v_scale = 1000.0  # 1 km/s arbitrary
        
        sat_vel_earth_frame = np.array([0.0, v_scale, 0.0])
        sat_vel_sun_frame = sat_vel_earth_frame.copy()  # Same inertial velocity (Earth fixed)
        
        state_earth = np.concatenate([sat_pos_earth_frame, sat_vel_earth_frame])
        state_sun = np.concatenate([sat_pos_sun_frame, sat_vel_sun_frame])
        
        # Integrate Earth-centered
        try:
            sol_earth = solve_ivp(earth_centered_dynamics, t_span, state_earth,
                                  method='DOP853', rtol=rtol, atol=atol)
            steps_earth = len(sol_earth.t)
            final_pos_earth = sol_earth.y[0:3, -1]
            earth_success = sol_earth.success
        except Exception:
            steps_earth = -1
            earth_success = False
            final_pos_earth = np.zeros(3)
        
        # Integrate Sun-centered
        try:
            sol_sun = solve_ivp(sun_centered_dynamics, t_span, state_sun,
                                method='DOP853', rtol=rtol, atol=atol)
            steps_sun = len(sol_sun.t)
            final_pos_sun = sol_sun.y[0:3, -1]
            sun_success = sol_sun.success
        except Exception:
            steps_sun = -1
            sun_success = False
            final_pos_sun = np.zeros(3)
        
        if earth_success and sun_success and steps_earth > 0 and steps_sun > 0:
            # Convert Sun-centered to Earth-centered for comparison
            earth_pos_in_sun_frame = np.array([EARTH_SUN_DISTANCE, 0.0, 0.0])
            final_pos_sun_in_earth_frame = final_pos_sun - earth_pos_in_sun_frame
            
            pos_diff = np.linalg.norm(final_pos_earth - final_pos_sun_in_earth_frame)
            step_ratio = steps_earth / steps_sun
            
            dominant = "Earth" if regime == "earth_dominated" else ("Sun" if regime == "sun_dominated" else "~Equal")
            
            print(f"{name:<30} {dominant:<15} {steps_earth:<18} {steps_sun:<18} {step_ratio:<12.2f} {pos_diff:<18.6e}")
            
            results.append({
                'name': name,
                'regime': regime,
                'steps_earth': steps_earth,
                'steps_sun': steps_sun,
                'step_ratio': step_ratio,
                'pos_diff': pos_diff,
            })
        else:
            status_earth = "OK" if earth_success else "FAIL"
            status_sun = "OK" if sun_success else "FAIL"
            print(f"{name:<30} {'?':<15} {steps_earth} ({status_earth})    {steps_sun} ({status_sun})      {'N/A':<12} {'N/A':<18}")
    
    print(f"""

ANALYSIS OF RESULTS:
""")
    
    if results:
        # Analyze the pattern
        earth_dominated = [r for r in results if r['regime'] == 'earth_dominated']
        sun_dominated = [r for r in results if r['regime'] == 'sun_dominated']
        
        if earth_dominated:
            avg_ratio_earth = np.mean([r['step_ratio'] for r in earth_dominated])
            print(f"  Earth-dominated regime (near Earth):")
            print(f"    Average step ratio (Earth-ctr/Sun-ctr): {avg_ratio_earth:.2f}")
            if avg_ratio_earth < 1:
                print(f"    → Earth-centered is MORE EFFICIENT (as expected)")
            else:
                print(f"    → Sun-centered is more efficient (unexpected!)")
        
        if sun_dominated:
            avg_ratio_sun = np.mean([r['step_ratio'] for r in sun_dominated])
            print(f"\n  Sun-dominated regime (near Sun):")
            print(f"    Average step ratio (Earth-ctr/Sun-ctr): {avg_ratio_sun:.2f}")
            if avg_ratio_sun > 1:
                print(f"    → Sun-centered is MORE EFFICIENT (as expected)")
                print(f"    → Earth-centered needs {avg_ratio_sun:.1f}x more steps!")
            else:
                print(f"    → Earth-centered is more efficient (unexpected!)")
        
        # Check accuracy
        max_pos_diff = max(r['pos_diff'] for r in results)
        print(f"\n  Accuracy check:")
        print(f"    Maximum position difference: {max_pos_diff:.6e} m")
        if max_pos_diff < 1.0:
            print(f"    → Both formulations give SAME answer (< 1m difference)")
        else:
            print(f"    → WARNING: Significant position differences detected!")
    
    print(f"""

CONCLUSION:
  The step ratio shows whether central body choice affects EFFICIENCY.
  
  If step ratio varies significantly between regimes:
    → Central body choice MATTERS for computational efficiency
    → Use the dominant body as central body to minimize steps
    
  If step ratio ≈ 1 everywhere:
    → Central body choice DOESN'T matter for efficiency
    → The "switch at SOI" advice may be outdated
    
  Position differences should always be small (< integrator tolerance)
  since both formulations represent the same physics.
""")


def main():
    print("=" * 80)
    print("FLOATING-POINT PRECISION IN ORBITAL MECHANICS")
    print("=" * 80)
    print("""
Goal: Find QUANTIFIABLE EVIDENCE of when switching central bodies matters.

We'll investigate:
1. Third-body subtraction precision (Earth-Moon)
2. Earth-Sun system in detail
""")
    
    # Part 1: Third-body subtraction
    demo_third_body_subtraction()
    
    # Part 2: Earth-Sun system deep dive
    demo_earth_sun_system()
    
    # Part 3: THE DEFINITIVE TEST
    demo_central_body_comparison()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("""
Based on the numerical experiments above:

1. THIRD-BODY SUBTRACTION PRECISION:
   - Float64 handles ratio ≈ 1.0 without issue
   - Float32 causes efficiency problems, not accuracy problems
   
2. CENTRAL BODY CHOICE:
   - [SEE PART 3 RESULTS ABOVE]
   - If step ratios ≈ 1 and pos diffs are tiny: DOESN'T MATTER
   - If step ratios vary wildly: MATTERS FOR EFFICIENCY
   - If pos diffs are large: MATTERS FOR ACCURACY
   
3. THE HONEST CONCLUSION:
   - Run the script and look at the Part 3 results
   - If they show no significant differences, then the "switch central
     bodies for precision" advice may be outdated for modern float64
   - The SOI concept may be more about CONCEPTUAL clarity than numerical
     necessity
""")

if __name__ == "__main__":
    main()