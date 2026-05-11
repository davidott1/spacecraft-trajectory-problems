"""
Integration Tests for Propagation Module
==========================================

Tests for Keplerian propagation, J2-perturbed propagation, third-body
frame consistency, and maneuver handling. All tests exercise multiple
components working together (dynamics + integrator + orbit converter).

Tests:
------
TestKeplerianPropagation
  - test_roundtrip_circular_orbit_period     : circular orbit returns to start after one period
  - test_energy_conservation                 : specific energy conserved over one orbit
  - test_angular_momentum_conservation       : angular momentum conserved over one orbit

TestJ2Perturbation
  - test_j2_raan_drift_direction             : J2 causes RAAN regression for prograde orbits
  - test_j2_sma_mean_preserved              : J2 oscillates SMA but preserves mean

TestThirdBodyFrameConsistency
  - test_moon_tidal_direction                : Moon acceleration aligns with SPICE tidal approximation
  - test_sun_tidal_direction                 : Sun acceleration aligns with SPICE tidal approximation

TestPropagateWithManeuvers
  - test_maneuver_increases_sma             : prograde burn increases SMA
  - test_no_maneuvers_matches_baseline      : empty maneuver list matches unperturbed propagation

Usage:
------
  python -m pytest src/verification/integration/test_propagation.py -v
"""
import pytest
import numpy as np
import spiceypy as spice

from datetime import datetime, timedelta

from scipy.integrate import solve_ivp

from src.model.dynamics        import AccelerationSTMDot, GeneralStateEquationsOfMotion, ThirdBodyGravity
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER
from src.model.time_converter  import utc_to_et
from src.schemas.gravity       import GravityModelConfig, SphericalHarmonicsConfig
from src.schemas.spacecraft    import SpacecraftProperties


class TestKeplerianPropagation:
  """
  Tests for Keplerian (point-mass) propagation.
  Exercises AccelerationSTMDot + GeneralStateEquationsOfMotion + solve_ivp together.
  """

  def test_roundtrip_circular_orbit_period(self):
    """
    Circular orbit must return to its initial state after exactly one period.
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma     = 7000.0e3
    vel_mag = np.sqrt(gp / sma)
    period  = 2.0 * np.pi * np.sqrt(sma**3 / gp)
    state_o = np.array([sma, 0.0, 0.0, 0.0, vel_mag, 0.0])

    gravity_config = GravityModelConfig(gp=gp)
    spacecraft     = SpacecraftProperties(mass=1000.0)
    accel          = AccelerationSTMDot(gravity_config=gravity_config, spacecraft=spacecraft)
    eom            = GeneralStateEquationsOfMotion(accel)

    sol     = solve_ivp(eom.state_time_derivative, (0, period), state_o,
                        method='DOP853', rtol=1e-12, atol=1e-14)
    state_f = sol.y[:, -1]

    np.testing.assert_allclose(state_o[0:3], state_f[0:3], atol=1e-4,
      err_msg="Position did not return to start after one period")
    np.testing.assert_allclose(state_o[3:6], state_f[3:6], atol=1e-4,
      err_msg="Velocity did not return to start after one period")


  def test_energy_conservation(self):
    """
    Specific mechanical energy must be conserved to 1e-10 relative error.
    """
    gp                = SOLARSYSTEMCONSTANTS.EARTH.GP
    periapsis_pos_mag = 7000.0e3
    apoapsis_pos_mag  = 14000.0e3
    sma               = (periapsis_pos_mag + apoapsis_pos_mag) / 2.0
    vel_mag           = np.sqrt(gp * (2.0 / periapsis_pos_mag - 1.0 / sma))
    state_o           = np.array([periapsis_pos_mag, 0.0, 0.0, 0.0, vel_mag, 0.0])

    def specific_energy(state):
      return 0.5 * np.linalg.norm(state[3:6])**2 - gp / np.linalg.norm(state[0:3])

    energy_o       = specific_energy(state_o)
    gravity_config = GravityModelConfig(gp=gp)
    spacecraft     = SpacecraftProperties(mass=1000.0)
    accel          = AccelerationSTMDot(gravity_config=gravity_config, spacecraft=spacecraft)
    eom            = GeneralStateEquationsOfMotion(accel)
    period         = 2.0 * np.pi * np.sqrt(sma**3 / gp)

    sol = solve_ivp(eom.state_time_derivative, (0, period), state_o,
                    method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)

    for t in np.linspace(0, period, 10):
      rel_error = abs(specific_energy(sol.sol(t)) - energy_o) / abs(energy_o)
      assert rel_error < 1e-10, f"Energy not conserved at t={t:.1f} s: rel_error={rel_error:.2e}"


  def test_angular_momentum_conservation(self):
    """
    Angular momentum vector magnitude must be conserved to 1e-10 relative error.
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    state_o = np.array([7000.0e3, 0.0, 0.0, 0.0, 6000.0, 4000.0])

    def ang_mom_mag(state):
      return np.linalg.norm(np.cross(state[0:3], state[3:6]))

    h_o            = ang_mom_mag(state_o)
    gravity_config = GravityModelConfig(gp=gp)
    spacecraft     = SpacecraftProperties(mass=1000.0)
    accel          = AccelerationSTMDot(gravity_config=gravity_config, spacecraft=spacecraft)
    eom            = GeneralStateEquationsOfMotion(accel)

    pos_mag = np.linalg.norm(state_o[0:3])
    vel_mag = np.linalg.norm(state_o[3:6])
    sma     = 1.0 / (2.0 / pos_mag - vel_mag**2 / gp)
    period  = 2.0 * np.pi * np.sqrt(sma**3 / gp)

    sol = solve_ivp(eom.state_time_derivative, (0, period), state_o,
                    method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)

    for t in np.linspace(0, period, 10):
      rel_error = abs(ang_mom_mag(sol.sol(t)) - h_o) / h_o
      assert rel_error < 1e-10, f"Angular momentum not conserved at t={t:.1f} s: rel_error={rel_error:.2e}"


class TestJ2Perturbation:
  """
  Tests for J2-perturbed propagation.
  Exercises AccelerationSTMDot + J2 harmonics + solve_ivp + OrbitConverter together.
  """

  @pytest.fixture
  def j2_dynamics(self):
    """AccelerationSTMDot with J2 only."""
    gp           = SOLARSYSTEMCONSTANTS.EARTH.GP
    earth_rad_eq = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    spacecraft   = SpacecraftProperties(mass=1000.0)
    accel        = AccelerationSTMDot(
      gravity_config = GravityModelConfig(
        gp                  = gp,
        spherical_harmonics = SphericalHarmonicsConfig(
          coefficients = ['J2'],
          gp           = gp,
          radius       = earth_rad_eq,
        ),
      ),
      spacecraft = spacecraft,
    )
    return accel, gp

  def test_j2_raan_drift_direction(self, j2_dynamics):
    """
    J2 must cause RAAN regression (westward drift) for prograde orbits.
    """
    accel, gp = j2_dynamics
    eom       = GeneralStateEquationsOfMotion(accel)

    sma     = 7000.0e3
    inc     = 45.0 * CONVERTER.RAD_PER_DEG
    vel_mag = np.sqrt(gp / sma)
    state_o = np.array([sma, 0.0, 0.0, 0.0, vel_mag * np.cos(inc), vel_mag * np.sin(inc)])
    coe_o   = OrbitConverter.pv_to_coe(state_o[0:3], state_o[3:6], gp)
    period  = 2.0 * np.pi * np.sqrt(sma**3 / gp)

    sol     = solve_ivp(eom.state_time_derivative, (0, 10 * period), state_o,
                        method='DOP853', rtol=1e-10, atol=1e-12)
    state_f = sol.y[:, -1]
    coe_f   = OrbitConverter.pv_to_coe(state_f[0:3], state_f[3:6], gp)

    delta_raan = coe_f.raan - coe_o.raan
    if delta_raan > np.pi:
      delta_raan -= 2.0 * np.pi
    elif delta_raan < -np.pi:
      delta_raan += 2.0 * np.pi

    assert delta_raan < 0, f"RAAN did not regress: delta_raan = {np.degrees(delta_raan):.4f} deg"


  def test_j2_sma_mean_preserved(self, j2_dynamics):
    """
    J2 causes short-period SMA oscillations, but the mean must remain
    within 0.1% of the initial value over one orbit.
    """
    accel, gp = j2_dynamics
    eom       = GeneralStateEquationsOfMotion(accel)

    sma     = 7000.0e3
    inc     = 45.0 * CONVERTER.RAD_PER_DEG
    vel_mag = np.sqrt(gp / sma)
    state_o = np.array([sma, 0.0, 0.0, 0.0, vel_mag * np.cos(inc), vel_mag * np.sin(inc)])
    period  = 2.0 * np.pi * np.sqrt(sma**3 / gp)

    sol = solve_ivp(eom.state_time_derivative, (0, period), state_o,
                    method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)

    sma_values = np.array([
      OrbitConverter.pv_to_coe(sol.sol(t)[0:3], sol.sol(t)[3:6], gp).sma
      for t in np.linspace(0, period, 100)
    ])

    assert np.isclose(np.mean(sma_values), sma, rtol=1e-3), \
      f"Mean SMA drifted: {np.mean(sma_values)/1e3:.3f} km vs {sma/1e3:.3f} km"
    assert np.std(sma_values) > 0, "Expected SMA oscillation but got constant value"


class TestThirdBodyFrameConsistency:
  """
  Tests that the third-body force model produces accelerations geometrically
  consistent with SPICE body positions.

  Uses the linear tidal approximation as the analytical reference:
    a_tidal ≈ (GM/R³)(3(r·R̂)R̂ - r)
  where R is the body position from SPICE and r is the satellite position.

  The tidal direction depends on the body's SPICE position (frame-dependent).
  If the force model queries SPICE in the wrong frame, the acceleration
  vector points in the wrong direction and the angular mismatch is large.
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    pass

  @pytest.fixture
  def epochs_et(self):
    """Three epochs spaced ~8 hours apart to avoid coincidental frame alignment."""
    return [
      utc_to_et(datetime(2025, 10, 1, 0, 0, 0)),
      utc_to_et(datetime(2025, 10, 1, 8, 0, 0)),
      utc_to_et(datetime(2025, 10, 1, 16, 0, 0)),
    ]

  @pytest.fixture
  def leo_pos_vec(self):
    """Satellite at [7000 km, 0, 0] in J2000."""
    return np.array([7000.0e3, 0.0, 0.0])

  def test_moon_tidal_direction(self, leo_pos_vec, epochs_et):
    """
    Third-body acceleration from the Moon must align with the linear tidal
    approximation derived from the Moon's SPICE J2000 position.

    The tidal approximation is accurate to ~1% for LEO (r/R ≈ 0.02).
    A frame error would cause a large angular mismatch at most epochs.
    Tested at 3 epochs spaced 8 hours apart to avoid coincidental alignment.
    """
    gp_moon    = SOLARSYSTEMCONSTANTS.MOON.GP
    third_body = ThirdBodyGravity(bodies=['MOON'])

    for epoch_et in epochs_et:
      acc_vec = third_body.point_mass(epoch_et, leo_pos_vec)

      # Moon position from SPICE (km → m)
      moon_state, _ = spice.spkez(301, epoch_et, 'J2000', 'NONE', 399)
      moon_pos_vec  = np.array(moon_state[0:3]) * 1000.0
      moon_pos_mag  = np.linalg.norm(moon_pos_vec)
      moon_pos_hat  = moon_pos_vec / moon_pos_mag

      # Linear tidal approximation: a ≈ (GM/R³)(3(r·R̂)R̂ - r)
      tidal_expected = (gp_moon / moon_pos_mag**3) * (
        3.0 * np.dot(leo_pos_vec, moon_pos_hat) * moon_pos_hat - leo_pos_vec
      )

      # Angular check: direction must match within 5°
      acc_hat   = acc_vec / np.linalg.norm(acc_vec)
      tidal_hat = tidal_expected / np.linalg.norm(tidal_expected)
      cos_angle = np.dot(acc_hat, tidal_hat)
      angle__deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

      assert angle__deg < 5.0, (
        f"Moon third-body acceleration misaligned by {angle__deg:.1f}° "
        f"at ET={epoch_et:.0f} — possible frame error"
      )

      # Magnitude check: must agree within 5%
      mag_ratio = np.linalg.norm(acc_vec) / np.linalg.norm(tidal_expected)
      assert abs(mag_ratio - 1.0) < 0.05, (
        f"Moon third-body magnitude ratio {mag_ratio:.4f} "
        f"at ET={epoch_et:.0f}, expected ~1.0"
      )

  def test_sun_tidal_direction(self, leo_pos_vec, epochs_et):
    """
    Third-body acceleration from the Sun must align with the linear tidal
    approximation derived from the Sun's SPICE J2000 position.

    The Sun is ~23,000 Earth radii away, so the tidal approximation is
    extremely accurate (r/R ≈ 5e-5). A frame error would cause a large
    angular mismatch at most epochs.
    Tested at 3 epochs spaced 8 hours apart to avoid coincidental alignment.
    """
    gp_sun     = SOLARSYSTEMCONSTANTS.SUN.GP
    third_body = ThirdBodyGravity(bodies=['SUN'])

    for epoch_et in epochs_et:
      acc_vec = third_body.point_mass(epoch_et, leo_pos_vec)

      # Sun position from SPICE (km → m)
      sun_state, _ = spice.spkez(10, epoch_et, 'J2000', 'NONE', 399)
      sun_pos_vec  = np.array(sun_state[0:3]) * 1000.0
      sun_pos_mag  = np.linalg.norm(sun_pos_vec)
      sun_pos_hat  = sun_pos_vec / sun_pos_mag

      # Linear tidal approximation: a ≈ (GM/R³)(3(r·R̂)R̂ - r)
      tidal_expected = (gp_sun / sun_pos_mag**3) * (
        3.0 * np.dot(leo_pos_vec, sun_pos_hat) * sun_pos_hat - leo_pos_vec
      )

      # Angular check: direction must match within 1° (Sun is very far)
      acc_hat   = acc_vec / np.linalg.norm(acc_vec)
      tidal_hat = tidal_expected / np.linalg.norm(tidal_expected)
      cos_angle = np.dot(acc_hat, tidal_hat)
      angle__deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

      assert angle__deg < 1.0, (
        f"Sun third-body acceleration misaligned by {angle__deg:.4f}° "
        f"at ET={epoch_et:.0f} — possible frame error"
      )

      # Magnitude check: must agree within 1%
      mag_ratio = np.linalg.norm(acc_vec) / np.linalg.norm(tidal_expected)
      assert abs(mag_ratio - 1.0) < 0.01, (
        f"Sun third-body magnitude ratio {mag_ratio:.6f} "
        f"at ET={epoch_et:.0f}, expected ~1.0"
      )


class TestPropagateWithManeuvers:
  """
  Tests for propagate_with_maneuvers — impulsive burn segmented propagation.
  Requires SPICE (dynamics model uses ET internally).
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    pass

  @pytest.fixture
  def epoch(self):
    return datetime(2025, 10, 1, 0, 0, 0)

  @pytest.fixture
  def keplerian_dynamics(self):
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    return AccelerationSTMDot(
      gravity_config = GravityModelConfig(gp=gp),
      spacecraft     = SpacecraftProperties(mass=1000.0),
    )

  @pytest.fixture
  def leo_state(self):
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma     = 7000.0e3
    vel_mag = np.sqrt(gp / sma)
    return np.array([sma, 0.0, 0.0, 0.0, vel_mag, 0.0])

  def test_maneuver_increases_sma(self, leo_state, keplerian_dynamics, epoch):
    """
    A prograde impulsive burn must increase SMA.
    The burn is applied at t=0 (before any propagation) in the prograde direction.
    """
    from src.propagation.numerical_propagator import propagate_with_maneuvers
    from src.schemas.spacecraft import ImpulsiveManeuver

    gp     = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma    = 7000.0e3
    period = 2.0 * np.pi * np.sqrt(sma**3 / gp)

    # Prograde direction at t=0 is +Y for our initial state [r,0,0,0,v,0]
    delta_vel_mag   = 50.0  # m/s
    maneuver_dt     = epoch + timedelta(seconds=1.0)  # Just after epoch
    delta_vel_vec   = np.array([0.0, delta_vel_mag, 0.0])  # +Y = prograde at t=0

    maneuver = ImpulsiveManeuver(
      time_dt       = maneuver_dt,
      delta_vel_vec = delta_vel_vec,
    )

    final_dt = epoch + timedelta(seconds=period)
    result   = propagate_with_maneuvers(
      initial_state = leo_state,
      initial_dt    = epoch,
      final_dt      = final_dt,
      dynamics      = keplerian_dynamics,
      maneuvers     = [maneuver],
    )

    assert result.success, f"Propagation failed: {result.message}"

    state_f = result.state[:, -1]
    coe_f   = OrbitConverter.pv_to_coe(state_f[0:3], state_f[3:6], gp)
    assert coe_f.sma > sma, \
      f"SMA did not increase after prograde burn: {coe_f.sma/1e3:.3f} km vs {sma/1e3:.3f} km"


  def test_no_maneuvers_matches_baseline(self, leo_state, keplerian_dynamics, epoch):
    """
    propagate_with_maneuvers with no maneuvers must yield the same
    final SMA as a bare two-body propagation.
    """
    from src.propagation.numerical_propagator import propagate_with_maneuvers

    gp     = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma    = 7000.0e3
    period = 2.0 * np.pi * np.sqrt(sma**3 / gp)

    final_dt = epoch + timedelta(seconds=period)
    result   = propagate_with_maneuvers(
      initial_state = leo_state,
      initial_dt    = epoch,
      final_dt      = final_dt,
      dynamics      = keplerian_dynamics,
      maneuvers     = [],
    )

    assert result.success

    state_f = result.state[:, -1]
    coe_f   = OrbitConverter.pv_to_coe(state_f[0:3], state_f[3:6], gp)

    # SMA must be conserved in Keplerian propagation
    assert np.isclose(coe_f.sma, sma, rtol=1e-6), \
      f"SMA changed without maneuvers: {coe_f.sma/1e3:.4f} km vs {sma/1e3:.4f} km"
