"""
Integration Tests for Propagation Module
==========================================

Tests for high-fidelity propagation with third-body forces and maneuvers.
Requires SPICE kernels to be loaded.

Tests:
------
TestThirdBodyFrameConsistency
  - test_moon_acceleration_points_toward_moon : third-body acceleration from Moon points
                                                toward SPICE Moon position (frame consistency)
  - test_sun_acceleration_points_toward_sun   : third-body acceleration from Sun points
                                                toward SPICE Sun position (frame consistency)

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
from src.model.constants       import SOLARSYSTEMCONSTANTS
from src.model.time_converter  import utc_to_et
from src.schemas.gravity       import GravityModelConfig
from src.schemas.spacecraft    import SpacecraftProperties


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
