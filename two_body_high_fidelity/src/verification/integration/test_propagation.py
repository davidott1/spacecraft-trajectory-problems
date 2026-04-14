"""
Integration Tests for Propagation Module
==========================================

Tests for high-fidelity propagation with third-body forces and maneuvers.
Requires SPICE kernels to be loaded.

Tests:
------
TestThirdBodyPropagation
  - test_third_body_perturbs_energy         : Moon/Sun third-body breaks energy conservation
  - test_third_body_raan_drift_differs_j2   : third-body changes RAAN drift vs J2-only

TestPropagateWithManeuvers
  - test_maneuver_increases_sma             : prograde burn increases SMA
  - test_no_maneuvers_matches_baseline      : empty maneuver list matches unperturbed propagation

Usage:
------
  python -m pytest src/verification/integration/test_propagation.py -v
"""
import pytest
import numpy as np

from datetime import datetime, timedelta

from scipy.integrate import solve_ivp

from src.model.dynamics        import AccelerationSTMDot, GeneralStateEquationsOfMotion
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER
from src.model.time_converter  import utc_to_et
from src.schemas.gravity       import GravityModelConfig, SphericalHarmonicsConfig, ThirdBodyConfig
from src.schemas.spacecraft    import SpacecraftProperties


class TestThirdBodyPropagation:
  """
  Tests for propagation with third-body perturbations (Moon and Sun).
  Requires SPICE kernels.
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    pass

  @pytest.fixture
  def epoch(self):
    return datetime(2025, 10, 1)

  @pytest.fixture
  def leo_state(self):
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma     = 7000.0e3
    vel_mag = np.sqrt(gp / sma)
    return np.array([sma, 0.0, 0.0, 0.0, vel_mag, 0.0])

  def test_third_body_perturbs_energy(self, leo_state, epoch):
    """
    Moon/Sun third-body forces must break energy conservation relative
    to the two-body case (energy no longer constant).
    """
    gp     = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma    = 7000.0e3
    period = 2.0 * np.pi * np.sqrt(sma**3 / gp)
    t0_et  = utc_to_et(epoch)

    def specific_energy(state):
      return 0.5 * np.linalg.norm(state[3:6])**2 - gp / np.linalg.norm(state[0:3])

    energy_o = specific_energy(leo_state)

    accel_3b = AccelerationSTMDot(
      gravity_config = GravityModelConfig(
        gp         = gp,
        third_body = ThirdBodyConfig(enabled=True, bodies=['MOON', 'SUN']),
      ),
      spacecraft = SpacecraftProperties(mass=1000.0),
    )

    eom_3b = GeneralStateEquationsOfMotion(accel_3b)
    sol_3b = solve_ivp(
      eom_3b.state_time_derivative, (t0_et, t0_et + 10 * period), leo_state,
      method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True,
    )

    energy_errors = [
      abs(specific_energy(sol_3b.sol(t)) - energy_o) / abs(energy_o)
      for t in np.linspace(t0_et, t0_et + 10 * period, 20)
    ]

    assert max(energy_errors) > 1e-12, \
      "Third-body forces had no effect on energy — check force model is active"


  def test_third_body_raan_drift_differs_j2(self, leo_state, epoch):
    """
    RAAN after 10 orbits must differ between J2-only and J2+third-body propagation.
    """
    gp           = SOLARSYSTEMCONSTANTS.EARTH.GP
    earth_rad_eq = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    sma          = 7000.0e3
    inc          = 45.0 * CONVERTER.RAD_PER_DEG
    vel_mag      = np.sqrt(gp / sma)
    state_o      = np.array([sma, 0.0, 0.0, 0.0, vel_mag * np.cos(inc), vel_mag * np.sin(inc)])
    period       = 2.0 * np.pi * np.sqrt(sma**3 / gp)
    t0_et        = utc_to_et(epoch)

    sh_config = SphericalHarmonicsConfig(coefficients=['J2'], gp=gp, radius=earth_rad_eq)

    accel_j2 = AccelerationSTMDot(
      gravity_config = GravityModelConfig(gp=gp, spherical_harmonics=sh_config),
      spacecraft     = SpacecraftProperties(mass=1000.0),
    )
    sol_j2  = solve_ivp(
      GeneralStateEquationsOfMotion(accel_j2).state_time_derivative,
      (t0_et, t0_et + 10 * period), state_o,
      method='DOP853', rtol=1e-10, atol=1e-12,
    )
    raan_j2 = OrbitConverter.pv_to_coe(sol_j2.y[0:3, -1], sol_j2.y[3:6, -1], gp).raan

    accel_j2_3b = AccelerationSTMDot(
      gravity_config = GravityModelConfig(
        gp                  = gp,
        spherical_harmonics = sh_config,
        third_body          = ThirdBodyConfig(enabled=True, bodies=['MOON', 'SUN']),
      ),
      spacecraft = SpacecraftProperties(mass=1000.0),
    )
    sol_j2_3b  = solve_ivp(
      GeneralStateEquationsOfMotion(accel_j2_3b).state_time_derivative,
      (t0_et, t0_et + 10 * period), state_o,
      method='DOP853', rtol=1e-10, atol=1e-12,
    )
    raan_j2_3b = OrbitConverter.pv_to_coe(sol_j2_3b.y[0:3, -1], sol_j2_3b.y[3:6, -1], gp).raan

    assert not np.isclose(raan_j2, raan_j2_3b, atol=1e-6), \
      "Third-body had no effect on RAAN — check force model is active"


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
