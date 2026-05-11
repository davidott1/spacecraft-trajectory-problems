"""
Integration Tests for Orbit Determination Module
==================================================

Tests for EKF and RTS smoother using simulated measurements.
Requires SPICE kernels.

Tests:
------
TestEKF
  - test_ekf_reduces_covariance     : EKF filter covariance decreases as measurements are processed
  - test_rts_smoother_reduces_covariance : RTS smoother covariance is <= filter covariance everywhere

Usage:
------
  python -m pytest src/verification/integration/test_od.py -v
"""
import pytest
import numpy as np

from datetime import datetime, timedelta
from scipy.integrate import solve_ivp

from src.model.dynamics                             import AccelerationSTMDot, GeneralStateEquationsOfMotion
from src.model.constants                            import SOLARSYSTEMCONSTANTS, CONVERTER
from src.model.time_converter                       import utc_to_et
from src.schemas.gravity                            import GravityModelConfig
from src.schemas.spacecraft                         import SpacecraftProperties
from src.schemas.propagation                        import PropagationResult
from src.schemas.state                              import (
  TrackerStation, TrackerPosition, TrackerPerformance,
  TrackerConstraints, TrackerUncertainty, ElevationLimits, RangeLimits,
)
from src.schemas.time                               import TimeStructure
from src.orbit_determination.ekf_processor         import (
  process_measurements_with_ekf,
  apply_rts_smoother,
  create_default_initial_covariance,
  create_default_process_noise,
)
from src.orbit_determination.measurement_simulator import MeasurementSimulator


EPOCH     = datetime(2025, 10, 1)
DURATION  = timedelta(minutes=10)


class TestEKF:
  """
  Integration tests for EKF and RTS smoother with simulated LEO measurements.
  Requires SPICE kernels for frame conversions.
  """

  @pytest.fixture(autouse=True)
  def _load_spice(self, load_spice_kernels):
    pass

  @pytest.fixture
  def epoch(self):
    return EPOCH

  @pytest.fixture
  def keplerian_dynamics(self):
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    return GeneralStateEquationsOfMotion(
      AccelerationSTMDot(
        gravity_config = GravityModelConfig(gp=gp),
        spacecraft     = SpacecraftProperties(mass=1000.0),
      )
    )

  @pytest.fixture
  def leo_truth_result(self, epoch, keplerian_dynamics):
    """
    Propagate a circular LEO orbit for DURATION to use as truth.
    Returns a PropagationResult with state (6, N) and time grid.
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma     = 7000.0e3
    vel_mag = np.sqrt(gp / sma)
    inc     = 51.6 * CONVERTER.RAD_PER_DEG
    state_o = np.array([sma, 0.0, 0.0, 0.0, vel_mag * np.cos(inc), vel_mag * np.sin(inc)])

    t0_et = utc_to_et(epoch)
    t_rel = np.arange(0.0, DURATION.total_seconds() + 1.0, 60.0)  # 1-min steps
    t_et  = t0_et + t_rel

    sol = solve_ivp(
      keplerian_dynamics.state_time_derivative,
      (t_et[0], t_et[-1]),
      state_o,
      method       = 'DOP853',
      t_eval       = t_et,
      rtol         = 1e-10,
      atol         = 1e-12,
      dense_output = False,
    )

    time_struct = TimeStructure(initial=epoch, grid_relative_initial=t_rel)

    return PropagationResult(
      success = True,
      message = "truth propagation",
      time    = time_struct,
      state   = sol.y,
    )

  @pytest.fixture
  def tracker(self):
    """Single ground station positioned directly under the initial orbit position."""
    return TrackerStation(
      name     = "TestStation",
      position = TrackerPosition(
        latitude  = np.radians(0.0),   # directly under initial state at [sma, 0, 0]
        longitude = np.radians(0.0),
        altitude  = 0.0,
      ),
      performance = TrackerPerformance(
        constraints = TrackerConstraints(
          elevation = ElevationLimits(min=np.radians(5.0), max=np.radians(90.0)),
          range     = RangeLimits(min=0.0, max=2000.0e3),
        ),
        uncertainty = TrackerUncertainty(
          range          = 10.0,
          range_rate     = 0.1,
          azimuth        = np.radians(0.1),
          azimuth_rate   = np.radians(0.01),
          elevation      = np.radians(0.1),
          elevation_rate = np.radians(0.01),
        ),
      ),
    )

  @pytest.fixture
  def measurements(self, leo_truth_result, tracker, epoch):
    """Simulate noisy measurements from the truth result."""
    simulator    = MeasurementSimulator(leo_truth_result, tracker, epoch)
    noise_config = simulator.get_tracker_noise_config()
    meas         = simulator.simulate(noise_config=noise_config, seed=42, include_rates=True)
    meas.truth    = meas.get_visible_truth()
    meas.measured = meas.get_visible_measured()
    return meas

  def test_ekf_reduces_covariance(self, leo_truth_result, tracker, measurements, epoch):
    """
    EKF filter position covariance trace must decrease after processing
    measurements relative to the initial covariance.
    """
    pytest.importorskip("src.orbit_determination.ekf_processor")

    if measurements.n_visible == 0:
      pytest.skip("No visible measurements from tracker — increase DURATION or reposition station")

    gp            = SOLARSYSTEMCONSTANTS.EARTH.GP
    initial_state = leo_truth_result.state[:, 0]
    ephem_times   = leo_truth_result.time.grid.relative_initial
    # Large initial uncertainty so EKF has room to converge.
    # Zero process noise: Keplerian dynamics are exact for our simulated truth,
    # so covariance should only decrease as measurements are processed.
    initial_cov   = create_default_initial_covariance(position_sigma=1000.0, velocity_sigma=10.0)
    process_noise = np.zeros((6, 6))

    filter_result, filter_covs, estimation_times, _ = process_measurements_with_ekf(
      measurements       = measurements,
      tracker            = tracker,
      initial_state      = initial_state,
      epoch_dt_utc       = epoch,
      ephemeris_times    = ephem_times,
      initial_covariance = initial_cov,
      process_noise      = process_noise,
    )

    pos_var_initial = initial_cov[0, 0] + initial_cov[1, 1] + initial_cov[2, 2]
    pos_var_series  = filter_covs[0, 0, :] + filter_covs[1, 1, :] + filter_covs[2, 2, :]
    pos_var_min     = float(np.min(pos_var_series))

    assert pos_var_min < pos_var_initial, (
      f"EKF position variance never decreased below initial: "
      f"initial={pos_var_initial:.2f} m², min achieved={pos_var_min:.2f} m²"
    )

  def test_rts_smoother_reduces_covariance(self, leo_truth_result, tracker, measurements, epoch, keplerian_dynamics):
    """
    RTS smoother position covariance must be <= filter covariance at every
    estimation time (smoother uses past and future measurements).
    """
    if measurements.n_visible == 0:
      pytest.skip("No visible measurements from tracker — increase DURATION or reposition station")

    initial_state = leo_truth_result.state[:, 0]
    ephem_times   = leo_truth_result.time.grid.relative_initial
    initial_cov   = create_default_initial_covariance(position_sigma=1000.0, velocity_sigma=10.0)
    process_noise = np.zeros((6, 6))

    filter_result, filter_covs, estimation_times, _ = process_measurements_with_ekf(
      measurements       = measurements,
      tracker            = tracker,
      initial_state      = initial_state,
      epoch_dt_utc       = epoch,
      ephemeris_times    = ephem_times,
      initial_covariance = initial_cov,
      process_noise      = process_noise,
    )

    smoother_result, smoother_covs = apply_rts_smoother(
      filter_result        = filter_result,
      filtered_covariances = filter_covs,
      estimation_times     = estimation_times,
      epoch_dt_utc         = epoch,
      dynamics             = keplerian_dynamics,
      process_noise        = process_noise,
    )

    filter_pos_var   = filter_covs[0, 0, :]   + filter_covs[1, 1, :]   + filter_covs[2, 2, :]
    smoother_pos_var = smoother_covs[0, 0, :] + smoother_covs[1, 1, :] + smoother_covs[2, 2, :]

    # Smoother variance must be <= filter variance at every time
    violations = np.sum(smoother_pos_var > filter_pos_var + 1e-6)
    assert violations == 0, (
      f"RTS smoother variance exceeded filter variance at {violations} time steps"
    )
