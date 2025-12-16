"""
Integration Tests for Orbit Propagator
======================================

Tests for the high-fidelity orbit propagator.

Tests:
------
TestKeplerianPropagation
  - test_roundtrip_circular_orbit_period             : verify circular orbit returns to start after one period
  - test_physical_laws_energy_conservation           : verify energy conservation for Keplerian propagation
  - test_physical_laws_angular_momentum_conservation : verify angular momentum conservation for Keplerian propagation

TestJ2Perturbation
  - test_known_solution_j2_raan_drift_direction : verify J2 causes RAAN regression for prograde orbits
  - test_sanity_check_j2_sma_oscillation        : verify J2 causes short-period SMA oscillations but preserves mean

Usage:
------
  python -m pytest src/validation/test_propagator.py -v

Note:
-----
  Some tests require SPICE kernels to be loaded.
"""
import pytest
import numpy as np

from scipy.integrate import solve_ivp

from src.model.dynamics        import Acceleration, GeneralStateEquationsOfMotion
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER


class TestKeplerianPropagation:
  """
  Tests for Keplerian (two-body) propagation.
  """
  
  def test_roundtrip_circular_orbit_period(self):
    """
    Test that circular orbit returns to start after one period.
    """
    # Define circular orbit
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma = 7000e3
    
    # Circular velocity
    vel_mag = np.sqrt(gp / sma)
    
    # Initial state
    state_o = np.array([sma, 0.0, 0.0, 0.0, vel_mag, 0.0])
    
    # Orbital period
    period = 2 * np.pi * np.sqrt(sma**3 / gp)
    
    # Create acceleration model (point mass only)
    accel = Acceleration(gp=gp)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    # Propagate for one period
    sol = solve_ivp(
      fun    = eom.state_time_derivative,
      t_span = (0, period),
      y0     = state_o,
      method = 'DOP853',
      rtol   = 1e-12,
      atol   = 1e-14,
    )
    
    # Extract final state
    state_f = sol.y[:, -1]
    
    # Should return close to initial state
    assert np.allclose(state_o[0:3], state_f[0:3], atol=1e-4)
    assert np.allclose(state_o[3:6], state_f[3:6], atol=1e-4)
  
  def test_physical_laws_energy_conservation(self):
    """
    Test energy conservation for Keplerian propagation.
    """
    # Define elliptical orbit
    gp                = SOLARSYSTEMCONSTANTS.EARTH.GP
    periapsis_pos_mag = 7000e3
    apoapsis_pos_mag  = 14000e3

    # Compute semi-major axis and periapsis velocity
    sma               = (periapsis_pos_mag + apoapsis_pos_mag) / 2
    periapsis_vel_mag = np.sqrt(gp * (2/periapsis_pos_mag - 1/sma))
    
    # Initial state at periapsis
    state_o = np.array([periapsis_pos_mag, 0.0, 0.0, 0.0, periapsis_vel_mag, 0.0])
    
    # Calculate initial energy using physics formula (not OrbitConverter)
    def _specific_energy(state):
      pos_mag = np.linalg.norm(state[0:3])
      vel_mag = np.linalg.norm(state[3:6])
      return 0.5 * vel_mag**2 - gp / pos_mag
    
    specific_energy_o = _specific_energy(state_o)

    # Initialize propagation
    accel = Acceleration(gp=gp)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    period = 2 * np.pi * np.sqrt(sma**3 / gp)
    
    # Propagate
    sol = solve_ivp(
      fun          = eom.state_time_derivative,
      t_span       = (0, period),
      y0           = state_o,
      method       = 'DOP853',
      rtol         = 1e-12,
      atol         = 1e-14,
      dense_output = True,
    )
    
    # Check energy at multiple points
    for t in np.linspace(0, period, 10):
      state           = sol.sol(t)
      specific_energy = _specific_energy(state)
      # Energy should be conserved to high precision
      assert np.isclose(specific_energy, specific_energy_o, rtol=1e-10)
  
  def test_physical_laws_angular_momentum_conservation(self):
    """
    Test angular momentum conservation for Keplerian propagation.
    """
    
    # Define inclined and elliptical orbit
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    state_o = np.array([7000e3, 0.0, 0.0, 0.0, 6000.0, 4000.0])
    
    # Calculate angular momentum using physics formula (not OrbitConverter)
    def _angular_momentum(state):
      return np.cross(state[0:3], state[3:6])
    
    ang_mom_mag_o = _angular_momentum(state_o)
    
    # Intialize propagation
    accel = Acceleration(gp=gp)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    pos_mag = np.linalg.norm(state_o[0:3])
    vel_mag = np.linalg.norm(state_o[3:6])
    sma     = 1.0 / (2.0/pos_mag - vel_mag**2/gp)
    period  = 2 * np.pi * np.sqrt(sma**3 / gp)

    # Propagation
    sol = solve_ivp(
      fun          = eom.state_time_derivative,
      t_span       = (0, period),
      y0           = state_o,
      method       = 'DOP853',
      rtol         = 1e-12,
      atol         = 1e-14,
      dense_output = True,
    )
    
    # Check angular momentum magnitude at multiple points
    #   - Use magnitude comparison to avoid issues with near-zero components
    for t in np.linspace(0, period, 10):
      state = sol.sol(t)
      ang_mom_vec = _angular_momentum(state)
      ang_mom_mag = np.linalg.norm(ang_mom_vec)
      assert np.isclose(ang_mom_mag, np.linalg.norm(ang_mom_mag_o), rtol=1e-10)


class TestJ2Perturbation:
  """
  Tests for J2 perturbed propagation.
  """
  
  def test_known_solution_j2_raan_drift_direction(self):
    """
    Test that J2 causes RAAN regression for prograde orbits.
    """
    # Retrieve constants
    gp           = SOLARSYSTEMCONSTANTS.EARTH.GP
    j2           = SOLARSYSTEMCONSTANTS.EARTH.J2
    earth_rad_eq = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    
    # Define inclined circular orbit
    sma     = 7000e3
    inc     = 45 * CONVERTER.RAD_PER_DEG
    vel_mag = np.sqrt(gp / sma)
    
    state_o = np.array([sma, 0.0, 0.0, 0.0, vel_mag * np.cos(inc), vel_mag * np.sin(inc)])
    
    coe_o = OrbitConverter.pv_to_coe(state_o[0:3], state_o[3:6], gp)
    
    # Initialize propagation
    accel = Acceleration(gp=gp, j2=j2, pos_ref=earth_rad_eq)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    period = 2 * np.pi * np.sqrt(sma**3 / gp)
    
    # Propagate
    sol = solve_ivp(
      fun    = eom.state_time_derivative,
      t_span = (0, 10 * period),  # 10 orbits
      y0     = state_o,
      method = 'DOP853',
      rtol   = 1e-10,
      atol   = 1e-12,
    )
    
    state_f = sol.y[:, -1]
    coe_f   = OrbitConverter.pv_to_coe(state_f[0:3], state_f[3:6], gp)
    
    # For prograde orbit, RAAN should decrease (regress westward)
    # Handle angle wrapping
    delta_raan = coe_f['raan'] - coe_o['raan']
    if delta_raan > np.pi:
      delta_raan -= 2 * np.pi
    elif delta_raan < -np.pi:
      delta_raan += 2 * np.pi
    
    assert delta_raan < 0  # RAAN should decrease
  
  def test_sanity_check_j2_sma_oscillation(self):
    """
    Test that J2 causes short-period SMA oscillations but preserves mean.
    """

    # Retrieve constant
    gp           = SOLARSYSTEMCONSTANTS.EARTH.GP
    j2           = SOLARSYSTEMCONSTANTS.EARTH.J2
    earth_rad_eq = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    
    # Define orbit
    sma     = 7000e3
    inc     = 45 * CONVERTER.RAD_PER_DEG
    vel_mag = np.sqrt(gp / sma)
    
    # Initialize propagation
    state_o = np.array([sma, 0.0, 0.0, 0.0, vel_mag * np.cos(inc), vel_mag * np.sin(inc)])
    
    accel = Acceleration(gp=gp, j2=j2, pos_ref=earth_rad_eq)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    period = 2 * np.pi * np.sqrt(sma**3 / gp)
    
    # Propagate
    sol = solve_ivp(
      fun          = eom.state_time_derivative,
      t_span       = (0, period),
      y0           = state_o,
      method       = 'DOP853',
      rtol         = 1e-10,
      atol         = 1e-12,
      dense_output = True,
    )
    
    # Sample SMA throughout orbit
    sma_values = []
    for t in np.linspace(0, period, 100):
      state = sol.sol(t)
      coe   = OrbitConverter.pv_to_coe(state[0:3], state[3:6], gp)
      sma_values.append(coe['sma'])
    
    sma_values = np.array(sma_values)
    
    # SMA should oscillate but mean should be close to initial
    mean_sma = np.mean(sma_values)
    assert np.isclose(mean_sma, sma, rtol=1e-3)  # 0.1% tolerance
    
    # There should be oscillation (not constant)
    assert np.std(sma_values) > 0


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
