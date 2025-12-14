"""
Integration Tests for Orbit Propagator
======================================

Tests for the high-fidelity orbit propagator.

Run with:
  pytest src/validation/test_propagator.py -v

Note: Some tests require SPICE kernels to be loaded.
"""
import pytest
import numpy as np

from src.model.dynamics        import Acceleration, GeneralStateEquationsOfMotion
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER


class TestKeplerianPropagation:
  """Tests for Keplerian (two-body) propagation."""
  
  def test_circular_orbit_period(self):
    """Test that circular orbit returns to start after one period."""
    from scipy.integrate import solve_ivp
    
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma = 7000e3
    
    # Circular velocity
    v = np.sqrt(gp / sma)
    
    # Initial state
    state_0 = np.array([sma, 0.0, 0.0, 0.0, v, 0.0])
    
    # Orbital period
    period = 2 * np.pi * np.sqrt(sma**3 / gp)
    
    # Create acceleration model (point mass only)
    accel = Acceleration(gp=gp)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    # Propagate for one period
    sol = solve_ivp(
      fun        = eom.state_time_derivative,
      t_span     = (0, period),
      y0         = state_0,
      method     = 'DOP853',
      rtol       = 1e-12,
      atol       = 1e-14,
    )
    
    state_f = sol.y[:, -1]
    
    # Should return close to initial state
    assert np.allclose(state_0[0:3], state_f[0:3], rtol=1e-6)
    assert np.allclose(state_0[3:6], state_f[3:6], rtol=1e-6)
  
  def test_energy_conservation_keplerian(self):
    """Test energy conservation for Keplerian propagation."""
    from scipy.integrate import solve_ivp
    
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Elliptical orbit
    rp = 7000e3
    ra = 14000e3
    a  = (rp + ra) / 2
    vp = np.sqrt(gp * (2/rp - 1/a))
    
    state_0 = np.array([rp, 0.0, 0.0, 0.0, vp, 0.0])
    
    # Calculate initial energy
    def specific_energy(state):
      r = np.linalg.norm(state[0:3])
      v = np.linalg.norm(state[3:6])
      return 0.5 * v**2 - gp / r
    
    E_0 = specific_energy(state_0)
    
    # Propagate
    accel = Acceleration(gp=gp)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    period = 2 * np.pi * np.sqrt(a**3 / gp)
    
    sol = solve_ivp(
      fun        = eom.state_time_derivative,
      t_span     = (0, period),
      y0         = state_0,
      method     = 'DOP853',
      rtol       = 1e-12,
      atol       = 1e-14,
      dense_output = True,
    )
    
    # Check energy at multiple points
    for t in np.linspace(0, period, 10):
      state = sol.sol(t)
      E     = specific_energy(state)
      # Energy should be conserved to high precision
      assert np.isclose(E, E_0, rtol=1e-10)
  
  def test_angular_momentum_conservation(self):
    """Test angular momentum conservation for Keplerian propagation."""
    from scipy.integrate import solve_ivp
    
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Inclined elliptical orbit
    state_0 = np.array([7000e3, 0.0, 0.0, 0.0, 6000.0, 4000.0])
    
    def angular_momentum(state):
      return np.cross(state[0:3], state[3:6])
    
    h_0 = angular_momentum(state_0)
    
    # Propagate
    accel = Acceleration(gp=gp)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    # Estimate period
    coe = OrbitConverter.pv_to_coe(state_0[0:3], state_0[3:6], gp)
    period = 2 * np.pi * np.sqrt(coe['sma']**3 / gp)
    
    sol = solve_ivp(
      fun          = eom.state_time_derivative,
      t_span       = (0, period),
      y0           = state_0,
      method       = 'DOP853',
      rtol         = 1e-12,
      atol         = 1e-14,
      dense_output = True,
    )
    
    # Check angular momentum at multiple points
    for t in np.linspace(0, period, 10):
      state = sol.sol(t)
      h     = angular_momentum(state)
      assert np.allclose(h, h_0, rtol=1e-10)


class TestJ2Perturbation:
  """Tests for J2 perturbed propagation."""
  
  def test_j2_raan_drift_direction(self):
    """Test that J2 causes RAAN regression for prograde orbits."""
    from scipy.integrate import solve_ivp
    
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    j2  = SOLARSYSTEMCONSTANTS.EARTH.J2
    req = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    
    # Inclined circular orbit
    sma = 7000e3
    inc = 45 * CONVERTER.RAD_PER_DEG
    v   = np.sqrt(gp / sma)
    
    state_0 = np.array([sma, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)])
    
    coe_0 = OrbitConverter.pv_to_coe(state_0[0:3], state_0[3:6], gp)
    
    # Propagate with J2
    accel = Acceleration(gp=gp, j2=j2, pos_ref=req)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    period = 2 * np.pi * np.sqrt(sma**3 / gp)
    
    sol = solve_ivp(
      fun    = eom.state_time_derivative,
      t_span = (0, 10 * period),  # 10 orbits
      y0     = state_0,
      method = 'DOP853',
      rtol   = 1e-10,
      atol   = 1e-12,
    )
    
    state_f = sol.y[:, -1]
    coe_f   = OrbitConverter.pv_to_coe(state_f[0:3], state_f[3:6], gp)
    
    # For prograde orbit, RAAN should decrease (regress westward)
    # Handle angle wrapping
    delta_raan = coe_f['raan'] - coe_0['raan']
    if delta_raan > np.pi:
      delta_raan -= 2 * np.pi
    elif delta_raan < -np.pi:
      delta_raan += 2 * np.pi
    
    assert delta_raan < 0  # RAAN should decrease
  
  def test_j2_sma_oscillation(self):
    """Test that J2 causes short-period SMA oscillations but preserves mean."""
    from scipy.integrate import solve_ivp
    
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    j2  = SOLARSYSTEMCONSTANTS.EARTH.J2
    req = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    
    sma = 7000e3
    inc = 45 * CONVERTER.RAD_PER_DEG
    v   = np.sqrt(gp / sma)
    
    state_0 = np.array([sma, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)])
    
    # Propagate with J2
    accel = Acceleration(gp=gp, j2=j2, pos_ref=req)
    eom   = GeneralStateEquationsOfMotion(accel)
    
    period = 2 * np.pi * np.sqrt(sma**3 / gp)
    
    sol = solve_ivp(
      fun          = eom.state_time_derivative,
      t_span       = (0, period),
      y0           = state_0,
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
    assert np.isclose(mean_sma, sma, rtol=1e-4)
    
    # There should be oscillation (not constant)
    assert np.std(sma_values) > 0


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
