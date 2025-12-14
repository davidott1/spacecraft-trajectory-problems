"""
Unit Tests for Orbit Converter Module
=====================================

Tests for conversions between Cartesian state vectors and orbital elements.

Run with:
  pytest src/validation/test_orbit_converter.py -v
"""
import pytest
import numpy as np

from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER


class TestCartesianToKeplerian:
  """Tests for Cartesian to Keplerian element conversion."""
  
  def test_circular_equatorial_orbit(self):
    """Test conversion for a circular equatorial orbit."""
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    r  = 7000e3
    v  = np.sqrt(gp / r)
    
    pos_vec = np.array([r, 0.0, 0.0])
    vel_vec = np.array([0.0, v, 0.0])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['sma'], r, rtol=1e-10)
    assert np.isclose(coe['ecc'], 0.0, atol=1e-10)
    assert np.isclose(coe['inc'], 0.0, atol=1e-10)
  
  def test_elliptical_orbit(self):
    """Test conversion for an elliptical orbit."""
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Periapsis at 7000 km, apoapsis at 14000 km
    rp = 7000e3
    ra = 14000e3
    a  = (rp + ra) / 2
    e  = (ra - rp) / (ra + rp)
    
    # At periapsis
    v_periapsis = np.sqrt(gp * (2/rp - 1/a))
    
    pos_vec = np.array([rp, 0.0, 0.0])
    vel_vec = np.array([0.0, v_periapsis, 0.0])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['sma'], a, rtol=1e-10)
    assert np.isclose(coe['ecc'], e, rtol=1e-10)
    assert np.isclose(coe['ta'], 0.0, atol=1e-10)  # At periapsis, TA = 0
  
  def test_inclined_orbit(self):
    """Test conversion for an inclined orbit."""
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    r   = 7000e3
    v   = np.sqrt(gp / r)
    inc = 45 * CONVERTER.RAD_PER_DEG
    
    pos_vec = np.array([r, 0.0, 0.0])
    vel_vec = np.array([0.0, v * np.cos(inc), v * np.sin(inc)])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['inc'], inc, rtol=1e-10)
  
  def test_roundtrip_conversion(self):
    """Test that pv -> coe -> pv gives the same result."""
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Arbitrary state
    pos_vec = np.array([6524.834e3, 6862.875e3, 6448.296e3])
    vel_vec = np.array([4.901327e3, 5.533756e3, -1.976341e3])
    
    # Convert to COE
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    # Convert back to Cartesian
    pos_vec_back, vel_vec_back = OrbitConverter.coe_to_pv(
      sma  = coe['sma'],
      ecc  = coe['ecc'],
      inc  = coe['inc'],
      raan = coe['raan'],
      aop  = coe['aop'],
      ta   = coe['ta'],
      gp   = gp,
    )
    
    assert np.allclose(pos_vec, pos_vec_back, rtol=1e-10)
    assert np.allclose(vel_vec, vel_vec_back, rtol=1e-10)


class TestKeplerianToCartesian:
  """Tests for Keplerian to Cartesian conversion."""
  
  def test_circular_orbit_at_ascending_node(self):
    """Test circular orbit at the ascending node."""
    gp   = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma  = 7000e3
    ecc  = 0.0
    inc  = 30 * CONVERTER.RAD_PER_DEG
    raan = 0.0
    aop  = 0.0
    ta   = 0.0
    
    pos_vec, vel_vec = OrbitConverter.coe_to_pv(sma, ecc, inc, raan, aop, ta, gp)
    
    # At ascending node with RAAN=0, should be on X-axis
    assert np.isclose(pos_vec[1], 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(pos_vec), sma, rtol=1e-10)
  
  def test_velocity_magnitude_circular(self):
    """Test velocity magnitude for circular orbit."""
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma = 7000e3
    
    pos_vec, vel_vec = OrbitConverter.coe_to_pv(
      sma=sma, ecc=0.0, inc=0.0, raan=0.0, aop=0.0, ta=0.0, gp=gp
    )
    
    expected_v = np.sqrt(gp / sma)
    actual_v   = np.linalg.norm(vel_vec)
    
    assert np.isclose(actual_v, expected_v, rtol=1e-10)


class TestAnomalyConversions:
  """Tests for anomaly conversions (TA, EA, MA)."""
  
  def test_circular_orbit_anomalies_equal(self):
    """For circular orbit, TA = EA = MA."""
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    r  = 7000e3
    v  = np.sqrt(gp / r)
    
    # Position at 45 degrees
    theta   = 45 * CONVERTER.RAD_PER_DEG
    pos_vec = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
    vel_vec = np.array([-v * np.sin(theta), v * np.cos(theta), 0.0])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    # For circular orbit, all anomalies should be equal
    assert np.isclose(coe['ta'], coe['ea'], atol=1e-8)
    assert np.isclose(coe['ta'], coe['ma'], atol=1e-8)
  
  def test_anomalies_at_periapsis(self):
    """At periapsis, all anomalies should be zero."""
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Elliptical orbit at periapsis
    rp = 7000e3
    ra = 14000e3
    a  = (rp + ra) / 2
    vp = np.sqrt(gp * (2/rp - 1/a))
    
    pos_vec = np.array([rp, 0.0, 0.0])
    vel_vec = np.array([0.0, vp, 0.0])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['ta'], 0.0, atol=1e-10)
    assert np.isclose(coe['ea'], 0.0, atol=1e-10)
    assert np.isclose(coe['ma'], 0.0, atol=1e-10)
  
  def test_anomalies_at_apoapsis(self):
    """At apoapsis, all anomalies should be pi."""
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Elliptical orbit at apoapsis
    rp = 7000e3
    ra = 14000e3
    a  = (rp + ra) / 2
    va = np.sqrt(gp * (2/ra - 1/a))
    
    pos_vec = np.array([-ra, 0.0, 0.0])  # Apoapsis on -X axis
    vel_vec = np.array([0.0, -va, 0.0])  # Velocity in -Y direction
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['ta'], np.pi, atol=1e-10)
    assert np.isclose(coe['ea'], np.pi, atol=1e-10)
    assert np.isclose(coe['ma'], np.pi, atol=1e-10)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
