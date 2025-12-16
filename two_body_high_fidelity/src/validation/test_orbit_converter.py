"""
Unit Tests for Orbit Converter Module
=====================================

Tests for conversions between Cartesian state vectors and orbital elements.

Tests:
------
TestCartesianToKeplerian
  - test_known_solution_circular_equatorial_orbit : verify COE for circular equatorial orbit
  - test_known_solution_elliptical_orbit          : verify COE for elliptical orbit at periapsis
  - test_known_solution_inclined_orbit            : verify inclination for inclined orbit
  - test_roundtrip_pv_to_coe_to_pv                : verify pv -> coe -> pv returns original state

TestKeplerianToCartesian
  - test_known_solution_circular_at_ascending_node : verify position at ascending node
  - test_known_solution_velocity_magnitude_circular : verify circular orbit velocity magnitude

TestAnomalyConversions
  - test_known_solution_circular_anomalies_equal : verify TA = EA = MA for circular orbit
  - test_known_solution_anomalies_at_periapsis   : verify all anomalies = 0 at periapsis
  - test_known_solution_anomalies_at_apoapsis    : verify all anomalies = π at apoapsis

Usage:
------
  python -m pytest src/validation/test_orbit_converter.py -v
"""
import pytest
import numpy as np

from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER


class TestCartesianToKeplerian:
  """
  Tests for Cartesian to Keplerian element conversion.
  """
  
  def test_known_solution_circular_equatorial_orbit(self):
    """
    Test conversion for a circular equatorial orbit.
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    pos_mag = 7000e3
    vel_mag = np.sqrt(gp / pos_mag)
    
    pos_vec = np.array([pos_mag, 0.0, 0.0])
    vel_vec = np.array([0.0, vel_mag, 0.0])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['sma'], pos_mag, rtol=1e-10)
    assert np.isclose(coe['ecc'],     0.0, atol=1e-10)
    assert np.isclose(coe['inc'],     0.0, atol=1e-10)
  
  def test_known_solution_elliptical_orbit(self):
    """
    Test conversion for an elliptical orbit at periapsis.
    """
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    periapsis_pos_mag = 7000e3
    apoapsis_pos_mag  = 14000e3
    sma = (periapsis_pos_mag + apoapsis_pos_mag) / 2
    ecc = (apoapsis_pos_mag - periapsis_pos_mag) / (apoapsis_pos_mag + periapsis_pos_mag)
    
    # At periapsis
    periapsis_vel_mag = np.sqrt(gp * (2/periapsis_pos_mag - 1/sma))
    
    pos_vec = np.array([periapsis_pos_mag, 0.0, 0.0])
    vel_vec = np.array([0.0, periapsis_vel_mag, 0.0])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['sma'], sma, rtol=1e-10)
    assert np.isclose(coe['ecc'], ecc, rtol=1e-10)
    assert np.isclose(coe[ 'ta'], 0.0, atol=1e-10) 
  
  def test_known_solution_inclined_orbit(self):
    """
    Test conversion for an inclined orbit.
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    pos_mag = 7000e3
    vel_mag = np.sqrt(gp / pos_mag)
    inc     = 45 * CONVERTER.RAD_PER_DEG
    
    pos_vec = np.array([pos_mag, 0.0, 0.0])
    vel_vec = np.array([0.0, vel_mag * np.cos(inc), vel_mag * np.sin(inc)])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['inc'], inc, rtol=1e-10)
  
  def test_roundtrip_pv_to_coe_to_pv(self):
    """
    Test that pv -> coe -> pv gives the same result.
    """
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Arbitrary state
    pos_vec = np.array([6524.834e3, 6862.875e3, 6448.296e3])
    vel_vec = np.array([4.901327e3, 5.533756e3, -1.976341e3])
    
    # Convert to COE
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    # Convert back to Cartesian
    back_pos_vec, back_vel_vec = OrbitConverter.coe_to_pv(
      coe = coe,
      gp  = gp,
    )
    
    assert np.allclose(pos_vec, back_pos_vec, rtol=1e-10)
    assert np.allclose(vel_vec, back_vel_vec, rtol=1e-10)


class TestKeplerianToCartesian:
  """
  Tests for Keplerian to Cartesian conversion.
  """
  
  def test_known_solution_circular_at_ascending_node(self):
    """
    Test circular orbit at the ascending node.
    """
    gp   = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma  = 7000e3
    ecc  = 0.0
    inc  = 30 * CONVERTER.RAD_PER_DEG
    raan = 0.0
    aop  = 0.0
    ta   = 0.0
    
    coe = {}
    coe[ 'sma'] = sma
    coe[ 'ecc'] = ecc
    coe[ 'inc'] = inc
    coe['raan'] = raan
    coe[ 'aop'] = aop
    coe[  'ta'] = ta

    pos_vec, vel_vec = OrbitConverter.coe_to_pv(
      coe = coe,
      gp  = gp,
    )
    
    # At ascending node with RAAN=0, should be on X-axis
    assert np.isclose(pos_vec[1], 0.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(pos_vec), sma, rtol=1e-10)
  
  def test_known_solution_velocity_magnitude_circular(self):
    """
    Test velocity magnitude for circular orbit.
    """
    gp  = SOLARSYSTEMCONSTANTS.EARTH.GP
    sma = 7000e3
    
    coe = {}
    coe[ 'sma'] = sma
    coe[ 'ecc'] = 0.0
    coe[ 'inc'] = 0.0
    coe['raan'] = 0.0
    coe[ 'aop'] = 0.0
    coe[  'ta'] = 0.0

    pos_vec, vel_vec = OrbitConverter.coe_to_pv(
      coe = coe, 
      gp  = gp,
    )
    
    expected_vel_mag = np.sqrt(gp / sma)
    actual_vel_mag   = np.linalg.norm(vel_vec)
    
    assert np.isclose(actual_vel_mag, expected_vel_mag, rtol=1e-10)


class TestAnomalyConversions:
  """
  Tests for anomaly conversions (TA, EA, MA).
  """
  
  def test_known_solution_circular_anomalies_equal(self):
    """
    For circular orbit, TA = EA = MA.
    """
    gp      = SOLARSYSTEMCONSTANTS.EARTH.GP
    pos_mag = 7000e3
    vel_mag = np.sqrt(gp / pos_mag)
    
    # Position at 45 degrees
    theta   = 45 * CONVERTER.RAD_PER_DEG
    pos_vec = np.array([pos_mag * np.cos(theta), pos_mag * np.sin(theta), 0.0])
    vel_vec = np.array([-vel_mag * np.sin(theta), vel_mag * np.cos(theta), 0.0])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    # For circular orbit, all anomalies should be equal
    assert np.isclose(coe['ta'], coe['ea'], atol=1e-8)
    assert np.isclose(coe['ta'], coe['ma'], atol=1e-8)
  
  def test_known_solution_anomalies_at_periapsis(self):
    """
    At periapsis, all anomalies should be zero.
    """
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Elliptical orbit at periapsis
    periapsis_pos_mag = 7000e3
    apoapsis_pos_mag  = 14000e3
    sma               = (periapsis_pos_mag + apoapsis_pos_mag) / 2
    periapsis_vel_mag = np.sqrt(gp * (2/periapsis_pos_mag - 1/sma))
    
    pos_vec = np.array([periapsis_pos_mag, 0.0, 0.0])
    vel_vec = np.array([0.0, periapsis_vel_mag, 0.0])
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['ta'], 0.0, atol=1e-10)
    assert np.isclose(coe['ea'], 0.0, atol=1e-10)
    assert np.isclose(coe['ma'], 0.0, atol=1e-10)
  
  def test_known_solution_anomalies_at_apoapsis(self):
    """
    At apoapsis, all anomalies should be pi.
    """
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    
    # Elliptical orbit at apoapsis
    periapsis_pos_mag = 7000e3
    apoapsis_pos_mag  = 14000e3
    sma               = (periapsis_pos_mag + apoapsis_pos_mag) / 2
    apoapsis_vel_mag  = np.sqrt(gp * (2/apoapsis_pos_mag - 1/sma))
    
    pos_vec = np.array([-apoapsis_pos_mag, 0.0, 0.0])  # Apoapsis on -X axis
    vel_vec = np.array([0.0, -apoapsis_vel_mag, 0.0])  # Velocity in -Y direction
    
    coe = OrbitConverter.pv_to_coe(pos_vec, vel_vec, gp)
    
    assert np.isclose(coe['ta'], np.pi, atol=1e-10)
    assert np.isclose(coe['ea'], np.pi, atol=1e-10)
    assert np.isclose(coe['ma'], np.pi, atol=1e-10)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
