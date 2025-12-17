"""
Dynamics Module Tests
=====================

Regression tests for gravity, drag, and SRP models.
"""
import pytest
import numpy as np

from src.model.dynamics import (
  TwoBodyGravity,
  ThirdBodyGravity,
  Gravity,
  AtmosphericDrag,
  SolarRadiationPressure,
  Acceleration,
  GeneralStateEquationsOfMotion,
)
from src.model.constants import SOLARSYSTEMCONSTANTS


class TestTwoBodyGravity:
  """Tests for TwoBodyGravity class."""
  
  def test_point_mass_acceleration_magnitude(self):
    """Test that point mass acceleration has correct magnitude."""
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    gravity = TwoBodyGravity(gp=gp)
    
    # Position at 7000 km altitude
    pos_vec = np.array([7000.0e3, 0.0, 0.0])
    acc_vec = gravity.point_mass(pos_vec)
    
    # Expected: a = -mu/r^2 in radial direction
    expected_mag = gp / (7000.0e3)**2
    actual_mag = np.linalg.norm(acc_vec)
    
    assert np.isclose(actual_mag, expected_mag, rtol=1e-10)
  
  def test_point_mass_acceleration_direction(self):
    """Test that point mass acceleration points toward Earth center."""
    gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    gravity = TwoBodyGravity(gp=gp)
    
    pos_vec = np.array([7000.0e3, 0.0, 0.0])
    acc_vec = gravity.point_mass(pos_vec)
    
    # Acceleration should be in -x direction (toward origin)
    assert acc_vec[0] < 0
    assert np.isclose(acc_vec[1], 0.0)
    assert np.isclose(acc_vec[2], 0.0)
  
  def test_j2_acceleration_nonzero(self):
    """Test that J2 produces non-zero acceleration."""
    gravity = TwoBodyGravity(
      gp=SOLARSYSTEMCONSTANTS.EARTH.GP,
      j2=SOLARSYSTEMCONSTANTS.EARTH.J2,
      pos_ref=SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    )
    
    # Position with non-zero z component
    pos_vec = np.array([5000.0e3, 3000.0e3, 4000.0e3])
    acc_vec = gravity.oblate_j2(0.0, pos_vec)
    
    assert np.linalg.norm(acc_vec) > 0
  
  def test_j2_zero_when_disabled(self):
    """Test that J2 returns zero when coefficient is zero."""
    gravity = TwoBodyGravity(gp=SOLARSYSTEMCONSTANTS.EARTH.GP, j2=0.0)
    
    pos_vec = np.array([7000.0e3, 0.0, 0.0])
    acc_vec = gravity.oblate_j2(0.0, pos_vec)
    
    assert np.allclose(acc_vec, np.zeros(3))
  
  def test_j2_magnitude_order(self):
    """Test that J2 acceleration is much smaller than point mass."""
    gravity = TwoBodyGravity(
      gp=SOLARSYSTEMCONSTANTS.EARTH.GP,
      j2=SOLARSYSTEMCONSTANTS.EARTH.J2,
      pos_ref=SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    )
    
    pos_vec = np.array([7000.0e3, 0.0, 1000.0e3])
    
    acc_point = gravity.point_mass(pos_vec)
    acc_j2 = gravity.oblate_j2(0.0, pos_vec)
    
    # J2 should be ~1000x smaller than point mass
    ratio = np.linalg.norm(acc_j2) / np.linalg.norm(acc_point)
    assert ratio < 0.01  # Less than 1%


class TestAtmosphericDrag:
  """Tests for AtmosphericDrag class."""
  
  def test_drag_zero_at_high_altitude(self):
    """Test that drag is negligible at high altitude."""
    drag = AtmosphericDrag(cd=2.2, area=10.0, mass=1000.0)
    
    # GEO altitude - negligible atmosphere
    pos_vec = np.array([42164.0e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 3075.0, 0.0])
    
    acc_vec = drag.compute(pos_vec, vel_vec)
    
    # Should be essentially zero
    assert np.linalg.norm(acc_vec) < 1e-20
  
  def test_drag_opposes_velocity(self):
    """Test that drag acceleration opposes relative velocity."""
    drag = AtmosphericDrag(cd=2.2, area=10.0, mass=1000.0)
    
    # LEO position and velocity
    pos_vec = np.array([6678.0e3, 0.0, 0.0])  # ~300 km altitude
    vel_vec = np.array([0.0, 7726.0, 0.0])
    
    acc_vec = drag.compute(pos_vec, vel_vec)
    
    # Drag should have negative component in velocity direction
    # (accounting for Earth rotation)
    assert acc_vec[1] < 0  # Opposing the +y velocity
  
  def test_drag_increases_with_lower_altitude(self):
    """Test that drag increases at lower altitudes."""
    drag = AtmosphericDrag(cd=2.2, area=10.0, mass=1000.0)
    
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    # Higher altitude
    pos_high = np.array([6800.0e3, 0.0, 0.0])
    acc_high = drag.compute(pos_high, vel_vec)
    
    # Lower altitude
    pos_low = np.array([6500.0e3, 0.0, 0.0])
    acc_low = drag.compute(pos_low, vel_vec)
    
    assert np.linalg.norm(acc_low) > np.linalg.norm(acc_high)


class TestAcceleration:
  """Tests for the Acceleration coordinator class."""
  
  def test_two_body_only(self):
    """Test acceleration with only two-body gravity."""
    acc_model = Acceleration(
      gp=SOLARSYSTEMCONSTANTS.EARTH.GP,
    )
    
    pos_vec = np.array([7000.0e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    acc_vec = acc_model.compute(0.0, pos_vec, vel_vec)
    
    # Should be non-zero and pointing toward Earth
    assert np.linalg.norm(acc_vec) > 0
    assert acc_vec[0] < 0
  
  def test_with_j2(self):
    """Test acceleration with J2 enabled."""
    acc_model = Acceleration(
      gp=SOLARSYSTEMCONSTANTS.EARTH.GP,
      j2=SOLARSYSTEMCONSTANTS.EARTH.J2,
      pos_ref=SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
    )
    
    pos_vec = np.array([7000.0e3, 0.0, 1000.0e3])
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    acc_vec = acc_model.compute(0.0, pos_vec, vel_vec)
    
    # Compare to two-body only
    acc_two_body = Acceleration(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)
    acc_two_body_vec = acc_two_body.compute(0.0, pos_vec, vel_vec)
    
    # Should be different due to J2
    assert not np.allclose(acc_vec, acc_two_body_vec)


class TestGeneralStateEquationsOfMotion:
  """Tests for the equations of motion class."""
  
  def test_state_derivative_shape(self):
    """Test that state derivative has correct shape."""
    acc_model = Acceleration(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)
    eom = GeneralStateEquationsOfMotion(acc_model)
    
    state = np.array([7000.0e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
    state_dot = eom.state_time_derivative(0.0, state)
    
    assert state_dot.shape == (6,)
  
  def test_velocity_in_derivative(self):
    """Test that velocity appears in position derivative."""
    acc_model = Acceleration(gp=SOLARSYSTEMCONSTANTS.EARTH.GP)
    eom = GeneralStateEquationsOfMotion(acc_model)
    
    vel = np.array([100.0, 200.0, 300.0])
    state = np.array([7000.0e3, 0.0, 0.0, vel[0], vel[1], vel[2]])
    state_dot = eom.state_time_derivative(0.0, state)
    
    # Position derivative should equal velocity
    assert np.allclose(state_dot[0:3], vel)
