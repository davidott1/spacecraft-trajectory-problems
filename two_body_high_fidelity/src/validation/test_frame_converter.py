"""
Unit Tests for Frame Converter Module
=====================================

Tests for reference frame transformations (J2000, TEME, RIC/RTN).

Run with:
  pytest src/validation/test_frame_converter.py -v
"""
import pytest
import numpy as np

from src.model.frame_converter import FrameConverter, VectorConverter


class TestRICFrameConversion:
  """Tests for RIC (Radial-Intrack-Crosstrack) frame conversions."""
  
  def test_ric_axes_orthonormal(self):
    """RIC rotation matrix should be orthonormal."""
    pos_vec = np.array([7000e3, 1000e3, 500e3])
    vel_vec = np.array([-500.0, 7000.0, 1000.0])
    
    R = FrameConverter.xyz_to_ric(pos_vec, vel_vec)
    
    # Check orthonormality: R @ R.T = I
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)
    
    # Check determinant = 1 (proper rotation)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-14)
  
  def test_radial_direction(self):
    """R-axis should point along position vector."""
    pos_vec = np.array([7000e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    R = FrameConverter.xyz_to_ric(pos_vec, vel_vec)
    
    # First row of R is the radial unit vector
    r_hat = R[0, :]
    expected_r_hat = pos_vec / np.linalg.norm(pos_vec)
    
    assert np.allclose(r_hat, expected_r_hat, atol=1e-14)
  
  def test_crosstrack_direction(self):
    """C-axis should be along angular momentum."""
    pos_vec = np.array([7000e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    R = FrameConverter.xyz_to_ric(pos_vec, vel_vec)
    
    # Third row of R is the crosstrack unit vector (h direction)
    c_hat = R[2, :]
    
    h_vec          = np.cross(pos_vec, vel_vec)
    expected_c_hat = h_vec / np.linalg.norm(h_vec)
    
    assert np.allclose(c_hat, expected_c_hat, atol=1e-14)
  
  def test_roundtrip_xyz_ric_xyz(self):
    """XYZ -> RIC -> XYZ should give original vector."""
    pos_ref = np.array([7000e3, 1000e3, 500e3])
    vel_ref = np.array([-500.0, 7000.0, 1000.0])
    
    R_xyz_to_ric = FrameConverter.xyz_to_ric(pos_ref, vel_ref)
    R_ric_to_xyz = FrameConverter.ric_to_xyz(pos_ref, vel_ref)
    
    # Test roundtrip
    test_vec     = np.array([1000.0, 2000.0, 500.0])
    ric_vec      = R_xyz_to_ric @ test_vec
    xyz_vec_back = R_ric_to_xyz @ ric_vec
    
    assert np.allclose(test_vec, xyz_vec_back, atol=1e-10)
  
  def test_rtn_alias(self):
    """RTN should be an alias for RIC."""
    pos_vec = np.array([7000e3, 1000e3, 500e3])
    vel_vec = np.array([-500.0, 7000.0, 1000.0])
    
    R_ric = FrameConverter.xyz_to_ric(pos_vec, vel_vec)
    R_rtn = FrameConverter.xyz_to_rtn(pos_vec, vel_vec)
    
    assert np.allclose(R_ric, R_rtn, atol=1e-14)


class TestVectorConverter:
  """Tests for VectorConverter class."""
  
  def test_position_error_in_ric(self):
    """Test converting position error to RIC frame."""
    # Reference state
    pos_ref = np.array([7000e3, 0.0, 0.0])
    vel_ref = np.array([0.0, 7500.0, 0.0])
    
    # Object with radial offset
    pos_obj = np.array([7001e3, 0.0, 0.0])  # 1 km radial offset
    
    ric_delta = VectorConverter.xyz_to_ric(
      xyz_ref_pos_vec = pos_ref,
      xyz_ref_vel_vec = vel_ref,
      xyz_obj_pos_vec = pos_obj,
    )
    
    # Should have ~1000 m radial error, near-zero in-track and cross-track
    assert np.isclose(ric_delta[0], 1000.0, rtol=1e-10)  # Radial
    assert np.isclose(ric_delta[1], 0.0, atol=1e-10)     # In-track
    assert np.isclose(ric_delta[2], 0.0, atol=1e-10)     # Cross-track
  
  def test_intrack_error_in_ric(self):
    """Test in-track offset converts correctly."""
    pos_ref = np.array([7000e3, 0.0, 0.0])
    vel_ref = np.array([0.0, 7500.0, 0.0])
    
    # Object with in-track offset (along velocity direction)
    pos_obj = np.array([7000e3, 1000.0, 0.0])  # 1 km along +Y
    
    ric_delta = VectorConverter.xyz_to_ric(
      xyz_ref_pos_vec = pos_ref,
      xyz_ref_vel_vec = vel_ref,
      xyz_obj_pos_vec = pos_obj,
    )
    
    # Should have ~1000 m in-track error
    assert np.isclose(ric_delta[0], 0.0, atol=1e-10)     # Radial
    assert np.isclose(ric_delta[1], 1000.0, rtol=1e-10)  # In-track
    assert np.isclose(ric_delta[2], 0.0, atol=1e-10)     # Cross-track
  
  def test_roundtrip_vector_conversion(self):
    """Test XYZ -> RIC -> XYZ vector conversion."""
    pos_ref = np.array([7000e3, 1000e3, 500e3])
    vel_ref = np.array([-500.0, 7000.0, 1000.0])
    
    pos_obj = np.array([7005e3, 1002e3, 498e3])
    vel_obj = np.array([-502.0, 7003.0, 999.0])
    
    # Convert to RIC
    ric_pos, ric_vel = VectorConverter.xyz_to_ric(
      xyz_ref_pos_vec = pos_ref,
      xyz_ref_vel_vec = vel_ref,
      xyz_obj_pos_vec = pos_obj,
      xyz_obj_vel_vec = vel_obj,
    )
    
    # Convert back to XYZ
    xyz_pos_back, xyz_vel_back = VectorConverter.ric_to_xyz(
      xyz_ref_pos_vec   = pos_ref,
      xyz_ref_vel_vec   = vel_ref,
      ric_delta_pos_vec = ric_pos,
      ric_delta_vel_vec = ric_vel,
    )
    
    assert np.allclose(pos_obj, xyz_pos_back, rtol=1e-10)
    assert np.allclose(vel_obj, xyz_vel_back, rtol=1e-10)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
