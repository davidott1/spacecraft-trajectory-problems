"""
Unit Tests for Frame Converter Module
=====================================

Tests for reference frame transformations (J2000, TEME, RIC/RTN).

Tests:
------
TestRICFrameConversion
  - test_sanity_check_ric_axes_orthonormal   : verify RIC rotation matrix is orthonormal (R @ R.T = I)
  - test_sanity_check_radial_direction       : verify R-axis points along position vector
  - test_sanity_check_crosstrack_direction   : verify C-axis points along angular momentum
  - test_roundtrip_xyz_ric_xyz               : verify XYZ -> RIC -> XYZ returns original vector
  - test_sanity_check_rtn_alias              : verify RTN is an alias for RIC

TestVectorConverter
  - test_sanity_check_radial_offset_in_ric   : verify radial offset converts correctly to RIC
  - test_sanity_check_intrack_offset_in_ric  : verify in-track offset converts correctly to RIC
  - test_roundtrip_xyz_ric_xyz_vectors       : verify XYZ -> RIC -> XYZ vector conversion roundtrip

Usage:
------
  python -m pytest src/validation/test_frame_converter.py -v
"""
import pytest
import numpy as np

from src.model.frame_converter import FrameConverter, VectorConverter


class TestRICFrameConversion:
  """
  Tests for RIC (Radial-Intrack-Crosstrack) frame conversions.
  """
  
  def test_sanity_check_ric_axes_orthonormal(self):
    """
    RIC rotation matrix should be orthonormal.
    """
    pos_vec = np.array([7000e3, 1000e3, 500e3])
    vel_vec = np.array([-500.0, 7000.0, 1000.0])
    
    R = FrameConverter.xyz_to_ric(pos_vec, vel_vec)
    
    # Check orthonormality: R @ R.T = I
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-14)
    
    # Check determinant = 1 (proper rotation)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-14)
  
  def test_sanity_check_radial_direction(self):
    """
    R-axis should point along position vector.
    """
    pos_vec = np.array([7000e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    rot_mat_xyz_to_ric = FrameConverter.xyz_to_ric(pos_vec, vel_vec)
    
    r_hat = rot_mat_xyz_to_ric[0, :]
    expected_r_hat = pos_vec / np.linalg.norm(pos_vec)
    
    assert np.allclose(r_hat, expected_r_hat, atol=1e-14)
  
  def test_sanity_check_crosstrack_direction(self):
    """
    C-axis should be along angular momentum.
    """
    pos_vec = np.array([7000e3, 0.0, 0.0])
    vel_vec = np.array([0.0, 7500.0, 0.0])
    
    rot_mat_xyz_to_ric = FrameConverter.xyz_to_ric(pos_vec, vel_vec)
    
    c_hat = rot_mat_xyz_to_ric[2, :]
    
    ang_mom_vec    = np.cross(pos_vec, vel_vec)
    expected_c_hat = ang_mom_vec / np.linalg.norm(ang_mom_vec)
    
    assert np.allclose(c_hat, expected_c_hat, atol=1e-14)
  
  def test_roundtrip_xyz_ric_xyz(self):
    """
    XYZ -> RIC -> XYZ should give original vector.
    """
    pos_ref = np.array([7000e3, 1000e3, 500e3])
    vel_ref = np.array([-500.0, 7000.0, 1000.0])
    
    rot_mat_xyz_to_ric = FrameConverter.xyz_to_ric(pos_ref, vel_ref)
    rot_mat_ric_to_xyz = FrameConverter.ric_to_xyz(pos_ref, vel_ref)
    
    # Test roundtrip
    xyz_test_pos_vec      = np.array([1000.0, 2000.0, 500.0])
    ric_test_pos_vec      = rot_mat_xyz_to_ric @ xyz_test_pos_vec
    xyz_test_back_pos_vec = rot_mat_ric_to_xyz @ ric_test_pos_vec
    
    assert np.allclose(xyz_test_pos_vec, xyz_test_back_pos_vec, atol=1e-10)
  
  def test_sanity_check_rtn_alias(self):
    """
    RTN should be an alias for RIC.
    """
    pos_vec = np.array([7000e3, 1000e3, 500e3])
    vel_vec = np.array([-500.0, 7000.0, 1000.0])
    
    rot_mat_xyz_to_ric = FrameConverter.xyz_to_ric(pos_vec, vel_vec)
    rot_mat_xyz_to_rtn = FrameConverter.xyz_to_rtn(pos_vec, vel_vec)
    
    assert np.allclose(rot_mat_xyz_to_ric, rot_mat_xyz_to_rtn, atol=1e-14)


class TestVectorConverter:
  """
  Tests for VectorConverter class.
  """
  
  def test_sanity_check_radial_offset_in_ric(self):
    """
    Radial offset should convert correctly to RIC frame.
    """
    # Reference state
    xyz_ref_pos_vec = np.array([7000e3, 0.0, 0.0])
    xyz_ref_vel_vec = np.array([0.0, 7500.0, 0.0])
    
    # Object with radial offset
    xyz_obj_pos_vec = np.array([7001e3, 0.0, 0.0])  # 1 km radial offset
    
    ric_delta = VectorConverter.xyz_to_ric(
      xyz_ref_pos_vec = xyz_ref_pos_vec,
      xyz_ref_vel_vec = xyz_ref_vel_vec,
      xyz_obj_pos_vec = xyz_obj_pos_vec,
    )
    
    # Should have ~1000 m radial error, near-zero in-track and cross-track
    assert isinstance(ric_delta, np.ndarray)             # now type checker knows it's an array
    assert np.isclose(ric_delta[0], 1000.0, rtol=1e-10)  # Radial
    assert np.isclose(ric_delta[1],    0.0, atol=1e-10)     # In-track
    assert np.isclose(ric_delta[2],    0.0, atol=1e-10)     # Cross-track
  
  def test_sanity_check_intrack_offset_in_ric(self):
    """
    In-track offset should convert correctly to RIC frame.
    """
    xyz_ref_pos_vec = np.array([7000e3, 0.0, 0.0])
    xyz_ref_vel_vec = np.array([0.0, 7500.0, 0.0])
    
    # Object with in-track offset (along velocity direction)
    xyz_obj_pos_vec = np.array([7000e3, 1000.0, 0.0])  # 1 km along +Y
    
    ric_delta = VectorConverter.xyz_to_ric(
      xyz_ref_pos_vec = xyz_ref_pos_vec,
      xyz_ref_vel_vec = xyz_ref_vel_vec,
      xyz_obj_pos_vec = xyz_obj_pos_vec,
    )
    
    # Should have ~1000 m in-track error
    assert isinstance(ric_delta, np.ndarray)             # now type checker knows it's an array
    assert np.isclose(ric_delta[0],    0.0, atol=1e-10)  # Radial
    assert np.isclose(ric_delta[1], 1000.0, rtol=1e-10)  # In-track
    assert np.isclose(ric_delta[2],    0.0, atol=1e-10)  # Cross-track
  
  def test_roundtrip_xyz_ric_xyz_vectors(self):
    """
    XYZ -> RIC -> XYZ vector conversion should return original vectors.
    """
    xyz_ref_pos_vec = np.array([7000e3, 1000e3, 500e3])
    xyz_ref_vel_vec = np.array([-500.0, 7000.0, 1000.0])
    
    xyz_obj_pos_vec = np.array([7005e3, 1002e3, 498e3])
    xyz_obj_vel_vec = np.array([-502.0, 7003.0, 999.0])
    
    # Convert to RIC
    ric_obj_pos_vec, ric_obj_vel_vec = VectorConverter.xyz_to_ric(
      xyz_ref_pos_vec = xyz_ref_pos_vec,
      xyz_ref_vel_vec = xyz_ref_vel_vec,
      xyz_obj_pos_vec = xyz_obj_pos_vec,
      xyz_obj_vel_vec = xyz_obj_vel_vec,
    )
    
    # Convert back to XYZ
    xyz_back_pos_vec, xyz_back_vel_vec = VectorConverter.ric_to_xyz(
      xyz_ref_pos_vec   = xyz_ref_pos_vec,
      xyz_ref_vel_vec   = xyz_ref_vel_vec,
      ric_delta_pos_vec = ric_obj_pos_vec,
      ric_delta_vel_vec = ric_obj_vel_vec,
    )
    
    assert np.allclose(xyz_obj_pos_vec, xyz_back_pos_vec, rtol=1e-10)
    assert np.allclose(xyz_obj_vel_vec, xyz_back_vel_vec, rtol=1e-10)

if __name__ == "__main__":
  pytest.main([__file__, "-v"])
