import numpy as np
import spiceypy as spice
from typing import Optional, Union

from astropy             import units as u
from astropy.time        import Time as AstropyTime
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential

class FrameConverter:
  @staticmethod
  def j2000_to_iau_earth(
    time_et : float,
  ) -> np.ndarray:
    """
    Get rotation matrix from J2000 to IAU_EARTH (body-fixed) frame using SPICE.
    
    Input:
    ------
      time_et : float
        Ephemeris Time (ET) in seconds past J2000 epoch.
    
    Output:
    -------
      rot_mat : np.ndarray
        3x3 rotation matrix such that: iau_earth_vec = rot_mat @ j2000_vec
    
    Notes:
    ------
      Requires SPICE kernels (PCK) to be loaded.
    """
    return spice.pxform('J2000', 'IAU_EARTH', time_et)

  @staticmethod
  def iau_earth_to_j2000(
    time_et : float,
  ) -> np.ndarray:
    """
    Get rotation matrix from IAU_EARTH (body-fixed) to J2000 frame using SPICE.
    
    Input:
    ------
      time_et : float
        Ephemeris Time (ET) in seconds past J2000 epoch.
    
    Output:
    -------
      rot_mat : np.ndarray
        3x3 rotation matrix such that: j2000_vec = rot_mat @ iau_earth_vec
    
    Notes:
    ------
      Requires SPICE kernels (PCK) to be loaded.
    """
    return spice.pxform('IAU_EARTH', 'J2000', time_et)

  @staticmethod
  def xyz_to_ric(
    xyz_ref_pos_vec : np.ndarray,
    xyz_ref_vel_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Calculate the rotation matrix from Inertial (XYZ) to Radial-Intrack-Crosstrack (RIC) frame.
    
    Input:
    ------
      xyz_ref_pos_vec : np.ndarray
        Reference position vector in inertial frame.
      xyz_ref_vel_vec : np.ndarray
        Reference velocity vector in inertial frame.
        
    Output:
    -------
      rot_mat_xyz_to_ric : np.ndarray
        3x3 Rotation matrix from Inertial to RIC frame.

    Usage:
    ------
      rot_mat_xyz_to_ric = FrameConverter.xyz_to_ric(
        xyz_ref_pos_vec = xyz_ref_pos_vec,
        xyz_ref_vel_vec = xyz_ref_vel_vec,
      )
    """
    # r_hat unit vector
    xyz_ref_pos_hat = xyz_ref_pos_vec / np.linalg.norm(xyz_ref_pos_vec)
    r_hat           = xyz_ref_pos_hat

    # c_hat unit vector
    ang_mom_vec = np.cross(xyz_ref_pos_vec, xyz_ref_vel_vec)
    ang_mom_hat = ang_mom_vec / np.linalg.norm(ang_mom_vec)
    c_hat       = ang_mom_hat
    
    # i_hat unit vector
    i_hat = np.cross(c_hat, r_hat)
    
    # Rotation matrix from inertial to RIC frame
    rot_mat_xyz_to_ric = np.vstack((r_hat, i_hat, c_hat))
    
    # Return rotation matrix
    return rot_mat_xyz_to_ric
  
  @staticmethod
  def ric_to_xyz(
    ric_ref_pos_vec : np.ndarray,
    ric_ref_vel_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Calculate the rotation matrix from Radial-Intrack-Crosstrack (RIC) to Inertial (XYZ) frame.
    
    Input:
    ------
      ric_ref_pos_vec : np.ndarray
        Reference position vector in inertial frame.
      ric_ref_vel_vec : np.ndarray
        Reference velocity vector in inertial frame.
        
    Output:
    -------
      rot_mat_ric_to_xyz : np.ndarray
        3x3 Rotation matrix from RIC to Inertial frame.

    Usage:
    ------
      rot_mat_ric_to_xyz = FrameConverter.ric_to_xyz(
        ric_ref_pos_vec = ric_ref_pos_vec,
        ric_ref_vel_vec = ric_ref_vel_vec,
      )
    """
    # Get rotation matrix from Inertial to RIC
    rot_mat_xyz_to_ric = FrameConverter.xyz_to_ric(
      xyz_ref_pos_vec = ric_ref_pos_vec,
      xyz_ref_vel_vec = ric_ref_vel_vec,
    )
    
    # Rotation matrix from RIC to Inertial is the transpose
    rot_mat_ric_to_xyz = rot_mat_xyz_to_ric.T
    
    # Return rotation matrix
    return rot_mat_ric_to_xyz

  @staticmethod
  def xyz_to_rtn(
    xyz_ref_pos_vec : np.ndarray,
    xyz_ref_vel_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Alias for xyz_to_ric. RIC is also known as RTN (Radial-Transverse-Normal).
    
    Input:
    ------
      xyz_ref_pos_vec : np.ndarray
        Reference position vector in inertial frame.
      xyz_ref_vel_vec : np.ndarray
        Reference velocity vector in inertial frame.
        
    Output:
    -------
      rot_mat_xyz_to_rtn : np.ndarray
        3x3 Rotation matrix from Inertial to RTN frame.
    """
    return FrameConverter.xyz_to_ric(xyz_ref_pos_vec, xyz_ref_vel_vec)

  @staticmethod
  def rtn_to_xyz(
    rtn_ref_pos_vec : np.ndarray,
    rtn_ref_vel_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Alias for ric_to_xyz. RIC is also known as RTN (Radial-Transverse-Normal).
    
    Input:
    ------
      rtn_ref_pos_vec : np.ndarray
        Reference position vector in inertial frame.
      rtn_ref_vel_vec : np.ndarray
        Reference velocity vector in inertial frame.
        
    Output:
    -------
      rot_mat_rtn_to_xyz : np.ndarray
        3x3 Rotation matrix from RTN to Inertial frame.
    """
    return FrameConverter.ric_to_xyz(rtn_ref_pos_vec, rtn_ref_vel_vec)


class VectorConverter:
  @staticmethod
  def teme_to_j2000(
    teme_pos_vec : np.ndarray,
    teme_vel_vec : np.ndarray,
    jd_utc       : Union[float, np.ndarray],
    units_pos    : str = 'm',
    units_vel    : str = 'm/s'
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert TEME (True Equator Mean Equinox) to J2000/GCRS using astropy.
    Supports vectorized inputs (3xN arrays) if jd_utc is an array of length N or scalar.
    
    Input:
    ------
      teme_pos_vec : np.ndarray
        Position in TEME frame [m]. Shape (3,) or (3, N).
      teme_vel_vec : np.ndarray
        Velocity in TEME frame [m/s]. Shape (3,) or (3, N).
      jd_utc : float | np.ndarray
        Julian date in UTC scale (e.g. 2460310.5). Scalar or array of length N.
      units_pos : str
        Units for position input (default 'm').
      units_vel : str
        Units for velocity input (default 'm/s').
    
    Output:
    -------
      gcrs_pos_vec : np.ndarray
        Position in GCRS frame [m]. Shape (3,) or (3, N).
      gcrs_vel_vec : np.ndarray
        Velocity in GCRS frame [m/s]. Shape (3,) or (3, N).

    Usage:
    ------
      gcrs_pos_vec, gcrs_vel_vec = VectorConverter.teme_to_j2000(
        teme_pos_vec = teme_pos_vec,
        teme_vel_vec = teme_vel_vec,
        jd_utc       = jd_utc,
      )
    """
    # Create astropy Time object from UTC
    astropy_time = AstropyTime(jd_utc, format='jd', scale='utc')
    
    # Determine units
    if units_pos.lower() == 'km':
      u_du = u.km # type: ignore
    else:
      u_du = u.m # type: ignore
    if units_vel.lower() == 'km/s':
      u_vu = u.km / u.s # type: ignore
    else:
      u_vu = u.m / u.s # type: ignore

    # Create CartesianRepresentation using position and velocity in TEME frame
    cart_rep = CartesianRepresentation(
      x = teme_pos_vec[0] * u_du,
      y = teme_pos_vec[1] * u_du,
      z = teme_pos_vec[2] * u_du,
      differentials = CartesianDifferential(
        d_x = teme_vel_vec[0] * u_vu,
        d_y = teme_vel_vec[1] * u_vu,
        d_z = teme_vel_vec[2] * u_vu,
      )
    )
    
    # Create a coordinate object in the TEME frame
    teme_cart_rep = TEME(cart_rep, obstime=astropy_time)
    
    # Transform the coordinates from the TEME frame to the GCRS (J2000) frame
    gcrs_cart_rep = teme_cart_rep.transform_to(GCRS(obstime=astropy_time))
      
    # Extract the numerical position and velocity vectors from the GCRS frame object
    gcrs_pos_vec = np.array([
      gcrs_cart_rep.cartesian.x.to(u_du).value,
      gcrs_cart_rep.cartesian.y.to(u_du).value,
      gcrs_cart_rep.cartesian.z.to(u_du).value,
    ])
    gcrs_vel_vec = np.array([
      gcrs_cart_rep.velocity.d_x.to(u_vu).value,
      gcrs_cart_rep.velocity.d_y.to(u_vu).value,
      gcrs_cart_rep.velocity.d_z.to(u_vu).value,
    ])
    
    return gcrs_pos_vec, gcrs_vel_vec

  @staticmethod
  def j2000_to_teme(
    j2000_pos_vec : np.ndarray,
    j2000_vel_vec : np.ndarray,
    jd_utc        : Union[float, np.ndarray],
    units_pos     : str = 'm',
    units_vel     : str = 'm/s',
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert J2000/GCRS to TEME (True Equator Mean Equinox) using astropy.
    Supports vectorized inputs (3xN arrays) if jd_utc is an array of length N or scalar.
    
    Input:
    ------
      j2000_pos_vec : np.ndarray
        Position in J2000/GCRS frame [m]. Shape (3,) or (3, N).
      j2000_vel_vec : np.ndarray
        Velocity in J2000/GCRS frame [m/s]. Shape (3,) or (3, N).
      jd_utc : float | np.ndarray
        Julian date in UTC scale (e.g. 2460310.5). Scalar or array of length N.
      units_pos : str
        Units for position input (default 'm').
      units_vel : str
        Units for velocity input (default 'm/s').
    
    Output:
    -------
      teme_pos_vec : np.ndarray
        Position in TEME frame [m]. Shape (3,) or (3, N).
      teme_vel_vec : np.ndarray
        Velocity in TEME frame [m/s]. Shape (3,) or (3, N).

    Usage:
    ------
      teme_pos_vec, teme_vel_vec = VectorConverter.j2000_to_teme(
        j2000_pos_vec = j2000_pos_vec,
        j2000_vel_vec = j2000_vel_vec,
        jd_utc        = jd_utc,
      )
    """
    # Create astropy Time object from UTC
    astropy_time = AstropyTime(jd_utc, format='jd', scale='utc')
    
    # Determine units
    if units_pos.lower() == 'km':
      u_du = u.km # type: ignore
    else:
      u_du = u.m # type: ignore
    if units_vel.lower() == 'km/s':
      u_vu = u.km / u.s # type: ignore
    else:
      u_vu = u.m / u.s # type: ignore

    # Create CartesianRepresentation using position and velocity in GCRS frame
    cart_rep = CartesianRepresentation(
      x = j2000_pos_vec[0] * u_du,
      y = j2000_pos_vec[1] * u_du,
      z = j2000_pos_vec[2] * u_du,
      differentials = CartesianDifferential(
        d_x = j2000_vel_vec[0] * u_vu,
        d_y = j2000_vel_vec[1] * u_vu,
        d_z = j2000_vel_vec[2] * u_vu,
      )
    )
    
    # Create a coordinate object in the GCRS frame
    gcrs_cart_rep = GCRS(cart_rep, obstime=astropy_time)
    
    # Transform the coordinates from the GCRS frame to the TEME frame
    teme_cart_rep = gcrs_cart_rep.transform_to(TEME(obstime=astropy_time))
      
    # Extract the numerical position and velocity vectors from the TEME frame object
    teme_pos_vec = np.array([
      teme_cart_rep.cartesian.x.to(u_du).value,
      teme_cart_rep.cartesian.y.to(u_du).value,
      teme_cart_rep.cartesian.z.to(u_du).value,
    ])
    teme_vel_vec = np.array([
      teme_cart_rep.velocity.d_x.to(u_vu).value,
      teme_cart_rep.velocity.d_y.to(u_vu).value,
      teme_cart_rep.velocity.d_z.to(u_vu).value,
    ])
    
    return teme_pos_vec, teme_vel_vec

  @staticmethod
  def xyz_to_ric(
    xyz_ref_pos_vec : np.ndarray,
    xyz_ref_vel_vec : np.ndarray,
    xyz_obj_pos_vec : Optional[np.ndarray] = None,
    xyz_obj_vel_vec : Optional[np.ndarray] = None,
  ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Convert position and/or velocity vectors from Inertial (XYZ) to Radial-Intrack-Cross-track (RIC) frame.
    Calculates the delta between object and reference before rotating.
    
    Input:
    ------
      xyz_ref_pos_vec : np.ndarray
        Reference position vector for RIC frame definition.
      xyz_ref_vel_vec : np.ndarray
        Reference velocity vector for RIC frame definition.
      xyz_obj_pos_vec : np.ndarray, optional
        Object position vector in inertial frame.
      xyz_obj_vel_vec : np.ndarray, optional
        Object velocity vector in inertial frame.
        
    Output:
    -------
      result : np.ndarray | tuple[np.ndarray, np.ndarray]
        If only pos provided: ric_delta_pos_vec
        If only vel provided: ric_delta_vel_vec
        If both provided: (ric_delta_pos_vec, ric_delta_vel_vec)

    Usage:
    ------
      # Convert both position and velocity
      ric_delta_pos_vec, ric_delta_vel_vec = VectorConverter.xyz_to_ric(
        xyz_ref_pos_vec = xyz_ref_pos_vec, 
        xyz_ref_vel_vec = xyz_ref_vel_vec, 
        xyz_obj_pos_vec = xyz_obj_pos_vec, 
        xyz_obj_vel_vec = xyz_obj_vel_vec,
      )
      
      # Convert only position
      ric_delta_pos_vec = VectorConverter.xyz_to_ric(
        xyz_ref_pos_vec = xyz_ref_pos_vec, 
        xyz_ref_vel_vec = xyz_ref_vel_vec, 
        xyz_obj_pos_vec = xyz_obj_pos_vec,
      )
    """
    if xyz_obj_pos_vec is None and xyz_obj_vel_vec is None:
      raise ValueError("At least one of xyz_obj_pos_vec or xyz_obj_vel_vec must be provided.")

    # Get rotation matrix
    rot_mat_xyz_to_ric = FrameConverter.xyz_to_ric(xyz_ref_pos_vec, xyz_ref_vel_vec)
    
    # Get delta vectors and rotate
    ric_delta_pos_vec = None
    ric_delta_vel_vec = None

    if xyz_obj_pos_vec is not None:
      # Calculate delta position
      xyz_delta_pos_vec = xyz_obj_pos_vec - xyz_ref_pos_vec
      # Rotate position
      ric_delta_pos_vec = rot_mat_xyz_to_ric @ xyz_delta_pos_vec

    if xyz_obj_vel_vec is not None:
      # Calculate delta velocity
      xyz_delta_vel_vec = xyz_obj_vel_vec - xyz_ref_vel_vec
      # Rotate velocity
      ric_delta_vel_vec = rot_mat_xyz_to_ric @ xyz_delta_vel_vec

    if ric_delta_pos_vec is not None and ric_delta_vel_vec is not None:
      return ric_delta_pos_vec, ric_delta_vel_vec
    elif ric_delta_pos_vec is not None:
      return ric_delta_pos_vec
    else:
      return ric_delta_vel_vec # type: ignore

  @staticmethod
  def ric_to_xyz(
    xyz_ref_pos_vec   : np.ndarray,
    xyz_ref_vel_vec   : np.ndarray,
    ric_delta_pos_vec : Optional[np.ndarray] = None,
    ric_delta_vel_vec : Optional[np.ndarray] = None,
  ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Convert delta position and/or velocity vectors from Radial-Intrack-Cross-track (RIC) to Inertial (XYZ) frame.
    Calculates the absolute object state by adding the reference state.
    
    Input:
    ------
      xyz_ref_pos_vec : np.ndarray
        Reference position vector for RIC frame definition (Inertial).
      xyz_ref_vel_vec : np.ndarray
        Reference velocity vector for RIC frame definition (Inertial).
      ric_delta_pos_vec : np.ndarray, optional
        Delta position vector in RIC frame.
      ric_delta_vel_vec : np.ndarray, optional
        Delta velocity vector in RIC frame.
        
    Output:
    -------
      result : np.ndarray | tuple[np.ndarray, np.ndarray]
        If only pos provided: xyz_obj_pos_vec
        If only vel provided: xyz_obj_vel_vec
        If both provided: (xyz_obj_pos_vec, xyz_obj_vel_vec)

    Usage:
    ------
      # Convert both position and velocity
      xyz_pos_vec, xyz_vel_vec = VectorConverter.ric_to_xyz(
        xyz_ref_pos_vec   = xyz_ref_pos_vec, 
        xyz_ref_vel_vec   = xyz_ref_vel_vec, 
        ric_delta_pos_vec = ric_delta_pos_vec, 
        ric_delta_vel_vec = ric_delta_vel_vec,
      )

      # Convert position only
      xyz_pos_vec = VectorConverter.ric_to_xyz(
        xyz_ref_pos_vec   = xyz_ref_pos_vec, 
        ric_delta_pos_vec = ric_delta_pos_vec, 
        ric_delta_vel_vec = ric_delta_vel_vec,
      )
    """
    if ric_delta_pos_vec is None and ric_delta_vel_vec is None:
      raise ValueError("At least one of ric_delta_pos_vec or ric_delta_vel_vec must be provided.")

    # Get rotation matrix (RIC to XYZ)
    rot_mat_ric_to_xyz = FrameConverter.ric_to_xyz(xyz_ref_pos_vec, xyz_ref_vel_vec)
    
    # Get object vectors by rotating and adding reference
    xyz_obj_pos_vec = None
    xyz_obj_vel_vec = None

    if ric_delta_pos_vec is not None:
      # Rotate delta position to inertial
      xyz_delta_pos_vec = rot_mat_ric_to_xyz @ ric_delta_pos_vec
      # Add reference position
      xyz_obj_pos_vec = xyz_ref_pos_vec + xyz_delta_pos_vec

    if ric_delta_vel_vec is not None:
      # Rotate delta velocity to inertial
      xyz_delta_vel_vec = rot_mat_ric_to_xyz @ ric_delta_vel_vec
      # Add reference velocity
      xyz_obj_vel_vec = xyz_ref_vel_vec + xyz_delta_vel_vec

    if xyz_obj_pos_vec is not None and xyz_obj_vel_vec is not None:
      return xyz_obj_pos_vec, xyz_obj_vel_vec
    elif xyz_obj_pos_vec is not None:
      return xyz_obj_pos_vec
    else:
      return xyz_obj_vel_vec # type: ignore

