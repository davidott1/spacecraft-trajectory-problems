import numpy as np

from astropy             import units as u
from astropy.time        import Time as AstropyTime
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential

class FrameConverter:
  @staticmethod
  def teme_to_j2000(
    teme_pos_vec : np.ndarray,
    teme_vel_vec : np.ndarray,
    jd_utc       : float,
    units_pos    : str = 'm',
    units_vel    : str = 'm/s'
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert TEME (True Equator Mean Equinox) to J2000/GCRS using astropy.
    
    Input:
    ------
      teme_pos_vec : np.ndarray
        Position in TEME frame [m].
      teme_vel_vec : np.ndarray
        Velocity in TEME frame [m/s].
      jd_utc : float
        Julian date in UTC scale (from SGP4).
      units_pos : str
        Units for position input (default 'm').
      units_vel : str
        Units for velocity input (default 'm/s').
    
    Output:
    -------
      gcrs_pos_vec : np.ndarray
        Position in GCRS frame [m].
      gcrs_vel_vec : np.ndarray
        Velocity in GCRS frame [m/s].
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
    jd_utc        : float,
    units_pos     : str = 'm',
    units_vel     : str = 'm/s',
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert J2000/GCRS to TEME (True Equator Mean Equinox) using astropy.
    
    Input:
    ------
      j2000_pos_vec : np.ndarray
        Position in J2000/GCRS frame [m].
      j2000_vel_vec : np.ndarray
        Velocity in J2000/GCRS frame [m/s].
      jd_utc : float
        Julian date in UTC scale.
      units_pos : str
        Units for position input (default 'm').
      units_vel : str
        Units for velocity input (default 'm/s').
    
    Output:
    -------
      teme_pos_vec : np.ndarray
        Position in TEME frame [m].
      teme_vel_vec : np.ndarray
        Velocity in TEME frame [m/s].
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
  ) -> np.ndarray:
    """
    Calculate the rotation matrix from Inertial (XYZ) to Radial-Intrack-Cross-track (RIC) frame.
    
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
    """
    # r_hat unit vector
    xyz_ref_pos_hat = xyz_ref_pos_vec / np.linalg.norm(xyz_ref_pos_vec)
    r_hat = xyz_ref_pos_hat

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
    Calculate the rotation matrix from Radial-Intrack-Cross-track (RIC) to Inertial (XYZ) frame.
    
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


