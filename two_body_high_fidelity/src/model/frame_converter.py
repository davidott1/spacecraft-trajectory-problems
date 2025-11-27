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
    
    Output:
    -------
      tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - gcrs_pos_vec: position in GCRS frame [m].
        - gcrs_vel_vec: velocity in GCRS frame [m/s].
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
    units_vel     : str = 'm/s'
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
    
    Output:
    -------
      tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - teme_pos_vec: position in TEME frame [m].
        - teme_vel_vec: velocity in TEME frame [m/s].
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
