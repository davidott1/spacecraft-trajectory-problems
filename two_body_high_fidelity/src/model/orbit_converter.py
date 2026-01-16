import warnings
import numpy    as np
import spiceypy as spice

from typing import Union

from src.model.constants import SOLARSYSTEMCONSTANTS, CONVERTER
from src.schemas.state   import (
  ClassicalOrbitalElements,
  ModifiedEquinoctialElements,
  GeodeticCoordinates,
  GeocentricCoordinates,
)

class TwoBody_RootSolvers:
  """
  Root solvers for two-body orbital mechanics.
  """
  
  @staticmethod
  def kepler(
    ma       : float,
    ecc      : float,
    tol      : float = 1e-10,
    max_iter : int   = 50
  ) -> float:
    """
    Solve Kepler's equation ma = ea - ecc*sin(ea) for eccentric anomaly ea.
    
    Input:
    ------
      ma : float
        Mean anomaly [rad]
      ecc : float
        Eccentricity
      tol : float
        Convergence tolerance
      max_iter : int
        Maximum iterations
    
    Output:
    -------
      ea : float
        Eccentric anomaly [rad]
    """
    # Initial guess
    if ecc < 0.8:
      ea = ma
    else:
      ea = np.pi
    
    # Newton-Raphson iteration
    for i in range(max_iter+1):
      func       = ea - ecc * np.sin(ea) - ma
      func_prime = 1 - ecc * np.cos(ea)
      delta_ea   = -func / func_prime
      if abs(delta_ea) < tol:
        return ea
      if i == max_iter:
        print("Kepler's equation not converged")
        break
      ea = ea + delta_ea
    
    return ea  # return best estimate if not converged
  
  @staticmethod
  def lambert(
    pos_o_vec : np.ndarray,
    pos_f_vec : np.ndarray,
    delta_t   : float,
    gp        : float,
    prograde  : bool  = True,
    max_iter  : int   = 100,
    tol       : float = 1e-6
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve Lambert's problem for transfer orbit between two position vectors.
    
    Input:
    ------
      pos_o_vec : np.ndarray
        Initial position vector [m]
      pos_f_vec : np.ndarray
        Final position vector [m]
      delta_t : float
        Time of flight [s]
      gp : float
        Gravitational parameter [m³/s²]
      prograde : bool
        True for prograde, False for retrograde
      max_iter : int
        Maximum iterations
      tol : float
        Convergence tolerance
    
    Output:
    -------
      vel_o_vec : np.ndarray
        Initial velocity vector [m/s]
      vel_f_vec : np.ndarray
        Final velocity vector [m/s]
    """
    # Compute magnitudes and chord
    pos_o_mag = np.linalg.norm(pos_o_vec)
    pos_f_mag = np.linalg.norm(pos_f_vec)
    c_vec     = pos_f_vec - pos_o_vec
    c_mag     = np.linalg.norm(c_vec)
    
    # Semi-latus rectum
    s = 0.5 * (pos_o_mag + pos_f_mag + c_mag)
    
    # Minimum energy semi-major axis
    a_min = s / 2
    
    # Time of flight for minimum energy orbit
    beta_min = 2 * np.arcsin(np.sqrt((s - c_mag) / s))
    if not prograde:
      beta_min = -beta_min
    tof_min = np.sqrt(a_min**3 / gp) * (np.pi - beta_min + np.sin(beta_min))
    
    if delta_t < tof_min:
      raise ValueError("Time of flight is less than minimum energy transfer time.")
    
    # Initial guess for semi-major axis
    a_low  = a_min
    a_high = 1e10 * a_min
    a      = 0.5 * (a_low + a_high)
    
    for _ in range(max_iter):
      # Compute time of flight for current semi-major axis
      alpha = 2 * np.arcsin(np.sqrt(s / (2 * a)))
      beta  = 2 * np.arcsin(np.sqrt((s - c_mag) / (2 * a)))
      if not prograde:
        beta = -beta
      
      tof = np.sqrt(a**3 / gp) * (alpha - beta - (np.sin(alpha) - np.sin(beta)))
      
      if abs(tof - delta_t) < tol:
        break
      
      if tof < delta_t:
        a_low = a
      else:
        a_high = a
      
      a = 0.5 * (a_low + a_high)
    
    # Compute velocities at pos_o_vec and pos_f_vec
    f     = 1 - pos_f_mag / a * (1 - np.cos(alpha))
    g     = pos_o_mag * pos_f_mag * np.sin(alpha) / np.sqrt(gp * a * (1 - np.cos(alpha)))
    g_dot = 1 - pos_o_mag / a * (1 - np.cos(alpha))
    
    vel_o_vec = (pos_f_vec - f * pos_o_vec) / g
    vel_f_vec = (g_dot * pos_f_vec - pos_o_vec) / g
    
    return vel_o_vec, vel_f_vec


class OrbitConverter:
  """
  Conversion between position/velocity and classical orbital elements.
  
  Provides comprehensive conversion utilities for orbital mechanics, including:
  - Cartesian state (position/velocity) ↔ classical orbital elements
  - Cartesian state (position/velocity) ↔ modified equinoctial elements
  - Anomaly transformations (true, eccentric, mean, hyperbolic, parabolic)
  - Support for all orbit types (circular, elliptical, parabolic, hyperbolic, rectilinear)
  """

  @staticmethod
  def pv_to_coe(
    pos_vec : np.ndarray,
    vel_vec : np.ndarray,
    gp      : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
  ) -> ClassicalOrbitalElements:
    """
    Convert Cartesian position and velocity vectors to classical orbital elements.
    
    Input:
    ------
      pos_vec : np.ndarray
        Position vector [m].
      vel_vec : np.ndarray
        Velocity vector [m/s].
      gp : float
        Gravitational parameter [m³/s²].
        
    Output:
    -------
      coe : ClassicalOrbitalElements | dict
        Classical orbital elements:
        - sma  : semi-major axis [m] (or np.inf for parabolic orbits)
        - ecc  : eccentricity [-]
        - inc  : inclination [rad]
        - raan : right ascension of the ascending node [rad]
        - aop  : argument of periapsis [rad]
        - ma   : mean anomaly [rad] (None for rectilinear or parabolic)
        - ta   : true anomaly [rad] (None for rectilinear)
        - ea   : eccentric anomaly [rad] (None for hyperbolic/parabolic)
        - ha   : hyperbolic anomaly [rad] (None for elliptic/parabolic)
        - pa   : parabolic anomaly [rad] (None for elliptic/hyperbolic)

    Notes:
    ------
      - Handles circular, elliptical, parabolic, hyperbolic, and rectilinear orbits
      - For the circular case, the ascending node (AN) and argument of periapsis (AP) 
        are ill-defined. The unit vector ie is set equal to the normalized inertial 
        position vector (ir).
      - Anomaly fields not applicable to the orbit type are set to None.

    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    # Small number for numerical comparisons
    eps = 1e-12
    
    # Ensure vectors are numpy arrays
    pos_vec = np.asarray(pos_vec).flatten()
    vel_vec = np.asarray(vel_vec).flatten()
    
    # Orbit radius
    pos_mag = np.linalg.norm(pos_vec)
    pos_dir = pos_vec / pos_mag

    # Angular momentum vector
    ang_mom_vec = np.cross(pos_vec, vel_vec)
    ang_mom_mag = np.linalg.norm(ang_mom_vec)

    # Eccentricity vector
    ecc_vec = np.cross(vel_vec, ang_mom_vec) / gp - pos_vec / pos_mag
    ecc_mag = np.linalg.norm(ecc_vec)

    # Compute semi-major axis
    sma_inv = 2.0 / pos_mag - np.dot(vel_vec, vel_vec) / gp
    if abs(sma_inv) > eps:
      # Elliptic or hyperbolic case
      sma = 1.0 / sma_inv
    else:
      # Parabolic case
      sma     = np.inf
      ecc_mag = 1.0

    # Handle rectilinear motion case
    if ang_mom_mag < eps:
      # periapsis_dir and ang_mom_dir are arbitrary
      ecc_dir       = pos_dir.copy()
      dum           = np.array([0, 0, 1])
      dum2          = np.array([0, 1, 0])
      ang_mom_dir   = np.cross(ecc_dir, dum)
      periapsis_dir = np.cross(ecc_dir, dum2)

      if np.linalg.norm(ang_mom_dir) > np.linalg.norm(periapsis_dir):
        ang_mom_dir = ang_mom_dir / np.linalg.norm(ang_mom_dir)
      else:
        ang_mom_dir = periapsis_dir / np.linalg.norm(periapsis_dir)
      periapsis_dir = np.cross(ang_mom_dir, ecc_dir)
    else:
      # Compute perifocal frame unit direction vectors
      ang_mom_dir = ang_mom_vec / ang_mom_mag
      if abs(ecc_mag) > eps:
        # Non-circular case
        ecc_dir = ecc_vec / ecc_mag
      else:
        # Circular orbit case
        ecc_dir = pos_dir.copy()
      periapsis_dir = np.cross(ang_mom_dir, ecc_dir)
    
    # Compute the 3-1-3 orbit plane orientation angles
    raan = np.arctan2(ang_mom_dir[0], -ang_mom_dir[1])
    inc  = np.arccos(ang_mom_dir[2])
    aop  = np.arctan2(ecc_dir[2], periapsis_dir[2])
    
    # Compute anomalies
    ma = None
    ta = None
    ea = None
    ha = None
    pa = None
    if ang_mom_mag < eps:
      # Rectilinear motion case
      if sma_inv > 0:
        # Elliptic case
        ea = np.arccos(1 - pos_mag * sma_inv)
        if np.dot(pos_vec, vel_vec) > 0:
          ea = 2 * np.pi - ea
      else:
        # Hyperbolic case
        ha = np.arccosh(pos_mag * sma_inv + 1)
        if np.dot(pos_vec, vel_vec) < 0:
          ha = 2 * np.pi - ha
    else:
      # Compute true anomaly
      dum = np.cross(ecc_dir, pos_dir)
      ta  = np.arctan2(np.dot(dum, ang_mom_dir), np.dot(ecc_dir, pos_dir))

      # Compute eccentric anomaly and mean anomaly
      if ecc_mag < 1.0 - eps:
        # Elliptical case - CORRECTED FORMULA
        ea = 2 * np.arctan2(
          np.sqrt(1 - ecc_mag) * np.sin(ta / 2),
          np.sqrt(1 + ecc_mag) * np.cos(ta / 2)
        )
        ma = ea - ecc_mag * np.sin(ea)
        ma = ma % (2 * np.pi)
      elif ecc_mag > 1.0 + eps:
        # Hyperbolic case
        ha  = 2 * np.arctanh(np.tan(ta / 2) * np.sqrt((ecc_mag - 1) / (ecc_mag + 1)))
        ma = ecc_mag * np.sinh(ha) - ha
      else:
        # Parabolic case
        pa = np.tan(ta / 2)
        ma = pa + pa**3 / 3

    coe = ClassicalOrbitalElements(
      sma  = sma,
      ecc  = ecc_mag,
      inc  = inc,
      raan = raan,
      aop  = aop,
      ta   = ta,
      ea   = ea,
      ma   = ma,
      ha   = ha,
      pa   = pa,
    )
    
    return coe

  @staticmethod
  def pv_to_mee(
    pos_vec    : np.ndarray,
    vel_vec    : np.ndarray,
    gp         : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
    retrograde : bool  = False,
  ) -> ModifiedEquinoctialElements:
    """
    Convert Cartesian position and velocity vectors to Modified Equinoctial Elements (MEE).
    
    Input:
    ------
      pos_vec : np.ndarray
        Position vector [m].
      vel_vec : np.ndarray
        Velocity vector [m/s].
      gp : float
        Gravitational parameter [m³/s²].
      retrograde : bool
        If True, use retrograde factor (I = -1) for inclinations > 90°.
        If False (default), use prograde factor (I = +1).
        
    Output:
    -------
      mee : dict
        Dictionary containing modified equinoctial elements:
        - p : semi-latus rectum [m]
        - f : e*cos(ω + I*Ω) [-]
        - g : e*sin(ω + I*Ω) [-]
        - h : tan(i/2)*cos(Ω) [-] (prograde) or cot(i/2)*cos(Ω) [-] (retrograde)
        - k : tan(i/2)*sin(Ω) [-] (prograde) or cot(i/2)*sin(Ω) [-] (retrograde)
        - L : true longitude = ω + I*Ω + ν [rad]
        - I : retrograde factor (+1 prograde, -1 retrograde)

    Notes:
    ------
      Modified Equinoctial Elements avoid singularities at:
      - Zero eccentricity (circular orbits)
      - Zero inclination (equatorial orbits) when using prograde formulation
      - 180° inclination when using retrograde formulation
      
      The retrograde formulation (I = -1) should be used for i > 90°.
      
    Source:
    -------
      Walker, Ireland, and Owens (1985)
      "A Set of Modified Equinoctial Orbit Elements"
      Celestial Mechanics, Vol. 36, pp. 409-419
    """
    # Ensure vectors are numpy arrays
    pos_vec = np.asarray(pos_vec).flatten()
    vel_vec = np.asarray(vel_vec).flatten()
    
    # Retrograde factor
    I = -1 if retrograde else 1
    
    # Position and velocity magnitudes
    pos_mag = np.linalg.norm(pos_vec)
    vel_mag = np.linalg.norm(vel_vec)
    
    # Angular momentum vector
    ang_mom_vec = np.cross(pos_vec, vel_vec)
    ang_mom_mag = np.linalg.norm(ang_mom_vec)
    
    # Semi-latus rectum
    p = ang_mom_mag**2 / gp
    
    # Eccentricity vector
    ecc_vec = np.cross(vel_vec, ang_mom_vec) / gp - pos_vec / pos_mag
    ecc_mag = np.linalg.norm(ecc_vec)
    
    # Unit vectors
    pos_dir     = pos_vec / pos_mag
    ang_mom_dir = ang_mom_vec / ang_mom_mag if ang_mom_mag > 1e-12 else np.array([0, 0, 1])
    
    # Compute h and k directly from angular momentum
    #   h = tan(i/2)*cos(Ω), k = tan(i/2)*sin(Ω) for prograde
    #   For h_hat = [hx, hy, hz], we have:
    #     hz = cos(i), so tan(i/2) = sqrt((1-hz)/(1+hz)) for prograde
    #     hx = sin(i)*sin(Ω), hy = -sin(i)*cos(Ω)
    ang_mom_dir_z = ang_mom_dir[2]
    
    if retrograde:
      # cot(i/2) = sqrt((1+hz)/(1-hz))
      if abs(1 - ang_mom_dir_z) > 1e-12:
        cot_half_i = np.sqrt((1 + ang_mom_dir_z) / (1 - ang_mom_dir_z))
      else:
        cot_half_i = np.inf
      # For retrograde: h = cot(i/2)*cos(Ω), k = cot(i/2)*sin(Ω)
      # From h_hat: sin(Ω) = hx/sin(i), cos(Ω) = -hy/sin(i)
      sin_i = np.sqrt(ang_mom_dir[0]**2 + ang_mom_dir[1]**2)
      if sin_i > 1e-12:
        h = -cot_half_i * ang_mom_dir[1] / sin_i
        k =  cot_half_i * ang_mom_dir[0] / sin_i
      else:
        h = 0.0
        k = 0.0
    else:
      # tan(i/2) = sqrt((1-hz)/(1+hz))
      if abs(1 + ang_mom_dir_z) > 1e-12:
        tan_half_i = np.sqrt((1 - ang_mom_dir_z) / (1 + ang_mom_dir_z))
      else:
        tan_half_i = np.inf
      # For prograde: h = tan(i/2)*cos(Ω), k = tan(i/2)*sin(Ω)
      sin_i = np.sqrt(ang_mom_dir[0]**2 + ang_mom_dir[1]**2)
      if sin_i > 1e-12:
        h = -tan_half_i * ang_mom_dir[1] / sin_i
        k =  tan_half_i * ang_mom_dir[0] / sin_i
      else:
        h = 0.0
        k = 0.0
    
    # Compute f and g from eccentricity vector
    #   Express ecc_vec in terms of f and g
    #     ecc_vec = ecc_mag * (cos(ω)*P_hat + sin(ω)*Q_hat)
    #   where P_hat, Q_hat are perifocal unit vectors
    #     f = ecc_mag * cos(ω + I*Ω)
    #     g = ecc_mag * sin(ω + I*Ω)
    
    # Define equinoctial frame unit vectors
    s_sq  = 1 + h**2 + k**2
    f_dir = np.array([1 - k**2 + h**2, 2*k*h, -2*I*k]) / s_sq
    g_dir = np.array([2*I*k*h, (1 + k**2 - h**2)*I, 2*h]) / s_sq
    
    # Project eccentricity vector onto equinoctial frame
    f = np.dot(ecc_vec, f_dir)
    g = np.dot(ecc_vec, g_dir)
    
    # Compute true longitude L
    #   L      = ω + I*Ω + ν
    #   cos(L) = (r_hat · f_hat)
    #   sin(L) = (r_hat · g_hat)
    cos_L = np.dot(pos_dir, f_dir)
    sin_L = np.dot(pos_dir, g_dir)
    L = np.arctan2(sin_L, cos_L)
    if L < 0:
      L += 2 * np.pi
    
    return ModifiedEquinoctialElements(
      p = p,
      f = f,
      g = g,
      h = h,
      k = k,
      L = L,
      I = I,
    )

  @staticmethod
  def coe_to_pv(
    coe : ClassicalOrbitalElements,
    gp  : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Classical Orbital Elements to position and velocity vectors.
    
    Input:
    ------
      coe : dict
        Dictionary containing classical orbital elements:
        - sma  : semi-major axis [m]
        - ecc  : eccentricity [-]
        - inc  : inclination [rad]
        - raan : right ascension of ascending node [rad]
        - aop  : argument of periapsis [rad]
        - ta   : true anomaly [rad]
      gp : float
        Gravitational parameter [m³/s²].
    
    Output:
    -------
      pos_vec : np.ndarray
        Position vector [m]
      vel_vec : np.ndarray
        Velocity vector [m/s]

    Notes:
    ------
      - For parabolic orbits, the semi-major axis (sma) should be set to np.inf.
      - The eccentricity (ecc) should be 1 for parabolic orbits.
      - The true anomaly (ta) is used to compute the position and velocity.
    """
    # Extract elements
    sma  = coe.sma
    ecc  = coe.ecc
    inc  = coe.inc
    raan = coe.raan
    aop  = coe.aop
    ta   = coe.ta
    
    # Gravitational parameter
    if hasattr(coe, 'gp'):
      gp = coe.gp
    
    # Compute semi-latus rectum
    if np.isinf(sma):
      # Parabolic case
      p = 0  # Semi-latus rectum is not used for parabolic orbits
    else:
      p = sma * (1 - ecc**2)
    
    # Position in orbital plane
    pqw_pos_vec = p / (1 + ecc * np.cos(ta)) * np.array([np.cos(ta), np.sin(ta), 0])
    
    # Velocity in orbital plane
    if np.isinf(sma):
      # Parabolic case
      pqw_vel_vec = np.sqrt(gp / (2 * np.linalg.norm(pqw_pos_vec))) * np.array([ -np.sin(ta), ecc - np.cos(ta), 0 ])
    else:
      # Elliptic case
      pqw_vel_vec = np.sqrt(gp / p) * np.array([-np.sin(ta), ecc + np.cos(ta), 0])
    
    # Precompute trigonometric functions
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_inc  = np.cos(inc)
    sin_inc  = np.sin(inc)
    cos_aop  = np.cos(aop)
    sin_aop  = np.sin(aop)

    # Combined rotation matrix (perifocal to inertial)
    #   rot_mat_pqw_to_xyz = rot_z(-raan) @ rot_x(-inc) @ rot_z(-aop)
    rot_mat_pqw_to_xyz = np.array([
      [ cos_raan * cos_aop - sin_raan * sin_aop * cos_inc, -cos_raan * sin_aop - sin_raan * cos_aop * cos_inc,  sin_raan * sin_inc ],
      [ sin_raan * cos_aop + cos_raan * sin_aop * cos_inc, -sin_raan * sin_aop + cos_raan * cos_aop * cos_inc, -cos_raan * sin_inc ],
      [                                 sin_aop * sin_inc,                                  cos_aop * sin_inc,             cos_inc ]
    ])
    
    # Transform to inertial frame
    xyz_pos_vec = rot_mat_pqw_to_xyz @ pqw_pos_vec
    xyz_vel_vec = rot_mat_pqw_to_xyz @ pqw_vel_vec
    
    return xyz_pos_vec, xyz_vel_vec

  @staticmethod
  def mee_to_pv(
    mee : ModifiedEquinoctialElements,
    gp  : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Modified Equinoctial Elements (MEE) to position and velocity vectors in the inertial frame.
    
    Input:
    ------
      mee : ModifiedEquinoctialElements
        Modified equinoctial elements dataclass:
        - p : semi-latus rectum [m]
        - f : e*cos(ω + I*Ω) [-]
        - g : e*sin(ω + I*Ω) [-]
        - h : tan(i/2)*cos(Ω) [-] or cot(i/2)*cos(Ω) [-] (retrograde)
        - k : tan(i/2)*sin(Ω) [-] or cot(i/2)*sin(Ω) [-] (retrograde)
        - L : true longitude = ω + I*Ω + ν [rad]
        - I : retrograde factor (+1 prograde, -1 retrograde)
      gp : float
        Gravitational parameter [m³/s²]
    
    Output:
    -------
      pos_vec : np.ndarray
        Position vector [m]
      vel_vec : np.ndarray
        Velocity vector [m/s]

    Notes:
    ------
      The conversion uses the direct transformation from MEE to Cartesian
      coordinates without going through classical elements.
      
    Source:
    -------
      Walker, Ireland, and Owens (1985)
      "A Set of Modified Equinoctial Orbit Elements"
      Celestial Mechanics, Vol. 36, pp. 409-419
    """
    # Extract elements from dataclass
    p = mee.p
    f = mee.f
    g = mee.g
    h = mee.h
    k = mee.k
    L = mee.L
    I = mee.I
    
    # Auxiliary quantities
    s_sq = 1 + h**2 + k**2
    w    = 1 + f * np.cos(L) + g * np.sin(L)
    r    = p / w
    
    # Trigonometric quantities
    cos_L = np.cos(L)
    sin_L = np.sin(L)
    
    # Transformation matrix components (unit vectors in equinoctial frame)
    f_hat = np.array([ 1 - k**2 + h**2,             2 * k * h, -2 * I * k ]) / s_sq
    g_hat = np.array([   2 * I * k * h, (1 + k**2 - h**2) * I,      2 * h ]) / s_sq
    
    # Position vector
    pos_vec = r * cos_L * f_hat + r * sin_L * g_hat
    
    # Velocity vector
    sqrt_gp_over_p = np.sqrt(gp / p)
    vel_vec        = -sqrt_gp_over_p * (sin_L + g) * f_hat + sqrt_gp_over_p * (cos_L + f) * g_hat
    
    return pos_vec, vel_vec

  @staticmethod
  def coe_to_mee(
    coe        : ClassicalOrbitalElements,
    retrograde : bool = False,
  ) -> ModifiedEquinoctialElements:
    """
    Convert Classical Orbital Elements to Modified Equinoctial Elements.
    
    Input:
    ------
      coe : ClassicalOrbitalElements
        Classical orbital elements dataclass:
        - sma  : semi-major axis [m]
        - ecc  : eccentricity [-]
        - inc  : inclination [rad]
        - raan : right ascension of ascending node [rad]
        - aop  : argument of periapsis [rad]
        - ta   : true anomaly [rad]
      retrograde : bool
        If True, use retrograde factor (I = -1).
        
    Output:
    -------
      mee : ModifiedEquinoctialElements
        Modified equinoctial elements dataclass.
    """
    # Extract elements from dataclass
    sma  = coe.sma
    ecc  = coe.ecc
    inc  = coe.inc
    raan = coe.raan
    aop  = coe.aop
    ta   = coe.ta
    
    # Retrograde factor
    I = -1 if retrograde else 1
    
    # Semi-latus rectum
    if np.isinf(sma):
      # Parabolic case - compute from eccentricity
      p = 0  # Cannot determine without additional info
    else:
      p = sma * (1 - ecc**2)
    
    # Longitude of periapsis
    varpi = aop + I * raan
    
    # f and g
    f = ecc * np.cos(varpi)
    g = ecc * np.sin(varpi)
    
    # h and k
    if retrograde:
      half_inc = inc / 2
      if np.abs(np.tan(half_inc)) > 1e-12:
        cot_half_inc = 1.0 / np.tan(half_inc)
      else:
        cot_half_inc = np.inf
      h = cot_half_inc * np.cos(raan)
      k = cot_half_inc * np.sin(raan)
    else:
      tan_half_inc = np.tan(inc / 2)
      h = tan_half_inc * np.cos(raan)
      k = tan_half_inc * np.sin(raan)
    
    # True longitude
    L = (varpi + ta) % (2 * np.pi)
    
    return ModifiedEquinoctialElements(
      p = p,
      f = f,
      g = g,
      h = h,
      k = k,
      L = L,
      I = I,
    )

  @staticmethod
  def mee_to_coe(
    mee : ModifiedEquinoctialElements,
  ) -> ClassicalOrbitalElements:
    """
    Convert Modified Equinoctial Elements to Classical Orbital Elements.
    
    Input:
    ------
      mee : ModifiedEquinoctialElements
        Modified equinoctial elements dataclass:
        - p : semi-latus rectum [m]
        - f : e*cos(ω + I*Ω) [-]
        - g : e*sin(ω + I*Ω) [-]
        - h : tan(i/2)*cos(Ω) [-] or cot(i/2)*cos(Ω) [-]
        - k : tan(i/2)*sin(Ω) [-] or cot(i/2)*sin(Ω) [-]
        - L : true longitude [rad]
        - I : retrograde factor (+1 or -1)
        
    Output:
    -------
      coe : ClassicalOrbitalElements
        Classical orbital elements dataclass.
    """
    # Extract elements from dataclass
    p = mee.p
    f = mee.f
    g = mee.g
    h = mee.h
    k = mee.k
    L = mee.L
    I = mee.I
    
    # Eccentricity
    ecc = np.sqrt(f**2 + g**2)
    
    # Semi-major axis
    if ecc < 1.0:
      sma = p / (1 - ecc**2)
    elif ecc > 1.0:
      sma = p / (ecc**2 - 1)  # Negative for hyperbolic
    else:
      sma = np.inf  # Parabolic
    
    # Inclination
    # From h² + k² = tan²(i/2) for prograde, cot²(i/2) for retrograde
    hk_sq = h**2 + k**2
    if I == 1:  # Prograde
      inc = 2 * np.arctan(np.sqrt(hk_sq))
    else:  # Retrograde
      if hk_sq > 1e-12:
        inc = 2 * np.arctan(1.0 / np.sqrt(hk_sq))
      else:
        inc = np.pi
    
    # RAAN
    raan = np.arctan2(k, h)
    if raan < 0:
      raan += 2 * np.pi
    
    # Longitude of periapsis
    varpi = np.arctan2(g, f)
    
    # Argument of periapsis
    aop = varpi - I * raan
    aop = aop % (2 * np.pi)
    
    # True anomaly
    ta = L - varpi
    ta = ta % (2 * np.pi)
    
    return ClassicalOrbitalElements(
      sma  = sma,
      ecc  = ecc,
      inc  = inc,
      raan = raan,
      aop  = aop,
      ta   = ta,
      ma   = None,  # Not computed here
      ea   = None,
      ha   = None,
      pa   = None,
    )

  @staticmethod
  def pv_to_specific_energy(
    pos_vec : np.ndarray,
    vel_vec : np.ndarray,
    gp      : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
  ) -> float:
    """
    Calculate specific mechanical energy from Cartesian state vectors.
    
    Input:
    ------
      pos_vec : np.ndarray
        Position vector [m].
      vel_vec : np.ndarray
        Velocity vector [m/s].
      gp : float
        Gravitational parameter [m³/s²].
        
    Output:
    -------
      epsilon : float
        Specific mechanical energy [m²/s²].
    """
    # Ensure vectors are numpy arrays and flattened
    pos_vec = np.asarray(pos_vec).flatten()
    vel_vec = np.asarray(vel_vec).flatten()

    pos_mag = np.linalg.norm(pos_vec)
    vel_mag = np.linalg.norm(vel_vec)
    
    specific_energy = vel_mag**2 / 2.0 - gp / pos_mag
    return float(specific_energy)

  @staticmethod
  def specific_energy_to_period(
    specific_energy : float,
    gp              : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
  ) -> float:
    """
    Calculate orbital period from specific mechanical energy.
    
    Input:
    ------
      specific_energy : float
        Specific mechanical energy [m²/s²].
      gp : float
        Gravitational parameter [m³/s²].
        
    Output:
    -------
      period : float
        Orbital period [s]. Returns np.inf for parabolic/hyperbolic orbits.
    """
    if specific_energy < 0:
      # Elliptical orbit
      sma    = -gp / (2.0 * specific_energy)
      period = 2.0 * np.pi * np.sqrt(sma**3 / gp)
      return float(period)
    else:
      # Parabolic or Hyperbolic
      return np.inf

  @staticmethod
  def pv_to_period(
    pos_vec : np.ndarray,
    vel_vec : np.ndarray,
    gp      : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
  ) -> float:
    """
    Calculate orbital period from Cartesian state vectors.
    
    Input:
    ------
      pos_vec : np.ndarray
        Position vector [m].
      vel_vec : np.ndarray
        Velocity vector [m/s].
      gp : float
        Gravitational parameter [m³/s²].
        
    Output:
    -------
      period : float
        Orbital period [s]. Returns np.inf for parabolic/hyperbolic orbits.
    """
    # Position and velocity vectors to specific energy
    specific_energy = OrbitConverter.pv_to_specific_energy(pos_vec, vel_vec, gp)
    
    
    # Specific energy to period
    return OrbitConverter.specific_energy_to_period(specific_energy, gp)

  @staticmethod
  def ea_to_ta(
    ea  : float,
    ecc : float,
  ) -> float:
    """
    Maps eccentric anomaly to true anomaly.
    For circular or non-rectilinear elliptic orbits.
    
    Input:
    ------
      ea : float
        Eccentric anomaly [rad]
      ecc : float
        Eccentricity (0 <= ecc < 1)
    
    Output:
    -------
      ta : float
        True anomaly [rad]

    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    if 0 <= ecc < 1:
      ta = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(ea / 2),
        np.sqrt(1 - ecc) * np.cos(ea / 2)
      )
    else:
        raise ValueError(f"E2f() requires 0 <= ecc < 1, received ecc = {ecc}")
    
    return ta
  
  @staticmethod
  def ea_to_ma(
    ea  : float, 
    ecc : float,
  ) -> float:
    """
    Maps eccentric anomaly to mean anomaly.
    For both 2D and 1D elliptic orbits.
    
    Input:
    ------
      ea : float
        Eccentric anomaly [rad]
      ecc : float
        Eccentricity (0 <= ecc < 1)
    
    Output:
    -------
      ma : float
        Mean anomaly [rad]

    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    if 0 <= ecc < 1:
      ma = ea - ecc * np.sin(ea)
    else:
      raise ValueError(f"ea_to_ma() requires 0 <= ecc < 1, received ecc = {ecc}")

    return ma
  
  @staticmethod
  def ta_to_ea(
    ta  : float,
    ecc : float,
  ) -> float:
    """
    Maps true anomaly to eccentric anomaly.
    For circular or non-rectilinear elliptic orbits.
    
    Input:
    ------
      ta : float
        True anomaly [rad]
      ecc : float
        Eccentricity (0 <= ecc < 1)
    
    Output:
    -------
      ea : float
        Eccentric anomaly [rad]
    
    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    if 0 <= ecc < 1:
      ea = 2 * np.arctan2(
        np.sqrt(1 - ecc) * np.sin(ta / 2),
        np.sqrt(1 + ecc) * np.cos(ta / 2)
      )
    else:
      raise ValueError(f"ta_to_ea() requires 0 <= ecc < 1, received ecc = {ecc}")
    
    return ea
  
  @staticmethod
  def ta_to_ha(
      ta  : float,
      ecc : float,
    ) -> float:
    """
    Maps true anomaly to hyperbolic anomaly for hyperbolic orbits.
    
    Input:
    ------
      ta : float
        True anomaly [rad]
      ecc : float
        Eccentricity (ecc > 1)

    Output:
    -------
      ha : float
        Hyperbolic anomaly [rad]

    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    if ecc > 1:
      ha = 2 * np.arctanh(
        np.sqrt((ecc - 1) / (ecc + 1)) * np.tan(ta / 2)
      )
    else:
      raise ValueError(f"ta_to_ha() requires ecc > 1, received ecc = {ecc}")

    return ha
  
  @staticmethod
  def ha_to_ta(
    ha  : float,
    ecc : float,
  ) -> float:
    """
    Maps hyperbolic anomaly to true anomaly for hyperbolic orbits.
    
    Input:
    ------
      ha : float
        Hyperbolic anomaly [rad]
      ecc : float
        Eccentricity (ecc > 1)
    
    Output:
    -------
      ta : float
        True anomaly [rad]

    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    if ecc > 1:
      ta = 2 * np.arctan(
          np.sqrt((ecc + 1) / (ecc - 1)) * np.tanh(ha / 2)
      )
    else:
      raise ValueError(f"ha_to_ta() requires ecc > 1, received ecc = {ecc}")
    
    return ta
  
  @staticmethod
  def ha_to_mha(
    ha  : float,
    ecc : float,
  ) -> float:
    """
    Maps hyperbolic anomaly to mean hyperbolic anomaly for hyperbolic orbits.
    
    Input:
    ------
      ha : float
        Hyperbolic anomaly [rad]
      ecc : float
        Eccentricity (ecc > 1)
    
    Output:
    -------
      mha : float
        Mean hyperbolic anomaly [rad]

    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    if ecc > 1:
      mha = ecc * np.sinh(ha) - ha
    else:
      raise ValueError(f"ha_to_mha() requires ecc > 1, received ecc = {ecc}")

    return mha

  @staticmethod
  def ma_to_ea(
    ma       : float,
    ecc      : float,
    tol      : float = 1e-13,
    max_iter : int   = 200,
  ) -> float:
    """
    Maps mean anomaly to eccentric anomaly using Newton-Raphson iteration for both 2D and 1D elliptic orbits.
    
    Alias for TwoBody_RootSolvers.kepler().
    
    Input:
    ------
      ma : float
        Mean anomaly [rad]
      ecc : float
        Eccentricity (0 <= ecc < 1)
      tol : float
        Convergence tolerance
      max_iter : int
        Maximum iterations
    
    Output:
    -------
      ea : float
        Eccentric anomaly [rad]
    """
    return TwoBody_RootSolvers.kepler(ma, ecc, tol, max_iter)
  
  @staticmethod
  def mha_to_ha(
    mha : float,
    ecc : float,
    tol : float = 1e-13,
    max_iter: int = 200,
  ) -> float:
    """
    Maps mean hyperbolic anomaly to hyperbolic anomaly using Newton-Raphson iteration.
    For hyperbolic orbits.
    
    Input:
    ------
      mha : float
        Mean hyperbolic anomaly [rad]
      ecc : float
        Eccentricity (ecc > 1)
      tol : float
        Convergence tolerance
      max_iter : int
        Maximum iterations
    
    Output:
    -------
      ha : float
        Hyperbolic anomaly [rad]

    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    if ecc > 1:
      # Initial guess
      ha = mha  

      # Iteration loop
      for i in range(max_iter+1):
        # ha step
        dha = (ecc * np.sinh(ha) - ha - mha) / (ecc * np.cosh(ha) - 1)

        # Check convergence
        if abs(dha) < tol:
          break

        # Check max iterations
        if i == max_iter:
          warnings.warn(f"mha_to_ha iteration did not converge for mha={mha}, ecc={ecc}")

        # Update ha
        ha += -dha
    else:
      raise ValueError(f"mha_to_ha() requires ecc > 1, received ecc = {ecc}")

    return ha


class GeographicCoordinateConverter:
  """
  Conversion between Cartesian position vectors and geographic coordinates.
  
  Supports both geocentric (spherical) and geodetic (ellipsoidal) representations.
  All methods assume the body-fixed reference frame.
  
  Coordinate Systems:
  -------------------
  - Geocentric: Latitude is angle from equatorial plane along position vector.
                Simple spherical geometry, ignores Earth's oblateness.
                
  - Geodetic:   Latitude is angle from equatorial plane along ellipsoid surface normal.
                Uses WGS84 ellipsoid parameters. This is what GPS and maps use.
  
  WGS84 Parameters:
  -----------------
  - Equatorial radius: 6378.137 km
  - Flattening: 1/298.257223563
  """
  
  # WGS84 ellipsoid parameters
  WGS84_RE = 6378.137          # Equatorial radius [km]
  WGS84_F  = 1/298.257223563   # Flattening
  
  # -------
  # Generic
  # -------

  @staticmethod
  def spherical_to_cartesian(
    lat_rad : Union[np.ndarray, float],
    lon_rad : Union[np.ndarray, float],
    radius  : Union[np.ndarray, float] = 1.0,
  ) -> np.ndarray:
    """
    Convert spherical coordinates (latitude, longitude, radius) to Cartesian.
    
    This is a low-level helper for both geocentric conversions and plotting utilities.
    
    Input:
    ------
      lat_rad : Union[np.ndarray, float]
        Latitude in radians (geocentric or geodetic)
      lon_rad : Union[np.ndarray, float]
        Longitude in radians
      radius : Union[np.ndarray, float]
        Radial distance from origin (e.g., Earth center) [m]. Default is 1.0 (unit sphere).
        
    Output:
    -------
      pos_vec : np.ndarray
        Cartesian position vector [m], shape (3,) or (3, N)
    """
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.array([x, y, z])
  
  # ----------------------------
  # Geocentric (Spherical)
  # ----------------------------
  
  @staticmethod
  def pos_to_geocentric(
    pos_vec : np.ndarray,
  ) -> GeocentricCoordinates:
    """
    Convert Cartesian position to geocentric (spherical) coordinates.

    Input:
    ------
      pos_vec : np.ndarray
        Position vector in body-fixed frame [m].

    Output:
    -------
      coords : GeocentricCoordinates
        Geocentric coordinates dataclass containing:
        - latitude  : float - Geocentric latitude [rad]
        - longitude : float - Longitude [rad]
        - altitude  : float - Altitude above spherical Earth [m]
    """
    # Delegate to array version with single position
    pos_vec = np.asarray(pos_vec).flatten().reshape(3, 1)
    coords_array = GeographicCoordinateConverter.pos_to_geocentric_array(pos_vec)

    return GeocentricCoordinates(
      latitude  = float(coords_array.latitude[0]),
      longitude = float(coords_array.longitude[0]),
      altitude  = float(coords_array.altitude[0]),
    )
  
  @staticmethod
  def geocentric_to_pos(
    coords : GeocentricCoordinates,
  ) -> np.ndarray:
    """
    Convert geocentric (spherical) coordinates to Cartesian position.
    
    Input:
    ------
      coords : GeocentricCoordinates
        Geocentric coordinates dataclass:
        - latitude  : float - Geocentric latitude [rad]
        - longitude : float - Longitude [rad]
        - altitude  : float - Altitude above spherical Earth [m]
        
    Output:
    -------
      pos_vec : np.ndarray
        Position vector in body-fixed frame [m].
    """
    radius = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + coords.altitude
    return GeographicCoordinateConverter.spherical_to_cartesian(
      coords.latitude, coords.longitude, radius
    )
  
  # ----------------------------
  # Geodetic (Ellipsoidal)
  # ----------------------------
  
  @staticmethod
  def pos_to_geodetic(
    pos_vec : np.ndarray,
  ) -> GeodeticCoordinates:
    """
    Convert Cartesian position to geodetic (ellipsoidal) coordinates.

    Uses SPICE's recgeo function with WGS84 ellipsoid parameters.

    Input:
    ------
      pos_vec : np.ndarray
        Position vector in IAU_EARTH frame [m].

    Output:
    -------
      coords : GeodeticCoordinates
        Geodetic coordinates dataclass containing:
        - latitude  : float - Geodetic latitude [rad]
        - longitude : float - Longitude [rad]
        - altitude  : float - Altitude above WGS84 ellipsoid [m]
    """
    # Delegate to array version with single position
    pos_vec = np.asarray(pos_vec).flatten().reshape(3, 1)
    coords_array = GeographicCoordinateConverter.pos_to_geodetic_array(pos_vec)

    return GeodeticCoordinates(
      latitude  = float(coords_array.latitude[0]),
      longitude = float(coords_array.longitude[0]),
      altitude  = float(coords_array.altitude[0]),
    )
  
  @staticmethod
  def geodetic_to_pos(
    coords : GeodeticCoordinates,
  ) -> np.ndarray:
    """
    Convert geodetic (ellipsoidal) coordinates to Cartesian position.
    
    Uses SPICE's georec function with WGS84 ellipsoid parameters.
    
    Input:
    ------
      coords : GeodeticCoordinates
        Geodetic coordinates dataclass:
        - latitude  : float - Geodetic latitude [rad]
        - longitude : float - Longitude [rad]
        - altitude  : float - Altitude above WGS84 ellipsoid [m]
        
    Output:
    -------
      pos_vec : np.ndarray
        Position vector in body-fixed frame [m].
    """
    # Convert m to km for SPICE
    altitude__km = coords.altitude * CONVERTER.KM_PER_M
    
    # SPICE georec expects (lon, lat, alt) in (rad, rad, km)
    pos_vec__km = spice.georec(
      coords.longitude,
      coords.latitude,
      altitude__km,
      GeographicCoordinateConverter.WGS84_RE,
      GeographicCoordinateConverter.WGS84_F,
    )
    
    # Convert km to m
    pos_vec = np.array(pos_vec__km) * 1000.0
    
    return pos_vec
  
  # ----------------------------
  # Vectorized versions
  # ----------------------------
  
  @staticmethod
  def pos_to_geocentric_array(
    pos_vec_array : np.ndarray,
  ) -> GeocentricCoordinates:
    """
    Convert array of Cartesian positions to geocentric coordinates.
    
    Input:
    ------
      pos_vec_array : np.ndarray
        Position vectors in body-fixed frame [m]. Shape (3, N).
        
    Output:
    -------
      coords : GeocentricCoordinates
        Geocentric coordinates dataclass with array attributes:
        - latitude  : np.ndarray - Geocentric latitudes [rad]
        - longitude : np.ndarray - Longitudes [rad]
        - altitude  : np.ndarray - Altitudes above spherical Earth [m]
    """
    pos_x = pos_vec_array[0, :]
    pos_y = pos_vec_array[1, :]
    pos_z = pos_vec_array[2, :]
    
    pos_mag   = np.linalg.norm(pos_vec_array, axis=0)
    latitude  = np.arcsin(pos_z / pos_mag)
    longitude = np.arctan2(pos_y, pos_x)
    altitude  = pos_mag - SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    
    return GeocentricCoordinates(
      latitude  = latitude,
      longitude = longitude,
      altitude  = altitude,
    )
  
  @staticmethod
  def pos_to_geodetic_array(
    pos_vec_array : np.ndarray,
  ) -> GeodeticCoordinates:
    """
    Convert array of Cartesian positions to geodetic coordinates.

    Input:
    ------
      pos_vec_array : np.ndarray
        Position vectors in body-fixed frame [m]. Shape (3, N).

    Output:
    -------
      coords : GeodeticCoordinates
        Geodetic coordinates dataclass with array attributes:
        - latitude  : np.ndarray - Geodetic latitudes [rad]
        - longitude : np.ndarray - Longitudes [rad]
        - altitude  : np.ndarray - Altitudes above WGS84 ellipsoid [m]
    """
    n_points = pos_vec_array.shape[1]

    latitude  = np.zeros(n_points)
    longitude = np.zeros(n_points)
    altitude  = np.zeros(n_points)

    # SPICE recgeo doesn't support vectorized calls, so we must loop
    for i in range(n_points):
      # Convert m to km for SPICE
      pos_vec__km = pos_vec_array[:, i] / 1000.0

      # SPICE recgeo returns (lon, lat, alt) in (rad, rad, km)
      longitude[i], latitude[i], altitude__km = spice.recgeo(
        pos_vec__km,
        GeographicCoordinateConverter.WGS84_RE,
        GeographicCoordinateConverter.WGS84_F,
      )

      # Convert alt back to m
      altitude[i] = altitude__km * 1000.0

    return GeodeticCoordinates(
      latitude  = latitude,
      longitude = longitude,
      altitude  = altitude,
    )
  
  # ----------------------------
  # Conversion between systems
  # ----------------------------
  
  @staticmethod
  def geocentric_to_geodetic(
    coords : GeocentricCoordinates,
  ) -> GeodeticCoordinates:
    """
    Convert geocentric coordinates to geodetic coordinates.
    
    Input:
    ------
      coords : GeocentricCoordinates
        Geocentric coordinates dataclass.
        
    Output:
    -------
      GeodeticCoordinates
        Geodetic coordinates dataclass.
    """
    pos_vec = GeographicCoordinateConverter.geocentric_to_pos(coords)
    return GeographicCoordinateConverter.pos_to_geodetic(pos_vec)
  
  @staticmethod
  def geodetic_to_geocentric(
    coords : GeodeticCoordinates,
  ) -> GeocentricCoordinates:
    """
    Convert geodetic coordinates to geocentric coordinates.
    
    Input:
    ------
      coords : GeodeticCoordinates
        Geodetic coordinates dataclass.
        
    Output:
    -------
      GeocentricCoordinates
        Geocentric coordinates dataclass.
    """
    pos_vec = GeographicCoordinateConverter.geodetic_to_pos(coords)
    return GeographicCoordinateConverter.pos_to_geocentric(pos_vec)


class TopocentricConverter:
  """
  Conversion between Cartesian position vectors and topocentric coordinates.

  Topocentric coordinates (azimuth, elevation, range) are the natural
  frame for ground-based observations and are fundamental to orbit
  determination from ground-based tracking data.
  """

  @staticmethod
  def pos_to_topocentric(
    sat_pos_vec : np.ndarray,
    tracker_lat : float,
    tracker_lon : float,
    tracker_alt : float,
  ) -> tuple[float, float, float]:
    """
    Convert satellite position to topocentric coordinates from a ground station.

    Input:
    ------
      sat_pos_vec : np.ndarray
        Satellite position vector in body-fixed frame [m].
      tracker_lat : float
        Tracker geodetic latitude [rad].
      tracker_lon : float
        Tracker geodetic longitude [rad].
      tracker_alt : float
        Tracker altitude above WGS84 ellipsoid [m].

    Output:
    -------
      azimuth : float
        Azimuth angle [rad] (0 = North, π/2 = East, clockwise positive).
      elevation : float
        Elevation angle [rad] (0 = horizon, π/2 = zenith).
      range : float
        Slant range to satellite [m].

    Notes:
    ------
      The azimuth convention is:
      -   0° = North
      -  90° = East
      - 180° = South
      - 270° = West

      The local topocentric frame (East-North-Up) is constructed using
      the geodetic latitude and longitude at the tracker location.
    """
    # Delegate to array version with single position
    sat_pos_vec = np.asarray(sat_pos_vec).flatten().reshape(3, 1)
    azimuth_arr, elevation_arr, range_arr = TopocentricConverter.pos_to_topocentric_array(
      sat_pos_vec, tracker_lat, tracker_lon, tracker_alt
    )
    return float(azimuth_arr[0]), float(elevation_arr[0]), float(range_arr[0])

  @staticmethod
  def pos_to_topocentric_array(
    sat_pos_array : np.ndarray,
    tracker_lat   : float,
    tracker_lon   : float,
    tracker_alt   : float,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert multiple satellite positions to topocentric coordinates from a ground station.

    This vectorized version is more efficient than calling pos_to_topocentric() in a loop.

    Input:
    ------
      sat_pos_array : np.ndarray
        Satellite position vectors in body-fixed frame [m], shape (3, N).
      tracker_lat : float
        Tracker geodetic latitude [rad].
      tracker_lon : float
        Tracker geodetic longitude [rad].
      tracker_alt : float
        Tracker altitude above WGS84 ellipsoid [m].

    Output:
    -------
      azimuth : np.ndarray
        Azimuth angles [rad] (0 = North, π/2 = East, clockwise positive), shape (N,).
      elevation : np.ndarray
        Elevation angles [rad] (0 = horizon, π/2 = zenith), shape (N,).
      range : np.ndarray
        Slant ranges to satellite [m], shape (N,).

    Notes:
    ------
      The azimuth convention is:
      -   0° = North
      -  90° = East
      - 180° = South
      - 270° = West

      The local topocentric frame (East-North-Up) is constructed using
      the geodetic latitude and longitude at the tracker location.
    """
    # Ensure array is numpy array with shape (3, N)
    sat_pos_array = np.asarray(sat_pos_array)
    if sat_pos_array.ndim == 1:
      sat_pos_array = sat_pos_array.reshape(3, 1)

    # Compute tracker position in body-fixed frame using existing converter
    tracker_coords = GeodeticCoordinates(
      latitude  = tracker_lat,
      longitude = tracker_lon,
      altitude  = tracker_alt,
    )
    tracker_pos_vec = GeographicCoordinateConverter.geodetic_to_pos(tracker_coords)

    # Compute relative position vectors (satellite - tracker) for all points
    # Shape: (3, N)
    rel_pos_array = sat_pos_array - tracker_pos_vec.reshape(3, 1)

    # Compute ranges for all points
    range_array = np.linalg.norm(rel_pos_array, axis=0)

    # Build local topocentric frame (ENU - East, North, Up) at tracker location
    sin_lat = np.sin(tracker_lat)
    cos_lat = np.cos(tracker_lat)
    sin_lon = np.sin(tracker_lon)
    cos_lon = np.cos(tracker_lon)

    # East unit vector
    east_dir = np.array([-sin_lon, cos_lon, 0.0])

    # North unit vector
    north_dir = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])

    # Up unit vector (geodetic normal)
    up_dir = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    # Project relative positions onto local ENU frame
    # Shape: (N,) for each component
    pos_east  =  east_dir @ rel_pos_array
    pos_north = north_dir @ rel_pos_array
    pos_up    =    up_dir @ rel_pos_array

    # Compute azimuth (from North, clockwise positive)
    azimuth_array = np.arctan2(pos_east, pos_north)

    # Compute elevation (angle above horizon)
    horizontal_range = np.sqrt(pos_east**2 + pos_north**2)
    elevation_array  = np.arctan2(pos_up, horizontal_range)

    return azimuth_array, elevation_array, range_array