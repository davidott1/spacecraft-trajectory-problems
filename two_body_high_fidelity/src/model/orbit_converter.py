import warnings
import numpy as np
import spiceypy as spice

from src.model.constants import SOLARSYSTEMCONSTANTS

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
  ) -> dict:
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
      coe : dict
        Dictionary containing orbital elements:
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

    return {
      'sma'  : sma,
      'ecc'  : ecc_mag,
      'inc'  : inc,
      'raan' : raan,
      'aop'  : aop,
      'ma'   : ma,
      'ta'   : ta,
      'ea'   : ea,
      'ha'   : ha,
      'pa'   : pa,
    }

  @staticmethod
  def pv_to_mee(
    pos_vec    : np.ndarray,
    vel_vec    : np.ndarray,
    gp         : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
    retrograde : bool  = False,
  ) -> dict:
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
    
    return {
      'p' : p,
      'f' : f,
      'g' : g,
      'h' : h,
      'k' : k,
      'L' : L,
      'I' : I,
    }

  @staticmethod
  def coe_to_pv(
    coe : dict,
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
    sma  = coe['sma']
    ecc  = coe['ecc']
    inc  = coe['inc']
    raan = coe['raan']
    aop  = coe['aop']
    ta   = coe['ta']
    
    # Gravitational parameter
    if 'gp' in coe:
      gp = coe['gp']
    
    # Compute semi-latus rectum
    if np.isinf(sma):
      # Parabolic case
      p = 0  # Semi-latus rectum is not used for parabolic orbits
    else:
      p = sma * (1 - ecc**2)
    
    # Position in orbital plane
    pqw_pos_vec = p * (1 - ecc) / (1 - ecc * np.cos(ta)) * np.array([ np.cos(ta), np.sin(ta), 0])
    
    # Velocity in orbital plane
    if np.isinf(sma):
      # Parabolic case
      pqw_vel_vec = np.sqrt(gp / (2 * np.linalg.norm(pqw_pos_vec))) * np.array([ -np.sin(ta), ecc - np.cos(ta), 0 ])
    else:
      # Elliptic case
      pqw_vel_vec = np.sqrt(gp / sma) * (1 - ecc * np.cos(ta)) * np.array([ -np.sin(ta), ecc - np.cos(ta), 0 ])
    
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
    mee : dict,
    gp  : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Modified Equinoctial Elements (MEE) to position and velocity vectors in the inertial frame.
    
    Input:
    ------
      mee : dict
        Dictionary containing modified equinoctial elements:
        - p : semi-latus rectum [m]
        - f : e*cos(ω + I*Ω) [-]
        - g : e*sin(ω + I*Ω) [-]
        - h : tan(i/2)*cos(Ω) [-] or cot(i/2)*cos(Ω) [-] (retrograde)
        - k : tan(i/2)*sin(Ω) [-] or cot(i/2)*sin(Ω) [-] (retrograde)
        - L : true longitude = ω + I*Ω + ν [rad]
        - I : retrograde factor (+1 prograde, -1 retrograde), optional (default +1)
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
    # Extract elements
    p = mee['p']
    f = mee['f']
    g = mee['g']
    h = mee['h']
    k = mee['k']
    L = mee['L']
    I = mee.get('I', 1)  # Default to prograde if not specified
    
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
    coe        : dict,
    retrograde : bool = False,
  ) -> dict:
    """
    Convert Classical Orbital Elements to Modified Equinoctial Elements.
    
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
      retrograde : bool
        If True, use retrograde factor (I = -1).
        
    Output:
    -------
      mee : dict
        Dictionary containing modified equinoctial elements.
    """
    # Extract elements
    sma  = coe['sma']
    ecc  = coe['ecc']
    inc  = coe['inc']
    raan = coe['raan']
    aop  = coe['aop']
    ta   = coe['ta']
    
    # Retrograde factor
    I = -1 if retrograde else 1
    
    # Semi-latus rectum
    if np.isinf(sma):
      # Parabolic case - need slr or periapsis from coe
      p = coe.get('slr', coe.get('periapsis', 0) * 2)
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
    
    return {
      'p' : p,
      'f' : f,
      'g' : g,
      'h' : h,
      'k' : k,
      'L' : L,
      'I' : I,
    }

  @staticmethod
  def mee_to_coe(
    mee : dict,
  ) -> dict:
    """
    Convert Modified Equinoctial Elements to Classical Orbital Elements.
    
    Input:
    ------
      mee : dict
        Dictionary containing modified equinoctial elements:
        - p : semi-latus rectum [m]
        - f : e*cos(ω + I*Ω) [-]
        - g : e*sin(ω + I*Ω) [-]
        - h : tan(i/2)*cos(Ω) [-] or cot(i/2)*cos(Ω) [-]
        - k : tan(i/2)*sin(Ω) [-] or cot(i/2)*sin(Ω) [-]
        - L : true longitude [rad]
        - I : retrograde factor (+1 or -1), optional (default +1)
        
    Output:
    -------
      coe : dict
        Dictionary containing classical orbital elements:
        - sma  : semi-major axis [m]
        - ecc  : eccentricity [-]
        - inc  : inclination [rad]
        - raan : right ascension of ascending node [rad]
        - aop  : argument of periapsis [rad]
        - ta   : true anomaly [rad]
    """
    # Extract elements
    p = mee['p']
    f = mee['f']
    g = mee['g']
    h = mee['h']
    k = mee['k']
    L = mee['L']
    I = mee.get('I', 1)
    
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
    
    return {
      'sma'  : sma,
      'ecc'  : ecc,
      'inc'  : inc,
      'raan' : raan,
      'aop'  : aop,
      'ta'   : ta,
      'ma'   : None,  # Not computed here
      'ea'   : None,
      'ha'   : None,
      'pa'   : None,
    }

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
  
  # ----------------------------
  # Geocentric (Spherical)
  # ----------------------------
  
  @staticmethod
  def pos_to_geocentric(
    pos_vec : np.ndarray,
  ) -> dict:
    """
    Convert Cartesian position to geocentric (spherical) coordinates.
    
    Input:
    ------
      pos_vec : np.ndarray
        Position vector in body-fixed frame [m].
        
    Output:
    -------
      coords : dict
        Dictionary containing:
        - latitude        : float - Geocentric latitude [rad]
        - longitude       : float - Longitude [rad]
        - altitude        : float - Altitude above spherical Earth [m]
    """
    pos_vec = np.asarray(pos_vec).flatten()
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]
    
    pos_mag   = np.linalg.norm(pos_vec)
    latitude  = np.arcsin(pos_z / pos_mag)
    longitude = np.arctan2(pos_y, pos_x)
    altitude  = pos_mag - SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    
    return {
      'latitude'  : latitude,
      'longitude' : longitude,
      'altitude'  : altitude,
    }
  
  @staticmethod
  def geocentric_to_pos(
    latitude  : float,
    longitude : float,
    altitude  : float,
  ) -> np.ndarray:
    """
    Convert geocentric (spherical) coordinates to Cartesian position.
    
    Input:
    ------
      latitude : float
        Geocentric latitude [rad].
      longitude : float
        Longitude [rad].
      altitude : float
        Altitude above spherical Earth [m].
        
    Output:
    -------
      pos_vec : np.ndarray
        Position vector in body-fixed frame [m].
    """
    pos_mag = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR + altitude
    
    pos_x = pos_mag * np.cos(latitude) * np.cos(longitude)
    pos_y = pos_mag * np.cos(latitude) * np.sin(longitude)
    pos_z = pos_mag * np.sin(latitude)
    
    return np.array([pos_x, pos_y, pos_z])
  
  # ----------------------------
  # Geodetic (Ellipsoidal)
  # ----------------------------
  
  @staticmethod
  def pos_to_geodetic(
    pos_vec : np.ndarray,
  ) -> dict:
    """
    Convert Cartesian position to geodetic (ellipsoidal) coordinates.
    
    Uses SPICE's recgeo function with WGS84 ellipsoid parameters.
    
    Input:
    ------
      pos_vec : np.ndarray
        Position vector in IAU_EARTH frame [m].
        
    Output:
    -------
      coords : dict
        Dictionary containing:
        - latitude  : float - Geodetic latitude [rad]
        - longitude : float - Longitude [rad]
        - altitude  : float - Altitude above WGS84 ellipsoid [m]
    """
    pos_vec = np.asarray(pos_vec).flatten()
    
    # Convert m to km for SPICE
    pos_vec__km = pos_vec / 1000.0
    
    # SPICE recgeo returns (lon, lat, alt) in (rad, rad, km)
    longitude, latitude, altitude__km = spice.recgeo(
      pos_vec__km,
      GeographicCoordinateConverter.WGS84_RE,
      GeographicCoordinateConverter.WGS84_F,
    )
    
    # Convert alt back to m
    altitude = altitude__km * 1000.0
    
    return {
      'latitude'  : latitude,
      'longitude' : longitude,
      'altitude'  : altitude,
    }
  
  @staticmethod
  def geodetic_to_pos(
    latitude  : float,
    longitude : float,
    altitude  : float,
  ) -> np.ndarray:
    """
    Convert geodetic (ellipsoidal) coordinates to Cartesian position.
    
    Uses SPICE's georec function with WGS84 ellipsoid parameters.
    
    Input:
    ------
      latitude : float
        Geodetic latitude [rad].
      longitude : float
        Longitude [rad].
      altitude : float
        Altitude above WGS84 ellipsoid [m].
        
    Output:
    -------
      pos_vec : np.ndarray
        Position vector in body-fixed frame [m].
    """
    # Convert m to km for SPICE
    altitude__km = altitude / 1000.0
    
    # SPICE georec expects (lon, lat, alt) in (rad, rad, km)
    pos_vec__km = spice.georec(
      longitude,
      latitude,
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
  ) -> dict:
    """
    Convert array of Cartesian positions to geocentric coordinates.
    
    Input:
    ------
      pos_vec_array : np.ndarray
        Position vectors in body-fixed frame [m]. Shape (3, N).
        
    Output:
    -------
      coords : dict
        Dictionary containing arrays:
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
    
    return {
      'latitude'  : latitude,
      'longitude' : longitude,
      'altitude'  : altitude,
    }
  
  @staticmethod
  def pos_to_geodetic_array(
    pos_vec_array : np.ndarray,
  ) -> dict:
    """
    Convert array of Cartesian positions to geodetic coordinates.
    
    Input:
    ------
      pos_vec_array : np.ndarray
        Position vectors in body-fixed frame [m]. Shape (3, N).
        
    Output:
    -------
      coords : dict
        Dictionary containing arrays:
        - latitude  : np.ndarray - Geodetic latitudes [rad]
        - longitude : np.ndarray - Longitudes [rad]
        - altitude  : np.ndarray - Altitudes above WGS84 ellipsoid [m]
    """
    n_points = pos_vec_array.shape[1]
    
    latitude  = np.zeros(n_points)
    longitude = np.zeros(n_points)
    altitude  = np.zeros(n_points)
    
    for i in range(n_points):
      coords = GeographicCoordinateConverter.pos_to_geodetic(pos_vec_array[:, i])
      latitude[i]  = coords['latitude']
      longitude[i] = coords['longitude']
      altitude[i]  = coords['altitude']
    
    return {
      'latitude'  : latitude,
      'longitude' : longitude,
      'altitude' : altitude,
    }
  
  # ----------------------------
  # Conversion between systems
  # ----------------------------
  
  @staticmethod
  def geocentric_to_geodetic(
    latitude_geocentric : float,
    longitude           : float,
    altitude_geocentric : float,
  ) -> dict:
    """
    Convert geocentric coordinates to geodetic coordinates.
    
    Input:
    ------
      latitude_geocentric : float
        Geocentric latitude [rad].
      longitude : float
        Longitude [rad] (same for both systems).
      altitude_geocentric : float
        Altitude above spherical Earth [m].
        
    Output:
    -------
      coords : dict
        Dictionary containing:
        - latitude  : float - Geodetic latitude [rad]
        - longitude : float - Longitude [rad]
        - altitude  : float - Altitude above WGS84 ellipsoid [m]
    """
    pos_vec = GeographicCoordinateConverter.geocentric_to_pos(latitude_geocentric, longitude, altitude_geocentric)
    return GeographicCoordinateConverter.pos_to_geodetic(pos_vec)
  
  @staticmethod
  def geodetic_to_geocentric(
    latitude_geodetic : float,
    longitude         : float,
    altitude_geodetic : float,
  ) -> dict:
    """
    Convert geodetic coordinates to geocentric coordinates.
    
    Input:
    ------
      latitude_geodetic : float
        Geodetic latitude [rad].
      longitude : float
        Longitude [rad] (same for both systems).
      altitude_geodetic : float
        Altitude above WGS84 ellipsoid [m].
        
    Output:
    -------
      coords : dict
        Dictionary containing:
        - latitude  : float - Geocentric latitude [rad]
        - longitude : float - Longitude [rad]
        - altitude  : float - Altitude above spherical Earth [m]
    """
    pos_vec = GeographicCoordinateConverter.geodetic_to_pos(latitude_geodetic, longitude, altitude_geodetic)
    return GeographicCoordinateConverter.pos_to_geocentric(pos_vec)