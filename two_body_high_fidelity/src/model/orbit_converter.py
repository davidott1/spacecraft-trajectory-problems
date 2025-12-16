import warnings
import numpy as np

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
  
  # Alias for kepler function - converts mean anomaly to eccentric anomaly
  M2E_kepler = kepler
  
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
  def coe_to_pv(
    coe : dict,
    gp  : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert classical orbital elements to position and velocity vectors.
    Handles all orbit types including parabolic and rectilinear cases.
    
    Input:
    ------
    coe : dict
      sma  : semi-major axis [m]
      ecc  : eccentricity [-]
      inc  : inclination [rad]
      raan : RAAN [rad]
      aop  : argument of periapsis [rad]
      ta   : true anomaly [rad] (for non-rectilinear orbits)
      ea   : eccentric anomaly [rad] (for rectilinear elliptic only)
      For parabolic orbits (ecc≈1), one of the following must be provided
      to define the orbit's size, as 'sma' is infinite:
        - periapsis   : periapsis radius [m]
        - slr         : semi-latus rectum [m]
        - ang_mom_mag : angular momentum magnitude [m²/s]
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
    The code can handle the following orbit types:
      - circular      :  e = 0           a > 0
      - elliptical-2D :  0 < e < 1       a > 0
      - elliptical-1D :  e = 1           a > 0 and finite (rectilinear)
      - parabolic-2D  :  e = 1           a = inf
      - hyperbolic-2D :  e > 1           a < 0
    The code does not handle the following orbit types:
      - parabolic-1D  :  e = 1           a = ? (rectilinear)
      - hyperbolic-1D :  e > 1           a < 0 and finite (rectilinear)

    Source:
    -------
      Modified from
      Analytical Mechanics of Space Systems, Fourth Edition
      Hanspeter Schaub and John L. Junkins
      DOI: https://doi.org/10.2514/4.105210
    """
    # Extract orbital elements
    sma  = coe['sma' ]
    ecc  = coe['ecc' ]
    inc  = coe['inc' ]
    raan = coe['raan']
    aop  = coe['aop']

    # Rectilinear vs. non-rectilinear case handling
    if ecc == 1.0 and sma > 0 and np.isfinite(sma):
      # Rectilinear elliptic orbit case

      # Extract eccentric anomaly
      ea = coe.get('ea', None)
      if ea is None:
        raise ValueError("Eccentric anomaly 'ea' must be provided for rectilinear elliptic orbits")

      # Position and velocity magnitudes
      pos_mag = sma * (1 - ecc * np.cos(ea))
      vel_mag = np.sqrt(2 * gp / pos_mag - gp / sma)
      
      # Position vector
      pos_dir = np.array([
        np.cos(raan) * np.cos(aop) - np.sin(raan) * np.sin(aop) * np.cos(inc),
        np.sin(raan) * np.cos(aop) + np.cos(raan) * np.sin(aop) * np.cos(inc),
        np.sin( aop) * np.sin(inc)
      ])
      pos_vec = pos_mag * pos_dir
      
      # Velocity direction (along or opposite to position direction)
      if np.sin(ea) > 0:
        vel_vec = -vel_mag * pos_dir
      else:
        vel_vec =  vel_mag * pos_dir
    
    else:
      # Non-rectilinear cases: elliptic-2D, hyperbolic, parabolic

      # Extract true anomaly
      ta = coe.get('ta', None)  
      if ta is None:
        raise ValueError("True anomaly 'ta' must be provided for non-rectilinear orbits")

      # Orbit conic cases: parabolic vs. elliptic/hyperbolic
      if ecc == 1:
        # Parabolic case
        #   Priority cascade for size input parameter:
        #   1. 'periapsis' (highest priority)
        #   2. 'slr'
        #   3. 'ang_mom_mag' (lowest priority)

        # Fetch all possible inputs first
        periapsis_mag = coe.get('periapsis', None)
        slr_val       = coe.get('slr', None)
        ang_mom_mag   = coe.get('ang_mom_mag', None)

        # Apply priority cascade
        if periapsis_mag is not None:
          slr = 2 * periapsis_mag
          # Build a list of ignored parameters for a specific warning
          ignored_params = []
          if slr_val is not None:
            ignored_params.append("'slr'")
          if ang_mom_mag is not None:
            ignored_params.append("'ang_mom_mag'")
          if ignored_params:
            ignored_str = " and ".join(ignored_params)
            warning_msg = (
              "Multiple size parameters for non-rectilinear parabolic orbit found in 'coe' dict. "
              f"Using 'periapsis' (highest priority). Ignoring {ignored_str}."
            )
            warnings.warn(warning_msg, UserWarning)

        elif slr_val is not None:
          slr = slr_val
          # Warn if lower-priority key was also present
          if ang_mom_mag is not None:
            warnings.warn("Multiple parabolic input parameters found to coe_to_pv function. 'periapsis' is None. Using 'slr' and ignoring 'ang_mom_mag'.", UserWarning)
        
        elif ang_mom_mag is not None:
            slr = ang_mom_mag**2 / gp
            # No warning needed, this is the last resort
        
        else:
            # All three are None, this is a fatal error
            raise ValueError("Either 'periapsis', 'slr', or 'ang_mom_mag' must be provided for parabolic orbits")
      
      else:
        # Elliptic and hyperbolic cases
        slr = sma * (1 - ecc**2)  # semi-latus rectum

      # Position magnitude, true latitude angle, angular momentum magnitude
      pos_mag     = slr / (1 + ecc * np.cos(ta))  # orbit radius
      theta       = aop + ta                      # true latitude angle
      ang_mom_mag = np.sqrt(gp * slr)             # orbit angular momentum magnitude

      # Position vector
      pos_vec = np.array([
        pos_mag * (np.cos(raan) * np.cos(theta) - np.sin(raan) * np.sin(theta) * np.cos(inc)),
        pos_mag * (np.sin(raan) * np.cos(theta) + np.cos(raan) * np.sin(theta) * np.cos(inc)),
        pos_mag * (                                              np.sin(theta) * np.sin(inc))
      ])
      
      # Velocity vector
      vel_vec = np.array([
        -gp / ang_mom_mag * (np.cos(raan) * (np.sin(theta) + ecc * np.sin(aop)) + np.sin(raan) * (np.cos(theta) + ecc * np.cos(aop)) * np.cos(inc)),
        -gp / ang_mom_mag * (np.sin(raan) * (np.sin(theta) + ecc * np.sin(aop)) - np.cos(raan) * (np.cos(theta) + ecc * np.cos(aop)) * np.cos(inc)),
        -gp / ang_mom_mag * (                                                                   -(np.cos(theta) + ecc * np.cos(aop)) * np.sin(inc))
      ])
    
    return pos_vec, vel_vec

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
