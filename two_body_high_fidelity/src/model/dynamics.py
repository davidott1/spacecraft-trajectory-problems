"""
Spacecraft Orbital Dynamics Module
==================================

High-fidelity orbital dynamics models for spacecraft trajectory propagation.

Summary:
--------
This module provides a comprehensive framework for modeling spacecraft orbital dynamics,
including gravitational forces (two-body and third-body), atmospheric drag, and solar 
radiation pressure. It features a hierarchical acceleration model architecture and 
integrates with SPICE for high-accuracy ephemerides.

Class Structure:
----------------
Acceleration Hierarchy:
    GeneralStateEquationsOfMotion (ODE interface)
    └── Acceleration (coordinator)
        ├── Gravity
        │   ├── TwoBodyGravity
        │   │   ├── point_mass()
        │   │   └── oblate (J2, J3, J4)
        │   └── ThirdBodyGravity
        │       ├── point_mass() (Sun, Moon)
        │       ├── oblate() (future)
        │       └── SPICE/analytical ephemerides
        ├── AtmosphericDrag
        │   └── Exponential density model
        └── SolarRadiationPressure

Main Components:
----------------
1. **GeneralStateEquationsOfMotion** - Defines the state derivative (d/dt [r, v] = [v, a]) for numerical integration.

2. **Acceleration** - Top-level coordinator that computes:
   total = gravity + drag + solar_radiation_pressure

3. **Gravity** - Gravitational acceleration coordinator with methods:
   - two_body_point_mass()
   - two_body_oblate()
   - third_body_point_mass()
   - third_body_oblate() (future)
   - relativity() (future)

4. **TwoBodyGravity** - Central body gravity:
   - point_mass() - Keplerian two-body
   - oblate_j2() - J2 oblateness
   - oblate_j3() - J3 oblateness
   - oblate_j4() - J4 oblateness
   - tesseral_22() - C22, S22 tesseral harmonics

5. **ThirdBodyGravity** - Perturbations from Sun, Moon, etc.:
   - point_mass() - Third-body point mass
   - SPICE ephemerides

6. **AtmosphericDrag** - Atmospheric drag model:
   - Exponential density model
   - Rotating atmosphere

7. **SolarRadiationPressure** - SRP model

Usage Example:
--------------
  from src.model.dynamics import Acceleration, GeneralStateEquationsOfMotion
  from src.model.constants import SOLARSYSTEMCONSTANTS
  
  # Initialize acceleration model
  acceleration = Acceleration(
      gp                = SOLARSYSTEMCONSTANTS.EARTH.GP,
      j2                = SOLARSYSTEMCONSTANTS.EARTH.J2,
      pos_ref           = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
      enable_drag       = True,
      cd                = 2.2,
      area_drag         = 10.0,
      mass              = 1000.0,
      enable_third_body = True,
  )
  
  # Create equations of motion
  eom = GeneralStateEquationsOfMotion(acceleration)
  
  # Integrate (e.g., with scipy.integrate.solve_ivp)
  state_dot = eom.state_time_derivative(time, state_vec)

Units:
------
- Position     : meters [m]
- Velocity     : meters per second [m/s]
- Acceleration : meters per second squared [m/s²]
- Time         : seconds [s]
- Angles       : radians [rad]

Notes:
------
- All calculations performed in inertial J2000 frame
- Third-body positions from SPICE kernels or analytical approximations
- Atmospheric model accounts for Earth rotation

Sources:
--------
- Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.). Microcosm Press.
- Montenbruck, O., & Gill, E. (2000). Satellite Orbits: Models, Methods and Applications. Springer.
"""

import numpy    as np
import spiceypy as spice

from pathlib import Path
from typing  import Optional

from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER, NAIFIDS
from src.model.frame_converter import FrameConverter


# =============================================================================
# Gravity Components (bottom of hierarchy)
# =============================================================================

class TwoBodyGravity:
  """
  Two-body gravitational acceleration components
  Handles point mass and oblateness (J2, J3, J4) perturbations
  """
  
  def __init__(
    self,
    gp      : float,
    j2      : float = 0.0,
    j3      : float = 0.0,
    j4      : float = 0.0,
    c22     : float = 0.0,
    s22     : float = 0.0,
    pos_ref : float = 0.0,
  ):
    """
    Initialize two-body gravity model
    
    Input:
    ------
      gp : float
        Gravitational parameter of central body [m³/s²]
      j2 : float
        J2 harmonic coefficient for oblateness
      j3 : float
        J3 harmonic coefficient for oblateness
      j4 : float
        J4 harmonic coefficient for oblateness
      c22 : float
        C22 tesseral harmonic coefficient
      s22 : float
        S22 tesseral harmonic coefficient
      pos_ref : float
        Reference radius for harmonic coefficients [m]
            
    Output:
    -------
      None
    """
    self.gp      = gp
    self.pos_ref = pos_ref
    self.j2      = j2
    self.j3      = j3
    self.j4      = j4
    self.c22     = c22
    self.s22     = s22
  
  def point_mass(
    self,
    pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Two-body point mass gravity
    
    Input:
    ------
      pos_vec : np.ndarray
        Position vector [m]
    
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²]
    """
    pos_mag = np.linalg.norm(pos_vec)
    return -self.gp * pos_vec / pos_mag**3
  
  def oblate_j2(
      self,
      time    : float,
      pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    J2 oblateness perturbation
    
    Input:
    ------
      time : float
        Current time [s]
      pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).
    
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²]

    Notes:
    ------
      Technically, zonal harmonics are defined in the Body-Fixed frame.
      However, Zonal harmonics (J2, J3...) are rotationally symmetric about the 
      Z-axis (longitude independent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).
      
      This implementation assumes the Inertial Z-axis is aligned with the 
      Body Z-axis (ignoring Precession/Nutation). Under this assumption, 
      inertial coordinates can be used directly.
    """
    if self.j2 == 0.0:
      return np.zeros(3)
    
    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2
    pos_mag_pwr5 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag
    
    factor = 1.5 * self.j2 * self.gp * self.pos_ref**2 / pos_mag_pwr5
    
    acc_vec    = np.zeros(3)
    acc_vec[0] = factor * pos_vec[0] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 1)
    acc_vec[1] = factor * pos_vec[1] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 1)
    acc_vec[2] = factor * pos_vec[2] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 3)
    
    return acc_vec
  
  def tesseral_22(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute C22 and S22 tesseral harmonic perturbation acceleration.
    
    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in inertial frame (J2000).
    
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²] in Inertial frame.
    """
    if self.c22 == 0.0 and self.s22 == 0.0:
      return np.zeros(3)

    # Get rotation matrix from J2000 to Body-Fixed (IAU_EARTH)
    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      # Fallback or return zero if kernels not loaded/available
      return np.zeros(3)

    # Rotate position to Body-Fixed frame
    iau_earth_pos_vec   = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    pos_x, pos_y, pos_z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]
    
    pos_mag_pwr2 = pos_x**2 + pos_y**2 + pos_z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr7 = pos_mag_pwr2**3 * pos_mag
    
    # Pre-compute terms for efficiency
    term_common = self.c22 * (pos_x**2 - pos_y**2) + 2.0 * self.s22 * pos_x * pos_y
    factor      = 3.0 * self.gp * self.pos_ref**2 / pos_mag_pwr7
    
    # Acceleration
    #   potential -> acc_vec = gradient(potential) 
    #     U22 = (3 * gp * earth_radius^2 / pos_mag^5) * (C22*(pos_x^2 - pos_y^2) + 2*S22*pos_x*pos_y)
    #     acc_vec = d/dpos_vec(U22)
    iau_earth_acc_x   = factor * (-5.0 * pos_x * term_common + pos_mag_pwr2 * ( 2.0 * self.c22 * pos_x + 2.0 * self.s22 * pos_y))
    iau_earth_acc_y   = factor * (-5.0 * pos_y * term_common + pos_mag_pwr2 * (-2.0 * self.c22 * pos_y + 2.0 * self.s22 * pos_x))
    iau_earth_acc_z   = factor * (-5.0 * pos_z * term_common)
    iau_earth_acc_vec = np.array([iau_earth_acc_x, iau_earth_acc_y, iau_earth_acc_z])
    
    # Rotate acceleration back to inertial frame
    j2000_acc_vec = rot_mat_j2000_to_iau_earth.T @ iau_earth_acc_vec
    
    # Return acceleration vector in inertial frame
    return j2000_acc_vec

  def oblate_j3(
    self,
    time    : float,
    pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute J3 oblateness perturbation acceleration.
    
    Input:
    ------
      time : float
        Current time [s]
      pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).
    
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²]

    Notes:
    ------
      Technically, zonal harmonics are defined in the Body-Fixed frame.
      However, Zonal harmonics (J2, J3...) are rotationally symmetric about the 
      Z-axis (longitude independent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).
      
      This implementation assumes the Inertial Z-axis is aligned with the 
      Body Z-axis (ignoring Precession/Nutation). Under this assumption, 
      inertial coordinates can be used directly.
    """
    if self.j3 == 0.0:
        return np.zeros(3)
      
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]

    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2
    pos_mag_pwr7 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag_pwr2 * pos_mag
    
    factor = 2.5 * self.j3 * self.gp * self.pos_ref**3 / pos_mag_pwr7
      
    acc_vec    = np.zeros(3)
    acc_vec[0] = factor * pos_x * pos_z * (3.0 - 7.0 * pos_z**2 / pos_mag_pwr2)
    acc_vec[1] = factor * pos_y * pos_z * (3.0 - 7.0 * pos_z**2 / pos_mag_pwr2)
    acc_vec[2] = factor * (3.0 * pos_z**2 - 7.0 * pos_z**4 / pos_mag_pwr2 - 0.6 * pos_mag_pwr2)
    
    return acc_vec
  
  def oblate_j4(
    self,
    time    : float,
    pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    J4 oblateness perturbation
    
    Input:
    ------
      time : float
        Current time [s]
      pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).
    
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²]

    Notes:
    ------
      Technically, zonal harmonics are defined in the Body-Fixed frame.
      However, Zonal harmonics (J2, J3...) are rotationally symmetric about the 
      Z-axis (longitude independent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).
      
      This implementation assumes the Inertial Z-axis is aligned with the 
      Body Z-axis (ignoring Precession/Nutation). Under this assumption, 
      inertial coordinates can be used directly.
    """
    if self.j4 == 0.0:
      return np.zeros(3)
    
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]
    
    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag
    
    term_common = pos_z**2 / pos_mag_pwr2
    factor      = 1.875 * self.j4 * self.gp * self.pos_ref**4 / pos_mag_pwr9
    
    acc_vec    = np.zeros(3)
    acc_vec[0] = factor * pos_x * (1.0 - 14.0 * term_common + 21.0 * term_common**2)
    acc_vec[1] = factor * pos_y * (1.0 - 14.0 * term_common + 21.0 * term_common**2)
    acc_vec[2] = factor * pos_z * (5.0 - 70.0 * term_common / 3.0 + 21.0 * term_common**2)
    
    return acc_vec
    

class ThirdBodyGravity:
    """
    Third-body gravitational perturbations from Sun, Moon, and other bodies.
    Uses SPICE ephemerides.
    """
    
    def __init__(
      self,
      bodies : list = None,
    ):
      """
      Initialize third-body gravity model.
      
      Input:
      ------
        bodies : list
            Which bodies to include (default: ['sun', 'moon']).
              
      Output:
      -------
        None
      """
      self.bodies = bodies if bodies else ['sun', 'moon']
    
    def _get_position_body_spice(
      self,
      body_name : str,
      time_et   : float,
      frame     : str = 'J2000',
    ) -> np.ndarray:
      """
      Get position of celestial body at given time using SPICE.
      
      Input:
      ------
        body_name : str
          Body name ('SUN' or 'MOON').
        time_et : float
          Ephemeris time in seconds past J2000 epoch.
        frame : str
          Reference frame (default: 'J2000').
      
      Output:
      -------
        pos_vec : np.ndarray
          Position vector [m].
      """
      # SPICE state relative to Earth
      state, _ = spice.spkez(
          targ   = self._get_naif_id(body_name),
          et     = time_et,
          ref    = frame,
          abcorr = 'NONE',
          obs    = 399  # relative to Earth
      )
      # SPICE returns km, convert to m
      return np.array(state[0:3]) * CONVERTER.M_PER_KM
    
    def _get_naif_id(
      self,
      body_name : str,
    ) -> int:
      """
      Get NAIF ID for body.
      
      Input:
      ------
        body_name : str
          Body name (e.g. 'SUN', 'JUPITER').
      
      Output:
      -------
        naif_id : int
          NAIF ID code.
      """
      body_upper = body_name.upper()
      if body_upper in NAIFIDS.NAME_TO_ID:
        return NAIFIDS.NAME_TO_ID[body_upper]
      
      raise ValueError(f"Unknown body name for NAIF ID lookup: {body_name}")

    def point_mass(
      self,
      time        : float,
      pos_sat_vec : np.ndarray,
    ) -> np.ndarray:
      """
      Compute third-body point mass perturbations (Sun, Moon, etc.).
      
      Input:
      ------
          time : float
              Current Ephemeris Time (ET) [s].
          pos_sat_vec : np.ndarray
              Satellite position vector [m].
      
      Output:
      -------
          acc_vec : np.ndarray
              Third-body acceleration [m/s²].
      """
      # Ephemeris time is seconds from J2000 epoch
      et_seconds = time
      
      # Compute acceleration for all bodies
      acc_vec = np.zeros(3)
      for body in self.bodies:
        body_upper = body.upper()
        
        # Skip Earth if it's in the list (it's the central body)
        if body_upper == 'EARTH':
          continue

        # Get gravitational parameter [m³/s²]
        if hasattr(SOLARSYSTEMCONSTANTS, body_upper):
          GP = getattr(SOLARSYSTEMCONSTANTS, body_upper).GP
        else:
          continue

        # Position of central body (Earth) to perturbing body [m]
        pos_centbody_to_pertbody_vec = self._get_position_body_spice(body, et_seconds)
        pos_centbody_to_pertbody_mag = np.linalg.norm(pos_centbody_to_pertbody_vec)
        
        # Safety check
        if pos_centbody_to_pertbody_mag == 0:
          continue

        # Position of satellite to perturbing body [m]
        pos_sat_to_pertbody_vec = pos_centbody_to_pertbody_vec - pos_sat_vec
        pos_sat_to_pertbody_mag = np.linalg.norm(pos_sat_to_pertbody_vec)

        # Third-body acceleration contribution [m/s²]
        acc_vec += (
            GP * pos_sat_to_pertbody_vec / pos_sat_to_pertbody_mag**3
            - GP * pos_centbody_to_pertbody_vec / pos_centbody_to_pertbody_mag**3
        )

      return acc_vec
    

class Gravity:
    """
    Gravitational acceleration coordinator.
    
    Computes gravity as:
        gravity = two_body_point_mass + two_body_oblate + third_body_point_mass + 
                  third_body_oblate (future) + relativity (future)
    """
    
    def __init__(
      self,
      gp                      : float,
      j2                      : float = 0.0,
      j3                      : float = 0.0,
      j4                      : float = 0.0,
      c22                     : float = 0.0,
      s22                     : float = 0.0,
      pos_ref                 : float = 0.0,
      enable_third_body       : bool  = False,
      third_body_bodies       : list  = None,
    ):
      """
      Initialize gravity acceleration components.
      
      Input:
      ------
        gp : float
          Gravitational parameter of central body [m³/s²].
        j2 : float
          J2 harmonic coefficient for oblateness.
        j3 : float
          J3 harmonic coefficient for oblateness.
        j4 : float
          J4 harmonic coefficient for oblateness.
        c22 : float
          C22 tesseral harmonic coefficient.
        s22 : float
          S22 tesseral harmonic coefficient.
        pos_ref : float
          Reference radius for harmonic coefficients [m].
        enable_third_body : bool
          Enable Sun/Moon gravitational perturbations.
        third_body_bodies : list
          Which bodies to include (default: ['sun', 'moon']).
              
      Output:
      -------
        None
      """
      # Two-body gravity
      self.two_body = TwoBodyGravity(
        gp      = gp,
        j2      = j2,
        j3      = j3,
        j4      = j4,
        c22     = c22,
        s22     = s22,
        pos_ref = pos_ref,
      )
        
      # Third-body gravity
      self.enable_third_body = enable_third_body
      if self.enable_third_body:
        self.third_body = ThirdBodyGravity(
          bodies = third_body_bodies,
        )
      else:
        self.third_body = None
    
    def compute(
      self,
      time    : float,
      pos_vec : np.ndarray,
    ) -> np.ndarray:
      """
      Compute total gravity acceleration.
      
      Input:
      ------
        time : float
          Current Ephemeris Time (ET) [s].
        pos_vec : np.ndarray
          Position vector [m].
      
      Output:
      -------
        acc_vec : np.ndarray
          Total gravity acceleration [m/s²].
      """
      # Initialize acceleration vector
      acc_vec = np.zeros(3)
      
      # Two-body point mass
      acc_vec += self.two_body.point_mass(pos_vec)
      
      # Two-body oblateness (J2, J3, J4)
      acc_vec += self.two_body.oblate_j2(time, pos_vec)
      acc_vec += self.two_body.oblate_j3(time, pos_vec)
      acc_vec += self.two_body.oblate_j4(time, pos_vec)
      
      # Two-body tesseral (C22, S22)
      acc_vec += self.two_body.tesseral_22(time, pos_vec)
      
      # Third-body contributions
      if self.enable_third_body and self.third_body is not None:
        acc_vec += self.third_body.point_mass(time, pos_vec)
      
      # Future: third_body_oblate, relativity
      
      return acc_vec
    

# =============================================================================
# Non-Gravitational Accelerations
# =============================================================================

class AtmosphericDrag:
    """
    Atmospheric drag acceleration using exponential atmosphere model
    """
    
    def __init__(
      self,
      cd   : float = 2.2,
      area : float = 0.0,
      mass : float = 1.0,
    ):
      """
      Initialize drag model
      
      Input:
      ------
        cd : float
          Drag coefficient.
        area : float
          Cross-sectional area [m²].
        mass : float
          Spacecraft mass [kg].
              
      Output:
      -------
        None
      """
      self.cd   = cd
      self.area = area
      self.mass = mass
    
    def compute(
      self,
      pos_vec : np.ndarray,
      vel_vec : np.ndarray,
    ) -> np.ndarray:
      """
      Compute drag acceleration
      
      Input:
      ------
        pos_vec : np.ndarray
          Position vector [m]
        vel_vec : np.ndarray
          Velocity vector [m/s]
      
      Output:
      -------
        acc_vec : np.ndarray
          Drag acceleration [m/s²]
      """
      # Position magnitude and altitude
      pos_mag = np.linalg.norm(pos_vec)
      alt     = pos_mag - SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
      
      # Atmospheric density at current altitude
      rho = self._atmospheric_density(float(alt))
      
      # Velocity relative to rotating atmosphere
      omega_earth = np.array([0, 0, SOLARSYSTEMCONSTANTS.EARTH.OMEGA])
      vel_rel_vec = vel_vec - np.cross(omega_earth, pos_vec)
      vel_rel_mag = np.linalg.norm(vel_rel_vec)
      
      if vel_rel_mag == 0:
        return np.zeros(3)
      
      # Drag acceleration
      acc_drag_mag = 0.5 * rho * (self.cd * self.area / self.mass) * vel_rel_mag**2
      acc_drag_dir = -vel_rel_vec / vel_rel_mag
      
      return acc_drag_mag * acc_drag_dir
    
    def _atmospheric_density(
      self,
      altitude : float,
    ) -> float:
      """
      Simplified exponential atmospheric density model
      
      Input:
      ------
        altitude : float
          Altitude above Earth's surface [m]
      
      Output:
      -------
        density : float
          Atmospheric density [kg/m³]
      """
      if altitude < 0:
        altitude = 0
      
      # Simplified exponential model
      rho = SOLARSYSTEMCONSTANTS.EARTH.RHO_0 * np.exp(-altitude / SOLARSYSTEMCONSTANTS.EARTH.H_0)
      
      return rho


class SolarRadiationPressure:
    """
    Solar radiation pressure acceleration model.
    
    Computes the acceleration due to solar radiation pressure on a spacecraft,
    accounting for the spacecraft's reflectivity, cross-sectional area, and mass.
    Includes cylindrical Earth shadow model.
    """
    
    def __init__(
      self,
      cr   : float = 1.3,
      area : float = 0.0,
      mass : float = 1.0,
    ):
      """
      Initialize SRP model
      
      Input:
      ------
        cr : float
          Radiation pressure coefficient (1.0 = absorbing, 2.0 = reflecting)
        area : float
          Cross-sectional area [m²]
        mass : float
          Spacecraft mass [kg]
              
      Output:
      -------
        None
      """
      self.cr   = cr
      self.area = area
      self.mass = mass
    
    def compute(
      self,
      time                 : float,
      earth_to_sat_pos_vec : np.ndarray,
    ) -> np.ndarray:
      """
      Compute SRP acceleration.
      
      Input:
      ------
        time : float
          Current Ephemeris Time (ET) [s]
        earth_to_sat_pos_vec : np.ndarray
          Spacecraft position vector relative to Earth [m], i.e. Earth to spacecraft vector.
      
      Output:
      -------
        earth_to_sat_acc_vec : np.ndarray
          SRP acceleration [m/s²]
      """
      # Check for valid parameters
      if self.area <= 0 or self.mass <= 0:
        return np.zeros(3)
      
      # Get Sun position relative to Earth using SPICE
      earth_to_sun_pos_vec = self._get_sun_position(time)
      
      # Vector from spacecraft to Sun
      sat_to_sun_pos_vec = earth_to_sun_pos_vec - earth_to_sat_pos_vec
      sat_to_sun_pos_mag = np.linalg.norm(sat_to_sun_pos_vec)
      sat_to_sun_pos_dir = sat_to_sun_pos_vec / sat_to_sun_pos_mag

      # Direction of solar radiation pressure force is from Sun to spacecraft
      acc_dir = -sat_to_sun_pos_dir
      
      # Compute shadow factor (0.0 = full shadow, 1.0 = full sunlight)
      shadow_factor = self._compute_shadow_factor(earth_to_sat_pos_vec, earth_to_sun_pos_vec)
      
      # If in full shadow, no SRP acceleration
      if shadow_factor == 0.0:
        return np.zeros(3)
      
      # Distance from Sun to spacecraft [m]
      sun_to_sat_pos_mag = sat_to_sun_pos_mag
      
      # Solar radiation pressure at spacecraft distance
      #   P = P_at_1au * ( 1_au / r_au )^2 = 4.56e-6 N/m² * ( 149597870700 m / r_m )^2
      pressure_srp  = SOLARSYSTEMCONSTANTS.EARTH.PRESSURE_SRP * (CONVERTER.M_PER_AU * CONVERTER.ONE_AU / sun_to_sat_pos_mag)**2
      
      # SRP acceleration magnitude
      acc_mag = (pressure_srp * self.cr * self.area / self.mass) * shadow_factor
      
      # SRP acceleration direction (away from Sun)
      acc_vec = acc_mag * acc_dir
      
      return acc_vec
    
    def _get_sun_position(
      self,
      time_et : float,
    ) -> np.ndarray:
      """
      Get Sun position relative to Earth at given time using SPICE.
      
      Input:
      ------
        time_et : float
          Ephemeris time in seconds past J2000 epoch.
      
      Output:
      -------
        sun_pos_vec : np.ndarray
          Sun position vector relative to Earth [m].
      """
      # Get Sun position relative to Earth
      state, _ = spice.spkez(
        targ   = 10,       # Sun NAIF ID
        et     = time_et,
        ref    = 'J2000',
        abcorr = 'NONE',
        obs    = 399       # Earth NAIF ID
      )
      
      # SPICE returns km, convert to m
      return np.array(state[0:3]) * CONVERTER.M_PER_KM
    
    def _compute_shadow_factor(
      self,
      earth_to_sat_pos_vec : np.ndarray,
      earth_to_sun_pos_vec : np.ndarray,
    ) -> float:
      """
      Compute shadow factor using cylindrical Earth shadow model.
      
      Input:
      ------
        earth_to_sat_pos_vec : np.ndarray
          Spacecraft position vector relative to Earth [m].
        earth_to_sun_pos_vec : np.ndarray
          Sun position vector relative to Earth [m].
      
      Output:
      -------
        shadow_factor : float
          0.0 = full shadow (umbra), 1.0 = full sunlight.
      
      Notes:
      ------
        Uses a simplified cylindrical shadow model where the shadow is a
        cylinder with radius equal to Earth's equatorial radius, extending
        from Earth in the anti-Sun direction.
      """
      # Direction from Earth to Sun
      earth_to_sun_pos_dir = earth_to_sun_pos_vec / np.linalg.norm(earth_to_sun_pos_vec)
      
      # Project spacecraft position onto Sun direction
      # (distance along Sun-Earth line, positive toward Sun)
      earth_to_sat_parallel_pos_mag = np.dot(earth_to_sat_pos_vec, earth_to_sun_pos_dir)
      
      # Check if spacecraft is in front of or behind Earth relative to Sun
      if earth_to_sat_parallel_pos_mag >= 0:
        # Spacecraft is on the sunlit side of Earth if positive projection
        shadow_factor = 1.0
      else:
        # Spacecraft is on the shadow side of Earth if negative projection

        # Calculate perpendicular distance from spacecraft to Sun-Earth line
        earth_to_sat_perpendicular_pos_vec = earth_to_sat_pos_vec - earth_to_sat_parallel_pos_mag * earth_to_sun_pos_dir
        earth_to_sat_perpendicular_pos_mag = np.linalg.norm(earth_to_sat_perpendicular_pos_vec)
        
        # Check if within Earth's shadow cylinder
        if earth_to_sat_perpendicular_pos_mag > SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR:
          # If perpendicular distance is greater than Earth radius, spacecraft is in sunlight
          shadow_factor = 1.0
        else:
          # Spacecraft is in Earth's cylindrical shadow
          shadow_factor = 0.0
        
      # Return shadow factor
      return shadow_factor


# =============================================================================
# Top-Level Coordinator
# =============================================================================

class Acceleration:
    """
    Acceleration coordinator - orchestrates all acceleration components
    
    Computes total acceleration as:
      total = gravity + drag + solar_radiation_pressure
    
    where:
      gravity = two_body_point_mass + third_body_point_mass + 
                two_body_oblate (J2, J3, J4) + relativity (future)
    """
    
    def __init__(
      self,
      gp                      : float,
      j2                      : float = 0.0,
      j3                      : float = 0.0,
      j4                      : float = 0.0,
      c22                     : float = 0.0,
      s22                     : float = 0.0,
      pos_ref                 : float = 0.0,
      mass                    : float = 1.0,
      enable_drag             : bool  = False,
      cd                      : float = 0.0,
      area_drag               : float = 0.0,
      enable_third_body       : bool  = False,
      third_body_bodies       : list  = None,
      enable_srp              : bool  = False,
      cr                      : float = 0.0,
      area_srp                : float = 0.0,
    ):
      """
      Initialize acceleration coordinator
      
      Input:
      ------
        gp : float
          Gravitational parameter of central body [m³/s²]
        j2 : float
          J2 harmonic coefficient for oblateness
        j3 : float
          J3 harmonic coefficient for oblateness
        j4 : float
          J4 harmonic coefficient for oblateness
        c22 : float
          C22 tesseral harmonic coefficient
        s22 : float
          S22 tesseral harmonic coefficient
        pos_ref : float
          Reference radius for harmonic coefficients [m]
        mass : float
          Spacecraft mass [kg]
        enable_drag : bool
          Enable atmospheric drag
        cd : float
          Drag coefficient
        area_drag : float
          Cross-sectional area [m²]
        enable_third_body : bool
          Enable Sun/Moon gravitational perturbations
        third_body_bodies : list of str
          Which bodies to include (default: ['sun', 'moon'])
        enable_srp : bool
          Enable solar radiation pressure (not yet implemented)
        cr : float
          Radiation pressure coefficient
        area_srp : float
          Cross-sectional area for SRP [m²]
              
      Output:
      -------
        None
      """
        # Create acceleration component instances
      self.gravity = Gravity(
        gp                      = gp,
        j2                      = j2,
        j3                      = j3,
        j4                      = j4,
        c22                     = c22,
        s22                     = s22,
        pos_ref                 = pos_ref,
        enable_third_body       = enable_third_body,
        third_body_bodies       = third_body_bodies,
      )
      
      self.enable_drag = enable_drag
      if self.enable_drag and cd > 0 and area_drag > 0 and mass > 0:
        self.drag = AtmosphericDrag(
          cd   = cd,
          area = area_drag,
          mass = mass,
        )
      else:
        self.drag = None
      
      self.enable_srp = enable_srp
      if self.enable_srp:
        self.srp = SolarRadiationPressure(
          cr   = cr,
          area = area_srp,
          mass = mass,
        )
      else:
          self.srp = None
    
    def compute(
      self,
      time    : float,
      pos_vec : np.ndarray,
      vel_vec : np.ndarray,
    ) -> np.ndarray:
      """
      Compute total acceleration from all components
      
      Input:
      ------
        time : float
          Current Ephemeris Time (ET) [s]
        pos_vec : np.ndarray
          Position vector [m]
        vel_vec : np.ndarray
          Velocity vector [m/s]
      
      Output:
      -------
        acc_vec : np.ndarray
          Total acceleration [m/s²]
      """
      # Gravity (always)
      acc_vec = self.gravity.compute(time, pos_vec)
      
      # Atmospheric drag (optional)
      if self.drag is not None:
        acc_vec += self.drag.compute(pos_vec, vel_vec)
      
      # Solar radiation pressure (optional)
      if self.srp is not None:
        acc_vec += self.srp.compute(time, pos_vec)
      
      return acc_vec


# =============================================================================
# Equations of Motion
# =============================================================================

class GeneralStateEquationsOfMotion:
  """
  General state equations of motion for orbit propagation
  """
  
  def __init__(
    self,
    acceleration : Acceleration,
  ):
    """
    Initialize equations of motion
    
    Input:
    ------
      acceleration : Acceleration
        Acceleration coordinator instance
            
    Output:
    -------
      None
    """
    self.acceleration = acceleration
  
  def state_time_derivative(
      self,
      time      : float,
      state_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute state time derivative for ODE integration
    
    Input:
    ------
      time : float
        Current Ephemeris Time (ET) [s]
      state_vec : np.ndarray
        Current state vector [pos, vel] [m, m/s]
    
    Output:
    -------
      state_dot_vec : np.ndarray
        Time derivative of state vector [vel, acc] [m/s, m/s²]
    """
    pos_vec = state_vec[0:3]
    vel_vec = state_vec[3:6]
    acc_vec = self.acceleration.compute(time, pos_vec, vel_vec)
    
    state_dot_vec      = np.zeros(6)
    state_dot_vec[0:3] = vel_vec
    state_dot_vec[3:6] = acc_vec
    
    return state_dot_vec




