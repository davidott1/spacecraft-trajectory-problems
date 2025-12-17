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
  Handles point mass and oblateness (J2, J3, J4, J5, J6) perturbations
  """
  
  def __init__(
    self,
    gp      : float,
    j2      : float = 0.0,
    j3      : float = 0.0,
    j4      : float = 0.0,
    j5      : float = 0.0,
    j6      : float = 0.0,
    c22     : float = 0.0,
    s22     : float = 0.0,
    pos_ref : float = 0.0,
  ):Initialize two-body gravity model
    """
    Initialize two-body gravity model
    ------
    Input: float
    ------avitational parameter of central body [m³/s²]
      gp : float
        Gravitational parameter of central body [m³/s²]
      j2 : float
        J2 harmonic coefficient for oblateness
      j3 : float
        J3 harmonic coefficient for oblateness
      j4 : floatt
        J4 harmonic coefficient for oblateness
      j5 : floatt
        J5 harmonic coefficient for oblateness
      j6 : floatfloat
        J6 harmonic coefficient for oblatenessents [m]
      c22 : float
        C22 tesseral harmonic coefficient
      s22 : float
        S22 tesseral harmonic coefficient
      pos_ref : float
        Reference radius for harmonic coefficients [m]
            _ref = pos_ref
    Output:      = j2
    -------      = j3
      None4      = j4
    """f.c22     = c22
    self.gp      = gp2
    self.pos_ref = pos_ref
    self.j2      = j2
    self.j3      = j3
    self.j4      = j4ray,
    self.j5      = j5
    self.j6      = j6
    self.c22     = c22s gravity
    self.s22     = s22
    Input:
  def point_mass(
    self,_vec : np.ndarray
    pos_vec : np.ndarray,m]
  ) -> np.ndarray:
    """put:
    Two-body point mass gravity
      acc_vec : np.ndarray
    Input:celeration vector [m/s²]
    ------
      pos_vec : np.ndarrayrm(pos_vec)
        Position vector [m]ec / pos_mag**3
    
    Output:e_j2(
    -------
      acc_vec : np.ndarrayt,rray
        Acceleration vector [m/s²][m/s²]
    """np.ndarray:
    pos_mag = np.linalg.norm(pos_vec)
    return -self.gp * pos_vec / pos_mag**3-fixed frame for higher accuracy)/ pos_mag**3
    
  def oblate_j2(
      self,
      time    : float,
      pos_vec : np.ndarray,
  ) -> np.ndarray:.ndarray
    """ Position vector [m] in Inertial frame (J2000).
    J2 oblateness perturbation
    Output:
    Input:-
    ------vec : np.ndarray
      time : floaton vector [m/s²]
        Current time [s]
      pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).
      Technically, zonal harmonics are defined in the Body-Fixed frame.
    Output:er, Zonal harmonics (J2, J3...) are rotationally symmetric about the 
    -------s (longitude independent). Therefore, the Earth's daily rotation (spin) 
      acc_vec : np.ndarrayforce, only the orientation of the Pole (Z-axis).
        Acceleration vector [m/s²]
      This implementation assumes the Inertial Z-axis is aligned with the 
    Notes: Z-axis (ignoring Precession/Nutation). Under this assumption, 
    ------tial coordinates can be used directly.
      Technically, zonal harmonics are defined in the Body-Fixed frame.
      However, Zonal harmonics (J2, J3...) are rotationally symmetric about the 
      Z-axis (longitude independent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).
      Transform to body-fixed frame for accurate computation
      This implementation assumes the Inertial Z-axis is aligned with the 
      Body Z-axis (ignoring Precession/Nutation). Under this assumption, _et)this assumption, 
      inertial coordinates can be used directly.arth @ j2000_pos_vec  inertial coordinates can be used directly.
    """os_vec = iau_earth_pos_vec
    if self.j2 == 0.0:== 0.0:
      return np.zeros(3)0 if transformation fails
      pos_vec = j2000_pos_vec
    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2norm(pos_vec)
    pos_mag_pwr5 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag * pos_mag_pwr2 * pos_mag
    pos_mag_pwr5 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag
    factor = 1.5 * self.j2 * self.gp * self.pos_ref**2 / pos_mag_pwr5r5
    factor = 1.5 * self.j2 * self.gp * self.pos_ref**2 / pos_mag_pwr5
    acc_vec    = np.zeros(3)
    acc_vec[0] = factor * pos_vec[0] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 1)1)
    acc_vec[1] = factor * pos_vec[1] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 1)ec[1] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 1)
    acc_vec[2] = factor * pos_vec[2] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 3)actor * pos_vec[2] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 3)
    acc_vec[2] = factor * pos_vec[2] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 3)
    return acc_vec
    # Transform acceleration back to J2000 if we used body-fixed
  def tesseral_22(
    self,'rot_mat_j2000_to_iau_earth' in locals():
    time_et       : float,j2000_to_iau_earth.T @ acc_vecloat,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute C22 and S22 tesseral harmonic perturbation acceleration. acceleration.
    
    Input:eral_22(
    ------
      time_et : floatloat,
        Current Ephemeris Time (ET) [s] Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in inertial frame (J2000).
    Compute C22 and S22 tesseral harmonic perturbation acceleration.    
    Output:
    ---------
      acc_vec : np.ndarray
        Acceleration vector [m/s²] in Inertial frame.
    """ Current Ephemeris Time (ET) [s]
    if self.c22 == 0.0 and self.s22 == 0.0: == 0.0:
      return np.zeros(3)[m] in inertial frame (J2000).      return np.zeros(3)
    
    # Get rotation matrix from J2000 to Body-Fixed (IAU_EARTH)
    try:---
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)_to_iau_earth(time_et)
    except Exception:vector [m/s²] in Inertial frame.
      # Fallback or return zero if kernels not loaded/available
      return np.zeros(3)nd self.s22 == 0.0:
      return np.zeros(3)
    # Rotate position to Body-Fixed frame
    iau_earth_pos_vec   = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    pos_x, pos_y, pos_z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    pos_mag_pwr2 = pos_x**2 + pos_y**2 + pos_z**2**2
    pos_mag      = np.sqrt(pos_mag_pwr2)ls not loaded/available
    pos_mag_pwr7 = pos_mag_pwr2**3 * pos_mag
    
    # Pre-compute terms for efficiencyame
    term_common = self.c22 * (pos_x**2 - pos_y**2) + 2.0 * self.s22 * pos_x * pos_y
    factor      = 3.0 * self.gp * self.pos_ref**2 / pos_mag_pwr7c[1], iau_earth_pos_vec[2]wr7
    
    # Acceleration pos_x**2 + pos_y**2 + pos_z**2# Acceleration
    #   potential -> acc_vec = gradient(potential) 
    #     U22 = (3 * gp * earth_radius^2 / pos_mag^5) * (C22*(pos_x^2 - pos_y^2) + 2*S22*pos_x*pos_y)
    #     acc_vec = d/dpos_vec(U22)
    iau_earth_acc_x   = factor * (-5.0 * pos_x * term_common + pos_mag_pwr2 * ( 2.0 * self.c22 * pos_x + 2.0 * self.s22 * pos_y))
    iau_earth_acc_y   = factor * (-5.0 * pos_y * term_common + pos_mag_pwr2 * (-2.0 * self.c22 * pos_y + 2.0 * self.s22 * pos_x))_pwr2 * (-2.0 * self.c22 * pos_y + 2.0 * self.s22 * pos_x))
    iau_earth_acc_z   = factor * (-5.0 * pos_z * term_common)wr7    iau_earth_acc_z   = factor * (-5.0 * pos_z * term_common)
    iau_earth_acc_vec = np.array([iau_earth_acc_x, iau_earth_acc_y, iau_earth_acc_z])
    # Acceleration
    # Rotate acceleration back to inertial framel) ation back to inertial frame
    j2000_acc_vec = rot_mat_j2000_to_iau_earth.T @ iau_earth_acc_vec2 - pos_y^2) + 2*S22*pos_x*pos_y)at_j2000_to_iau_earth.T @ iau_earth_acc_vec
    #     acc_vec = d/dpos_vec(U22)
    # Return acceleration vector in inertial frameerm_common + pos_mag_pwr2 * ( 2.0 * self.c22 * pos_x + 2.0 * self.s22 * pos_y))eturn acceleration vector in inertial frame
    return j2000_acc_vecfactor * (-5.0 * pos_y * term_common + pos_mag_pwr2 * (-2.0 * self.c22 * pos_y + 2.0 * self.s22 * pos_x))
    iau_earth_acc_z   = factor * (-5.0 * pos_z * term_common)
  def oblate_j3(c_vec = np.array([iau_earth_acc_x, iau_earth_acc_y, iau_earth_acc_z])te_j3(
    self,
    time    : float,ation back to inertial framet,
    pos_vec : np.ndarray,at_j2000_to_iau_earth.T @ iau_earth_acc_vec,
  ) -> np.ndarray:
    """eturn acceleration vector in inertial frame
    Compute J3 oblateness perturbation acceleration.bation acceleration.
    
    Input:te_j3(
    ------
      time : floatt,
        Current time [s],        Current time [s]
      pos_vec : np.ndarray.ndarray
        Position vector [m] in Inertial frame (J2000).
    Compute J3 oblateness perturbation acceleration.
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²]ector [m/s²]
        Current time [s]
    Notes:vec : np.ndarray
    ------sition vector [m] in Inertial frame (J2000).
      Technically, zonal harmonics are defined in the Body-Fixed frame.
      However, Zonal harmonics (J2, J3...) are rotationally symmetric about the 
      Z-axis (longitude independent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).f the Pole (Z-axis).
        Acceleration vector [m/s²]
      This implementation assumes the Inertial Z-axis is aligned with the 
      Body Z-axis (ignoring Precession/Nutation). Under this assumption, 
      inertial coordinates can be used directly.
    """echnically, zonal harmonics are defined in the Body-Fixed frame.
    if self.j3 == 0.0:armonics (J2, J3...) are rotationally symmetric about the if self.j3 == 0.0:
        return np.zeros(3)dependent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]
      This implementation assumes the Inertial Z-axis is aligned with the 
    pos_mag      = np.linalg.norm(pos_vec)ation). Under this assumption, 
    pos_mag_pwr2 = pos_mag**2n be used directly.
    pos_mag_pwr7 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag_pwr2 * pos_magmag
    if self.j3 == 0.0:
    factor = 2.5 * self.j3 * self.gp * self.pos_ref**3 / pos_mag_pwr7os_ref**3 / pos_mag_pwr7
      
    acc_vec    = np.zeros(3)s_vec[0], pos_vec[1], pos_vec[2]ec    = np.zeros(3)
    acc_vec[0] = factor * pos_x * pos_z * (3.0 - 7.0 * pos_z**2 / pos_mag_pwr2)
    acc_vec[1] = factor * pos_y * pos_z * (3.0 - 7.0 * pos_z**2 / pos_mag_pwr2)2 / pos_mag_pwr2)
    acc_vec[2] = factor * (3.0 * pos_z**2 - 7.0 * pos_z**4 / pos_mag_pwr2 - 0.6 * pos_mag_pwr2)s_mag_pwr2)
    pos_mag_pwr7 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag_pwr2 * pos_mag
    return acc_vec
    factor = 2.5 * self.j3 * self.gp * self.pos_ref**3 / pos_mag_pwr7
  def oblate_j4(
    self,ec    = np.zeros(3)
    time    : float,tor * pos_x * pos_z * (3.0 - 7.0 * pos_z**2 / pos_mag_pwr2)t,
    pos_vec : np.ndarray, pos_y * pos_z * (3.0 - 7.0 * pos_z**2 / pos_mag_pwr2),
  ) -> np.ndarray:actor * (3.0 * pos_z**2 - 7.0 * pos_z**4 / pos_mag_pwr2 - 0.6 * pos_mag_pwr2)
    """
    J4 oblateness perturbations perturbation
    
    Input:te_j4(
    ------
      time : floatt,
        Current time [s],        Current time [s]
      pos_vec : np.ndarray.ndarray
        Position vector [m] in Inertial frame (J2000).
    J4 oblateness perturbation
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²]ector [m/s²]
        Current time [s]
    Notes:vec : np.ndarray
    ------sition vector [m] in Inertial frame (J2000).
      Technically, zonal harmonics are defined in the Body-Fixed frame.
      However, Zonal harmonics (J2, J3...) are rotationally symmetric about the 
      Z-axis (longitude independent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis). of the Pole (Z-axis).
        Acceleration vector [m/s²]
      This implementation assumes the Inertial Z-axis is aligned with the 
      Body Z-axis (ignoring Precession/Nutation). Under this assumption, 
      inertial coordinates can be used directly.
    """echnically, zonal harmonics are defined in the Body-Fixed frame.
    if self.j4 == 0.0:armonics (J2, J3...) are rotationally symmetric about the if self.j4 == 0.0:
      return np.zeros(3)independent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]2]
      This implementation assumes the Inertial Z-axis is aligned with the 
    pos_mag      = np.linalg.norm(pos_vec)ation). Under this assumption, 
    pos_mag_pwr2 = pos_mag**2n be used directly.
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag
    if self.j4 == 0.0:
    term_common = pos_z**2 / pos_mag_pwr2g_pwr2
    factor      = 1.875 * self.j4 * self.gp * self.pos_ref**4 / pos_mag_pwr9
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]    
    acc_vec    = np.zeros(3)
    acc_vec[0] = factor * pos_x * (1.0 - 14.0 * term_common + 21.0 * term_common**2)erm_common + 21.0 * term_common**2)
    acc_vec[1] = factor * pos_y * (1.0 - 14.0 * term_common + 21.0 * term_common**2)
    acc_vec[2] = factor * pos_z * (5.0 - 70.0 * term_common / 3.0 + 21.0 * term_common**2)* term_common**2)
    
    return acc_vecpos_z**2 / pos_mag_pwr2return acc_vec
    factor      = 1.875 * self.j4 * self.gp * self.pos_ref**4 / pos_mag_pwr9
  def oblate_j5(
    self,ec    = np.zeros(3)
    time    : float,tor * pos_x * (1.0 - 14.0 * term_common + 21.0 * term_common**2)me    : float,
    pos_vec : np.ndarray, pos_y * (1.0 - 14.0 * term_common + 21.0 * term_common**2)ec : np.ndarray,
  ) -> np.ndarray:actor * pos_z * (5.0 - 70.0 * term_common / 3.0 + 21.0 * term_common**2)
    """
    J5 oblateness perturbationbation
    
    Input:
    ------dBodyGravity:
      time : float
        Current time [s]onal perturbations from Sun, Moon, and other bodies.nt time [s]
      pos_vec : np.ndarray. : np.ndarray
        Position vector [m] in Inertial frame (J2000).
    
    Output:nit__(
    --------------
      acc_vec : np.ndarray,
        Acceleration vector [m/s²]
      """
    Notes:ialize third-body gravity model.
    ------
      Technically, zonal harmonics are defined in the Body-Fixed frame.
      However, Zonal harmonics (J2, J3...) are rotationally symmetric about the he 
      Z-axis (longitude independent). Therefore, the Earth's daily rotation (spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).ect the force, only the orientation of the Pole (Z-axis).
              
      This implementation assumes the Inertial Z-axis is aligned with the  
      Body Z-axis (ignoring Precession/Nutation). Under this assumption, 
      inertial coordinates can be used directly.
    """""
    if self.j5 == 0.0:dies if bodies else ['sun', 'moon']
      return np.zeros(3)
    def _get_position_body_spice(
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]ec[2]
    pos_mag = np.linalg.norm(pos_vec)(pos_vec)
    pos_mag_pwr2 = pos_mag**2 pos_mag**2
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag_mag
    ) -> np.ndarray:
    factor = 1.875 * self.j5 * self.gp * self.pos_ref**5 / pos_mag_pwr9
    z_term = pos_z / pos_magial body at given time using SPICE.
      
    acc_vec = np.zeros(3)
    acc_vec[0] = factor * pos_x * z_term * (3.0 - 30.0 * z_term**2 + 35.0 * z_term**4)
    acc_vec[1] = factor * pos_y * z_term * (3.0 - 30.0 * z_term**2 + 35.0 * z_term**4)
    acc_vec[2] = factor * (3.0 * z_term**2 - 30.0 * z_term**4 + 35.0 * z_term**6 - 0.6 * pos_mag_pwr2 / pos_mag_pwr2)os_mag_pwr2)
        time_et : float
    return acc_vecs time in seconds past J2000 epoch.urn acc_vec
        frame : str
  def oblate_j6(nce frame (default: 'J2000').
    self,f,
    time    : float,
    pos_vec : np.ndarray,y,
  ) -> np.ndarray:np.ndarray
    """   Position vector [m].
    J6 oblateness perturbation
      # SPICE state relative to Earth
    Input:e, _ = spice.spkez(put:
    ------targ   = self._get_naif_id(body_name),
      time : float time_et, float
        Current time [s],]
      pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).vector [m] in Inertial frame (J2000).
      )
    Output:CE returns km, convert to m
    -------n np.array(state[0:3]) * CONVERTER.M_PER_KM
      acc_vec : np.ndarray
        Acceleration vector [m/s²]ector [m/s²]
      self,
    Notes:_name : str,
    ------nt:
      Technically, zonal harmonics are defined in the Body-Fixed frame.me.
      However, Zonal harmonics (J2, J3...) are rotationally symmetric about the 
      Z-axis (longitude independent). Therefore, the Earth's daily rotation (spin) spin) 
      does not affect the force, only the orientation of the Pole (Z-axis).
      ------
      This implementation assumes the Inertial Z-axis is aligned with the 
      Body Z-axis (ignoring Precession/Nutation). Under this assumption,  assumption, 
      inertial coordinates can be used directly.
    """utput:
    if self.j6 == 0.0:
      return np.zeros(3)urn np.zeros(3)
          NAIF ID code.
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]
    pos_mag = np.linalg.norm(pos_vec).norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2S.NAME_TO_ID:
    pos_mag_pwr11 = pos_mag_pwr2**5 * pos_mag5 * pos_mag
      
    factor = 2.1875 * self.j6 * self.gp * self.pos_ref**6 / pos_mag_pwr11}")ctor = 2.1875 * self.j6 * self.gp * self.pos_ref**6 / pos_mag_pwr11
    z_term = pos_z**2 / pos_mag_pwr2
    def point_mass(
    acc_vec = np.zeros(3)
    acc_vec[0] = factor * pos_x * (1.0 - 21.0 * z_term + 63.0 * z_term**2 - 63.0 * z_term**3)
    acc_vec[1] = factor * pos_y * (1.0 - 21.0 * z_term + 63.0 * z_term**2 - 63.0 * z_term**3)*2 - 63.0 * z_term**3)
    acc_vec[2] = factor * pos_z * (7.0 - 63.0 * z_term + 126.0 * z_term**2 - 63.0 * z_term**3)
      """
    return acc_vecd-body point mass perturbations (Sun, Moon, etc.).turn acc_vec
      
      Input:
class ThirdBodyGravity:
    """   time : float
    Third-body gravitational perturbations from Sun, Moon, and other bodies.ions from Sun, Moon, and other bodies.
    Uses SPICE ephemerides.ndarray
    """       Satellite position vector [m].
      
    def __init__(def __init__(
      self,--
      bodies : list = None,ray
    ):        Third-body acceleration [m/s²].
      """
      Initialize third-body gravity model.00 epochhird-body gravity model.
      et_seconds = time      
      Input:
      ------ute acceleration for all bodies
        bodies : listros(3)
            Which bodies to include (default: ['sun', 'moon']).ult: ['sun', 'moon']).
              pper = body.upper()
      Output:
      -------p Earth if it's in the list (it's the central body)
        Noneody_upper == 'EARTH':        None
      """ continue
      self.bodies = bodies if bodies else ['sun', 'moon']
        # Get gravitational parameter [m³/s²]
    def _get_position_body_spice(STANTS, body_upper):    def _get_position_body_spice(
      self,P = getattr(SOLARSYSTEMCONSTANTS, body_upper).GP
      body_name : str,
      time_et   : float,
      frame     : str = 'J2000',
    ) -> np.ndarray:f central body (Earth) to perturbing body [m]np.ndarray:
      """os_centbody_to_pertbody_vec = self._get_position_body_spice(body, et_seconds)      """
      Get position of celestial body at given time using SPICE.ody_to_pertbody_vec)f celestial body at given time using SPICE.
          
      Input:fety check      Input:
      ------os_centbody_to_pertbody_mag == 0:
        body_name : str_name : str
          Body name ('SUN' or 'MOON').
        time_et : floatatellite to perturbing body [m]    time_et : float
          Ephemeris time in seconds past J2000 epoch.tbody_vec - pos_sat_vec in seconds past J2000 epoch.
        frame : strpertbody_mag = np.linalg.norm(pos_sat_to_pertbody_vec)
          Reference frame (default: 'J2000').
        # Third-body acceleration contribution [m/s²]
      Output:ec += (  Output:
      -------P * pos_sat_to_pertbody_vec / pos_sat_to_pertbody_mag**3
        pos_vec : np.ndarrayody_to_pertbody_vec / pos_centbody_to_pertbody_mag**3_vec : np.ndarray
          Position vector [m].
      """
      # SPICE state relative to Earth
      state, _ = spice.spkez(
          targ   = self._get_naif_id(body_name),
          et     = time_et,
          ref    = frame,
          abcorr = 'NONE',tion coordinator.
          obs    = 399  # relative to Earth
      )putes gravity as:)
      # SPICE returns km, convert to m+ two_body_oblate + third_body_point_mass + PICE returns km, convert to m
      return np.array(state[0:3]) * CONVERTER.M_PER_KMity (future)ER_KM
    """
    def _get_naif_id(
      self,nit__(
      body_name : str,
    ) -> int:                 : float,
      """                     : float = 0.0,
      Get NAIF ID for body.   : float = 0.0,
      j4                      : float = 0.0,
      Input:                  : float = 0.0,
      ------                  : float = 0.0,
        body_name : str       : float = 0.0,
          Body name (e.g. 'SUN', 'JUPITER').e, (e.g. 'SUN', 'JUPITER').
      third_body_bodies       : list  = None,
      Output:
      -------
        naif_id : intity acceleration components.
          NAIF ID code.
      """ut:
      body_upper = body_name.upper()
      if body_upper in NAIFIDS.NAME_TO_ID:
        return NAIFIDS.NAME_TO_ID[body_upper]body [m³/s²].
        j2 : float
      raise ValueError(f"Unknown body name for NAIF ID lookup: {body_name}")me for NAIF ID lookup: {body_name}")
        j3 : float
    def point_mass(ic coefficient for oblateness.t_mass(
      self,: floatf,
      time        : float,ficient for oblateness.t,
      pos_sat_vec : np.ndarray,
    ) -> np.ndarray:al harmonic coefficient.
      """22 : float
      Compute third-body point mass perturbations (Sun, Moon, etc.).ions (Sun, Moon, etc.).
        pos_ref : float
      Input:ference radius for harmonic coefficients [m].
      ------le_third_body : bool
          time : floatoon gravitational perturbations.
              Current Ephemeris Time (ET) [s]. Ephemeris Time (ET) [s].
          pos_sat_vec : np.ndarray(default: ['sun', 'moon']).  pos_sat_vec : np.ndarray
              Satellite position vector [m].
      Output:
      Output:
      -------
          acc_vec : np.ndarray
              Third-body acceleration [m/s²].eration [m/s²].
      """f.two_body = TwoBodyGravity(
      # Ephemeris time is seconds from J2000 epoch
      et_seconds = timeet_seconds = time
        j3      = j3,
      # Compute acceleration for all bodiesall bodies
      acc_vec = np.zeros(3)
      for body in self.bodies:
        body_upper = body.upper()pper()
        
        # Skip Earth if it's in the list (it's the central body)
        if body_upper == 'EARTH':dy_upper == 'EARTH':
          continuethird_body = enable_third_bodyntinue
      if self.enable_third_body:
        # Get gravitational parameter [m³/s²]ional parameter [m³/s²]
        if hasattr(SOLARSYSTEMCONSTANTS, body_upper):
          GP = getattr(SOLARSYSTEMCONSTANTS, body_upper).GP
        else:
          continue_body = None    continue
    
        # Position of central body (Earth) to perturbing body [m][m]
        pos_centbody_to_pertbody_vec = self._get_position_body_spice(body, et_seconds)
        pos_centbody_to_pertbody_mag = np.linalg.norm(pos_centbody_to_pertbody_vec)
        s_vec : np.ndarray,
        # Safety check
        if pos_centbody_to_pertbody_mag == 0:
          continuel gravity acceleration.    continue
      
        # Position of satellite to perturbing body [m]
        pos_sat_to_pertbody_vec = pos_centbody_to_pertbody_vec - pos_sat_vecat_vec
        pos_sat_to_pertbody_mag = np.linalg.norm(pos_sat_to_pertbody_vec)
          Current Ephemeris Time (ET) [s].
        # Third-body acceleration contribution [m/s²]
        acc_vec += (ector [m].
            GP * pos_sat_to_pertbody_vec / pos_sat_to_pertbody_mag**3
            - GP * pos_centbody_to_pertbody_vec / pos_centbody_to_pertbody_mag**3
        )----
        acc_vec : np.ndarray
      return acc_vecity acceleration [m/s²].
      """
      # Initialize acceleration vector
class Gravity:= np.zeros(3)Gravity:
    """
    Gravitational acceleration coordinator.leration coordinator.
      acc_vec += self.two_body.point_mass(pos_vec)
    Computes gravity as:s:
        gravity = two_body_point_mass + two_body_oblate + third_body_point_mass + o_body_oblate + third_body_point_mass + 
                  third_body_oblate (future) + relativity (future)
    """cc_vec += self.two_body.oblate_j3(time, pos_vec)
      acc_vec += self.two_body.oblate_j4(time, pos_vec)
    def __init__(nit__(
      self,-body tesseral (C22, S22)
      gp                      : float,l_22(time, pos_vec)p                      : float,
      j2                      : float = 0.0,
      j3                      : float = 0.0,           : float = 0.0,
      j4                      : float = 0.0,rd_body is not None:  j4                      : float = 0.0,
      j5                      : float = 0.0,s(time, pos_vec)             : float = 0.0,
      j6                      : float = 0.0,
      c22                     : float = 0.0,y     : float = 0.0,
      s22                     : float = 0.0,
      pos_ref                 : float = 0.0,
      enable_third_body       : bool  = False,
      third_body_bodies       : list  = None,
    ):=========================================================================
      """vitational Accelerations"""
      Initialize gravity acceleration components.==============================lize gravity acceleration components.
      
      Input:hericDrag:
      ------
        gp : floatag acceleration using exponential atmosphere model
          Gravitational parameter of central body [m³/s²].
        j2 : float
          J2 harmonic coefficient for oblateness.
        j3 : float
          J3 harmonic coefficient for oblateness. oblateness.
        j4 : float = 0.0,float
          J4 harmonic coefficient for oblateness.r oblateness.
        j5 : float
          J5 harmonic coefficient for oblateness.
        j6 : floatrag model
          J6 harmonic coefficient for oblateness.
        c22 : float : float
          C22 tesseral harmonic coefficient.
        s22 : float: float
          S22 tesseral harmonic coefficient.
        pos_ref : float
          Reference radius for harmonic coefficients [m].oefficients [m].
        enable_third_body : boolbody : bool
          Enable Sun/Moon gravitational perturbations.
        third_body_bodies : lists : list
          Which bodies to include (default: ['sun', 'moon'])..
               
      Output:
      -------
        Noned   = cd
      """f.area = area
      # Two-body gravityTwo-body gravity
      self.two_body = TwoBodyGravity(
        gp      = gp,gp,
        acc_vec : np.ndarray
          Drag acceleration [m/s²]
      """_vec : np.ndarray,4      = j4,
      # Position magnitude and altitude
      pos_mag = np.linalg.norm(pos_vec)
      alt     = pos_mag - SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
        s22     = s22,
      # Atmospheric density at current altitude
      rho = self._atmospheric_density(float(alt))
        pos_vec : np.ndarray  
      # Velocity relative to rotating atmosphere
      omega_earth = np.array([0, 0, SOLARSYSTEMCONSTANTS.EARTH.OMEGA])
      vel_rel_vec = vel_vec - np.cross(omega_earth, pos_vec)
      vel_rel_mag = np.linalg.norm(vel_rel_vec)
      Output:    bodies = third_body_bodies,
      if vel_rel_mag == 0:
        return np.zeros(3)ay
          Drag acceleration [m/s²]  self.third_body = None
      # Drag acceleration
      acc_drag_mag = 0.5 * rho * (self.cd * self.area / self.mass) * vel_rel_mag**2
      acc_drag_dir = -vel_rel_vec / vel_rel_mag
      alt     = pos_mag - SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATORtime    : float,
      return acc_drag_mag * acc_drag_dir
      # Atmospheric density at current altitude) -> np.ndarray:
    def _atmospheric_density(_density(float(alt))
      self,tal gravity acceleration.
      altitude : float,ve to rotating atmosphere
    ) -> float:th = np.array([0, 0, SOLARSYSTEMCONSTANTS.EARTH.OMEGA])
      """_rel_vec = vel_vec - np.cross(omega_earth, pos_vec)---
      Simplified exponential atmospheric density model
          Current Ephemeris Time (ET) [s].
      Input:_rel_mag == 0:vec : np.ndarray
      ------rn np.zeros(3)sition vector [m].
        altitude : float
          Altitude above Earth's surface [m]
      acc_drag_mag = 0.5 * rho * (self.cd * self.area / self.mass) * vel_rel_mag**2-------
      Output:g_dir = -vel_rel_vec / vel_rel_magec : np.ndarray
      -------ity acceleration [m/s²].
        density : floatag * acc_drag_dir
          Atmospheric density [kg/m³]
      """atmospheric_density(_vec = np.zeros(3)
      if altitude < 0:
        altitude = 0at,nt mass
      -> float:acc_vec += self.two_body.point_mass(pos_vec)
      # Simplified exponential model
      rho = SOLARSYSTEMCONSTANTS.EARTH.RHO_0 * np.exp(-altitude / SOLARSYSTEMCONSTANTS.EARTH.H_0)
      acc_vec += self.two_body.oblate_j2(time, pos_vec)
      return rhof.two_body.oblate_j3(time, pos_vec)
      ------      acc_vec += self.two_body.oblate_j4(time, pos_vec)
        altitude : float      acc_vec += self.two_body.oblate_j5(time, pos_vec)
class SolarRadiationPressure:h's surface [m]y.oblate_j6(time, pos_vec)
    """
    Solar radiation pressure acceleration model.
      -------  acc_vec += self.two_body.tesseral_22(time, pos_vec)
    Computes the acceleration due to solar radiation pressure on a spacecraft,
    accounting for the spacecraft's reflectivity, cross-sectional area, and mass.
    Includes cylindrical Earth shadow model.
    """f altitude < 0: acc_vec += self.third_body.point_mass(time, pos_vec)
        altitude = 0  
    def __init__(blate, relativity
      self,plified exponential model
      cr   : float = 1.3,NSTANTS.EARTH.RHO_0 * np.exp(-altitude / SOLARSYSTEMCONSTANTS.EARTH.H_0)
      area : float = 0.0,
      mass : float = 1.0,
    ):===================================================================
      """l Accelerations
      Initialize SRP modelre:=====================================================
      "
      Input:diation pressure acceleration model.hericDrag:
      ------
        cr : floatcceleration due to solar radiation pressure on a spacecraft,ag acceleration using exponential atmosphere model
          Radiation pressure coefficient (1.0 = absorbing, 2.0 = reflecting)mass.
        area : floatical Earth shadow model.
          Cross-sectional area [m²]
        mass : float
          Spacecraft mass [kg]
              t = 0.0,
      Output:float = 1.3,float = 1.0,
      -------float = 0.0,
        None float = 1.0,
      """lize drag model
      self.cr   = cr
      self.area = areaodel
      self.mass = mass
      Input:    cd : float
    def compute(icient.
      self,: floata : float
      time                 : float,cient (1.0 = absorbing, 2.0 = reflecting).
      earth_to_sat_pos_vec : np.ndarray,
    ) -> np.ndarray:ional area [m²] mass [kg].
      """ass : float     
      Compute SRP acceleration.
              -------
      Input::
      -------
        time : float
          Current Ephemeris Time (ET) [s]
        earth_to_sat_pos_vec : np.ndarray
          Spacecraft position vector relative to Earth [m], i.e. Earth to spacecraft vector.
      self.mass = massf compute(
      Output:
      -------te( : np.ndarray,
        earth_to_sat_acc_vec : np.ndarray
          SRP acceleration [m/s²]t,
      """th_to_sat_pos_vec : np.ndarray,
      # Check for valid parameters
      if self.area <= 0 or self.mass <= 0:
        return np.zeros(3)tion.
      ------
      # Get Sun position relative to Earth using SPICE
      earth_to_sun_pos_vec = self._get_sun_position(time)
        time : float  vel_vec : np.ndarray
      # Vector from spacecraft to Sun [s]
      sat_to_sun_pos_vec = earth_to_sun_pos_vec - earth_to_sat_pos_vec
      sat_to_sun_pos_mag = np.linalg.norm(sat_to_sun_pos_vec).e. Earth to spacecraft vector.
      sat_to_sun_pos_dir = sat_to_sun_pos_vec / sat_to_sun_pos_mag
      Output:        acc_vec : np.ndarray
      # Direction of solar radiation pressure force is from Sun to spacecraft
      acc_dir = -sat_to_sun_pos_dirdarray
          SRP acceleration [m/s²]# Position magnitude and altitude
      # Compute shadow factor (0.0 = full shadow, 1.0 = full sunlight)
      shadow_factor = self._compute_shadow_factor(earth_to_sat_pos_vec, earth_to_sun_pos_vec)
      if self.area <= 0 or self.mass <= 0:
      # If in full shadow, no SRP acceleration
      if shadow_factor == 0.0:
        return np.zeros(3)elative to Earth using SPICE
      earth_to_sun_pos_vec = self._get_sun_position(time)# Velocity relative to rotating atmosphere
      # Distance from Sun to spacecraft [m]
      sun_to_sat_pos_mag = sat_to_sun_pos_magos_vec)
      sat_to_sun_pos_vec = earth_to_sun_pos_vec - earth_to_sat_pos_vecvel_rel_mag = np.linalg.norm(vel_rel_vec)
      # Solar radiation pressure at spacecraft distances_vec)
      #   P = P_at_1au * ( 1_au / r_au )^2 = 4.56e-6 N/m² * ( 149597870700 m / r_m )^2
      pressure_srp  = SOLARSYSTEMCONSTANTS.EARTH.PRESSURE_SRP * (CONVERTER.M_PER_AU * CONVERTER.ONE_AU / sun_to_sat_pos_mag)**2
      # Direction of solar radiation pressure force is from Sun to spacecraft
      # SRP acceleration magnituder
      acc_mag = (pressure_srp * self.cr * self.area / self.mass) * shadow_factor
      # Compute shadow factor (0.0 = full shadow, 1.0 = full sunlight)acc_drag_dir = -vel_rel_vec / vel_rel_mag
      # SRP acceleration direction (away from Sun)earth_to_sat_pos_vec, earth_to_sun_pos_vec)
      acc_vec = acc_mag * acc_dir
      # If in full shadow, no SRP acceleration
      return acc_vecor == 0.0:_density(
        return np.zeros(3)  self,
    def _get_sun_position(
      self,tance from Sun to spacecraft [m]oat:
      time_et : float,ag = sat_to_sun_pos_mag
    ) -> np.ndarray:pheric density model
      """olar radiation pressure at spacecraft distance
      Get Sun position relative to Earth at given time using SPICE.7870700 m / r_m )^2
      pressure_srp  = SOLARSYSTEMCONSTANTS.EARTH.PRESSURE_SRP * (CONVERTER.M_PER_AU * CONVERTER.ONE_AU / sun_to_sat_pos_mag)**2------
      Input: float
      ------acceleration magnitudetitude above Earth's surface [m]
        time_et : floatre_srp * self.cr * self.area / self.mass) * shadow_factor
          Ephemeris time in seconds past J2000 epoch.
      # SRP acceleration direction (away from Sun)-------
      Output: = acc_mag * acc_dirty : float
      -------c density [kg/m³]
        sun_pos_vec : np.ndarray
          Sun position vector relative to Earth [m].
      """get_sun_position(ltitude = 0
      # Get Sun position relative to Earth
      state, _ = spice.spkez(
        targ   = 10,       # Sun NAIF IDude / SOLARSYSTEMCONSTANTS.EARTH.H_0)
        et     = time_et,
        ref    = 'J2000',lative to Earth at given time using SPICE.
        abcorr = 'NONE',
        obs    = 399       # Earth NAIF ID
      )-----olarRadiationPressure:
        time_et : float"
      # SPICE returns km, convert to mst J2000 epoch.ion model.
      return np.array(state[0:3]) * CONVERTER.M_PER_KM
      Output:Computes the acceleration due to solar radiation pressure on a spacecraft,
    def _compute_shadow_factor( cross-sectional area, and mass.
      self,_pos_vec : np.ndarrays cylindrical Earth shadow model.
      earth_to_sat_pos_vec : np.ndarray,o Earth [m].
      earth_to_sun_pos_vec : np.ndarray,
    ) -> float: position relative to Earth_(
      """te, _ = spice.spkez(f,
      Compute shadow factor using cylindrical Earth shadow model.
        et     = time_et,area : float = 0.0,
      Input:   = 'J2000', float = 1.0,
      ------rr = 'NONE',
        earth_to_sat_pos_vec : np.ndarrayD
          Spacecraft position vector relative to Earth [m].
        earth_to_sun_pos_vec : np.ndarray
          Sun position vector relative to Earth [m].
      return np.array(state[0:3]) * CONVERTER.M_PER_KM------
      Output:
      -------ute_shadow_factor(iation pressure coefficient (1.0 = absorbing, 2.0 = reflecting)
        shadow_factor : float
          0.0 = full shadow (umbra), 1.0 = full sunlight.
      earth_to_sun_pos_vec : np.ndarray,  mass : float
      Notes:at:acecraft mass [kg]
      ------
        Uses a simplified cylindrical shadow model where the shadow is a
        cylinder with radius equal to Earth's equatorial radius, extending
        from Earth in the anti-Sun direction.
      """---
      # Direction from Earth to Sundarray
      earth_to_sun_pos_dir = earth_to_sun_pos_vec / np.linalg.norm(earth_to_sun_pos_vec)
        earth_to_sun_pos_vec : np.ndarrayself.mass = mass
      # Project spacecraft position onto Sun direction
      # (distance along Sun-Earth line, positive toward Sun)
      earth_to_sat_parallel_pos_mag = np.dot(earth_to_sat_pos_vec, earth_to_sun_pos_dir)
      -------time                 : float,
      # Check if spacecraft is in front of or behind Earth relative to Sun
      if earth_to_sat_parallel_pos_mag >= 0:ull sunlight.
        # Spacecraft is on the sunlit side of Earth if positive projection
        shadow_factor = 1.0
      else:-
        # Spacecraft is on the shadow side of Earth if negative projection
        cylinder with radius equal to Earth's equatorial radius, extending      ------
        # Calculate perpendicular distance from spacecraft to Sun-Earth line
        earth_to_sat_perpendicular_pos_vec = earth_to_sat_pos_vec - earth_to_sat_parallel_pos_mag * earth_to_sun_pos_dir
        earth_to_sat_perpendicular_pos_mag = np.linalg.norm(earth_to_sat_perpendicular_pos_vec)
        rth_to_sun_pos_dir = earth_to_sun_pos_vec / np.linalg.norm(earth_to_sun_pos_vec)  Spacecraft position vector relative to Earth [m], i.e. Earth to spacecraft vector.
        # Check if within Earth's shadow cylinder
        if earth_to_sat_perpendicular_pos_mag > SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR:
          # If perpendicular distance is greater than Earth radius, spacecraft is in sunlight
          shadow_factor = 1.0os_mag = np.dot(earth_to_sat_pos_vec, earth_to_sun_pos_dir): np.ndarray
        else:ration [m/s²]
          # Spacecraft is in Earth's cylindrical shadowrth relative to Sun
          shadow_factor = 0.0l_pos_mag >= 0:eters
        # Spacecraft is on the sunlit side of Earth if positive projection self.area <= 0 or self.mass <= 0:
      # Return shadow factor
      return shadow_factor
            # Spacecraft is on the shadow side of Earth if negative projection      # Get Sun position relative to Earth using SPICE
    def _compute_shadow_factor_conical(      earth_to_sun_pos_vec = self._get_sun_position(time)
      self,
      earth_to_sat_pos_vec : np.ndarray,rpendicular_pos_vec = earth_to_sat_pos_vec - earth_to_sat_parallel_pos_mag * earth_to_sun_pos_dircecraft to Sun
      earth_to_sun_pos_vec : np.ndarray,dicular_pos_vec)
    ) -> float:              sat_to_sun_pos_mag = np.linalg.norm(sat_to_sun_pos_vec)
      """within Earth's shadow cylinders_dir = sat_to_sun_pos_vec / sat_to_sun_pos_mag
      Compute shadow factor using conical Earth shadow model with penumbra. if earth_to_sat_perpendicular_pos_mag > SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR:
      More accurate than cylindrical model.cecraft is in sunlightecraft
      """      shadow_factor = 1.0  acc_dir = -sat_to_sun_pos_dir
      # Sun and Earth radii
      R_sun = SOLARSYSTEMCONSTANTS.SUN.RADIUS.EQUATOR full sunlight)
      R_earth = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR      shadow_factor = 0.0  shadow_factor = self._compute_shadow_factor(earth_to_sat_pos_vec, earth_to_sun_pos_vec)
      
      # Distances
      earth_to_sun_dist = np.linalg.norm(earth_to_sun_pos_vec)
      earth_to_sat_dist = np.linalg.norm(earth_to_sat_pos_vec) np.zeros(3)
      
      # Angle between sat and anti-sun direction==============================================================from Sun to spacecraft [m]
      cos_angle = -np.dot(earth_to_sat_pos_vec, earth_to_sun_pos_vec) / (earth_to_sat_dist * earth_to_sun_dist) Coordinatoro_sat_pos_mag = sat_to_sun_pos_mag
      =========================================
      if cos_angle < 0:
        return 1.0  # Satellite on sunlit side70700 m / r_m )^2
      AU * CONVERTER.ONE_AU / sun_to_sat_pos_mag)**2
      # Angular radii of Sun and Earth as seen from satelliteall acceleration components
      alpha_sun = np.arcsin(R_sun / earth_to_sun_dist)
      alpha_earth = np.arcsin(R_earth / earth_to_sat_dist) self.mass) * shadow_factor
      on_pressure
      # Angular separation between Sun and Earth centers
      angle = np.arccos(cos_angle)
      body_point_mass + 
      # Check umbra/penumbra/sunlight relativity (future)
      if angle < alpha_earth - alpha_sun:
        return 0.0  # Umbra (total eclipse)
      elif angle < alpha_earth + alpha_sun:
        # Penumbra (partial eclipse) - approximate area fraction
        return 0.5  # Simplified; could compute actual fractiongp                      : float,-> np.ndarray:
      else:                     : float = 0.0,
        return 1.0  # Full sunlight.0,at given time using SPICE.
j4                      : float = 0.0,
                  : float = 0.0,
# =============================================================================                  : float = 0.0,
# Top-Level Coordinator            : float = 0.0,float
# =============================================================================
            : bool  = False,
class Acceleration:
    """            : float = 0.0,
    Acceleration coordinator - orchestrates all acceleration components
    odies       : list  = None,tion vector relative to Earth [m].
    Computes total acceleration as:
      total = gravity + drag + solar_radiation_pressure           : float = 0.0,ition relative to Earth
    ,
    where: NAIF ID
      gravity = two_body_point_mass + third_body_point_mass + 
                two_body_oblate (J2, J3, J4) + relativity (future)ration coordinator',
    """
     Earth NAIF ID
    def __init__(
      self,
      gp                      : float, of central body [m³/s²] to m
      j2                      : float = 0.0,ray(state[0:3]) * CONVERTER.M_PER_KM
      j3                      : float = 0.0,ficient for oblateness
      j4                      : float = 0.0,
      c22                     : float = 0.0,or oblateness
      s22                     : float = 0.0,
      pos_ref                 : float = 0.0,
      mass                    : float = 1.0,
      enable_drag             : bool  = False,
      cd                      : float = 0.0,ng cylindrical Earth shadow model.
      area_drag               : float = 0.0,
      enable_third_body       : bool  = False,float
      third_body_bodies       : list  = None,coefficients [m]
      enable_srp              : bool  = False, : np.ndarray
      cr                      : float = 0.0,m].
      area_srp                : float = 0.0,_drag : boolto_sun_pos_vec : np.ndarray
    ):ble atmospheric drag position vector relative to Earth [m].
      """float
      Initialize acceleration coordinatorag coefficient:
      rea_drag : float----
      Input:
      ------oolumbra), 1.0 = full sunlight.
        gp : floatal perturbations
          Gravitational parameter of central body [m³/s²]tr
        j2 : floatfault: ['sun', 'moon'])
          J2 harmonic coefficient for oblatenessl where the shadow is a
        j3 : floature (not yet implemented)Earth's equatorial radius, extending
          J3 harmonic coefficient for oblateness
        j4 : float
          J4 harmonic coefficient for oblateness
        c22 : float.norm(earth_to_sun_pos_vec)
          C22 tesseral harmonic coefficient       
        s22 : floatOutput:# Project spacecraft position onto Sun direction
          S22 tesseral harmonic coefficient)
        pos_ref : float
          Reference radius for harmonic coefficients [m]
        mass : floateleration component instancescecraft is in front of or behind Earth relative to Sun
          Spacecraft mass [kg]y(lel_pos_mag >= 0:
        enable_drag : bool          = gp,s on the sunlit side of Earth if positive projection
          Enable atmospheric drag2                      = j2,hadow_factor = 1.0
        cd : float                     = j3,
          Drag coefficient        = j4,on the shadow side of Earth if negative projection
        area_drag : float  c22                     = c22,
          Cross-sectional area [m²]s22,distance from spacecraft to Sun-Earth line
        enable_third_body : bool       = pos_ref,endicular_pos_vec = earth_to_sat_pos_vec - earth_to_sat_parallel_pos_mag * earth_to_sun_pos_dir
          Enable Sun/Moon gravitational perturbationshird_body, = np.linalg.norm(earth_to_sat_perpendicular_pos_vec)
        third_body_bodies : list of strodies       = third_body_bodies,
          Which bodies to include (default: ['sun', 'moon'])nder
        enable_srp : boolpos_mag > SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR:
          Enable solar radiation pressure (not yet implemented)f.enable_drag = enable_drag # If perpendicular distance is greater than Earth radius, spacecraft is in sunlight
        cr : floatlf.enable_drag and cd > 0 and area_drag > 0 and mass > 0:hadow_factor = 1.0
          Radiation pressure coefficientphericDrag(
        area_srp : float      cd   = cd,      # Spacecraft is in Earth's cylindrical shadow
          Cross-sectional area for SRP [m²] area_drag,_factor = 0.0
              ass = mass,
      Output:
      -------
        None
      """
        # Create acceleration component instancesf.enable_srp = enable_srp======================================================================
      self.gravity = Gravity(
        gp                      = gp,  self.srp = SolarRadiationPressure(=========================================================================
        j2                      = j2,   = cr,
        j3                      = j3,ea = area_srp,ration:
        j4                      = j4,s,
        c22                     = c22,
        s22                     = s22,
        pos_ref                 = pos_ref,s:
        enable_third_body       = enable_third_body,ure
        third_body_bodies       = third_body_bodies,
      )self,ere:
       : float, = two_body_point_mass + third_body_point_mass + 
      self.enable_drag = enable_drag : np.ndarray,   two_body_oblate (J2, J3, J4) + relativity (future)
      if self.enable_drag and cd > 0 and area_drag > 0 and mass > 0:
        self.drag = AtmosphericDrag(
          cd   = cd,_init__(
          area = area_drag,leration from all components
          mass = mass,
        )Input:j2                      : float = 0.0,
      else:
        self.drag = None0,
      
      self.enable_srp = enable_srp  pos_vec : np.ndarrayj6                      : float = 0.0,
      if self.enable_srp:
        self.srp = SolarRadiationPressure(float = 0.0,
          cr   = cr,
          area = area_srp,mass                    : float = 1.0,
          mass = mass,   : bool  = False,
        )      -------      cd                      : float = 0.0,
      else:        acc_vec : np.ndarray      area_drag               : float = 0.0,
          self.srp = None
    ist  = None,
    def compute(
      self,      acc_vec = self.gravity.compute(time, pos_vec)      cr                      : float = 0.0,
      time    : float,
      pos_vec : np.ndarray, # Atmospheric drag (optional):
      vel_vec : np.ndarray,
    ) -> np.ndarray:   acc_vec += self.drag.compute(pos_vec, vel_vec) Initialize acceleration coordinator
      """        
      Compute total acceleration from all componentsadiation pressure (optional)
      self.srp is not None:---
      Input:ute(time, pos_vec)
      ------        Gravitational parameter of central body [m³/s²]
        time : floateturn acc_vec j2 : float
          Current Ephemeris Time (ET) [s]
        pos_vec : np.ndarrayj3 : float
          Position vector [m]=====================================================================J3 harmonic coefficient for oblateness
        vel_vec : np.ndarrays of Motion : float
          Velocity vector [m/s]============================================== for oblateness
      
      Output:lStateEquationsOfMotion: harmonic coefficient for oblateness
      -------t
        acc_vec : np.ndarraytate equations of motion for orbit propagation6 harmonic coefficient for oblateness
          Total acceleration [m/s²]loat
      """2 tesseral harmonic coefficient
      # Gravity (always)
      acc_vec = self.gravity.compute(time, pos_vec)  self,        S22 tesseral harmonic coefficient
      ion,
      # Atmospheric drag (optional)e radius for harmonic coefficients [m]
      if self.drag is not None:
        acc_vec += self.drag.compute(pos_vec, vel_vec)otion]
      
      # Solar radiation pressure (optional)ut:   Enable atmospheric drag
      if self.srp is not None:
        acc_vec += self.srp.compute(time, pos_vec)  acceleration : Acceleration      Drag coefficient
      celeration coordinator instanceea_drag : float
      return acc_vec  Cross-sectional area [m²]
 : bool

# =============================================================================
# Equations of Motion
# =============================================================================self.acceleration = acceleration    enable_srp : bool
ar radiation pressure (not yet implemented)
class GeneralStateEquationsOfMotion:_time_derivative(: float
  """
  General state equations of motion for orbit propagation
  """tate_vec : np.ndarray,   Cross-sectional area for SRP [m²]
  
  def __init__(
    self,
    acceleration : Acceleration,    None
  ):
    """
    Initialize equations of motion
        Current Ephemeris Time (ET) [s]    gp                      = gp,
    Input:rray        = j2,
    ------        Current state vector [pos, vel] [m, m/s]        j3                      = j3,
      acceleration : Acceleration            j4                      = j4,
        Acceleration coordinator instance    Output:        j5                      = j5,
                -------        j6                      = j6,
    Output:      state_dot_vec : np.ndarray        c22                     = c22,







































    return state_dot_vec        state_dot_vec[3:6] = acc_vec    state_dot_vec[0:3] = vel_vec    state_dot_vec      = np.zeros(6)        acc_vec = self.acceleration.compute(time, pos_vec, vel_vec)    vel_vec = state_vec[3:6]    pos_vec = state_vec[0:3]    """        Time derivative of state vector [vel, acc] [m/s, m/s²]      state_dot_vec : np.ndarray    -------    Output:            Current state vector [pos, vel] [m, m/s]      state_vec : np.ndarray        Current Ephemeris Time (ET) [s]      time : float    ------    Input:        Compute state time derivative for ODE integration    """  ) -> np.ndarray:      state_vec : np.ndarray,      time      : float,      self,  def state_time_derivative(      self.acceleration = acceleration    """      None    -------















    return state_dot_vec        state_dot_vec[3:6] = acc_vec    state_dot_vec[0:3] = vel_vec    state_dot_vec      = np.zeros(6)        acc_vec = self.acceleration.compute(time, pos_vec, vel_vec)    vel_vec = state_vec[3:6]    pos_vec = state_vec[0:3]    """        Time derivative of state vector [vel, acc] [m/s, m/s²]        s22                     = s22,
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




