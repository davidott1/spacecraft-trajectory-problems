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
    └── AccelerationSTMDot (coordinator)
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
  from src.model.dynamics import AccelerationSTMDot, GeneralStateEquationsOfMotion
  from src.model.constants import SOLARSYSTEMCONSTANTS
  from src.schemas.spacecraft import SpacecraftProperties, DragConfig
  
  # Create spacecraft with drag enabled
  spacecraft = SpacecraftProperties(
      mass = 1000.0,
      drag = DragConfig(enabled=True, cd=2.2, area=10.0),
  )
  
  # Initialize acceleration model
  acceleration = AccelerationSTMDot(
      gp                = SOLARSYSTEMCONSTANTS.EARTH.GP,
      spacecraft        = spacecraft,
      j2                = SOLARSYSTEMCONSTANTS.EARTH.J2,
      pos_ref           = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
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

from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER, NAIFIDS, PHYSICALCONSTANTS
from src.model.frame_converter import FrameConverter
from src.schemas.spacecraft    import SpacecraftProperties, DragConfig, SRPConfig
from src.schemas.gravity       import GravityModelConfig


# =============================================================================
# Gravity Components (bottom of hierarchy)
# =============================================================================

class TwoBodyGravity:
  """
  Two-body gravitational acceleration components
  Handles point mass and oblateness (J2, J3) perturbations
  """
  
  def __init__(
    self,
    gp      : float,
    j2      : float = 0.0,
    j3      : float = 0.0,
    c21     : float = 0.0,
    s21     : float = 0.0,
    c22     : float = 0.0,
    s22     : float = 0.0,
    c31     : float = 0.0,
    s31     : float = 0.0,
    c32     : float = 0.0,
    s32     : float = 0.0,
    c33     : float = 0.0,
    s33     : float = 0.0,
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
      c21 : float
        C21 tesseral harmonic coefficient
      s21 : float
        S21 tesseral harmonic coefficient
      c22 : float
        C22 tesseral harmonic coefficient
      s22 : float
        S22 tesseral harmonic coefficient
      c31 : float
        C31 tesseral harmonic coefficient
      s31 : float
        S31 tesseral harmonic coefficient
      c32 : float
        C32 tesseral harmonic coefficient
      s32 : float
        S32 tesseral harmonic coefficient
      c33 : float
        C33 tesseral harmonic coefficient
      s33 : float
        S33 tesseral harmonic coefficient
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
    self.c21     = c21
    self.s21     = s21
    self.c22     = c22
    self.s22     = s22
    self.c31     = c31
    self.s31     = s31
    self.c32     = c32
    self.s32     = s32
    self.c33     = c33
    self.s33     = s33
  
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

  def point_mass_jacobian(
    self,
    pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Analytical 6x6 Jacobian matrix A for two-body point mass gravity.

    The state transition matrix Φ satisfies: dΦ/dt = A Φ

    where the A matrix (Jacobian of state derivative w.r.t. state) is:
      A = [      0   I  ]
          [  ∂a/∂r   0  ]

    The gravity gradient ∂a/∂r for a = -μr/r³ is:
      ∂a/∂r = -μ/r³ * I + 3μ/r⁵ * (r ⊗ r)

    where r ⊗ r is the outer product of position with itself.

    Input:
    ------
      pos_vec : np.ndarray
        Position vector [m]

    Output:
    -------
      dposveldotvec__dposvelvec : np.ndarray (6x6)
        Jacobian matrix for STM propagation
    """
    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr3 = pos_mag**3
    pos_mag_pwr5 = pos_mag**5

    # d(acc_vec)/d(pos_vec)
    daccvec__dposvec = \
      -self.gp / pos_mag_pwr3 * np.eye(3) \
      + 3.0 * self.gp / pos_mag_pwr5 * np.outer(pos_vec, pos_vec)
    
    # Build 6x6 Jacobian: d(posveldotvec)/d(posvelvec)
    dposveldotvec__dposvelvec           = np.zeros((6, 6))
    dposveldotvec__dposvelvec[0:3, 3:6] = np.eye(3)         # d(pos_dot_vec)/d(vel_vec) = I
    dposveldotvec__dposvelvec[3:6, 0:3] = daccvec__dposvec  # d(vel_dot_vec)/d(pos_vec) = d(acc_vec)/d(pos_vec)
    
    return dposveldotvec__dposvelvec
 
  def oblate_j2(
      self,
      time_et       : float,
      j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    J2 oblateness perturbation
    
    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).
    
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²] in J2000 frame.

    Notes:
    ------
      Zonal harmonics are defined in the Body-Fixed frame. This method transforms
      the position to IAU_EARTH, computes the acceleration, then transforms back
      to J2000 to properly account for precession/nutation.
    """
    if self.j2 == 0.0:
      return np.zeros(3)
    
    # Get rotation matrix from J2000 to Body-Fixed (IAU_EARTH)
    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      # Fallback to J2000 if transformation fails (kernels not loaded)
      rot_mat_j2000_to_iau_earth = np.eye(3)
    
    # Transform position to body-fixed frame
    pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    
    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2
    pos_mag_pwr5 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag
    
    factor = 1.5 * self.j2 * self.gp * self.pos_ref**2 / pos_mag_pwr5
    
    # Compute acceleration in body-fixed frame
    iau_earth_acc_vec    = np.zeros(3)
    iau_earth_acc_vec[0] = factor * pos_vec[0] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 1)
    iau_earth_acc_vec[1] = factor * pos_vec[1] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 1)
    iau_earth_acc_vec[2] = factor * pos_vec[2] * (5 * pos_vec[2]**2 / pos_mag_pwr2 - 3)
    
    # Transform acceleration back to J2000
    j2000_acc_vec = rot_mat_j2000_to_iau_earth.T @ iau_earth_acc_vec
    
    return j2000_acc_vec
  
  def oblate_j2_jacobian(
    self,
    time_et : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Analytical 3x3 Jacobian matrix for J2 oblateness perturbation.

    Computes ∂a_J2/∂r, the partial derivative of J2 acceleration with respect
    to position. This is added to the gravity gradient in the STM A matrix.

    The J2 acceleration in body-fixed frame is:
      a_x = k * x * (5z²/r² - 1)
      a_y = k * y * (5z²/r² - 1)
      a_z = k * z * (5z²/r² - 3)

    where k = (3/2) * J2 * μ * R_e² / r⁵

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).

    Output:
    -------
      daccvec__dposvec : np.ndarray (3x3)
        Jacobian matrix ∂a_J2/∂r in J2000 frame
    """
    if self.j2 == 0.0:
      return np.zeros((3, 3))

    # Get rotation matrix from J2000 to Body-Fixed (IAU_EARTH)
    rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)

    # Transform position to body-fixed frame
    pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    x, y, z = pos_vec[0], pos_vec[1], pos_vec[2]

    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2
    pos_mag_pwr5 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag

    # Common factors
    k  = 1.5 * self.j2 * self.gp * self.pos_ref**2
    z2 = z**2
    z2_over_r2 = z2 / pos_mag_pwr2

    # Jacobian in body-fixed frame
    # Derived from partial derivatives of J2 acceleration equations
    jac_bf = np.zeros((3, 3))

    # Common terms
    term1 = 5.0 * z2_over_r2 - 1.0
    term2 = 5.0 * z2_over_r2 - 3.0

    # ∂a_x/∂x, ∂a_x/∂y, ∂a_x/∂z
    jac_bf[0, 0] = k / pos_mag_pwr5 * (term1 - 5.0 * x**2 / pos_mag_pwr2 * (7.0 * z2_over_r2 - 1.0))
    jac_bf[0, 1] = k / pos_mag_pwr5 * (-5.0 * x * y / pos_mag_pwr2 * (7.0 * z2_over_r2 - 1.0))
    jac_bf[0, 2] = k / pos_mag_pwr5 * (10.0 * x * z / pos_mag_pwr2 * (1.0 - 7.0 * z2_over_r2 / 2.0 + 0.5))

    # ∂a_y/∂x, ∂a_y/∂y, ∂a_y/∂z
    jac_bf[1, 0] = k / pos_mag_pwr5 * (-5.0 * y * x / pos_mag_pwr2 * (7.0 * z2_over_r2 - 1.0))
    jac_bf[1, 1] = k / pos_mag_pwr5 * (term1 - 5.0 * y**2 / pos_mag_pwr2 * (7.0 * z2_over_r2 - 1.0))
    jac_bf[1, 2] = k / pos_mag_pwr5 * (10.0 * y * z / pos_mag_pwr2 * (1.0 - 7.0 * z2_over_r2 / 2.0 + 0.5))

    # ∂a_z/∂x, ∂a_z/∂y, ∂a_z/∂z
    jac_bf[2, 0] = k / pos_mag_pwr5 * (-5.0 * z * x / pos_mag_pwr2 * (7.0 * z2_over_r2 - 3.0))
    jac_bf[2, 1] = k / pos_mag_pwr5 * (-5.0 * z * y / pos_mag_pwr2 * (7.0 * z2_over_r2 - 3.0))
    jac_bf[2, 2] = k / pos_mag_pwr5 * (term2 + 10.0 * z2 / pos_mag_pwr2 * (1.0 - 7.0 * z2_over_r2 / 2.0 + 1.5))

    # Transform Jacobian back to J2000 frame
    # For a tensor transformation: J_j2000 = R^T * J_bf * R
    rot_T = rot_mat_j2000_to_iau_earth.T
    j2000_jac = rot_T @ jac_bf @ rot_mat_j2000_to_iau_earth

    return j2000_jac

  def oblate_j3(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute J3 oblateness perturbation acceleration.
    
    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).
    
    Output:
    -------
      acc_vec : np.ndarray
        Acceleration vector [m/s²] in J2000 frame.

    Notes:
    ------
      Zonal harmonics are defined in the Body-Fixed frame. This method transforms
      the position to IAU_EARTH, computes the acceleration, then transforms back
      to J2000 to properly account for precession/nutation.
    """
    if self.j3 == 0.0:
      return np.zeros(3)
    
    # Get rotation matrix from J2000 to Body-Fixed (IAU_EARTH)
    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      rot_mat_j2000_to_iau_earth = np.eye(3)
    
    # Transform position to body-fixed frame
    pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    pos_x, pos_y, pos_z = pos_vec[0], pos_vec[1], pos_vec[2]

    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2
    pos_mag_pwr7 = pos_mag_pwr2 * pos_mag_pwr2 * pos_mag_pwr2 * pos_mag
    
    factor = 2.5 * self.j3 * self.gp * self.pos_ref**3 / pos_mag_pwr7
    
    # Compute acceleration in body-fixed frame
    iau_earth_acc_vec    = np.zeros(3)
    iau_earth_acc_vec[0] = factor * pos_x * pos_z * (3.0 - 7.0 * pos_z**2 / pos_mag_pwr2)
    iau_earth_acc_vec[1] = factor * pos_y * pos_z * (3.0 - 7.0 * pos_z**2 / pos_mag_pwr2)
    iau_earth_acc_vec[2] = factor * (3.0 * pos_z**2 - 7.0 * pos_z**4 / pos_mag_pwr2 - 0.6 * pos_mag_pwr2)
    
    # Transform acceleration back to J2000
    j2000_acc_vec = rot_mat_j2000_to_iau_earth.T @ iau_earth_acc_vec

    return j2000_acc_vec

  def oblate_j3_jacobian(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Analytical 3x3 Jacobian matrix for J3 oblateness perturbation.

    Computes ∂a_J3/∂r, the partial derivative of J3 acceleration with respect
    to position. This is added to the gravity gradient in the STM A matrix.

    The J3 acceleration in body-fixed frame is:
      a_x = k * x * z * (3 - 7z²/r²)
      a_y = k * y * z * (3 - 7z²/r²)
      a_z = k * (3z² - 7z⁴/r² - 0.6r²)

    where k = 2.5 * J3 * μ * R_e³ / r⁷

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in Inertial frame (J2000).

    Output:
    -------
      daccvec__dposvec : np.ndarray (3x3)
        Jacobian matrix ∂a_J3/∂r in J2000 frame
    """
    if self.j3 == 0.0:
      return np.zeros((3, 3))

    # Get rotation matrix from J2000 to Body-Fixed (IAU_EARTH)
    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      rot_mat_j2000_to_iau_earth = np.eye(3)

    # Transform position to body-fixed frame
    pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    x, y, z = pos_vec[0], pos_vec[1], pos_vec[2]

    pos_mag      = np.linalg.norm(pos_vec)
    pos_mag_pwr2 = pos_mag**2
    pos_mag_pwr4 = pos_mag_pwr2**2
    pos_mag_pwr7 = pos_mag_pwr4 * pos_mag_pwr2 * pos_mag

    # Common factors
    k  = 2.5 * self.j3 * self.gp * self.pos_ref**3
    z2 = z**2
    z2_over_r2 = z2 / pos_mag_pwr2

    # Jacobian in body-fixed frame
    jac_bf = np.zeros((3, 3))

    # Common terms
    term1 = 3.0 - 7.0 * z2_over_r2

    # ∂a_x/∂x, ∂a_x/∂y, ∂a_x/∂z
    jac_bf[0, 0] = k / pos_mag_pwr7 * z * (term1 - 7.0 * x**2 / pos_mag_pwr2 * (1.0 - 9.0 * z2_over_r2))
    jac_bf[0, 1] = k / pos_mag_pwr7 * z * (-7.0 * x * y / pos_mag_pwr2 * (1.0 - 9.0 * z2_over_r2))
    jac_bf[0, 2] = k / pos_mag_pwr7 * x * (3.0 * pos_mag_pwr2 - 21.0 * z2 + 14.0 * z * x * z / pos_mag_pwr2 * (1.0 - 9.0 * z2_over_r2))

    # ∂a_y/∂x, ∂a_y/∂y, ∂a_y/∂z
    jac_bf[1, 0] = k / pos_mag_pwr7 * z * (-7.0 * y * x / pos_mag_pwr2 * (1.0 - 9.0 * z2_over_r2))
    jac_bf[1, 1] = k / pos_mag_pwr7 * z * (term1 - 7.0 * y**2 / pos_mag_pwr2 * (1.0 - 9.0 * z2_over_r2))
    jac_bf[1, 2] = k / pos_mag_pwr7 * y * (3.0 * pos_mag_pwr2 - 21.0 * z2 + 14.0 * z * y * z / pos_mag_pwr2 * (1.0 - 9.0 * z2_over_r2))

    # ∂a_z/∂x, ∂a_z/∂y, ∂a_z/∂z
    jac_bf[2, 0] = k / pos_mag_pwr7 * x * (-1.2 * pos_mag_pwr2 + 21.0 * z2 - 28.0 * z2_over_r2**2)
    jac_bf[2, 1] = k / pos_mag_pwr7 * y * (-1.2 * pos_mag_pwr2 + 21.0 * z2 - 28.0 * z2_over_r2**2)
    jac_bf[2, 2] = k / pos_mag_pwr7 * (6.0 * z * pos_mag_pwr2 - 42.0 * z2_over_r2 * z + 14.0 * z / pos_mag_pwr2 * (3.0 * z2 - 7.0 * z2_over_r2**2 - 0.6 * pos_mag_pwr2))

    # Transform Jacobian back to J2000 frame
    rot_T = rot_mat_j2000_to_iau_earth.T
    j2000_jac = rot_T @ jac_bf @ rot_mat_j2000_to_iau_earth

    return j2000_jac

  def tesseral_21(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute C21 and S21 tesseral harmonic perturbation acceleration.
    
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
    if self.c21 == 0.0 and self.s21 == 0.0:
      return np.zeros(3)

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros(3)

    iau_earth_pos_vec   = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    pos_x, pos_y, pos_z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]
    
    pos_mag_pwr2 = pos_x**2 + pos_y**2 + pos_z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr7 = pos_mag_pwr2**3 * pos_mag
    
    term_common = self.c21 * pos_x + self.s21 * pos_y
    factor      = 3.0 * self.gp * self.pos_ref**2 / pos_mag_pwr7
    
    iau_earth_acc_x   = factor * pos_z * (self.c21 * pos_mag_pwr2 - 5.0 * pos_x * term_common)
    iau_earth_acc_y   = factor * pos_z * (self.s21 * pos_mag_pwr2 - 5.0 * pos_y * term_common)
    iau_earth_acc_z   = factor * (term_common * pos_mag_pwr2 - 5.0 * pos_z**2 * term_common)
    iau_earth_acc_vec = np.array([iau_earth_acc_x, iau_earth_acc_y, iau_earth_acc_z])

    return rot_mat_j2000_to_iau_earth.T @ iau_earth_acc_vec

  def tesseral_21_jacobian(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Analytical 3x3 Jacobian matrix for C21 and S21 tesseral harmonic perturbation.

    Computes ∂a_T21/∂r, the partial derivative of T21 acceleration with respect
    to position.

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in inertial frame (J2000).

    Output:
    -------
      daccvec__dposvec : np.ndarray (3x3)
        Jacobian matrix ∂a_T21/∂r in J2000 frame
    """
    if self.c21 == 0.0 and self.s21 == 0.0:
      return np.zeros((3, 3))

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros((3, 3))

    iau_earth_pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    x, y, z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]

    pos_mag_pwr2 = x**2 + y**2 + z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr7 = pos_mag_pwr2**3 * pos_mag

    term_common = self.c21 * x + self.s21 * y
    factor      = 3.0 * self.gp * self.pos_ref**2 / pos_mag_pwr7

    # Jacobian in body-fixed frame
    jac_bf = np.zeros((3, 3))

    # ∂a_x/∂x, ∂a_x/∂y, ∂a_x/∂z
    jac_bf[0, 0] = factor * z * (self.c21 * 2.0 * x - 5.0 * (self.c21 * x + term_common * x / pos_mag_pwr2) - 7.0 * x / pos_mag_pwr2 * (self.c21 * pos_mag_pwr2 - 5.0 * x * term_common))
    jac_bf[0, 1] = factor * z * (self.c21 * 2.0 * y - 5.0 * (self.s21 * x + term_common * y / pos_mag_pwr2) - 7.0 * y / pos_mag_pwr2 * (self.c21 * pos_mag_pwr2 - 5.0 * x * term_common))
    jac_bf[0, 2] = factor * (self.c21 * pos_mag_pwr2 - 5.0 * x * term_common - 7.0 * z**2 / pos_mag_pwr2 * (self.c21 * pos_mag_pwr2 - 5.0 * x * term_common) + 10.0 * z * term_common)

    # ∂a_y/∂x, ∂a_y/∂y, ∂a_y/∂z
    jac_bf[1, 0] = factor * z * (self.s21 * 2.0 * x - 5.0 * (self.c21 * y + term_common * x / pos_mag_pwr2) - 7.0 * x / pos_mag_pwr2 * (self.s21 * pos_mag_pwr2 - 5.0 * y * term_common))
    jac_bf[1, 1] = factor * z * (self.s21 * 2.0 * y - 5.0 * (self.s21 * y + term_common * y / pos_mag_pwr2) - 7.0 * y / pos_mag_pwr2 * (self.s21 * pos_mag_pwr2 - 5.0 * y * term_common))
    jac_bf[1, 2] = factor * (self.s21 * pos_mag_pwr2 - 5.0 * y * term_common - 7.0 * z**2 / pos_mag_pwr2 * (self.s21 * pos_mag_pwr2 - 5.0 * y * term_common) + 10.0 * z * term_common)

    # ∂a_z/∂x, ∂a_z/∂y, ∂a_z/∂z
    jac_bf[2, 0] = factor * (self.c21 * 2.0 * x - 5.0 * term_common * 2.0 * x / pos_mag_pwr2 - 7.0 * x / pos_mag_pwr2 * (term_common * pos_mag_pwr2 - 5.0 * z**2 * term_common))
    jac_bf[2, 1] = factor * (self.s21 * 2.0 * y - 5.0 * term_common * 2.0 * y / pos_mag_pwr2 - 7.0 * y / pos_mag_pwr2 * (term_common * pos_mag_pwr2 - 5.0 * z**2 * term_common))
    jac_bf[2, 2] = factor * (-10.0 * z * term_common - 7.0 * z / pos_mag_pwr2 * (term_common * pos_mag_pwr2 - 5.0 * z**2 * term_common) + 10.0 * term_common * 2.0 * z)

    # Transform Jacobian back to J2000 frame
    rot_T = rot_mat_j2000_to_iau_earth.T
    j2000_jac = rot_T @ jac_bf @ rot_mat_j2000_to_iau_earth

    return j2000_jac

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

  def tesseral_22_jacobian(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Analytical 3x3 Jacobian matrix for C22 and S22 tesseral harmonic perturbation.

    Computes ∂a_T22/∂r, the partial derivative of T22 acceleration with respect
    to position.

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in inertial frame (J2000).

    Output:
    -------
      daccvec__dposvec : np.ndarray (3x3)
        Jacobian matrix ∂a_T22/∂r in J2000 frame
    """
    if self.c22 == 0.0 and self.s22 == 0.0:
      return np.zeros((3, 3))

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros((3, 3))

    iau_earth_pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    x, y, z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]

    pos_mag_pwr2 = x**2 + y**2 + z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr7 = pos_mag_pwr2**3 * pos_mag

    term_common = self.c22 * (x**2 - y**2) + 2.0 * self.s22 * x * y
    factor      = 3.0 * self.gp * self.pos_ref**2 / pos_mag_pwr7

    # Derivatives of term_common
    d_term_dx = 2.0 * self.c22 * x + 2.0 * self.s22 * y
    d_term_dy = -2.0 * self.c22 * y + 2.0 * self.s22 * x

    # Jacobian in body-fixed frame
    jac_bf = np.zeros((3, 3))

    # ∂a_x/∂x, ∂a_x/∂y, ∂a_x/∂z
    jac_bf[0, 0] = factor * (-5.0 * d_term_dx * x - 5.0 * term_common + 2.0 * self.c22 * pos_mag_pwr2 + 2.0 * d_term_dx * x - 7.0 * x / pos_mag_pwr2 * (-5.0 * x * term_common + pos_mag_pwr2 * d_term_dx))
    jac_bf[0, 1] = factor * (-5.0 * d_term_dy * x - 7.0 * y / pos_mag_pwr2 * (-5.0 * x * term_common + pos_mag_pwr2 * d_term_dx) + 2.0 * self.s22 * pos_mag_pwr2 + 2.0 * d_term_dy * x)
    jac_bf[0, 2] = factor * (-7.0 * z / pos_mag_pwr2 * (-5.0 * x * term_common + pos_mag_pwr2 * d_term_dx))

    # ∂a_y/∂x, ∂a_y/∂y, ∂a_y/∂z
    jac_bf[1, 0] = factor * (-5.0 * d_term_dx * y - 7.0 * x / pos_mag_pwr2 * (-5.0 * y * term_common + pos_mag_pwr2 * d_term_dy) + 2.0 * self.s22 * pos_mag_pwr2 + 2.0 * d_term_dx * y)
    jac_bf[1, 1] = factor * (-5.0 * d_term_dy * y - 5.0 * term_common - 2.0 * self.c22 * pos_mag_pwr2 + 2.0 * d_term_dy * y - 7.0 * y / pos_mag_pwr2 * (-5.0 * y * term_common + pos_mag_pwr2 * d_term_dy))
    jac_bf[1, 2] = factor * (-7.0 * z / pos_mag_pwr2 * (-5.0 * y * term_common + pos_mag_pwr2 * d_term_dy))

    # ∂a_z/∂x, ∂a_z/∂y, ∂a_z/∂z
    jac_bf[2, 0] = factor * (-5.0 * d_term_dx * z - 7.0 * x / pos_mag_pwr2 * (-5.0 * z * term_common))
    jac_bf[2, 1] = factor * (-5.0 * d_term_dy * z - 7.0 * y / pos_mag_pwr2 * (-5.0 * z * term_common))
    jac_bf[2, 2] = factor * (-5.0 * term_common - 7.0 * z / pos_mag_pwr2 * (-5.0 * z * term_common))

    # Transform Jacobian back to J2000 frame
    rot_T = rot_mat_j2000_to_iau_earth.T
    j2000_jac = rot_T @ jac_bf @ rot_mat_j2000_to_iau_earth

    return j2000_jac

  def tesseral_31(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute C31 and S31 tesseral harmonic perturbation acceleration.
    
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
    if self.c31 == 0.0 and self.s31 == 0.0:
      return np.zeros(3)

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros(3)

    iau_earth_pos_vec   = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    pos_x, pos_y, pos_z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]
    
    pos_mag_pwr2 = pos_x**2 + pos_y**2 + pos_z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag
    
    term_common = self.c31 * pos_x + self.s31 * pos_y
    factor      = 1.5 * self.gp * self.pos_ref**3 / pos_mag_pwr9
    
    # (5z^2 - r^2) term
    z2_term = 5.0 * pos_z**2 - pos_mag_pwr2
    
    iau_earth_acc_x   = factor * (self.c31 * z2_term - 7.0 * pos_x * term_common * z2_term / pos_mag_pwr2 + 10.0 * pos_x * pos_z**2 * term_common / pos_mag_pwr2)
    iau_earth_acc_y   = factor * (self.s31 * z2_term - 7.0 * pos_y * term_common * z2_term / pos_mag_pwr2 + 10.0 * pos_y * pos_z**2 * term_common / pos_mag_pwr2)
    iau_earth_acc_z   = factor * (10.0 * pos_z * term_common - 7.0 * pos_z * term_common * z2_term / pos_mag_pwr2)
    iau_earth_acc_vec = np.array([iau_earth_acc_x, iau_earth_acc_y, iau_earth_acc_z])

    return rot_mat_j2000_to_iau_earth.T @ iau_earth_acc_vec

  def tesseral_31_jacobian(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Analytical 3x3 Jacobian matrix for C31 and S31 tesseral harmonic perturbation.

    Computes ∂a_T31/∂r, the partial derivative of T31 acceleration with respect
    to position.

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in inertial frame (J2000).

    Output:
    -------
      daccvec__dposvec : np.ndarray (3x3)
        Jacobian matrix ∂a_T31/∂r in J2000 frame
    """
    if self.c31 == 0.0 and self.s31 == 0.0:
      return np.zeros((3, 3))

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros((3, 3))

    iau_earth_pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    x, y, z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]

    pos_mag_pwr2 = x**2 + y**2 + z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag

    term_common = self.c31 * x + self.s31 * y
    factor      = 1.5 * self.gp * self.pos_ref**3 / pos_mag_pwr9
    z2_term     = 5.0 * z**2 - pos_mag_pwr2

    # Jacobian in body-fixed frame
    jac_bf = np.zeros((3, 3))

    # ∂a_x/∂x, ∂a_x/∂y, ∂a_x/∂z
    jac_bf[0, 0] = factor * (self.c31 * z2_term - 2.0 * self.c31 * x - 7.0 * x * term_common * z2_term / pos_mag_pwr2 + 14.0 * x**2 * term_common * z2_term / pos_mag_pwr2**2 + 10.0 * x * z**2 * term_common / pos_mag_pwr2 - 20.0 * x**3 * z**2 * term_common / pos_mag_pwr2**2 + 20.0 * z**2 * term_common / pos_mag_pwr2 - 9.0 * x / pos_mag_pwr2 * (self.c31 * z2_term - 7.0 * x * term_common * z2_term / pos_mag_pwr2 + 10.0 * x * z**2 * term_common / pos_mag_pwr2))
    jac_bf[0, 1] = factor * (self.s31 * z2_term - 2.0 * self.c31 * y - 7.0 * y * term_common * z2_term / pos_mag_pwr2 + 14.0 * x * y * term_common * z2_term / pos_mag_pwr2**2 + 10.0 * y * z**2 * term_common / pos_mag_pwr2 - 20.0 * x**2 * y * z**2 * term_common / pos_mag_pwr2**2 - 9.0 * y / pos_mag_pwr2 * (self.c31 * z2_term - 7.0 * x * term_common * z2_term / pos_mag_pwr2 + 10.0 * x * z**2 * term_common / pos_mag_pwr2))
    jac_bf[0, 2] = factor * (10.0 * self.c31 * z - 14.0 * z * term_common * z2_term / pos_mag_pwr2 + 14.0 * x * z / pos_mag_pwr2 * term_common * z2_term * (1.0 - z**2 / pos_mag_pwr2) + 20.0 * x * z * term_common / pos_mag_pwr2 - 20.0 * x * z**3 * term_common / pos_mag_pwr2**2 - 9.0 * z / pos_mag_pwr2 * (self.c31 * z2_term - 7.0 * x * term_common * z2_term / pos_mag_pwr2 + 10.0 * x * z**2 * term_common / pos_mag_pwr2))

    # ∂a_y/∂x, ∂a_y/∂y, ∂a_y/∂z (similar structure)
    jac_bf[1, 0] = factor * (self.s31 * z2_term - 2.0 * self.s31 * x - 7.0 * x * term_common * z2_term / pos_mag_pwr2 + 14.0 * x * y * term_common * z2_term / pos_mag_pwr2**2 + 10.0 * x * z**2 * term_common / pos_mag_pwr2 - 20.0 * x * y**2 * z**2 * term_common / pos_mag_pwr2**2 - 9.0 * x / pos_mag_pwr2 * (self.s31 * z2_term - 7.0 * y * term_common * z2_term / pos_mag_pwr2 + 10.0 * y * z**2 * term_common / pos_mag_pwr2))
    jac_bf[1, 1] = factor * (self.s31 * z2_term - 2.0 * self.s31 * y - 7.0 * y * term_common * z2_term / pos_mag_pwr2 + 14.0 * y**2 * term_common * z2_term / pos_mag_pwr2**2 + 10.0 * y * z**2 * term_common / pos_mag_pwr2 - 20.0 * y**3 * z**2 * term_common / pos_mag_pwr2**2 + 20.0 * z**2 * term_common / pos_mag_pwr2 - 9.0 * y / pos_mag_pwr2 * (self.s31 * z2_term - 7.0 * y * term_common * z2_term / pos_mag_pwr2 + 10.0 * y * z**2 * term_common / pos_mag_pwr2))
    jac_bf[1, 2] = factor * (10.0 * self.s31 * z - 14.0 * z * term_common * z2_term / pos_mag_pwr2 + 14.0 * y * z / pos_mag_pwr2 * term_common * z2_term * (1.0 - z**2 / pos_mag_pwr2) + 20.0 * y * z * term_common / pos_mag_pwr2 - 20.0 * y * z**3 * term_common / pos_mag_pwr2**2 - 9.0 * z / pos_mag_pwr2 * (self.s31 * z2_term - 7.0 * y * term_common * z2_term / pos_mag_pwr2 + 10.0 * y * z**2 * term_common / pos_mag_pwr2))

    # ∂a_z/∂x, ∂a_z/∂y, ∂a_z/∂z
    jac_bf[2, 0] = factor * (10.0 * self.c31 * z - 7.0 * self.c31 * z * z2_term / pos_mag_pwr2 + 14.0 * self.c31 * z * x / pos_mag_pwr2 - 7.0 * x * term_common * z2_term / pos_mag_pwr2 + 14.0 * x * z * term_common * z2_term / pos_mag_pwr2**2 - 9.0 * x / pos_mag_pwr2 * (10.0 * z * term_common - 7.0 * z * term_common * z2_term / pos_mag_pwr2))
    jac_bf[2, 1] = factor * (10.0 * self.s31 * z - 7.0 * self.s31 * z * z2_term / pos_mag_pwr2 + 14.0 * self.s31 * z * y / pos_mag_pwr2 - 7.0 * y * term_common * z2_term / pos_mag_pwr2 + 14.0 * y * z * term_common * z2_term / pos_mag_pwr2**2 - 9.0 * y / pos_mag_pwr2 * (10.0 * z * term_common - 7.0 * z * term_common * z2_term / pos_mag_pwr2))
    jac_bf[2, 2] = factor * (10.0 * term_common - 7.0 * term_common * z2_term / pos_mag_pwr2 + 20.0 * term_common * z**2 / pos_mag_pwr2 - 14.0 * term_common * z**2 * z2_term / pos_mag_pwr2**2 - 9.0 * z / pos_mag_pwr2 * (10.0 * z * term_common - 7.0 * z * term_common * z2_term / pos_mag_pwr2))

    # Transform Jacobian back to J2000 frame
    rot_T = rot_mat_j2000_to_iau_earth.T
    j2000_jac = rot_T @ jac_bf @ rot_mat_j2000_to_iau_earth

    return j2000_jac

  def tesseral_32(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute C32 and S32 tesseral harmonic perturbation acceleration.
    
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
    if self.c32 == 0.0 and self.s32 == 0.0:
      return np.zeros(3)

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros(3)

    iau_earth_pos_vec   = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    pos_x, pos_y, pos_z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]
    
    pos_mag_pwr2 = pos_x**2 + pos_y**2 + pos_z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag
    
    term_common = self.c32 * (pos_x**2 - pos_y**2) + 2.0 * self.s32 * pos_x * pos_y
    factor      = 3.0 * self.gp * self.pos_ref**3 / pos_mag_pwr9
    
    iau_earth_acc_x   = factor * pos_z * (2.0 * self.c32 * pos_x + 2.0 * self.s32 * pos_y - 7.0 * pos_x * term_common / pos_mag_pwr2)
    iau_earth_acc_y   = factor * pos_z * (-2.0 * self.c32 * pos_y + 2.0 * self.s32 * pos_x - 7.0 * pos_y * term_common / pos_mag_pwr2)
    iau_earth_acc_z   = factor * (term_common - 7.0 * pos_z**2 * term_common / pos_mag_pwr2)
    iau_earth_acc_vec = np.array([iau_earth_acc_x, iau_earth_acc_y, iau_earth_acc_z])

    return rot_mat_j2000_to_iau_earth.T @ iau_earth_acc_vec

  def tesseral_32_jacobian(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Analytical 3x3 Jacobian matrix for C32 and S32 tesseral harmonic perturbation.

    Computes ∂a_T32/∂r, the partial derivative of T32 acceleration with respect
    to position.

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in inertial frame (J2000).

    Output:
    -------
      daccvec__dposvec : np.ndarray (3x3)
        Jacobian matrix ∂a_T32/∂r in J2000 frame
    """
    if self.c32 == 0.0 and self.s32 == 0.0:
      return np.zeros((3, 3))

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros((3, 3))

    iau_earth_pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    x, y, z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]

    pos_mag_pwr2 = x**2 + y**2 + z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag

    term_common = self.c32 * (x**2 - y**2) + 2.0 * self.s32 * x * y
    factor      = 3.0 * self.gp * self.pos_ref**3 / pos_mag_pwr9

    # Derivatives of term_common
    d_term_dx = 2.0 * self.c32 * x + 2.0 * self.s32 * y
    d_term_dy = -2.0 * self.c32 * y + 2.0 * self.s32 * x

    # Jacobian in body-fixed frame
    jac_bf = np.zeros((3, 3))

    # ∂a_x/∂x, ∂a_x/∂y, ∂a_x/∂z
    jac_bf[0, 0] = factor * z * (2.0 * self.c32 + 2.0 * d_term_dx - 7.0 * d_term_dx * x / pos_mag_pwr2 + 14.0 * x**2 / pos_mag_pwr2**2 * (2.0 * self.c32 * x + 2.0 * self.s32 * y - 7.0 * x * term_common / pos_mag_pwr2) - 9.0 * x / pos_mag_pwr2 * (2.0 * self.c32 * x + 2.0 * self.s32 * y - 7.0 * x * term_common / pos_mag_pwr2))
    jac_bf[0, 1] = factor * z * (2.0 * self.s32 + 2.0 * d_term_dy - 7.0 * d_term_dy * x / pos_mag_pwr2 + 14.0 * x * y / pos_mag_pwr2**2 * (2.0 * self.c32 * x + 2.0 * self.s32 * y - 7.0 * x * term_common / pos_mag_pwr2) - 9.0 * y / pos_mag_pwr2 * (2.0 * self.c32 * x + 2.0 * self.s32 * y - 7.0 * x * term_common / pos_mag_pwr2))
    jac_bf[0, 2] = factor * (2.0 * self.c32 * x + 2.0 * self.s32 * y - 7.0 * x * term_common / pos_mag_pwr2 + 14.0 * x * z**2 / pos_mag_pwr2**2 * term_common - 9.0 * z / pos_mag_pwr2 * (2.0 * self.c32 * x + 2.0 * self.s32 * y - 7.0 * x * term_common / pos_mag_pwr2))

    # ∂a_y/∂x, ∂a_y/∂y, ∂a_y/∂z
    jac_bf[1, 0] = factor * z * (2.0 * self.s32 - 2.0 * d_term_dx - 7.0 * d_term_dx * y / pos_mag_pwr2 + 14.0 * x * y / pos_mag_pwr2**2 * (-2.0 * self.c32 * y + 2.0 * self.s32 * x - 7.0 * y * term_common / pos_mag_pwr2) - 9.0 * x / pos_mag_pwr2 * (-2.0 * self.c32 * y + 2.0 * self.s32 * x - 7.0 * y * term_common / pos_mag_pwr2))
    jac_bf[1, 1] = factor * z * (-2.0 * self.c32 - 2.0 * d_term_dy - 7.0 * d_term_dy * y / pos_mag_pwr2 + 14.0 * y**2 / pos_mag_pwr2**2 * (-2.0 * self.c32 * y + 2.0 * self.s32 * x - 7.0 * y * term_common / pos_mag_pwr2) - 9.0 * y / pos_mag_pwr2 * (-2.0 * self.c32 * y + 2.0 * self.s32 * x - 7.0 * y * term_common / pos_mag_pwr2))
    jac_bf[1, 2] = factor * (-2.0 * self.c32 * y + 2.0 * self.s32 * x - 7.0 * y * term_common / pos_mag_pwr2 + 14.0 * y * z**2 / pos_mag_pwr2**2 * term_common - 9.0 * z / pos_mag_pwr2 * (-2.0 * self.c32 * y + 2.0 * self.s32 * x - 7.0 * y * term_common / pos_mag_pwr2))

    # ∂a_z/∂x, ∂a_z/∂y, ∂a_z/∂z
    jac_bf[2, 0] = factor * (d_term_dx - 7.0 * d_term_dx * z**2 / pos_mag_pwr2 + 14.0 * x * z**2 / pos_mag_pwr2**2 * term_common - 9.0 * x / pos_mag_pwr2 * (term_common - 7.0 * z**2 * term_common / pos_mag_pwr2))
    jac_bf[2, 1] = factor * (d_term_dy - 7.0 * d_term_dy * z**2 / pos_mag_pwr2 + 14.0 * y * z**2 / pos_mag_pwr2**2 * term_common - 9.0 * y / pos_mag_pwr2 * (term_common - 7.0 * z**2 * term_common / pos_mag_pwr2))
    jac_bf[2, 2] = factor * (-14.0 * z * term_common / pos_mag_pwr2 + 14.0 * z**3 / pos_mag_pwr2**2 * term_common - 9.0 * z / pos_mag_pwr2 * (term_common - 7.0 * z**2 * term_common / pos_mag_pwr2))

    # Transform Jacobian back to J2000 frame
    rot_T = rot_mat_j2000_to_iau_earth.T
    j2000_jac = rot_T @ jac_bf @ rot_mat_j2000_to_iau_earth

    return j2000_jac

  def tesseral_33(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute C33 and S33 tesseral harmonic perturbation acceleration.
    
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
    if self.c33 == 0.0 and self.s33 == 0.0:
      return np.zeros(3)

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros(3)

    iau_earth_pos_vec   = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    pos_x, pos_y, pos_z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]
    
    pos_mag_pwr2 = pos_x**2 + pos_y**2 + pos_z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag
    
    # cos(3*lon) term: x*(x^2-3y^2), sin(3*lon) term: y*(3x^2-y^2)
    term_c33    = pos_x * (pos_x**2 - 3.0 * pos_y**2)
    term_s33    = pos_y * (3.0 * pos_x**2 - pos_y**2)
    term_common = self.c33 * term_c33 + self.s33 * term_s33
    factor      = 1.0 * self.gp * self.pos_ref**3 / pos_mag_pwr9
    
    # Partial derivatives of term_c33 and term_s33
    d_term_c33_dx = 3.0 * pos_x**2 - 3.0 * pos_y**2
    d_term_c33_dy = -6.0 * pos_x * pos_y
    d_term_s33_dx = 6.0 * pos_x * pos_y
    d_term_s33_dy = 3.0 * pos_x**2 - 3.0 * pos_y**2
    
    iau_earth_acc_x   = factor * (self.c33 * d_term_c33_dx + self.s33 * d_term_s33_dx - 7.0 * pos_x * term_common / pos_mag_pwr2)
    iau_earth_acc_y   = factor * (self.c33 * d_term_c33_dy + self.s33 * d_term_s33_dy - 7.0 * pos_y * term_common / pos_mag_pwr2)
    iau_earth_acc_z   = factor * (-7.0 * pos_z * term_common / pos_mag_pwr2)
    iau_earth_acc_vec = np.array([iau_earth_acc_x, iau_earth_acc_y, iau_earth_acc_z])

    return rot_mat_j2000_to_iau_earth.T @ iau_earth_acc_vec

  def tesseral_33_jacobian(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Analytical 3x3 Jacobian matrix for C33 and S33 tesseral harmonic perturbation.

    Computes ∂a_T33/∂r, the partial derivative of T33 acceleration with respect
    to position.

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in inertial frame (J2000).

    Output:
    -------
      daccvec__dposvec : np.ndarray (3x3)
        Jacobian matrix ∂a_T33/∂r in J2000 frame
    """
    if self.c33 == 0.0 and self.s33 == 0.0:
      return np.zeros((3, 3))

    try:
      rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      return np.zeros((3, 3))

    iau_earth_pos_vec = rot_mat_j2000_to_iau_earth @ j2000_pos_vec
    x, y, z = iau_earth_pos_vec[0], iau_earth_pos_vec[1], iau_earth_pos_vec[2]

    pos_mag_pwr2 = x**2 + y**2 + z**2
    pos_mag      = np.sqrt(pos_mag_pwr2)
    pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag

    # cos(3*lon) term: x*(x^2-3y^2), sin(3*lon) term: y*(3x^2-y^2)
    term_c33    = x * (x**2 - 3.0 * y**2)
    term_s33    = y * (3.0 * x**2 - y**2)
    term_common = self.c33 * term_c33 + self.s33 * term_s33
    factor      = 1.0 * self.gp * self.pos_ref**3 / pos_mag_pwr9

    # Partial derivatives of term_c33 and term_s33
    d_term_c33_dx = 3.0 * x**2 - 3.0 * y**2
    d_term_c33_dy = -6.0 * x * y
    d_term_s33_dx = 6.0 * x * y
    d_term_s33_dy = 3.0 * x**2 - 3.0 * y**2

    d_term_common_dx = self.c33 * d_term_c33_dx + self.s33 * d_term_s33_dx
    d_term_common_dy = self.c33 * d_term_c33_dy + self.s33 * d_term_s33_dy

    # Jacobian in body-fixed frame
    jac_bf = np.zeros((3, 3))

    # ∂a_x/∂x, ∂a_x/∂y, ∂a_x/∂z
    jac_bf[0, 0] = factor * (d_term_common_dx - 7.0 * d_term_common_dx * x / pos_mag_pwr2 + 14.0 * x**2 / pos_mag_pwr2**2 * (self.c33 * d_term_c33_dx + self.s33 * d_term_s33_dx - 7.0 * x * term_common / pos_mag_pwr2) + self.c33 * 6.0 * x - 9.0 * x / pos_mag_pwr2 * (self.c33 * d_term_c33_dx + self.s33 * d_term_s33_dx - 7.0 * x * term_common / pos_mag_pwr2))
    jac_bf[0, 1] = factor * (d_term_common_dy - 7.0 * d_term_common_dy * x / pos_mag_pwr2 + 14.0 * x * y / pos_mag_pwr2**2 * (self.c33 * d_term_c33_dx + self.s33 * d_term_s33_dx - 7.0 * x * term_common / pos_mag_pwr2) + self.c33 * (-6.0 * y) + self.s33 * 6.0 * x - 9.0 * y / pos_mag_pwr2 * (self.c33 * d_term_c33_dx + self.s33 * d_term_s33_dx - 7.0 * x * term_common / pos_mag_pwr2))
    jac_bf[0, 2] = factor * (-7.0 * z / pos_mag_pwr2 * (self.c33 * d_term_c33_dx + self.s33 * d_term_s33_dx) + 14.0 * x * z**2 / pos_mag_pwr2**2 * term_common - 9.0 * z / pos_mag_pwr2 * (self.c33 * d_term_c33_dx + self.s33 * d_term_s33_dx - 7.0 * x * term_common / pos_mag_pwr2))

    # ∂a_y/∂x, ∂a_y/∂y, ∂a_y/∂z
    jac_bf[1, 0] = factor * (d_term_common_dx - 7.0 * d_term_common_dx * y / pos_mag_pwr2 + 14.0 * x * y / pos_mag_pwr2**2 * (self.c33 * d_term_c33_dy + self.s33 * d_term_s33_dy - 7.0 * y * term_common / pos_mag_pwr2) + self.c33 * (-6.0 * y) + self.s33 * 6.0 * y - 9.0 * x / pos_mag_pwr2 * (self.c33 * d_term_c33_dy + self.s33 * d_term_s33_dy - 7.0 * y * term_common / pos_mag_pwr2))
    jac_bf[1, 1] = factor * (d_term_common_dy - 7.0 * d_term_common_dy * y / pos_mag_pwr2 + 14.0 * y**2 / pos_mag_pwr2**2 * (self.c33 * d_term_c33_dy + self.s33 * d_term_s33_dy - 7.0 * y * term_common / pos_mag_pwr2) + self.s33 * (-6.0 * y) - 9.0 * y / pos_mag_pwr2 * (self.c33 * d_term_c33_dy + self.s33 * d_term_s33_dy - 7.0 * y * term_common / pos_mag_pwr2))
    jac_bf[1, 2] = factor * (-7.0 * z / pos_mag_pwr2 * (self.c33 * d_term_c33_dy + self.s33 * d_term_s33_dy) + 14.0 * y * z**2 / pos_mag_pwr2**2 * term_common - 9.0 * z / pos_mag_pwr2 * (self.c33 * d_term_c33_dy + self.s33 * d_term_s33_dy - 7.0 * y * term_common / pos_mag_pwr2))

    # ∂a_z/∂x, ∂a_z/∂y, ∂a_z/∂z
    jac_bf[2, 0] = factor * (-7.0 * d_term_common_dx * z / pos_mag_pwr2 + 14.0 * x * z**2 / pos_mag_pwr2**2 * term_common - 9.0 * x / pos_mag_pwr2 * (-7.0 * z * term_common / pos_mag_pwr2))
    jac_bf[2, 1] = factor * (-7.0 * d_term_common_dy * z / pos_mag_pwr2 + 14.0 * y * z**2 / pos_mag_pwr2**2 * term_common - 9.0 * y / pos_mag_pwr2 * (-7.0 * z * term_common / pos_mag_pwr2))
    jac_bf[2, 2] = factor * (-7.0 * term_common / pos_mag_pwr2 + 14.0 * z**3 / pos_mag_pwr2**2 * term_common - 9.0 * z / pos_mag_pwr2 * (-7.0 * z * term_common / pos_mag_pwr2))

    # Transform Jacobian back to J2000 frame
    rot_T = rot_mat_j2000_to_iau_earth.T
    j2000_jac = rot_T @ jac_bf @ rot_mat_j2000_to_iau_earth

    return j2000_jac


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
      # Simple cache for SPICE body positions to reduce repeated calls
      self._spice_pos_cache = {}
    
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
      # Cache by rounded time to avoid repeated SPICE calls
      cache_key = (body_name.upper(), frame, round(time_et, 3))
      if cache_key in self._spice_pos_cache:
        return self._spice_pos_cache[cache_key]

      # SPICE state relative to Earth
      state, _ = spice.spkez(
          targ   = self._get_naif_id(body_name),
          et     = time_et,
          ref    = frame,
          abcorr = 'NONE',
          obs    = 399  # relative to Earth
      )
      # SPICE returns km, convert to m
      pos_vec = np.array(state[0:3]) * CONVERTER.M_PER_KM
      self._spice_pos_cache[cache_key] = pos_vec
      return pos_vec
    
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

    def jacobian(
      self,
      time        : float,
      pos_sat_vec : np.ndarray,
    ) -> np.ndarray:
      """
      Compute Jacobian of third-body acceleration with respect to satellite position.

      Returns the 3x3 matrix ∂a_3rd/∂r for all third bodies combined.

      Input:
      ------
        time : float
          Current Ephemeris Time (ET) [s]
        pos_sat_vec : np.ndarray
          Satellite position vector [m]

      Output:
      -------
        jacobian : np.ndarray (3, 3)
          Partial derivative of third-body acceleration w.r.t. position [1/s²]

      Notes:
      ------
        For each third body, the Jacobian is:
          ∂a/∂r = μ * [ I/d³ - 3*ρ⊗ρ/d⁵ ]

        where:
          μ = gravitational parameter of third body
          ρ = r_sat - r_3rd (vector from third body to satellite)
          d = ||ρ|| (distance from satellite to third body)
          I = identity matrix
          ⊗ = outer product

        This formula comes from differentiating the third-body point-mass acceleration:
          a = μ * [ρ/d³ - r_3rd/||r_3rd||³]

        The second term (indirect part) doesn't depend on satellite position, so its
        derivative is zero. Only the direct term contributes to the Jacobian.
      """
      et_seconds = time
      jac_total = np.zeros((3, 3))

      for body in self.bodies:
        body_upper = body.upper()

        # Skip Earth if it's in the list
        if body_upper == 'EARTH':
          continue

        # Get gravitational parameter
        if hasattr(SOLARSYSTEMCONSTANTS, body_upper):
          GP = getattr(SOLARSYSTEMCONSTANTS, body_upper).GP
        else:
          continue

        # Position of Earth to third body [m]
        pos_centbody_to_pertbody_vec = self._get_position_body_spice(body, et_seconds)

        # Position of satellite to third body [m]
        pos_sat_to_pertbody_vec = pos_centbody_to_pertbody_vec - pos_sat_vec
        pos_sat_to_pertbody_mag = np.linalg.norm(pos_sat_to_pertbody_vec)

        # Safety check
        if pos_sat_to_pertbody_mag < 1e6:  # Less than 1000 km (unphysical)
          continue

        # Compute Jacobian for this body
        # ∂a/∂r = μ * [ I/d³ - 3*ρ⊗ρ/d⁵ ]
        d = pos_sat_to_pertbody_mag
        d3 = d**3
        d5 = d**5

        I = np.eye(3)
        rho_outer_rho = np.outer(pos_sat_to_pertbody_vec, pos_sat_to_pertbody_vec)

        jac_body = GP * (I / d3 - 3.0 * rho_outer_rho / d5)

        jac_total += jac_body

      return jac_total


class GeneralRelativity:
  """
  General relativistic corrections to Newtonian gravity.

  Implements the Schwarzschild (point-mass) post-Newtonian correction,
  which is the dominant relativistic effect for satellite orbits.
  """

  def __init__(
    self,
    gp : float,
  ):
    """
    Initialize general relativity model.

    Input:
    ------
      gp : float
        Gravitational parameter of central body [m³/s²]

    Output:
    -------
      None
    """
    self.gp = gp
    self.c  = PHYSICALCONSTANTS.speed_of_light  # Speed of light [m/s]
    self.c2 = self.c**2

  def schwarzschild(
    self,
    pos_vec : np.ndarray,
    vel_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute Schwarzschild (point-mass) relativistic correction.

    This is the first post-Newtonian (1PN) correction to Newtonian gravity,
    accounting for the curvature of spacetime around a massive body.

    The acceleration is:
      a_GR = (μ/c²r³) × [(4μ/r - v²)r + 4(r·v)v]

    where:
      μ   = GM (gravitational parameter)
      c   = speed of light
      r   = position vector, r = |r|
      v   = velocity vector, v² = |v|²
      r·v = dot product of position and velocity

    Input:
    ------
      pos_vec : np.ndarray
        Position vector [m]
      vel_vec : np.ndarray
        Velocity vector [m/s]

    Output:
    -------
      acc_vec : np.ndarray
        Relativistic acceleration correction [m/s²]

    References:
    -----------
      - Moyer, T. D. (2003). Formulation for Observed and Computed Values
        of Deep Space Network Data Types for Navigation. JPL Publication 00-7.
      - Montenbruck & Gill (2000). Satellite Orbits. Springer. Section 3.4.3.
    """
    pos_mag     = np.linalg.norm(pos_vec)
    vel_mag_sqr = np.dot(vel_vec, vel_vec)
    pos_dot_vel = np.dot(pos_vec, vel_vec)

    # Schwarzschild factor: μ/(c²r³)
    factor = self.gp / (self.c2 * pos_mag**3)

    # Terms in the acceleration
    term1 = 4.0 * self.gp / pos_mag - vel_mag_sqr
    term2 = 4.0 * pos_dot_vel

    # a_GR = factor × [term1 * r + term2 * v]
    acc_vec = factor * (term1 * pos_vec + term2 * vel_vec)

    return acc_vec


class SolidEarthTides:
  """
  Solid Earth tide model following IERS 2010 Conventions.

  Computes the time-varying gravitational acceleration due to tidal deformations
  of the Earth caused by the Moon and Sun. The Earth deforms elastically in
  response to tidal forces, changing the gravity field.

  This uses a simplified degree-2 model which captures ~95% of the total effect.
  """

  def __init__(self):
    """
    Initialize solid Earth tide model using constants from SOLARSYSTEMCONSTANTS.

    Input:
    ------
      None

    Output:
    -------
      None
    """
    self.gp           = SOLARSYSTEMCONSTANTS.EARTH.GP
    self.earth_radius = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    self.k2           = SOLARSYSTEMCONSTANTS.EARTH.K2_LOVE
    self.k3           = SOLARSYSTEMCONSTANTS.EARTH.K3_LOVE

  def compute(
    self,
    time_et     : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute solid Earth tide acceleration on satellite.

    Uses degree-2 approximation which is sufficient for most applications.
    The full IERS model includes permanent tide, frequency-dependent Love
    numbers, and higher degrees, but this simplified model captures the
    dominant effect.

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Satellite position vector [m] in J2000 frame

    Output:
    -------
      acc_vec : np.ndarray
        Solid tide acceleration [m/s²] in J2000 frame

    References:
    -----------
      - IERS Conventions 2010, Chapter 6
      - Petit, G., & Luzum, B. (2010). IERS Conventions (2010).
        IERS Technical Note No. 36.
      - Montenbruck & Gill (2000). Satellite Orbits, Section 3.2.5
    """
    # Initialize acceleration vector
    acc_vec = np.zeros(3)

    # Compute satellite position magnitude and direction once (used for all bodies)
    j2000_pos_mag = np.linalg.norm(j2000_pos_vec)
    if j2000_pos_mag == 0:
      return acc_vec
    j2000_pos_dir = j2000_pos_vec / j2000_pos_mag

    # Get positions of tide-generating bodies (Moon and Sun)
    for body_name, body_gp in [('MOON', SOLARSYSTEMCONSTANTS.MOON.GP),
                               ( 'SUN', SOLARSYSTEMCONSTANTS.SUN.GP )]:
      try:
        # Get body position relative to Earth using SPICE
        state, _ = spice.spkez(
          targ   = NAIFIDS.NAME_TO_ID[body_name],
          et     = time_et,
          ref    = 'J2000',
          abcorr = 'NONE',
          obs    = 399  # Earth
        )
        body_pos_vec = np.array(state[0:3]) * CONVERTER.M_PER_KM
      except:
        continue

      # Distance from Earth center to tide-generating body
      body_pos_mag = np.linalg.norm(body_pos_vec)

      if body_pos_mag == 0:
        continue

      # Unit vector toward perturbing body
      body_pos_dir = body_pos_vec / body_pos_mag

      # Degree-2 solid tide acceleration (IERS 2010 simplified)
      # Based on Montenbruck & Gill equation 3.81
      cos_psi = np.dot(j2000_pos_dir, body_pos_dir)
      # Solid tide factor
      factor = (1.5 * self.k2 * body_gp) * (self.earth_radius**5 / (j2000_pos_mag**4 * body_pos_mag**3))

      # Acceleration components (Montenbruck & Gill Eq. 3.81)
      # a = factor * [(5*cos²(ψ) - 1)*ŝ - 2*cos(ψ)*d̂]
      # where ŝ is satellite direction and d̂ is body direction (UNIT VECTORS)
      acc_tide = factor * ((5.0 * cos_psi**2 - 1.0) * j2000_pos_dir - 2.0 * cos_psi * body_pos_dir)

      acc_vec += acc_tide

    return acc_vec


class OceanTides:
  """
  Ocean tide model following IERS 2010 Conventions.

  Computes the time-varying gravitational acceleration due to ocean tides
  caused by the Moon and Sun. Ocean tides represent the redistribution of
  ocean mass in response to tidal forces.

  This uses a simplified degree-2 model. The ocean tide Love number k2
  has opposite sign to solid Earth tides (k2 ≈ -0.31 for oceans vs +0.30 for solid Earth).
  """

  def __init__(self):
    """
    Initialize ocean tide model using constants.

    Input:
    ------
      None

    Output:
    -------
      None
    """
    self.gp           = SOLARSYSTEMCONSTANTS.EARTH.GP
    self.earth_radius = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    # Ocean tide Love numbers from IERS 2010 Conventions
    # k2_ocean is negative (mass deficit below tidal bulge)
    self.k2_ocean     = -0.3075  # Degree-2 ocean tide Love number

  def compute(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute ocean tide acceleration on satellite.

    Uses degree-2 approximation following IERS 2010 simplified model.
    The formula is identical to solid Earth tides but with different Love number.

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Satellite position vector [m] in J2000 frame

    Output:
    -------
      acc_vec : np.ndarray
        Ocean tide acceleration [m/s²] in J2000 frame

    References:
    -----------
      - IERS Conventions 2010, Chapter 6
      - Petit, G., & Luzum, B. (2010). IERS Conventions (2010).
        IERS Technical Note No. 36.
    """
    # Initialize acceleration vector
    acc_vec = np.zeros(3)

    # Compute satellite position magnitude and direction once
    j2000_pos_mag = np.linalg.norm(j2000_pos_vec)
    if j2000_pos_mag == 0:
      return acc_vec
    j2000_pos_dir = j2000_pos_vec / j2000_pos_mag

    # Get positions of tide-generating bodies (Moon and Sun)
    for body_name, body_gp in [('MOON', SOLARSYSTEMCONSTANTS.MOON.GP),
                               ('SUN',  SOLARSYSTEMCONSTANTS.SUN.GP)]:
      try:
        # Get body position relative to Earth using SPICE
        state, _ = spice.spkez(
          targ   = NAIFIDS.NAME_TO_ID[body_name],
          et     = time_et,
          ref    = 'J2000',
          abcorr = 'NONE',
          obs    = 399  # Earth
        )
        body_pos_vec = np.array(state[0:3]) * CONVERTER.M_PER_KM
      except:
        continue

      # Distance from Earth center to tide-generating body
      body_pos_mag = np.linalg.norm(body_pos_vec)

      if body_pos_mag == 0:
        continue

      # Unit vector toward perturbing body
      body_pos_dir = body_pos_vec / body_pos_mag

      # Degree-2 ocean tide acceleration (same formula as solid tides, different k2)
      cos_psi = np.dot(j2000_pos_dir, body_pos_dir)

      # Ocean tide factor (note: k2_ocean is negative)
      factor = (1.5 * self.k2_ocean * body_gp) * (self.earth_radius**5 / (j2000_pos_mag**4 * body_pos_mag**3))

      # Acceleration components
      acc_tide = factor * ((5.0 * cos_psi**2 - 1.0) * j2000_pos_dir - 2.0 * cos_psi * body_pos_dir)

      acc_vec += acc_tide

    return acc_vec


def _get_harmonic_coefficients(
  gravity_harmonics_list : list,
) -> dict:
  """
  Map harmonic names to their coefficient values from constants.
  
  Input:
  ------
    gravity_harmonics_list : list
      List of harmonic names (e.g., ['J2', 'J3', 'C22', 'S22']).
      
  Output:
  -------
    coeffs : dict
      Dictionary mapping parameter names to values.
  """
  coeffs = {
    'j2'  : 0.0,
    'j3'  : 0.0,
    'c21' : 0.0,
    's21' : 0.0,
    'c22' : 0.0,
    's22' : 0.0,
    'c31' : 0.0,
    's31' : 0.0,
    'c32' : 0.0,
    's32' : 0.0,
    'c33' : 0.0,
    's33' : 0.0,
  }
  
  # Map harmonic names to constants
  harmonic_map = {
    'J2'  : ('j2',  SOLARSYSTEMCONSTANTS.EARTH.J2),
    'J3'  : ('j3',  SOLARSYSTEMCONSTANTS.EARTH.J3),
    'C21' : ('c21', SOLARSYSTEMCONSTANTS.EARTH.C21),
    'S21' : ('s21', SOLARSYSTEMCONSTANTS.EARTH.S21),
    'C22' : ('c22', SOLARSYSTEMCONSTANTS.EARTH.C22),
    'S22' : ('s22', SOLARSYSTEMCONSTANTS.EARTH.S22),
    'C31' : ('c31', SOLARSYSTEMCONSTANTS.EARTH.C31),
    'S31' : ('s31', SOLARSYSTEMCONSTANTS.EARTH.S31),
    'C32' : ('c32', SOLARSYSTEMCONSTANTS.EARTH.C32),
    'S32' : ('s32', SOLARSYSTEMCONSTANTS.EARTH.S32),
    'C33' : ('c33', SOLARSYSTEMCONSTANTS.EARTH.C33),
    'S33' : ('s33', SOLARSYSTEMCONSTANTS.EARTH.S33),
  }
  
  for harmonic in gravity_harmonics_list:
    harmonic_upper = harmonic.upper()
    if harmonic_upper in harmonic_map:
      key, value = harmonic_map[harmonic_upper]
      coeffs[key] = value
  
  return coeffs
    

class Gravity:
    """
    Gravitational acceleration coordinator.
    
    Computes gravity as:
        gravity = two_body_point_mass + two_body_oblate + third_body_point_mass + 
                  third_body_oblate (future) + relativity (future)
    
    If a spherical harmonics gravity model is provided, it replaces the 
    two-body point mass and oblateness terms.
    """
    
    def __init__(
      self,
      gravity_config : GravityModelConfig,
    ):
      """
      Initialize gravity acceleration components.
      
      Input:
      ------
        gravity_config : GravityModelConfig
          Gravity model configuration containing:
          - gp: Gravitational parameter of central body [m³/s²]
          - spherical_harmonics: SphericalHarmonicsConfig with degree, order, coefficients, model
          - third_body: ThirdBodyConfig with enabled flag and list of bodies
              
      Output:
      -------
        None
      """
      # Spherical harmonics gravity model (if provided, replaces two-body terms)
      self.spherical_harmonics_model = gravity_config.spherical_harmonics.model

      # Jacobian selection
      use_approx = gravity_config.use_approx_jacobian is True
      use_analytic = gravity_config.use_analytic_jacobian is True
      if use_approx and use_analytic:
        raise ValueError("Only one of use_approx_jacobian or use_analytic_jacobian can be True")
      
      # Default behavior logic
      if not use_approx and not use_analytic:
        # If we have a high-fidelity spherical harmonics model, we prefer the exact analytic Jacobian (Vines)
        # unless specifically told otherwise. 
        # For legacy/simple cases, maybe default to approx? 
        # But generally analytic is preferred if available.
        if self.spherical_harmonics_model is not None:
             use_analytic = True
             use_approx   = False
        else:
             # For simple 2-body, we also have analytic J2
             use_analytic = True
             use_approx   = False

      self.use_approx_jacobian   = use_approx
      self.use_analytic_jacobian = use_analytic
      self.jacobian_approx_eps   = gravity_config.jacobian_approx_eps
      
      # Two-body gravity (used if no spherical harmonics model provided)
      # Get harmonic coefficients from the config's coefficient list
      harmonic_coeffs = _get_harmonic_coefficients(gravity_config.spherical_harmonics.coefficients)
      
      self.two_body = TwoBodyGravity(
        gp      = gravity_config.gp,
        j2      = harmonic_coeffs['j2'],
        j3      = harmonic_coeffs['j3'],
        c21     = harmonic_coeffs['c21'],
        s21     = harmonic_coeffs['s21'],
        c22     = harmonic_coeffs['c22'],
        s22     = harmonic_coeffs['s22'],
        c31     = harmonic_coeffs['c31'],
        s31     = harmonic_coeffs['s31'],
        c32     = harmonic_coeffs['c32'],
        s32     = harmonic_coeffs['s32'],
        c33     = harmonic_coeffs['c33'],
        s33     = harmonic_coeffs['s33'],
        pos_ref = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR,
      )
        
      # Third-body gravity
      self.enable_third_body = gravity_config.third_body.enabled
      if self.enable_third_body:
        self.third_body = ThirdBodyGravity(
          bodies = gravity_config.third_body.bodies,
        )
      else:
        self.third_body = None

      # General relativity
      self.enable_relativity = gravity_config.relativity.enabled
      if self.enable_relativity:
        self.relativity = GeneralRelativity(
          gp = gravity_config.gp,
        )
      else:
        self.relativity = None

      # Solid Earth tides
      self.enable_solid_tides = gravity_config.solid_tides.enabled
      if self.enable_solid_tides:
        self.solid_tides = SolidEarthTides()
      else:
        self.solid_tides = None

      # Ocean tides
      self.enable_ocean_tides = gravity_config.ocean_tides.enabled
      if self.enable_ocean_tides:
        self.ocean_tides = OceanTides()
      else:
        self.ocean_tides = None

    def compute(
      self,
      time        : float,
      pos_vec     : np.ndarray,
      vel_vec     : np.ndarray = None,
      include_stm : bool       = False,
      stm         : np.ndarray = None,
    ):
      """
      Compute total gravity acceleration, optionally with STM time derivative.

      Input:
      ------
        time : float
          Current Ephemeris Time (ET) [s].
        pos_vec : np.ndarray
          Position vector [m].
        vel_vec : np.ndarray, optional
          Velocity vector [m/s] (required if relativity is enabled).
        include_stm : bool
          If True, also compute STM time derivative (default: False).
        stm : np.ndarray (6, 6), optional
          State transition matrix (required if include_stm=True).

      Output:
      -------
        acc_vec : np.ndarray
          Total gravity acceleration [m/s²].
        OR
        (acc_vec, stm_dot) : tuple
          If include_stm=True, returns tuple of acceleration and STM time derivative.
      """
      # Initialize acceleration vector
      acc_vec = np.zeros(3)

      # Use spherical harmonics model if available
      if self.spherical_harmonics_model is not None:
        # Spherical harmonics includes point mass and all harmonic terms
        acc_vec += self.spherical_harmonics_model.compute(time, pos_vec)
      else:
        # Fall back to analytical two-body terms
        # Two-body point mass
        acc_vec += self.two_body.point_mass(pos_vec)

        # Two-body oblateness (J2, J3)
        acc_vec += self.two_body.oblate_j2(time, pos_vec)
        acc_vec += self.two_body.oblate_j3(time, pos_vec)

        # Two-body tesseral (C21, S21, C22, S22)
        acc_vec += self.two_body.tesseral_21(time, pos_vec)
        acc_vec += self.two_body.tesseral_22(time, pos_vec)

        # Two-body tesseral (C31, S31, C32, S32, C33, S33)
        acc_vec += self.two_body.tesseral_31(time, pos_vec)
        acc_vec += self.two_body.tesseral_32(time, pos_vec)
        acc_vec += self.two_body.tesseral_33(time, pos_vec)

      # Third-body contributions
      if self.enable_third_body and self.third_body is not None:
        acc_vec += self.third_body.point_mass(time, pos_vec)

      # General relativity corrections
      if self.enable_relativity and self.relativity is not None:
        if vel_vec is None:
          raise ValueError("vel_vec parameter required when relativity is enabled")
        acc_vec += self.relativity.schwarzschild(pos_vec, vel_vec)

      # Solid Earth tides
      if self.enable_solid_tides and self.solid_tides is not None:
        acc_vec += self.solid_tides.compute(time, pos_vec)

      # Ocean tides
      if self.enable_ocean_tides and self.ocean_tides is not None:
        acc_vec += self.ocean_tides.compute(time, pos_vec)

      # Return just acceleration if STM not requested
      if not include_stm:
        return acc_vec

      # Compute STM time derivative if requested
      if stm is None:
        raise ValueError("stm parameter required when include_stm=True")

      # Compute Jacobian matrix A for STM propagation
      if self.spherical_harmonics_model is not None:
        if self.use_analytic_jacobian:
          # Analytical Jacobian using Spherical Harmonics (Pines implementation)
          daccvec__dposvec = self.spherical_harmonics_model.jacobian(time, pos_vec)
          A_matrix = np.zeros((6, 6))
          A_matrix[0:3, 3:6] = np.eye(3)
          A_matrix[3:6, 0:3] = daccvec__dposvec
        else:
          # Numerical Jacobian from spherical harmonics model
          eps = self.jacobian_approx_eps if self.jacobian_approx_eps is not None else 1.0e-6
          daccvec__dposvec = self.spherical_harmonics_model.jacobian_approx(time, pos_vec, eps=eps)
          A_matrix = np.zeros((6, 6))
          A_matrix[0:3, 3:6] = np.eye(3)
          A_matrix[3:6, 0:3] = daccvec__dposvec
      else:
        # Use analytical Jacobians matching the acceleration model
        A_matrix = self.two_body.point_mass_jacobian(pos_vec)
        A_matrix[3:6, 0:3] += self.two_body.oblate_j2_jacobian(time, pos_vec)

      # Add third-body Jacobian contribution if enabled
      if self.third_body is not None:
        A_matrix[3:6, 0:3] += self.third_body.jacobian(time, pos_vec)

      # STM time derivative: dΦ/dt = A * Φ
      stm_dot = A_matrix @ stm

      return acc_vec, stm_dot
    

# =============================================================================
# Non-Gravitational Accelerations
# =============================================================================


class AtmosphericDrag:
    """
    Atmospheric drag acceleration using exponential atmosphere model with layers.
    Ref: Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
    """
    
    # Exponential atmosphere model coefficients (Vallado, 2013, Table 8-4)
    # Metric units: h_base [km], rho_base [kg/m^3], H [km]
    ATMOSPHERE_LAYERS = [
        (0.0,    1.225,       7.249),
        (25.0,   3.899e-2,    6.349),
        (30.0,   1.774e-2,    6.682),
        (40.0,   3.972e-3,    7.554),
        (50.0,   1.057e-3,    8.382),
        (60.0,   3.206e-4,    7.714),
        (70.0,   8.770e-5,    6.549),
        (80.0,   1.905e-5,    5.799),
        (90.0,   3.396e-6,    5.382),
        (100.0,  5.297e-7,    5.877),
        (110.0,  9.661e-8,    7.263),
        (120.0,  2.438e-8,    9.473),
        (130.0,  8.484e-9,    12.636),
        (140.0,  3.845e-9,    16.149),
        (150.0,  2.070e-9,    22.523),
        (180.0,  5.464e-10,   29.740),
        (200.0,  2.789e-10,   37.105),
        (250.0,  7.248e-11,   45.546),
        (300.0,  2.418e-11,   53.628),
        (350.0,  9.518e-12,   53.298),
        (400.0,  3.725e-12,   58.515),
        (450.0,  1.585e-12,   60.828),
        (500.0,  6.967e-13,   63.822),
        (600.0,  1.454e-13,   71.835),
        (700.0,  3.614e-14,   88.667),
        (800.0,  1.170e-14,   124.64),
        (900.0,  5.245e-15,   181.05),
        (1000.0, 3.019e-15,   268.00)
    ]
    
    def __init__(
      self,
      drag_config : DragConfig,
      mass        : float = 1.0,
    ):
      """
      Initialize drag model
      
      Input:
      ------
        drag_config : DragConfig
          Drag configuration dataclass containing:
          - enabled : bool - Whether drag is enabled
          - cd      : float - Drag coefficient
          - area    : float - Cross-sectional area [m²]
        mass : float
          Spacecraft mass [kg].
              
      Output:
      -------
        None
      """
      self.cd   = drag_config.cd
      self.area = drag_config.area
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
    
    def jacobian(
      self,
      pos_vec : np.ndarray,
      vel_vec : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
      """
      Compute Jacobians of drag acceleration with respect to position and velocity.

      Input:
      ------
        pos_vec : np.ndarray
          Position vector [m]
        vel_vec : np.ndarray
          Velocity vector [m/s]

      Output:
      -------
        dacc__dpos : np.ndarray (3, 3)
          Partial derivative of acceleration w.r.t. position
        dacc__dvel : np.ndarray (3, 3)
          Partial derivative of acceleration w.r.t. velocity
      """
      # Position magnitude and altitude
      r       = np.linalg.norm(pos_vec)
      alt     = r - SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
      
      # Constants
      omega_earth = np.array([0, 0, SOLARSYSTEMCONSTANTS.EARTH.OMEGA])
      
      # Atmospheric density
      rho = self._atmospheric_density(float(alt))
      
      # Look up scale height H for current altitude
      # This is an approximation: assuming H from the static table lookup
      # corresponding to the current altitude layer.
      H = 0.0
      for h_b, rho_b, H_b in self.ATMOSPHERE_LAYERS:
        # Convert h_b from km to m
        if alt >= h_b * 1000.0:
          H = H_b * 1000.0 # Convert km to m
        else:
          break
      if H == 0.0: H = 10000.0 # Fallback 

      # Gradient of density w.r.t. position
      # d(rho)/d(r) = -rho/H * (r_vec / r)
      drho__dpos = -(rho / H) * (pos_vec / r)

      # Relative velocity
      vel_rel_vec = vel_vec - np.cross(omega_earth, pos_vec)
      v_rel       = np.linalg.norm(vel_rel_vec)
      
      if v_rel == 0:
        return np.zeros((3, 3)), np.zeros((3, 3))

      # Drag factor B = 0.5 * Cd * A / m
      B = 0.5 * self.cd * self.area / self.mass

      # d(v_rel_vec) / d(pos_vec) = - skew(omega)
      # skew(omega) = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
      # Since omega = [0, 0, wz]:
      skew_omega = np.array([
        [0.0, -omega_earth[2], 0.0],
        [omega_earth[2], 0.0, 0.0],
        [0.0, 0.0, 0.0]
      ])
      dv_rel_vec__dpos = -skew_omega

      # d(v_rel) / d(pos_vec) = (v_rel_vec^T / v_rel) @ d(v_rel_vec)/d(pos_vec)
      dv_rel__dpos = (vel_rel_vec @ dv_rel_vec__dpos) / v_rel

      # Velocity Jacobian: d(acc) / d(vel)
      # a = -B * rho * v_rel * v_rel_vec
      # da/dv = -B * rho * ( v_rel * I + v_rel_vec * (v_rel_vec^T / v_rel) )
      term_vv = np.outer(vel_rel_vec, vel_rel_vec) / v_rel
      dacc__dvel = -B * rho * (v_rel * np.eye(3) + term_vv)

      # Position Jacobian: d(acc) / d(pos)
      # a = -B * rho * v_rel * v_rel_vec
      # da/dp = -B * [ (d(rho)/dp * v_rel * v_rel_vec) + (rho * d(v_rel)/dp * v_rel_vec) + (rho * v_rel * d(v_rel_vec)/dp) ]
      
      # Term 1: Variation of density
      # result is 3x3 matrix. outer product of (v_rel * v_rel_vec) and drho__dpos
      term1 = np.outer(v_rel * vel_rel_vec, drho__dpos)

      # Term 2: Variation of relative speed magnitude
      # result is 3x3. outer product of v_rel_vec and dv_rel__dpos
      term2 = rho * np.outer(vel_rel_vec, dv_rel__dpos)

      # Term 3: Variation of relative velocity vector direction
      # result is 3x3. rho * v_rel * dv_rel_vec__dpos
      term3 = rho * v_rel * dv_rel_vec__dpos

      dacc__dpos = -B * (term1 + term2 + term3)

      return dacc__dpos, dacc__dvel

    def _atmospheric_density(
      self,
      altitude : float,
    ) -> float:
      """
      Calculate density using layered exponential model.
      
      Input:
      ------
        altitude_m : float
          Altitude above Earth's surface [m]
      
      Output:
      -------
        density : float
          Atmospheric density [kg/m³]
      """
      altitude_km = altitude / 1000.0
      
      # Find the appropriate layer
      # Default to the highest layer if above (or vacuum)
      if altitude_km > 1000.0:
          return 0.0
          
      # Find layer: last layer where h_base <= altitude
      h_base = 0.0
      rho_base = 1.225
      H = 7.249
                 
      for layer in self.ATMOSPHERE_LAYERS:
          if altitude_km >= layer[0]:
              h_base = layer[0]
              rho_base = layer[1]
              H = layer[2]
          else:
              break
              
      # Exponential model for the layer
      # rho = rho_base * exp(-(h - h_base) / H)
      rho = rho_base * np.exp(-(altitude_km - h_base) / H)
      
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
      srp_config : SRPConfig,
      mass       : float = 1.0,
    ):
      """
      Initialize SRP model
      
      Input:
      ------
        srp_config : SRPConfig
          SRP configuration dataclass containing:
          - enabled : bool - Whether SRP is enabled
          - cr      : float - Radiation pressure coefficient (1.0 = absorbing, 2.0 = reflecting)
          - area    : float - Cross-sectional area [m²]
        mass : float
          Spacecraft mass [kg]
              
      Output:
      -------
        None
      """
      self.cr   = srp_config.cr
      self.area = srp_config.area
      self.mass = mass
    
    def compute(
      self,
      time                 : float,
      earth_to_sat_pos_vec : np.ndarray,
      include_stm          : bool       = False,
      stm                  : np.ndarray = None,
    ):
      """
      Compute SRP acceleration, optionally with STM time derivative.

      Input:
      ------
        time : float
          Current Ephemeris Time (ET) [s]
        earth_to_sat_pos_vec : np.ndarray
          Spacecraft position vector relative to Earth [m], i.e. Earth to spacecraft vector.
        include_stm : bool
          If True, also compute STM time derivative (default: False)
        stm : np.ndarray (6, 6), optional
          State transition matrix (required if include_stm=True)

      Output:
      -------
        earth_to_sat_acc_vec : np.ndarray
          SRP acceleration [m/s²]
        OR
        (earth_to_sat_acc_vec, stm_dot) : tuple
          If include_stm=True, returns tuple of acceleration and STM time derivative
      """
      # Check for valid parameters
      if self.area <= 0 or self.mass <= 0:
        if include_stm:
          return np.zeros(3), np.zeros((6, 6))
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
        if include_stm:
          return np.zeros(3), np.zeros((6, 6))
        return np.zeros(3)
      
      # Solar radiation pressure at spacecraft distance
      #   P = P_at_1au * ( 1_au / r_au )^2 = 4.56e-6 N/m² * ( 149597870700 m / r_m )^2
      pressure_srp  = SOLARSYSTEMCONSTANTS.EARTH.PRESSURE_SRP * (CONVERTER.M_PER_AU * CONVERTER.ONE_AU / sat_to_sun_pos_mag)**2
      
      # SRP acceleration magnitude
      acc_mag = (pressure_srp * self.cr * self.area / self.mass) * shadow_factor
      
      # SRP acceleration direction (away from Sun)
      acc_vec = acc_mag * acc_dir

      # Return just acceleration if STM not requested
      if not include_stm:
        return acc_vec

      # Compute STM time derivative if requested
      if stm is None:
        raise ValueError("stm parameter required when include_stm=True")

      # Get SRP Jacobian
      srp_jac = self.jacobian(time, earth_to_sat_pos_vec)

      # Build A matrix for SRP
      A_matrix = np.zeros((6, 6))
      A_matrix[3:6, 0:3] = srp_jac

      # STM time derivative: dΦ/dt = A * Φ
      stm_dot = A_matrix @ stm

      return acc_vec, stm_dot

    def jacobian(
      self,
      time                 : float,
      earth_to_sat_pos_vec : np.ndarray,
    ) -> np.ndarray:
      """
      Compute Jacobian of SRP acceleration with respect to position.

      Returns the 3x3 matrix ∂a_SRP/∂r.

      Input:
      ------
        time : float
          Current Ephemeris Time (ET) [s]
        earth_to_sat_pos_vec : np.ndarray
          Spacecraft position vector relative to Earth [m]

      Output:
      -------
        jacobian : np.ndarray (3, 3)
          Partial derivative of SRP acceleration w.r.t. position [1/s²]

      Notes:
      ------
        This Jacobian assumes the shadow factor ν is constant (i.e., ignores
        the discontinuity at shadow boundaries). This is a standard approximation
        for EKF propagation as shadow transitions are brief and the linearization
        error is small compared to other uncertainties.

        The analytical formula is:
          ∂a/∂r = -K * ν * [ I/r - 3*(ŝ⊗ŝ)/r ]

        where:
          K = P * C_r * A / m  (SRP coefficient)
          ν = shadow factor
          ŝ = unit vector from spacecraft to Sun
          r = distance from spacecraft to Sun
          I = identity matrix
          ⊗ = outer product
      """
      # Check for valid parameters
      if self.area <= 0 or self.mass <= 0:
        return np.zeros((3, 3))

      # Get Sun position relative to Earth
      earth_to_sun_pos_vec = self._get_sun_position(time)

      # Vector from spacecraft to Sun
      sat_to_sun_pos_vec = earth_to_sun_pos_vec - earth_to_sat_pos_vec
      sat_to_sun_pos_mag = np.linalg.norm(sat_to_sun_pos_vec)
      sat_to_sun_pos_dir = sat_to_sun_pos_vec / sat_to_sun_pos_mag

      # Compute shadow factor
      shadow_factor = self._compute_shadow_factor(earth_to_sat_pos_vec, earth_to_sun_pos_vec)

      # If in full shadow, Jacobian is zero
      if shadow_factor == 0.0:
        return np.zeros((3, 3))

      # Solar radiation pressure at spacecraft distance
      pressure_srp = SOLARSYSTEMCONSTANTS.EARTH.PRESSURE_SRP * (CONVERTER.M_PER_AU * CONVERTER.ONE_AU / sat_to_sun_pos_mag)**2

      # SRP coefficient: K = P * C_r * A / m * ν
      K = (pressure_srp * self.cr * self.area / self.mass) * shadow_factor

      # Partial derivative of SRP acceleration w.r.t. position
      # ∂a/∂r = -K * [ I/r - 3*(ŝ⊗ŝ)/r ]
      #
      # Physical interpretation:
      #   - First term (I/r): as spacecraft moves away from Sun, SRP decreases
      #   - Second term (ŝ⊗ŝ): directional dependency (stronger along Sun line)

      I = np.eye(3)
      s_outer_s = np.outer(sat_to_sun_pos_dir, sat_to_sun_pos_dir)

      jacobian = -K * (I / sat_to_sun_pos_mag - 3.0 * s_outer_s / sat_to_sun_pos_mag)

      return jacobian

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
      Compute shadow factor using conical Earth shadow model with penumbra.

      Input:
      ------
        earth_to_sat_pos_vec : np.ndarray
          Spacecraft position vector relative to Earth [m].
        earth_to_sun_pos_vec : np.ndarray
          Sun position vector relative to Earth [m].

      Output:
      -------
        shadow_factor : float
          0.0 = full shadow (umbra), 1.0 = full sunlight, (0,1) = penumbra.

      Notes:
      ------
        Uses a conical shadow model accounting for the Sun's apparent diameter.
        Penumbra is the region of partial shadow where the Sun is partially
        occluded by Earth. This is important for high-precision orbit determination
        of geodetic satellites like LAGEOS-2.

      References:
      -----------
        - Montenbruck & Gill (2000). Satellite Orbits. Springer. Section 3.4.2.
        - Vokrouhlický, D. (1993). A&A 280, 295-304.
      """
      # Earth and Sun radii
      r_earth = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
      r_sun   = SOLARSYSTEMCONSTANTS.SUN.RADIUS.EQUATOR

      # Position magnitudes
      r_sat = np.linalg.norm(earth_to_sat_pos_vec)
      r_sun_mag = np.linalg.norm(earth_to_sun_pos_vec)

      # Unit vector from Earth to Sun
      earth_to_sun_dir = earth_to_sun_pos_vec / r_sun_mag

      # Satellite position projected onto Sun direction
      sat_proj = np.dot(earth_to_sat_pos_vec, earth_to_sun_dir)

      # If satellite is on sunlit side (positive projection), no shadow
      if sat_proj >= 0:
        return 1.0

      # Apparent radii of Sun and Earth as seen from satellite
      # (using small angle approximation for Sun's angular radius)
      sun_angular_radius = np.arcsin(r_sun / (r_sun_mag - sat_proj))
      earth_angular_radius = np.arcsin(r_earth / r_sat)

      # Perpendicular distance from satellite to Sun-Earth line
      sat_perp_vec = earth_to_sat_pos_vec - sat_proj * earth_to_sun_dir
      sat_perp_dist = np.linalg.norm(sat_perp_vec)

      # Angular separation between Earth and Sun centers as seen from satellite
      # This is approximately sat_perp_dist / abs(sat_proj) for small angles
      angular_separation = sat_perp_dist / abs(sat_proj)

      # Determine shadow condition
      # Total shadow (umbra): Earth completely blocks Sun
      if angular_separation <= earth_angular_radius - sun_angular_radius:
        return 0.0

      # Full sunlight: No overlap between Earth and Sun disks
      if angular_separation >= earth_angular_radius + sun_angular_radius:
        return 1.0

      # Partial shadow (penumbra): Sun is partially occluded by Earth
      # Use approximate formula for penumbra shadow factor
      # Based on the fraction of Sun's disk that is visible

      # This is a simplified penumbra model using linear interpolation
      # More sophisticated models use the actual overlapping disk area
      x = angular_separation
      a = earth_angular_radius
      b = sun_angular_radius

      # Linear approximation of shadow factor in penumbra
      # shadow_factor goes from 0 (at umbra boundary) to 1 (at full sunlight)
      shadow_factor = (x - (a - b)) / (2.0 * b)

      # Clamp to [0, 1] range
      shadow_factor = max(0.0, min(1.0, shadow_factor))

      return shadow_factor


# =============================================================================
# Top-Level Coordinator
# =============================================================================

class AccelerationSTMDot:
    """
    Acceleration and STM time-derivative coordinator - orchestrates all acceleration components
    
    Computes total acceleration (velocity time-derivative) as:
      total = gravity + drag + solar_radiation_pressure
    
    where:
      gravity = two_body_point_mass + third_body_point_mass + 
                two_body_oblate (J2, J3) + relativity (future)
    
    Or if a spherical harmonics gravity model is provided:
      gravity = spherical_harmonics_model + third_body_point_mass
    
    Also computes STM time-derivative for EKF orbit determination.
    """
    
    def __init__(
      self,
      gravity_config : GravityModelConfig,
      spacecraft     : SpacecraftProperties,
    ):
      """
      Initialize acceleration coordinator
      
      Input:
      ------
        gravity_config : GravityModelConfig
          Gravity model configuration containing:
          - gp: Gravitational parameter of central body [m³/s²]
          - spherical_harmonics: SphericalHarmonicsConfig with degree, order, coefficients, model
          - third_body: ThirdBodyConfig with enabled flag and list of bodies
        spacecraft : SpacecraftProperties
          Spacecraft properties dataclass containing mass, drag config, and SRP config
              
      Output:
      -------
        None
      """
      # Create gravity component
      self.gravity = Gravity(gravity_config=gravity_config)
      
      self.enable_drag = spacecraft.drag.enabled
      if self.enable_drag and spacecraft.drag.is_valid and spacecraft.mass > 0:
        self.drag = AtmosphericDrag(
          drag_config = spacecraft.drag,
          mass        = spacecraft.mass,
        )
      else:
        self.drag = None
      
      self.enable_srp = spacecraft.srp.enabled
      if self.enable_srp and spacecraft.srp.is_valid and spacecraft.mass > 0:
        self.srp = SolarRadiationPressure(
          srp_config = spacecraft.srp,
          mass       = spacecraft.mass,
        )
      else:
          self.srp = None
    
    def compute(
      self,
      time        : float,
      pos_vec     : np.ndarray,
      vel_vec     : np.ndarray,
      include_stm : bool       = False,
      stm         : np.ndarray = None,
    ):
      """
      Compute total acceleration from all components, optionally with STM time derivative.

      Input:
      ------
        time : float
          Current Ephemeris Time (ET) [s]
        pos_vec : np.ndarray
          Position vector [m]
        vel_vec : np.ndarray
          Velocity vector [m/s]
        include_stm : bool
          If True, also compute STM time derivative (default: False)
        stm : np.ndarray (6, 6), optional
          State transition matrix (required if include_stm=True)

      Output:
      -------
        acc_vec : np.ndarray
          Total acceleration [m/s²]
        OR
        (acc_vec, stm_dot) : tuple
          If include_stm=True, returns tuple of acceleration and STM time derivative
      """
      # Compute gravity acceleration and optionally STM derivative
      if include_stm:
        if stm is None:
          raise ValueError("stm parameter required when include_stm=True")
        acc_vec, stm_dot = self.gravity.compute(time, pos_vec, vel_vec, include_stm=True, stm=stm)
      else:
        acc_vec = self.gravity.compute(time, pos_vec, vel_vec)

      # Atmospheric drag (optional)
      if self.drag is not None:
        acc_vec += self.drag.compute(pos_vec, vel_vec)
        if include_stm:
           drag_jac_pos, drag_jac_vel = self.drag.jacobian(pos_vec, vel_vec)
           # Add drag contributions to STM time derivative
           # d (stm_vel) / dt += d(acc)/d(pos) * stm_pos + d(acc)/d(vel) * stm_vel
           stm_dot[3:6, :] += drag_jac_pos @ stm[0:3, :] + drag_jac_vel @ stm[3:6, :]

      # Solar radiation pressure (optional)
      if self.srp is not None:
        if include_stm:
          acc_srp, stm_dot_srp = self.srp.compute(time, pos_vec, include_stm=True, stm=stm)
          acc_vec += acc_srp
          stm_dot += stm_dot_srp
        else:
          acc_vec += self.srp.compute(time, pos_vec)

      # Return based on whether STM was requested
      if include_stm:
        return acc_vec, stm_dot
      else:
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
    acceleration : AccelerationSTMDot,
  ):
    """
    Initialize equations of motion

    Input:
    ------
      acceleration : AccelerationSTMDot
        Acceleration and STM time-derivative coordinator instance

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

  def state_stm_time_derivative(
      self,
      time_et   : float,
      state_stm : np.ndarray,
  ) -> np.ndarray:
    """
    Compute combined state and STM time derivative for EKF integration

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s past J2000]
      state_stm : np.ndarray (42,)
        Combined state and STM vector

    Output:
    -------
      state_stm_dot : np.ndarray (42,)
        Time derivative of combined state and STM
    """
    # Extract state components
    pos_vec = state_stm[0:3]
    vel_vec = state_stm[3:6]
    stm     = state_stm[6:42].reshape((6, 6))

    # Compute acceleration and STM time derivative
    acc_vec, stm_dot = self.acceleration.compute(time_et, pos_vec, vel_vec, include_stm=True, stm=stm)

    # State time derivative
    state_dot      = np.zeros(6)
    state_dot[0:3] = vel_vec
    state_dot[3:6] = acc_vec

    # Combine
    state_stm_dot       = np.zeros(42)
    state_stm_dot[0:6]  = state_dot
    state_stm_dot[6:42] = stm_dot.flatten()

    return state_stm_dot




