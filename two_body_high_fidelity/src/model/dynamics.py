"""
Spacecraft Orbital Dynamics Module
==================================

High-fidelity orbital dynamics models for spacecraft trajectory propagation.

Summary:
--------
This module provides a comprehensive framework for modeling spacecraft orbital dynamics,
including gravitational forces (two-body and third-body), atmospheric drag, and solar 
radiation pressure. It features a hierarchical acceleration model architecture, orbital 
element conversions, anomaly transformations, and specialized solvers for Kepler's equation 
and Lambert's problem. The module supports all orbit types (circular, elliptical, parabolic, 
hyperbolic, and rectilinear) and integrates with SPICE for high-accuracy ephemerides.

Class Structure:
----------------
Acceleration Hierarchy:
    Acceleration (coordinator)
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
    └── SolarRadiationPressure (future)

Main Components:
----------------
1. **Acceleration** - Top-level coordinator that computes:
   total = gravity + drag + solar_radiation_pressure

2. **Gravity** - Gravitational acceleration coordinator with methods:
   - two_body_point_mass()
   - two_body_oblate()
   - third_body_point_mass()
   - third_body_oblate() (future)
   - relativity() (future)

3. **TwoBodyGravity** - Central body gravity:
   - point_mass() - Keplerian two-body
   - oblate_j2() - J2 oblateness
   - oblate_j3() - J3 oblateness
   - oblate_j4() - J4 oblateness

4. **ThirdBodyGravity** - Perturbations from Sun, Moon, etc.:
   - point_mass() - Third-body point mass
   - SPICE ephemerides or analytical approximations

5. **AtmosphericDrag** - Atmospheric drag model:
   - Exponential density model
   - Rotating atmosphere

6. **SolarRadiationPressure** - SRP model (future)

Utility Classes:
----------------
- GeneralStateEquationsOfMotion - ODE integration interface
- TwoBody_RootSolvers - Kepler's equation, Lambert's problem
- CoordinateSystemConverter - Position/velocity ↔ orbital elements

Usage Example:
--------------
  from src.model.dynamics import Acceleration, GeneralStateEquationsOfMotion
  from src.model.constants import PHYSICALCONSTANTS
  
  # Initialize acceleration model
  acceleration = Acceleration(
      gp                = PHYSICALCONSTANTS.EARTH.GP,
      j2                = PHYSICALCONSTANTS.EARTH.J2,
      pos_ref           = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR,
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

import numpy as np
from pathlib import Path
from typing import Optional
import warnings

from src.model.constants import PHYSICALCONSTANTS, CONVERTER


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
        pos_ref : float = 0.0,
    ):
        """
        Initialize two-body gravity model
        
        Input:
        ------
        gp : float
            Gravitational parameter of central body [m³/s²]
        j2, j3, j4 : float
            Harmonic coefficients for oblateness
        pos_ref : float
            Reference radius for harmonic coefficients [m]
        """
        self.gp      = gp
        self.pos_ref = pos_ref
        self.j2      = j2
        self.j3      = j3
        self.j4      = j4
    
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
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """
        J2 oblateness perturbation
        
        Input:
        ------
        pos_vec : np.ndarray
            Position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            Acceleration vector [m/s²]
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
    
    def oblate_j3(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """
        J3 oblateness perturbation
        
        Input:
        ------
        pos_vec : np.ndarray
            Position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            Acceleration vector [m/s²]
        """
        if self.j3 == 0.0:
            return np.zeros(3)
        
        x, y, z = pos_vec[0], pos_vec[1], pos_vec[2]
        pos_mag = np.linalg.norm(pos_vec)
        posmag2 = pos_mag**2
        posmag7 = posmag2 * posmag2 * posmag2 * pos_mag
        
        factor = 2.5 * self.j3 * self.gp * self.pos_ref**3 / posmag7
        
        acc_vec    = np.zeros(3)
        acc_vec[0] = factor * x * z * (3 - 7 * z**2 / posmag2)
        acc_vec[1] = factor * y * z * (3 - 7 * z**2 / posmag2)
        acc_vec[2] = factor * (3 * z**2 - 7 * z**4 / posmag2 - 0.6 * posmag2)
        
        return acc_vec
    
    def oblate_j4(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """
        J4 oblateness perturbation
        
        Input:
        ------
        pos_vec : np.ndarray
            Position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            Acceleration vector [m/s²]
        """
        if self.j4 == 0.0:
            return np.zeros(3)
        
        x, y, z = pos_vec[0], pos_vec[1], pos_vec[2]
        
        pos_mag      = np.linalg.norm(pos_vec)
        pos_mag_pwr2 = pos_mag**2
        pos_mag_pwr9 = pos_mag_pwr2**4 * pos_mag
        
        z2_r2  = z**2 / pos_mag_pwr2
        factor = 1.875 * self.j4 * self.gp * self.pos_ref**4 / pos_mag_pwr9
        
        acc_vec    = np.zeros(3)
        acc_vec[0] = factor * x * (1 - 14 * z2_r2 + 21 * z2_r2**2)
        acc_vec[1] = factor * y * (1 - 14 * z2_r2 + 21 * z2_r2**2)
        acc_vec[2] = factor * z * (5 - 70 * z2_r2 / 3 + 21 * z2_r2**2)
        
        return acc_vec
    

class ThirdBodyGravity:
    """
    Third-body gravitational perturbations from Sun, Moon, and other bodies
    Uses SPICE ephemerides or analytical approximations
    """
    
    def __init__(
        self,
        et_offset               : float = 0.0,
        use_spice               : bool  = True,
        bodies                  : list  = None,
        spice_kernel_folderpath : str   = None,
    ):
        """
        Initialize third-body gravity model
        
        Input:
        ------
        et_offset : float
            Offset to convert integrator time to ET: et = et_offset + time [s]
        use_spice : bool
            Use SPICE ephemerides (True) or analytical approximations (False)
        bodies : list of str
            Which bodies to include (default: ['sun', 'moon'])
        spice_kernel_folderpath : str
            Path to SPICE kernel folderpath
        """
        self.et_offset = et_offset
        self.time_o    = et_offset
        self.bodies    = bodies if bodies else ['sun', 'moon']
        self.use_spice = use_spice
        
        if use_spice:
            self._load_spice_kernels(spice_kernel_folderpath)
    
    def _load_spice_kernels(
        self,
        kernel_folderpath : Optional[Path],
    ) -> None:
        """
        Load required SPICE kernels
        
        Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
        
        Required kernels:
        - LSK (Leap Second Kernel): naif0012.tls
        - SPK (Planetary Ephemeris): de430.bsp or de440.bsp
        - PCK (Planetary Constants): pck00010.tpc
        """
        import spiceypy as spice

        if kernel_folderpath is None:
            # Default to a kernels folderpath in the project
            kernel_folderpath = Path(__file__).parent.parent.parent / 'data' / 'spice_kernels'
        
        kernel_folderpath = Path(kernel_folderpath)
        
        if not kernel_folderpath.exists():
            raise FileNotFoundError(
                f"SPICE kernel folderpath not found: {kernel_folderpath}\n"
                f"Please download kernels from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/\n"
                f"Required files:\n"
                f"  - lsk/naif0012.tls\n"
                f"  - spk/planets/de440.bsp (or de430.bsp)\n"
                f"  - pck/pck00010.tpc"
            )
        
        # Load planetary ephemeris
        spk_files = list(kernel_folderpath.glob('de*.bsp'))
        if spk_files:
            spice.furnsh(str(spk_files[0]))  # Use first found
        else:
            raise FileNotFoundError(f"No SPK files (de*.bsp) found in {kernel_folderpath}")
        
        # Load planetary constants
        pck_file = kernel_folderpath / 'pck00010.tpc'
        if pck_file.exists():
            spice.furnsh(str(pck_file))
        else:
            raise FileNotFoundError(f"PCK file not found: {pck_file}")
    
    def _get_position_body_spice(
        self,
        body_name  : str,
        et_seconds : float,
        frame      : str = 'J2000',
    ) -> np.ndarray:
        """
        Get position of celestial body at given time
        
        Input:
        ------
        body_name : str
            'SUN' or 'MOON'
        et_seconds : float
            Ephemeris time in seconds past J2000 epoch
        frame : str
            Reference frame (default: 'J2000')
        
        Output:
        -------
        pos_vec : np.ndarray (3,)
            Position vector [m]
        """
        if self.use_spice:
            import spiceypy as spice
            # SPICE state relative to Earth
            state, _ = spice.spkez(
                targ   = self._get_naif_id(body_name),
                et     = et_seconds,
                ref    = frame,
                abcorr = 'NONE',
                obs    = 399  # relative to Earth
            )
            # SPICE returns km, convert to m
            return np.array(state[0:3]) * CONVERTER.M_PER_KM
        else:
            # Use analytical approximation (returns km, convert to m)
            return self._get_position_body_analytical(body_name, et_seconds) * CONVERTER.M_PER_KM
    
    def _get_naif_id(
        self,
        body_name : str,
    ) -> int:
        """
        Get NAIF ID for body
        
        Input:
        ------
        body_name : str
            Body name ('SUN' or 'MOON')
        
        Output:
        -------
        naif_id : int
            NAIF ID code
        """
        naif_ids = {
            'SUN'  : 10,
            'MOON' : 301,
        }
        return naif_ids[body_name.upper()]

    def _get_position_body_analytical(
        self,
        body_name  : str,
        et_seconds : float,
    ) -> np.ndarray:
        """
        Simple analytical approximation for Sun/Moon position
        Lower accuracy (~1000 km for Moon, ~10,000 km for Sun)
        Good enough for rough estimates

        Input:
        ------
        body_name : str
            'SUN' or 'MOON'
        et_seconds : float
            Ephemeris time in seconds past J2000 epoch

        Output:
        -------
        pos_vec : np.ndarray (3,)
            Position vector [km]  # Note: returns km (caller converts to m)
        """
        # Convert to Julian centuries from J2000
        T = et_seconds / (86400.0 * 36525.0)
        
        if body_name.upper() == 'SUN':
            # Very simplified Sun position (ecliptic plane approximation)
            # Mean longitude
            L = np.radians(280.460 + 36000.771 * T)
            # Mean anomaly
            g = np.radians(357.528 + 35999.050 * T)
            # Ecliptic longitude
            lambda_sun = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2*g)
            
            # Distance (AU to km)
            r_sun = 149597870.7 * (1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2*g))
            
            # Ecliptic to equatorial (simple rotation)
            epsilon = np.radians(23.439)  # Obliquity
            
            x = r_sun * np.cos(lambda_sun)
            y = r_sun * np.sin(lambda_sun) * np.cos(epsilon)
            z = r_sun * np.sin(lambda_sun) * np.sin(epsilon)
            
            return np.array([x, y, z])
        
        elif body_name.upper() == 'MOON':
            # Very simplified Moon position
            # Mean longitude
            L = np.radians(218.316 + 481267.881 * T)
            # Mean anomaly
            M = np.radians(134.963 + 477198.868 * T)
            # Mean distance of Moon from ascending node
            F = np.radians(93.272 + 483202.018 * T)
            
            # Longitude
            lambda_moon = L + np.radians(6.289) * np.sin(M)
            # Latitude
            beta = np.radians(5.128) * np.sin(F)
            # Distance
            r_moon = 385000.0 - 20905.0 * np.cos(M)
            
            # Ecliptic to equatorial
            epsilon = np.radians(23.439)
            
            x = r_moon * np.cos(beta) * np.cos(lambda_moon)
            y = r_moon * np.cos(beta) * np.sin(lambda_moon) * np.cos(epsilon) - np.sin(beta) * np.sin(epsilon)
            z = r_moon * np.cos(beta) * np.sin(lambda_moon) * np.sin(epsilon) + np.sin(beta) * np.cos(epsilon)
            
            return np.array([x, y, z])
        
        else:
            raise ValueError(f"Unknown body: {body_name}")
    
    def point_mass(
        self,
        time        : float,
        pos_sat_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Third-body point mass perturbations (Sun, Moon)
        
        Input:
        ------
        time : float
            Current time [s]
        pos_sat_vec : np.ndarray
            Satellite position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            Third-body acceleration [m/s²]
        """
        # Ephemeris time is seconds from J2000 epoch
        et_seconds = self.et_offset + time
        
        # Compute acceleration for all bodies
        acc_vec = np.zeros(3)
        for body in self.bodies:

            # Get gravitational parameter [m³/s²]
            if body.upper() == 'SUN':
                GP = PHYSICALCONSTANTS.SUN.GP
            elif body.upper() == 'MOON':
                GP = PHYSICALCONSTANTS.MOON.GP
            else:
                continue

            # Position of central body (Earth) to perturbing body [m]
            pos_centbody_to_pertbody_vec = self._get_position_body_spice(body, et_seconds)
            pos_centbody_to_pertbody_mag = np.linalg.norm(pos_centbody_to_pertbody_vec)
            
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
    Gravitational acceleration coordinator
    
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
        pos_ref                 : float = 0.0,
        enable_third_body       : bool  = False,
        et_offset               : float = 0.0,
        third_body_use_spice    : bool  = True,
        third_body_bodies       : list  = None,
        spice_kernel_folderpath : str   = None,
    ):
        """
        Initialize gravity acceleration components
        
        Input:
        ------
        gp : float
            Gravitational parameter of central body [m³/s²]
        j2, j3, j4 : float
            Harmonic coefficients for oblateness
        pos_ref : float
            Reference radius for harmonic coefficients [m]
        enable_third_body : bool
            Enable Sun/Moon gravitational perturbations
        et_offset : float
            Offset to convert integrator time to ET: et = et_offset + time [s]
        third_body_use_spice : bool
            Use SPICE ephemerides (True) or analytical approximations (False)
        third_body_bodies : list of str
            Which bodies to include (default: ['sun', 'moon'])
        spice_kernel_folderpath : str
            Path to SPICE kernel folderpath
        """
        # Two-body gravity
        self.two_body = TwoBodyGravity(
            gp      = gp,
            j2      = j2,
            j3      = j3,
            j4      = j4,
            pos_ref = pos_ref,
        )
        
        # Third-body gravity
        self.enable_third_body = enable_third_body
        if self.enable_third_body:
            self.third_body = ThirdBodyGravity(
                et_offset               = et_offset,
                use_spice               = third_body_use_spice,
                bodies                  = third_body_bodies,
                spice_kernel_folderpath = spice_kernel_folderpath,
            )
        else:
            self.third_body = None
    
    def compute(
        self,
        time    : float,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Compute total gravity acceleration
        
        Input:
        ------
        time : float
            Current time [s]
        pos_vec : np.ndarray
            Position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            Total gravity acceleration [m/s²]
        """
        # Initialize acceleration vector
        acc_vec = np.zeros(3)
        
        # Two-body contributions
        acc_vec += self.two_body_point_mass(pos_vec)
        acc_vec += self.two_body_oblate(pos_vec)
        
        # Third-body contributions
        if self.enable_third_body:
            acc_vec += self.third_body_point_mass(time, pos_vec)
        
        # Future: third_body_oblate, relativity
        
        return acc_vec
    
    def two_body_point_mass(
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
        return self.two_body.point_mass(pos_vec)
    
    def two_body_oblate(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Two-body oblateness (J2, J3, J4)
        
        Input:
        ------
        pos_vec : np.ndarray
            Position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            Acceleration vector [m/s²]
        """
        acc_vec  = self.two_body.oblate_j2(pos_vec)
        acc_vec += self.two_body.oblate_j3(pos_vec)
        acc_vec += self.two_body.oblate_j4(pos_vec)
        return acc_vec
    
    def third_body_point_mass(
        self,
        time    : float,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Third-body point mass perturbations
        
        Input:
        ------
        time : float
            Current time [s]
        pos_vec : np.ndarray
            Position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            Acceleration vector [m/s²]
        """
        if self.third_body is None:
            return np.zeros(3)
        return self.third_body.point_mass(time, pos_vec)
    
    def third_body_oblate(
        self,
        time    : float,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Third-body oblateness perturbations (future implementation)
        
        Input:
        ------
        time : float
            Current time [s]
        pos_vec : np.ndarray
            Position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            Acceleration vector [m/s²]
        """
        # TODO: Implement third-body oblateness
        return np.zeros(3)
    
    def relativity(
        self,
        pos_vec : np.ndarray,
        vel_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Relativistic corrections (future implementation)
        
        Input:
        ------
        pos_vec : np.ndarray
            Position vector [m]
        vel_vec : np.ndarray
            Velocity vector [m/s]
        
        Output:
        -------
        acc_vec : np.ndarray
            Acceleration vector [m/s²]
        """
        # TODO: Implement post-Newtonian corrections
        return np.zeros(3)


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
            Drag coefficient
        area : float
            Cross-sectional area [m²]
        mass : float
            Spacecraft mass [kg]
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
        alt     = pos_mag - PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR
        
        # Atmospheric density at current altitude
        rho = self._atmospheric_density(float(alt))
        
        # Velocity relative to rotating atmosphere
        omega_earth = np.array([0, 0, PHYSICALCONSTANTS.EARTH.OMEGA])
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
        rho = PHYSICALCONSTANTS.EARTH.RHO_0 * np.exp(-altitude / PHYSICALCONSTANTS.EARTH.H_0)
        
        return rho


class SolarRadiationPressure:
    """
    Solar radiation pressure acceleration (placeholder for future implementation)
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
            Radiation pressure coefficient
        area : float
            Cross-sectional area [m²]
        mass : float
            Spacecraft mass [kg]
        """
        self.cr   = cr
        self.area = area
        self.mass = mass
    
    def compute(
        self,
        time    : float,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Compute SRP acceleration (not yet implemented)
        
        Input:
        ------
        time : float
            Current time [s]
        pos_vec : np.ndarray
            Position vector [m]
        
        Output:
        -------
        acc_vec : np.ndarray
            SRP acceleration [m/s²]
        """
        # TODO: Implement solar radiation pressure
        return np.zeros(3)


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
        time_et_o               : float = 0.0,
        time_o                  : float = 0.0,
        j2                      : float = 0.0,
        j3                      : float = 0.0,
        j4                      : float = 0.0,
        pos_ref                 : float = 0.0,
        mass                    : float = 1.0,
        enable_drag             : bool  = False,
        cd                      : float = 0.0,
        area_drag               : float = 0.0,
        enable_third_body       : bool  = False,
        third_body_use_spice    : bool  = True,
        third_body_bodies       : list  = None,
        spice_kernel_folderpath : str   = None,
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
        time_et_o : float
            Ephemeris Time (ET) seconds from J2000 at the initial time [s]
        time_o : float
            Initial time in integrator's time system [s]
        j2, j3, j4 : float
            Harmonic coefficients for oblateness
        pos_ref : float
            Reference radius for harmonic coefficients [m]
        cd : float
            Drag coefficient
        area_drag : float
            Cross-sectional area [m²]
        mass : float
            Spacecraft mass [kg]
        enable_drag : bool
            Enable atmospheric drag
        enable_third_body : bool
            Enable Sun/Moon gravitational perturbations
        third_body_use_spice : bool
            Use SPICE ephemerides (True) or analytical approximations (False)
        third_body_bodies : list of str
            Which bodies to include (default: ['sun', 'moon'])
        spice_kernel_folderpath : str
            Path to SPICE kernel folderpath
        enable_srp : bool
            Enable solar radiation pressure (not yet implemented)
        cr : float
            Radiation pressure coefficient
        area_srp : float
            Cross-sectional area for SRP [m²]
        """
        # Compute ET offset internally
        et_offset = time_et_o - time_o
        
        # Create acceleration component instances
        self.gravity = Gravity(
            gp                      = gp,
            j2                      = j2,
            j3                      = j3,
            j4                      = j4,
            pos_ref                 = pos_ref,
            enable_third_body       = enable_third_body,
            et_offset               = et_offset,
            third_body_use_spice    = third_body_use_spice,
            third_body_bodies       = third_body_bodies,
            spice_kernel_folderpath = spice_kernel_folderpath,
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
            Current time [s]
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
            Current time [s]
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


# =============================================================================
# Utility Classes
# =============================================================================

class TwoBody_RootSolvers:
    """
    Root solvers for two-body orbital mechanics
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
    Conversion between position/velocity and classical orbital elements
    
    Summary:
    --------
    Provides comprehensive conversion utilities for orbital mechanics, including:
    - Cartesian state (position/velocity) ↔ classical orbital elements
    - Anomaly transformations (true, eccentric, mean, hyperbolic, parabolic)
    - Support for all orbit types (circular, elliptical, parabolic, hyperbolic, rectilinear)
    
    Key Methods:
    ------------
    - pv_to_coe() : Convert position/velocity to orbital elements
    - coe_to_pv() : Convert orbital elements to position/velocity
    - Anomaly conversions: ea_to_ta, ta_to_ea, ma_to_ea, ea_to_ma, etc.
    """
    
    @staticmethod
    def pv_to_coe(
      pos_vec : np.ndarray,
      vel_vec : np.ndarray,
      gp      : float = PHYSICALCONSTANTS.EARTH.GP,
    ) -> dict:
      """
      Convert Cartesian position and velocity vectors to classical orbital elements.
      
      Input:
      ------
      pos_vec : np.ndarray
          Position vector [m]
      vel_vec : np.ndarray
          Velocity vector [m/s]
      gp : float
          Gravitational parameter [m³/s²]
          
      Output:
      -------
      dict : Dictionary containing orbital elements:
        sma  : semi-major axis [m] (or np.inf for parabolic orbits)
        ecc  : eccentricity [-]
        inc  : inclination [rad]
        raan : right ascension of the ascending node [rad]
        argp : argument of periapsis [rad]
        ma   : mean anomaly [rad] (None for rectilinear or parabolic)
        ta   : true anomaly [rad] (None for rectilinear)
        ea   : eccentric anomaly [rad] (None for hyperbolic/parabolic/non-rectilinear-elliptic)
        ha   : hyperbolic anomaly [rad] (None for elliptic/parabolic/non-rectilinear-hyperbolic)
        pa   : parabolic anomaly [rad] (None for elliptic/hyperbolic)

      Notes:
      ------      
      - Handles circular, elliptical, parabolic, hyperbolic, and rectilinear orbits
        - circular:       e = 0           a > 0
        - elliptical-2D:  0 < e < 1       a > 0
        - elliptical-1D:  e = 1           a > 0 (rectilinear)
        - parabolic:      e = 1           a = inf
        - hyperbolic:     e > 1           a < 0
      - For rectilinear motion:
          * Elliptic   : returns ea (eccentric anomaly)
          * Hyperbolic : returns ha (hyperbolic anomaly)
      - For non-rectilinear motion:
          * Elliptic   : returns ta, ea, ma
          * Hyperbolic : returns ta, ha, ma
          * Parabolic  : returns ta, pa, ma
      - For the circular case, the ascending node (AN) and argument of periapsis (AP) 
        are ill-defined, along with the associated eccentricity direction vector (ie) 
        and periapsis direction vector (ip) of the perifocal frame. In this circular 
        orbit case, the unit vector ie is set equal to the normalized inertial 
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
      argp = np.arctan2(ecc_dir[2], periapsis_dir[2])
      
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
        'argp' : argp,
        'ma'   : ma,
        'ta'   : ta,
        'ea'   : ea,
        'ha'   : ha,
        'pa'   : pa,
      }


    @staticmethod
    def coe_to_pv(
        coe : dict,
        gp  : float = PHYSICALCONSTANTS.EARTH.GP,
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
          argp : argument of periapsis [rad]
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
        argp = coe['argp']

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
            np.cos(raan) * np.cos(argp) - np.sin(raan) * np.sin(argp) * np.cos(inc),
            np.sin(raan) * np.cos(argp) + np.cos(raan) * np.sin(argp) * np.cos(inc),
            np.sin(argp) * np.sin(inc)
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
          theta       = argp + ta                     # true latitude angle
          ang_mom_mag = np.sqrt(gp * slr)             # orbit angular momentum magnitude

          # Position vector
          pos_vec = np.array([
            pos_mag * (np.cos(raan) * np.cos(theta) - np.sin(raan) * np.sin(theta) * np.cos(inc)),
            pos_mag * (np.sin(raan) * np.cos(theta) + np.cos(raan) * np.sin(theta) * np.cos(inc)),
            pos_mag * (                                              np.sin(theta) * np.sin(inc))
          ])
          
          # Velocity vector
          vel_vec = np.array([
            -gp / ang_mom_mag * (np.cos(raan) * (np.sin(theta) + ecc * np.sin(argp)) + np.sin(raan) * (np.cos(theta) + ecc * np.cos(argp)) * np.cos(inc)),
            -gp / ang_mom_mag * (np.sin(raan) * (np.sin(theta) + ecc * np.sin(argp)) - np.cos(raan) * (np.cos(theta) + ecc * np.cos(argp)) * np.cos(inc)),
            -gp / ang_mom_mag * (                                                                    -(np.cos(theta) + ecc * np.cos(argp)) * np.sin(inc))
          ])
        
        return pos_vec, vel_vec

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
      ta  : float
        True anomaly [rad]
      ecc : float
        Eccentricity (ecc > 1)

      Output:
      -------
      ha : float
          Hyperbolic anomaly [rad]

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
          Eccentricity (e > 1)
      
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
      max_iter : int = 200,
    ) -> float:
      """
      Maps mean anomaly to eccentric anomaly using Newton-Raphson iteration for both 2D and 1D elliptic orbits.
      
      Alias for TwoBody_RootSolvers.kepler().
      
      Input:
      ------
      ma : float
          Mean anomaly [rad]
      ecc : float
          Eccentricity (0 <= e < 1)
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
      ma : float
          Mean hyperbolic anomaly [rad]
      ecc : float
          Eccentricity (e > 1)
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




