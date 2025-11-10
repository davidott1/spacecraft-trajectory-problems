"""
Spacecraft Orbital Dynamics Module
==================================

High-fidelity orbital dynamics models for spacecraft trajectory propagation.

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
        gp      = PHYSICALCONSTANTS.EARTH.GP,
        j2      = PHYSICALCONSTANTS.EARTH.J2,
        pos_ref = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR,
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
- Position: meters [m]
- Velocity: meters per second [m/s]
- Acceleration: meters per second squared [m/s²]
- Time: seconds [s]
- Angles: radians [rad]

Notes:
------
- All calculations performed in inertial J2000 frame
- Third-body positions from SPICE kernels or analytical approximations
- Atmospheric model accounts for Earth rotation
"""

import numpy as np
import spiceypy as spice
from pathlib import Path
from typing import Optional

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
        time_o                  : float = 0.0,
        use_spice               : bool  = True,
        bodies                  : list  = None,
        spice_kernel_folderpath : str   = None,
    ):
        """
        Initialize third-body gravity model
        
        Input:
        ------
        time_o : float
            Initial epoch time [s from J2000]
        use_spice : bool
            Use SPICE ephemerides (True) or analytical approximations (False)
        bodies : list of str
            Which bodies to include (default: ['sun', 'moon'])
        spice_kernel_folderpath : str
            Path to SPICE kernel directory
        """
        self.time_o    = time_o
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
        if kernel_folderpath is None:
            # Default to a kernels directory in the project
            kernel_folderpath = Path(__file__).parent.parent.parent / 'data' / 'spice_kernels'
        
        kernel_folderpath = Path(kernel_folderpath)
        
        if not kernel_folderpath.exists():
            raise FileNotFoundError(
                f"SPICE kernel directory not found: {kernel_folderpath}\n"
                f"Please download kernels from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/\n"
                f"Required files:\n"
                f"  - lsk/naif0012.tls\n"
                f"  - spk/planets/de440.bsp (or de430.bsp)\n"
                f"  - pck/pck00010.tpc"
            )
        
        # Load leap second kernel
        lsk_file = kernel_folderpath / 'naif0012.tls'
        if lsk_file.exists():
            spice.furnsh(str(lsk_file))
        else:
            raise FileNotFoundError(f"LSK file not found: {lsk_file}")
        
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
        
        print(f"SPICE kernels loaded from: {kernel_folderpath}")
    
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
            y = r_moon * (np.cos(beta) * np.sin(lambda_moon) * np.cos(epsilon) - 
                          np.sin(beta) * np.sin(epsilon))
            z = r_moon * (np.cos(beta) * np.sin(lambda_moon) * np.sin(epsilon) + 
                          np.sin(beta) * np.cos(epsilon))
            
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
        et_seconds = self.time_o + time
        
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
    
    def __del__(self):
        """
        Unload SPICE kernels on cleanup
        """
        if self.use_spice:
            try:
                spice.kclear()
            except:
                pass


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
        time_o                  : float = 0.0,
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
        time_o : float
            Initial epoch time [s]
        third_body_use_spice : bool
            Use SPICE ephemerides (True) or analytical approximations (False)
        third_body_bodies : list of str
            Which bodies to include (default: ['sun', 'moon'])
        spice_kernel_folderpath : str
            Path to SPICE kernel directory
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
                time_o                  = time_o,
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
        time_o : float
            Initial epoch time [s]
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
            Path to SPICE kernel directory
        enable_srp : bool
            Enable solar radiation pressure (not yet implemented)
        cr : float
            Radiation pressure coefficient
        area_srp : float
            Cross-sectional area for SRP [m²]
        """
        # Create acceleration component instances
        self.gravity = Gravity(
            gp                      = gp,
            j2                      = j2,
            j3                      = j3,
            j4                      = j4,
            pos_ref                 = pos_ref,
            enable_third_body       = enable_third_body,
            time_o                  = time_o,
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
        
        for _ in range(max_iter):
            func       = ea - ecc * np.sin(ea) - ma
            func_prime = 1 - ecc * np.cos(ea)
            ea_new = ea - func / func_prime
            
            if abs(ea_new - ea) < tol:
                return ea_new
            ea = ea_new
        
        return ea  # Return best estimate if not converged
    
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
        
        # Semi-perimeter
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


class CoordinateSystemConverter:
    """
    Conversion between position/velocity and classical orbital elements
    """
    
    def __init__(
        self,
        gp: float = PHYSICALCONSTANTS.EARTH.GP,
    ):
        """
        Initialize converter
        
        Input:
        ------
        gp : float
            Gravitational parameter [m³/s²]
        """
        self.gp = gp
    
    def pv_to_coe(
        self,
        pos_vec: np.ndarray,
        vel_vec: np.ndarray,
    ) -> dict:
        """
        Convert position and velocity vectors to classical orbital elements.
        
        Input:
        ------
        pos_vec : np.ndarray
            Position vector [m]
        vel_vec : np.ndarray
            Velocity vector [m/s]
        
        Output:
        -------
        dict : Dictionary containing orbital elements:
            sma  : semi-major axis [m]
            ecc  : eccentricity
            inc  : inclination [rad]
            raan : right ascension of ascending node [rad]
            argp : argument of perigee [rad]
            ma   : mean anomaly [rad]
            ta   : true anomaly [rad]
            ea   : eccentric anomaly [rad]
        """
        # Convert inputs to numpy arrays
        pos_vec = np.array(pos_vec)
        vel_vec = np.array(vel_vec)
        
        # Specific angular momentum
        ang_mom_vec = np.cross(pos_vec, vel_vec)
        ang_mom_mag = np.linalg.norm(ang_mom_vec)
        
        # Eccentricity vector
        pos_mag = np.linalg.norm(pos_vec)
        ecc_vec = np.cross(vel_vec, ang_mom_vec) / self.gp - pos_vec / pos_mag
        ecc_mag = np.linalg.norm(ecc_vec)
        
        # Semi-major axis
        vel_mag     = np.linalg.norm(vel_vec)
        spec_energy = vel_mag**2 / 2 - self.gp / pos_mag
        sma         = -self.gp / (2 * spec_energy)
        
        # Inclination
        inc = np.arccos(ang_mom_vec[2] / ang_mom_mag)
        
        # Node vector
        node_vec = np.cross([0, 0, 1], ang_mom_vec)
        node_mag = np.linalg.norm(node_vec)
        
        # RAAN
        if node_mag - 1e-10 * self.gp > 0:
            raan = np.arccos(np.clip(node_vec[0] / node_mag, -1, 1))
            if node_vec[1] + 1e-10 * self.gp < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0
        
        # Argument of perigee
        if node_mag - 1e-10 * self.gp > 0 and ecc_mag + 1e-10 > 0:
            argp = np.arccos(np.clip(np.dot(node_vec/node_mag, ecc_vec/ecc_mag), -1, 1))
            if ecc_vec[2] < 0:
                argp = 2 * np.pi - argp
        else:
            argp = 0
        
        # True anomaly
        if ecc_mag - 1e-10 > 0:
            cos_ta = np.dot(ecc_vec, pos_vec) / (ecc_mag * pos_mag)
            cos_ta = np.clip(cos_ta, -1, 1)
            ta     = np.arccos(cos_ta)
            if np.dot(pos_vec, vel_vec) < 0:
                ta = 2 * np.pi - ta
            
            # Eccentric anomaly
            ea = 2 * np.arctan(np.sqrt((1 - ecc_mag) / (1 + ecc_mag)) * np.tan(ta / 2))
            if ea < 0:
                ea = ea + 2 * np.pi
            
            # Mean anomaly
            ma = ea - ecc_mag * np.sin(ea)
            ma = ma % (2 * np.pi)
        else:
            ta = 0
            ea = 0
            ma = 0
        
        return {
            'sma'  : sma,
            'ecc'  : ecc_mag,
            'inc'  : inc,
            'raan' : raan,
            'argp' : argp,
            'ma'   : ma,
            'ta'   : ta,
            'ea'   : ea
        }
    
    def coe_to_pv(
        self,
        coe: dict,
    ) -> tuple:
        """
        Convert classical orbital elements to position and velocity vectors.
        
        Input:
        ------
        coe : dict
            Dictionary containing:
                - sma  : semi-major axis [m]
                - ecc  : eccentricity [-]
                - inc  : inclination [rad]
                - raan : RAAN [rad]
                - argp : argument of perigee [rad]
                - ta   : true anomaly [rad] (or use 'ma' for mean anomaly)
        
        Output:
        -------
        pos : np.ndarray
            Position vector [m]
        vel : np.ndarray
            Velocity vector [m/s]
        """
        sma  = coe['sma']
        ecc  = coe['ecc']
        inc  = coe['inc']
        raan = coe['raan']
        argp = coe['argp']
        
        # Convert mean anomaly to true anomaly if needed
        if 'ta' in coe:
            ta = coe['ta']
        elif 'ma' in coe:
            # Solve Kepler's equation: M = E - e*sin(E)
            ma = coe['ma']
            ea = TwoBody_RootSolvers.kepler(ma, ecc)
            
            # Convert eccentric anomaly to true anomaly
            ta = 2 * np.arctan2(
                np.sqrt(1 + ecc) * np.sin(ea / 2),
                np.sqrt(1 - ecc) * np.cos(ea / 2)
            )
        else:
            raise ValueError("Must provide either 'ta' or 'ma' in coe dict")
        
        # Orbital plane (perifocal) coordinates
        p = sma * (1 - ecc**2)  # Semi-latus rectum
        r_mag = p / (1 + ecc * np.cos(ta))
        
        # Position in perifocal frame
        r_pqw = np.array([
            r_mag * np.cos(ta),
            r_mag * np.sin(ta),
            0.0
        ])
        
        # Velocity in perifocal frame
        v_pqw = np.array([
            -np.sqrt(self.gp / p) * np.sin(ta),
            np.sqrt(self.gp / p) * (ecc + np.cos(ta)),
            0.0
        ])
        
        # Rotation matrix from perifocal to inertial (3-1-3 Euler angles)
        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)
        cos_inc  = np.cos(inc)
        sin_inc  = np.sin(inc)
        cos_argp = np.cos(argp)
        sin_argp = np.sin(argp)
        
        R = np.array([
            [ cos_raan * cos_argp - sin_raan * sin_argp * cos_inc, -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc,  sin_raan * sin_inc],
            [ sin_raan * cos_argp + cos_raan * sin_argp * cos_inc, -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc, -cos_raan * sin_inc],
            [ sin_argp * sin_inc, cos_argp * sin_inc, cos_inc]
        ])
        
        # Transform to inertial frame
        pos = R @ r_pqw
        vel = R @ v_pqw
        
        return pos, vel
