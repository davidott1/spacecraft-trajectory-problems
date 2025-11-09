"""
Third-body gravitational perturbations using SPICE ephemerides
"""

import numpy as np
import spiceypy as spice
from pathlib import Path
from typing import Optional

from two_body_high_fidelity.src.model.constants import PHYSICALCONSTANTS, CONVERTER

class ThirdBodyPerturbations:
    """
    Third-body gravitational perturbations from Sun and Moon.
    Uses SPICE ephemerides for body positions

    Methods
        get_body_position(body_name, et_seconds, frame='J2000')
            Get position of celestial body at given time
        compute_acceleration(r_sat, et_seconds, bodies=['SUN', 'MOON'])
            Compute third-body gravitational acceleration on satellite
    
    Private Methods
        _load_spice_kernels(kernel_dir)
            Load required SPICE kernels from directory
        _get_naif_id(body_name)
            Get NAIF ID for celestial body
        _analytical_position(body_name, et_seconds)
            Compute approximate position using analytical formulas
    """
    def __init__(
        self,
        use_spice        : bool           = True,
        spice_kernel_dir : Optional[Path] = None,
    ):
        """
        Initialize third-body perturbations
        
        Input
            use_spice : bool
                If True, use SPICE ephemerides (high accuracy)
                If False, use analytical approximations (faster, less accurate)
            spice_kernel_dir : str or Path
                Directory containing SPICE kernel files
        """
        self.use_spice = use_spice
        if use_spice:
            self._load_spice_kernels(spice_kernel_dir)
    
    def _load_spice_kernels(
        self,
        kernel_dir: Optional[Path],
    ):  
        """
        Load required SPICE kernels
        
        Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
        
        Required kernels:
        - LSK (Leap Second Kernel): naif0012.tls
        - SPK (Planetary Ephemeris): de430.bsp or de440.bsp
        - PCK (Planetary Constants): pck00010.tpc
        """
        if kernel_dir is None:
            # Default to a kernels directory in the project
            kernel_dir = Path(__file__).parent.parent.parent / 'data' / 'spice_kernels'
        
        kernel_dir = Path(kernel_dir)
        
        if not kernel_dir.exists():
            raise FileNotFoundError(
                f"SPICE kernel directory not found: {kernel_dir}\n"
                f"Please download kernels from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/\n"
                f"Required files:\n"
                f"  - lsk/naif0012.tls\n"
                f"  - spk/planets/de440.bsp (or de430.bsp)\n"
                f"  - pck/pck00010.tpc"
            )
        
        # Load leap second kernel
        lsk_file = kernel_dir / 'naif0012.tls'
        if lsk_file.exists():
            spice.furnsh(str(lsk_file))
        else:
            raise FileNotFoundError(f"LSK file not found: {lsk_file}")
        
        # Load planetary ephemeris
        spk_files = list(kernel_dir.glob('de*.bsp'))
        if spk_files:
            spice.furnsh(str(spk_files[0]))  # Use first found
        else:
            raise FileNotFoundError(f"No SPK files (de*.bsp) found in {kernel_dir}")
        
        # Load planetary constants
        pck_file = kernel_dir / 'pck00010.tpc'
        if pck_file.exists():
            spice.furnsh(str(pck_file))
        else:
            raise FileNotFoundError(f"PCK file not found: {pck_file}")
        
        print(f"SPICE kernels loaded from: {kernel_dir}")
    
    def get_body_position(
        self,
        body_name  : str,
        et_seconds : float,
        frame      : str   = 'J2000',
    ) -> np.ndarray:
        """
        Get position of celestial body at given time
        
        Input
            body_name : str
                'SUN' or 'MOON'
            et_seconds : float
                Ephemeris time in seconds past J2000 epoch
            frame : str
                Reference frame (default: 'J2000')
        
        Output
            pos_vec : np.ndarray (3,)
                Position vector [km]
        """
        if self.use_spice:
            # SPICE state relative to Earth
            state, _ = spice.spkez(
                targ   = self._get_naif_id(body_name),
                et     = et_seconds,
                ref    = frame,
                abcorr = 'NONE',
                obs    = 399  # Earth
            )
            return np.array(state[0:3])  # position only
        else:
            # Use analytical approximation
            return self._analytical_position(body_name, et_seconds)
    
    def _get_naif_id(
        self,
        body_name : str,
    ) -> int:
        """
        Get NAIF ID for body
        """
        naif_ids = {
            'SUN'  : 10,
            'MOON' : 301,
        }
        return naif_ids[body_name.upper()]
    
    def _analytical_position(self, body_name, et_seconds):
        """
        Simple analytical approximation for Sun/Moon position
        Lower accuracy (~1000 km for Moon, ~10,000 km for Sun)
        Good enough for rough estimates
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
    
    def compute_acceleration(
        self,
        pos_sat_vec : np.ndarray,
        et_seconds  : float,
        bodies      : list = ['SUN', 'MOON'],
    ) -> np.ndarray:
        """
        Compute third-body gravitational acceleration on satellite
        
        Input
        pos_sat_vec : np.ndarray (3,)
            Satellite position relative to Earth [km]
        et_seconds : float
            Ephemeris time in seconds past J2000
        bodies : list of str
            Which bodies to include (default: ['SUN', 'MOON'])
        
        Output
        acc_vec : np.ndarray (3,)
            Acceleration vector [km/s²]
        """
        acc_vec = np.zeros(3)
        for body in bodies:
            # Get position of perturbing body relative to Earth
            pos_body_vec = self.get_body_position(body, et_seconds)
            
            # Position of satellite relative to perturbing body
            pos_sat_to_body = pos_body_vec - pos_sat_vec
            
            # Get gravitational parameter
            if body.upper() == 'SUN':
                GP = PHYSICALCONSTANTS.SUN.GP  * CONVERTER.KM_PER_M**3  # [m³/s²] -> [km³/s²]
            elif body.upper() == 'MOON':
                GP = PHYSICALCONSTANTS.MOON.GP  * CONVERTER.KM_PER_M**3  # [m³/s²] -> [km³/s²]
            else:
                continue
            
            # Third-body acceleration (point mass approximation)
            #   a = GM * (r_sat_to_body / |r_sat_to_body|³ - r_body / |r_body|³)
            pos_sat_to_body_mag = np.linalg.norm(pos_sat_to_body)
            pos_body_mag        = np.linalg.norm(pos_body_vec)

            acc_vec += GP * (
                pos_sat_to_body / pos_sat_to_body_mag**3
                - pos_body_vec / pos_body_mag**3
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


# Example usage and testing
if __name__ == "__main__":
    # Test with and without SPICE
    print("Testing third-body perturbations...")
    
    # Without SPICE (analytical approximation)
    print("\n1. Analytical approximation (no SPICE):")
    tb_analytical = ThirdBodyPerturbations(use_spice=False)
    
    et = 0.0  # J2000 epoch
    r_sat = np.array([7000.0, 0.0, 0.0])  # LEO satellite position
    
    accel_analytical = tb_analytical.compute_acceleration(r_sat, et)
    print(f"   Sun+Moon acceleration magnitude: {np.linalg.norm(accel_analytical):.6e} km/s^2")
    
    # With SPICE (if available)
    print("\n2. SPICE ephemerides (high accuracy):")
    try:
        spice_kernels_folderpath = Path(__file__).parent.parent.parent.parent / 'data' / 'spice_kernels'
        tb_spice = ThirdBodyPerturbations(use_spice=True, spice_kernel_dir=spice_kernels_folderpath)
        accel_spice = tb_spice.compute_acceleration(r_sat, et)
        print(f"   Sun+Moon acceleration magnitude: {np.linalg.norm(accel_spice):.6e} km/s^2")
        
        # Compare
        diff = np.linalg.norm(accel_spice - accel_analytical)
        print(f"   Difference: {diff:.6e} km/s^2 ({diff/np.linalg.norm(accel_spice)*100:.2f}%)")
    except FileNotFoundError as e:
        print(f"   SPICE kernels not available: {e}")
        print(f"   Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/")
