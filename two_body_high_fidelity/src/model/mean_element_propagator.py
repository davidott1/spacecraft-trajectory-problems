import numpy as np
from model.dynamics import PHYSICALCONSTANTS


class MeanElementPropagator:
    """
    Mean element propagator using Brouwer-Lyddane theory.
    Propagates mean elements with J2 secular and long-period effects.
    """
    
    def __init__(self, 
                 gp: float = PHYSICALCONSTANTS.EARTH.GP,
                 j2: float = PHYSICALCONSTANTS.EARTH.J_2,
                 r_eq: float = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR):
        self.gp = gp
        self.j2 = j2
        self.r_eq = r_eq
    
    def propagate_mean_elements(self, 
                                mean_coe_0: dict,
                                time_array: np.ndarray) -> dict:
        """
        Propagate mean orbital elements using Brouwer-Lyddane theory.
        
        Input:
            mean_coe_0: Initial mean orbital elements dict
                - sma  : semi-major axis [m]
                - ecc  : eccentricity [-]
                - inc  : inclination [rad]
                - raan : RAAN [rad]
                - argp : argument of perigee [rad]
                - ma   : mean anomaly [rad]
            time_array: Array of times to propagate to [s]
        
        Output:
            mean_coe_history: dict of arrays for each element over time
        """
        num_steps = len(time_array)
        
        # Initialize output arrays
        mean_coe_history = {
            'sma'  : np.zeros(num_steps),
            'ecc'  : np.zeros(num_steps),
            'inc'  : np.zeros(num_steps),
            'raan' : np.zeros(num_steps),
            'argp' : np.zeros(num_steps),
            'ma'   : np.zeros(num_steps),
        }
        
        # Extract initial mean elements
        a0 = mean_coe_0['sma']
        e0 = mean_coe_0['ecc']
        i0 = mean_coe_0['inc']
        raan0 = mean_coe_0['raan']
        argp0 = mean_coe_0['argp']
        M0 = mean_coe_0['ma']
        
        # Compute secular rates (Brouwer-Lyddane)
        n0 = np.sqrt(self.gp / a0**3)  # Mean motion
        p0 = a0 * (1 - e0**2)  # Semi-latus rectum
        
        # J2 coefficient
        k2 = 0.5 * self.j2 * self.r_eq**2
        
        # Auxiliary variables
        eta = np.sqrt(1 - e0**2)
        cos_i = np.cos(i0)
        sin_i = np.sin(i0)
        
        # Secular rates (deg/s or rad/s)
        raan_dot = -(3/2) * (k2 / p0**2) * n0 * cos_i
        
        argp_dot = (3/4) * (k2 / p0**2) * n0 * (4 - 5 * sin_i**2)
        
        M_dot = n0 + (3/4) * (k2 / p0**2) * n0 * eta * (2 - 3 * sin_i**2)
        
        # Propagate mean elements (semi-major axis, eccentricity, inclination constant for J2-only)
        for idx, t in enumerate(time_array):
            dt = t  # Time since epoch
            
            # Secular variations (linear with time for J2)
            mean_coe_history['sma'][idx] = a0
            mean_coe_history['ecc'][idx] = e0
            mean_coe_history['inc'][idx] = i0
            mean_coe_history['raan'][idx] = (raan0 + raan_dot * dt) % (2 * np.pi)
            mean_coe_history['argp'][idx] = (argp0 + argp_dot * dt) % (2 * np.pi)
            mean_coe_history['ma'][idx] = (M0 + M_dot * dt) % (2 * np.pi)
        
        return mean_coe_history
    
    def mean_coe_to_state(self, mean_coe: dict) -> np.ndarray:
        """
        Convert mean orbital elements to Cartesian state.
        Uses mean elements directly (no osculating conversion).
        
        Input:
            mean_coe: Mean orbital elements dict
        
        Output:
            state: [x, y, z, vx, vy, vz] in meters and m/s
        """
        from model.coordinate_system_converter import CoordinateSystemConverter
        
        converter = CoordinateSystemConverter(self.gp)
        pos, vel = converter.coe2rv(mean_coe)
        
        return np.concatenate([pos, vel])
