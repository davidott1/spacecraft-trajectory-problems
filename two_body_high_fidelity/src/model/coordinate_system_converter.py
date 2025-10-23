import numpy as np
from model.dynamics import PHYSICALCONSTANTS


class CoordinateSystemConverter:
    def __init__(self, gp: float):
        """
        Initialize the converter with the gravitational parameter.

        Input:
            gp: gravitational parameter [m³/s²]
        """
        self.gp = gp

    def rv2coe(self, pos, vel):
        """
        Convert position and velocity vectors to classical orbital elements.
        
        Input:
            pos: position vector [m] (3D numpy array)
            vel: velocity vector [m/s] (3D numpy array)
            
        Output:
            dict: Dictionary containing orbital elements:
                sma  : semi-major axis [m]
                ecc  : eccentricity
                inc  : inclination [rad]
                raan : right ascension of ascending node [rad]
                argp : argument of perigee [rad]
                ma   : mean anomaly [rad]
                ta   : true anomaly [rad]
                ea   : eccentric anomaly [rad]
        """
        pos = np.array(pos)
        vel = np.array(vel)
        
        # Specific angular momentum
        ang_mom_vec = np.cross(pos, vel)
        ang_mom_mag = np.linalg.norm(ang_mom_vec)

        # Eccentricity vector
        pos_mag = np.linalg.norm(pos)
        ecc_vec = np.cross(vel, ang_mom_vec) / self.gp - pos / pos_mag
        ecc_mag = np.linalg.norm(ecc_vec)
        
        # Semi-major axis
        vel_mag     = np.linalg.norm(vel)
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
            cos_ta = np.dot(ecc_vec, pos) / (ecc_mag * pos_mag)
            cos_ta = np.clip(cos_ta, -1, 1)
            ta     = np.arccos(cos_ta)
            if np.dot(pos, vel) < 0:
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
    
    def coe2rv(self, coe: dict) -> tuple:
        """
        Convert classical orbital elements to position and velocity vectors.
        
        Input:
            coe: Dictionary containing:
                - sma  : semi-major axis [m]
                - ecc  : eccentricity [-]
                - inc  : inclination [rad]
                - raan : RAAN [rad]
                - argp : argument of perigee [rad]
                - ta   : true anomaly [rad] (or use 'ma' for mean anomaly)
        
        Output:
            pos: position vector [m] (3D numpy array)
            vel: velocity vector [m/s] (3D numpy array)
        """
        sma = coe['sma']
        ecc = coe['ecc']
        inc = coe['inc']
        raan = coe['raan']
        argp = coe['argp']
        
        # Convert mean anomaly to true anomaly if needed
        if 'ta' in coe:
            ta = coe['ta']
        elif 'ma' in coe:
            # Solve Kepler's equation: M = E - e*sin(E)
            ma = coe['ma']
            ea = self._solve_kepler(ma, ecc)
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
        cos_inc = np.cos(inc)
        sin_inc = np.sin(inc)
        cos_argp = np.cos(argp)
        sin_argp = np.sin(argp)
        
        R = np.array([
            [cos_raan * cos_argp - sin_raan * sin_argp * cos_inc,
             -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc,
             sin_raan * sin_inc],
            [sin_raan * cos_argp + cos_raan * sin_argp * cos_inc,
             -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc,
             -cos_raan * sin_inc],
            [sin_argp * sin_inc,
             cos_argp * sin_inc,
             cos_inc]
        ])
        
        # Transform to inertial frame
        pos = R @ r_pqw
        vel = R @ v_pqw
        
        return pos, vel
    
    def _solve_kepler(self, ma: float, ecc: float, tol: float = 1e-10) -> float:
        """
        Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
        
        Input:
            ma: mean anomaly [rad]
            ecc: eccentricity [-]
            tol: convergence tolerance
        
        Output:
            ea: eccentric anomaly [rad]
        """
        # Initial guess
        ea = ma if ecc < 0.8 else np.pi
        
        # Newton-Raphson iteration
        for _ in range(50):
            f = ea - ecc * np.sin(ea) - ma
            fp = 1 - ecc * np.cos(ea)
            ea_new = ea - f / fp
            
            if abs(ea_new - ea) < tol:
                return ea_new
            ea = ea_new
        
        return ea  # Return best estimate if not converged