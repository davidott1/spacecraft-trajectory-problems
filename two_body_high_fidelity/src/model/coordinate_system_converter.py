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