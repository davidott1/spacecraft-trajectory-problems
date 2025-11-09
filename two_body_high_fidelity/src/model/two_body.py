import numpy as np
from src.model.constants import PHYSICALCONSTANTS
from src.model.third_body import ThirdBodyPerturbations


class TwoBodyDynamics:
    """
    Two-body dynamics with perturbations
    """
    
    def __init__(
        self,
        gp          : float,
        time_o      : float = 0.0,
        j_2         : float = 0.0,
        j_3         : float = 0.0,
        j_4         : float = 0.0,
        pos_ref     : float = 0.0,
        cd          : float = 0.0,  # Drag coefficient
        area        : float = 0.0,  # Cross-sectional area [m²]
        mass        : float = 1.0,  # Spacecraft mass [kg]
        # NEW: Third-body perturbations
        enable_third_body: bool = False,
        third_body_use_spice: bool = True,
        third_body_bodies: list = None,
        spice_kernel_dir: str = None,
    ):
        """
        Initialize dynamics model
        
        Parameters:
        -----------
        gp : float
            Gravitational parameter of the central body [m³/s²]
        time_o : float, optional
            Initial time [s]
        j_2 : float, optional
            J2 coefficient (Earth's oblateness)
        j_3 : float, optional
            J3 coefficient (Earth's asymmetry)
        j_4 : float, optional
            J4 coefficient (Earth's asymmetry)
        pos_ref : float, optional
            Reference position for acceleration calculations [m]
        cd : float, optional
            Drag coefficient
        area : float, optional
            Cross-sectional area [m²]
        mass : float, optional
            Spacecraft mass [kg]
        enable_third_body : bool
            Enable Sun/Moon gravitational perturbations
        third_body_use_spice : bool
            Use SPICE ephemerides (True) or analytical approximations (False)
        third_body_bodies : list of str
            Which bodies to include (default: ['SUN', 'MOON'])
        spice_kernel_dir : str
            Path to SPICE kernel directory (optional)
        """
        self.gp      = gp
        self.time_o  = time_o
        self.j_2     = j_2
        self.j_3     = j_3
        self.j_4     = j_4
        self.pos_ref = pos_ref
        self.cd      = cd
        self.area    = area
        self.mass    = mass
        
        # Third-body perturbations
        self.enable_third_body = enable_third_body
        self.third_body_bodies = third_body_bodies if third_body_bodies else ['SUN', 'MOON']
        
        if self.enable_third_body:
            self.third_body = ThirdBodyPerturbations(
                use_spice=third_body_use_spice,
                spice_kernel_dir=spice_kernel_dir
            )
        else:
            self.third_body = None

    def acceleration(
        self,
        time      : float,
        state_vec : np.ndarray,
    ) -> np.ndarray:
        pos_vec = state_vec[0:3]

        acc_vec = self.acc_two_body_problem(time, pos_vec)
        acc_vec += self.acc_j_2(pos_vec)
        acc_vec += self.acc_j_3(pos_vec)
        acc_vec += self.acc_j_4(pos_vec)
        
        if self.cd > 0 and self.area > 0 and self.mass > 0:
            acc_vec += self.acc_drag(state_vec)

        # Third-body perturbations (Sun/Moon)
        if self.enable_third_body:
            # Ephemeris time is seconds from J2000 epoch
            et_seconds = self.time_o + time
            
            accel_third_body_km = self.third_body.compute_acceleration(
                r_sat=pos_vec / 1000.0,
                et_seconds=et_seconds,
                bodies=self.third_body_bodies
            )
            
            # Convert km/s^2 to m/s^2
            acc_vec += accel_third_body_km * 1000.0

        return acc_vec

    def acc_two_body_problem(
        self,
        time    : float,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        pos_mag = np.sqrt(pos_vec[0]**2 + pos_vec[1]**2 + pos_vec[2]**2)

        posmag3 = pos_mag**3

        acc_vec    = np.zeros(3)
        acc_vec[0] = -self.gp * pos_vec[0] / posmag3
        acc_vec[1] = -self.gp * pos_vec[1] / posmag3
        acc_vec[2] = -self.gp * pos_vec[2] / posmag3

        return acc_vec

    def acc_j_2(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        
        pos_mag = np.sqrt(pos_vec[0]**2 + pos_vec[1]**2 + pos_vec[2]**2)
        posmag2 = pos_mag**2
        posmag5 = posmag2 * posmag2 * pos_mag

        factor = 1.5 * self.j_2 * self.gp * self.pos_ref**2 / posmag5

        acc_vec    = np.zeros(3)
        acc_vec[0] = factor * pos_vec[0] * (5 * pos_vec[2]**2 / posmag2 - 1)
        acc_vec[1] = factor * pos_vec[1] * (5 * pos_vec[2]**2 / posmag2 - 1)
        acc_vec[2] = factor * pos_vec[2] * (5 * pos_vec[2]**2 / posmag2 - 3)

        return acc_vec

    def acc_j_3(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        
        x, y, z = pos_vec[0], pos_vec[1], pos_vec[2]
        pos_mag = np.sqrt(x**2 + y**2 + z**2)
        posmag2 = pos_mag**2
        posmag7 = posmag2 * posmag2 * posmag2 * pos_mag

        factor = 2.5 * self.j_3 * self.gp * self.pos_ref**3 / posmag7

        acc_vec    = np.zeros(3)
        acc_vec[0] = factor * x * z * (3 - 7 * z**2 / posmag2)
        acc_vec[1] = factor * y * z * (3 - 7 * z**2 / posmag2)
        acc_vec[2] = factor * (3 * z**2 - 7 * z**4 / posmag2 - 0.6 * posmag2)

        return acc_vec

    def acc_j_4(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        
        x, y, z = pos_vec[0], pos_vec[1], pos_vec[2]
        pos_mag = np.sqrt(x**2 + y**2 + z**2)
        posmag2 = pos_mag**2
        posmag9 = posmag2**4 * pos_mag

        z2_r2 = z**2 / posmag2
        factor = 1.875 * self.j_4 * self.gp * self.pos_ref**4 / posmag9

        acc_vec    = np.zeros(3)
        acc_vec[0] = factor * x * (1 - 14 * z2_r2 + 21 * z2_r2**2)
        acc_vec[1] = factor * y * (1 - 14 * z2_r2 + 21 * z2_r2**2)
        acc_vec[2] = factor * z * (5 - 70 * z2_r2 / 3 + 21 * z2_r2**2)

        return acc_vec

    def atmospheric_density(
        self,
        altitude : float,
    ) -> float:
        """
        Simplified exponential atmospheric density model.
        Similar to SGP4's atmosphere model.
        
        Input:
            altitude: altitude above Earth's surface [m]
        
        Output:
            density: atmospheric density [kg/m³]
        """
        if altitude < 0:
            altitude = 0
        
        # Simplified exponential model
        rho = PHYSICALCONSTANTS.EARTH.RHO_0 * np.exp(-altitude / PHYSICALCONSTANTS.EARTH.H_0)
        
        return rho

    def acc_drag(
        self,
        state_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Atmospheric drag acceleration using exponential atmosphere model.
        
        Input:
            state_vec: state vector in inertial frame [pos, vel] [m, m/s]
        
        Output:
            acc_vec: drag acceleration vector [m/s²]
        """
        pos_vec = state_vec[0:3]
        vel_vec = state_vec[3:6]
        
        # Position magnitude
        pos_mag = np.linalg.norm(pos_vec)
        alt     = pos_mag - PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR
        
        # Atmospheric density at current altitude
        rho = self.atmospheric_density(float(alt))

        # Velocity relative to rotating atmosphere
        omega_earth = np.array([0, 0, PHYSICALCONSTANTS.EARTH.OMEGA])
        vel_rel_vec = vel_vec - np.cross(omega_earth, pos_vec)
        vel_rel_mag = np.linalg.norm(vel_rel_vec)

        # Drag acceleration
        acc_drag_mag = 1/2 * rho * (self.cd * self.area / self.mass) * vel_rel_mag**2
        acc_drag_dir = -vel_rel_vec / vel_rel_mag 
        acc_vec      = acc_drag_mag * acc_drag_dir

        return acc_vec
    
    def acc_third_body():
        pass
    



class TwoBodyEquationsOfMotion:
    def __init__(
        self,
        dynamics : TwoBodyDynamics,
    ):
        self.dynamics = dynamics

    def state_time_derivative(
        self,
        time      : float,
        state_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Compute state time derivative for ODE integration

        Input
            time      : current time [s]
            state_vec : current state vector [pos, vel] [m, m/s]
        Output
            state_dot_vec : time derivative of state vector [vel, acc] [m/s, m/s²]
        """
        pos_vec = state_vec[0:3]
        vel_vec = state_vec[3:6]

        acc_vec = self.dynamics.acceleration(time, state_vec)

        state_dot_vec      = np.zeros(6)
        state_dot_vec[0:3] = vel_vec
        state_dot_vec[3:6] = acc_vec

        return state_dot_vec
    

class TwoBodyRootSolvers:
    @staticmethod
    def kepler(
        ma       : float,
        ecc      : float,
        tol      : float = 1e-10,
        max_iter : int   = 50
    ) -> float:
        """
        Solve Kepler's equation ma = ea - ecc*sin(ea) for eccentric anomaly ea.
        
        Input
            ma       : mean anomaly [rad]
            ecc      : eccentricity
            tol      : convergence tolerance
            max_iter : maximum iterations
        Output
            ea : eccentric anomaly [rad]
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
        
        Input
            pos_o_vec : initial position vector [m]
            pos_f_vec : final position vector [m]
            delta_t   : time of flight [s]
            gp        : gravitational parameter [m³/s²]
            prograde  : True for prograde, False for retrograde
            max_iter  : maximum iterations
            tol       : convergence tolerance
        Output
            vel_o_vec : initial velocity vector [m/s]
            vel_f_vec : final velocity vector [m/s]
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
    def __init__(
        self,
        gp: float = PHYSICALCONSTANTS.EARTH.GP,
    ):
        """
        Initialize the converter with the gravitational parameter.

        Input:
            gp: gravitational parameter [m³/s²]
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
            pos_vec: position vector [m] (3D numpy array)
            vel_vec: velocity vector [m/s] (3D numpy array)

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
        # Convert inputs to numpy arrays (if not already)
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
            ea = TwoBodyRootSolvers.kepler(ma, ecc)

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