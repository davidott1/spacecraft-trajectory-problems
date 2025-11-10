import numpy as np
from src.model.constants import PHYSICALCONSTANTS
from src.model.third_body import ThirdBodyPerturbations


class GravityAcceleration:
    """
    Gravitational acceleration components:
    - Two-body point mass
    - Two-body oblateness (J2, J3, J4)
    - Third-body point mass (Sun, Moon)
    """
    
    def __init__(
        self,
        gp                   : float,
        j_2                  : float = 0.0,
        j_3                  : float = 0.0,
        j_4                  : float = 0.0,
        pos_ref              : float = 0.0,
        enable_third_body    : bool  = False,
        time_o               : float = 0.0,
        third_body_use_spice : bool  = True,
        third_body_bodies    : list  = None,
        spice_kernel_dir     : str   = None,
    ):
        """
        Initialize gravity acceleration components
        
        Parameters:
        -----------
        gp : float
            Gravitational parameter of central body [m³/s²]
        j_2, j_3, j_4 : float
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
        spice_kernel_dir : str
            Path to SPICE kernel directory
        """
        self.gp      = gp
        self.j_2     = j_2
        self.j_3     = j_3
        self.j_4     = j_4
        self.pos_ref = pos_ref
        
        # Third-body perturbations
        self.enable_third_body = enable_third_body
        self.time_o            = time_o
        self.third_body_bodies = third_body_bodies if third_body_bodies else ['sun', 'moon']
        
        if self.enable_third_body:
            self.third_body = ThirdBodyPerturbations(
                use_spice        = third_body_use_spice,
                spice_kernel_dir = spice_kernel_dir,
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
        
        Parameters:
        -----------
        time : float
            Current time [s]
        pos_vec : np.ndarray
            Position vector [m]
        
        Returns:
        --------
        acc_vec : np.ndarray
            Total gravity acceleration [m/s²]
        """
        acc_vec  = self.two_body_point_mass(pos_vec)
        acc_vec += self.two_body_oblate_j2(pos_vec)
        acc_vec += self.two_body_oblate_j3(pos_vec)
        acc_vec += self.two_body_oblate_j4(pos_vec)
        
        if self.enable_third_body:
            acc_vec += self.third_body_point_mass(time, pos_vec)
        
        return acc_vec
    
    def two_body_point_mass(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """Two-body point mass gravity"""
        pos_mag = np.linalg.norm(pos_vec)
        return -self.gp * pos_vec / pos_mag**3
    
    def two_body_oblate_j2(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """J2 oblateness perturbation"""
        if self.j_2 == 0.0:
            return np.zeros(3)
        
        pos_mag = np.linalg.norm(pos_vec)
        posmag2 = pos_mag**2
        posmag5 = posmag2 * posmag2 * pos_mag
        
        factor = 1.5 * self.j_2 * self.gp * self.pos_ref**2 / posmag5
        
        acc_vec    = np.zeros(3)
        acc_vec[0] = factor * pos_vec[0] * (5 * pos_vec[2]**2 / posmag2 - 1)
        acc_vec[1] = factor * pos_vec[1] * (5 * pos_vec[2]**2 / posmag2 - 1)
        acc_vec[2] = factor * pos_vec[2] * (5 * pos_vec[2]**2 / posmag2 - 3)
        
        return acc_vec
    
    def two_body_oblate_j3(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """J3 oblateness perturbation"""
        if self.j_3 == 0.0:
            return np.zeros(3)
        
        x, y, z = pos_vec[0], pos_vec[1], pos_vec[2]
        pos_mag = np.linalg.norm(pos_vec)
        posmag2 = pos_mag**2
        posmag7 = posmag2 * posmag2 * posmag2 * pos_mag
        
        factor = 2.5 * self.j_3 * self.gp * self.pos_ref**3 / posmag7
        
        acc_vec    = np.zeros(3)
        acc_vec[0] = factor * x * z * (3 - 7 * z**2 / posmag2)
        acc_vec[1] = factor * y * z * (3 - 7 * z**2 / posmag2)
        acc_vec[2] = factor * (3 * z**2 - 7 * z**4 / posmag2 - 0.6 * posmag2)
        
        return acc_vec
    
    def two_body_oblate_j4(
        self,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """J4 oblateness perturbation"""
        if self.j_4 == 0.0:
            return np.zeros(3)
        
        x, y, z = pos_vec[0], pos_vec[1], pos_vec[2]
        pos_mag = np.linalg.norm(pos_vec)
        posmag2 = pos_mag**2
        posmag9 = posmag2**4 * pos_mag
        
        z2_r2  = z**2 / posmag2
        factor = 1.875 * self.j_4 * self.gp * self.pos_ref**4 / posmag9
        
        acc_vec    = np.zeros(3)
        acc_vec[0] = factor * x * (1 - 14 * z2_r2 + 21 * z2_r2**2)
        acc_vec[1] = factor * y * (1 - 14 * z2_r2 + 21 * z2_r2**2)
        acc_vec[2] = factor * z * (5 - 70 * z2_r2 / 3 + 21 * z2_r2**2)
        
        return acc_vec
    
    def third_body_point_mass(
        self,
        time    : float,
        pos_vec : np.ndarray,
    ) -> np.ndarray:
        """Third-body point mass perturbations (Sun, Moon)"""
        if not self.enable_third_body or self.third_body is None:
            return np.zeros(3)
        
        # Ephemeris time is seconds from J2000 epoch
        et_seconds = self.time_o + time
        
        # Third-body acceleration in km/s²
        accel_third_body_km = self.third_body.compute_acceleration(
            pos_sat_vec = pos_vec / 1000.0,  # Convert m to km
            et_seconds  = et_seconds,
            bodies      = self.third_body_bodies,
        )
        
        # Convert km/s² to m/s²
        return accel_third_body_km * 1000.0


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
        
        Parameters:
        -----------
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
        
        Parameters:
        -----------
        pos_vec : np.ndarray
            Position vector [m]
        vel_vec : np.ndarray
            Velocity vector [m/s]
        
        Returns:
        --------
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
        
        Parameters:
        -----------
        altitude : float
            Altitude above Earth's surface [m]
        
        Returns:
        --------
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
        
        Parameters:
        -----------
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
        
        Parameters:
        -----------
        time : float
            Current time [s]
        pos_vec : np.ndarray
            Position vector [m]
        
        Returns:
        --------
        acc_vec : np.ndarray
            SRP acceleration [m/s²]
        """
        # TODO: Implement solar radiation pressure
        return np.zeros(3)


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
        gp                   : float,
        time_o               : float = 0.0,
        j_2                  : float = 0.0,
        j_3                  : float = 0.0,
        j_4                  : float = 0.0,
        pos_ref              : float = 0.0,
        cd                   : float = 0.0,
        area                 : float = 0.0,
        mass                 : float = 1.0,
        enable_drag          : bool  = False,
        enable_third_body    : bool  = False,
        third_body_use_spice : bool  = True,
        third_body_bodies    : list  = None,
        spice_kernel_dir     : str   = None,
        enable_srp           : bool  = False,
        cr                   : float = 1.3,
    ):
        """
        Initialize acceleration coordinator
        
        Parameters:
        -----------
        gp : float
            Gravitational parameter of central body [m³/s²]
        time_o : float
            Initial epoch time [s]
        j_2, j_3, j_4 : float
            Harmonic coefficients for oblateness
        pos_ref : float
            Reference radius for harmonic coefficients [m]
        cd : float
            Drag coefficient
        area : float
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
        spice_kernel_dir : str
            Path to SPICE kernel directory
        enable_srp : bool
            Enable solar radiation pressure (not yet implemented)
        cr : float
            Radiation pressure coefficient
        """
        # Create acceleration component instances
        self.gravity = GravityAcceleration(
            gp                   = gp,
            j_2                  = j_2,
            j_3                  = j_3,
            j_4                  = j_4,
            pos_ref              = pos_ref,
            enable_third_body    = enable_third_body,
            time_o               = time_o,
            third_body_use_spice = third_body_use_spice,
            third_body_bodies    = third_body_bodies,
            spice_kernel_dir     = spice_kernel_dir,
        )
        
        self.enable_drag = enable_drag
        if self.enable_drag and cd > 0 and area > 0 and mass > 0:
            self.drag = AtmosphericDrag(cd=cd, area=area, mass=mass)
        else:
            self.drag = None
        
        self.enable_srp = enable_srp
        if self.enable_srp:
            self.srp = SolarRadiationPressure(cr=cr, area=area, mass=mass)
        else:
            self.srp = None
    
    def total(
        self,
        time    : float,
        pos_vec : np.ndarray,
        vel_vec : np.ndarray,
    ) -> np.ndarray:
        """
        Compute total acceleration from all components
        
        Parameters:
        -----------
        time : float
            Current time [s]
        pos_vec : np.ndarray
            Position vector [m]
        vel_vec : np.ndarray
            Velocity vector [m/s]
        
        Returns:
        --------
        acc_vec : np.ndarray
            Total acceleration [m/s²]
        """
        # Gravity (always included)
        acc_vec = self.gravity.compute(time, pos_vec)
        
        # Atmospheric drag (optional)
        if self.drag is not None:
            acc_vec += self.drag.compute(pos_vec, vel_vec)
        
        # Solar radiation pressure (optional, future)
        if self.srp is not None:
            acc_vec += self.srp.compute(time, pos_vec)
        
        return acc_vec


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
        
        Parameters:
        -----------
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
        
        Parameters:
        -----------
        time : float
            Current time [s]
        state_vec : np.ndarray
            Current state vector [pos, vel] [m, m/s]
        
        Returns:
        --------
        state_dot_vec : np.ndarray
            Time derivative of state vector [vel, acc] [m/s, m/s²]
        """
        pos_vec = state_vec[0:3]
        vel_vec = state_vec[3:6]
        acc_vec = self.acceleration.total(time, pos_vec, vel_vec)
        
        state_dot_vec      = np.zeros(6)
        state_dot_vec[0:3] = vel_vec
        state_dot_vec[3:6] = acc_vec
        
        return state_dot_vec


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
        
        Parameters:
        -----------
        ma : float
            Mean anomaly [rad]
        ecc : float
            Eccentricity
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations
        
        Returns:
        --------
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
        
        Parameters:
        -----------
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
        
        Returns:
        --------
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
        
        Parameters:
        -----------
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
        
        Parameters:
        -----------
        pos_vec : np.ndarray
            Position vector [m]
        vel_vec : np.ndarray
            Velocity vector [m/s]
        
        Returns:
        --------
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
        
        Parameters:
        -----------
        coe : dict
            Dictionary containing:
                - sma  : semi-major axis [m]
                - ecc  : eccentricity [-]
                - inc  : inclination [rad]
                - raan : RAAN [rad]
                - argp : argument of perigee [rad]
                - ta   : true anomaly [rad] (or use 'ma' for mean anomaly)
        
        Returns:
        --------
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
