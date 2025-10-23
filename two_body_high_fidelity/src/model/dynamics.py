import numpy as np


class PHYSICALCONSTANTS:
    """
    Class to hold physical constants
    """
    class EARTH:
        class RADIUS:
            EQUATOR = 6378137.0  # Earth's WGS84 equatorial radius [m]
            POLAR   = 6356752.3  # Earth's WGS84 polar radius [m]

        GP = 3.986004418e14      # Earth's gravitational parameter [m³/s²]

        # Spherical harmonic coefficients
        J_2 = 1.08263e-3         # Earth's J2 coefficient [-]
        J_3 = -2.532153e-6       # Earth's J3 coefficient [-]
        J_4 = -1.61962159137e-6  # Earth's J4 coefficient [-]
        
        # Rotation rate
        OMEGA = 7.2921150e-5     # Earth's rotation rate [rad/s]
        
        # Reference atmosphere parameters (simplified exponential model)
        RHO_0 = 1.225            # Earth's sea level density [kg/m³]
        H_0   = 8500.0           # Earth's scale height [m]

    class MOON:
        class RADIUS:
            EQUATOR = 1737400.0  # Moon's equatorial radius [m]
            POLAR   = 1737400.0  # Moon's polar radius [m] (approximately spherical)

        GP  = 4.9048695e12 # Moon's gravitational parameter [m³/s²]
        J_2 = 2.032e-4     # Moon's J2 coefficient
        J_3 = 0.0          # Moon's J3 coefficient
        J_4 = 0.0          # Moon's J4 coefficient


class TwoBodyDynamics:
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
        mass        : float = 0.0,  # Spacecraft mass [kg]
    ):
        self.gp      = gp
        self.time_o  = time_o
        self.j_2     = j_2
        self.j_3     = j_3
        self.j_4     = j_4
        self.pos_ref = pos_ref
        self.cd      = cd
        self.area    = area
        self.mass    = mass

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


class EquationsOfMotion:
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
        pos_vec = state_vec[0:3]
        vel_vec = state_vec[3:6]

        acc_vec = self.dynamics.acceleration(time, state_vec)

        state_dot_vec      = np.zeros(6)
        state_dot_vec[0:3] = vel_vec
        state_dot_vec[3:6] = acc_vec

        return state_dot_vec