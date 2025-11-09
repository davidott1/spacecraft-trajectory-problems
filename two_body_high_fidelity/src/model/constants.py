class CONVERTER:
    # Angle Conversions
    RAD_PER_DEG = 180.0 / 3.141592653589793  # [degree] per [radian]
    DEG_PER_RAD = 3.141592653589793 / 180.0  # [radian] per [degree]

    # Time Conversions
    SEC_PER_YEAR = 31536000  # [seconds] per [year]
    SEC_PER_DAY  = 86400     # [seconds] per [day]
    SEC_PER_HOUR = 3600      # [seconds] per [hour]
    SEC_PER_MIN  = 60        # [seconds] per [minute]
    SEC_PER_SEC  = 1         # [seconds] per [second]

    # Distance Conversions
    M_PER_KM = 1000.0          # [meters] per [kilometer]
    KM_PER_M = 1.0 / 1000.0    # [kilometers] per [meter]
    M_PER_AU = 149597870700.0  # [meters] per [astronomical unit]
    
    # Velocity Conversions
    M_PER_SEC__PER__AU_PER_DAY = M_PER_AU / SEC_PER_DAY  # [meters/second] per [astronomical units/day]

class PHYSICALCONSTANTS:
    """
    Class to hold physical constants
    """
    class SUN:
        class RADIUS:
            EQUATOR = 696340000.0  # Sun's equatorial radius [m]
            POLAR   = 696340000.0  # Sun's polar radius [m]

        GP  = 1.32712440018e20  # Sun's gravitational parameter [m³/s²]
        J_2 = 0.0               # Sun's J2 coefficient (negligible)
        J_3 = 0.0               # Sun's J3 coefficient
        J_4 = 0.0               # Sun's J4 coefficient

    class EARTH:
        class RADIUS:
            EQUATOR = 6378137.0  # Earth's WGS84 equatorial radius [m]
            POLAR   = 6356752.3  # Earth's WGS84 polar radius [m]

        GP = 3.986004418e14     # Earth's gravitational parameter [m³/s²]

        # SGP4 uses WGS-72 constants (not WGS-84)
        # Standard WGS-84 values:
        # J_2 = 1.08263e-3
        # J_3 = -2.532153e-6
        # J_4 = -1.61962159137e-6
        
        # SGP4/WGS-72 values for better agreement:
        J_2_WGS84 =  1.08263e-3        # WGS-84 J2 coefficient
        J_3_WGS84 = -2.532153e-6       # WGS-84 J3 coefficient
        J_4_WGS84 = -1.61962159137e-6  # WGS-84 J4 coefficient
        
        J_2_WGS72 =  1.082616e-3       # WGS-72 J2 (SGP4 uses this)
        J_3_WGS72 = -2.53881e-6        # WGS-72 J3 (SGP4 uses this)
        J_4_WGS72 = -1.65597e-6        # WGS-72 J4 (SGP4 uses this)
        
        # Use WGS-72 for SGP4 comparison
        J_2 = J_2_WGS72
        J_3 = J_3_WGS72
        J_4 = J_4_WGS72
        
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