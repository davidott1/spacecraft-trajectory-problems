class CONVERTER:
  # Unity Values
  ONE_AU  = 1.0                            # [astronomical unit]
  ONE_M   = 1.0                            # [meter]
  ONE_SEC = 1.0                            # [second]
  ONE_KG  = 1.0                            # [kilogram]

  # Angle Conversions
  RAD_PER_DEG = 3.141592653589793 / 180.0  # [radian] per [degree]
  DEG_PER_RAD = 180.0 / 3.141592653589793  # [degree] per [radian]

  # Time Conversions
  SEC_PER_YEAR = 31536000                  # [seconds] per [year]
  SEC_PER_DAY  = 86400                     # [seconds] per [day]
  SEC_PER_HOUR = 3600                      # [seconds] per [hour]
  SEC_PER_MIN  = 60                        # [seconds] per [minute]
  SEC_PER_SEC  = 1                         # [seconds] per [second]

  # Distance Conversions
  M_PER_KM = 1000.0                        # [meters] per [kilometer]
  KM_PER_M = 1.0 / 1000.0                  # [kilometers] per [meter]
  M_PER_AU = 149597870700.0                # [meters] per [astronomical unit]

  # Velocity Conversions
  M_PER_SEC__PER__AU_PER_DAY = M_PER_AU / SEC_PER_DAY  # [meters/second] per [astronomical units/day]

class PHYSICALCONSTANTS:
  speed_of_light = 299792458.0  # Speed of light in vacuum [m/s]

class NAIFIDS:
  """
  NAIF ID codes for celestial bodies used by SPICE.
  Reference: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html
  
  Notes:
  ------
  For planets with moons, DE440 ephemeris contains barycenters (1-9), not planet centers.
  For third-body gravitational perturbations, barycenters are appropriate since:
  1. The barycenter represents the center of mass of the planet system
  2. The difference between barycenter and planet center is negligible at interplanetary distances
  """
  # Sun
  SUN     = 10
  
  # Inner planets (planet centers available in DE440)
  MERCURY = 199
  VENUS   = 299
  EARTH   = 399
  MOON    = 301
  
  # Outer planets - use barycenters (planet centers not in DE440)
  MARS            = 4    # Mars Barycenter
  JUPITER         = 5    # Jupiter Barycenter
  SATURN          = 6    # Saturn Barycenter
  URANUS          = 7    # Uranus Barycenter
  NEPTUNE         = 8    # Neptune Barycenter
  PLUTO           = 9    # Pluto Barycenter
  
  # Planet center IDs (for reference, may not be in all SPK files)
  MARS_CENTER     = 499
  JUPITER_CENTER  = 599
  SATURN_CENTER   = 699
  URANUS_CENTER   = 799
  NEPTUNE_CENTER  = 899
  PLUTO_CENTER    = 999
  
  # Mapping from body name to NAIF ID (uses barycenters for outer planets)
  NAME_TO_ID = {
    'SUN'     : SUN,
    'MERCURY' : MERCURY,
    'VENUS'   : VENUS,
    'EARTH'   : EARTH,
    'MOON'    : MOON,
    'MARS'    : MARS,
    'JUPITER' : JUPITER,
    'SATURN'  : SATURN,
    'URANUS'  : URANUS,
    'NEPTUNE' : NEPTUNE,
    'PLUTO'   : PLUTO,
  }

class SOLARSYSTEMCONSTANTS:
  """
  Class to hold physical constants.
  Some constants are from "OrbitalMotion", created by Hanspeter Schaub on 6/19/05.
  """
  
  # G_SC      = 1361.0   # solar constant [W/m²] at 1 AU
  # P_SRP_1AU = G_SC / PHYSICALCONSTANTS.speed_of_light  # solar radiation pressure at 1 AU [N/m²]. approx 4.56e-6 N/m².

  class SUN:
    class RADIUS:
      EQUATOR = 696340000.0                 # Sun's equatorial radius [m]
      POLAR   = 696340000.0                 # Sun's polar radius [m]

    GP = 1.32712440018e20                   # Sun's gravitational parameter [m³/s²]
    J2 = 0.0                                # Sun's J2 coefficient (negligible)
    J3 = 0.0                                # Sun's J3 coefficient
    J4 = 0.0                                # Sun's J4 coefficient

  class MERCURY:
    class RADIUS:
      EQUATOR = 2439700.0                   # Mercury's equatorial radius [m]
    GP  = 2.2032e13                         # Mercury's gravitational parameter [m³/s²]
    J2  = 60.0e-6                           # Mercury's J2 coefficient
    SMA = 0.38709893 * CONVERTER.M_PER_AU   # Semi-major axis [m]
    ECC = 0.20563069                        # Eccentricity
    INC = 7.00487    * CONVERTER.DEG_PER_RAD  # Inclination [rad]

  class VENUS:
    class RADIUS:
      EQUATOR = 6051800.0                   # Venus's equatorial radius [m]
    GP  = 3.2485859e14                      # Venus's gravitational parameter [m³/s²]
    J2  = 4.458e-6                          # Venus's J2 coefficient
    SMA = 0.72333199 * CONVERTER.M_PER_AU   # Semi-major axis [m]
    ECC = 0.00677323                        # Eccentricity
    INC = 3.39471    * CONVERTER.DEG_PER_RAD  # Inclination [rad]

  class EARTH:
    class RADIUS:
      EQUATOR = 6378137.0                     # Earth's WGS84 equatorial radius [m]
      POLAR   = 6356752.3                     # Earth's WGS84 polar radius [m]

    GP  = 3.986004418e14                      # Earth's gravitational parameter [m³/s²]
    SMA = 1.00000011 * CONVERTER.M_PER_AU     # Semi-major axis [m]
    ECC = 0.01671022                          # Eccentricity
    INC = 0.00005    * CONVERTER.DEG_PER_RAD  # Inclination [rad]

    # SGP4 uses WGS-72 constants (not WGS-84)
    # Standard WGS-84 values:
    # J2 = 1.08263e-3
    # J3 = -2.532153e-6
    # J4 = -1.61962159137e-6
    
    # SGP4/WGS-72 values for better agreement:
    J2_WGS84 =  1.08263e-3                  # WGS-84 J2 coefficient
    J3_WGS84 = -2.532153e-6                 # WGS-84 J3 coefficient
    J4_WGS84 = -1.61962159137e-6            # WGS-84 J4 coefficient
    
    J2_WGS72 =  1.082616e-3                 # WGS-72 J2 (SGP4 uses this)
    J3_WGS72 = -2.53881e-6                  # WGS-72 J3 (SGP4 uses this)
    J4_WGS72 = -1.65597e-6                  # WGS-72 J4 (SGP4 uses this)
    
    # Additional Earth zonal harmonics
    J5 = -0.15e-6
    J6 = 0.57e-6

    # Use WGS-72 for SGP4 comparison
    J2 = J2_WGS72
    J3 = J3_WGS72
    J4 = J4_WGS72
    
    # Tesseral harmonics (degree 2, order 1 and 2)
    C21 = -1.86e-10                     # C21 tesseral harmonic coefficient (normalized)
    S21 = -1.19e-9                      # S21 tesseral harmonic coefficient (normalized) 
    C22 =  2.43914352e-6                # C22 tesseral harmonic coefficient (normalized)
    S22 = -1.40016683e-6                # S22 tesseral harmonic coefficient (normalized)
    
    # Tesseral harmonics (degree 3)
    C31 =  2.03046e-6                   # C31 tesseral harmonic coefficient (normalized)
    S31 =  2.48513e-7                   # S31 tesseral harmonic coefficient (normalized)
    C32 =  9.04788e-7                   # C32 tesseral harmonic coefficient (normalized)
    S32 = -6.19005e-7                   # S32 tesseral harmonic coefficient (normalized)
    C33 =  7.21282e-7                   # C33 tesseral harmonic coefficient (normalized)
    S33 =  1.41434e-6                   # S33 tesseral harmonic coefficient (normalized)
    
    # Rotation rate
    OMEGA = 7.2921150e-5                    # Earth's rotation rate [rad/s]
    
    # Reference atmosphere parameters (simplified exponential model)
    RHO_0 = 1.225                           # Earth's sea level density [kg/m³]
    H_0   = 8500.0                          # Earth's scale height [m]

    # Solar radiation pressure at Earth's distance
    G_SC         = 1361.0                                    # solar constant [W/m²] at 1 AU
    PRESSURE_SRP = G_SC / PHYSICALCONSTANTS.speed_of_light   # solar radiation pressure at 1 AU [N/m²]. approx 4.56e-6 N/m².

  class MOON:
    class RADIUS:
      EQUATOR = 1737400.0                   # Moon's equatorial radius [m]
      POLAR   = 1737400.0                   # Moon's polar radius [m] (approximately spherical)

    GP  = 4.9048695e12                      # Moon's gravitational parameter [m³/s²]
    J2  = 2.032e-4                          # Moon's J2 coefficient
    J3  = 0.0                               # Moon's J3 coefficient
    J4  = 0.0                               # Moon's J4 coefficient
    SMA = 3.844e8                           # Semi-major axis [m]
    ECC = 0.0549                            # Eccentricity

  class MARS:
    class RADIUS:
      EQUATOR = 3397200.0                   # Mars's equatorial radius [m]
    GP  = 4.28283e13                          # Mars's gravitational parameter [m³/s²]
    J2  = 1960.45e-6                          # Mars's J2 coefficient
    SMA = 1.52366231 * CONVERTER.M_PER_AU     # Semi-major axis [m]
    ECC = 0.09341233                          # Eccentricity
    INC = 1.85061    * CONVERTER.DEG_PER_RAD  # Inclination [rad]

  class JUPITER:
    class RADIUS:
      EQUATOR = 71492000.0                    # Jupiter's equatorial radius [m]
    GP  = 1.2671277e17                        # Jupiter's gravitational parameter [m³/s²]
    J2  = 14736.e-6                           # Jupiter's J2 coefficient
    SMA = 5.20336301 * CONVERTER.M_PER_AU     # Semi-major axis [m]
    ECC = 0.04839266                          # Eccentricity
    INC = 1.30530    * CONVERTER.DEG_PER_RAD  # Inclination [rad]

  class SATURN:
    class RADIUS:
      EQUATOR = 60268000.0                    # Saturn's equatorial radius [m]
    GP  = 3.79406e16                          # Saturn's gravitational parameter [m³/s²]
    J2  = 16298.e-6                           # Saturn's J2 coefficient
    SMA = 9.53707032 * CONVERTER.M_PER_AU     # Semi-major axis [m]
    ECC = 0.05415060                          # Eccentricity
    INC = 2.48446    * CONVERTER.DEG_PER_RAD  # Inclination [rad]

  class URANUS:
    class RADIUS:
      EQUATOR = 25559000.0                     # Uranus's equatorial radius [m]
    GP  = 5.79455e15                           # Uranus's gravitational parameter [m³/s²]
    J2  = 3343.43e-6                           # Uranus's J2 coefficient
    SMA = 19.19126393 * CONVERTER.M_PER_AU     # Semi-major axis [m]
    ECC = 0.04716771                           # Eccentricity
    INC = 0.76986     * CONVERTER.DEG_PER_RAD  # Inclination [rad]

  class NEPTUNE:
    class RADIUS:
      EQUATOR = 24746000.0                     # Neptune's equatorial radius [m]
    GP  = 6.83653e15                           # Neptune's gravitational parameter [m³/s²]
    J2  = 3411.e-6                             # Neptune's J2 coefficient
    SMA = 30.06896348 * CONVERTER.M_PER_AU     # Semi-major axis [m]
    ECC = 0.00858587                           # Eccentricity
    INC = 1.76917     * CONVERTER.DEG_PER_RAD  # Inclination [rad]

  class PLUTO:
    class RADIUS:
      EQUATOR = 1137000.0                      # Pluto's equatorial radius [m]
    GP  = 9.830e11                             # Pluto's gravitational parameter [m³/s²]
    SMA = 39.48168677 * CONVERTER.M_PER_AU     # Semi-major axis [m]
    ECC = 0.24880766                           # Eccentricity
    INC = 17.14175    * CONVERTER.DEG_PER_RAD  # Inclination [rad]


