class CONVERTER:
    # Angle Conversions
    RAD2DEG = 180.0 / 3.141592653589793  # [degree] per [radian]
    DEG2RAD = 3.141592653589793 / 180.0  # [radian] per [degree]

    # Time Conversions
    SEC_PER_DAY = 86400.0         # [seconds] per [day]

    # Distance Conversions
    M_PER_KM = 1000.0          # [meters] per [kilometer]
    KM_PER_M = 1.0 / 1000.0    # [kilometers] per [meter]
    M_PER_AU = 149597870700.0  # [meters] per [astronomical unit]
    
    # Velocity Conversions
    M_PER_SEC__PER__AU_PER_DAY = M_PER_AU / SEC_PER_DAY  # [meters/second] per [astronomical units/day]

class TIMEVALUES:
    ONE_YEAR   = 31536000   # [s]
    ONE_DAY    = 86400      # [s]
    ONE_HOUR   = 3600       # [s]
    ONE_MINUTE = 60         # [s]
    ONE_SECOND = 1          # [s]