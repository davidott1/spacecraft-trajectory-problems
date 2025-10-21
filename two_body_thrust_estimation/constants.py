import numpy as np


class Constants:
    """
    Physical and mathematical constants for orbital mechanics.
    """
    
    # Mathematical constants
    PI      = np.pi
    TWO_PI  = 2.0 * np.pi
    DEG2RAD = np.pi / 180.0
    RAD2DEG = 180.0 / np.pi
    
    # Earth gravitational parameter
    MU_EARTH = 398600441800000.0  # [m³/s²]
    
    # Earth radius
    RADIUS_EARTH = 6378137.0  # [m]
    
    # Time conversions
    SEC_PER_MIN  = 60.0
    MIN_PER_HOUR = 60.0
    HOUR_PER_DAY = 24.0
    SEC_PER_HOUR = SEC_PER_MIN * MIN_PER_HOUR
    SEC_PER_DAY  = SEC_PER_HOUR * HOUR_PER_DAY
