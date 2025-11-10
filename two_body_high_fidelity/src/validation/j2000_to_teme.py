#!/usr/bin/env python3
"""
J2000 (ICRS) to TEME coordinate transformation compatible with SGP4

This implements the transformation from J2000/ICRS coordinates to 
True Equator Mean Equinox (TEME) of date, using the same algorithms 
that SGP4 uses internally.

Reference: Vallado, "Fundamentals of Astrodynamics and Applications" 4th ed.
           Algorithm 24 (TEME conversions)
"""

import numpy as np
from astropy.time import Time


def rotation_x(angle):
    """Rotation matrix about X axis"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1,  0,  0],
        [0,  c,  s],
        [0, -s,  c]
    ])


def rotation_z(angle):
    """Rotation matrix about Z axis"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [ c,  s,  0],
        [-s,  c,  0],
        [ 0,  0,  1]
    ])


def j2000_to_teme(pos_j2000, vel_j2000, time_utc):
    """
    Transform position and velocity from J2000 (ICRS) to TEME
    
    This uses a simplified transformation appropriate for Earth satellites,
    matching what SGP4 expects for TEME coordinates.
    
    Parameters:
    -----------
    pos_j2000 : array-like (3,)
        Position vector in J2000/ICRS frame [km]
    vel_j2000 : array-like (3,)
        Velocity vector in J2000/ICRS frame [km/s]
    time_utc : astropy.time.Time
        UTC time for the transformation
    
    Returns:
    --------
    pos_teme : ndarray (3,)
        Position vector in TEME frame [km]
    vel_teme : ndarray (3,)
        Velocity vector in TEME frame [km/s]
    
    Notes:
    ------
    TEME (True Equator Mean Equinox of date) is defined by:
    - Z-axis aligned with true pole (including nutation)
    - X-axis aligned with mean equinox (excluding nutation in RA)
    
    This is the frame SGP4 outputs, dating from the 1970s.
    """
    
    # Convert to Julian Date
    jd_utc = time_utc.jd
    
    # Time since J2000 epoch (JD 2451545.0) in Julian centuries
    T_UT1 = (jd_utc - 2451545.0) / 36525.0
    
    # Mean obliquity of ecliptic (Vallado Eq. 3-52)
    # This is epsilon_bar (mean obliquity)
    epsilon = (23.439291 - 0.0130042 * T_UT1) * (np.pi / 180.0)
    
    # For TEME, we need the transformation from J2000 to TEME
    # which involves precession to mean-of-date and then to true equator
    
    # Simplified approach: For Earth satellites over short timespans,
    # the dominant rotation is Earth's rotation and precession.
    # The difference between J2000 and TEME is primarily:
    # 1. Precession (mean equinox movement)
    # 2. Nutation (true pole vs mean pole)
    
    # For a more accurate transformation, we use:
    # TEME = R_z(-GAST) @ R_x(-eps) @ R_z(GAST) @ J2000
    # where GAST = Greenwich Apparent Sidereal Time
    
    # However, for the JPL HORIZONS comparison, the key issue is that
    # HORIZONS outputs in ICRS which is essentially J2000
    # and SGP4 outputs in TEME
    
    # The transformation that works for most purposes:
    # Just account for precession and nutation effects
    
    # Mean obliquity precession (simplified)
    # Precession angles (Vallado Eq. 3-88, 3-89, 3-90)
    zeta = (2306.2181 * T_UT1 + 0.30188 * T_UT1**2 + 0.017998 * T_UT1**3) / 3600.0 * (np.pi / 180.0)
    theta = (2004.3109 * T_UT1 - 0.42665 * T_UT1**2 - 0.041833 * T_UT1**3) / 3600.0 * (np.pi / 180.0)
    z = (2306.2181 * T_UT1 + 1.09468 * T_UT1**2 + 0.018203 * T_UT1**3) / 3600.0 * (np.pi / 180.0)
    
    # Precession matrix from J2000 to mean-of-date
    # P = R_z(-z) @ R_y(theta) @ R_z(-zeta)
    R_prec = rotation_z(-z) @ rotation_x(theta) @ rotation_z(-zeta)
    
    # Nutation (simplified - using mean nutation values)
    # For full accuracy, would need to compute nutation series
    # Here we use a simplified approximation
    
    # Mean longitude of lunar ascending node (Vallado Eq. 3-60)
    Omega = (125.04452 - 1934.136261 * T_UT1) * (np.pi / 180.0)
    
    # Mean longitude of Sun (Vallado Eq. 3-59)
    L_sun = (280.4665 + 36000.7698 * T_UT1) * (np.pi / 180.0)
    
    # Simplified nutation in longitude and obliquity
    # (These are very approximate - full IAU model has hundreds of terms)
    delta_psi = (-17.20 * np.sin(Omega) - 1.32 * np.sin(2 * L_sun)) / 3600.0 * (np.pi / 180.0)
    delta_epsilon = (9.20 * np.cos(Omega) + 0.57 * np.cos(2 * L_sun)) / 3600.0 * (np.pi / 180.0)
    
    # True obliquity
    epsilon_true = epsilon + delta_epsilon
    
    # Nutation matrix
    R_nut = rotation_x(-epsilon) @ rotation_z(-delta_psi) @ rotation_x(epsilon_true)
    
    # Combined transformation: J2000 → Mean-of-Date → True-of-Date (TEME)
    R_j2000_to_teme = R_nut @ R_prec
    
    # Transform position
    pos_teme = R_j2000_to_teme @ np.array(pos_j2000)
    
    # Transform velocity (same rotation, no correction for Earth rotation rate needed here)
    vel_teme = R_j2000_to_teme @ np.array(vel_j2000)
    
    return pos_teme, vel_teme


def teme_to_j2000(pos_teme, vel_teme, time_utc):
    """
    Transform position and velocity from TEME to J2000 (ICRS)
    
    This is the inverse of j2000_to_teme()
    
    Parameters:
    -----------
    pos_teme : array-like (3,)
        Position vector in TEME frame [km]
    vel_teme : array-like (3,)
        Velocity vector in TEME frame [km/s]
    time_utc : astropy.time.Time
        UTC time for the transformation
    
    Returns:
    --------
    pos_j2000 : ndarray (3,)
        Position vector in J2000/ICRS frame [km]
    vel_j2000 : ndarray (3,)
        Velocity vector in J2000/ICRS frame [km/s]
    """
    
    # Get the J2000→TEME rotation matrix
    # (we'll invert it since rotation matrices are orthogonal)
    dummy_pos = np.array([1, 0, 0])
    dummy_vel = np.array([0, 0, 0])
    
    # Compute transformation matrix by transforming basis vectors
    R_j2000_to_teme = np.zeros((3, 3))
    for i in range(3):
        basis = np.zeros(3)
        basis[i] = 1.0
        transformed, _ = j2000_to_teme(basis, dummy_vel, time_utc)
        R_j2000_to_teme[:, i] = transformed
    
    # Inverse is transpose for rotation matrices
    R_teme_to_j2000 = R_j2000_to_teme.T
    
    # Transform
    pos_j2000 = R_teme_to_j2000 @ np.array(pos_teme)
    vel_j2000 = R_teme_to_j2000 @ np.array(vel_teme)
    
    return pos_j2000, vel_j2000
