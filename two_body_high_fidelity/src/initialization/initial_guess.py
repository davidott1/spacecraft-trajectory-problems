import numpy as np
from model.dynamics import PHYSICALCONSTANTS
from initialization.utility import rotation_matrix

def get_initial_state(
    initial_guess_selection : str   = 'circular',
    alt                     : float = 500e3,
    sma                     : float = 0.0,
    ecc                     : float = 0.0,
    inc                     : float = 0.0,
    raan                    : float = 0.0,
    argp                    : float = 0.0,
    ta                      : float = 0.0,
) -> np.ndarray:
    """
    Generate an initial state vector based on the selected initial guess type.
    """
    if initial_guess_selection.lower() == 'circular':
        if ecc != 0.0:
            ecc = 0.0

        pos_mag_o = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR + alt
        pos_x     = pos_mag_o
        pos_y     = 0.0
        pos_z     = 0.0
        pos_vec   = np.array([pos_x, pos_y, pos_z])

        vel_x     = 0.0
        vel_y     = np.sqrt(PHYSICALCONSTANTS.EARTH.GP / pos_mag_o)
        vel_z     = 0.0
        vel_vec   = np.array([vel_x, vel_y, vel_z])
        vel_vec   = rotation_matrix(axis='x', angle=inc, rotate_type='vector') @ vel_vec
        vel_mag_o = np.sqrt(PHYSICALCONSTANTS.EARTH.GP / pos_mag_o)

        initial_state = np.array([
            pos_vec[0],
            pos_vec[1],
            pos_vec[2],
            vel_vec[0],
            vel_vec[1],
            vel_vec[2]
        ])

        return initial_state
    elif initial_guess_selection.lower() == 'elliptical':
        if ecc == 0.0:
            ecc = 0.1

        pos_mag_o  = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR + alt
        pos_x      = pos_mag_o
        pos_y      = 0.0
        pos_z      = 0.0
        pos_vec    = np.array([pos_x, pos_y, pos_z])

        vel_x      = 0.0
        vel_y      = np.sqrt(PHYSICALCONSTANTS.EARTH.GP / pos_mag_o)
        vel_y     *= np.sqrt((1 + ecc) / (1 - ecc))  # adjust for eccentricity
        vel_z      = 0.0
        vel_vec    = np.array([vel_x, vel_y, vel_z])
        vel_vec    = rotation_matrix(axis='x', angle=inc, rotate_type='vector') @ vel_vec
        vel_mag_o  = np.sqrt(PHYSICALCONSTANTS.EARTH.GP / pos_mag_o)

        initial_state = np.array([
            pos_vec[0],
            pos_vec[1],
            pos_vec[2],
            vel_vec[0],
            vel_vec[1],
            vel_vec[2]
        ])

        return initial_state
    else:
        pos_mag_o = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR + alt
        vel_mag_o = np.sqrt(PHYSICALCONSTANTS.EARTH.GP / pos_mag_o)
        initial_state = np.array([
            pos_mag_o,
            0.0,
            0.0,
            0.0,
            vel_mag_o,
            0.0,
        ])
        return initial_state