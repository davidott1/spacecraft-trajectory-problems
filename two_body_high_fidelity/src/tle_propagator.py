import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from model.coordinate_system_converter import CoordinateSystemConverter
from model.dynamics import PHYSICALCONSTANTS


def propagate_tle(
    tle_line1        : str,
    tle_line2        : str,
    time_o           : float,
    time_f           : float,
    num_points       : int = 1000,
) -> dict:
    """
    Propagate a TLE using SGP4.
    
    Input:
        tle_line1: First line of TLE
        tle_line2: Second line of TLE
        time_o: Initial time [s]
        time_f: Final time [s]
        num_points: Number of time points to evaluate
    
    Output:
        dict: Dictionary containing:
            - success: propagation success flag
            - time: time array [s]
            - state: state array [6 x num_points] [m, m/s]
            - coe: classical orbital elements time series
    """
    # Initialize satellite from TLE
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    
    # Get epoch from TLE
    year = satellite.epochyr
    if year < 57:
        year += 2000
    else:
        year += 1900
    
    epoch_days = satellite.epochdays
    epoch_datetime = datetime(year, 1, 1) + timedelta(days=epoch_days - 1)
    
    # Create time array
    time_array = np.linspace(time_o, time_f, num_points)
    
    # Initialize arrays
    state_array = np.zeros((6, num_points))
    coe_time_series = {
        'sma'  : np.zeros(num_points),
        'ecc'  : np.zeros(num_points),
        'inc'  : np.zeros(num_points),
        'raan' : np.zeros(num_points),
        'argp' : np.zeros(num_points),
        'ma'   : np.zeros(num_points),
        'ta'   : np.zeros(num_points),
        'ea'   : np.zeros(num_points),
    }
    
    coord_sys_converter = CoordinateSystemConverter(PHYSICALCONSTANTS.EARTH.GP)
    
    # Propagate at each time step
    for i, t in enumerate(time_array):
        # Convert time to datetime
        dt = epoch_datetime + timedelta(seconds=float(t))
        
        # Get Julian date
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond/1e6)
        
        # Propagate
        error_code, r_teme, v_teme = satellite.sgp4(jd, fr)
        
        if error_code != 0:
            return {
                'success': False,
                'message': f'SGP4 error code: {error_code}',
                'time': time_array,
                'state': state_array,
                'coe': coe_time_series,
            }
        
        # Convert from km to m and km/s to m/s
        pos = np.array(r_teme) * 1000.0  # km -> m
        vel = np.array(v_teme) * 1000.0  # km/s -> m/s
        
        # Store state (TEME frame - approximately inertial for short propagations)
        state_array[0:3, i] = pos
        state_array[3:6, i] = vel
        
        # Compute osculating elements
        coe = coord_sys_converter.rv2coe(pos, vel)
        for key in coe_time_series.keys():
            coe_time_series[key][i] = coe[key]
    
    return {
        'success': True,
        'message': 'SGP4 propagation successful',
        'time': time_array,
        'state': state_array,
        'final_state': state_array[:, -1],
        'coe': coe_time_series,
    }


def get_tle_initial_state(
    tle_line1 : str,
    tle_line2 : str,
) -> np.ndarray:
    """
    Extract initial position and velocity from TLE at epoch.
    
    Input:
        tle_line1: First line of TLE
        tle_line2: Second line of TLE
    
    Output:
        initial_state: [x, y, z, vx, vy, vz] in meters and m/s
    """
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    
    # Get epoch
    year = satellite.epochyr
    if year < 57:
        year += 2000
    else:
        year += 1900
    
    epoch_days = satellite.epochdays
    epoch_datetime = datetime(year, 1, 1) + timedelta(days=epoch_days - 1)
    
    jd, fr = jday(epoch_datetime.year, epoch_datetime.month, epoch_datetime.day,
                  epoch_datetime.hour, epoch_datetime.minute, 
                  epoch_datetime.second + epoch_datetime.microsecond/1e6)
    
    error_code, r_teme, v_teme = satellite.sgp4(jd, fr)
    
    if error_code != 0:
        raise ValueError(f'SGP4 error code: {error_code}')
    
    # Convert from km to m and km/s to m/s
    pos = np.array(r_teme) * 1000.0
    vel = np.array(v_teme) * 1000.0
    
    return np.concatenate([pos, vel])
