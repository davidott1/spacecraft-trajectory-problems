import numpy as np
import math
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from model.coordinate_system_converter import CoordinateSystemConverter
from model.dynamics import PHYSICALCONSTANTS


def teme_to_j2000(teme_pos_vec, teme_vel_vec, jd_ut1):
    """
    Convert TEME (True Equator Mean Equinox) to J2000/GCRS using astropy.
    
    Input:
        teme_pos_vec: position in TEME frame [km]
        teme_vel_vec: velocity in TEME frame [km/s]
        jd_ut1: Julian date (UT1)
    
    Output:
        j2000_pos_vec: position in J2000/GCRS frame [km]
        j2000_vel_vec: velocity in J2000/GCRS frame [km/s]
    """
    # Create astropy Time object
    t = Time(jd_ut1, format='jd', scale='ut1')
    
    # Create CartesianRepresentation with velocity differential
    cart_rep = CartesianRepresentation(
        x=teme_pos_vec[0] * u.km, # type: ignore
        y=teme_pos_vec[1] * u.km, # type: ignore
        z=teme_pos_vec[2] * u.km, # type: ignore
        differentials=CartesianDifferential(
            d_x=teme_vel_vec[0] * u.km / u.s, # type: ignore
            d_y=teme_vel_vec[1] * u.km / u.s, # type: ignore
            d_z=teme_vel_vec[2] * u.km / u.s  # type: ignore
        )
    )
    
    # Create TEME coordinate
    teme_coord = TEME(cart_rep, obstime=t)
    
    # Transform to GCRS (J2000)
    gcrs_coord = teme_coord.transform_to(GCRS(obstime=t))
    
    # Extract position and velocity
    j2000_pos_vec = np.array([
        gcrs_coord.cartesian.x.to(u.km).value, # type: ignore
        gcrs_coord.cartesian.y.to(u.km).value, # type: ignore
        gcrs_coord.cartesian.z.to(u.km).value  # type: ignore
    ])
    j2000_vel_vec = np.array([
        gcrs_coord.velocity.d_x.to(u.km / u.s).value, # type: ignore
        gcrs_coord.velocity.d_y.to(u.km / u.s).value, # type: ignore
        gcrs_coord.velocity.d_z.to(u.km / u.s).value  # type: ignore
    ])
    
    return j2000_pos_vec, j2000_vel_vec


def modify_tle_bstar(tle_line1: str, bstar_value: float = 0.0) -> str:
    """
    Modify the B* drag term in TLE line 1.
    
    Input:
        tle_line1: First line of TLE
        bstar_value: New B* value (default 0.0)
    
    Output:
        Modified TLE line 1 with new B* value
    """
    # B* is in columns 54-61 (0-indexed: 53-60) in format: ±.nnnnn±n
    # Example: " 12345-3" means 0.12345 × 10^-3
    
    if bstar_value == 0.0:
        bstar_str = " 00000-0"
    else:
        # Convert to scientific notation and format
        if bstar_value != 0:
            exponent = math.floor(math.log10(abs(bstar_value)))
            mantissa = bstar_value / (10 ** exponent)
            sign = '-' if bstar_value < 0 else ' '
            exp_sign = '-' if exponent < 0 else '+'
            bstar_str = f"{sign}{abs(mantissa):.5f}[1:]{exp_sign}{abs(exponent)}"
        else:
            bstar_str = " 00000-0"
    
    # Replace B* in TLE line 1 (columns 53-60)
    modified_line1 = tle_line1[:53] + bstar_str + tle_line1[61:]
    
    # Recalculate checksum (last character)
    checksum = 0
    for char in modified_line1[:-1]:
        if char.isdigit():
            checksum += int(char)
        elif char == '-':
            checksum += 1
    modified_line1 = modified_line1[:-1] + str(checksum % 10)
    
    return modified_line1


def propagate_tle(
    tle_line1    : str,
    tle_line2    : str,
    time_o       : float,
    time_f       : float,
    num_points   : int = 1000,
    disable_drag : bool = False,
    to_j2000     : bool = True,
) -> dict:
    """
    Propagate a TLE using SGP4.
    
    Input:
        tle_line1    : First line of TLE
        tle_line2    : Second line of TLE
        time_o       : Initial time [s]
        time_f       : Final time [s]
        num_points   : Number of time points to evaluate
        disable_drag : If True, set B* drag term to zero
    
    Output:
        dict: Dictionary containing:
            - success : propagation success flag
            - time    : time array [s]
            - state   : state array [6 x num_points] [m, m/s]
            - coe     : classical orbital elements time series
    """
    # Modify TLE to disable drag if requested
    if disable_drag:
        tle_line1 = modify_tle_bstar(tle_line1, 0.0)
    
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
        error_code, teme_pos_vec, v_teme = satellite.sgp4(jd, fr)
        if error_code != 0:
            return {
                'success' : False,
                'message' : f'SGP4 error code: {error_code}',
                'time'    : time_array,
                'state'   : state_array,
                'coe'     : coe_time_series,
            }
        
        # Transform TEME to J2000/GCRS
        if to_j2000:
            j2000_pos_vec, j2000_vel_vec = teme_to_j2000(teme_pos_vec, v_teme, jd + fr)
        
        # Convert from km to m and km/s to m/s
        j2000_pos_vec = np.array(j2000_pos_vec) * 1000.0  # km -> m
        j2000_vel_vec = np.array(j2000_vel_vec) * 1000.0  # km/s -> m/s
        
        # Store state (TEME frame - approximately inertial for short propagations)
        state_array[0:3, i] = j2000_pos_vec
        state_array[3:6, i] = j2000_vel_vec
        
        # Compute osculating elements
        coe = coord_sys_converter.rv2coe(j2000_pos_vec, j2000_vel_vec)
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
    disable_drag : bool = False,
) -> np.ndarray:
    """
    Extract initial position and velocity from TLE at epoch.
    
    Input:
        tle_line1: First line of TLE
        tle_line2: Second line of TLE
        disable_drag: If True, set B* drag term to zero
    
    Output:
        initial_state: [x, y, z, vx, vy, vz] in meters and m/s
    """
    # Modify TLE to disable drag if requested
    if disable_drag:
        tle_line1 = modify_tle_bstar(tle_line1, 0.0)
    
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
    
    error_code, teme_pos_vec, v_teme = satellite.sgp4(jd, fr)
    
    if error_code != 0:
        raise ValueError(f'SGP4 error code: {error_code}')
    
    # Convert from km to m and km/s to m/s
    pos = np.array(teme_pos_vec) * 1000.0
    vel = np.array(v_teme) * 1000.0
    
    return np.concatenate([pos, vel])


def get_tle_osculating_state(
    tle_line1 : str,
    tle_line2 : str,
) -> np.ndarray:
    """
    Extract mean elements from TLE, convert to osculating, then to Cartesian state.
    
    Input:
        tle_line1: First line of TLE
        tle_line2: Second line of TLE
    
    Output:
        initial_state: [x, y, z, vx, vy, vz] in meters and m/s (osculating)
    """
    from model.coordinate_system_converter import CoordinateSystemConverter
    from model.brouwer_lyddane import BrouwerLyddane
    
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    
    # Extract mean elements from TLE
    n_mean = satellite.no_kozai  # Mean motion [rad/min]
    ecc = satellite.ecco         # Eccentricity
    inc = satellite.inclo        # Inclination [rad]
    raan = satellite.nodeo       # RAAN [rad]
    argp = satellite.argpo       # Argument of perigee [rad]
    ma = satellite.mo            # Mean anomaly [rad]
    
    # Convert mean motion to semi-major axis
    n_rad_per_sec = n_mean / 60.0
    sma = (PHYSICALCONSTANTS.EARTH.GP / (n_rad_per_sec**2))**(1/3)
    
    # Create mean COE dict
    mean_coe = {
        'sma'  : sma,
        'ecc'  : ecc,
        'inc'  : inc,
        'raan' : raan,
        'argp' : argp,
        'ma'   : ma,
    }
    
    # Convert mean to osculating elements
    bl = BrouwerLyddane()
    osc_coe = bl.mean_to_osculating(mean_coe)
    
    # Convert osculating elements to Cartesian state
    converter = CoordinateSystemConverter(PHYSICALCONSTANTS.EARTH.GP)
    pos, vel = converter.coe2rv(osc_coe)
    
    return np.concatenate([pos, vel])
