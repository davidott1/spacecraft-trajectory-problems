import numpy as np
from sgp4.api import Satrec
from sgp4.api import jday

from tle_propagator import propagate_tle
from model.coordinate_system_converter import CoordinateSystemConverter
from constants import CONVERTER, TIMEVALUES
from two_body_high_fidelity.src.model.two_body import PHYSICALCONSTANTS

def solve_kepler(M, e, tol=1e-12):
    """Solves Kepler's equation M = E - e*sin(E) for E using Newton's method."""
    E = M # Initial guess
    for _ in range(100): # Max 100 iterations
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        delta = f / f_prime
        E = E - delta
        if abs(delta) < tol:
            return E
    return E # Return best guess if convergence fails

def propagate_analytical(
    tle_line1: str, 
    tle_line2: str, 
    time_o: float, 
    time_f: float, 
    num_points: int
) -> dict:
    """
    Propagates a TLE using simplified analytical expressions for secular perturbations.
    This version is updated to use canonical SGP4 units (Earth radii, minutes).
    """
    # --- 1. Initialization from TLE and SGP4 Constants ---
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    
    # SGP4 constants
    J2 = 1.082616e-3  # SGP4 J2 constant
    KE = 0.0743669161 # sqrt(GM) in Earth radii^1.5 / min
    AE = 1.0          # Canonical distance unit is 1 Earth radius

    # Get initial mean elements at epoch from the satellite object (already in SGP4 units)
    n_0 = satellite.no_kozai # rad/min
    e_0 = satellite.ecco
    i_0 = satellite.inclo
    raan_0 = satellite.nodeo
    argp_0 = satellite.argpo
    ma_0 = satellite.mo
    bstar = satellite.bstar

    # --- 2. Calculate Secular Rates of Change (in SGP4 units) ---
    a_0 = (KE / n_0)**(2/3.0) # Semi-major axis in Earth radii
    p_0 = a_0 * (1 - e_0**2)

    # Rate of change of RAAN due to J2 (rad/min)
    raan_dot = - (3/2) * n_0 * J2 * (AE/p_0)**2 * np.cos(i_0)

    # Rate of change of Argument of Perigee due to J2 (rad/min)
    argp_dot = (3/4) * n_0 * J2 * (AE/p_0)**2 * (5 * np.cos(i_0)**2 - 1)

    # Rate of change of Mean Anomaly due to J2 (rad/min)
    ma_dot_j2 = (3/4) * n_0 * J2 * (AE/p_0)**2 * (3 * np.cos(i_0)**2 - 1) * np.sqrt(1 - e_0**2)

    # --- 3. Propagate Mean Elements over the time series ---
    time_vec_sec = np.linspace(time_o, time_f, num_points)
    state_vec = np.zeros((6, num_points))
    
    # Initialize COE time series dictionary
    coe_time_series = {
        'sma'  : np.zeros(num_points), 'ecc'  : np.zeros(num_points),
        'inc'  : np.zeros(num_points), 'raan' : np.zeros(num_points),
        'argp' : np.zeros(num_points), 'ma'   : np.zeros(num_points),
        'ta'   : np.zeros(num_points), 'ea'   : np.zeros(num_points),
    }
    
    # Use a single converter with SGP4's MU in km^3/s^2 for the final conversion
    mu_km_s2 = PHYSICALCONSTANTS.EARTH.GP
    coord_converter = CoordinateSystemConverter(mu_km_s2)
    earth_radius_km = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR

    for idx, t_sec in enumerate(time_vec_sec):
        dt_min = t_sec / 60.0 # Convert time from seconds to minutes

        # --- SECULAR PROPAGATION ---
        # This block calculates the smooth, long-term trend of the orbit.
        raan_t_secular = raan_0 + raan_dot * dt_min
        argp_t_secular = argp_0 + argp_dot * dt_min
        
        # Define terms from SGP4 theory for secular drag
        a1 = (KE / n_0)**(2/3.0)
        delta1 = (3/2) * (J2 / a1**2) * (3 * np.cos(i_0)**2 - 1) / (1 - e_0**2)**1.5
        a0 = a1 * (1 - (1/3) * delta1 - delta1**2 - (134/81) * delta1**3)
        delta0 = (3/2) * (J2 / a0**2) * (3 * np.cos(i_0)**2 - 1) / (1 - e_0**2)**1.5
        n0_prime = n_0 / (1 + delta0)
        a0_prime = a0 / (1 - delta0)

        # Update SMA (a) and Eccentricity (e) due to drag
        a_t_secular = a0_prime * (1 - bstar * delta0 * dt_min)**2
        e_t_secular = e_0 - bstar * (1 - e_0) * dt_min 
        if e_t_secular < 0: e_t_secular = 0

        # Update Mean Anomaly (M)
        ma_t_secular = ma_0 + n0_prime * dt_min + ma_dot_j2 * dt_min
        
        i_t_secular = i_0

        # --- PERIODIC VARIATIONS ---
        # This block adds the "wiggles" on top of the smooth secular trend.
        
        # Long-period periodic terms (due to J2, depend on argument of perigee)
        axnl = e_t_secular * np.cos(argp_t_secular)
        aynl = e_t_secular * np.sin(argp_t_secular) - (3/2) * J2 * (AE/p_0) * np.sin(i_0)
        
        # Add long-period variations to get the "mean" elements at time t
        e_t_mean = np.sqrt(axnl**2 + aynl**2)
        argp_t_mean = np.arctan2(aynl, axnl)
        
        # Short-period periodic terms (depend on position in orbit, i.e., mean anomaly)
        # These are simplified expressions for the primary effects.
        C = (J2 / p_0) * (3 * np.cos(i_0)**2 - 1) / 4.0
        delta_r = (J2 / p_0) * (1 - np.cos(i_0)**2) * np.cos(2 * argp_t_mean + 2 * ma_t_secular) / 4.0
        delta_u = - (J2 / p_0) * (7 * np.cos(i_0)**2 - 1) * np.sin(2 * argp_t_mean + 2 * ma_t_secular) / 8.0
        delta_raan = (3 * J2 / (2 * p_0**2)) * np.cos(i_0) * np.sin(2 * argp_t_mean + 2 * ma_t_secular)
        
        # Apply periodic variations to get final osculating elements
        a_t = a_t_secular * (1 - C * np.cos(2 * argp_t_mean + 2 * ma_t_secular))
        e_t = e_t_mean
        i_t = i_t_secular
        raan_t = raan_t_secular + delta_raan
        argp_t = argp_t_mean
        ma_t = ma_t_secular + delta_u

        # Normalize angles
        raan_t %= (2 * np.pi)
        argp_t %= (2 * np.pi)
        ma_t %= (2 * np.pi)

        # Convert final mean elements to state vector for this time step
        final_elements = {
            'sma': a_t * earth_radius_km, # Convert to km for coe2rv
            'ecc': e_t, 'inc': i_t,
            'raan': raan_t, 'argp': argp_t, 'ma': ma_t
        }
        pos, vel = coord_converter.coe2rv(final_elements)
        state_vec[:, idx] = np.concatenate((pos, vel))

        # Store COE values for plotting
        for key in ['sma', 'ecc', 'inc', 'raan', 'argp', 'ma']:
            coe_time_series[key][idx] = final_elements[key]
        
        # Calculate and store EA and TA from MA
        ea = solve_kepler(ma_t, e_t)
        # Convert EA to TA
        ta = 2 * np.arctan2(np.sqrt(1 + e_t) * np.sin(ea / 2), np.sqrt(1 - e_t) * np.cos(ea / 2))
        
        coe_time_series['ea'][idx] = ea
        coe_time_series['ta'][idx] = ta

    return {
        'success': True,
        'message': 'Custom analytical propagation complete.',
        'time': time_vec_sec,
        'state': state_vec,
        'coe': coe_time_series
    }

