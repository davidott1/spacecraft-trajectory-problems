import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skyfield.api import load, EarthSatellite
from skyfield.timelib import Time
import pytz

from src.model.constants import CONVERTER, PHYSICALCONSTANTS
from src.model.two_body import CoordinateSystemConverter

# Initialize coordinate converter
coord_sys_converter = CoordinateSystemConverter(gp=PHYSICALCONSTANTS.EARTH.GP)

def load_horizons_data(filepath):
    """
    Load Horizons ephemeris data from CSV with units row.
    """
    # Read CSV, skipping the units row (row 1)
    df = pd.read_csv(filepath, skiprows=[1])
    
    # Convert datetime string to datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Data is already in meters and m/s, convert to km and km/s
    df['pos_x__km']       = df['pos_x'] * CONVERTER.KM_PER_M
    df['pos_y__km']       = df['pos_y'] * CONVERTER.KM_PER_M
    df['pos_z__km']       = df['pos_z'] * CONVERTER.KM_PER_M
    df['vel_x__km_per_s'] = df['vel_x'] * CONVERTER.KM_PER_M
    df['vel_y__km_per_s'] = df['vel_y'] * CONVERTER.KM_PER_M
    df['vel_z__km_per_s'] = df['vel_z'] * CONVERTER.KM_PER_M
    
    return df

def compute_orbital_elements(df):
    """
    Compute orbital elements for each time step.
    """
    # List to hold results
    elements = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        # Convert km to m for the converter
        pos_vec = np.array([row['pos_x__km']      , row['pos_y__km']      , row['pos_z__km']      ]) * CONVERTER.M_PER_KM  # [m]
        vel_vec = np.array([row['vel_x__km_per_s'], row['vel_y__km_per_s'], row['vel_z__km_per_s']]) * CONVERTER.M_PER_KM  # [m/s]
        
        # Compute classical orbital elements
        coe = coord_sys_converter.rv2coe(pos_vec, vel_vec)

        # Store elements
        elements.append({
            'datetime' : row['datetime'], 
            'sma'      : coe['sma' ] * CONVERTER.M_PER_KM,     # [m]
            'ecc'      : coe['ecc' ],                          # [-]
            'inc'      : coe['inc' ] * CONVERTER.DEG_PER_RAD,  # [deg]
            'raan'     : coe['raan'] * CONVERTER.DEG_PER_RAD,  # [deg]
            'argp'     : coe['argp'] * CONVERTER.DEG_PER_RAD,  # [deg]
            'ta'       : coe['ta'  ] * CONVERTER.DEG_PER_RAD,  # [deg]
        })
    
    return pd.DataFrame(elements)

def load_tle_data(filepath):
    """
    Load TLE data from file with multiple TLEs.
    """

    # List to hold TLEs
    tles = []

    # Read all lines from the file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse TLEs (every 2 lines is one TLE)
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            line1 = lines[i].strip()
            line2 = lines[i + 1].strip()
            if line1.startswith('1 ') and line2.startswith('2 '):
                tles.append((line1, line2))
    
    return tles

def propagate_tle_to_times(
    tle_line1,
    tle_line2,
    times,
):
    """
    Propagate a TLE to specified times using Skyfield.
    
    Input
        tle_line1, tle_line2 : TLE lines
        times                : pandas Series of datetime objects
    Output
        DataFrame with propagated positions and velocities
    """
    import pytz
    ts = load.timescale()
    satellite = EarthSatellite(tle_line1, tle_line2, 'ISS', ts)
    
    results = []
    for dt in times:
        # Ensure datetime is timezone-aware (UTC)
        dt_py = dt.to_pydatetime()
        if dt_py.tzinfo is None:
            dt_py = pytz.utc.localize(dt_py)
        
        t = ts.from_datetime(dt_py)
        geocentric = satellite.at(t)
        
        # Get position and velocity in km and km/s
        pos_vec:np.ndarray = geocentric.position.km        # type: ignore
        vel_vec:np.ndarray = geocentric.velocity.km_per_s  # type: ignore

        results.append({
            'datetime'        : dt,
            'pos_x__km'       : float(pos_vec[0]),
            'pos_y__km'       : float(pos_vec[1]),
            'pos_z__km'       : float(pos_vec[2]),
            'vel_x__km_per_s' : float(vel_vec[0]),
            'vel_y__km_per_s' : float(vel_vec[1]),
            'vel_z__km_per_s' : float(vel_vec[2])
        })
    
    return pd.DataFrame(results)

def propagate_all_tles_and_select_best(tles, times):
    """
    Propagate all TLEs to specified times and select the best TLE state for each time.
    
    Args:
        tles: List of (line1, line2) tuples
        times: pandas Series of datetime objects
    
    Returns:
        DataFrame with propagated positions and velocities from the best TLE at each time
    """
    import pytz
    ts = load.timescale()
    
    # Get TLE epochs
    tle_epochs = []
    satellites = []
    for line1, line2 in tles:
        sat = EarthSatellite(line1, line2, 'ISS', ts)
        satellites.append(sat)
        epoch_dt = sat.epoch.utc_datetime()
        if epoch_dt.tzinfo is None: # type: ignore
            epoch_dt = pytz.utc.localize(epoch_dt) # type: ignore
        tle_epochs.append(epoch_dt)
    
    results = []
    for dt in times:
        # Ensure datetime is timezone-aware (UTC)
        dt_py = dt.to_pydatetime()
        if dt_py.tzinfo is None:
            dt_py = pytz.utc.localize(dt_py)
        
        # Find closest TLE epoch
        min_diff = float('inf')
        best_idx = 0
        for idx, tle_epoch in enumerate(tle_epochs):
            diff = abs((tle_epoch - dt_py).total_seconds())
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
        
        # Use the best TLE to propagate to this time
        t = ts.from_datetime(dt_py)
        geocentric = satellites[best_idx].at(t)
        
        # Get position and velocity in km and km/s
        pos_vec:np.ndarray = geocentric.position.km        # type: ignore
        vel_vec:np.ndarray = geocentric.velocity.km_per_s  # type: ignore

        results.append({
            'datetime'        : dt,
            'pos_x__km'       : float(pos_vec[0]),
            'pos_y__km'       : float(pos_vec[1]),
            'pos_z__km'       : float(pos_vec[2]),
            'vel_x__km_per_s' : float(vel_vec[0]),
            'vel_y__km_per_s' : float(vel_vec[1]),
            'vel_z__km_per_s' : float(vel_vec[2]),
            'tle_index'       : best_idx
        })
    
    return pd.DataFrame(results)

def get_tle_epoch_states(
    tles : list[tuple[str, str]]
) -> pd.DataFrame:
    """
    Get the state vector at the epoch time for each TLE.
    
    Args:
        tles: List of (line1, line2) tuples
    
    Returns:
        DataFrame with epoch times and state vectors
    """
    ts = load.timescale()
    
    results = []
    for line1, line2 in tles:
        sat = EarthSatellite(line1, line2, 'ISS', ts)
        epoch_time = sat.epoch
        
        # Get state at epoch
        geocentric = sat.at(epoch_time)
        pos_vec:np.ndarray = geocentric.position.km        # type: ignore
        vel_vec:np.ndarray = geocentric.velocity.km_per_s  # type: ignore

        # Convert epoch to datetime
        epoch_dt = epoch_time.utc_datetime()
        if epoch_dt.tzinfo is None: # type: ignore
            epoch_dt = pytz.utc.localize(epoch_dt) # type: ignore
        
        results.append({
            'datetime'        : epoch_dt,
            'pos_x__km'       : float(pos_vec[0]),
            'pos_y__km'       : float(pos_vec[1]),
            'pos_z__km'       : float(pos_vec[2]),
            'vel_x__km_per_s' : float(vel_vec[0]),
            'vel_y__km_per_s' : float(vel_vec[1]),
            'vel_z__km_per_s' : float(vel_vec[2])
        })
    
    return pd.DataFrame(results)

def get_best_tle_indices(horizons_df, tles):
    """
    Determine which TLE is best (closest epoch) for each time in horizons_df.
    
    Args:
        horizons_df: DataFrame with datetime column
        tles: List of (line1, line2) tuples
    
    Returns:
        DataFrame with datetime and tle_index columns
    """
    import pytz
    ts = load.timescale()
    
    # Get TLE epochs
    tle_epochs = []
    for line1, line2 in tles:
        sat = EarthSatellite(line1, line2, 'ISS', ts)
        epoch_dt = sat.epoch.utc_datetime()
        if epoch_dt.tzinfo is None: # type: ignore
            epoch_dt = pytz.utc.localize(epoch_dt) # type: ignore
        tle_epochs.append(epoch_dt)
    
    # For each horizons time, find closest TLE
    results = []
    for dt in horizons_df['datetime']:
        dt_py = dt.to_pydatetime()
        if dt_py.tzinfo is None:
            dt_py = pytz.utc.localize(dt_py)
        
        # Find closest TLE epoch
        min_diff = float('inf')
        best_idx = 0
        for idx, tle_epoch in enumerate(tle_epochs):
            diff = abs((tle_epoch - dt_py).total_seconds())
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
        
        results.append({
            'datetime': dt,
            'tle_index': best_idx
        })
    
    return pd.DataFrame(results)

def plot_horizons_vs_tle_with_index(horizons_df, tle_df, tle_epochs_df=None, tles=None):
    """Plot ISS position and velocity comparing Horizons and TLE data, with TLE index markers."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, 
                             gridspec_kw={'height_ratios': [1, 4, 4]})
    fig.suptitle('ISS State: Horizons vs TLE Propagation (with TLE Index)', fontsize=16)
    
    # Compute magnitudes
    r_mag_hor = np.sqrt(horizons_df['pos_x__km']**2 + horizons_df['pos_y__km']**2 + horizons_df['pos_z__km']**2)
    v_mag_hor = np.sqrt(horizons_df['vel_x__km_per_s']**2 + horizons_df['vel_y__km_per_s']**2 + horizons_df['vel_z__km_per_s']**2)
    
    r_mag_tle = np.sqrt(tle_df['pos_x__km']**2 + tle_df['pos_y__km']**2 + tle_df['pos_z__km']**2)
    v_mag_tle = np.sqrt(tle_df['vel_x__km_per_s']**2 + tle_df['vel_y__km_per_s']**2 + tle_df['vel_z__km_per_s']**2)
    
    # Store transition times for vertical lines
    transition_times = []
    
    # TLE Index plot
    ax = axes[0]
    if tles is not None:
        # Get best TLE index for each time
        tle_index_df = get_best_tle_indices(horizons_df, tles)
        
        # Find continuous segments of the same TLE index
        current_idx = tle_index_df.iloc[0]['tle_index']
        start_time = tle_index_df.iloc[0]['datetime']
        
        for i in range(1, len(tle_index_df)):
            if tle_index_df.iloc[i]['tle_index'] != current_idx or i == len(tle_index_df) - 1:
                # End of segment
                if i == len(tle_index_df) - 1 and tle_index_df.iloc[i]['tle_index'] == current_idx:
                    end_time = tle_index_df.iloc[i]['datetime']
                else:
                    end_time = tle_index_df.iloc[i-1]['datetime']
                
                # Ensure timezone consistency
                if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                    if not hasattr(end_time, 'tzinfo') or end_time.tzinfo is None:
                        import pytz
                        end_time = pd.Timestamp(end_time).tz_localize('UTC')
                
                # Plot horizontal line for this segment
                ax.hlines(y=0, xmin=start_time, xmax=end_time, colors='black', linewidth=3)
                
                # Add vertical line at segment start
                ax.vlines(x=start_time, ymin=-0.3, ymax=0.3, colors='black', linewidth=2)
                transition_times.append(start_time)
                
                # Add text label in the middle of the segment
                mid_time = start_time + (end_time - start_time) / 2
                ax.text(mid_time, 0, f'TLE {current_idx}', ha='center', va='center',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Start new segment
                current_idx = tle_index_df.iloc[i]['tle_index']
                start_time = tle_index_df.iloc[i]['datetime']
        
        # Add final vertical line at the end
        final_time = tle_index_df.iloc[-1]['datetime']
        ax.vlines(x=final_time, ymin=-0.3, ymax=0.3, colors='black', linewidth=2)
        transition_times.append(final_time)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    
    ax.set_ylabel('Active TLE', fontsize=12)
    ax.set_title('TLE Timeline', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Position plot
    ax = axes[1]
    ax.plot(horizons_df['datetime'], horizons_df['pos_x__km'], 'r-', label='X (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['pos_y__km'], 'g-', label='Y (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['pos_z__km'], 'b-', label='Z (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], r_mag_hor, 'k-', label='Magnitude (Horizons)', linewidth=1.5, alpha=0.7)
    
    ax.plot(tle_df['datetime'], tle_df['pos_x__km'], 'r--', label='X (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['pos_y__km'], 'g--', label='Y (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['pos_z__km'], 'b--', label='Z (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], r_mag_tle, 'k--', label='Magnitude (TLE)', linewidth=1.5, alpha=0.7)
    
    # Plot TLE epoch positions as markers
    if tle_epochs_df is not None:
        r_mag_epochs = np.sqrt(tle_epochs_df['pos_x__km']**2 + tle_epochs_df['pos_y__km']**2 + tle_epochs_df['pos_z__km']**2)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['pos_x__km'], c='red', marker='o', s=50, zorder=5)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['pos_y__km'], c='green', marker='o', s=50, zorder=5)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['pos_z__km'], c='blue', marker='o', s=50, zorder=5)
        ax.scatter(tle_epochs_df['datetime'], r_mag_epochs, c='black', marker='o', s=50, zorder=5)
    
    # Add vertical dotted lines at TLE transitions
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('Position (km)', fontsize=12)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Velocity plot
    ax = axes[2]
    ax.plot(horizons_df['datetime'], horizons_df['vel_x__km_per_s'], 'r-', label='X (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['vel_y__km_per_s'], 'g-', label='Y (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['vel_z__km_per_s'], 'b-', label='Z (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], v_mag_hor, 'k-', label='Magnitude (Horizons)', linewidth=1.5, alpha=0.7)
    
    ax.plot(tle_df['datetime'], tle_df['vel_x__km_per_s'], 'r--', label='X (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['vel_y__km_per_s'], 'g--', label='Y (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['vel_z__km_per_s'], 'b--', label='Z (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], v_mag_tle, 'k--', label='Magnitude (TLE)', linewidth=1.5, alpha=0.7)
    
    # Plot TLE epoch velocities as markers
    if tle_epochs_df is not None:
        v_mag_epochs = np.sqrt(tle_epochs_df['vel_x__km_per_s']**2 + tle_epochs_df['vel_y__km_per_s']**2 + tle_epochs_df['vel_z__km_per_s']**2)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vel_x__km_per_s'], c='red', marker='o', s=50, zorder=5)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vel_y__km_per_s'], c='green', marker='o', s=50, zorder=5)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vel_z__km_per_s'], c='blue', marker='o', s=50, zorder=5)
        ax.scatter(tle_epochs_df['datetime'], v_mag_epochs, c='black', marker='o', s=50, zorder=5)
    
    # Add vertical dotted lines at TLE transitions
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('Velocity (km/s)', fontsize=12)
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=90)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))

    # Rotate x-axis labels
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    return fig

def plot_orbital_elements_tle_comparison(horizons_oe_df, tle_oe_df, tles=None, tle_epochs_df=None):
    """
    Plot orbital elements comparing Horizons and TLE data, with TLE index markers.
    """
    fig, axes = plt.subplots(7, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 2, 2, 2, 2, 2]})
    fig.suptitle('ISS Orbital Elements: Horizons vs TLE Propagation (with TLE Index)', fontsize=16)
    
    # Compute orbital elements for TLE epochs if provided
    tle_epochs_oe_df = None
    if tle_epochs_df is not None:
        tle_epochs_oe_df = compute_orbital_elements(tle_epochs_df)
    
    # Store transition times for vertical lines
    transition_times = []
    
    # TLE Index plot
    ax = axes[0]
    if tles is not None:
        # Get best TLE index for each time
        tle_index_df = get_best_tle_indices(horizons_oe_df, tles)
        
        # Find continuous segments of the same TLE index
        current_idx = tle_index_df.iloc[0]['tle_index']
        start_time = tle_index_df.iloc[0]['datetime']
        
        for i in range(1, len(tle_index_df)):
            if tle_index_df.iloc[i]['tle_index'] != current_idx or i == len(tle_index_df) - 1:
                # End of segment
                if i == len(tle_index_df) - 1 and tle_index_df.iloc[i]['tle_index'] == current_idx:
                    end_time = tle_index_df.iloc[i]['datetime']
                else:
                    end_time = tle_index_df.iloc[i-1]['datetime']
                
                # Ensure timezone consistency
                if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                    if not hasattr(end_time, 'tzinfo') or end_time.tzinfo is None:
                        import pytz
                        end_time = pd.Timestamp(end_time).tz_localize('UTC')
                
                # Plot horizontal line for this segment
                ax.hlines(y=0, xmin=start_time, xmax=end_time, colors='black', linewidth=3)
                
                # Add vertical line at segment start
                ax.vlines(x=start_time, ymin=-0.3, ymax=0.3, colors='black', linewidth=2)
                transition_times.append(start_time)
                
                # Add text label in the middle of the segment
                mid_time = start_time + (end_time - start_time) / 2
                ax.text(mid_time, 0, f'TLE {current_idx}', ha='center', va='center',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Start new segment
                current_idx = tle_index_df.iloc[i]['tle_index']
                start_time = tle_index_df.iloc[i]['datetime']
        
        # Add final vertical line at the end
        final_time = tle_index_df.iloc[-1]['datetime']
        ax.vlines(x=final_time, ymin=-0.3, ymax=0.3, colors='black', linewidth=2) 
        transition_times.append(final_time)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    
    # Labeling for TLE index plot
    ax.set_ylabel('Active TLE', fontsize=12)
    ax.set_title('TLE Timeline', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Semi-major axis
    ax = axes[1]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['sma'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['sma'], 'k--', label='TLE', linewidth=1.5, alpha=0.7)
    if tle_epochs_oe_df is not None:
        ax.scatter(tle_epochs_oe_df['datetime'], tle_epochs_oe_df['sma'], c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Semi-major axis (km)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Eccentricity
    ax = axes[2]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['ecc'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['ecc'], 'k--', label='TLE', linewidth=1.5, alpha=0.7)
    if tle_epochs_oe_df is not None:
        ax.scatter(tle_epochs_oe_df['datetime'], tle_epochs_oe_df['ecc'], c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Eccentricity', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Inclination
    ax = axes[3]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['inc'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['inc'], 'k--', label='TLE', linewidth=1.5, alpha=0.7)
    if tle_epochs_oe_df is not None:
        ax.scatter(tle_epochs_oe_df['datetime'], tle_epochs_oe_df['inc'], c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Inclination (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # RAAN
    ax = axes[4]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['raan'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['raan'], 'k--', label='TLE', linewidth=1.5, alpha=0.7)
    if tle_epochs_oe_df is not None:
        ax.scatter(tle_epochs_oe_df['datetime'], tle_epochs_oe_df['raan'], c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('RAAN (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Argument of Periapsis
    ax = axes[5]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['argp'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['argp'], 'k--', label='TLE', linewidth=1.5, alpha=0.7)
    if tle_epochs_oe_df is not None:
        ax.scatter(tle_epochs_oe_df['datetime'], tle_epochs_oe_df['argp'], c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Arg. of Periapsis (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # True Anomaly
    ax = axes[6]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['ta'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['ta'], 'k--', label='TLE', linewidth=1.5, alpha=0.7)
    if tle_epochs_oe_df is not None:
        ax.scatter(tle_epochs_oe_df['datetime'], tle_epochs_oe_df['ta'], c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('True Anomaly (deg)', fontsize=11)
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=90)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    
    # Rotate x-axis labels
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    return fig

def plot_position_velocity_errors(horizons_df, tle_df, tles=None, tle_epochs_df=None):
    """Plot position and velocity errors between Horizons and TLE data, with TLE index markers."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 4, 4]})
    fig.suptitle('ISS Position and Velocity Errors: Horizons - TLE (with TLE Index)', fontsize=16)
    
    # Compute errors
    pos_error_x = horizons_df['pos_x__km'].values - tle_df['pos_x__km'].values
    pos_error_y = horizons_df['pos_y__km'].values - tle_df['pos_y__km'].values
    pos_error_z = horizons_df['pos_z__km'].values - tle_df['pos_z__km'].values
    pos_error_mag = np.sqrt(pos_error_x**2 + pos_error_y**2 + pos_error_z**2)
    
    vel_error_x = horizons_df['vel_x__km_per_s'].values - tle_df['vel_x__km_per_s'].values
    vel_error_y = horizons_df['vel_y__km_per_s'].values - tle_df['vel_y__km_per_s'].values
    vel_error_z = horizons_df['vel_z__km_per_s'].values - tle_df['vel_z__km_per_s'].values
    vel_error_mag = np.sqrt(vel_error_x**2 + vel_error_y**2 + vel_error_z**2)
    
    # Store transition times for vertical lines
    transition_times = []
    
    # TLE Index plot
    ax = axes[0]
    if tles is not None:
        tle_index_df = get_best_tle_indices(horizons_df, tles)
        
        current_idx = tle_index_df.iloc[0]['tle_index']
        start_time = tle_index_df.iloc[0]['datetime']
        
        for i in range(1, len(tle_index_df)):
            if tle_index_df.iloc[i]['tle_index'] != current_idx or i == len(tle_index_df) - 1:
                if i == len(tle_index_df) - 1 and tle_index_df.iloc[i]['tle_index'] == current_idx:
                    end_time = tle_index_df.iloc[i]['datetime']
                else:
                    end_time = tle_index_df.iloc[i-1]['datetime']
                
                if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                    if not hasattr(end_time, 'tzinfo') or end_time.tzinfo is None:
                        end_time = pd.Timestamp(end_time).tz_localize('UTC')
                
                ax.hlines(y=0, xmin=start_time, xmax=end_time, colors='black', linewidth=3)
                ax.vlines(x=start_time, ymin=-0.3, ymax=0.3, colors='black', linewidth=2)
                transition_times.append(start_time)
                
                mid_time = start_time + (end_time - start_time) / 2
                ax.text(mid_time, 0, f'TLE {current_idx}', ha='center', va='center',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                current_idx = tle_index_df.iloc[i]['tle_index']
                start_time = tle_index_df.iloc[i]['datetime']
        
        final_time = tle_index_df.iloc[-1]['datetime']
        ax.vlines(x=final_time, ymin=-0.3, ymax=0.3, colors='black', linewidth=2)
        transition_times.append(final_time)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    
    ax.set_ylabel('Active TLE', fontsize=12)
    ax.set_title('TLE Timeline', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Position error plot
    ax = axes[1]
    ax.plot(horizons_df['datetime'], pos_error_x, 'r-', label='X error', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], pos_error_y, 'g-', label='Y error', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], pos_error_z, 'b-', label='Z error', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], pos_error_mag, 'k-', label='Magnitude', linewidth=2)
    # Add markers at TLE epochs (error should be zero at epoch times)
    if tle_epochs_df is not None:
        ax.scatter(tle_epochs_df['datetime'], np.zeros(len(tle_epochs_df)), c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Position Error (km)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Velocity error plot
    ax = axes[2]
    ax.plot(horizons_df['datetime'], vel_error_x, 'r-', label='X error', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], vel_error_y, 'g-', label='Y error', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], vel_error_z, 'b-', label='Z error', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], vel_error_mag, 'k-', label='Magnitude', linewidth=2)
    # Add markers at TLE epochs (error should be zero at epoch times)
    if tle_epochs_df is not None:
        ax.scatter(tle_epochs_df['datetime'], np.zeros(len(tle_epochs_df)), c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Velocity Error (km/s)', fontsize=12)
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='right')
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    
    return fig

def plot_orbital_element_errors(horizons_oe_df, tle_oe_df, tles=None, tle_epochs_df=None):
    """Plot orbital element errors between Horizons and TLE data, with TLE index markers."""
    fig, axes = plt.subplots(7, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 2, 2, 2, 2, 2]})
    fig.suptitle('ISS Orbital Element Errors: Horizons - TLE (with TLE Index)', fontsize=16)
    
    # Compute errors
    sma_error  = horizons_oe_df['sma' ].values - tle_oe_df['sma' ].values
    ecc_error  = horizons_oe_df['ecc' ].values - tle_oe_df['ecc' ].values
    inc_error  = horizons_oe_df['inc' ].values - tle_oe_df['inc' ].values
    raan_error = horizons_oe_df['raan'].values - tle_oe_df['raan'].values
    argp_error = horizons_oe_df['argp'].values - tle_oe_df['argp'].values
    ta_error   = horizons_oe_df['ta'  ].values - tle_oe_df['ta'  ].values
    
    # Store transition times for vertical lines
    transition_times = []
    
    # TLE Index plot
    ax = axes[0]
    if tles is not None:
        tle_index_df = get_best_tle_indices(horizons_oe_df, tles)
        
        current_idx = tle_index_df.iloc[0]['tle_index']
        start_time = tle_index_df.iloc[0]['datetime']
        
        for i in range(1, len(tle_index_df)):
            if tle_index_df.iloc[i]['tle_index'] != current_idx or i == len(tle_index_df) - 1:
                # End of segment
                if i == len(tle_index_df) - 1 and tle_index_df.iloc[i]['tle_index'] == current_idx:
                    end_time = tle_index_df.iloc[i]['datetime']
                else:
                    end_time = tle_index_df.iloc[i-1]['datetime']
                
                # Ensure timezone consistency
                if hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
                    if not hasattr(end_time, 'tzinfo') or end_time.tzinfo is None:
                        end_time = pd.Timestamp(end_time).tz_localize('UTC')
                
                # Plot horizontal line for this segment
                ax.hlines(y=0, xmin=start_time, xmax=end_time, colors='black', linewidth=3)
                
                # Add vertical line at segment start
                ax.vlines(x=start_time, ymin=-0.3, ymax=0.3, colors='black', linewidth=2)
                transition_times.append(start_time)
                
                # Add text label in the middle of the segment
                mid_time = start_time + (end_time - start_time) / 2
                ax.text(mid_time, 0, f'TLE {current_idx}', ha='center', va='center',
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Start new segment
                current_idx = tle_index_df.iloc[i]['tle_index']
                start_time = tle_index_df.iloc[i]['datetime']
        
        # Add final vertical line at the end
        final_time = tle_index_df.iloc[-1]['datetime']
        ax.vlines(x=final_time, ymin=-0.3, ymax=0.3, colors='black', linewidth=2) 
        transition_times.append(final_time)
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    
    ax.set_ylabel('Active TLE', fontsize=10)
    ax.set_title('TLE Timeline', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Semi-major axis error
    ax = axes[1]
    ax.plot(horizons_oe_df['datetime'], sma_error, 'k-', linewidth=1.5)
    if tle_epochs_df is not None:
        ax.scatter(tle_epochs_df['datetime'], np.zeros(len(tle_epochs_df)), c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('a error (km)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Eccentricity error
    ax = axes[2]
    ax.plot(horizons_oe_df['datetime'], ecc_error, 'k-', linewidth=1.5)
    if tle_epochs_df is not None:
        ax.scatter(tle_epochs_df['datetime'], np.zeros(len(tle_epochs_df)), c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('e error', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Inclination error
    ax = axes[3]
    ax.plot(horizons_oe_df['datetime'], inc_error, 'k-', linewidth=1.5)
    if tle_epochs_df is not None:
        ax.scatter(tle_epochs_df['datetime'], np.zeros(len(tle_epochs_df)), c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('i error (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # RAAN error
    ax = axes[4]
    ax.plot(horizons_oe_df['datetime'], raan_error, 'k-', linewidth=1.5)
    if tle_epochs_df is not None:
        ax.scatter(tle_epochs_df['datetime'], np.zeros(len(tle_epochs_df)), c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('RAAN error (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Argument of Periapsis error
    ax = axes[5]
    ax.plot(horizons_oe_df['datetime'], argp_error, 'k-', linewidth=1.5)
    if tle_epochs_df is not None:
        ax.scatter(tle_epochs_df['datetime'], np.zeros(len(tle_epochs_df)), c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('ω error (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # True Anomaly error
    ax = axes[6]
    ax.plot(horizons_oe_df['datetime'], ta_error, 'k-', linewidth=1.5)
    if tle_epochs_df is not None:
        ax.scatter(tle_epochs_df['datetime'], np.zeros(len(tle_epochs_df)), c='black', marker='o', s=50, zorder=5)
    for t in transition_times:
        ax.axvline(x=t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('ν error (deg)', fontsize=11)
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='right')
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    
    return fig

def main():
    # Define file paths
    base_path = Path(__file__).parent.parent.parent.parent / 'data'
    horizons_file = base_path / 'ephems' / 'horizons_ephem_25544_iss_20251001_20251008_1m.csv'
    tle_file      = base_path / 'ephems' / 'celestrak_tles_25544_iss_20251001_20251008.txt'
    
    # Define output directory
    output_folderpath = Path(__file__).parent.parent.parent / 'output' / 'figures'
    output_folderpath.mkdir(parents=True, exist_ok=True)
    
    # NORAD ID for ISS
    norad_id = 25544
    
    print(f"Loading Horizons data from: {horizons_file}")
    
    # Load data
    horizons_df = load_horizons_data(horizons_file)
    
    print(f"Loaded {len(horizons_df)} data points")
    print(f"Time range: {horizons_df['datetime'].min()} to {horizons_df['datetime'].max()}")
    print(f"\nSample data:")
    print(horizons_df[['datetime', 'pos_x__km', 'pos_y__km', 'pos_z__km', 'vel_x__km_per_s', 'vel_y__km_per_s', 'vel_z__km_per_s']].head())
    
    # Load TLEs
    print(f"\nLoading TLE data from: {tle_file}")
    tles = load_tle_data(tle_file)
    print(f"Loaded {len(tles)} TLEs")
    
    # Get TLE epoch states
    print("\nExtracting TLE epoch states...")
    tle_epochs_df = get_tle_epoch_states(tles)
    print(f"TLE epochs:")
    print(tle_epochs_df[['datetime']])
    
    # Propagate all TLEs and select best for each time
    print("\nPropagating all TLEs and selecting best for each time...")
    tle_df = propagate_all_tles_and_select_best(tles, horizons_df['datetime'])
    print(f"\nSample TLE data:")
    print(tle_df[['datetime', 'pos_x__km', 'pos_y__km', 'pos_z__km', 'tle_index']].head())
    print(f"\nTLE index distribution:")
    print(tle_df['tle_index'].value_counts().sort_index())
    
    # Compute orbital elements
    print("\nComputing orbital elements...")
    oe_df = compute_orbital_elements(horizons_df)
    print(f"\nSample orbital elements:")
    print(oe_df[['datetime', 'sma', 'ecc', 'inc', 'raan', 'argp', 'ta']].head())
    
    # Compute orbital elements for TLE data
    print("\nComputing orbital elements for TLE data...")
    tle_oe_df = compute_orbital_elements(tle_df)
    print(f"\nSample TLE orbital elements:")
    print(tle_oe_df[['datetime', 'sma', 'ecc', 'inc', 'raan', 'argp', 'ta']].head())
    
    # Create plots
    fig1 = plot_horizons_vs_tle_with_index(horizons_df, tle_df, tle_epochs_df, tles)
    fig2 = plot_orbital_elements_tle_comparison(oe_df, tle_oe_df, tles, tle_epochs_df)
    fig3 = plot_position_velocity_errors(horizons_df, tle_df, tles, tle_epochs_df)
    fig4 = plot_orbital_element_errors(oe_df, tle_oe_df, tles, tle_epochs_df)
    
    # Save plots
    output_file4 = output_folderpath / f'iss_{norad_id}_horizons_vs_tle_with_index_plot.png'
    fig1.savefig(output_file4, dpi=150, bbox_inches='tight')
    print(f"Horizons vs TLE with index plot saved to: {output_file4}")
    
    output_file5 = output_folderpath / f'iss_{norad_id}_orbital_elements_tle_comparison_plot.png'
    fig2.savefig(output_file5, dpi=150, bbox_inches='tight')
    print(f"Orbital elements TLE comparison plot saved to: {output_file5}")
    
    output_file6 = output_folderpath / f'iss_{norad_id}_position_velocity_errors_plot.png'
    fig3.savefig(output_file6, dpi=150, bbox_inches='tight')
    print(f"Position/velocity errors plot saved to: {output_file6}")
    
    output_file7 = output_folderpath / f'iss_{norad_id}_orbital_element_errors_plot.png'
    fig4.savefig(output_file7, dpi=150, bbox_inches='tight')
    print(f"Orbital element errors plot saved to: {output_file7}")
    
    plt.show()

if __name__ == '__main__':
    main()
