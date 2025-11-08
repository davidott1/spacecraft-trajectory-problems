import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skyfield.api import load, EarthSatellite
from skyfield.timelib import Time

# Constants
AU_TO_KM = 149597870.7  # km per AU
MU_EARTH = 398600.4418  # km^3/s^2, Earth's gravitational parameter

def load_horizons_data(filepath):
    """Load Horizons ephemeris data from CSV."""
    df = pd.read_csv(filepath)
    
    # Convert datetime string to datetime object
    df['datetime'] = pd.to_datetime(df['datetime_str_utc'])
    
    # Convert AU to km and AU/day to km/s
    df['x_km'] = df['x'] * AU_TO_KM
    df['y_km'] = df['y'] * AU_TO_KM
    df['z_km'] = df['z'] * AU_TO_KM
    df['vx_km_s'] = df['vx'] * AU_TO_KM / 86400
    df['vy_km_s'] = df['vy'] * AU_TO_KM / 86400
    df['vz_km_s'] = df['vz'] * AU_TO_KM / 86400
    
    return df

def cartesian_to_orbital_elements(x, y, z, vx, vy, vz, mu=MU_EARTH):
    """
    Convert Cartesian state vectors to classical orbital elements.
    
    Returns: a, e, i, raan, arg_pe, true_anomaly (all in degrees except a in km)
    """
    # Position and velocity vectors
    r_vec = np.array([x, y, z])
    v_vec = np.array([vx, vy, vz])
    
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # Node vector
    k_vec = np.array([0, 0, 1])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)
    
    # Eccentricity vector
    e_vec = ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)
    
    # Specific orbital energy
    epsilon = v**2 / 2 - mu / r
    
    # Semi-major axis
    if abs(e - 1.0) > 1e-10:
        a = -mu / (2 * epsilon)
    else:
        a = np.inf
    
    # Inclination
    i = np.arccos(h_vec[2] / h)
    
    # RAAN
    if n > 1e-10:
        raan = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0
    
    # Argument of periapsis
    if n > 1e-10 and e > 1e-10:
        arg_pe = np.arccos(np.dot(n_vec, e_vec) / (n * e))
        if e_vec[2] < 0:
            arg_pe = 2 * np.pi - arg_pe
    else:
        arg_pe = 0.0
    
    # True anomaly
    if e > 1e-10:
        true_anomaly = np.arccos(np.dot(e_vec, r_vec) / (e * r))
        if np.dot(r_vec, v_vec) < 0:
            true_anomaly = 2 * np.pi - true_anomaly
    else:
        true_anomaly = 0.0
    
    return a, e, np.degrees(i), np.degrees(raan), np.degrees(arg_pe), np.degrees(true_anomaly)

def compute_orbital_elements(df):
    """Compute orbital elements for each time step."""
    elements = []
    
    for idx, row in df.iterrows():
        a, e, i, raan, arg_pe, ta = cartesian_to_orbital_elements(
            row['x_km'], row['y_km'], row['z_km'],
            row['vx_km_s'], row['vy_km_s'], row['vz_km_s']
        )
        elements.append({
            'datetime': row['datetime'],
            'a': a,
            'e': e,
            'i': i,
            'raan': raan,
            'arg_pe': arg_pe,
            'ta': ta
        })
    
    return pd.DataFrame(elements)

def plot_horizons(df):
    """Plot ISS position and velocity from Horizons data."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('ISS State from Horizons Ephemeris', fontsize=16)
    
    # Compute magnitudes
    r_mag = np.sqrt(df['x_km']**2 + df['y_km']**2 + df['z_km']**2)
    v_mag = np.sqrt(df['vx_km_s']**2 + df['vy_km_s']**2 + df['vz_km_s']**2)
    
    # Position plot
    ax = axes[0]
    ax.plot(df['datetime'], df['x_km'], 'r-', label='X', linewidth=1.5)
    ax.plot(df['datetime'], df['y_km'], 'g-', label='Y', linewidth=1.5)
    ax.plot(df['datetime'], df['z_km'], 'b-', label='Z', linewidth=1.5)
    ax.plot(df['datetime'], r_mag, 'k-', label='Magnitude', linewidth=1.5)
    ax.set_ylabel('Position (km)', fontsize=12)
    ax.set_title('Position Components', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Velocity plot
    ax = axes[1]
    ax.plot(df['datetime'], df['vx_km_s'], 'r-', label='X', linewidth=1.5)
    ax.plot(df['datetime'], df['vy_km_s'], 'g-', label='Y', linewidth=1.5)
    ax.plot(df['datetime'], df['vz_km_s'], 'b-', label='Z', linewidth=1.5)
    ax.plot(df['datetime'], v_mag, 'k-', label='Magnitude', linewidth=1.5)
    ax.set_ylabel('Velocity (km/s)', fontsize=12)
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.set_title('Velocity Components', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Rotate x-axis labels
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    return fig

def plot_orbital_elements(oe_df):
    """Plot orbital elements from Horizons data."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
    fig.suptitle('ISS Orbital Elements from Horizons Ephemeris', fontsize=16)
    
    # Semi-major axis
    ax = axes[0, 0]
    ax.plot(oe_df['datetime'], oe_df['a'], 'k-', linewidth=1.5)
    ax.set_ylabel('Semi-major axis (km)', fontsize=11)
    ax.set_title('Semi-major Axis', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Eccentricity
    ax = axes[0, 1]
    ax.plot(oe_df['datetime'], oe_df['e'], 'k-', linewidth=1.5)
    ax.set_ylabel('Eccentricity', fontsize=11)
    ax.set_title('Eccentricity', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Inclination
    ax = axes[1, 0]
    ax.plot(oe_df['datetime'], oe_df['i'], 'k-', linewidth=1.5)
    ax.set_ylabel('Inclination (deg)', fontsize=11)
    ax.set_title('Inclination', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # RAAN
    ax = axes[1, 1]
    ax.plot(oe_df['datetime'], oe_df['raan'], 'k-', linewidth=1.5)
    ax.set_ylabel('RAAN (deg)', fontsize=11)
    ax.set_title('Right Ascension of Ascending Node', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Argument of Periapsis
    ax = axes[2, 0]
    ax.plot(oe_df['datetime'], oe_df['arg_pe'], 'k-', linewidth=1.5)
    ax.set_ylabel('Arg. of Periapsis (deg)', fontsize=11)
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.set_title('Argument of Periapsis', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # True Anomaly
    ax = axes[2, 1]
    ax.plot(oe_df['datetime'], oe_df['ta'], 'k-', linewidth=1.5)
    ax.set_ylabel('True Anomaly (deg)', fontsize=11)
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.set_title('True Anomaly', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in axes[-1, :]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    plt.tight_layout()
    
    return fig

def load_tle_data(filepath):
    """Load TLE data from file."""
    tles = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse TLEs (2 lines per TLE)
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            line1 = lines[i].strip()
            line2 = lines[i + 1].strip()
            tles.append((line1, line2))
    
    return tles

def propagate_tle_to_times(tle_line1, tle_line2, times):
    """
    Propagate a TLE to specified times using Skyfield.
    
    Args:
        tle_line1, tle_line2: TLE lines
        times: pandas Series of datetime objects
    
    Returns:
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
        pos = geocentric.position.km
        vel = geocentric.velocity.km_per_s
        
        results.append({
            'datetime': dt,
            'x_km': pos[0],
            'y_km': pos[1],
            'z_km': pos[2],
            'vx_km_s': vel[0],
            'vy_km_s': vel[1],
            'vz_km': vel[2]
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
        if epoch_dt.tzinfo is None:
            epoch_dt = pytz.utc.localize(epoch_dt)
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
        pos = geocentric.position.km
        vel = geocentric.velocity.km_per_s
        
        results.append({
            'datetime': dt,
            'x_km': pos[0],
            'y_km': pos[1],
            'z_km': pos[2],
            'vx_km_s': vel[0],
            'vy_km_s': vel[1],
            'vz_km_s': vel[2],
            'tle_index': best_idx
        })
    
    return pd.DataFrame(results)

def get_tle_epoch_states(tles):
    """
    Get the state vector at the epoch time for each TLE.
    
    Args:
        tles: List of (line1, line2) tuples
    
    Returns:
        DataFrame with epoch times and state vectors
    """
    import pytz
    ts = load.timescale()
    
    results = []
    for line1, line2 in tles:
        sat = EarthSatellite(line1, line2, 'ISS', ts)
        epoch_time = sat.epoch
        
        # Get state at epoch
        geocentric = sat.at(epoch_time)
        pos = geocentric.position.km
        vel = geocentric.velocity.km_per_s
        
        # Convert epoch to datetime
        epoch_dt = epoch_time.utc_datetime()
        if epoch_dt.tzinfo is None:
            epoch_dt = pytz.utc.localize(epoch_dt)
        
        results.append({
            'datetime': epoch_dt,
            'x_km': pos[0],
            'y_km': pos[1],
            'z_km': pos[2],
            'vx_km_s': vel[0],
            'vy_km_s': vel[1],
            'vz_km_s': vel[2]
        })
    
    return pd.DataFrame(results)

def plot_horizons_vs_tle(horizons_df, tle_df, tle_epochs_df=None):
    """Plot ISS position and velocity comparing Horizons and TLE data."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle('ISS State: Horizons vs TLE Propagation', fontsize=16)
    
    # Compute magnitudes
    r_mag_hor = np.sqrt(horizons_df['x_km']**2 + horizons_df['y_km']**2 + horizons_df['z_km']**2)
    v_mag_hor = np.sqrt(horizons_df['vx_km_s']**2 + horizons_df['vy_km_s']**2 + horizons_df['vz_km_s']**2)
    
    r_mag_tle = np.sqrt(tle_df['x_km']**2 + tle_df['y_km']**2 + tle_df['z_km']**2)
    v_mag_tle = np.sqrt(tle_df['vx_km_s']**2 + tle_df['vy_km_s']**2 + tle_df['vz_km_s']**2)
    
    # Position plot
    ax = axes[0]
    ax.plot(horizons_df['datetime'], horizons_df['x_km'], 'r-', label='X (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['y_km'], 'g-', label='Y (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['z_km'], 'b-', label='Z (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], r_mag_hor, 'k-', label='Magnitude (Horizons)', linewidth=1.5, alpha=0.7)
    
    ax.plot(tle_df['datetime'], tle_df['x_km'], 'r--', label='X (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['y_km'], 'g--', label='Y (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['z_km'], 'b--', label='Z (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], r_mag_tle, 'k--', label='Magnitude (TLE)', linewidth=1.5, alpha=0.7)
    
    # Plot TLE epoch positions as markers
    if tle_epochs_df is not None:
        r_mag_epochs = np.sqrt(tle_epochs_df['x_km']**2 + tle_epochs_df['y_km']**2 + tle_epochs_df['z_km']**2)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['x_km'], c='red', marker='o', s=50, zorder=5, label='TLE Epochs (X)')
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['y_km'], c='green', marker='o', s=50, zorder=5, label='TLE Epochs (Y)')
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['z_km'], c='blue', marker='o', s=50, zorder=5, label='TLE Epochs (Z)')
        ax.scatter(tle_epochs_df['datetime'], r_mag_epochs, c='black', marker='o', s=50, zorder=5, label='TLE Epochs (Mag)')
    
    ax.set_ylabel('Position (km)', fontsize=12)
    ax.set_title('Position Components', fontsize=14)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Velocity plot
    ax = axes[1]
    ax.plot(horizons_df['datetime'], horizons_df['vx_km_s'], 'r-', label='X (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['vy_km_s'], 'g-', label='Y (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['vz_km_s'], 'b-', label='Z (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], v_mag_hor, 'k-', label='Magnitude (Horizons)', linewidth=1.5, alpha=0.7)
    
    ax.plot(tle_df['datetime'], tle_df['vx_km_s'], 'r--', label='X (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['vy_km_s'], 'g--', label='Y (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['vz_km_s'], 'b--', label='Z (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], v_mag_tle, 'k--', label='Magnitude (TLE)', linewidth=1.5, alpha=0.7)
    
    # Plot TLE epoch velocities as markers
    if tle_epochs_df is not None:
        v_mag_epochs = np.sqrt(tle_epochs_df['vx_km_s']**2 + tle_epochs_df['vy_km_s']**2 + tle_epochs_df['vz_km_s']**2)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vx_km_s'], c='red', marker='o', s=50, zorder=5, label='TLE Epochs (X)')
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vy_km_s'], c='green', marker='o', s=50, zorder=5, label='TLE Epochs (Y)')
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vz_km_s'], c='blue', marker='o', s=50, zorder=5, label='TLE Epochs (Z)')
        ax.scatter(tle_epochs_df['datetime'], v_mag_epochs, c='black', marker='o', s=50, zorder=5, label='TLE Epochs (Mag)')
    
    ax.set_ylabel('Velocity (km/s)', fontsize=12)
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.set_title('Velocity Components', fontsize=14)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Rotate x-axis labels
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    return fig

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
        if epoch_dt.tzinfo is None:
            epoch_dt = pytz.utc.localize(epoch_dt)
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
    r_mag_hor = np.sqrt(horizons_df['x_km']**2 + horizons_df['y_km']**2 + horizons_df['z_km']**2)
    v_mag_hor = np.sqrt(horizons_df['vx_km_s']**2 + horizons_df['vy_km_s']**2 + horizons_df['vz_km_s']**2)
    
    r_mag_tle = np.sqrt(tle_df['x_km']**2 + tle_df['y_km']**2 + tle_df['z_km']**2)
    v_mag_tle = np.sqrt(tle_df['vx_km_s']**2 + tle_df['vy_km_s']**2 + tle_df['vz_km_s']**2)
    
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
    ax.plot(horizons_df['datetime'], horizons_df['x_km'], 'r-', label='X (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['y_km'], 'g-', label='Y (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['z_km'], 'b-', label='Z (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], r_mag_hor, 'k-', label='Magnitude (Horizons)', linewidth=1.5, alpha=0.7)
    
    ax.plot(tle_df['datetime'], tle_df['x_km'], 'r--', label='X (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['y_km'], 'g--', label='Y (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['z_km'], 'b--', label='Z (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], r_mag_tle, 'k--', label='Magnitude (TLE)', linewidth=1.5, alpha=0.7)
    
    # Plot TLE epoch positions as markers
    if tle_epochs_df is not None:
        r_mag_epochs = np.sqrt(tle_epochs_df['x_km']**2 + tle_epochs_df['y_km']**2 + tle_epochs_df['z_km']**2)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['x_km'], c='red', marker='o', s=50, zorder=5, label='TLE Epochs (X)')
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['y_km'], c='green', marker='o', s=50, zorder=5, label='TLE Epochs (Y)')
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['z_km'], c='blue', marker='o', s=50, zorder=5, label='TLE Epochs (Z)')
        ax.scatter(tle_epochs_df['datetime'], r_mag_epochs, c='black', marker='o', s=50, zorder=5, label='TLE Epochs (Mag)')
    
    # Add vertical dotted lines at TLE transitions
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('Position (km)', fontsize=12)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Velocity plot
    ax = axes[2]
    ax.plot(horizons_df['datetime'], horizons_df['vx_km_s'], 'r-', label='X (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['vy_km_s'], 'g-', label='Y (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], horizons_df['vz_km_s'], 'b-', label='Z (Horizons)', linewidth=1.5, alpha=0.7)
    ax.plot(horizons_df['datetime'], v_mag_hor, 'k-', label='Magnitude (Horizons)', linewidth=1.5, alpha=0.7)
    
    ax.plot(tle_df['datetime'], tle_df['vx_km_s'], 'r--', label='X (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['vy_km_s'], 'g--', label='Y (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], tle_df['vz_km_s'], 'b--', label='Z (TLE)', linewidth=1.5, alpha=0.7)
    ax.plot(tle_df['datetime'], v_mag_tle, 'k--', label='Magnitude (TLE)', linewidth=1.5, alpha=0.7)
    
    # Plot TLE epoch velocities as markers
    if tle_epochs_df is not None:
        v_mag_epochs = np.sqrt(tle_epochs_df['vx_km_s']**2 + tle_epochs_df['vy_km_s']**2 + tle_epochs_df['vz_km_s']**2)
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vx_km_s'], c='red', marker='o', s=50, zorder=5, label='TLE Epochs (X)')
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vy_km_s'], c='green', marker='o', s=50, zorder=5, label='TLE Epochs (Y)')
        ax.scatter(tle_epochs_df['datetime'], tle_epochs_df['vz_km_s'], c='blue', marker='o', s=50, zorder=5, label='TLE Epochs (Z)')
        ax.scatter(tle_epochs_df['datetime'], v_mag_epochs, c='black', marker='o', s=50, zorder=5, label='TLE Epochs (Mag)')
    
    # Add vertical dotted lines at TLE transitions
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('Velocity (km/s)', fontsize=12)
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Rotate x-axis labels
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    return fig

def plot_orbital_elements_tle_comparison(horizons_oe_df, tle_oe_df, tles=None):
    """Plot orbital elements comparing Horizons and TLE data, with TLE index markers."""
    fig, axes = plt.subplots(7, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 2, 2, 2, 2, 2]})
    fig.suptitle('ISS Orbital Elements: Horizons vs TLE Propagation (with TLE Index)', fontsize=16)
    
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
    
    ax.set_ylabel('Active TLE', fontsize=12)
    ax.set_title('TLE Timeline', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Semi-major axis
    ax = axes[1]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['a'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['a'], 'r--', label='TLE', linewidth=1.5, alpha=0.7)
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_ylabel('Semi-major axis (km)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Eccentricity
    ax = axes[2]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['e'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['e'], 'r--', label='TLE', linewidth=1.5, alpha=0.7)
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_ylabel('Eccentricity', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Inclination
    ax = axes[3]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['i'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['i'], 'r--', label='TLE', linewidth=1.5, alpha=0.7)
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_ylabel('Inclination (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # RAAN
    ax = axes[4]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['raan'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['raan'], 'r--', label='TLE', linewidth=1.5, alpha=0.7)
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_ylabel('RAAN (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Argument of Periapsis
    ax = axes[5]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['arg_pe'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['arg_pe'], 'r--', label='TLE', linewidth=1.5, alpha=0.7)
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_ylabel('Arg. of Periapsis (deg)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # True Anomaly
    ax = axes[6]
    ax.plot(horizons_oe_df['datetime'], horizons_oe_df['ta'], 'k-', label='Horizons', linewidth=1.5, alpha=0.7)
    ax.plot(tle_oe_df['datetime'], tle_oe_df['ta'], 'r--', label='TLE', linewidth=1.5, alpha=0.7)
    for t in transition_times:
        ax.axvline(x=t, color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_ylabel('True Anomaly (deg)', fontsize=11)
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Rotate x-axis labels
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    
    return fig

def main():
    # Define file paths
    base_path = Path(__file__).parent.parent.parent.parent / 'data'
    horizons_file = base_path / 'ephems' / 'large' / 'horizons_iss_highres_1m_20251001_utc.csv'
    tle_file      = base_path / 'ephems' / 'tle_history_iss.txt'
    
    print(f"Loading Horizons data from: {horizons_file}")
    
    # Load data
    horizons_df = load_horizons_data(horizons_file)
    
    print(f"Loaded {len(horizons_df)} data points")
    print(f"Time range: {horizons_df['datetime'].min()} to {horizons_df['datetime'].max()}")
    print(f"\nSample data:")
    print(horizons_df[['datetime', 'x_km', 'y_km', 'z_km', 'vx_km_s', 'vy_km_s', 'vz_km_s']].head())
    
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
    print(tle_df[['datetime', 'x_km', 'y_km', 'z_km', 'tle_index']].head())
    print(f"\nTLE index distribution:")
    print(tle_df['tle_index'].value_counts().sort_index())
    
    # Compute orbital elements
    print("\nComputing orbital elements...")
    oe_df = compute_orbital_elements(horizons_df)
    print(f"\nSample orbital elements:")
    print(oe_df[['datetime', 'a', 'e', 'i', 'raan', 'arg_pe', 'ta']].head())
    
    # Compute orbital elements for TLE data
    print("\nComputing orbital elements for TLE data...")
    tle_oe_df = compute_orbital_elements(tle_df)
    print(f"\nSample TLE orbital elements:")
    print(tle_oe_df[['datetime', 'a', 'e', 'i', 'raan', 'arg_pe', 'ta']].head())
    
    # Create plots
    fig1 = plot_horizons(horizons_df)
    fig2 = plot_orbital_elements(oe_df)
    fig3 = plot_horizons_vs_tle(horizons_df, tle_df, tle_epochs_df)
    fig4 = plot_horizons_vs_tle_with_index(horizons_df, tle_df, tle_epochs_df, tles)
    fig5 = plot_orbital_elements_tle_comparison(oe_df, tle_oe_df, tles)
    
    # Save plots
    output_file1 = Path(__file__).parent / 'iss_horizons_plot.png'
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"\nPosition/Velocity plot saved to: {output_file1}")
    
    output_file2 = Path(__file__).parent / 'iss_horizons_elements_plot.png'
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Orbital elements plot saved to: {output_file2}")
    
    output_file3 = Path(__file__).parent / 'iss_horizons_vs_tle_plot.png'
    fig3.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"Horizons vs TLE comparison plot saved to: {output_file3}")
    
    output_file4 = Path(__file__).parent / 'iss_horizons_vs_tle_with_index_plot.png'
    fig4.savefig(output_file4, dpi=150, bbox_inches='tight')
    print(f"Horizons vs TLE comparison with index plot saved to: {output_file4}")
    
    output_file5 = Path(__file__).parent / 'iss_orbital_elements_tle_comparison_plot.png'
    fig5.savefig(output_file5, dpi=150, bbox_inches='tight')
    print(f"Orbital elements TLE comparison plot saved to: {output_file5}")
    
    plt.show()

if __name__ == '__main__':
    main()
