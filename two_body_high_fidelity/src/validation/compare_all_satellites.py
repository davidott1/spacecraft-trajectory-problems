#!/usr/bin/env python3
"""
Compare TLE vs HORIZONS ephemeris for multiple satellites across LEO, MEO, and GEO
"""

from pathlib import Path
import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astroquery.jplhorizons import Horizons
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Extended satellite catalog
ALL_SATELLITES = {
    # LEO satellites (Low Earth Orbit, ~400-700 km altitude)
    'ISS': 25544,         # International Space Station
    'TERRA': 25994,       # Earth observation satellite
    'AQUA': 27424,        # Earth observation satellite
    
    # MEO satellites (Medium Earth Orbit, ~20,200 km altitude)
    'GPS-IIR-5': 26407,   # NAVSTAR-48
    'GPS-IIF-2': 38833,   # NAVSTAR-67
    'GPS-IIF-3': 39166,   # NAVSTAR-68
    
    # GEO satellites (Geostationary Orbit, ~35,786 km altitude)
    'GOES-16': 41866,     # GOES-R (weather)
    'GOES-17': 43226,     # GOES-S (weather)
    'GOES-18': 51850,     # GOES-T (weather)
}

# Constants
AU_TO_KM = 149597870.7
SECONDS_PER_DAY = 86400.0

DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'ephems'


def compare_satellite(norad_id, sat_name):
    """
    Compare TLE history with HORIZONS ephemeris for a single satellite
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog number
    sat_name : str
        Satellite name (e.g., 'goes16', 'iss', 'gpsbiia27')
    
    Returns:
    --------
    dict : Comparison results including error statistics
    """
    print(f"\n{'='*80}")
    print(f"Processing {sat_name.upper()}")
    print(f"{'='*80}")
    
    sat_id = -(100000 + norad_id)
    
    # Load TLE history
    tle_history_file = DATA_DIR / f'tle_history_{sat_name}.txt'
    
    try:
        with open(tle_history_file, 'r') as f:
            lines = f.readlines()
        
        # Parse TLEs
        tle_list = []
        for i in range(0, len(lines)-1, 2):
            if lines[i].strip().startswith('1 ') and lines[i+1].strip().startswith('2 '):
                line1 = lines[i].strip()
                line2 = lines[i+1].strip()
                
                # Extract epoch
                tle_epoch_str = line1[18:32]
                tle_year = int(tle_epoch_str[0:2])
                tle_year = 2000 + tle_year if tle_year < 57 else 1900 + tle_year
                tle_day_of_year = float(tle_epoch_str[2:])
                
                tle_epoch_time = Time(f"{tle_year}-01-01", format='iso') + (tle_day_of_year - 1)
                
                sat_obj = Satrec.twoline2rv(line1, line2)
                
                tle_list.append({
                    'line1': line1,
                    'line2': line2,
                    'epoch_jd': tle_epoch_time.jd,
                    'epoch_time': tle_epoch_time,
                    'satrec': sat_obj
                })
        
        print(f"Loaded {len(tle_list)} TLEs from history file")
        
    except FileNotFoundError:
        print(f"TLE history file not found: {tle_history_file}")
        return None
    
    # Query HORIZONS for October 1-8, 2025
    horizons_start = '2025-10-01 00:00:00'
    horizons_stop = '2025-10-08 00:00:00'
    
    print(f"Querying HORIZONS from {horizons_start} to {horizons_stop}...")
    
    obj_full = Horizons(id=sat_id, 
                       location='500@399',
                       epochs={'start': horizons_start,
                              'stop': horizons_stop,
                              'step': '1h'})
    
    vectors_full = obj_full.vectors(refplane='earth', delta_T=True)
    
    n_points = len(vectors_full)
    print(f"Retrieved {n_points} HORIZONS data points")
    
    # Arrays to store results
    horizons_pos = np.zeros((n_points, 3))
    horizons_vel = np.zeros((n_points, 3))
    horizons_times = np.zeros(n_points)
    
    # For each TLE, evaluate at its epoch
    tle_pos_list = []
    tle_vel_list = []
    tle_times_list = []
    
    oct1_2025_jd = Time('2025-10-01 00:00:00', format='iso', scale='utc').jd
    
    print("Processing HORIZONS data...")
    
    for i in range(n_points):
        # Get HORIZONS data
        pos_h_j2000 = np.array([
            vectors_full['x'][i] * AU_TO_KM,
            vectors_full['y'][i] * AU_TO_KM,
            vectors_full['z'][i] * AU_TO_KM
        ])
        
        vel_h_j2000 = np.array([
            vectors_full['vx'][i] * AU_TO_KM / SECONDS_PER_DAY,
            vectors_full['vy'][i] * AU_TO_KM / SECONDS_PER_DAY,
            vectors_full['vz'][i] * AU_TO_KM / SECONDS_PER_DAY
        ])
        
        # Convert HORIZONS to TEME
        t_i = Time(vectors_full['datetime_jd'][i], format='jd', scale='utc')
        cart_i = CartesianRepresentation(
            x=pos_h_j2000[0] * u.km,
            y=pos_h_j2000[1] * u.km,
            z=pos_h_j2000[2] * u.km,
            differentials=CartesianDifferential(
                d_x=vel_h_j2000[0] * u.km / u.s,
                d_y=vel_h_j2000[1] * u.km / u.s,
                d_z=vel_h_j2000[2] * u.km / u.s
            )
        )
        gcrs_i = GCRS(cart_i, obstime=t_i)
        teme_i = gcrs_i.transform_to(TEME(obstime=t_i))
        
        horizons_pos[i] = np.array([
            teme_i.cartesian.x.to(u.km).value,
            teme_i.cartesian.y.to(u.km).value,
            teme_i.cartesian.z.to(u.km).value
        ])
        
        horizons_vel[i] = np.array([
            teme_i.velocity.d_x.to(u.km / u.s).value,
            teme_i.velocity.d_y.to(u.km / u.s).value,
            teme_i.velocity.d_z.to(u.km / u.s).value
        ])
        
        horizons_times[i] = (vectors_full['datetime_jd'][i] - oct1_2025_jd) * 24
    
    print("Processing TLE data at epochs...")
    
    for tle in tle_list:
        # Evaluate TLE at its epoch
        dt_tle = tle['epoch_time'].datetime
        jd_tle, fr_tle = jday(dt_tle.year, dt_tle.month, dt_tle.day,
                              dt_tle.hour, dt_tle.minute,
                              dt_tle.second + dt_tle.microsecond/1e6)
        
        err_code, pos_tle, vel_tle = tle['satrec'].sgp4(jd_tle, fr_tle)
        
        if err_code == 0:
            tle_pos_list.append(np.array(pos_tle))
            tle_vel_list.append(np.array(vel_tle))
            tle_times_list.append((tle['epoch_jd'] - oct1_2025_jd) * 24)
    
    tle_pos = np.array(tle_pos_list)
    tle_vel = np.array(tle_vel_list)
    tle_times = np.array(tle_times_list)
    
    print(f"Processed {len(tle_pos)} TLE points")
    
    if len(tle_pos) == 0:
        print("No valid TLE data points")
        return None
    
    # Interpolate HORIZONS to TLE epochs
    horizons_interp_x = interp1d(horizons_times, horizons_pos[:, 0], kind='cubic', fill_value='extrapolate')
    horizons_interp_y = interp1d(horizons_times, horizons_pos[:, 1], kind='cubic', fill_value='extrapolate')
    horizons_interp_z = interp1d(horizons_times, horizons_pos[:, 2], kind='cubic', fill_value='extrapolate')
    
    horizons_at_tle = np.column_stack([
        horizons_interp_x(tle_times),
        horizons_interp_y(tle_times),
        horizons_interp_z(tle_times)
    ])
    
    pos_errors = np.linalg.norm(horizons_at_tle - tle_pos, axis=1)
    
    # Get orbital regime
    mean_altitude = np.mean(np.linalg.norm(tle_pos, axis=1)) - 6371  # Earth radius ~6371 km
    if mean_altitude < 2000:
        regime = "LEO"
    elif mean_altitude < 35000:
        regime = "MEO"
    else:
        regime = "GEO"
    
    print(f"\nError statistics for {sat_name.upper()} ({regime}):")
    print(f"  Mean altitude: {mean_altitude:.1f} km")
    print(f"  Mean error: {np.mean(pos_errors):.3f} km")
    print(f"  Min error:  {np.min(pos_errors):.3f} km")
    print(f"  Max error:  {np.max(pos_errors):.3f} km")
    print(f"  Std error:  {np.std(pos_errors):.3f} km")
    
    return {
        'sat_name': sat_name,
        'norad_id': norad_id,
        'regime': regime,
        'mean_altitude': mean_altitude,
        'horizons_pos': horizons_pos,
        'horizons_vel': horizons_vel,
        'horizons_times': horizons_times,
        'tle_pos': tle_pos,
        'tle_vel': tle_vel,
        'tle_times': tle_times,
        'pos_errors': pos_errors,
        'error_mean': np.mean(pos_errors),
        'error_min': np.min(pos_errors),
        'error_max': np.max(pos_errors),
        'error_std': np.std(pos_errors),
    }


def plot_comparison(results):
    """
    Create comparison plots for all satellites grouped by orbital regime
    
    Parameters:
    -----------
    results : list of dict
        Results from compare_satellite for each satellite
    """
    # Group by regime
    regimes = {'LEO': [], 'MEO': [], 'GEO': []}
    for res in results:
        regimes[res['regime']].append(res)
    
    # Create regime comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    regime_names = ['LEO', 'MEO', 'GEO']
    for idx, regime in enumerate(regime_names):
        ax = axes[idx]
        
        if not regimes[regime]:
            ax.text(0.5, 0.5, f'No {regime} satellites', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{regime} Satellites')
            continue
        
        for res in regimes[regime]:
            ax.scatter(res['tle_times'], res['pos_errors'], 
                      s=50, marker='o', label=res['sat_name'].upper(), alpha=0.7)
        
        ax.set_xlabel('Hours from Oct 1, 2025 00:00 UTC')
        ax.set_ylabel('Position Error (km)')
        ax.set_title(f'{regime} Satellites - TLE vs HORIZONS Position Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_satellites_by_regime.png', dpi=150)
    print(f"\nRegime comparison plot saved to: all_satellites_by_regime.png")
    
    # Create individual satellite plots
    n_sats = len(results)
    fig, axes = plt.subplots(n_sats, 2, figsize=(14, 5*n_sats))
    if n_sats == 1:
        axes = axes.reshape(1, -1)
    
    for idx, res in enumerate(results):
        sat_name = res['sat_name']
        
        # Position X component
        ax = axes[idx, 0]
        ax.plot(res['horizons_times'], res['horizons_pos'][:, 0], 
                'b-', label='HORIZONS', linewidth=2, alpha=0.7)
        ax.scatter(res['tle_times'], res['tle_pos'][:, 0], 
                  c='red', s=50, marker='o', label='TLE Epochs', zorder=5)
        ax.set_xlabel('Hours from Oct 1, 2025 00:00:00 UTC')
        ax.set_ylabel('X Position (km)')
        ax.set_title(f'{sat_name.upper()} ({res["regime"]}) - X Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Position errors
        ax = axes[idx, 1]
        ax.scatter(res['tle_times'], res['pos_errors'], c='green', s=50, marker='o')
        ax.set_xlabel('Hours from Oct 1, 2025 00:00:00 UTC')
        ax.set_ylabel('Position Error (km)')
        ax.set_title(f'{sat_name.upper()} ({res["regime"]}) - Error (mean={res["error_mean"]:.1f} km)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_satellites_detailed.png', dpi=150)
    print(f"Detailed comparison plot saved to: all_satellites_detailed.png")


def main():
    """Main function to compare all satellites"""
    results = []
    
    for sat_name, norad_id in ALL_SATELLITES.items():
        sat_name_lower = sat_name.lower().replace('-', '')
        result = compare_satellite(norad_id, sat_name_lower)
        if result is not None:
            results.append(result)
    
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Satellite':15s} {'Regime':6s} {'Alt (km)':>10s} {'Mean Err':>10s} {'Min':>10s} {'Max':>10s} {'Std':>10s}")
        print("-"*80)
        for res in results:
            print(f"{res['sat_name'].upper():15s} {res['regime']:6s} "
                  f"{res['mean_altitude']:10.1f} "
                  f"{res['error_mean']:10.1f} "
                  f"{res['error_min']:10.1f} "
                  f"{res['error_max']:10.1f} "
                  f"{res['error_std']:10.1f}")
        
        print("\nGenerating plots...")
        plot_comparison(results)
    else:
        print("\n✗ No successful comparisons")


if __name__ == "__main__":
    main()
