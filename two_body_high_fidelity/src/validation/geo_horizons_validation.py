"""
Compare TLE propagation with HORIZONS ephemeris data
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from astropy.table import Table
from astropy.time import Time
from skyfield.api import EarthSatellite, load

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import TIMEVALUES
from model.dynamics import PHYSICALCONSTANTS
from model.coordinate_system_converter import CoordinateSystemConverter
from tle_propagator import propagate_tle


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_horizons_csv(filepath):
    """Load HORIZONS ephemeris from CSV file"""
    data = Table.read(filepath, format='csv')
    
    print("\nLoading HORIZONS data:")
    print(f"  Columns: {data.colnames}")
    print(f"  Target: {data['targetname'][0]}")  # Check what target this is
    
    # Extract times
    times = Time(data['datetime_jd'], format='jd')
    epoch = times[0]
    time_sec = (times - epoch).sec
    
    # Convert from AU and AU/day to m and m/s
    AU_TO_M = 149597870700.0
    DAY_TO_SEC = 86400.0
    
    x  = data['x'].data * AU_TO_M
    y  = data['y'].data * AU_TO_M
    z  = data['z'].data * AU_TO_M
    vx = data['vx'].data * AU_TO_M / DAY_TO_SEC
    vy = data['vy'].data * AU_TO_M / DAY_TO_SEC
    vz = data['vz'].data * AU_TO_M / DAY_TO_SEC
    
    state = np.vstack([x, y, z, vx, vy, vz])
    
    # Verify
    r = np.sqrt(x**2 + y**2 + z**2)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    
    print(f"  Number of points: {len(time_sec)}")
    print(f"  Time span: {time_sec[0]/TIMEVALUES.ONE_DAY:.2f} to {time_sec[-1]/TIMEVALUES.ONE_DAY:.2f} days")
    print(f"  Position magnitude: {np.mean(r)/1000:.2f} km (sample: {r[0]/1000:.2f} km)")
    print(f"  Velocity magnitude: {np.mean(v)/1000:.6f} km/s (sample: {v[0]/1000:.6f} km/s)")
    print(f"\n  ⚠️  WARNING: This appears to be {data['targetname'][0]}")
    print(f"      For GEO satellite: expect ~42,164 km position, ~3.075 km/s velocity")
    print(f"      For Mars: expect ~200-400 million km from Sun")
    
    return {
        'time': time_sec,
        'state': state,
        'epoch_jd': epoch.jd,
    }


def compare_tle_with_horizons(horizons_file, tle_file):
    """Compare TLE propagation with HORIZONS ephemeris"""
    
    print("="*80)
    print(" TLE vs HORIZONS COMPARISON")
    print("="*80)
    
    # Load HORIZONS data
    horizons_data = load_horizons_csv(horizons_file)
    
    # Load TLE
    print(f"\nLoading TLE from: {tle_file}")
    with open(tle_file, 'r') as f:
        lines = f.readlines()
    tle_line1 = lines[0].strip()
    tle_line2 = lines[1].strip()
    
    # DIAGNOSTIC: Check TLE epoch vs HORIZONS time range
    print("\n" + "="*80)
    print(" EPOCH ALIGNMENT CHECK")
    print("="*80)
    
    ts = load.timescale()
    sat = EarthSatellite(tle_line1, tle_line2, ts=ts)
    
    # Correct conversion: SGP4 epoch is already in JD (with a different base)
    # The model.jdsatepoch is days since 1949 Dec 31 00:00 UT
    # To convert to standard JD, add 2433281.5
    tle_epoch_jd = sat.model.jdsatepoch + 2433281.5
    
    # Alternative: use skyfield's time object directly
    tle_epoch_time = ts.tt_jd(sat.model.jdsatepoch + 2433281.5)
    # Even better: get it from the satellite's epoch attribute
    tle_epoch_jd_skyfield = sat.epoch.tt  # This is the actual JD in TT
    
    print(f"TLE Epoch (JD TT)   :    {tle_epoch_jd_skyfield:.6f}")
    print(f"HORIZONS Start (JD) :    {horizons_data['epoch_jd']:.6f}")
    print(f"HORIZONS End (JD)   :    {horizons_data['epoch_jd'] + horizons_data['time'][-1]/86400:.6f}")

    days_before_start = horizons_data['epoch_jd'] - tle_epoch_jd_skyfield
    print(f"\nTLE epoch is {days_before_start:.2f} days {'before' if days_before_start > 0 else 'after'} HORIZONS start")
    
    if abs(days_before_start) > 7:
        print(f"⚠️  WARNING: TLE epoch is {abs(days_before_start):.1f} days from comparison period!")
        print(f"   This will cause significant errors. TLEs degrade ~1km/day.")
        print(f"   Expected error: ~{abs(days_before_start):.0f} km just from age")
    
    print("="*80)
    
    # Propagate TLE
    print("\nPropagating TLE...")
    result_tle = propagate_tle(
        tle_line1=tle_line1,
        tle_line2=tle_line2,
        time_o=horizons_data['time'][0],
        time_f=horizons_data['time'][-1],
        num_points=len(horizons_data['time']),
        disable_drag=True,
        to_j2000=True,
    )
    
    if not result_tle['success']:
        print(f"✗ TLE propagation failed: {result_tle['message']}")
        return
    
    print("✓ TLE propagation successful")
    
    # DIAGNOSTIC: Compare first state vectors to identify frame issue
    print("\n" + "="*80)
    print(" STATE VECTOR COMPARISON (at t=0)")
    print("="*80)
    print(f"Position vectors (km):")
    print(f"  HORIZONS: [{horizons_data['state'][0,0]/1000:10.3f}, {horizons_data['state'][1,0]/1000:10.3f}, {horizons_data['state'][2,0]/1000:10.3f}]")
    print(f"  TLE:      [{result_tle['state'][0,0]/1000:10.3f}, {result_tle['state'][1,0]/1000:10.3f}, {result_tle['state'][2,0]/1000:10.3f}]")
    print(f"  Diff:     [{(result_tle['state'][0,0]-horizons_data['state'][0,0])/1000:10.3f}, "
          f"{(result_tle['state'][1,0]-horizons_data['state'][1,0])/1000:10.3f}, "
          f"{(result_tle['state'][2,0]-horizons_data['state'][2,0])/1000:10.3f}]")
    
    print(f"\nVelocity vectors (km/s):")
    print(f"  HORIZONS: [{horizons_data['state'][3,0]/1000:10.6f}, {horizons_data['state'][4,0]/1000:10.6f}, {horizons_data['state'][5,0]/1000:10.6f}]")
    print(f"  TLE:      [{result_tle['state'][3,0]/1000:10.6f}, {result_tle['state'][4,0]/1000:10.6f}, {result_tle['state'][5,0]/1000:10.6f}]")
    print(f"  Diff:     [{(result_tle['state'][3,0]-horizons_data['state'][3,0])/1000:10.6f}, "
          f"{(result_tle['state'][4,0]-horizons_data['state'][4,0])/1000:10.6f}, "
          f"{(result_tle['state'][5,0]-horizons_data['state'][5,0])/1000:10.6f}]")
    
    # Check if HORIZONS might be in wrong frame
    print(f"\n🔍 FRAME DIAGNOSTIC:")
    print(f"   Check your HORIZONS download settings:")
    print(f"   - Must use: Reference frame = 'ICRF/J2000.0'")
    print(f"   - NOT: 'Earth True Equator and Equinox of date'")
    print("="*80)
    
    # Calculate position/velocity errors
    pos_error = np.linalg.norm(
        result_tle['state'][0:3, :] - horizons_data['state'][0:3, :], 
        axis=0
    ) / 1000.0  # km
    
    vel_error = np.linalg.norm(
        result_tle['state'][3:6, :] - horizons_data['state'][3:6, :], 
        axis=0
    ) / 1000.0  # km/s
    
    print(f"\nPosition Error Statistics:")
    print(f"  Initial: {pos_error[0]:.6f} km")
    print(f"  Final:   {pos_error[-1]:.2f} km")
    print(f"  Mean:    {np.mean(pos_error):.2f} km")
    print(f"  RMS:     {np.sqrt(np.mean(pos_error**2)):.2f} km")
    print(f"  Max:     {np.max(pos_error):.2f} km")
    
    print(f"\nVelocity Error Statistics:")
    print(f"  Initial: {vel_error[0]:.9f} km/s")
    print(f"  Final:   {vel_error[-1]:.6f} km/s")
    print(f"  Mean:    {np.mean(vel_error):.6f} km/s")
    print(f"  RMS:     {np.sqrt(np.mean(vel_error**2)):.6f} km/s")
    
    # DIAGNOSTIC: Assessment
    print("\n" + "="*80)
    print(" ERROR ASSESSMENT")
    print("="*80)
    if pos_error[0] < 1.0 and vel_error[0] < 0.001:
        print("✓ EXCELLENT: Initial errors are very small")
        print("  This suggests good epoch alignment and frame conversion")
    elif pos_error[0] < 10.0 and vel_error[0] < 0.01:
        print("✓ GOOD: Initial errors are acceptable")
    else:
        print("✗ POOR: Large initial errors suggest:")
        print("  - Possible reference frame conversion issue")
        print("  - Possible epoch mismatch")
        print("  - Check if TLE and HORIZONS use same coordinate system")
    
    error_growth_rate = (pos_error[-1] - pos_error[0]) / (horizons_data['time'][-1] / 86400)
    print(f"\nPosition error growth rate: {error_growth_rate:.2f} km/day")
    if error_growth_rate > 10:
        print("✗ High growth rate suggests significant model differences")
    elif error_growth_rate > 1:
        print("⚠️  Moderate growth rate (expected for TLE vs high-fidelity)")
    else:
        print("✓ Low growth rate")
    print("="*80)
    
    # Convert to orbital elements
    print("\nConverting to orbital elements...")
    converter = CoordinateSystemConverter(PHYSICALCONSTANTS.EARTH.GP)
    
    time_days = horizons_data['time'] / TIMEVALUES.ONE_DAY
    n_points = len(time_days)
    
    # HORIZONS orbital elements
    horizons_coe = {key: np.zeros(n_points) for key in ['sma', 'ecc', 'inc', 'raan', 'argp', 'ta']}
    for i in range(n_points):
        coe = converter.rv2coe(horizons_data['state'][0:3, i], horizons_data['state'][3:6, i])
        for key in horizons_coe.keys():
            horizons_coe[key][i] = coe[key]
    
    # TLE orbital elements
    tle_coe = {key: np.zeros(n_points) for key in ['sma', 'ecc', 'inc', 'raan', 'argp', 'ta']}
    for i in range(n_points):
        coe = converter.rv2coe(result_tle['state'][0:3, i], result_tle['state'][3:6, i])
        for key in tle_coe.keys():
            tle_coe[key][i] = coe[key]
    
    # Unwrap angles
    for key in ['inc', 'raan', 'argp', 'ta']:
        horizons_coe[key] = np.unwrap(horizons_coe[key])
        tle_coe[key] = np.unwrap(tle_coe[key])
    
    # Plotting
    print("\nGenerating plots...")
    
    # Figure 1: Position and Velocity Errors
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('TLE vs HORIZONS: Position and Velocity Errors', fontsize=16, fontweight='bold')
    
    # Position error
    axes1[0, 0].plot(time_days, pos_error, 'b-', linewidth=2)
    axes1[0, 0].set_xlabel('Time (days)')
    axes1[0, 0].set_ylabel('Position Error (km)')
    axes1[0, 0].set_title('Position Error (Linear Scale)')
    axes1[0, 0].grid(True, alpha=0.3)
    
    axes1[0, 1].semilogy(time_days, pos_error, 'b-', linewidth=2)
    axes1[0, 1].set_xlabel('Time (days)')
    axes1[0, 1].set_ylabel('Position Error (km)')
    axes1[0, 1].set_title('Position Error (Log Scale)')
    axes1[0, 1].grid(True, alpha=0.3, which='both')
    
    # Velocity error
    axes1[1, 0].plot(time_days, vel_error, 'r-', linewidth=2)
    axes1[1, 0].set_xlabel('Time (days)')
    axes1[1, 0].set_ylabel('Velocity Error (km/s)')
    axes1[1, 0].set_title('Velocity Error (Linear Scale)')
    axes1[1, 0].grid(True, alpha=0.3)
    
    axes1[1, 1].semilogy(time_days, vel_error, 'r-', linewidth=2)
    axes1[1, 1].set_xlabel('Time (days)')
    axes1[1, 1].set_ylabel('Velocity Error (km/s)')
    axes1[1, 1].set_title('Velocity Error (Log Scale)')
    axes1[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Figure 2: Position and Velocity Components
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Position and Velocity Components', fontsize=16, fontweight='bold')
    
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes2[0, i].plot(time_days, horizons_data['state'][i, :]/1000.0, 'k-', label='HORIZONS', linewidth=2)
        axes2[0, i].plot(time_days, result_tle['state'][i, :]/1000.0, 'b--', label='TLE', linewidth=1.5, alpha=0.7)
        axes2[0, i].set_xlabel('Time (days)')
        axes2[0, i].set_ylabel(f'{label} Position (km)')
        axes2[0, i].set_title(f'{label} Position')
        axes2[0, i].legend()
        axes2[0, i].grid(True, alpha=0.3)
        
        axes2[1, i].plot(time_days, horizons_data['state'][3+i, :]/1000.0, 'k-', label='HORIZONS', linewidth=2)
        axes2[1, i].plot(time_days, result_tle['state'][3+i, :]/1000.0, 'r--', label='TLE', linewidth=1.5, alpha=0.7)
        axes2[1, i].set_xlabel('Time (days)')
        axes2[1, i].set_ylabel(f'V{label} (km/s)')
        axes2[1, i].set_title(f'{label} Velocity')
        axes2[1, i].legend()
        axes2[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Figure 3: Orbital Elements
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 12))
    fig3.suptitle('Orbital Elements Comparison', fontsize=16, fontweight='bold')
    
    elements = [
        ('sma', 'Semi-major Axis (km)', 1.0/1000.0),
        ('ecc', 'Eccentricity', 1.0),
        ('inc', 'Inclination (deg)', np.rad2deg(1.0)),
        ('raan', 'RAAN (deg)', np.rad2deg(1.0)),
        ('argp', 'Arg of Perigee (deg)', np.rad2deg(1.0)),
        ('ta', 'True Anomaly (deg)', np.rad2deg(1.0)),
    ]
    
    for idx, (key, label, scale) in enumerate(elements):
        ax = axes3[idx // 2, idx % 2]
        ax.plot(time_days, horizons_coe[key] * scale, 'k-', label='HORIZONS', linewidth=2)
        ax.plot(time_days, tle_coe[key] * scale, 'b--', label='TLE', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Figure 4: Orbital Element Errors
    fig4, axes4 = plt.subplots(3, 2, figsize=(14, 12))
    fig4.suptitle('Orbital Element Errors (TLE - HORIZONS)', fontsize=16, fontweight='bold')
    
    for idx, (key, label, scale) in enumerate(elements):
        ax = axes4[idx // 2, idx % 2]
        error = (tle_coe[key] - horizons_coe[key]) * scale
        ax.plot(time_days, error, 'r-', linewidth=2)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(f'Error in {label}')
        ax.set_title(f'{label} Error')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    plt.show()
    
    return {
        'horizons': horizons_data,
        'tle': result_tle,
        'pos_error': pos_error,
        'vel_error': vel_error,
        'horizons_coe': horizons_coe,
        'tle_coe': tle_coe,
    }


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / 'data' / 'ephems'
    horizons_file = data_dir / 'horizons_goes16.csv'
    tle_file = data_dir / 'tle_goes16.txt'
    
    if not horizons_file.exists() or not tle_file.exists():
        print(f"Data files not found!")
        print(f"HORIZONS: {horizons_file}")
        print(f"TLE:      {tle_file}")
    else:
        compare_tle_with_horizons(horizons_file, tle_file)
