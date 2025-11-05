#!/usr/bin/env python3
"""
Diagnostic script to understand HORIZONS vs TLE coordinate frame differences.
"""

from astroquery.jplhorizons import Horizons
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
import numpy as np

# GOES-16 identifier
norad_id = 41866
sat_id = -(100000 + norad_id)

# Query for October 1-8, 2025 time range
start_time = '2025-10-01 00:00:00'
stop_time = '2025-10-01 01:00:00'   # Just for initial check

epochs = {'start': start_time,
          'stop': stop_time,
          'step': '1h'}

# Query Horizons
obj = Horizons(id=sat_id, 
               location='500@399',  # Earth center
               epochs=epochs)

vectors = obj.vectors(refplane='earth', delta_T=True)

# Convert from AU to km
AU_TO_KM = 149597870.7
SECONDS_PER_DAY = 86400.0

pos_horizons = np.array([
    vectors['x'][0] * AU_TO_KM,
    vectors['y'][0] * AU_TO_KM,
    vectors['z'][0] * AU_TO_KM
])

vel_horizons = np.array([
    vectors['vx'][0] * AU_TO_KM / SECONDS_PER_DAY,
    vectors['vy'][0] * AU_TO_KM / SECONDS_PER_DAY,
    vectors['vz'][0] * AU_TO_KM / SECONDS_PER_DAY
])

print("HORIZONS Data (ICRS/J2000):")
print(f"  JD: {vectors['datetime_jd'][0]}")
print(f"  Position: {pos_horizons} km")
print(f"  Velocity: {vel_horizons} km/s")
print(f"  Delta-T: {vectors['delta_T'][0]:.3f} s")

# Now try converting HORIZONS GCRS to TEME
t = Time(vectors['datetime_jd'][0], format='jd', scale='utc')

cart_rep = CartesianRepresentation(
    x=pos_horizons[0] * u.km,
    y=pos_horizons[1] * u.km,
    z=pos_horizons[2] * u.km,
    differentials=CartesianDifferential(
        d_x=vel_horizons[0] * u.km / u.s,
        d_y=vel_horizons[1] * u.km / u.s,
        d_z=vel_horizons[2] * u.km / u.s
    )
)

gcrs_coord = GCRS(cart_rep, obstime=t)
teme_coord = gcrs_coord.transform_to(TEME(obstime=t))

pos_teme = np.array([
    teme_coord.cartesian.x.to(u.km).value,
    teme_coord.cartesian.y.to(u.km).value,
    teme_coord.cartesian.z.to(u.km).value
])

vel_teme = np.array([
    teme_coord.velocity.d_x.to(u.km / u.s).value,
    teme_coord.velocity.d_y.to(u.km / u.s).value,
    teme_coord.velocity.d_z.to(u.km / u.s).value
])

print("\nConverted to TEME:")
print(f"  Position: {pos_teme} km")
print(f"  Velocity: {vel_teme} km/s")
print(f"  Position change: {np.linalg.norm(pos_teme - pos_horizons):.3f} km")

# Load and compare with TLE
print("\n" + "="*60)
print("Now checking TLE...")

from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

# Read TLE
tle_file = '/Users/davidottesen/github/spacecraft-trajectory-problems/data/ephems/tle_goes16.txt'
with open(tle_file, 'r') as f:
    lines = f.readlines()

# Handle TLE format - could be 2 lines or 3 lines (with title)
lines = [line.strip() for line in lines if line.strip()]
if len(lines) == 2:
    tle_line1, tle_line2 = lines[0], lines[1]
elif len(lines) >= 3:
    tle_line1, tle_line2 = lines[1], lines[2]
else:
    raise ValueError(f"Invalid TLE format: expected 2 or 3 lines, got {len(lines)}")

print(f"TLE Line 1: {tle_line1}")
print(f"TLE Line 2: {tle_line2}")

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

print("\n" + "="*60)
print("EPOCH COMPARISON")
print("="*60)
print(f"TLE Epoch:")
print(f"  Datetime: {epoch_datetime}")
print(f"  JD (UTC): {jd + fr:.6f}")
print(f"\nHORIZONS Query Time:")
print(f"  Datetime: {t.datetime}")
print(f"  JD (UTC): {vectors['datetime_jd'][0]:.6f}")

time_diff_days = (jd + fr) - vectors['datetime_jd'][0]
print(f"\nTime Difference: {time_diff_days:.1f} days ({time_diff_days*24:.1f} hours)")

if abs(time_diff_days) > 1:
    print(f"\n❌ ERROR: TLE and HORIZONS times don't match!")
    print(f"   To fix this, either:")
    print(f"   1. Get a TLE from around {t.datetime.strftime('%Y-%m-%d')}")
    print(f"   2. Query HORIZONS for {epoch_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNow querying HORIZONS at TLE epoch for comparison...")
    
    # Query HORIZONS at TLE epoch
    tle_epoch_str = epoch_datetime.strftime('%Y-%m-%d %H:%M:%S')
    tle_epoch_stop = (epoch_datetime + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
    
    obj2 = Horizons(id=sat_id, 
                   location='500@399',
                   epochs={'start': tle_epoch_str,
                          'stop': tle_epoch_stop,
                          'step': '1h'})
    
    vectors2 = obj2.vectors(refplane='earth', delta_T=True)
    
    pos_horizons2 = np.array([
        vectors2['x'][0] * AU_TO_KM,
        vectors2['y'][0] * AU_TO_KM,
        vectors2['z'][0] * AU_TO_KM
    ])
    
    vel_horizons2 = np.array([
        vectors2['vx'][0] * AU_TO_KM / SECONDS_PER_DAY,
        vectors2['vy'][0] * AU_TO_KM / SECONDS_PER_DAY,
        vectors2['vz'][0] * AU_TO_KM / SECONDS_PER_DAY
    ])
    
    # Convert to TEME
    t2 = Time(vectors2['datetime_jd'][0], format='jd', scale='utc')
    
    print(f"\nHORIZONS at TLE epoch (ICRS/J2000):")
    print(f"  Time: {t2.datetime} (JD: {vectors2['datetime_jd'][0]:.6f})")
    print(f"  Position: {pos_horizons2} km")
    print(f"  Velocity: {vel_horizons2} km/s")
    
    cart_rep2 = CartesianRepresentation(
        x=pos_horizons2[0] * u.km,
        y=pos_horizons2[1] * u.km,
        z=pos_horizons2[2] * u.km,
        differentials=CartesianDifferential(
            d_x=vel_horizons2[0] * u.km / u.s,
            d_y=vel_horizons2[1] * u.km / u.s,
            d_z=vel_horizons2[2] * u.km / u.s
        )
    )
    gcrs_coord2 = GCRS(cart_rep2, obstime=t2)
    teme_coord2 = gcrs_coord2.transform_to(TEME(obstime=t2))
    
    pos_teme2 = np.array([
        teme_coord2.cartesian.x.to(u.km).value,
        teme_coord2.cartesian.y.to(u.km).value,
        teme_coord2.cartesian.z.to(u.km).value
    ])
    
    vel_teme2 = np.array([
        teme_coord2.velocity.d_x.to(u.km / u.s).value,
        teme_coord2.velocity.d_y.to(u.km / u.s).value,
        teme_coord2.velocity.d_z.to(u.km / u.s).value
    ])
    
    print(f"\nHORIZONS at TLE epoch (converted to TEME):")
    print(f"  Time: {t2.datetime} (JD: {vectors2['datetime_jd'][0]:.6f})")
    print(f"  Position: {pos_teme2} km")
    print(f"  Velocity: {vel_teme2} km/s")
    
    # Get TLE state at epoch
    error_code, teme_pos_tle, teme_vel_tle = satellite.sgp4(jd, fr)
    
    print(f"\nTLE at epoch (TEME):")
    print(f"  Time: {epoch_datetime} (JD: {jd + fr:.6f})")
    print(f"  Position: {np.array(teme_pos_tle)} km")
    print(f"  Velocity: {np.array(teme_vel_tle)} km/s")
    
    print(f"\nDifference (HORIZONS TEME - TLE TEME) at matching epoch:")
    print(f"  Position: {pos_teme2 - np.array(teme_pos_tle)} km")
    print(f"  Position mag: {np.linalg.norm(pos_teme2 - np.array(teme_pos_tle)):.3f} km")
    
    vel_diff = np.linalg.norm(vel_teme2 - np.array(teme_vel_tle))
    print(f"  Velocity: {vel_teme2 - np.array(teme_vel_tle)} km/s")
    print(f"  Velocity mag: {vel_diff:.6f} km/s")
    
    if np.linalg.norm(pos_teme2 - np.array(teme_pos_tle)) < 500:
        print(f"\n✓ GOOD: Error is reasonable when epochs match!")
    else:
        print(f"\n⚠️  Still large error even with matching epochs")
    
    # Now propagate TLE over the same timespan as HORIZONS data and compare
    print("\n" + "="*60)
    print("COMPARING HORIZONS WITH TLE HISTORY")
    print("="*60)
    
    # Load TLE history
    tle_history_file = '/Users/davidottesen/github/spacecraft-trajectory-problems/data/ephems/tle_history_goes16.txt'
    
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
        print("Using single TLE propagation instead...")
        tle_list = [{
            'line1': tle_line1,
            'line2': tle_line2,
            'epoch_jd': jd + fr,
            'epoch_time': Time(jd + fr, format='jd'),
            'satrec': satellite
        }]
    
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
        
        # Time relative to October 1, 2025 00:00
        oct1_2025_jd = Time('2025-10-01 00:00:00', format='iso', scale='utc').jd
        horizons_times[i] = (vectors_full['datetime_jd'][i] - oct1_2025_jd) * 24  # hours from Oct 1
    
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
            tle_times_list.append((tle['epoch_jd'] - oct1_2025_jd) * 24)  # hours from Oct 1
    
    tle_pos = np.array(tle_pos_list)
    tle_vel = np.array(tle_vel_list)
    tle_times = np.array(tle_times_list)
    
    print(f"Processed {len(tle_pos)} TLE points")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Position components
        ax = axes[0, 0]
        ax.plot(horizons_times, horizons_pos[:, 0], 'b-', label='HORIZONS', linewidth=2, alpha=0.7)
        ax.scatter(tle_times, tle_pos[:, 0], c='red', s=50, marker='o', label='TLE Epochs', zorder=5)
        ax.set_xlabel('Hours from Oct 1, 2025 00:00 UTC')
        ax.set_ylabel('X Position (km)')
        ax.set_title('X Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(horizons_times, horizons_pos[:, 1], 'b-', label='HORIZONS', linewidth=2, alpha=0.7)
        ax.scatter(tle_times, tle_pos[:, 1], c='red', s=50, marker='o', label='TLE Epochs', zorder=5)
        ax.set_xlabel('Hours from Oct 1, 2025 00:00 UTC')
        ax.set_ylabel('Y Position (km)')
        ax.set_title('Y Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity components
        ax = axes[1, 0]
        ax.plot(horizons_times, horizons_vel[:, 0], 'b-', label='HORIZONS', linewidth=2, alpha=0.7)
        ax.scatter(tle_times, tle_vel[:, 0], c='red', s=50, marker='o', label='TLE Epochs', zorder=5)
        ax.set_xlabel('Hours from Oct 1, 2025 00:00 UTC')
        ax.set_ylabel('VX (km/s)')
        ax.set_title('VX Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compute errors at TLE epochs by interpolating HORIZONS
        from scipy.interpolate import interp1d
        
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
        
        ax = axes[1, 1]
        ax.scatter(tle_times, pos_errors, c='green', s=50, marker='o')
        ax.set_xlabel('Hours from Oct 1, 2025 00:00 UTC')
        ax.set_ylabel('Position Error (km)')
        ax.set_title('Position Error at TLE Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tle_horizons_comparison.png', dpi=150)
        print(f"\nPlot saved to: tle_horizons_comparison.png")
        
        print(f"\nError statistics:")
        print(f"  Mean error: {np.mean(pos_errors):.3f} km")
        print(f"  Min error:  {np.min(pos_errors):.3f} km")
        print(f"  Max error:  {np.max(pos_errors):.3f} km")
        print(f"  Std error:  {np.std(pos_errors):.3f} km")
        
        # Additional 3D trajectory plot
        fig2 = plt.figure(figsize=(10, 10))
        ax3d = fig2.add_subplot(111, projection='3d')
        
        ax3d.plot(horizons_pos[:, 0], horizons_pos[:, 1], horizons_pos[:, 2],
                 'b-', label='HORIZONS', linewidth=2, alpha=0.7)
        ax3d.scatter(tle_pos[:, 0], tle_pos[:, 1], tle_pos[:, 2],
                    c='red', s=50, marker='o', label='TLE Epochs')
        
        ax3d.set_xlabel('X (km)')
        ax3d.set_ylabel('Y (km)')
        ax3d.set_zlabel('Z (km)')
        ax3d.set_title('3D Trajectory Comparison (TEME Frame)')
        ax3d.legend()
        ax3d.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tle_horizons_3d_trajectory.png', dpi=150)
        print(f"3D plot saved to: tle_horizons_3d_trajectory.png")
        
    except ImportError as e:
        print(f"\nMatplotlib or scipy not available for plotting: {e}")

else:
    error_code, teme_pos_tle, teme_vel_tle = satellite.sgp4(jd, fr)
    
    print(f"\nTLE at epoch (TEME):")
    print(f"  JD: {jd + fr}")
    print(f"  Position: {np.array(teme_pos_tle)} km")
    print(f"  Velocity: {np.array(teme_vel_tle)} km/s")
    
    print(f"\nDifference (HORIZONS TEME - TLE TEME):")
    print(f"  Position: {pos_teme - np.array(teme_pos_tle)} km")
    print(f"  Position mag: {np.linalg.norm(pos_teme - np.array(teme_pos_tle)):.3f} km")
