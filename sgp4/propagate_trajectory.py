from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dynamics import Dynamics
from sgp4_helper import SGP4Helper

# Example TLE data for ISS (International Space Station)
# You can get updated TLEs from https://celestrak.org/
line1 = '1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005'
line2 = '2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.54225995 12345'

# Initialize the satellite object
satellite = Satrec.twoline2rv(line1, line2)

# Define the epoch time
year   = 2024
month  = 1
day    = 1
hour   = 12
minute = 0
second = 0.0

# Convert to Julian date
jd, fr = jday(year, month, day, hour, minute, second)

print(f"Propagating spacecraft trajectory using SGP4")

# Convert TLE epoch to datetime
epoch_datetime = SGP4Helper.get_epoch_as_datetime(satellite)
print(f"TLE Epoch: {epoch_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Propagate for one week in 1-hour increments
print("Propagating for one week in 1-hour increments...\n")

start_time      = datetime(2024, 1, 1, 12, 0, 0)
minutes_to_propagate = 2 * 60 
time_hours      = []
semi_major_axis = []
eccentricity    = []
inclination     = []
raan            = []
arg_perigee     = []
mean_anomaly    = []

# Get initial orbital elements from TLE
tle_coe = SGP4Helper.get_orbital_elements(satellite)
print(f"Initial TLE orbital elements:")
print(f"  SMA  : {tle_coe['sma']/1000.0:>13.6e} m")
print(f"  ECC  : {tle_coe['ecc']:>13.6e}")
print(f"  INC  : {np.rad2deg(tle_coe['inc']):>13.6e} deg")
print(f"  RAAN : {np.rad2deg(tle_coe['raan']):>13.6e} deg")
print(f"  ARGP : {np.rad2deg(tle_coe['argp']):>13.6e} deg")
print(f"  MA   : {np.rad2deg(tle_coe['ma']):>13.6e} deg")

# Initialize dynamics calculator (for comparison if needed)
dynamics = Dynamics()

# Store position data for plotting
pos_x = []
pos_y = []
pos_z = []

# for hour in range(0, hours_in_week + 1):
for min in range(0, minutes_to_propagate + 1):
    current_time = start_time + timedelta(minutes=min)
    jd, fr = \
        jday(
            current_time.year, current_time.month, current_time.day,
            current_time.hour, current_time.minute, current_time.second,
        )
    
    error_code, position, velocity = satellite.sgp4(jd, fr)
    
    if error_code == 0:
        time_hours.append(min / 60.0)  # Convert minutes to hours
        
        # Store position components (in km)
        pos_x.append(position[0])
        pos_y.append(position[1])
        pos_z.append(position[2])
        
        # Convert position and velocity to orbital elements
        pos_m = [p * 1000.0 for p in position]
        vel_m = [v * 1000.0 for v in velocity]
        coe = dynamics.rv2coe(pos_m, vel_m)
        
        semi_major_axis.append(coe['sma'] / 1000.0)  # Convert back to km for plotting
        eccentricity.append(coe['ecc'])
        inclination.append(np.rad2deg(coe['inc']))
        raan.append(np.rad2deg(coe['raan']))
        arg_perigee.append(np.rad2deg(coe['argp']))
        mean_anomaly.append(np.rad2deg(coe['ma']))

# Convert to numpy arrays
time_hours = np.array(time_hours)
time_days = time_hours / 24.0

# Create plots with 3D trajectory
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Trajectory and Orbital Elements Evolution', fontsize=16)

# Create 3x3 grid
gs = fig.add_gridspec(3, 3)

# Left column: 3D trajectory (spans all 3 rows)
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_3d.plot(pos_x, pos_y, pos_z, 'k-', linewidth=1.5)
ax_3d.scatter(pos_x[0], pos_y[0], pos_z[0], edgecolors='k', facecolors='w', marker='^', s=100, linewidths=2, label='Start')
ax_3d.scatter(pos_x[-1], pos_y[-1], pos_z[-1], edgecolors='k', facecolors='w', marker='s', s=100, linewidths=2, label='Stop')
ax_3d.set_xlabel('X (km)')
ax_3d.set_ylabel('Y (km)')
ax_3d.set_zlabel('Z (km)')
ax_3d.set_title('3D Trajectory')
ax_3d.legend(loc='best')
ax_3d.grid(True, alpha=0.3)

# Middle column: First 3 orbital elements
ax_sma = fig.add_subplot(gs[0, 1])
ax_sma.plot(time_days, semi_major_axis, 'k-', linewidth=1.5)
ax_sma.set_ylabel('Semi-major Axis (km)')
ax_sma.grid(True, alpha=0.3)

ax_ecc = fig.add_subplot(gs[1, 1], sharex=ax_sma)
ax_ecc.plot(time_days, eccentricity, 'k-', linewidth=1.5)
ax_ecc.set_ylabel('Eccentricity')
ax_ecc.grid(True, alpha=0.3)

ax_inc = fig.add_subplot(gs[2, 1], sharex=ax_sma)
ax_inc.plot(time_days, inclination, 'k-', linewidth=1.5)
ax_inc.set_ylabel('Inclination (deg)')
ax_inc.set_xlabel('Time (days)')
ax_inc.grid(True, alpha=0.3)

# Right column: Last 3 orbital elements
ax_raan = fig.add_subplot(gs[0, 2], sharex=ax_sma)
ax_raan.plot(time_days, raan, 'k-', linewidth=1.5)
ax_raan.set_ylabel('RAAN (deg)')
ax_raan.grid(True, alpha=0.3)

ax_argp = fig.add_subplot(gs[1, 2], sharex=ax_sma)
ax_argp.plot(time_days, arg_perigee, 'k-', linewidth=1.5)
ax_argp.set_ylabel('Arg. of Perigee (deg)')
ax_argp.grid(True, alpha=0.3)

ax_ma = fig.add_subplot(gs[2, 2], sharex=ax_sma)
ax_ma.plot(time_days, mean_anomaly, 'k-', linewidth=1.5)
ax_ma.set_ylabel('Mean Anomaly (deg)')
ax_ma.set_xlabel('Time (days)')
ax_ma.grid(True, alpha=0.3)

plt.tight_layout()
output_file = Path.cwd() / 'orbital_elements.png'
plt.savefig(output_file, dpi=150)
plt.show()

print(f"Propagation complete. Plotted {len(time_hours)} data points.")
print(f"Plot saved to {output_file}")