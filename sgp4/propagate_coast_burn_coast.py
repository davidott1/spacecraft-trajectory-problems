from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dynamics import Dynamics
from sgp4_helper import SGP4Helper
from constants import Constants
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Propagate coast-burn-coast trajectory using SGP4')
parser.add_argument('--show-burns', action='store_true', 
                    help='Show red arrows indicating burn directions on 3D plot')
args = parser.parse_args()

# Example TLE data for satellite at ~5000 km altitude (circular orbit)
# SMA = 5000 + 6371 = 11371 km, very stable orbit
line1 = '1 39634U 14016A   24001.50000000  .00000001  00000-0  10000-5 0  9998'
line2 = '2 39634  55.0000 150.0000 0000500  45.0000 315.0000 12.56637061123456'

# # Example TLE data for satellite at ~5000 km altitude (circular orbit)
# # SMA = 5000 + 6371 = 11371 km, very stable orbit
# line1 = '1 39634U 14016A   24001.50000000  .00000001  00000-0  10000-5 0  9998'
# line2 = '2 39634  55.0000 150.0000 0000500  45.0000 315.0000 12.56637061123456'

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
print("Propagating coast-burn-coast trajectory...\n")

# Use TLE epoch as the start time to avoid propagation errors
start_time           = epoch_datetime
minutes_to_propagate = int(90 * 15)

# Define two burn periods
# First burn period: 10% to 50% of total time
burn_1_start   = int(minutes_to_propagate * 0.10)
burn_1_end     = int(minutes_to_propagate * 0.15)
burn_minutes_1 = list(range(burn_1_start, burn_1_end + 1))

# Second burn period: 60% to 90% of total time
burn_2_start = int(minutes_to_propagate * 0.60)
burn_2_end = int(minutes_to_propagate * 0.65)
burn_minutes_2 = list(range(burn_2_start, burn_2_end + 1))

burn_minutes = burn_minutes_1 + burn_minutes_2

# Define delta-V in velocity-normal-binormal (VNB) frame [m/s]
delta_v_vnb = np.array([1.0e0, 0.0, 0.0])  # 10 m/s prograde burn

print(f"Total propagation time: {minutes_to_propagate} minutes")
print(f"\nFirst burn period ({len(burn_minutes_1)} maneuvers):")
for i, bm in enumerate(burn_minutes_1, 1):
    print(f"  Burn {i} at t = {bm} minutes")
print(f"\nCoast phase: {burn_minutes_1[-1]} to {burn_minutes_2[0]} minutes")
print(f"\nSecond burn period ({len(burn_minutes_2)} maneuvers):")
for i, bm in enumerate(burn_minutes_2, len(burn_minutes_1) + 1):
    print(f"  Burn {i} at t = {bm} minutes")
print(f"\nDelta-V (VNB frame): [{delta_v_vnb[0]:.1f}, {delta_v_vnb[1]:.1f}, {delta_v_vnb[2]:.1f}] m/s\n")

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
print(f"  SMA  : {tle_coe['sma']/1000.0:>13.6e} km")
print(f"  ECC  : {tle_coe['ecc']:>13.6e}")
print(f"  INC  : {np.rad2deg(tle_coe['inc']):>13.6e} deg")
print(f"  RAAN : {np.rad2deg(tle_coe['raan']):>13.6e} deg")
print(f"  ARGP : {np.rad2deg(tle_coe['argp']):>13.6e} deg")
print(f"  MA   : {np.rad2deg(tle_coe['ma']):>13.6e} deg")
print("")

# Initialize dynamics calculator
dynamics = Dynamics()

# Store position and velocity data
pos_x = []
pos_y = []
pos_z = []
burn_count = 0
burn_locations = []
burn_data = []  # Store position and velocity at each burn
delta_v_history = []  # Store delta-v at each time step

for min in range(0, minutes_to_propagate + 1):
    current_time = start_time + timedelta(minutes=min)
    jd, fr = jday(
        current_time.year, current_time.month, current_time.day,
        current_time.hour, current_time.minute, current_time.second,
    )
    
    error_code, position, velocity = satellite.sgp4(jd, fr)
    
    if error_code == 0:
        # Apply delta-V at burn times
        if min in burn_minutes and burn_count < len(burn_minutes):
            burn_num = burn_count + 1
            # print(f"Applying delta-V #{burn_num} at t = {min} minutes...")
            
            # Convert to numpy arrays and meters
            pos_m = np.array(position) * 1000.0  # km to m
            vel_m = np.array(velocity) * 1000.0  # km/s to m/s
            
            # Store burn data for visualization
            vel_dir = vel_m / np.linalg.norm(vel_m)
            burn_data.append({
                'pos': np.array(position),  # in km
                'vel_dir': vel_dir
            })
            
            # Construct VNB frame
            v_hat = vel_m / np.linalg.norm(vel_m)  # Velocity direction
            h = np.cross(pos_m, vel_m)  # Angular momentum
            n_hat = h / np.linalg.norm(h)  # Normal direction
            b_hat = np.cross(v_hat, n_hat)  # Binormal direction
            
            # Rotation matrix from VNB to inertial
            R_vnb_to_eci = np.column_stack([v_hat, n_hat, b_hat])
            
            # Convert delta-V to inertial frame
            delta_v_eci = R_vnb_to_eci @ delta_v_vnb
            
            # Apply delta-V
            vel_m_new = vel_m + delta_v_eci
            
            # Store delta-v in VNB frame and magnitude
            delta_v_mag = np.linalg.norm(delta_v_vnb)
            delta_v_history.append({
                'v': delta_v_vnb[0],
                'n': delta_v_vnb[1],
                'b': delta_v_vnb[2],
                'mag': delta_v_mag,
                'x': delta_v_eci[0],
                'y': delta_v_eci[1],
                'z': delta_v_eci[2]
            })
            
            # Update velocity for this point
            velocity = tuple(vel_m_new / 1000.0)  # Convert back to km/s
            
            # print(f"  Pre-burn velocity  : [{vel_m[0]/1000:.3f}, {vel_m[1]/1000:.3f}, {vel_m[2]/1000:.3f}] km/s")
            # print(f"  Post-burn velocity : [{vel_m_new[0]/1000:.3f}, {vel_m_new[1]/1000:.3f}, {vel_m_new[2]/1000:.3f}] km/s\n")
        
            # Create new Satrec from post-burn state using mean elements conversion
            satellite = SGP4Helper.create_satrec_from_state2(
                pos_m, vel_m_new, current_time, bstar=0.0
            )
            
            burn_locations.append(min)
            burn_count += 1
            # print(f"  Created new TLE from post-burn state\n")
        
        else:
            # No burn at this time
            delta_v_history.append({
                'v': 0.0,
                'n': 0.0,
                'b': 0.0,
                'mag': 0.0,
                'x': 0.0,
                'y': 0.0,
                'z': 0.0
            })
        
        time_hours.append(min / 60.0)
        
        # Store position components (in km)
        pos_x.append(position[0])
        pos_y.append(position[1])
        pos_z.append(position[2])
        
        # Convert position and velocity to orbital elements
        pos_m = np.array(position) * 1000.0
        vel_m = np.array(velocity) * 1000.0
        coe = dynamics.rv2coe(pos_m, vel_m)
        
        semi_major_axis.append(coe['sma'] / 1000.0)
        eccentricity.append(coe['ecc'])
        inclination.append(np.rad2deg(coe['inc']))
        raan.append(np.rad2deg(coe['raan']))
        arg_perigee.append(np.rad2deg(coe['argp']))
        mean_anomaly.append(np.rad2deg(coe['ma']))

    else:
        print(f"\nERROR: SGP4 propagation failed at t = {min} minutes")
        print(f"Error code: {error_code}")
        error_messages = {
            1: "Mean elements invalid (ecc >= 1.0 or ecc < -0.001 or a < 0.95 er)",
            2: "Mean motion less than 0.0",
            3: "Perturbed elements, ecc < 0.0 or ecc > 1.0",
            4: "Semi-latus rectum < 0.0",
            5: "Epoch elements are sub-orbital",
            6: "Satellite has decayed"
        }
        print(f"Error message: {error_messages.get(error_code, 'Unknown error')}")
        print(f"Last successful propagation at t = {min-1} minutes")
        if len(pos_x) > 0:
            last_pos = np.array([pos_x[-1], pos_y[-1], pos_z[-1]])
            print(f"Position (km): {np.linalg.norm(last_pos):.3f} [{last_pos[0]:.3f}, {last_pos[1]:.3f}, {last_pos[2]:.3f}]")
        # Update burn end times if propagation failed during burn
        if min <= burn_1_end:
            burn_1_end = min - 1
        if min <= burn_2_end:
            burn_2_end = min - 1
        break

# Convert to numpy arrays
time_hours = np.array(time_hours)
time_days = time_hours / 24.0

# Extract delta-v components
delta_v_v = np.array([dv['v'] for dv in delta_v_history])
delta_v_n = np.array([dv['n'] for dv in delta_v_history])
delta_v_b = np.array([dv['b'] for dv in delta_v_history])
delta_v_mag = np.array([dv['mag'] for dv in delta_v_history])
delta_v_x = np.array([dv['x'] for dv in delta_v_history])
delta_v_y = np.array([dv['y'] for dv in delta_v_history])
delta_v_z = np.array([dv['z'] for dv in delta_v_history])

# Calculate acceleration (delta-V / 60 seconds)
accel_x = delta_v_x / 60.0  # m/s^2
accel_y = delta_v_y / 60.0  # m/s^2
accel_z = delta_v_z / 60.0  # m/s^2
accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

# Calculate mass using rocket equation: m = m0 * exp(-ΔV/Ve)
# Exhaust velocity and initial mass
v_exhaust = 3000.0  # m/s
mass_initial = 1000.0  # kg

# Calculate cumulative delta-V
cumulative_dv = np.cumsum(delta_v_mag)  # m/s

# Calculate mass at each time step
mass = mass_initial * np.exp(-cumulative_dv / v_exhaust)

# Calculate thrust: F = m * a = m * (ΔV / Δt)
# where Δt = 60 seconds per our acceleration calculation
thrust = mass * (delta_v_mag / 60.0)  # Newtons
thrust_x = mass * (delta_v_x / 60.0)  # Newtons
thrust_y = mass * (delta_v_y / 60.0)  # Newtons
thrust_z = mass * (delta_v_z / 60.0)  # Newtons

# Create plots with 3D trajectory
fig = plt.figure(figsize=(16, 8))
fig.suptitle('Coast-Burn-Coast Trajectory and Orbital Elements Evolution', fontsize=16)

# Create 3x3 grid
gs = fig.add_gridspec(3, 3)

# Left column: 3D trajectory (spans all 3 rows)
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')

# Add Earth sphere
earth_radius = 6371.0  # km
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax_3d.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3)

# Plot trajectory with different colors for burn and coast phases
# First, plot coast phase from start to first burn
if burn_1_start < len(pos_x):
    ax_3d.plot(pos_x[:burn_1_start], pos_y[:burn_1_start], pos_z[:burn_1_start], 
               'k-', linewidth=1.5, label='Coast')

# Plot first burn phase in red
if burn_1_end < len(pos_x):
    ax_3d.plot(pos_x[burn_1_start:burn_1_end+1], pos_y[burn_1_start:burn_1_end+1], 
               pos_z[burn_1_start:burn_1_end+1], 'r-', linewidth=1.5, label='Burn')

# Plot coast phase between burns
if burn_2_start < len(pos_x):
    ax_3d.plot(pos_x[burn_1_end:burn_2_start], pos_y[burn_1_end:burn_2_start], 
               pos_z[burn_1_end:burn_2_start], 'k-', linewidth=1.5)

# Plot second burn phase in red
if burn_2_end < len(pos_x):
    ax_3d.plot(pos_x[burn_2_start:burn_2_end+1], pos_y[burn_2_start:burn_2_end+1], 
               pos_z[burn_2_start:burn_2_end+1], 'r-', linewidth=1.5)

# Plot final coast phase
if burn_2_end < len(pos_x):
    ax_3d.plot(pos_x[burn_2_end:], pos_y[burn_2_end:], pos_z[burn_2_end:], 
               'k-', linewidth=1.5)

ax_3d.scatter(pos_x[0], pos_y[0], pos_z[0], edgecolors='k', facecolors='w', marker='^', s=100, linewidths=2, label='Start')
# Draw red lines showing burn directions (if flag is set)
if args.show_burns:
    imp_mnv_scale = 3.0
    for i, burn in enumerate(burn_data):
        pos = burn['pos']
        vel_dir = burn['vel_dir'] * 1000.0  # Scale by 1000 km for visibility
        end_pos = pos + imp_mnv_scale * vel_dir
        ax_3d.plot([pos[0], end_pos[0]], 
                   [pos[1], end_pos[1]], 
                   [pos[2], end_pos[2]], 
                   'r-', linewidth=1, alpha=0.3)
ax_3d.scatter(pos_x[-1], pos_y[-1], pos_z[-1], edgecolors='k', facecolors='w', marker='s', s=100, linewidths=2, label='Stop')
ax_3d.set_xlabel('X [km]')
ax_3d.set_ylabel('Y [km]')
ax_3d.set_zlabel('Z [km]')
ax_3d.set_xlim([-15000, 15000])
ax_3d.set_ylim([-15000, 15000])
ax_3d.set_zlim([-15000, 15000])
ax_3d.set_title('3D Trajectory')
ax_3d.legend(loc='best')
ax_3d.grid(True, alpha=0.3)
ax_3d.set_box_aspect([1,1,1])  # Equal aspect ratio

# Middle column: Orbital elements
ax_sma = fig.add_subplot(gs[0, 1])
ax_sma.plot(time_days, semi_major_axis, 'k-', linewidth=1.5)
# Add red segments during burns
if burn_1_end < len(time_days):
    ax_sma.plot(time_days[burn_1_start:burn_1_end+1], semi_major_axis[burn_1_start:burn_1_end+1], 'r-', linewidth=1.5)
if burn_2_end < len(time_days):
    ax_sma.plot(time_days[burn_2_start:burn_2_end+1], semi_major_axis[burn_2_start:burn_2_end+1], 'r-', linewidth=1.5)
ax_sma.set_ylabel('Semi-major Axis (km)')
ax_sma.grid(True, alpha=0.3)

ax_ecc = fig.add_subplot(gs[1, 1], sharex=ax_sma)
ax_ecc.plot(time_days, eccentricity, 'k-', linewidth=1.5)
# Add red segments during burns
if burn_1_end < len(time_days):
    ax_ecc.plot(time_days[burn_1_start:burn_1_end+1], eccentricity[burn_1_start:burn_1_end+1], 'r-', linewidth=1.5)
if burn_2_end < len(time_days):
    ax_ecc.plot(time_days[burn_2_start:burn_2_end+1], eccentricity[burn_2_start:burn_2_end+1], 'r-', linewidth=1.5)
ax_ecc.set_ylabel('Eccentricity')
ax_ecc.grid(True, alpha=0.3)

ax_inc = fig.add_subplot(gs[2, 1], sharex=ax_sma)
ax_inc.plot(time_days, inclination, 'k-', linewidth=1.5)
# Add red segments during burns
if burn_1_end < len(time_days):
    ax_inc.plot(time_days[burn_1_start:burn_1_end+1], inclination[burn_1_start:burn_1_end+1], 'r-', linewidth=1.5)
if burn_2_end < len(time_days):
    ax_inc.plot(time_days[burn_2_start:burn_2_end+1], inclination[burn_2_start:burn_2_end+1], 'r-', linewidth=1.5)
ax_inc.set_ylabel('Inclination (deg)')
ax_inc.set_xlabel('Time (days)')
ax_inc.grid(True, alpha=0.3)

# Right column top 3 plots: Remaining orbital elements
ax_raan = fig.add_subplot(gs[0, 2], sharex=ax_sma)
ax_raan.plot(time_days, raan, 'k-', linewidth=1.5)
# Add red segments during burns
if burn_1_end < len(time_days):
    ax_raan.plot(time_days[burn_1_start:burn_1_end+1], raan[burn_1_start:burn_1_end+1], 'r-', linewidth=1.5)
if burn_2_end < len(time_days):
    ax_raan.plot(time_days[burn_2_start:burn_2_end+1], raan[burn_2_start:burn_2_end+1], 'r-', linewidth=1.5)
ax_raan.set_ylabel('RAAN (deg)')
ax_raan.grid(True, alpha=0.3)

ax_argp = fig.add_subplot(gs[1, 2], sharex=ax_sma)
ax_argp.plot(time_days, arg_perigee, 'k-', linewidth=1.5)
# Add red segments during burns
if burn_1_end < len(time_days):
    ax_argp.plot(time_days[burn_1_start:burn_1_end+1], arg_perigee[burn_1_start:burn_1_end+1], 'r-', linewidth=1.5)
if burn_2_end < len(time_days):
    ax_argp.plot(time_days[burn_2_start:burn_2_end+1], arg_perigee[burn_2_start:burn_2_end+1], 'r-', linewidth=1.5)
ax_argp.set_ylabel('Arg. of Perigee (deg)')
ax_argp.grid(True, alpha=0.3)

ax_ma = fig.add_subplot(gs[2, 2], sharex=ax_sma)
ax_ma.plot(time_days, mean_anomaly, 'k-', linewidth=1.5)
# Add red segments during burns
if burn_1_end < len(time_days):
    ax_ma.plot(time_days[burn_1_start:burn_1_end+1], mean_anomaly[burn_1_start:burn_1_end+1], 'r-', linewidth=1.5)
if burn_2_end < len(time_days):
    ax_ma.plot(time_days[burn_2_start:burn_2_end+1], mean_anomaly[burn_2_start:burn_2_end+1], 'r-', linewidth=1.5)
ax_ma.set_ylabel('Mean Anomaly (deg)')
ax_ma.set_xlabel('Time (days)')
ax_ma.grid(True, alpha=0.3)

plt.tight_layout()
output_file = Path.cwd() / 'coast_burn_coast.png'
plt.savefig(output_file, dpi=150)

# Create second figure for delta-V and acceleration
fig2 = plt.figure(figsize=(16, 8))
fig2.suptitle('Delta-V, Acceleration, Mass, and Thrust History', fontsize=16)

# Create 4x1 grid for stacked plots
gs2 = fig2.add_gridspec(4, 1)

# Top: Delta-V plot
ax_dv = fig2.add_subplot(gs2[0, 0])
ax_dv.plot(time_days, delta_v_x, 'b-', linewidth=1.5, label='ΔV-X (m/s)', alpha=0.7)
ax_dv.plot(time_days, delta_v_y, 'g-', linewidth=1.5, label='ΔV-Y (m/s)', alpha=0.7)
ax_dv.plot(time_days, delta_v_z, 'c-', linewidth=1.5, label='ΔV-Z (m/s)', alpha=0.7)
ax_dv.plot(time_days, delta_v_mag, 'r-', linewidth=2.0, label='ΔV-Mag (m/s)')
ax_dv.set_ylabel('Delta-V (m/s)')
ax_dv.legend(loc='best', fontsize=10)
ax_dv.grid(True, alpha=0.3)
ax_dv.set_title('Delta-V Components (ECI)')

# Second: Acceleration plot
ax_accel = fig2.add_subplot(gs2[1, 0], sharex=ax_dv)
ax_accel.plot(time_days, accel_x * 1000, 'b-', linewidth=1.5, label='Accel-X (mm/s²)', alpha=0.7)
ax_accel.plot(time_days, accel_y * 1000, 'g-', linewidth=1.5, label='Accel-Y (mm/s²)', alpha=0.7)
ax_accel.plot(time_days, accel_z * 1000, 'c-', linewidth=1.5, label='Accel-Z (mm/s²)', alpha=0.7)
ax_accel.plot(time_days, accel_mag * 1000, 'r-', linewidth=2.0, label='Accel-Mag (mm/s²)')
ax_accel.set_ylabel('Acceleration (mm/s²)')
ax_accel.legend(loc='best', fontsize=10)
ax_accel.grid(True, alpha=0.3)
ax_accel.set_title('Acceleration (ΔV/60s, ECI)')

# Third: Thrust plot
ax_thrust = fig2.add_subplot(gs2[2, 0], sharex=ax_dv)
ax_thrust.plot(time_days, thrust_x, 'b-', linewidth=1.5, label='Thrust-X (N)', alpha=0.7)
ax_thrust.plot(time_days, thrust_y, 'g-', linewidth=1.5, label='Thrust-Y (N)', alpha=0.7)
ax_thrust.plot(time_days, thrust_z, 'c-', linewidth=1.5, label='Thrust-Z (N)', alpha=0.7)
ax_thrust.plot(time_days, thrust, 'r-', linewidth=2.0, label='Thrust-Mag (N)')
ax_thrust.set_ylabel('Thrust (N)')
ax_thrust.legend(loc='best', fontsize=10)
ax_thrust.grid(True, alpha=0.3)
ax_thrust.set_title('Thrust Components (F = m × ΔV/60s, ECI)')

# Bottom: Mass plot
ax_mass = fig2.add_subplot(gs2[3, 0], sharex=ax_dv)
ax_mass.plot(time_days, mass, 'k-', linewidth=2.0, label=f'Mass (Ve={v_exhaust}m/s, m0={mass_initial}kg)')
ax_mass.set_ylabel('Mass (kg)')
ax_mass.set_xlabel('Time (days)')
ax_mass.legend(loc='best', fontsize=10)
ax_mass.grid(True, alpha=0.3)
ax_mass.set_title('Spacecraft Mass (Rocket Equation)')

plt.tight_layout()
output_file2 = Path.cwd() / 'coast_burn_coast_deltav.png'
fig2.savefig(output_file2, dpi=150)

plt.show()

print(f"Propagation complete. Plotted {len(time_hours)} data points.")
print(f"Plot saved to {output_file}")
print(f"Delta-V plot saved to {output_file2}")
print(f"Final mass: {mass[-1]:.2f} kg (propellant used: {mass_initial - mass[-1]:.2f} kg)")
print(f"Peak thrust: {np.max(thrust):.2f} N")
