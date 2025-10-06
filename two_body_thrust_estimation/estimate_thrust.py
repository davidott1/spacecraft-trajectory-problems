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

# Define constant thrust instead of constant delta-V
constant_thrust = 15.0  # Newtons (constant)
delta_time = 60.0  # seconds per burn step

# Exhaust velocity and initial mass (needed for mass calculation)
v_exhaust = 3000.0  # m/s
mass_initial = 1000.0  # kg

print(f"Total propagation time: {minutes_to_propagate} minutes")
print(f"\nFirst burn period ({len(burn_minutes_1)} maneuvers):")
for i, bm in enumerate(burn_minutes_1, 1):
    print(f"  Burn {i} at t = {bm} minutes")
print(f"\nCoast phase: {burn_minutes_1[-1]} to {burn_minutes_2[0]} minutes")
print(f"\nSecond burn period ({len(burn_minutes_2)} maneuvers):")
for i, bm in enumerate(burn_minutes_2, len(burn_minutes_1) + 1):
    print(f"  Burn {i} at t = {bm} minutes")
print(f"\nConstant thrust: {constant_thrust:.1f} N")
print(f"Delta-time per burn: {delta_time:.1f} seconds")
print(f"Initial mass: {mass_initial:.1f} kg")
print(f"Exhaust velocity: {v_exhaust:.1f} m/s\n")

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
vel_x = []
vel_y = []
vel_z = []
burn_count = 0
burn_locations = []
burn_data = []  # Store position and velocity at each burn
delta_v_history = []  # Store delta-v at each time step
current_mass = mass_initial  # Track current mass

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
            
            # Calculate delta-V from constant thrust: ΔV = F * Δt / m
            delta_v_magnitude = (constant_thrust * delta_time) / current_mass  # m/s
            
            # Delta-V only in velocity direction (VNB: V component only)
            delta_v_vnb = np.array([delta_v_magnitude, 0.0, 0.0])
            
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
            
            # Update mass using rocket equation: m_new = m_old * exp(-ΔV/Ve)
            current_mass = current_mass * np.exp(-delta_v_magnitude / v_exhaust)
            
            # Store delta-v in VNB frame and magnitude
            delta_v_history.append({
                'v': delta_v_vnb[0],
                'n': delta_v_vnb[1],
                'b': delta_v_vnb[2],
                'mag': delta_v_magnitude,
                'x': delta_v_eci[0],
                'y': delta_v_eci[1],
                'z': delta_v_eci[2],
                'mass': current_mass
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
                'z': 0.0,
                'mass': current_mass
            })
        
        time_hours.append(min / 60.0)
        
        # Store position components (in km)
        pos_x.append(position[0])
        pos_y.append(position[1])
        pos_z.append(position[2])
        
        # Store velocity components (in km/s)
        vel_x.append(velocity[0])
        vel_y.append(velocity[1])
        vel_z.append(velocity[2])
        
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
mass = np.array([dv['mass'] for dv in delta_v_history])

# Calculate acceleration (delta-V / 60 seconds)
accel_x = delta_v_x / delta_time  # m/s^2
accel_y = delta_v_y / delta_time  # m/s^2
accel_z = delta_v_z / delta_time  # m/s^2
accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

# Calculate thrust: F = m * a = m * (ΔV / Δt)
thrust = mass * (delta_v_mag / delta_time)  # Newtons (should be constant at 15 N during burns)
thrust_x = mass * (delta_v_x / delta_time)  # Newtons
thrust_y = mass * (delta_v_y / delta_time)  # Newtons
thrust_z = mass * (delta_v_z / delta_time)  # Newtons

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
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_sma.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                   color='purple', alpha=0.2)
if burn_2_end < len(time_days):
    ax_sma.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                   color='purple', alpha=0.2)
ax_sma.set_ylabel('Semi-major Axis (km)')
ax_sma.grid(True, alpha=0.3)

ax_ecc = fig.add_subplot(gs[1, 1], sharex=ax_sma)
ax_ecc.plot(time_days, eccentricity, 'k-', linewidth=1.5)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_ecc.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                   color='purple', alpha=0.2)
if burn_2_end < len(time_days):
    ax_ecc.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                   color='purple', alpha=0.2)
ax_ecc.set_ylabel('Eccentricity')
ax_ecc.grid(True, alpha=0.3)

ax_inc = fig.add_subplot(gs[2, 1], sharex=ax_sma)
ax_inc.plot(time_days, inclination, 'k-', linewidth=1.5)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_inc.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                   color='purple', alpha=0.2)
if burn_2_end < len(time_days):
    ax_inc.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                   color='purple', alpha=0.2)
ax_inc.set_ylabel('Inclination (deg)')
ax_inc.set_xlabel('Time (days)')
ax_inc.grid(True, alpha=0.3)

# Right column top 3 plots: Remaining orbital elements
ax_raan = fig.add_subplot(gs[0, 2], sharex=ax_sma)
ax_raan.plot(time_days, raan, 'k-', linewidth=1.5)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_raan.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                    color='purple', alpha=0.2)
if burn_2_end < len(time_days):
    ax_raan.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                    color='purple', alpha=0.2)
ax_raan.set_ylabel('RAAN (deg)')
ax_raan.grid(True, alpha=0.3)

ax_argp = fig.add_subplot(gs[1, 2], sharex=ax_sma)
ax_argp.plot(time_days, arg_perigee, 'k-', linewidth=1.5)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_argp.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                    color='purple', alpha=0.2)
if burn_2_end < len(time_days):
    ax_argp.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                    color='purple', alpha=0.2)
ax_argp.set_ylabel('Arg. of Perigee (deg)')
ax_argp.grid(True, alpha=0.3)

ax_ma = fig.add_subplot(gs[2, 2], sharex=ax_sma)
ax_ma.plot(time_days, mean_anomaly, 'k-', linewidth=1.5)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_ma.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                  color='purple', alpha=0.2)
if burn_2_end < len(time_days):
    ax_ma.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                  color='purple', alpha=0.2)
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

# Create third figure for position, velocity, and acceleration vs time
fig3 = plt.figure(figsize=(16, 8))
fig3.suptitle('Position, Velocity, Acceleration, and Jerk vs Time (ECI)', fontsize=16)

# Create 6x1 grid for stacked plots
gs3 = fig3.add_gridspec(6, 1)

# Convert to numpy arrays
pos_x_arr = np.array(pos_x)
pos_y_arr = np.array(pos_y)
pos_z_arr = np.array(pos_z)
vel_x_arr = np.array(vel_x)
vel_y_arr = np.array(vel_y)
vel_z_arr = np.array(vel_z)
vel_mag = np.sqrt(vel_x_arr**2 + vel_y_arr**2 + vel_z_arr**2)

# Calculate analytical total acceleration (two-body gravity + thrust)
total_accel_x = np.zeros(len(pos_x_arr))
total_accel_y = np.zeros(len(pos_y_arr))
total_accel_z = np.zeros(len(pos_z_arr))

for i in range(len(pos_x_arr)):
    # Position in meters
    pos_vec = np.array([pos_x_arr[i], pos_y_arr[i], pos_z_arr[i]]) * 1000.0  # km to m
    r_mag = np.linalg.norm(pos_vec)
    
    # Two-body gravity acceleration (in m/s^2)
    gravity_accel = -(Constants.MU_EARTH / r_mag**3) * pos_vec
    
    # Thrust acceleration (already calculated, in m/s^2)
    thrust_accel = np.array([accel_x[i], accel_y[i], accel_z[i]])
    
    # Total acceleration
    total_accel = gravity_accel + thrust_accel
    total_accel_x[i] = total_accel[0]
    total_accel_y[i] = total_accel[1]
    total_accel_z[i] = total_accel[2]

total_accel_mag = np.sqrt(total_accel_x**2 + total_accel_y**2 + total_accel_z**2)

# Calculate analytical jerk: d/dt(a_gravity + a_thrust)
# For gravity: jerk_gravity = d/dt(-μ/r³ * r) = -μ * d/dt(r/r³) = -μ * (v/r³ - 3*(r·v)*r/r⁵)
# For thrust: jerk_thrust = d/dt(F/m) = (F/m²) * dm/dt (if thrust is constant)
jerk_x = np.zeros(len(pos_x_arr))
jerk_y = np.zeros(len(pos_y_arr))
jerk_z = np.zeros(len(pos_z_arr))
jerk_gravity_x = np.zeros(len(pos_x_arr))
jerk_gravity_y = np.zeros(len(pos_y_arr))
jerk_gravity_z = np.zeros(len(pos_z_arr))
jerk_thrust_x = np.zeros(len(pos_x_arr))
jerk_thrust_y = np.zeros(len(pos_y_arr))
jerk_thrust_z = np.zeros(len(pos_z_arr))

for i in range(len(pos_x_arr)):
    pos_vec = np.array([pos_x_arr[i], pos_y_arr[i], pos_z_arr[i]]) * 1000.0  # km to m
    vel_vec = np.array([vel_x_arr[i], vel_y_arr[i], vel_z_arr[i]]) * 1000.0  # km/s to m/s
    r_mag = np.linalg.norm(pos_vec)
    r_dot_v = np.dot(pos_vec, vel_vec)
    
    # Gravity jerk: -μ * (v/r³ - 3*(r·v)*r/r⁵)
    gravity_jerk = -Constants.MU_EARTH * (vel_vec / r_mag**3 - 3 * r_dot_v * pos_vec / r_mag**5)
    jerk_gravity_x[i] = gravity_jerk[0]
    jerk_gravity_y[i] = gravity_jerk[1]
    jerk_gravity_z[i] = gravity_jerk[2]
    
    # Thrust jerk: only non-zero if mass is changing (during burns)
    # d/dt(F/m) = (F/m²) * dm/dt, where dm/dt = -F/v_exhaust (from rocket equation)
    thrust_accel_vec = np.array([accel_x[i], accel_y[i], accel_z[i]])
    if np.linalg.norm(thrust_accel_vec) > 1e-10:  # During burn
        # dm/dt = -F/v_exhaust = -m*a/v_exhaust
        dm_dt = -mass[i] * np.linalg.norm(thrust_accel_vec) / v_exhaust
        thrust_jerk = (thrust_accel_vec / mass[i]) * dm_dt
    else:
        thrust_jerk = np.zeros(3)
    
    jerk_thrust_x[i] = thrust_jerk[0]
    jerk_thrust_y[i] = thrust_jerk[1]
    jerk_thrust_z[i] = thrust_jerk[2];
    
    # Total jerk
    total_jerk = gravity_jerk + thrust_jerk
    jerk_x[i] = total_jerk[0]
    jerk_y[i] = total_jerk[1]
    jerk_z[i] = total_jerk[2]

jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
jerk_gravity_mag = np.sqrt(jerk_gravity_x**2 + jerk_gravity_y**2 + jerk_gravity_z**2)
jerk_thrust_mag = np.sqrt(jerk_thrust_x**2 + jerk_thrust_y**2 + jerk_thrust_z**2)

# Position plot
ax_pos = fig3.add_subplot(gs3[0, 0])
ax_pos.plot(time_days, pos_x_arr, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_pos.plot(time_days, pos_y_arr, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_pos.plot(time_days, pos_z_arr, 'b-', linewidth=1.5, label='Z', alpha=0.8)
pos_mag = np.sqrt(pos_x_arr**2 + pos_y_arr**2 + pos_z_arr**2)
ax_pos.plot(time_days, pos_mag, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_pos.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                   color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_pos.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                   color='purple', alpha=0.2)
ax_pos.set_ylabel('Position (km)')
ax_pos.grid(True, alpha=0.3)
ax_pos.legend(loc='best', fontsize=10)
ax_pos.set_title('Position Components (X=red, Y=green, Z=blue, Mag=black)')

# Velocity plot
ax_vel = fig3.add_subplot(gs3[1, 0], sharex=ax_pos)
ax_vel.plot(time_days, vel_x_arr, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_vel.plot(time_days, vel_y_arr, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_vel.plot(time_days, vel_z_arr, 'b-', linewidth=1.5, label='Z', alpha=0.8)
ax_vel.plot(time_days, vel_mag, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_vel.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                   color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_vel.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                   color='purple', alpha=0.2)
ax_vel.set_ylabel('Velocity (km/s)')
ax_vel.grid(True, alpha=0.3)
ax_vel.legend(loc='best', fontsize=10)
ax_vel.set_title('Velocity Components (X=red, Y=green, Z=blue, Mag=black)')

# Total acceleration plot (gravity + thrust)
ax_accel = fig3.add_subplot(gs3[2, 0], sharex=ax_pos)
ax_accel.plot(time_days, total_accel_x * 1000, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_accel.plot(time_days, total_accel_y * 1000, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_accel.plot(time_days, total_accel_z * 1000, 'b-', linewidth=1.5, label='Z', alpha=0.8)
ax_accel.plot(time_days, total_accel_mag * 1000, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_accel.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                     color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_accel.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                     color='purple', alpha=0.2)
ax_accel.set_ylabel('Acceleration (mm/s²)')
ax_accel.set_xlabel('Time (days)')
ax_accel.grid(True, alpha=0.3)
ax_accel.legend(loc='best', fontsize=10)
ax_accel.set_title('Total Acceleration: Gravity + Thrust (X=red, Y=green, Z=blue, Mag=black)')

# Total jerk plot
ax_jerk = fig3.add_subplot(gs3[3, 0], sharex=ax_pos)
ax_jerk.plot(time_days, jerk_x * 1e6, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_jerk.plot(time_days, jerk_y * 1e6, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_jerk.plot(time_days, jerk_z * 1e6, 'b-', linewidth=1.5, label='Z', alpha=0.8)
ax_jerk.plot(time_days, jerk_mag * 1e6, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_jerk.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                    color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_jerk.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                    color='purple', alpha=0.2)
ax_jerk.set_ylabel('Total Jerk (μm/s³)')
ax_jerk.grid(True, alpha=0.3)
ax_jerk.legend(loc='best', fontsize=10)
ax_jerk.set_title('Total Jerk: Gravity + Thrust (X=red, Y=green, Z=blue, Mag=black)')

# Gravity jerk plot
ax_jerk_grav = fig3.add_subplot(gs3[4, 0], sharex=ax_pos)
ax_jerk_grav.plot(time_days, jerk_gravity_x * 1e6, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_jerk_grav.plot(time_days, jerk_gravity_y * 1e6, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_jerk_grav.plot(time_days, jerk_gravity_z * 1e6, 'b-', linewidth=1.5, label='Z', alpha=0.8)
ax_jerk_grav.plot(time_days, jerk_gravity_mag * 1e6, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_jerk_grav.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                         color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_jerk_grav.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                         color='purple', alpha=0.2)
ax_jerk_grav.set_ylabel('Gravity Jerk (μm/s³)')
ax_jerk_grav.grid(True, alpha=0.3)
ax_jerk_grav.legend(loc='best', fontsize=10)
ax_jerk_grav.set_title('Gravity Jerk Component (X=red, Y=green, Z=blue, Mag=black)')

# Thrust jerk plot
ax_jerk_thrust = fig3.add_subplot(gs3[5, 0], sharex=ax_pos)
ax_jerk_thrust.plot(time_days, jerk_thrust_x * 1e6, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_jerk_thrust.plot(time_days, jerk_thrust_y * 1e6, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_jerk_thrust.plot(time_days, jerk_thrust_z * 1e6, 'b-', linewidth=1.5, label='Z', alpha=0.8)
ax_jerk_thrust.plot(time_days, jerk_thrust_mag * 1e6, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_jerk_thrust.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                           color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_jerk_thrust.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                           color='purple', alpha=0.2)
ax_jerk_thrust.set_ylabel('Thrust Jerk (μm/s³)')
ax_jerk_thrust.set_xlabel('Time (days)')
ax_jerk_thrust.grid(True, alpha=0.3)
ax_jerk_thrust.legend(loc='best', fontsize=10)
ax_jerk_thrust.set_title('Thrust Jerk Component (X=red, Y=green, Z=blue, Mag=black)')

plt.tight_layout()
output_file3 = Path.cwd() / 'coast_burn_coast_position.png'
fig3.savefig(output_file3, dpi=150)

# Create fourth figure for finite-difference jerk comparison
fig4 = plt.figure(figsize=(16, 8))
fig4.suptitle('Finite-Difference Jerk vs Time (ECI)', fontsize=16)

# Create 3x1 grid for stacked plots
gs4 = fig4.add_gridspec(3, 1)

# Calculate gravity-only acceleration for finite-difference jerk
gravity_accel_x = np.zeros(len(pos_x_arr))
gravity_accel_y = np.zeros(len(pos_y_arr))
gravity_accel_z = np.zeros(len(pos_z_arr))

for i in range(len(pos_x_arr)):
    pos_vec = np.array([pos_x_arr[i], pos_y_arr[i], pos_z_arr[i]]) * 1000.0  # km to m
    r_mag = np.linalg.norm(pos_vec)
    gravity_accel = -(Constants.MU_EARTH / r_mag**3) * pos_vec
    gravity_accel_x[i] = gravity_accel[0]
    gravity_accel_y[i] = gravity_accel[1]
    gravity_accel_z[i] = gravity_accel[2]

# Calculate finite-difference jerks
jerk_fd_total_x = np.gradient(total_accel_x, time_days * 86400)  # m/s^3
jerk_fd_total_y = np.gradient(total_accel_y, time_days * 86400)  # m/s^3
jerk_fd_total_z = np.gradient(total_accel_z, time_days * 86400)  # m/s^3
jerk_fd_total_mag = np.sqrt(jerk_fd_total_x**2 + jerk_fd_total_y**2 + jerk_fd_total_z**2)

jerk_fd_gravity_x = np.gradient(gravity_accel_x, time_days * 86400)  # m/s^3
jerk_fd_gravity_y = np.gradient(gravity_accel_y, time_days * 86400)  # m/s^3
jerk_fd_gravity_z = np.gradient(gravity_accel_z, time_days * 86400)  # m/s^3
jerk_fd_gravity_mag = np.sqrt(jerk_fd_gravity_x**2 + jerk_fd_gravity_y**2 + jerk_fd_gravity_z**2)

jerk_fd_thrust_x = np.gradient(accel_x, time_days * 86400)  # m/s^3
jerk_fd_thrust_y = np.gradient(accel_y, time_days * 86400)  # m/s^3
jerk_fd_thrust_z = np.gradient(accel_z, time_days * 86400)  # m/s^3
jerk_fd_thrust_mag = np.sqrt(jerk_fd_thrust_x**2 + jerk_fd_thrust_y**2 + jerk_fd_thrust_z**2)

# Total jerk (finite-difference)
ax_jerk_fd_total = fig4.add_subplot(gs4[0, 0])
ax_jerk_fd_total.plot(time_days, jerk_fd_total_x * 1e6, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_jerk_fd_total.plot(time_days, jerk_fd_total_y * 1e6, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_jerk_fd_total.plot(time_days, jerk_fd_total_z * 1e6, 'b-', linewidth=1.5, label='Z', alpha=0.8)
ax_jerk_fd_total.plot(time_days, jerk_fd_total_mag * 1e6, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_jerk_fd_total.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                             color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_jerk_fd_total.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                             color='purple', alpha=0.2)
ax_jerk_fd_total.set_ylabel('Total Jerk FD (μm/s³)')
ax_jerk_fd_total.grid(True, alpha=0.3)
ax_jerk_fd_total.legend(loc='best', fontsize=10)
ax_jerk_fd_total.set_title('Total Jerk (Finite-Difference): Gravity + Thrust (X=red, Y=green, Z=blue, Mag=black)')

# Gravity jerk (finite-difference)
ax_jerk_fd_grav = fig4.add_subplot(gs4[1, 0], sharex=ax_jerk_fd_total)
ax_jerk_fd_grav.plot(time_days, jerk_fd_gravity_x * 1e6, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_jerk_fd_grav.plot(time_days, jerk_fd_gravity_y * 1e6, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_jerk_fd_grav.plot(time_days, jerk_fd_gravity_z * 1e6, 'b-', linewidth=1.5, label='Z', alpha=0.8)
ax_jerk_fd_grav.plot(time_days, jerk_fd_gravity_mag * 1e6, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_jerk_fd_grav.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                            color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_jerk_fd_grav.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                            color='purple', alpha=0.2)
ax_jerk_fd_grav.set_ylabel('Gravity Jerk FD (μm/s³)')
ax_jerk_fd_grav.grid(True, alpha=0.3)
ax_jerk_fd_grav.legend(loc='best', fontsize=10)
ax_jerk_fd_grav.set_title('Gravity Jerk (Finite-Difference) (X=red, Y=green, Z=blue, Mag=black)')

# Thrust jerk (finite-difference)
ax_jerk_fd_thrust = fig4.add_subplot(gs4[2, 0], sharex=ax_jerk_fd_total)
ax_jerk_fd_thrust.plot(time_days, jerk_fd_thrust_x * 1e6, 'r-', linewidth=1.5, label='X', alpha=0.8)
ax_jerk_fd_thrust.plot(time_days, jerk_fd_thrust_y * 1e6, 'g-', linewidth=1.5, label='Y', alpha=0.8)
ax_jerk_fd_thrust.plot(time_days, jerk_fd_thrust_z * 1e6, 'b-', linewidth=1.5, label='Z', alpha=0.8)
ax_jerk_fd_thrust.plot(time_days, jerk_fd_thrust_mag * 1e6, 'k-', linewidth=2.0, label='Magnitude', alpha=0.9)
# Add purple shading for thrust periods
if burn_1_end < len(time_days):
    ax_jerk_fd_thrust.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                              color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_jerk_fd_thrust.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                              color='purple', alpha=0.2)
ax_jerk_fd_thrust.set_ylabel('Thrust Jerk FD (μm/s³)')
ax_jerk_fd_thrust.set_xlabel('Time (days)')
ax_jerk_fd_thrust.grid(True, alpha=0.3)
ax_jerk_fd_thrust.legend(loc='best', fontsize=10)
ax_jerk_fd_thrust.set_title('Thrust Jerk (Finite-Difference) (X=red, Y=green, Z=blue, Mag=black)')

plt.tight_layout()
output_file4 = Path.cwd() / 'coast_burn_coast_jerk_fd.png'
fig4.savefig(output_file4, dpi=150)

# Generate position measurements with Gaussian noise
print("\n=== Generating Position Measurements ===")
sigma_pos = 1e+0 / 1000  # km (10 m position uncertainty)
sigma_vel = 1e-2 / 1000  # km/s (0.1 m/s velocity uncertainty)
print(f"Position measurement noise: σ = {sigma_pos*1000:.1f} m")
print(f"Velocity measurement noise: σ = {sigma_vel*1000:.1f} m/s")

measurements_pos = []
measurements_vel = []
for i in range(len(pos_x_arr)):
    true_pos = np.array([pos_x_arr[i], pos_y_arr[i], pos_z_arr[i]])
    true_vel = np.array([vel_x_arr[i], vel_y_arr[i], vel_z_arr[i]])
    
    noise_pos = np.random.normal(0, sigma_pos, size=3)
    noise_vel = np.random.normal(0, sigma_vel, size=3)
    
    meas_pos = true_pos + noise_pos
    meas_vel = true_vel + noise_vel
    
    measurements_pos.append(meas_pos)
    measurements_vel.append(meas_vel)

measurements_pos = np.array(measurements_pos)
measurements_vel = np.array(measurements_vel)

meas_pos_x = measurements_pos[:, 0]
meas_pos_y = measurements_pos[:, 1]
meas_pos_z = measurements_pos[:, 2]

meas_vel_x = measurements_vel[:, 0]
meas_vel_y = measurements_vel[:, 1]
meas_vel_z = measurements_vel[:, 2]

# Calculate measurement errors
pos_errors = measurements_pos - np.column_stack([pos_x_arr, pos_y_arr, pos_z_arr])
pos_error_mag = np.linalg.norm(pos_errors, axis=1)

vel_errors = measurements_vel - np.column_stack([vel_x_arr, vel_y_arr, vel_z_arr])
vel_error_mag = np.linalg.norm(vel_errors, axis=1)

print(f"Mean position error: {np.mean(pos_error_mag)*1000:.2f} m")
print(f"Std position error: {np.std(pos_error_mag)*1000:.2f} m")
print(f"Max position error: {np.max(pos_error_mag)*1000:.2f} m")
print(f"Mean velocity error: {np.mean(vel_error_mag)*1000:.2f} m/s")
print(f"Std velocity error: {np.std(vel_error_mag)*1000:.2f} m/s")
print(f"Max velocity error: {np.max(vel_error_mag)*1000:.2f} m/s")

# Create fifth figure for true vs measured position and velocity
fig5 = plt.figure(figsize=(16, 8))
fig5.suptitle('True vs Measured Position and Velocity (ECI)', fontsize=16)

# Create 4x1 grid for stacked plots
gs5 = fig5.add_gridspec(4, 1)

# Position XYZ combined
ax_pos = fig5.add_subplot(gs5[0, 0])
ax_pos.plot(time_days, pos_x_arr, 'r-', linewidth=2.0, label='True X', alpha=0.4)
ax_pos.plot(time_days, pos_y_arr, 'g-', linewidth=2.0, label='True Y', alpha=0.4)
ax_pos.plot(time_days, pos_z_arr, 'b-', linewidth=2.0, label='True Z', alpha=0.4)
ax_pos.plot(time_days, meas_pos_x, 'o', color='r', markersize=4, label='Meas X', alpha=0.6)
ax_pos.plot(time_days, meas_pos_y, 'o', color='g', markersize=4, label='Meas Y', alpha=0.6)
ax_pos.plot(time_days, meas_pos_z, 'o', color='b', markersize=4, label='Meas Z', alpha=0.6)
if burn_1_end < len(time_days):
    ax_pos.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                   color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_pos.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                   color='purple', alpha=0.2)
ax_pos.set_ylabel('Position (km)')
ax_pos.grid(True, alpha=0.3)
ax_pos.legend(loc='best', fontsize=9, ncol=2)
ax_pos.set_title(f'Position: True vs Measured (σ = {sigma_pos*1000:.1f} m)')

# Position error magnitude
ax_pos_err = fig5.add_subplot(gs5[1, 0], sharex=ax_pos)
ax_pos_err.plot(time_days, pos_error_mag * 1000, 'k-', linewidth=1.5, label='Error Magnitude', alpha=0.8)
ax_pos_err.axhline(y=sigma_pos*1000, color='r', linestyle='--', linewidth=2, label=f'1σ = {sigma_pos*1000:.1f} m')
ax_pos_err.axhline(y=3*sigma_pos*1000, color='orange', linestyle='--', linewidth=2, label=f'3σ = {3*sigma_pos*1000:.1f} m')
if burn_1_end < len(time_days):
    ax_pos_err.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                       color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_pos_err.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                       color='purple', alpha=0.2)
ax_pos_err.set_ylabel('Position Error (m)')
ax_pos_err.grid(True, alpha=0.3)
ax_pos_err.legend(loc='best', fontsize=10)
ax_pos_err.set_title('Position Measurement Error Magnitude')

# Velocity plot
ax_vel = fig5.add_subplot(gs5[2, 0], sharex=ax_pos)
ax_vel.plot(time_days, vel_x_arr, 'r-', linewidth=2.0, label='True X', alpha=0.4)
ax_vel.plot(time_days, vel_y_arr, 'g-', linewidth=2.0, label='True Y', alpha=0.4)
ax_vel.plot(time_days, vel_z_arr, 'b-', linewidth=2.0, label='True Z', alpha=0.4)
ax_vel.plot(time_days, meas_vel_x, 'o', color='r', markersize=4, label='Meas X', alpha=0.6)
ax_vel.plot(time_days, meas_vel_y, 'o', color='g', markersize=4, label='Meas Y', alpha=0.6)
ax_vel.plot(time_days, meas_vel_z, 'o', color='b', markersize=4, label='Meas Z', alpha=0.6)
if burn_1_end < len(time_days):
    ax_vel.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                   color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_vel.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                   color='purple', alpha=0.2)
ax_vel.set_ylabel('Velocity (km/s)')
ax_vel.grid(True, alpha=0.3)
ax_vel.legend(loc='best', fontsize=9, ncol=2)
ax_vel.set_title(f'Velocity: True vs Measured (σ = {sigma_vel*1000:.1f} m/s)')

# Velocity error magnitude
ax_vel_err = fig5.add_subplot(gs5[3, 0], sharex=ax_pos)
ax_vel_err.plot(time_days, vel_error_mag * 1000, 'k-', linewidth=1.5, label='Error Magnitude', alpha=0.8)
ax_vel_err.axhline(y=sigma_vel*1000, color='r', linestyle='--', linewidth=2, label=f'1σ = {sigma_vel*1000:.1f} m/s')
ax_vel_err.axhline(y=3*sigma_vel*1000, color='orange', linestyle='--', linewidth=2, label=f'3σ = {3*sigma_vel*1000:.1f} m/s')
if burn_1_end < len(time_days):
    ax_vel_err.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                       color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_vel_err.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                       color='purple', alpha=0.2)
ax_vel_err.set_ylabel('Velocity Error (m/s)')
ax_vel_err.set_xlabel('Time (days)')
ax_vel_err.grid(True, alpha=0.3)
ax_vel_err.legend(loc='best', fontsize=10)
ax_vel_err.set_title('Velocity Measurement Error Magnitude')

plt.tight_layout()
output_file5 = Path.cwd() / 'coast_burn_coast_measurements.png'
fig5.savefig(output_file5, dpi=150)

# Create sixth figure with position estimates
# Implement Extended Kalman Filter for state estimation
print("\n=== Running Extended Kalman Filter ===")

# EKF State: [x, y, z, vx, vy, vz] in km and km/s
state_dim = 6
meas_dim = 6  # measuring position and velocity

# Initialize state with true initial conditions
x_est = np.zeros((len(time_days), state_dim))
x_est[0, 0:3] = np.array([pos_x_arr[0], pos_y_arr[0], pos_z_arr[0]])  # km
x_est[0, 3:6] = np.array([vel_x_arr[0], vel_y_arr[0], vel_z_arr[0]])  # km/s

# Initialize covariance matrix (small initial uncertainty)
P = np.eye(state_dim) * 1e-6

# Process noise covariance (accounts for model uncertainty)
dt_sec = 60.0  # seconds between measurements
Q = np.eye(state_dim) * 1e-10  # Very small process noise

# Measurement noise covariance
R = np.zeros((meas_dim, meas_dim))
R[0:3, 0:3] = (sigma_pos**2) * np.eye(3)  # Position variance (km^2)
R[3:6, 3:6] = (sigma_vel**2) * np.eye(3)  # Velocity variance (km^2/s^2)

print(f"Initial state: pos = [{x_est[0,0]:.3f}, {x_est[0,1]:.3f}, {x_est[0,2]:.3f}] km")
print(f"               vel = [{x_est[0,3]:.6f}, {x_est[0,4]:.6f}, {x_est[0,5]:.6f}] km/s")
print(f"Process noise Q diagonal: {Q[0,0]:.2e}")
print(f"Measurement noise R (pos): {R[0,0]:.2e} km^2 ({np.sqrt(R[0,0])*1000:.1f} m)")
print(f"Measurement noise R (vel): {R[3,3]:.2e} km^2/s^2 ({np.sqrt(R[3,3])*1000:.1f} m/s)")

# Run EKF
for k in range(1, len(time_days)):
    # Previous state
    x_prev = x_est[k-1, :]
    
    # === PREDICTION STEP ===
    # State transition using two-body dynamics
    pos_m = x_prev[0:3] * 1000.0  # km to m
    vel_m = x_prev[3:6] * 1000.0  # km/s to m/s
    
    r_mag = np.linalg.norm(pos_m)
    
    # Two-body gravity acceleration
    accel_gravity = -(Constants.MU_EARTH / r_mag**3) * pos_m  # m/s^2
    
    # Simple Euler integration (could use RK4 for better accuracy)
    pos_new = pos_m + vel_m * dt_sec  # m
    vel_new = vel_m + accel_gravity * dt_sec  # m/s
    
    # Predicted state
    x_pred = np.zeros(state_dim)
    x_pred[0:3] = pos_new / 1000.0  # m to km
    x_pred[3:6] = vel_new / 1000.0  # m/s to km/s
    
    # State transition Jacobian F
    F = np.eye(state_dim)
    F[0:3, 3:6] = np.eye(3) * dt_sec  # Position changes with velocity
    
    # Gravity gradient for velocity changes
    mu = Constants.MU_EARTH / 1e9  # Convert to km^3/s^2
    r_km = np.linalg.norm(x_prev[0:3])
    
    # ∂a/∂r = -μ/r³ * I + 3μ/r⁵ * r*r^T
    gravity_jacobian = (-mu / r_km**3) * np.eye(3) + (3 * mu / r_km**5) * np.outer(x_prev[0:3], x_prev[0:3])
    F[3:6, 0:3] = gravity_jacobian * dt_sec
    
    # Predict covariance
    P_pred = F @ P @ F.T + Q
    
    # === UPDATE STEP ===
    # Measurement: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    z = np.zeros(meas_dim)
    z[0:3] = measurements_pos[k, :]
    z[3:6] = measurements_vel[k, :]
    
    # Measurement matrix (direct observation)
    H = np.eye(meas_dim, state_dim)
    
    # Innovation
    y = z - H @ x_pred
    
    # Innovation covariance
    S = H @ P_pred @ H.T + R
    
    # Kalman gain
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    # Update state
    x_est[k, :] = x_pred + K @ y
    
    # Update covariance
    P = (np.eye(state_dim) - K @ H) @ P_pred

# Extract estimated position and velocity
est_pos_x = x_est[:, 0]
est_pos_y = x_est[:, 1]
est_pos_z = x_est[:, 2]
est_vel_x = x_est[:, 3]
est_vel_y = x_est[:, 4]
est_vel_z = x_est[:, 5]

# Calculate estimation errors
est_pos_errors = x_est[:, 0:3] - np.column_stack([pos_x_arr, pos_y_arr, pos_z_arr])
est_pos_error_mag = np.linalg.norm(est_pos_errors, axis=1)

est_vel_errors = x_est[:, 3:6] - np.column_stack([vel_x_arr, vel_y_arr, vel_z_arr])
est_vel_error_mag = np.linalg.norm(est_vel_errors, axis=1)

print(f"\n=== EKF Results ===")
print(f"Mean position estimation error: {np.mean(est_pos_error_mag)*1000:.2f} m")
print(f"Std position estimation error: {np.std(est_pos_error_mag)*1000:.2f} m")
print(f"Max position estimation error: {np.max(est_pos_error_mag)*1000:.2f} m")
print(f"Mean velocity estimation error: {np.mean(est_vel_error_mag)*1000:.2f} m/s")
print(f"Std velocity estimation error: {np.std(est_vel_error_mag)*1000:.2f} m/s")
print(f"Max velocity estimation error: {np.max(est_vel_error_mag)*1000:.2f} m/s")

fig6 = plt.figure(figsize=(16, 8))
fig6.suptitle('EKF: True vs Measured vs Estimated Position and Velocity (ECI)', fontsize=16)

# Create 6x1 grid for stacked plots
gs6 = fig6.add_gridspec(6, 1)

# Position XYZ combined (with estimates)
ax_pos6 = fig6.add_subplot(gs6[0, 0])
ax_pos6.plot(time_days, pos_x_arr, 'r-', linewidth=2.0, label='True X', alpha=0.4)
ax_pos6.plot(time_days, pos_y_arr, 'g-', linewidth=2.0, label='True Y', alpha=0.4)
ax_pos6.plot(time_days, pos_z_arr, 'b-', linewidth=2.0, label='True Z', alpha=0.4)
ax_pos6.plot(time_days, meas_pos_x, 'o', color='r', markersize=3, label='Meas X', alpha=0.4)
ax_pos6.plot(time_days, meas_pos_y, 'o', color='g', markersize=3, label='Meas Y', alpha=0.4)
ax_pos6.plot(time_days, meas_pos_z, 'o', color='b', markersize=3, label='Meas Z', alpha=0.4)
ax_pos6.plot(time_days, est_pos_x, 'r--', linewidth=2.0, label='EKF X', alpha=0.8)
ax_pos6.plot(time_days, est_pos_y, 'g--', linewidth=2.0, label='EKF Y', alpha=0.8)
ax_pos6.plot(time_days, est_pos_z, 'b--', linewidth=2.0, label='EKF Z', alpha=0.8)
if burn_1_end < len(time_days):
    ax_pos6.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                    color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_pos6.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                    color='purple', alpha=0.2)
ax_pos6.set_ylabel('Position (km)')
ax_pos6.grid(True, alpha=0.3)
ax_pos6.legend(loc='best', fontsize=8, ncol=3)
ax_pos6.set_title(f'Position: True vs Measured vs EKF Estimated (σ = {sigma_pos*1000:.1f} m)')

# Position error magnitude (measurement)
ax_pos_err6 = fig6.add_subplot(gs6[1, 0], sharex=ax_pos6)
ax_pos_err6.plot(time_days, pos_error_mag * 1000, 'k-', linewidth=1.5, label='Meas Error Mag', alpha=0.8)
ax_pos_err6.axhline(y=sigma_pos*1000, color='r', linestyle='--', linewidth=2, label=f'1σ = {sigma_pos*1000:.1f} m')
ax_pos_err6.axhline(y=3*sigma_pos*1000, color='orange', linestyle='--', linewidth=2, label=f'3σ = {3*sigma_pos*1000:.1f} m')
if burn_1_end < len(time_days):
    ax_pos_err6.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                        color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_pos_err6.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                        color='purple', alpha=0.2)
ax_pos_err6.set_ylabel('Meas Error (m)')
ax_pos_err6.grid(True, alpha=0.3)
ax_pos_err6.legend(loc='best', fontsize=10)
ax_pos_err6.set_title('Position Measurement Error Magnitude')

# Position estimation error magnitude
ax_est_err6 = fig6.add_subplot(gs6[2, 0], sharex=ax_pos6)
ax_est_err6.plot(time_days, est_pos_error_mag * 1000, 'b-', linewidth=1.5, label='EKF Error Mag', alpha=0.8)
ax_est_err6.axhline(y=sigma_pos*1000, color='r', linestyle='--', linewidth=2, label=f'1σ = {sigma_pos*1000:.1f} m')
ax_est_err6.axhline(y=3*sigma_pos*1000, color='orange', linestyle='--', linewidth=2, label=f'3σ = {3*sigma_pos*1000:.1f} m')
if burn_1_end < len(time_days):
    ax_est_err6.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                        color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_est_err6.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                        color='purple', alpha=0.2)
ax_est_err6.set_ylabel('EKF Error (m)')
ax_est_err6.grid(True, alpha=0.3)
ax_est_err6.legend(loc='best', fontsize=10)
ax_est_err6.set_title('Position EKF Estimation Error Magnitude (vs Truth)')

# Velocity XYZ combined (with EKF estimates)
ax_vel6 = fig6.add_subplot(gs6[3, 0], sharex=ax_pos6)
ax_vel6.plot(time_days, vel_x_arr, 'r-', linewidth=2.0, label='True X', alpha=0.4)
ax_vel6.plot(time_days, vel_y_arr, 'g-', linewidth=2.0, label='True Y', alpha=0.4)
ax_vel6.plot(time_days, vel_z_arr, 'b-', linewidth=2.0, label='True Z', alpha=0.4)
ax_vel6.plot(time_days, meas_vel_x, 'o', color='r', markersize=3, label='Meas X', alpha=0.4)
ax_vel6.plot(time_days, meas_vel_y, 'o', color='g', markersize=3, label='Meas Y', alpha=0.4)
ax_vel6.plot(time_days, meas_vel_z, 'o', color='b', markersize=3, label='Meas Z', alpha=0.4)
ax_vel6.plot(time_days, est_vel_x, 'r--', linewidth=2.0, label='EKF X', alpha=0.8)
ax_vel6.plot(time_days, est_vel_y, 'g--', linewidth=2.0, label='EKF Y', alpha=0.8)
ax_vel6.plot(time_days, est_vel_z, 'b--', linewidth=2.0, label='EKF Z', alpha=0.8)
if burn_1_end < len(time_days):
    ax_vel6.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                    color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_vel6.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                    color='purple', alpha=0.2)
ax_vel6.set_ylabel('Velocity (km/s)')
ax_vel6.grid(True, alpha=0.3)
ax_vel6.legend(loc='best', fontsize=8, ncol=3)
ax_vel6.set_title(f'Velocity: True vs Measured vs EKF Estimated (σ = {sigma_vel*1000:.1f} m/s)')

# Velocity measurement error magnitude
ax_vel_meas_err6 = fig6.add_subplot(gs6[4, 0], sharex=ax_pos6)
ax_vel_meas_err6.plot(time_days, vel_error_mag * 1000, 'k-', linewidth=1.5, label='Meas Error Mag', alpha=0.8)
ax_vel_meas_err6.axhline(y=sigma_vel*1000, color='r', linestyle='--', linewidth=2, label=f'1σ = {sigma_vel*1000:.1f} m/s')
ax_vel_meas_err6.axhline(y=3*sigma_vel*1000, color='orange', linestyle='--', linewidth=2, label=f'3σ = {3*sigma_vel*1000:.1f} m/s')
if burn_1_end < len(time_days):
    ax_vel_meas_err6.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                             color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_vel_meas_err6.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                             color='purple', alpha=0.2)
ax_vel_meas_err6.set_ylabel('Meas Error (m/s)')
ax_vel_meas_err6.grid(True, alpha=0.3)
ax_vel_meas_err6.legend(loc='best', fontsize=10)
ax_vel_meas_err6.set_title('Velocity Measurement Error Magnitude')

# Velocity estimation error magnitude
ax_vel_err6 = fig6.add_subplot(gs6[5, 0], sharex=ax_pos6)
ax_vel_err6.plot(time_days, est_vel_error_mag * 1000, 'b-', linewidth=1.5, label='EKF Error Mag', alpha=0.8)
ax_vel_err6.axhline(y=sigma_vel*1000, color='r', linestyle='--', linewidth=2, label=f'1σ = {sigma_vel*1000:.1f} m/s')
ax_vel_err6.axhline(y=3*sigma_vel*1000, color='orange', linestyle='--', linewidth=2, label=f'3σ = {3*sigma_vel*1000:.1f} m/s')
if burn_1_end < len(time_days):
    ax_vel_err6.axvspan(time_days[burn_1_start], time_days[burn_1_end], 
                        color='purple', alpha=0.2, label='Thrust On')
if burn_2_end < len(time_days):
    ax_vel_err6.axvspan(time_days[burn_2_start], time_days[burn_2_end], 
                        color='purple', alpha=0.2)
ax_vel_err6.set_ylabel('EKF Vel Error (m/s)')
ax_vel_err6.set_xlabel('Time (days)')
ax_vel_err6.grid(True, alpha=0.3)
ax_vel_err6.legend(loc='best', fontsize=10)
ax_vel_err6.set_title('Velocity EKF Estimation Error Magnitude (vs Truth)')

plt.tight_layout()
output_file6 = Path.cwd() / 'coast_burn_coast_estimation.png'
fig6.savefig(output_file6, dpi=150)

plt.show()

print(f"Propagation complete. Plotted {len(time_hours)} data points.")
print(f"Plot saved to {output_file}")
print(f"Delta-V plot saved to {output_file2}")
print(f"Position plot saved to {output_file3}")
print(f"Jerk FD plot saved to {output_file4}")
print(f"Measurements plot saved to {output_file5}")
print(f"Estimation plot saved to {output_file6}")
print(f"Final mass: {mass[-1]:.2f} kg (propellant used: {mass_initial - mass[-1]:.2f} kg)")
print(f"Peak thrust: {np.max(thrust):.2f} N (should be constant at {constant_thrust:.2f} N during burns)")
print(f"Total delta-V: {np.sum(delta_v_mag):.2f} m/s")
