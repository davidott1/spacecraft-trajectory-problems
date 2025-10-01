from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import numpy as np

# Example TLE data for ISS (International Space Station)
# You can get updated TLEs from https://celestrak.org/
line1 = '1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005'
line2 = '2 25544  51.6400 208.9163 0006317  69.9862  25.2906 15.54225995 12345'

# Initialize the satellite object
satellite = Satrec.twoline2rv(line1, line2)

# Define the epoch time
year = 2024
month = 1
day = 1
hour = 12
minute = 0
second = 0.0

# Convert to Julian date
jd, fr = jday(year, month, day, hour, minute, second)

print(f"Propagating spacecraft trajectory using SGP4")
print(f"TLE Epoch: {satellite.epochyr + 2000}-{satellite.epochdays:.2f}")
print(f"\nPropagation results:\n")
print(f"{'Time':>13}  {'Pos-X':>13}  {'Pos-Y':>13}  {'Pos-Z':>13}  {'Vel-X':>13}  {'Vel-Y':>13}  {'Vel-Z':>13}")
print(f"{'min':>13}  {'km':>13}  {'km':>13}  {'km':>13}  {'km/s':>13}  {'km/s':>13}  {'km/s':>13}")
print(("-" * 13 + "  ") * 7)

# Propagate at different times (0 to 90 minutes in 15-minute intervals)
for minutes in range(0, 91, 15):
    # Calculate time since epoch in minutes
    tsince = minutes
    
    # Propagate
    error_code, position, velocity = satellite.sgp4(jd, fr + minutes / 1440.0)
    
    if error_code == 0:
        x, y, z = position
        vx, vy, vz = velocity
        print(f"{minutes:>13}  {x:>13.6e}  {y:>13.6e}  {z:>13.6e}  {vx:>13.6e}  {vy:>13.6e}  {vz:>13.6e}")
    else:
        print(f"{minutes:>13} Error code: {error_code}")

# Example: Propagate over a specific time range
print("\n\nPropagating over 24 hours:")
start_time = datetime(2024, 1, 1, 12, 0, 0)

for hours in [0, 6, 12, 18, 24]:
    current_time = start_time + timedelta(hours=hours)
    jd, fr = jday(current_time.year, current_time.month, current_time.day,
                  current_time.hour, current_time.minute, current_time.second)
    
    error_code, position, velocity = satellite.sgp4(jd, fr)
    
    if error_code == 0:
        print(f"\nTime: {current_time}")
        print(f"  Position (km): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        print(f"  Velocity (km/s): [{velocity[0]:.6f}, {velocity[1]:.6f}, {velocity[2]:.6f}]")
        print(f"  Altitude (km): {np.linalg.norm(position) - 6371:.3f}")