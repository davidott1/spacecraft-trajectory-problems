#!/usr/bin/env python3
"""
Example: Query JPL Horizons for GOES-16 satellite ephemeris in J2000 frame
GOES-16 NORAD ID: 41866
"""

from astroquery.jplhorizons import Horizons
from astropy.time import Time
import numpy as np

# GOES-16 identifier
# Note: JPL Horizons uses different identifiers than NORAD IDs
# GOES-16 in Horizons is identified as '-21416' or 'GOES-16'
goes16_id = 'GOES-16'  # or use '-21416'

# Define time range for ephemeris
# You can use a single epoch or a range
start_time = '2024-01-01 00:00'
stop_time = '2024-01-01 01:00'
step = '10m'  # 10 minute intervals

# Alternative: Single epoch
# epochs = Time('2024-01-01 00:00:00').jd

# For a time range
epochs = {'start': start_time,
          'stop': stop_time,
          'step': step}

# Query for vectors (position and velocity) from Earth's center
# location='500@399' means Earth center (399 is Earth's ID, 500@ means body center)
obj = Horizons(id=goes16_id, 
               location='500@399',  # Earth center as origin
               epochs=epochs)

# Get vector ephemeris in J2000 equatorial frame
# refplane='earth' gives equatorial, refplane='ecliptic' would give ecliptic
vectors = obj.vectors(refplane='earth',    # Earth equatorial plane
                      refsystem='J2000',    # J2000.0 reference system
                      aberrations='none',   # No aberration corrections
                      delta_T=True)         # Include Delta-T values

print("GOES-16 Ephemeris in J2000 Equatorial Frame")
print("=" * 60)
print(f"Reference Frame: {vectors.meta.get('reference_system', 'Not specified')}")
print(f"Reference Plane: {vectors.meta.get('reference_plane', 'Not specified')}")
print()

# Display the vectors
print("Position and Velocity Vectors:")
print("-" * 60)
for i in range(len(vectors)):
    print(f"Time: {vectors['datetime_str'][i]}")
    print(f"  Position (km): X={vectors['x'][i]:.3f}, "
          f"Y={vectors['y'][i]:.3f}, Z={vectors['z'][i]:.3f}")
    print(f"  Velocity (km/s): VX={vectors['vx'][i]:.6f}, "
          f"VY={vectors['vy'][i]:.6f}, VZ={vectors['vz'][i]:.6f}")
    print(f"  Range from Earth center: {vectors['range'][i]:.3f} km")
    print()

# Alternative: Get orbital elements instead of vectors
print("\nAlternative: Orbital Elements")
print("=" * 60)
elements = obj.elements(refsystem='J2000',
                        refplane='earth')

if len(elements) > 0:
    print(f"Semimajor axis: {elements['a'][0]:.3f} km")
    print(f"Eccentricity: {elements['e'][0]:.6f}")
    print(f"Inclination: {elements['incl'][0]:.3f} degrees")
    print(f"RAAN: {elements['Omega'][0]:.3f} degrees")
    print(f"Arg of periapsis: {elements['omega'][0]:.3f} degrees")
    print(f"Mean anomaly: {elements['M'][0]:.3f} degrees")

# Example: Convert to astropy SkyCoord object for further manipulation
from astropy.coordinates import SkyCoord
from astropy import units as u

# Create SkyCoord object from the first position
if len(vectors) > 0:
    coord = SkyCoord(x=vectors['x'][0] * u.km,
                     y=vectors['y'][0] * u.km,
                     z=vectors['z'][0] * u.km,
                     v_x=vectors['vx'][0] * u.km/u.s,
                     v_y=vectors['vy'][0] * u.km/u.s,
                     v_z=vectors['vz'][0] * u.km/u.s,
                     frame='icrs',  # ICRS is essentially J2000
                     obstime=Time(vectors['datetime_jd'][0], format='jd'))
    
    print("\nAs Astropy SkyCoord:")
    print("=" * 60)
    print(f"RA: {coord.ra.deg:.6f} degrees")
    print(f"Dec: {coord.dec.deg:.6f} degrees")
    print(f"Distance: {coord.distance.km:.3f} km")

# Note about alternative queries:
print("\n" + "=" * 60)
print("Notes:")
print("- JPL Horizons may not have all satellites. For GOES-16, try:")
print("  - id='GOES-16' or id='-21416'")
print("- If satellite not found in Horizons, consider using:")
print("  - SGP4 propagation with TLE data from Space-Track or Celestrak")
print("  - NASA NAIF SPICE kernels if available")
print("- Reference frame options:")
print("  - refsystem='J2000' for J2000.0 epoch")
print("  - refplane='earth' for equatorial, 'ecliptic' for ecliptic")