#!/usr/bin/env python3
"""
Example: Query JPL Horizons for GOES-16 satellite ephemeris in J2000 frame
GOES-16 NORAD ID: 41866
"""

from astroquery.jplhorizons import Horizons
from astropy.time import Time
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential

# GOES-16 identifier
# Note: JPL Horizons uses different identifiers than NORAD IDs
# GOES-16 in Horizons is identified as '-21416' or 'GOES-16'
norad_id = 41866  # GOES-16
sat_id = -(100000 + norad_id)

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
obj = Horizons(id=sat_id, 
               location='500@399',  # Earth center as origin
               epochs=epochs)

# Get vector ephemeris in J2000 equatorial frame
# refplane='earth' gives equatorial, refplane='ecliptic' would give ecliptic
# delta_T is the difference TT-UT (Terrestrial Time - Universal Time)
# Used to account for Earth's irregular rotation when converting between time scales
vectors = obj.vectors(refplane='earth',    # Earth equatorial plane
                      delta_T=True)        # Include Delta-T values in output table
# vectors = obj.vectors(refplane='earth', aberrations='geometric')

# Convert from AU and AU/day to km and km/s
AU_TO_KM = 149597870.7  # 1 AU in kilometers
SECONDS_PER_DAY = 86400.0

# Convert positions from AU to km
vectors['x'] = vectors['x'] * AU_TO_KM
vectors['y'] = vectors['y'] * AU_TO_KM
vectors['z'] = vectors['z'] * AU_TO_KM
vectors['range'] = vectors['range'] * AU_TO_KM

# Convert velocities from AU/day to km/s
vectors['vx'] = vectors['vx'] * AU_TO_KM / SECONDS_PER_DAY
vectors['vy'] = vectors['vy'] * AU_TO_KM / SECONDS_PER_DAY
vectors['vz'] = vectors['vz'] * AU_TO_KM / SECONDS_PER_DAY
vectors['range_rate'] = vectors['range_rate'] * AU_TO_KM / SECONDS_PER_DAY

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
    # Convert semimajor axis from AU to km
    a_km = elements['a'][0] * AU_TO_KM if elements['a'].unit == u.AU else elements['a'][0]
    print(f"SMA  : {a_km:.3f} km")
    print(f"ECC  : {elements['e'][0]:.6f}")
    print(f"INC  : {elements['incl'][0]:.3f} degrees")
    print(f"RAAN : {elements['Omega'][0]:.3f} degrees")
    print(f"ARGP : {elements['w'][0]:.3f} degrees")
    print(f"MA   : {elements['M'][0]:.3f} degrees")

# Example: Convert to astropy SkyCoord object for further manipulation
# Create SkyCoord object from the first position
if len(vectors) > 0:
    # Create Cartesian representation with position and velocity
    cart_rep = CartesianRepresentation(
        x=vectors['x'][0] * u.km,
        y=vectors['y'][0] * u.km,
        z=vectors['z'][0] * u.km,
        differentials=CartesianDifferential(
            d_x=vectors['vx'][0] * u.km/u.s,
            d_y=vectors['vy'][0] * u.km/u.s,
            d_z=vectors['vz'][0] * u.km/u.s
        )
    )
    
    coord = SkyCoord(cart_rep,
                     frame='icrs',  # ICRS is essentially J2000
                     obstime=Time(vectors['datetime_jd'][0], format='jd'))
    
    print("\nAs Astropy SkyCoord:")
    print("=" * 60)
    print(f"RA: {coord.ra.deg:.6f} degrees")
    print(f"Dec: {coord.dec.deg:.6f} degrees")
    print(f"Distance: {coord.distance.km:.3f} km")
    print()

