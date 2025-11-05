#!/usr/bin/env python3
"""
Practical example for getting GOES-16 ephemeris in J2000 frame
Includes error handling since not all satellites are in JPL Horizons
"""

from astroquery.jplhorizons import Horizons
from astropy.time import Time
import numpy as np

def get_satellite_ephemeris(sat_id, time_str, location='500@399'):
    """
    Get satellite position and velocity in J2000 equatorial frame.
    
    Parameters:
    -----------
    sat_id : str
        Satellite identifier for Horizons (e.g., 'GOES-16', '-21416')
    time_str : str
        Time string (e.g., '2024-01-01 00:00:00')
    location : str
        Observer location (default: '500@399' for Earth center)
    
    Returns:
    --------
    dict with position, velocity, and time information
    """
    
    try:
        # Convert time to Julian Date
        epoch = Time(time_str).jd
        
        # Create Horizons query object
        obj = Horizons(id=sat_id,
                      location=location,
                      epochs=epoch)
        
        # Query for vectors in J2000 equatorial frame
        vectors = obj.vectors(
            refplane='earth',      # Equatorial plane
            refsystem='J2000',     # J2000.0 reference frame
            aberrations='none',    # No light-time corrections
            delta_T=True          # Include Delta-T
        )
        
        if len(vectors) == 0:
            raise ValueError("No data returned from Horizons")
        
        # Extract results
        result = {
            'time': time_str,
            'jd': vectors['datetime_jd'][0],
            'position_km': {
                'x': vectors['x'][0],
                'y': vectors['y'][0],
                'z': vectors['z'][0]
            },
            'velocity_km_s': {
                'vx': vectors['vx'][0],
                'vy': vectors['vy'][0],
                'vz': vectors['vz'][0]
            },
            'range_km': vectors['range'][0],
            'range_rate_km_s': vectors['range_rate'][0]
        }
        
        return result
        
    except Exception as e:
        print(f"Error querying Horizons: {e}")
        print("\nTroubleshooting:")
        print("1. GOES-16 might not be in JPL Horizons database")
        print("2. Try alternative IDs: 'GOES-16', '-21416', or 'GOES 16'")
        print("3. For real-time satellite tracking, consider using TLE data instead")
        return None

# Example usage
if __name__ == "__main__":
    
    # Try different possible IDs for GOES-16
    possible_ids = ['GOES-16', '-21416', 'GOES 16', '41866']
    
    time_to_query = '2024-01-01 00:00:00'
    
    print("Attempting to query GOES-16 ephemeris from JPL Horizons")
    print("=" * 60)
    
    for sat_id in possible_ids:
        print(f"\nTrying ID: '{sat_id}'")
        result = get_satellite_ephemeris(sat_id, time_to_query)
        
        if result:
            print(f"✓ Success with ID: '{sat_id}'")
            print(f"\nGOES-16 State Vectors in J2000 Frame at {result['time']}:")
            print("-" * 60)
            print(f"Position (J2000 equatorial):")
            print(f"  X = {result['position_km']['x']:12.3f} km")
            print(f"  Y = {result['position_km']['y']:12.3f} km")
            print(f"  Z = {result['position_km']['z']:12.3f} km")
            print(f"\nVelocity (J2000 equatorial):")
            print(f"  VX = {result['velocity_km_s']['vx']:9.6f} km/s")
            print(f"  VY = {result['velocity_km_s']['vy']:9.6f} km/s")
            print(f"  VZ = {result['velocity_km_s']['vz']:9.6f} km/s")
            print(f"\nRange from Earth center: {result['range_km']:12.3f} km")
            print(f"Range rate: {result['range_rate_km_s']:9.6f} km/s")
            break
    else:
        print("\n" + "=" * 60)
        print("GOES-16 not found in JPL Horizons database.")
        print("\nAlternative approaches for GOES-16 ephemeris:")
        print("1. Use TLE data from Space-Track.org or Celestrak.com")
        print("2. Use SGP4 propagator with the skyfield or sgp4 Python packages")
        print("3. Query NOAA's satellite data services directly")
        print("\nExample using TLE/SGP4 approach:")
        print("-" * 40)
        print("""
from skyfield.api import load, EarthSatellite
from skyfield.api import wgs84

# Load TLE data (example - get current from Celestrak)
line1 = '1 41866U 16071A   24001.50000000  .00000000  00000-0  00000-0 0  9999'
line2 = '2 41866   0.0400  75.1234 0001234  90.0000 270.0000  1.00270000 12345'

ts = load.timescale()
satellite = EarthSatellite(line1, line2, 'GOES-16', ts)
t = ts.utc(2024, 1, 1, 0, 0, 0)

# Get position in GCRS (essentially J2000)
geocentric = satellite.at(t)
position = geocentric.position.km
velocity = geocentric.velocity.km_per_s
""")

    # Show how to do a batch query for multiple times
    print("\n" + "=" * 60)
    print("Example: Batch query for multiple epochs")
    print("-" * 40)
    print("""
# For multiple epochs, use a dictionary:
epochs = {
    'start': '2024-01-01 00:00',
    'stop': '2024-01-01 06:00',
    'step': '1h'  # 1 hour steps
}

obj = Horizons(id='GOES-16', location='500@399', epochs=epochs)
vectors = obj.vectors(refplane='earth', refsystem='J2000')

# This would give you ephemeris at hourly intervals
""")
