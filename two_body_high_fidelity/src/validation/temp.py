from astroquery.jplhorizons import Horizons
from astropy.time import Time

# The JPL Horizons ID for NORAD 25544 (ISS) is -125544
# 37846
norad_id = 37846
sat_id   = -(100000 + norad_id)

# Location '@399' is the Earth Geocenter
obj = Horizons(id=sat_id,
               location='@399',  # Coordinate origin: Earth Geocenter
               epochs={'start' : '2025-11-03 12:00',
                       'stop'  : '2025-11-03 18:00',
                       'step'  : '1h'})

# Get state vectors (x, y, z, vx, vy, vz)
# This will return data in km and km/s, not AU
vec = obj.vectors()

print(vec)