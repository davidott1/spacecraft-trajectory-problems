import spiceypy as spice
from datetime import datetime


def utc_to_et(
  utc_dt : datetime,
) -> float:
  """
  Convert a UTC datetime object to Ephemeris Time (ET) (seconds past J2000).
  
  Input:
  ------
    utc_dt : datetime
      The UTC datetime to convert.
  
  Output:
  -------
    et_float : float
      The corresponding Ephemeris Time (ET) in seconds past J2000.
  """
  utc_str  = utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
  et_float = spice.str2et(utc_str)
  return et_float