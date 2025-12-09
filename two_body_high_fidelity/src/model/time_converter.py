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
  et_float = float(spice.str2et(utc_str))
  return et_float


def et_to_utc(
  et                : float,
  precision_seconds : int = 0,
) -> datetime:
  """
  Convert Ephemeris Time (ET) to UTC datetime object.
  
  Input:
  ------
    et : float
      Ephemeris Time (ET) in seconds past J2000.
    precision_seconds : int
      Number of decimal places for the seconds component (0-6).
  
  Output:
  -------
    utc_dt : datetime
      UTC time as a datetime object.
  """
  # Ensure precision doesn't exceed 6 for datetime compatibility
  if precision_seconds > 6:
    precision_seconds = 6

  # Convert ET to UTC string 
  # 'ISOC' specifies the output format as ISO Calendar (YYYY-MM-DDThh:mm:ss.sss)
  utc_str = str(spice.et2utc(et, 'ISOC', precision_seconds))

  # Convert and return UTC string to datetime object
  return datetime.fromisoformat(utc_str)