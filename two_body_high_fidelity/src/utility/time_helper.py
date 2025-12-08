"""
Time Utilities
==============

Utility functions for time parsing and formatting.
"""
from datetime import datetime


def format_time_offset(
  seconds : float,
) -> str:
  """
  Format a time offset in seconds as a human-readable string.

  Examples:
    12345.678 -> "+0d 03h 25m 45.678s"
   -98765.432 -> "-1d 03h 26m 05.432s"
  
  Input:
  ------
    seconds : float
      Time offset in seconds (can be positive or negative).
  
  Output:
  -------
    str
      Formatted string like "+47d 21h 30m 20.357s" or "-1d 02h 15m 00.000s"
  """
  sign    = '+' if seconds >= 0 else '-'
  abs_sec = abs(seconds)
  
  days    = int(abs_sec // 86400)
  hours   = int((abs_sec % 86400) // 3600)
  minutes = int((abs_sec % 3600) // 60)
  secs    = abs_sec % 60
  
  return f"{sign}{days}d {hours:02d}h {minutes:02d}m {secs:06.3f}s"


def parse_time(
  time_str : str,
) -> datetime:
  """
  Parse a time string into a datetime object.
  
  Accepted formats include:
  - ISO 8601 with 'T' separator: "2025-10-01T00:00:00"
  - ISO 8601 with 'Z' suffix: "2025-10-01T00:00:00Z"
  - Space-separated: "2025-10-01 00:00:00"
  - With microseconds: "2025-10-01 00:00:00.123456"
  
  The 'Z' suffix (UTC indicator) is stripped before parsing for 
  Python < 3.11 compatibility.
  
  Input:
  ------
    time_str : str
      Time string to parse.
      
  Output:
  -------
    datetime
      Parsed datetime object.
  """
  # Handle 'Z' suffix for Python < 3.11 compatibility
  if time_str.endswith('Z'):
    time_str = time_str[:-1]

  try:
    return datetime.fromisoformat(time_str)
  except ValueError:
    # Fallback for other formats if needed
    formats = [
      "%Y-%m-%d %H:%M:%S",
      "%Y-%m-%d %H:%M:%S.%f",
    ]
    for fmt in formats:
      try:
        return datetime.strptime(time_str, fmt)
      except ValueError:
        continue
    raise ValueError(f"Cannot parse time string: {time_str}")
