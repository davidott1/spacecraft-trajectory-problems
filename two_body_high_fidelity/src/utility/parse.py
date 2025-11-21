from datetime import datetime

def parse_time(
  time_str : str,
) -> datetime:
  """
  Parse time string in multiple formats (all assumed UTC)
  
  Input:
  ------
  time_str : str
    Time string in various formats
  
  Output:
  -------
  datetime : Parsed datetime object
  
  Supported formats:
  - YYYY-MM-DD
  - YYYY-MM-DD HH:MM
  - YYYY-MM-DD HH:MM:SS
  - YYYY-MM-DD HH:MM:SS.ssssss
  - YYYY-MM-DDTHH:MM
  - YYYY-MM-DDTHH:MM:SS
  - YYYY-MM-DDTHH:MM:SS.ssssss
  - YYYY-MM-DDTHH:MM:SSZ (with trailing Z)
  """
  # Remove trailing 'Z' if present (ISO 8601 UTC designator)
  time_str = time_str.rstrip('Z')
  
  # Try different formats
  formats = [
    '%Y-%m-%dT%H:%M:%S.%f', # ISO 8601 with microseconds
    '%Y-%m-%dT%H:%M:%S',    # ISO 8601 with seconds: 2025-10-01T00:00:00
    '%Y-%m-%dT%H:%M',       # ISO 8601 without seconds: 2025-10-01T00:00
    '%Y-%m-%d %H:%M:%S.%f', # Space-separated with microseconds
    '%Y-%m-%d %H:%M:%S',    # Space-separated with seconds: 2025-10-01 00:00:00
    '%Y-%m-%d %H:%M',       # Space-separated without seconds: 2025-10-01 00:00
    '%Y-%m-%d',             # Date only: 2025-10-01
  ]
  
  for fmt in formats:
    try:
      return datetime.strptime(time_str, fmt)
    except ValueError:
      continue
  
  raise ValueError(f"Cannot parse time string: {time_str}")
