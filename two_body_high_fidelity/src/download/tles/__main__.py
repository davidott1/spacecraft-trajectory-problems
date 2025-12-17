"""
Download TLE data for any satellite by NORAD ID

Usage:
  python -m src.download.tles <norad_id> <start_time> <end_time>
  
Examples:
  python -m src.download.tles 25544 "2025-10-01" "2025-10-08"
  python -m src.download.tles 39166 "2025-10-01T00:00:00Z" "2025-10-02T00:00:00Z"
"""

import sys
from pathlib import Path
from datetime import datetime

# Import shared functionality from ephems_and_tles module
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.download.ephems_and_tles.__main__ import (
    download_ephems_and_tles,
    parse_time
)

def parse_command_line_args():
    """Parse command line arguments for TLE download only."""
    if len(sys.argv) < 4:
        print("Usage: python -m src.download.tles <norad_id> <start_time> <end_time>")
        print("\nExamples:")
        print('  python -m src.download.tles 25544 "2025-10-01" "2025-10-08"')
        print('  python -m src.download.tles 39166 "2025-10-01T00:00:00Z" "2025-10-02T00:00:00Z"')
        print("\nTime format (UTC assumed):")
        print("  YYYY-MM-DD")
        print("  YYYY-MM-DD HH:MM")
        print("  YYYY-MM-DD HH:MM:SS")
        print("  YYYY-MM-DDTHH:MM")
        print("  YYYY-MM-DDTHH:MM:SS")
        print("  YYYY-MM-DDTHH:MM:SSZ")
        sys.exit(1)
    
    norad_id = int(sys.argv[1])
    start_str = sys.argv[2]
    end_str = sys.argv[3]
    
    try:
        start_time = parse_time(start_str)
        end_time = parse_time(end_str)
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)
    
    if end_time <= start_time:
        print("Error: End time must be after start time")
        sys.exit(1)
    
    return norad_id, start_time, end_time

if __name__ == "__main__":
    norad_id, start_time, end_time = parse_command_line_args()
    # TLE downloads don't need a step parameter
    download_ephems_and_tles(norad_id, start_time, end_time, step='1h', download_ephem=False, download_tle=True)
