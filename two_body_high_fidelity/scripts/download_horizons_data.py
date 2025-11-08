"""
Download ephemeris and TLE data for any satellite by NORAD ID

Usage:
    python download_horizons_data.py <norad_id> <start_time> <end_time> [step]
    
Example:
    python download_horizons_data.py 25544 "2025-10-01" "2025-10-08"
    python download_horizons_data.py 25544 "2025-10-01T00:00:00Z" "2025-10-08T00:00:00Z" 1m
    python download_horizons_data.py 25544 "2025-10-01 00:00" "2025-10-08 00:00"
"""

from astroquery.jplhorizons import Horizons
from pathlib import Path
from datetime import datetime, timedelta
import requests
from astropy.time import Time
import sys

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Output directory
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'ephems'

def get_satellite_name(norad_id):
    """Get satellite name from NORAD ID or return generic name"""
    known_satellites = {
        25544: 'iss',
        25994: 'terra',
        27424: 'aqua',
        26407: 'gps_iir_5',
        38833: 'gps_iif_2',
        39166: 'gps_iif_3',
        41866: 'goes16',
        43226: 'goes17',
        51850: 'goes18',
    }
    return known_satellites.get(norad_id, f'sat{norad_id}')

def download_tle_for_satellite(norad_id, output_file=None):
    """
    Download TLE for a satellite from Celestrak
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog number
    output_file : str or Path
        Output file path
    
    Returns:
    --------
    tuple : (line1, line2, epoch_jd) TLE lines and epoch
    """
    # Try multiple sources
    sources = [
        f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE",
        f"https://celestrak.org/satcat/tle.php?CATNR={norad_id}",
    ]
    
    for url in sources:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                if len(lines) >= 2:
                    # Find the two TLE lines (starting with '1 ' and '2 ')
                    tle_lines = [l for l in lines if l.startswith('1 ') or l.startswith('2 ')]
                    if len(tle_lines) >= 2:
                        line1 = tle_lines[0]
                        line2 = tle_lines[1]
                        
                        # Extract TLE epoch
                        tle_epoch_str = line1[18:32]
                        tle_year = int(tle_epoch_str[0:2])
                        tle_year = 2000 + tle_year if tle_year < 57 else 1900 + tle_year
                        tle_day_of_year = float(tle_epoch_str[2:])
                        
                        tle_epoch_time = Time(f"{tle_year}-01-01", format='iso') + (tle_day_of_year - 1)
                        epoch_jd = tle_epoch_time.jd
                        
                        print(f"Downloaded TLE for NORAD {norad_id}")
                        print(f"  Epoch: {tle_epoch_time.iso} (JD {epoch_jd})")
                        print(f"  Line 1: {line1}")
                        print(f"  Line 2: {line2}")
                        
                        if output_file:
                            with open(output_file, 'w') as f:
                                f.write(f"{line1}\n{line2}\n")
                            print(f"  Saved TLE to: {output_file}")
                        
                        return line1, line2, epoch_jd
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    
    raise RuntimeError(f"Could not download TLE for NORAD {norad_id}")

def download_historical_tles(norad_id, start_time, end_time, output_file=None):
    """
    Download historical TLEs from Space-Track.org for a given time range.
    
    NOTE: Requires Space-Track.org credentials. Set environment variables:
          SPACETRACK_USER and SPACETRACK_PASSWORD
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog number
    start_time : datetime
        Start of time range
    end_time : datetime  
        End of time range
    output_file : str or Path, optional
        Output file path
    
    Returns:
    --------
    list of dict : List of TLE data with keys 'line1', 'line2', 'epoch_jd'
    """
    import os
    
    username = os.getenv('SPACETRACK_USER')
    password = os.getenv('SPACETRACK_PASSWORD')
    
    if not username or not password:
        print("WARNING: Space-Track credentials not found!")
        print("Set SPACETRACK_USER and SPACETRACK_PASSWORD environment variables")
        print("You can register for free at: https://www.space-track.org/auth/createAccount")
        print("\nFalling back to Celestrak for single TLE...")
        
        # Fall back to single TLE from Celestrak
        line1, line2, epoch_jd = download_tle_for_satellite(norad_id)
        return [{'line1': line1, 'line2': line2, 'epoch_jd': epoch_jd}]
    
    # Space-Track API
    base_url = "https://www.space-track.org"
    login_url = f"{base_url}/ajaxauth/login"
    query_url = f"{base_url}/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{norad_id}/orderby/EPOCH asc/EPOCH/{start_time.strftime('%Y-%m-%d')}--{end_time.strftime('%Y-%m-%d')}/format/3le"
    
    session = requests.Session()
    
    try:
        # Login
        print(f"Logging into Space-Track.org...")
        response = session.post(login_url, data={'identity': username, 'password': password})
        if response.status_code != 200:
            raise RuntimeError(f"Login failed: {response.status_code}")
        
        # Query TLEs
        print(f"Querying TLEs for NORAD {norad_id} from {start_time} to {end_time}...")
        response = session.get(query_url)
        if response.status_code != 200:
            raise RuntimeError(f"Query failed: {response.status_code}")
        
        # Parse TLEs
        lines = response.text.strip().split('\n')
        tle_data = []
        
        for i in range(0, len(lines)-1, 2):
            if lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
                line1 = lines[i].strip()
                line2 = lines[i+1].strip()
                
                # Extract epoch
                tle_epoch_str = line1[18:32]
                tle_year = int(tle_epoch_str[0:2])
                tle_year = 2000 + tle_year if tle_year < 57 else 1900 + tle_year
                tle_day_of_year = float(tle_epoch_str[2:])
                
                tle_epoch_time = Time(f"{tle_year}-01-01", format='iso') + (tle_day_of_year - 1)
                
                tle_data.append({
                    'line1': line1,
                    'line2': line2,
                    'epoch_jd': tle_epoch_time.jd,
                    'epoch_iso': tle_epoch_time.iso
                })
        
        print(f"Retrieved {len(tle_data)} TLEs")
        
        if output_file:
            with open(output_file, 'w') as f:
                for tle in tle_data:
                    f.write(f"{tle['line1']}\n{tle['line2']}\n")
            print(f"Saved TLEs to: {output_file}")
        
        return tle_data
        
    except Exception as e:
        print(f"Error downloading from Space-Track: {e}")
        print("Falling back to single TLE from Celestrak...")
        line1, line2, epoch_jd = download_tle_for_satellite(norad_id)
        return [{'line1': line1, 'line2': line2, 'epoch_jd': epoch_jd}]
    finally:
        session.close()

def download_horizons_ephemeris(norad_id, start_time, end_time, step='1h', output_file=None):
    """
    Download satellite ephemeris from HORIZONS
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog number
    start_time : datetime
        Start time for ephemeris
    end_time : datetime
        End time for ephemeris
    step : str
        Time step (default '1h')
    output_file : str or Path
        Output file path
    
    Returns:
    --------
    astropy.table.Table with ephemeris data
    """
    print(f"Downloading HORIZONS ephemeris for NORAD {norad_id}")
    print(f"Time range: {start_time} to {end_time}")
    print(f"Time step: {step}")
    
    # Query HORIZONS using NORAD catalog number (negative ID format)
    sat_id = -(100000 + norad_id)
    obj = Horizons(
        id=f"{sat_id}",
        location='@399',  # Earth center (geocentric)
        epochs={'start' : start_time.strftime('%Y-%m-%d %H:%M'),
                'stop'  : end_time.strftime('%Y-%m-%d %H:%M'),
                'step'  : step}
    )
    
    # Get vectors in ICRF/J2000 equatorial frame
    vectors = obj.vectors(refplane='earth', cache=False)
    
    print(f"Retrieved {len(vectors)} data points")
    print(f"Reference frame: ICRF/J2000 Earth equatorial (refplane='earth')")
    
    if output_file:
        vectors.write(output_file, format='csv', overwrite=True)
        print(f"Saved ephemeris to: {output_file}")
        file_size = Path(output_file).stat().st_size / (1024*1024)
        print(f"File size: {file_size:.2f} MB")
    
    return vectors

def download_satellite_data(norad_id, start_time, end_time, step='1h'):
    """
    Download both HORIZONS ephemeris and TLE history for a satellite
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog number
    start_time : datetime
        Start time
    end_time : datetime
        End time
    step : str
        Time step for ephemeris (default: '1h')
    
    Returns:
    --------
    dict with file paths and data
    """
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    sat_name = get_satellite_name(norad_id)
    timestamp_str = start_time.strftime('%Y%m%d')
    
    print("="*80)
    print(f"Downloading data for {sat_name.upper()} (NORAD {norad_id})")
    print(f"Time range: {start_time} to {end_time}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    # Download HORIZONS ephemeris
    print("\n" + "="*80)
    print("DOWNLOADING HORIZONS EPHEMERIS")
    print("="*80)
    horizons_file = OUTPUT_DIR / f'horizons_{sat_name}_{step}_{timestamp_str}_utc.csv'
    horizons_data = download_horizons_ephemeris(
        norad_id=norad_id,
        start_time=start_time,
        end_time=end_time,
        step=step,
        output_file=horizons_file
    )
    
    # Download TLE history
    print("\n" + "="*80)
    print("DOWNLOADING TLE HISTORY")
    print("="*80)
    tle_history_file = OUTPUT_DIR / f'tle_history_{sat_name}.txt'
    tle_history = download_historical_tles(
        norad_id=norad_id,
        start_time=start_time,
        end_time=end_time,
        output_file=tle_history_file
    )
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"HORIZONS file:    {horizons_file}")
    print(f"TLE history file: {tle_history_file}")
    print(f"Data points:      {len(horizons_data)}")
    print(f"Number of TLEs:   {len(tle_history)}")
    
    return {
        'horizons_file': horizons_file,
        'tle_history_file': tle_history_file,
        'horizons_data': horizons_data,
        'tle_history': tle_history,
        'sat_name': sat_name,
        'norad_id': norad_id,
    }

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python download_horizons_data.py <norad_id> <start_time> <end_time> [step]")
        print("\nExample:")
        print('  python download_horizons_data.py 25544 "2025-10-01" "2025-10-08"')
        print('  python download_horizons_data.py 25544 "2025-10-01T00:00:00Z" "2025-10-08T00:00:00Z" 1m')
        print('  python download_horizons_data.py 25544 "2025-10-01 00:00" "2025-10-08 00:00"')
        print("\nTime format (UTC assumed):")
        print("  YYYY-MM-DD")
        print("  YYYY-MM-DD HH:MM")
        print("  YYYY-MM-DD HH:MM:SS")
        print("  YYYY-MM-DDTHH:MM")
        print("  YYYY-MM-DDTHH:MM:SS")
        print("  YYYY-MM-DDTHH:MM:SSZ")
        print("\nCommon satellites:")
        print("  25544 - ISS")
        print("  41866 - GOES-16")
        print("  43226 - GOES-17")
        print("  51850 - GOES-18")
        sys.exit(1)
    
    norad_id = int(sys.argv[1])
    start_str = sys.argv[2]
    end_str = sys.argv[3]
    step = sys.argv[4] if len(sys.argv) > 4 else '1h'
    
    # Parse start and end times
    def parse_time(time_str):
        """Parse time string in multiple formats (all assumed UTC)"""
        # Remove trailing 'Z' if present (ISO 8601 UTC designator)
        time_str = time_str.rstrip('Z')
        
        # Try different formats
        formats = [
            '%Y-%m-%dT%H:%M:%S',  # ISO 8601 with seconds: 2025-10-01T00:00:00
            '%Y-%m-%dT%H:%M',     # ISO 8601 without seconds: 2025-10-01T00:00
            '%Y-%m-%d %H:%M:%S',  # Space-separated with seconds: 2025-10-01 00:00:00
            '%Y-%m-%d %H:%M',     # Space-separated without seconds: 2025-10-01 00:00
            '%Y-%m-%d',           # Date only: 2025-10-01
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Cannot parse time string: {time_str}")
    
    try:
        start_time = parse_time(start_str)
        end_time = parse_time(end_str)
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print('Supported formats (UTC assumed):')
        print('  "YYYY-MM-DD"')
        print('  "YYYY-MM-DD HH:MM"')
        print('  "YYYY-MM-DD HH:MM:SS"')
        print('  "YYYY-MM-DDTHH:MM"')
        print('  "YYYY-MM-DDTHH:MM:SS"')
        print('  "YYYY-MM-DDTHH:MM:SSZ"')
        sys.exit(1)
    
    # Validate time range
    if end_time <= start_time:
        print("Error: End time must be after start time")
        sys.exit(1)
    
    download_satellite_data(norad_id, start_time, end_time, step)
