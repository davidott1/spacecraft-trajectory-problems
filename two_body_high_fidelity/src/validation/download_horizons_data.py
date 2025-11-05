"""
Download GOES-16 ephemeris from NASA HORIZONS

Requires: pip install astroquery

GOES-16: NORAD 41866
Use NORAD catalog number as HORIZONS ID
"""

from astroquery.jplhorizons import Horizons
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import requests
from astropy.time import Time

# Get the project root directory (assuming this script is at src/validation/download_horizons_data.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# GOES satellite catalog
GOES_SATELLITES = {
    'GOES-16': 41866,
    'GOES-17': 43226,
    'GOES-18': 51850,
}

# Extended satellite catalog including LEO and MEO
ALL_SATELLITES = {
    # LEO satellites (Low Earth Orbit, ~400-700 km altitude)
    'ISS': 25544,         # International Space Station
    'TERRA': 25994,       # Earth observation satellite
    'AQUA': 27424,        # Earth observation satellite
    
    # MEO satellites (Medium Earth Orbit, ~20,200 km altitude)
    'GPS-IIR-5': 26407,   # NAVSTAR-48
    'GPS-IIF-2': 38833,   # NAVSTAR-67
    'GPS-IIF-3': 39166,   # NAVSTAR-68
    
    # GEO satellites (Geostationary Orbit, ~35,786 km altitude)
    'GOES-16': 41866,     # GOES-R (weather)
    'GOES-17': 43226,     # GOES-S (weather)
    'GOES-18': 51850,     # GOES-T (weather)
}

def get_satellite_name(norad_id):
    """Get satellite name from NORAD ID"""
    for name, id in ALL_SATELLITES.items():
        if id == norad_id:
            return name.lower().replace('-', '')
    return f"sat{norad_id}"


def download_tle_for_satellite(norad_id, output_file=None):
    """
    Download TLE for a satellite from Celestrak
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog number (GOES-16 = 41866, GOES-17 = 43226, GOES-18 = 51850)
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
                        
                        from astropy.time import Time
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


def download_goes_ephemeris(
    norad_id    = 41866,
    days        = 7,
    output_file = None,
    start_time  = None,
):
    """
    Download GOES satellite ephemeris from HORIZONS
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog number (GOES-16 = 41866, GOES-17 = 43226, GOES-18 = 51850)
    days : float
        Number of days to download
    output_file : str or Path
        Output file path
    start_time : datetime or astropy.time.Time, optional
        Start time for ephemeris. If None, uses current time.
    """
    # Set time range
    if start_time is None:
        start_time = datetime.now()
    elif hasattr(start_time, 'datetime'):  # astropy Time object
        start_time = start_time.datetime
    
    end_time = start_time + timedelta(days=days)
    
    print(f"Downloading ephemeris for NORAD {norad_id}")
    print(f"Time range: {start_time} to {end_time}")
    
    # Query HORIZONS using NORAD catalog number
    sat_id = -(100000 + norad_id)
    obj = Horizons(
        id=f"{sat_id}",
        location='@399',  # Earth center (geocentric)
        epochs={'start' : start_time.strftime('%Y-%m-%d %H:%M'),
                'stop'  : end_time.strftime('%Y-%m-%d %H:%M'),
                'step'  : '1h'}  # 1 hour intervals
    )
    
    # Get vectors in ICRF/J2000 equatorial frame
    # refplane='earth' gives Earth mean equator and equinox of J2000.0 (ICRF)
    vectors = obj.vectors(refplane='earth') # type: ignore
    
    # # refplane='earth' gives equatorial, refplane='ecliptic' would give ecliptic
    # vectors = obj.vectors(refplane='earth',    # Earth equatorial plane
    #                       delta_T=True)         # Include Delta-T values
    # # vectors = obj.vectors(refplane='earth', aberrations='geometric')


    print(f"Retrieved {len(vectors)} data points")
    print(f"Reference frame: ICRF/J2000 Earth equatorial (refplane='earth')")
    
    # Diagnostics
    print("\nData preview:")
    print(f"  Columns: {vectors.colnames}")
    print(f"  Target: {vectors['targetname'][0]}")
    if 'x' in vectors.colnames:
        print(f"  Sample X: {vectors['x'][0]}")
        print(f"  Sample VX: {vectors['vx'][0]}")
        r_sample = np.sqrt(vectors['x'][0]**2 + vectors['y'][0]**2 + vectors['z'][0]**2)
        print(f"  Sample |r|: {r_sample}")
    
    # Save to file
    if output_file is None:
        data_dir = PROJECT_ROOT / 'data' / 'ephems'
        data_dir.mkdir(exist_ok=True, parents=True)
        output_file = data_dir / f'horizons_norad{norad_id}.csv'
    
    vectors.write(output_file, format='csv', overwrite=True)
    print(f"Saved to: {output_file}")
    
    return vectors


def download_goes_data_package(norad_id=41866, days=7):
    """
    Download both HORIZONS ephemeris and historical TLEs for GOES satellite
    Uses specific date range: October 1-8, 2025
    
    Parameters:
    -----------
    norad_id : int
        NORAD catalog number (GOES-16 = 41866, GOES-18 = 51850, GOES-19 = 60134)
    days : float
        Number of days for HORIZONS ephemeris (default 7)
    
    Returns:
    --------
    dict with 'horizons' ephemeris table and 'tle' lines
    """
    data_dir = PROJECT_ROOT / 'data' / 'ephems'
    data_dir.mkdir(exist_ok=True, parents=True)
    
    sat_name = get_satellite_name(norad_id)
    
    # Use fixed time range: October 1-8, 2025
    start_time = datetime(2025, 10, 1, 0, 0, 0)
    end_time = datetime(2025, 10, 8, 0, 0, 0)
    
    print("="*80)
    print(f"Downloading data for {sat_name.upper()} (NORAD {norad_id})")
    print(f"Time range: {start_time} to {end_time}")
    print("="*80)
    
    # Download HORIZONS ephemeris for Oct 1-8, 2025
    print("\n" + "="*80)
    print(f"Downloading HORIZONS ephemeris for {sat_name.upper()}")
    print("="*80)
    horizons_file = data_dir / f'horizons_{sat_name}.csv'
    horizons_data = download_goes_ephemeris(
        norad_id=norad_id, 
        days=(end_time - start_time).days, 
        output_file=horizons_file,
        start_time=start_time
    )
    
    # Download historical TLEs for the same time period
    print("\n" + "="*80)
    print(f"Downloading historical TLEs for {sat_name.upper()}")
    print("="*80)
    tle_history_file = data_dir / f'tle_history_{sat_name}.txt'
    tle_history = download_historical_tles(
        norad_id=norad_id,
        start_time=start_time,
        end_time=end_time,
        output_file=tle_history_file
    )
    
    # Download latest TLE as reference
    print("\n" + "="*80)
    print(f"Downloading latest TLE for reference")
    print("="*80)
    tle_file = data_dir / f'tle_{sat_name}.txt'
    tle_line1, tle_line2, tle_epoch_jd = download_tle_for_satellite(norad_id, output_file=tle_file)
    
    # Get HORIZONS epoch from the data
    horizons_epoch_jd = Time(horizons_data['datetime_jd'][0], format='jd').jd
    horizons_epoch_time = Time(horizons_epoch_jd, format='jd')
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"HORIZONS start:     {start_time}")
    print(f"HORIZONS end:       {end_time}")
    print(f"Number of TLEs:     {len(tle_history)}")
    print(f"HORIZONS file:      {horizons_file}")
    print(f"TLE file:           {tle_file}")
    print(f"TLE history file:   {tle_history_file}")
    
    return {
        'horizons'      : horizons_data,
        'tle'           : (tle_line1, tle_line2),
        'tle_history'   : tle_history,
        'start_time'    : start_time,
        'end_time'      : end_time,
        'horizons_file' : horizons_file,
        'tle_file'      : tle_file,
        'tle_history_file': tle_history_file,
        'sat_name'      : sat_name,
        'norad_id'      : norad_id,
    }


def download_all_goes_satellites():
    """
    Download data for all GOES satellites (16, 18, 19)
    
    Returns:
    --------
    dict : Dictionary with satellite names as keys and download results as values
    """
    results = {}
    
    for sat_name, norad_id in GOES_SATELLITES.items():
        print("\n" + "="*80)
        print(f"PROCESSING {sat_name}")
        print("="*80 + "\n")
        
        try:
            result = download_goes_data_package(norad_id=norad_id, days=7)
            results[sat_name] = result
            print(f"\n✓ Successfully downloaded data for {sat_name}")
        except Exception as e:
            print(f"\n✗ Failed to download data for {sat_name}: {e}")
            results[sat_name] = {'error': str(e)}
    
    print("\n" + "="*80)
    print("ALL DOWNLOADS COMPLETE")
    print("="*80)
    print(f"Successfully downloaded: {sum(1 for r in results.values() if 'error' not in r)}/{len(GOES_SATELLITES)}")
    
    return results


def download_all_satellites():
    """
    Download data for all satellites (LEO, MEO, GEO)
    
    Returns:
    --------
    dict : Dictionary with satellite names as keys and download results as values
    """
    results = {}
    
    for sat_name, norad_id in ALL_SATELLITES.items():
        print("\n" + "="*80)
        print(f"PROCESSING {sat_name}")
        print("="*80 + "\n")
        
        try:
            result = download_goes_data_package(norad_id=norad_id, days=7)
            results[sat_name] = result
            print(f"\n✓ Successfully downloaded data for {sat_name}")
        except Exception as e:
            print(f"\n✗ Failed to download data for {sat_name}: {e}")
            results[sat_name] = {'error': str(e)}
    
    print("\n" + "="*80)
    print("ALL DOWNLOADS COMPLETE")
    print("="*80)
    print(f"Successfully downloaded: {sum(1 for r in results.values() if 'error' not in r)}/{len(ALL_SATELLITES)}")
    
    return results


if __name__ == "__main__":
    # Download data for all satellites (LEO, MEO, GEO)
    download_all_satellites()
