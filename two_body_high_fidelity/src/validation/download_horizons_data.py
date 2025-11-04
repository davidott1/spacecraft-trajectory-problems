"""
Download GOES-16 ephemeris from NASA HORIZONS

Requires: pip install astroquery

GOES-16 ID in HORIZONS: -76 (NOAA designation)
Other GOES satellites:
  GOES-17: -77
  GOES-18: -78
"""

from astroquery.jplhorizons import Horizons
from pathlib import Path
from datetime import datetime, timedelta

def download_goes_ephemeris(satellite_id=-76, days=7, output_file=None):
    """
    Download GOES satellite ephemeris from HORIZONS
    
    Parameters:
    -----------
    satellite_id : int
        HORIZONS ID (GOES-16 = -76, GOES-17 = -77, etc.)
    days : float
        Number of days to download
    output_file : str or Path
        Output file path
    """
    # Set time range
    start_time = datetime.now()
    end_time = start_time + timedelta(days=days)
    
    print(f"Downloading ephemeris for satellite {satellite_id}")
    print(f"Time range: {start_time} to {end_time}")
    
    # Query HORIZONS
    obj = Horizons(
        id=satellite_id,
        location='500@399',  # Earth center
        epochs={'start': start_time.strftime('%Y-%m-%d'),
                'stop': end_time.strftime('%Y-%m-%d'),
                'step': '1h'}  # 1 hour intervals
    )
    
    # Get vectors (position and velocity)
    vectors = obj.vectors(refplane='earth')
    
    print(f"Retrieved {len(vectors)} data points")
    
    # Save to file
    if output_file is None:
        data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'ephems'
        data_dir.mkdir(exist_ok=True)
        output_file = data_dir / f'horizons_goes{abs(satellite_id)}.csv'
    
    vectors.write(output_file, format='csv', overwrite=True)
    print(f"Saved to: {output_file}")
    
    return vectors


if __name__ == "__main__":
    # Download GOES-16 data for 7 days
    download_goes_ephemeris(satellite_id=-76, days=7)
