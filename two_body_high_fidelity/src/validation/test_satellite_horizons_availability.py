#!/usr/bin/env python3
"""
Test which satellites are available in JPL HORIZONS across LEO, MEO, and GEO
"""

from astroquery.jplhorizons import Horizons

# Satellite catalog organized by orbital regime
SATELLITE_CATALOG = {
    'LEO': {
        # Low Earth Orbit (< 2000 km altitude)
        'ISS': 25544,                    # International Space Station
        'HUBBLE': 20580,                 # Hubble Space Telescope
        'TIANGONG': 48274,               # Chinese Space Station
        'STARLINK-1007': 44713,          # Starlink satellite
        'SENTINEL-1A': 39634,            # Earth observation
        'SENTINEL-2A': 40697,            # Earth observation
        'SENTINEL-3A': 41335,            # Earth observation
        'LANDSAT-8': 39084,              # Earth observation
        'LANDSAT-9': 49260,              # Earth observation
        'TERRA': 25994,                  # Earth observation
        'AQUA': 27424,                   # Earth observation
    },
    'MEO': {
        # Medium Earth Orbit (2000 - 35000 km altitude)
        # GPS satellites
        'GPS-IIR-5': 26407,              # NAVSTAR-48
        'GPS-IIF-2': 38833,              # NAVSTAR-67
        'GPS-IIF-3': 39166,              # NAVSTAR-68
        # Galileo satellites
        'GALILEO-101': 37846,            # Galileo IOV-FM1
        'GALILEO-102': 37847,            # Galileo IOV-FM2
        'GALILEO-103': 38857,            # Galileo IOV-FM3
        # GLONASS satellites
        'GLONASS-730': 36111,            # GLONASS-M
        'GLONASS-747': 37829,            # GLONASS-M
        # BeiDou satellites
        'BEIDOU-M1': 36287,              # BeiDou-2
        'BEIDOU-3M1': 43001,             # BeiDou-3
    },
    'GEO': {
        # Geostationary Orbit (~35,786 km altitude)
        'GOES-16': 41866,                # GOES-R (weather)
        'GOES-17': 43226,                # GOES-S (weather)
        'GOES-18': 51850,                # GOES-T (weather)
        'GOES-19': 60134,                # GOES-U (weather)
        'INTELSAT-39': 45364,            # Communications
        'ASTRA-2E': 38087,               # Communications
        'EUTELSAT-7B': 39215,            # Communications
        'SES-14': 43227,                 # Communications
        'METEOSAT-11': 41105,            # Weather (European)
        'HIMAWARI-8': 40267,             # Weather (Japanese)
        'ELEKTRO-L-2': 41105,            # Weather (Russian)
    }
}

def test_horizons_availability(norad_id, sat_name):
    """
    Test if a satellite is available in HORIZONS
    
    Returns:
    --------
    bool : True if available, False otherwise
    """
    sat_id = -(100000 + norad_id)
    
    try:
        obj = Horizons(
            id=sat_id,
            location='@399',
            epochs={'start': '2025-10-01 00:00',
                   'stop': '2025-10-08 00:00',
                   'step': '1h'}
        )
    
        vec = obj.vectors(refplane='earth')
        
        if len(vec) > 0:
            print(f"✓ {sat_name:20s} (NORAD {norad_id:5d}) - AVAILABLE")
            print(f"  Target: {vec['targetname'][0]}")
            return True
        else:
            print(f"✗ {sat_name:20s} (NORAD {norad_id:5d}) - No data returned")
            return False
            
    except Exception as e:
        print(f"✗ {sat_name:20s} (NORAD {norad_id:5d}) - NOT AVAILABLE")
        # print(f"  Error: {str(e)[:100]}")
        return False


def main():
    """Test all satellites across LEO, MEO, and GEO"""
    print("="*80)
    print("Testing Satellite Availability in JPL HORIZONS")
    print("="*80)
    print()
    
    results_by_regime = {}
    
    for regime, satellites in SATELLITE_CATALOG.items():
        print(f"\n{regime} Satellites:")
        print("-" * 80)
        
        available = []
        not_available = []
        
        for sat_name, norad_id in satellites.items():
            if test_horizons_availability(norad_id, sat_name):
                available.append((sat_name, norad_id))
            else:
                not_available.append((sat_name, norad_id))
        
        results_by_regime[regime] = {
            'available': available,
            'not_available': not_available,
            'total': len(satellites)
        }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY BY ORBITAL REGIME")
    print("="*80)
    
    for regime in ['LEO', 'MEO', 'GEO']:
        res = results_by_regime[regime]
        print(f"\n{regime}:")
        print(f"  Available: {len(res['available'])}/{res['total']}")
        
        if res['available']:
            print(f"  Recommended satellites:")
            for sat_name, norad_id in res['available'][:3]:  # Show top 3
                print(f"    '{sat_name}': {norad_id},")
    
    # Overall summary
    total_available = sum(len(r['available']) for r in results_by_regime.values())
    total_satellites = sum(r['total'] for r in results_by_regime.values())
    
    print("\n" + "="*80)
    print(f"OVERALL: {total_available}/{total_satellites} satellites available in HORIZONS")
    print("="*80)


if __name__ == "__main__":
    main()
