#!/usr/bin/env python3
"""
Test which MEO satellites are available in JPL HORIZONS
"""

from astroquery.jplhorizons import Horizons

# MEO satellite candidates (GNSS and others)
MEO_SATELLITES = {
    # GPS satellites (Block II, IIA, IIR, IIRM, IIF, III)
    'GPS-II-1': 20061,      # GPS (1989)
    'GPS-II-2': 20185,      # GPS (1989)
    'GPS-IIA-12': 21552,    # GPS (1991)
    'GPS-IIA-15': 21890,    # GPS (1992)
    'GPS-IIA-18': 21930,    # GPS (1992)
    'GPS-IIA-21': 22014,    # GPS (1990)
    'GPS-IIA-23': 22108,    # GPS (1993)
    'GPS-IIA-24': 22231,    # GPS (1993)
    'GPS-IIA-25': 22275,    # GPS (1992)
    'GPS-IIA-26': 22446,    # GPS (1993)
    'GPS-IIA-27': 22877,    # GPS (1992)
    'GPS-IIA-28': 22700,    # GPS (1992)
    'GPS-IIA-29': 22779,    # GPS (1992)
    'GPS-IIA-31': 23027,    # GPS (1993)
    'GPS-IIA-32': 23833,    # GPS (1996)
    'GPS-IIA-33': 23953,    # GPS (1996)
    'GPS-IIA-35': 24876,    # GPS (1997)
    'GPS-IIA-36': 25030,    # GPS (1997)
    'GPS-IIA-37': 25933,    # GPS (2000)
    'GPS-IIR-2': 24876,     # GPS (1997)
    'GPS-IIR-3': 25933,     # GPS (2000)
    'GPS-IIR-4': 26360,     # GPS (2000)
    'GPS-IIR-5': 26407,     # GPS (2000)
    'GPS-IIR-6': 26605,     # GPS (2003)
    'GPS-IIR-7': 26690,     # GPS (2003)
    'GPS-IIR-8': 27663,     # GPS (2003)
    'GPS-IIR-9': 27704,     # GPS (2003)
    'GPS-IIR-10': 28129,    # GPS (2004)
    'GPS-IIR-11': 28190,    # GPS (2004)
    'GPS-IIR-12': 28361,    # GPS (2004)
    'GPS-IIR-13': 28474,    # GPS (2005)
    'GPS-IIR-14': 28874,    # GPS (2005)
    'GPS-IIR-15': 29486,    # GPS (2006)
    'GPS-IIR-16': 29601,    # GPS (2006)
    'GPS-IIR-17': 32260,    # GPS (2007)
    'GPS-IIR-18': 32384,    # GPS (2007)
    'GPS-IIR-19': 32711,    # GPS (2008)
    'GPS-IIR-20': 35752,    # GPS (2009)
    'GPS-IIF-1': 37753,     # GPS (2010)
    'GPS-IIF-2': 38833,     # GPS (2011)
    'GPS-IIF-3': 39166,     # GPS (2012)
    'GPS-IIF-4': 39533,     # GPS (2012)
    'GPS-IIF-5': 39741,     # GPS (2014)
    'GPS-IIF-6': 40105,     # GPS (2014)
    'GPS-IIF-7': 40294,     # GPS (2014)
    'GPS-IIF-8': 40534,     # GPS (2015)
    'GPS-IIF-9': 40730,     # GPS (2015)
    'GPS-IIF-10': 41019,    # GPS (2015)
    'GPS-IIF-11': 41328,    # GPS (2016)
    'GPS-IIF-12': 41549,    # GPS (2016)
    'GPS-IIIA-1': 43873,    # GPS (2018)
    'GPS-IIIA-2': 44506,    # GPS (2019)
    'GPS-IIIA-3': 45854,    # GPS (2020)
    'GPS-IIIA-4': 46826,    # GPS (2020)
    'GPS-IIIA-5': 48859,    # GPS (2021)
    
    # Galileo satellites
    'GALILEO-101': 37846,   # Galileo (2011) - IOV-FM1
    'GALILEO-102': 37847,   # Galileo (2011) - IOV-FM2
    'GALILEO-103': 38857,   # Galileo (2012) - IOV-FM3
    'GALILEO-104': 38858,   # Galileo (2012) - IOV-FM4
    'GALILEO-201': 40128,   # Galileo (2014) - FOC-FM1
    'GALILEO-202': 40129,   # Galileo (2014) - FOC-FM2
    'GALILEO-203': 40544,   # Galileo (2015) - FOC-FM3
    'GALILEO-204': 40545,   # Galileo (2015) - FOC-FM4
    'GALILEO-205': 40889,   # Galileo (2015) - FOC-FM5
    'GALILEO-206': 40890,   # Galileo (2015) - FOC-FM6
    'GALILEO-207': 41174,   # Galileo (2015) - FOC-FM7
    'GALILEO-208': 41175,   # Galileo (2015) - FOC-FM8
    'GALILEO-209': 41549,   # Galileo (2016) - FOC-FM9
    'GALILEO-210': 41550,   # Galileo (2016) - FOC-FM10
    'GALILEO-211': 41859,   # Galileo (2016) - FOC-FM11
    'GALILEO-212': 41860,   # Galileo (2016) - FOC-FM12
    'GALILEO-213': 41861,   # Galileo (2016) - FOC-FM13
    'GALILEO-214': 41862,   # Galileo (2016) - FOC-FM14
    
    # GLONASS satellites
    'GLONASS-730': 36111,   # GLONASS-M (2010)
    'GLONASS-731': 36112,   # GLONASS-M (2010)
    'GLONASS-732': 36113,   # GLONASS-M (2010)
    'GLONASS-735': 36400,   # GLONASS-M (2010)
    'GLONASS-736': 36401,   # GLONASS-M (2010)
    'GLONASS-737': 36402,   # GLONASS-M (2010)
    'GLONASS-744': 37139,   # GLONASS-M (2011)
    'GLONASS-745': 37140,   # GLONASS-M (2011)
    'GLONASS-747': 37829,   # GLONASS-M (2011)
    'GLONASS-748': 37867,   # GLONASS-M (2011)
    'GLONASS-749': 37868,   # GLONASS-M (2011)
    'GLONASS-750': 37869,   # GLONASS-M (2011)
    'GLONASS-751': 38723,   # GLONASS-M (2012)
    'GLONASS-752': 38724,   # GLONASS-M (2012)
    'GLONASS-753': 38725,   # GLONASS-M (2012)
    'GLONASS-754': 39155,   # GLONASS-M (2013)
    'GLONASS-755': 39620,   # GLONASS-M (2013)
    'GLONASS-K1-11': 41330, # GLONASS-K1 (2016)
    'GLONASS-K2-12': 42939, # GLONASS-K2 (2017)
    
    # BeiDou satellites
    'BEIDOU-M1': 36287,     # BeiDou-2 (2010)
    'BEIDOU-M2': 36828,     # BeiDou-2 (2010)
    'BEIDOU-M3': 37210,     # BeiDou-2 (2011)
    'BEIDOU-M4': 37256,     # BeiDou-2 (2011)
    'BEIDOU-M5': 37384,     # BeiDou-2 (2011)
    'BEIDOU-M6': 37763,     # BeiDou-2 (2011)
    'BEIDOU-3M1': 43001,    # BeiDou-3 (2017)
    'BEIDOU-3M2': 43002,    # BeiDou-3 (2017)
    'BEIDOU-3M3': 43107,    # BeiDou-3 (2018)
    'BEIDOU-3M4': 43108,    # BeiDou-3 (2018)
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
    """Test all MEO satellites"""
    print("="*80)
    print("Testing MEO Satellite Availability in JPL HORIZONS")
    print("="*80)
    print()
    
    available = []
    not_available = []
    
    # Group by constellation
    constellations = {
        'GPS': [k for k in MEO_SATELLITES.keys() if 'GPS' in k],
        'Galileo': [k for k in MEO_SATELLITES.keys() if 'GALILEO' in k],
        'GLONASS': [k for k in MEO_SATELLITES.keys() if 'GLONASS' in k],
        'BeiDou': [k for k in MEO_SATELLITES.keys() if 'BEIDOU' in k],
        'Other': [k for k in MEO_SATELLITES.keys() if not any(x in k for x in ['GPS', 'GALILEO', 'GLONASS', 'BEIDOU'])],
    }
    
    for constellation, satellites in constellations.items():
        if satellites:
            print(f"\n{constellation} Satellites:")
            print("-" * 80)
            
            for sat_name in satellites:
                norad_id = MEO_SATELLITES[sat_name]
                if test_horizons_availability(norad_id, sat_name):
                    available.append((sat_name, norad_id))
                else:
                    not_available.append((sat_name, norad_id))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Available satellites: {len(available)}/{len(MEO_SATELLITES)}")
    
    if available:
        print("\nRecommended MEO satellites for analysis:")
        for sat_name, norad_id in available[:5]:  # Show first 5
            print(f"  '{sat_name}': {norad_id},")
    else:
        print("\n⚠️  No MEO satellites found in HORIZONS!")
        print("Consider using TLE-only analysis for MEO satellites.")


if __name__ == "__main__":
    main()
