from datetime import datetime, timedelta
from sgp4.api import Satrec
from constants import Constants


class SGP4Helper:
    """
    Helper class for SGP4 operations and conversions.
    """
    
    @staticmethod
    def get_epoch_as_datetime(satellite):
        """
        Convert SGP4 satellite epoch to datetime object.
        
        Args:
            satellite: Satrec object with epochyr and epochdays attributes
            
        Returns:
            datetime: Epoch as datetime object
        """
        epoch_year = satellite.epochyr + 2000 if satellite.epochyr < 57 else satellite.epochyr + 1900
        epoch_days = satellite.epochdays
        epoch_datetime = datetime(epoch_year, 1, 1) + timedelta(days=epoch_days - 1)
        return epoch_datetime

    @staticmethod
    def get_orbital_elements(satellite):
        """
        Extract orbital elements directly from TLE.
        
        Args:
            satellite: Satrec object
            
        Returns:
            dict: Dictionary containing orbital elements:
                - sma  : semi-major axis [m]
                - ecc  : eccentricity
                - inc  : inclination [rad]
                - raan : right ascension of ascending node [rad]
                - argp : argument of perigee [rad]
                - ma   : mean anomaly [rad]
                - n    : mean motion [rad/min]
        """
        # Extract from TLE (note: SGP4 uses slightly different names)
        ecc  = satellite.ecco      # eccentricity
        inc  = satellite.inclo     # inclination [rad]
        raan = satellite.nodeo     # right-ascension of the ascending node [rad]
        argp = satellite.argpo     # argument of perigee [rad]
        ma   = satellite.mo        # mean anomaly [rad]
        mm   = satellite.no_kozai  # mean motion [rad/min]

        # Calculate semi-major axis from mean motion
        # n = sqrt(mu / a^3), so a = (mu / n^2)^(1/3)
        n_rad_per_sec = mm / Constants.SEC_PER_MIN  # Convert to rad/s
        sma = (Constants.MU_EARTH / n_rad_per_sec**2)**(1/3)
        
        return {
            'sma'  : sma,
            'ecc'  : ecc,
            'inc'  : inc,
            'raan' : raan,
            'argp' : argp,
            'ma'   : ma,
            'n'    : mm
        }
