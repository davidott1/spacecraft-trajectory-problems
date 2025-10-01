from datetime import datetime, timedelta
from sgp4.api import Satrec
from constants import Constants
from dynamics import Dynamics
from sgp4.api import WGS72OLD, WGS72, WGS84


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
    
    @staticmethod
    def create_satrec_from_state(pos, vel, epoch_datetime, bstar=0.0):
        """
        Create a Satrec object from position and velocity vectors.
        
        WARNING: Treats osculating elements as mean (approximation).
        
        Args:
            pos: Position vector [m] (3D numpy array)
            vel: Velocity vector [m/s] (3D numpy array)
            epoch_datetime: Epoch time as datetime object
            bstar: Drag term (default 0.0)
            
        Returns:
            Satrec: New satellite object initialized from state
        """
        import numpy as np
        from sgp4.api import Satrec, jday
        
        # Get osculating elements (treating as mean - approximation!)
        dynamics = Dynamics()
        coe = dynamics.rv2coe(pos, vel)
        
        # Extract elements (in radians)
        ecc  = coe['ecc']
        inc  = coe['inc']
        raan = coe['raan']
        argp = coe['argp']
        ma   = coe['ma']
        
        # Calculate mean motion from semi-major axis
        # n = sqrt(mu / a^3)
        sma = coe['sma']
        n_rad_per_sec = np.sqrt(Constants.MU_EARTH / sma**3)
        n_rad_per_min = n_rad_per_sec * Constants.SEC_PER_MIN
        
        # Convert epoch to SGP4 format
        year = epoch_datetime.year
        epoch_year = year % 100  # Two-digit year
        
        # Calculate day of year with fractional part
        jan1 = datetime(year, 1, 1)
        epoch_days = (epoch_datetime - jan1).total_seconds() / Constants.SEC_PER_DAY + 1.0
        
        # FIX: Compute epoch as days since 1949-12-31 00:00 UT (required by sgp4init).
        # Previous implementation used: epoch_year + epoch_days / 365.25 (incorrect for sgp4init).
        jd, fr = jday(epoch_datetime.year, epoch_datetime.month, epoch_datetime.day,
                      epoch_datetime.hour, epoch_datetime.minute,
                      epoch_datetime.second + epoch_datetime.microsecond * 1e-6)
        epoch_days_since_1949 = (jd + fr) - 2433281.5  # 1949-12-31 00:00 UT JD = 2433281.5

        satrec = Satrec()
        satrec.sgp4init(
            WGS72,             # gravity model
            'i',               # improved mode
            99999,             # dummy sat number
            epoch_days_since_1949,  # epoch (days since 1949-12-31)
            bstar,             # bstar drag
            0.0,               # ndot
            0.0,               # nddot
            ecc,               # ecco
            argp,              # argpo
            inc,               # inclo
            ma,                # mo
            n_rad_per_min,     # no_kozai
            raan,              # nodeo
        )
        return satrec

    @staticmethod
    def osculating_to_mean_elements(osc_coe, simplified=True):
        """
        Convert osculating elements to mean elements.
        
        Args:
            osc_coe: Dictionary of osculating elements (from rv2coe)
            simplified: If True, use first-order J2 correction only
            
        Returns:
            Dictionary of mean elements
        """
        import numpy as np
        from constants import Constants
        
        if simplified:
            # First-order J2 correction (Brouwer theory)
            # This removes dominant short-period variations
            
            J2 = 1.08263e-3  # Earth's J2 coefficient
            R_earth = Constants.RADIUS_EARTH
            
            a_osc = osc_coe['sma']
            e_osc = osc_coe['ecc']
            i_osc = osc_coe['inc']
            raan_osc = osc_coe['raan']
            argp_osc = osc_coe['argp']
            ma_osc = osc_coe['ma']
            
            # Calculate mean elements (first-order corrections)
            p = a_osc * (1 - e_osc**2)  # Semi-latus rectum
            eta = np.sqrt(1 - e_osc**2)
            
            # Mean semi-major axis (same as osculating to first order)
            a_mean = a_osc
            
            # Mean eccentricity
            delta_e = -J2 * (R_earth / p)**2 * e_osc * (1 - 1.5 * np.sin(i_osc)**2) / 4
            e_mean = e_osc - delta_e
            
            # Mean inclination (same as osculating)
            i_mean = i_osc
            
            # Mean RAAN
            delta_raan = J2 * (R_earth / p)**2 * np.cos(i_osc) / 2
            raan_mean = raan_osc - delta_raan
            
            # Mean argument of perigee
            delta_argp = J2 * (R_earth / p)**2 * (2 - 2.5 * np.sin(i_osc)**2) / 4
            argp_mean = argp_osc - delta_argp
            
            # Mean mean anomaly
            delta_ma = J2 * (R_earth / p)**2 * eta * (1 - 1.5 * np.sin(i_osc)**2) / 4
            ma_mean = ma_osc - delta_ma
            
            return {
                'sma': a_mean,
                'ecc': e_mean,
                'inc': i_mean,
                'raan': raan_mean % (2 * np.pi),
                'argp': argp_mean % (2 * np.pi),
                'ma': ma_mean % (2 * np.pi)
            }
        else:
            # For higher accuracy, would need full Brouwer-Lyddane theory
            # This is very complex - recommend using a library
            raise NotImplementedError("Full mean element conversion not implemented. Use simplified=True")

