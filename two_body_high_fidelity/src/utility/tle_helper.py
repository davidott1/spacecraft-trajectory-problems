import math
from datetime import datetime, timedelta
from sgp4.api import Satrec

def modify_tle_bstar(
  tle_line1   : str,
  bstar_value : float = 0.0,
) -> str:
  """
  Modify the B* drag term in TLE line 1.
  
  Input:
  ------
    tle_line1 : str
      First line of TLE.
    bstar_value : float
      New B* value (default 0.0).
  
  Output:
  -------
    str
      Modified TLE line 1 with new B* value.
  """
  # B* is in columns 54-61 (0-indexed: 53-60) in format: ±.nnnnn±n
  # Example: " 12345-3" means 0.12345 × 10^-3
  
  if bstar_value == 0.0:
    bstar_str = " 00000-0"
  else:
    # Convert to scientific notation and format
    exponent = math.floor(math.log10(abs(bstar_value)))
    mantissa = bstar_value / (10 ** exponent)
    # Adjust to TLE format: mantissa in [0.1, 1) range
    mantissa = mantissa / 10
    exponent = exponent + 1
    sign = '-' if bstar_value < 0 else ' '
    exp_sign = '-' if exponent < 0 else '+'
    # Format mantissa as "0.nnnnn" and extract the 5 decimal digits
    # Use replace to handle edge case where rounding might produce "1.00000"
    mantissa_formatted = f"{abs(mantissa):.5f}"
    if mantissa_formatted.startswith('0.'):
      mantissa_digits = mantissa_formatted[2:]
    else:
      # Handle edge case where rounding produces >= 1.0
      mantissa = mantissa / 10
      exponent = exponent + 1
      mantissa_digits = f"{abs(mantissa):.5f}"[2:]
    bstar_str = f"{sign}{mantissa_digits}{exp_sign}{abs(exponent)}"
  
  # Replace B* in TLE line 1 (columns 53-60)
  modified_line1 = tle_line1[:53] + bstar_str + tle_line1[61:]
  
  # Recalculate checksum (last character)
  checksum = 0
  for char in modified_line1[:-1]:
    if char.isdigit():
      checksum += int(char)
    elif char == '-':
      checksum += 1
  modified_line1 = modified_line1[:-1] + str(checksum % 10)
  
  return modified_line1


def get_tle_satellite_and_tle_epoch(
  tle_line1 : str,
  tle_line2 : str,
) -> tuple[datetime, Satrec]:
  """
  Create Satrec object and extract epoch from TLE. Deconstruct datetime from year 
  and fractional days to make it precise.
  
  Input:
  ------
    tle_line1 : str
      First line of TLE.
    tle_line2 : str
      Second line of TLE.
  
  Output:
  -------
    tuple[datetime, Satrec]
      Epoch datetime and Satellite object.
      
  Notes:
  ------
    Preferred method: Explicitly handle year rollover and fractional days.
  
    Explanation:
      Precision: 
        Julian Dates are large numbers (~2.45e6). Performing arithmetic 
        on them (like subtracting J2000 epoch) before converting to datetime can 
        introduce small floating-point errors compared to using the specific 
        year and fractional day provided directly by the TLE.

      Mathematical Detail:
      - A standard 64-bit float (double) has ~15-17 significant decimal digits.
      - A modern Julian Date is approx 2,460,000.xxxxxx
      - To represent 1 microsecond (1e-6 s), we need a day fraction of:
        1e-6 / 86400 ≈ 1.157e-11
      - So we need precision down to the 11th decimal place.
      - JD = 2,460,219.50000000001157...
        Digits before decimal: 7
        Digits needed after decimal: 11
        Total digits needed: 18
      - Since 18 > 15-17 (double precision limit), the last digits are truncated 
        or rounded, losing the microsecond precision.
      - By using the fractional day directly (0.50000000001157...), we only need 
        11 digits total, which fits comfortably within double precision.
  """
  # Satellite object of TLE
  tle_satellite = Satrec.twoline2rv(
    tle_line1,
    tle_line2,
  )
  
  # Extract year
  tle_year = tle_satellite.epochyr
  if tle_year < 57:
    tle_year += 2000
  else:
    tle_year += 1900

  # Extract days of year
  tle_days = tle_satellite.epochdays

  # Convert to datetime object
  tle_time_datetime = datetime(tle_year, 1, 1) + timedelta(days=tle_days - 1)
  
  # Return epoch datetime and satellite object
  return tle_time_datetime, tle_satellite
