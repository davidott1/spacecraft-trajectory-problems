from pathlib            import Path
from datetime           import datetime, timedelta
from types              import SimpleNamespace
from typing             import Optional
from sgp4.api           import Satrec

from src.load.loader    import load_supported_objects
from src.load.parser    import parse_time


def build_config(
  input_object_type      : str,
  norad_id               : str,
  timespan               : list,
  use_spice              : bool = False,
  include_third_body     : bool = False,
  include_zonal_harmonics: bool = False,
  zonal_harmonics_list   : Optional[list] = None,
  include_srp            : bool = False,
) -> SimpleNamespace:
  """
  Parse, validate, and set up input parameters for orbit propagation.
  
  Input:
  ------
    input_object_type : str
      Type of input object (e.g., norad-id).
    norad_id : str
      NORAD catalog ID of the satellite.
    timespan : list
      Start and end time in ISO format (e.g., ['2025-10-01T00:00:00', '2025-10-02T00:00:00']).
    use_spice : bool
      Flag to enable/disable SPICE usage.
    include_third_body : bool
      Flag to enable/disable third-body gravity.
    include_zonal_harmonics : bool
      Flag to enable/disable zonal harmonics.
    zonal_harmonics_list : list
      List of zonal harmonics to include (e.g., ['J2', 'J3']).
    include_srp : bool
      Flag to enable/disable Solar Radiation Pressure.
  
  Output:
  -------
    SimpleNamespace
      Configuration object containing parsed and calculated propagation parameters.
  
  Raises:
  -------
    ValueError
      If NORAD ID is not supported.
  """
  # Normalize input object type
  input_object_type = input_object_type.replace('-', '_').replace(' ', '_')

  # Unpack timespan
  start_time_str = timespan[0]
  end_time_str   = timespan[1]

  # Validate conditional arguments
  if input_object_type == 'norad_id' and not norad_id:
    raise ValueError("NORAD ID is required when input-object-type is 'norad-id'")

  # Enforce dependencies: SRP requires SPICE
  if include_srp:
    use_spice = True

  # Load supported objects
  supported_objects = load_supported_objects()

  # Validate norad id input
  if norad_id not in supported_objects:
    raise ValueError(f"NORAD ID {norad_id} is not supported. Supported IDs: {list(supported_objects.keys())}")

  # Get object properties
  obj_props = supported_objects[norad_id]

  # Extract TLE lines
  tle_line1 = obj_props['tle']['line_1']
  tle_line2 = obj_props['tle']['line_2']

  # Parse TLE epoch
  satellite    = Satrec.twoline2rv(tle_line1, tle_line2)
  tle_epoch_jd = satellite.jdsatepoch + satellite.jdsatepochF
  tle_epoch_dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=tle_epoch_jd - 2451545.0)
  
  # Target propagation start/end times from arguments
  target_start_dt = parse_time(start_time_str)
  target_end_dt   = parse_time(end_time_str)
  delta_time      = (target_end_dt - target_start_dt).total_seconds()
  
  # Integration time bounds (seconds from TLE epoch)
  integ_time_o     = (target_start_dt - tle_epoch_dt).total_seconds()
  integ_time_f     = integ_time_o + delta_time
  delta_integ_time = integ_time_f - integ_time_o
  
  # Set up paths and files
  paths = setup_paths_and_files(
    norad_id        = norad_id,
    obj_name        = obj_props['name'],
    target_start_dt = target_start_dt,
    target_end_dt   = target_end_dt,
  )
  
  return SimpleNamespace(
    obj_props                = obj_props,
    tle_line1                = tle_line1,
    tle_line2                = tle_line2,
    tle_epoch_dt             = tle_epoch_dt,
    tle_epoch_jd             = tle_epoch_jd,
    target_start_dt          = target_start_dt,
    target_end_dt            = target_end_dt,
    delta_time               = delta_time,
    integ_time_o             = integ_time_o,
    integ_time_f             = integ_time_f,
    delta_integ_time         = delta_integ_time,
    mass                     = obj_props['mass'],
    cd                       = obj_props['drag']['coeff'],
    area_drag                = obj_props['drag']['area'],
    cr                       = obj_props['srp']['coeff'],
    area_srp                 = obj_props['srp']['area'],
    use_spice                = use_spice,
    include_third_body       = include_third_body,
    include_zonal_harmonics  = include_zonal_harmonics,
    zonal_harmonics_list     = zonal_harmonics_list if zonal_harmonics_list else [],
    include_srp              = include_srp,
    output_folderpath        = paths['output_folderpath'],
    spice_kernels_folderpath = paths['spice_kernels_folderpath'],
    horizons_filepath        = paths['horizons_filepath'],
    lsk_filepath             = paths['lsk_filepath'],
  )


def setup_paths_and_files(
  norad_id        : str,
  obj_name        : str,
  target_start_dt : datetime,
  target_end_dt   : datetime,
) -> dict:
  """
  Set up all required folder paths and file names for the propagation.
  
  Input:
  ------
    norad_id : str
      NORAD catalog ID of the satellite.
    obj_name : str
      Name of the object (e.g., 'ISS').
    target_start_dt : datetime
      Target start time as a datetime object.
    target_end_dt : datetime
      Target end time as a datetime object.
  
  Output:
  -------
    dict
      A dictionary containing paths to output, data, SPICE kernels,
      Horizons ephemeris, and leap seconds files.
  """
  # Output directory for figures
  output_folderpath = Path('./output/figures')
  output_folderpath.mkdir(parents=True, exist_ok=True)
  
  # Project and data paths
  # Adjusted for location in src/config/parser.py (depth: src/config/parser.py -> config -> src -> root)
  project_root    = Path(__file__).parent.parent.parent
  data_folderpath = project_root / 'data'
  
  # SPICE kernels path
  spice_kernels_folderpath = data_folderpath / 'spice_kernels'
  lsk_filepath             = spice_kernels_folderpath / 'naif0012.tls'
  
  # Horizons ephemeris file (dynamically named)
  start_str         = target_start_dt.strftime('%Y%m%dT%H%M%SZ')
  end_str           = target_end_dt.strftime('%Y%m%dT%H%M%SZ')
  horizons_filename = f"horizons_ephem_{norad_id}_{obj_name.lower()}_{start_str}_{end_str}_1m.csv"
  horizons_filepath = data_folderpath / 'ephems' / horizons_filename
  
  return {
    'output_folderpath'        : output_folderpath,
    'spice_kernels_folderpath' : spice_kernels_folderpath,
    'horizons_filepath'        : horizons_filepath,
    'lsk_filepath'             : lsk_filepath,
  }
