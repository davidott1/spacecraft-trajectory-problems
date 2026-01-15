"""
Topocentric coordinate computation for orbit determination.

This module computes topocentric coordinates (azimuth, elevation, range)
from a ground station's perspective, which are fundamental measurements
for orbit determination systems.
"""

import datetime
import numpy as np

from typing import Optional

from src.model.frame_converter import FrameConverter
from src.model.orbit_converter import TopocentricConverter
from src.model.time_converter  import utc_to_et
from src.schemas.propagation   import PropagationResult
from src.schemas.state         import TrackerStation, TopocentricCoordinates


def compute_topocentric_coordinates(
  result       : PropagationResult,
  tracker      : TrackerStation,
  epoch_dt_utc : Optional[datetime.datetime] = None,
) -> TopocentricCoordinates:
  """
  Compute topocentric coordinates (azimuth, elevation, range) from a ground station.

  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_time_s'.
    tracker : TrackerStation
      Ground tracking station with latitude, longitude, altitude.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.

  Output:
  -------
    topo : TopocentricCoordinates
      Topocentric coordinates (azimuth, elevation, range) arrays.
  """
  # Extract J2000 state vectors
  j2000_state   = result.state
  j2000_pos_vec = j2000_state[0:3, :]
  time_s        = result.plot_time_s
  n_points      = j2000_state.shape[1]

  # Convert epoch to ET
  if epoch_dt_utc is not None:
    epoch_et = utc_to_et(epoch_dt_utc)
  else:
    epoch_et = 0.0

  # Initialize output arrays
  azimuth   = np.zeros(n_points)
  elevation = np.zeros(n_points)
  range_arr = np.zeros(n_points)

  for i in range(n_points):
    # Current ephemeris time
    epoch_et_i = epoch_et + time_s[i]

    # Transform satellite position from J2000 to IAU_EARTH
    rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(epoch_et_i)
    sat_pos_iau_earth = rot_mat_j2000_to_iau_earth @ j2000_pos_vec[:, i]

    # Compute topocentric coordinates using orbit_converter
    azimuth[i], elevation[i], range_arr[i] = TopocentricConverter.pos_to_topocentric(
      sat_pos_vec = sat_pos_iau_earth,
      tracker_lat = tracker.position.latitude,
      tracker_lon = tracker.position.longitude,
      tracker_alt = tracker.position.altitude,
    )

  return TopocentricCoordinates(
    azimuth   = azimuth,
    elevation = elevation,
    range     = range_arr,
  )
