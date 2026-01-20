"""
Topocentric coordinate computation for orbit determination.

This module computes topocentric coordinates (azimuth, elevation, range)
from a ground station's perspective, which are fundamental measurements
for orbit determination systems.
"""

import datetime
import numpy as np

from typing import Optional

from src.model.frame_converter import FrameConverter, VectorConverter
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
      Propagation result containing 'state' (6xN array) and 'plot_delta_time'.
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
  time_s        = result.time_grid.deltas
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


def compute_topocentric_coordinates_with_rates(
  result       : PropagationResult,
  tracker      : TrackerStation,
  epoch_dt_utc : Optional[datetime.datetime] = None,
) -> TopocentricCoordinates:
  """
  Compute topocentric coordinates and rates from a ground station.

  This function computes azimuth, elevation, range and their time derivatives
  (azimuth_dot, elevation_dot, range_dot) which are useful for orbit
  determination and tracking applications.

  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_delta_time'.
    tracker : TrackerStation
      Ground tracking station with latitude, longitude, altitude.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.

  Output:
  -------
    topo : TopocentricCoordinates
      Topocentric coordinates and rates (azimuth, elevation, range,
      azimuth_dot, elevation_dot, range_dot) arrays.
  """
  # Extract J2000 state vectors
  j2000_state   = result.state
  j2000_pos_vec = j2000_state[0:3, :]
  j2000_vel_vec = j2000_state[3:6, :]
  time_s        = result.time_grid.deltas
  n_points      = j2000_state.shape[1]

  # Convert epoch to ET
  if epoch_dt_utc is not None:
    epoch_et = utc_to_et(epoch_dt_utc)
  else:
    epoch_et = 0.0

  # Transform all positions and velocities to body-fixed frame
  sat_pos_bf_array = np.zeros((3, n_points))
  sat_vel_bf_array = np.zeros((3, n_points))

  for i in range(n_points):
    epoch_et_i = epoch_et + time_s[i]
    sat_pos_bf_array[:, i], sat_vel_bf_array[:, i] = VectorConverter.j2000_to_iau_earth(
      j2000_pos_vec = j2000_pos_vec[:, i],
      j2000_vel_vec = j2000_vel_vec[:, i],
      time_et       = epoch_et_i,
    )

  # Compute topocentric coordinates and rates using vectorized converter
  azimuth, elevation, range_arr, az_dot, el_dot, rng_dot = TopocentricConverter.posvel_to_topocentric_array(
    sat_pos_array = sat_pos_bf_array,
    sat_vel_array = sat_vel_bf_array,
    tracker_lat   = tracker.position.latitude,
    tracker_lon   = tracker.position.longitude,
    tracker_alt   = tracker.position.altitude,
  )

  return TopocentricCoordinates(
    azimuth       = azimuth,
    elevation     = elevation,
    range         = range_arr,
    azimuth_dot   = az_dot,
    elevation_dot = el_dot,
    range_dot     = rng_dot,
  )
