"""
Ground track plotting functions.

This module contains functions for plotting ground tracks on 2D maps
and 3D globes.
"""
import numpy             as np
import matplotlib.pyplot as plt
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature

from datetime          import datetime, timedelta
from typing            import Optional
from matplotlib.figure import Figure

from src.model.constants       import CONVERTER, SOLARSYSTEMCONSTANTS
from src.model.frame_converter import FrameConverter
from src.model.time_converter  import utc_to_et
from src.model.orbit_converter import GeographicCoordinateConverter
from src.schemas.propagation   import PropagationResult
from src.schemas.state         import TrackerStation


def _get_coastline_coordinates():
  """
  Extract coastline coordinates from Natural Earth dataset via cartopy.
  
  Returns a list of (lon, lat) arrays for each coastline segment.
  """
  coastlines = cfeature.NaturalEarthFeature('physical', 'coastline', '110m')
  segments = []
  for geom in coastlines.geometries():
    if geom.geom_type == 'LineString':
      coords = np.array(geom.coords)
      segments.append((coords[:, 0], coords[:, 1]))  # (lon, lat)
    elif geom.geom_type == 'MultiLineString':
      for line in geom.geoms:
        coords = np.array(line.coords)
        segments.append((coords[:, 0], coords[:, 1]))
  return segments


def _create_tracker_fov_hemisphere(
  tracker_lat_deg : float,
  tracker_lon_deg : float,
  earth_radius_m  : float,
  fov_radius_m    : float,
  resolution      : int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Create a half-sphere representing the tracker's field of view.

  The hemisphere is tangent to the Earth's surface at the tracker location and
  extends outward with the specified radius.

  Input:
  ------
    tracker_lat_deg : float
      Tracker latitude in degrees
    tracker_lon_deg : float
      Tracker longitude in degrees
    earth_radius_m : float
      Earth radius in meters
    fov_radius_m : float
      Field of view hemisphere radius in meters
    resolution : int
      Resolution for the hemisphere mesh (default: 30)

  Output:
  -------
    x, y, z : tuple[np.ndarray, np.ndarray, np.ndarray]
      Cartesian coordinates of the hemisphere surface (2D arrays)
  """
  # Get the tracker position on Earth's surface
  tracker_pos = GeographicCoordinateConverter.spherical_to_cartesian(
    tracker_lat_deg * CONVERTER.RAD_PER_DEG, 
    tracker_lon_deg * CONVERTER.RAD_PER_DEG, 
    earth_radius_m
  )

  # Calculate the outward normal vector (from Earth center to tracker)
  normal = tracker_pos.copy()
  normal = normal / np.linalg.norm(normal)

  # Create hemisphere centered at origin first (pointing in +z direction)
  u = np.linspace(0, 2 * np.pi, resolution)
  v = np.linspace(0, np.pi / 2, resolution // 2)  # Only upper hemisphere (0 to π/2)
  u_grid, v_grid = np.meshgrid(u, v)

  x_sphere = fov_radius_m * np.sin(v_grid) * np.cos(u_grid)
  y_sphere = fov_radius_m * np.sin(v_grid) * np.sin(u_grid)
  z_sphere = fov_radius_m * np.cos(v_grid)

  # Rotation matrix to align +z axis with normal vector
  # Using Rodrigues' rotation formula
  z_axis = np.array([0, 0, 1])
  rotation_axis = np.cross(z_axis, normal)
  rotation_axis_norm = np.linalg.norm(rotation_axis)

  if rotation_axis_norm > 1e-10:  # Not parallel
    rotation_axis = rotation_axis / rotation_axis_norm
    angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))

    # Rodrigues' rotation matrix
    K = np.array([
      [0, -rotation_axis[2], rotation_axis[1]],
      [rotation_axis[2], 0, -rotation_axis[0]],
      [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
  else:
    # Parallel or anti-parallel
    if np.dot(z_axis, normal) > 0:
      R = np.eye(3)  # No rotation needed
    else:
      R = -np.eye(3)  # Flip
      R[2, 2] = 1  # Keep z positive

  # Apply rotation to all points
  points = np.stack([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])
  rotated_points = R @ points

  # Translate to tracker position
  x_fov = rotated_points[0].reshape(x_sphere.shape) + tracker_pos[0]
  y_fov = rotated_points[1].reshape(y_sphere.shape) + tracker_pos[1]
  z_fov = rotated_points[2].reshape(z_sphere.shape) + tracker_pos[2]

  return x_fov, y_fov, z_fov


def _calculate_hemisphere_ground_track(
  tracker_lat_deg : float,
  tracker_lon_deg : float,
  earth_radius_m  : float,
  fov_radius_m    : float,
  num_points      : int = 100,
) -> tuple[np.ndarray, np.ndarray]:
  """
  Calculate the ground track (lat/lon circle) where the hemisphere's base intersects Earth's surface.

  The base of the hemisphere forms a circle on Earth's surface at a fixed angular distance
  from the tracker location.

  Input:
  ------
    tracker_lat_deg : float
      Tracker latitude in degrees
    tracker_lon_deg : float
      Tracker longitude in degrees
    earth_radius_m : float
      Earth radius in meters
    fov_radius_m : float
      Field of view hemisphere radius in meters
    num_points : int
      Number of points to generate around the circle (default: 100)

  Output:
  -------
    lat_circle, lon_circle : tuple[np.ndarray, np.ndarray]
      Latitudes and longitudes of the circle points in degrees
  """
  # Calculate the angular radius of the FOV circle on Earth's surface
  # This is the angle subtended at Earth's center by the FOV radius
  angular_radius_rad = fov_radius_m / earth_radius_m

  # Convert tracker position to radians
  tracker_lat_rad = tracker_lat_deg * CONVERTER.RAD_PER_DEG
  tracker_lon_rad = tracker_lon_deg * CONVERTER.RAD_PER_DEG

  # Generate points around the circle using spherical geometry
  # We'll use the haversine formula to generate points at a fixed angular distance
  azimuths = np.linspace(0, 2 * np.pi, num_points)

  lat_circle = []
  lon_circle = []

  for az in azimuths:
    # Use spherical trigonometry to find point at angular_radius_rad distance
    # along azimuth az from the tracker location
    lat_rad = np.arcsin(
      np.sin(tracker_lat_rad) * np.cos(angular_radius_rad) +
      np.cos(tracker_lat_rad) * np.sin(angular_radius_rad) * np.cos(az)
    )

    lon_rad = tracker_lon_rad + np.arctan2(
      np.sin(az) * np.sin(angular_radius_rad) * np.cos(tracker_lat_rad),
      np.cos(angular_radius_rad) - np.sin(tracker_lat_rad) * np.sin(lat_rad)
    )

    lat_circle.append(lat_rad * CONVERTER.DEG_PER_RAD)
    lon_circle.append(lon_rad * CONVERTER.DEG_PER_RAD)

  # Convert to arrays and ensure longitude wraps properly
  lat_circle = np.array(lat_circle)
  lon_circle = np.array(lon_circle)

  # Wrap longitude to [-180, 180]
  lon_circle = np.mod(lon_circle + 180, 360) - 180

  return lat_circle, lon_circle


def plot_ground_track(
  result                  : PropagationResult,
  epoch_dt_utc            : Optional[datetime] = None,
  title_text              : str = "Ground Track",
  trackers                : Optional[list['TrackerStation']] = None,
  include_tracker_on_body : bool = False,
) -> Figure:
  """
  Plot ground track with 3D globe (left) and 2D map projection (right).
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_delta_time'.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.
    title_text : str
      Base title for the plot.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the ground track plots (3D and 2D side by side).
  """
  fig = plt.figure(figsize=(22, 9))
  
  # Extract J2000 state vectors
  j2000_state   = result.state
  j2000_pos_vec = j2000_state[0:3, :]
  time_s        = result.time.grid.relative_initial
  n_points      = j2000_state.shape[1]
  
  # Convert epoch to ET
  if epoch_dt_utc is not None:
    epoch_et = utc_to_et(epoch_dt_utc)
  else:
    epoch_et = 0.0
  
  # Transform each position to body-fixed frame
  iau_earth_pos_vec = np.zeros((3, n_points))
  for i in range(n_points):
    epoch_et_i                 = epoch_et + time_s[i]
    rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(epoch_et_i)
    iau_earth_pos_vec[:, i]    = rot_mat_j2000_to_iau_earth @ j2000_pos_vec[:, i]
    
  # Compute geodetic coordinates
  geo_coords = GeographicCoordinateConverter.pos_to_geodetic_array(iau_earth_pos_vec)
  lat = geo_coords.latitude  * CONVERTER.DEG_PER_RAD
  lon = geo_coords.longitude * CONVERTER.DEG_PER_RAD
  
  # Handle longitude wrapping for plotting
  # Split trajectory at discontinuities (where lon jumps by more than 180 deg)
  lon_diff = np.abs(np.diff(lon))
  split_indices = np.where(lon_diff > 180)[0] + 1
  
  # Split into segments
  lat_segments = np.split(lat, split_indices)
  lon_segments = np.split(lon, split_indices)
  
  # Earth radius for 3D plotting (use equatorial radius for simplicity on globe)
  r_earth = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
  
  # ========== LEFT SUBPLOT: 3D Globe ==========
  ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
  
  # Draw Earth wireframe sphere (black)
  u = np.linspace(0, 2 * np.pi, 24)
  v = np.linspace(0, np.pi, 12)
  x_sphere = r_earth * np.outer(np.cos(u), np.sin(v))
  y_sphere = r_earth * np.outer(np.sin(u), np.sin(v))
  z_sphere = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
  ax_3d.plot_wireframe(x_sphere, y_sphere, z_sphere, color='black', linewidth=0.3, alpha=0.3, zorder=1)  # type: ignore
  
  # Draw coastlines on the 3D sphere
  coastline_segments = _get_coastline_coordinates()
  for lon_coast, lat_coast in coastline_segments:
    coast_pos_vec = GeographicCoordinateConverter.spherical_to_cartesian(
      lat_coast * CONVERTER.RAD_PER_DEG, 
      lon_coast * CONVERTER.RAD_PER_DEG, 
      r_earth * 1.001
    )  # Slightly above surface
    ax_3d.plot(coast_pos_vec[0], coast_pos_vec[1], coast_pos_vec[2], 'k-', linewidth=0.5, alpha=0.8, zorder=2)
  
  # Draw latitude/longitude grid lines
  grid_alpha = 0.2
  grid_color = 'gray'
  # Latitude lines (every 30 degrees)
  for lat_line in [-60, -30, 0, 30, 60]:
    lon_grid = np.linspace(-180, 180, 100)
    lat_grid = np.full_like(lon_grid, lat_line)
    grid_pos_vec = GeographicCoordinateConverter.spherical_to_cartesian(
      lat_grid * CONVERTER.RAD_PER_DEG, 
      lon_grid * CONVERTER.RAD_PER_DEG, 
      r_earth * 1.002
    )
    ax_3d.plot(grid_pos_vec[0], grid_pos_vec[1], grid_pos_vec[2], color=grid_color, linewidth=0.3, alpha=grid_alpha, zorder=1)
    
  # Longitude lines (every 30 degrees)
  for lon_line in range(-180, 180, 30):
    lat_grid = np.linspace(-90, 90, 50)
    lon_grid = np.full_like(lat_grid, lon_line)
    grid_pos_vec = GeographicCoordinateConverter.spherical_to_cartesian(
      lat_grid * CONVERTER.RAD_PER_DEG, 
      lon_grid * CONVERTER.RAD_PER_DEG, 
      r_earth * 1.002
    )
    ax_3d.plot(grid_pos_vec[0], grid_pos_vec[1], grid_pos_vec[2], color=grid_color, linewidth=0.3, alpha=grid_alpha, zorder=1)
  
  # Plot ground track on 3D sphere (projected to surface)
  x_track_all = []
  y_track_all = []
  z_track_all = []
  for lat_seg, lon_seg in zip(lat_segments, lon_segments):
    track_pos_vec = GeographicCoordinateConverter.spherical_to_cartesian(
      lat_seg * CONVERTER.RAD_PER_DEG, 
      lon_seg * CONVERTER.RAD_PER_DEG, 
      r_earth * 1.01
    )  # Slightly above surface
    ax_3d.plot(track_pos_vec[0], track_pos_vec[1], track_pos_vec[2], 'b-', linewidth=2.0, zorder=3)
    x_track_all.extend(track_pos_vec[0])
    y_track_all.extend(track_pos_vec[1])
    z_track_all.extend(track_pos_vec[2])
  x_track_all = np.array(x_track_all)
  y_track_all = np.array(y_track_all)
  z_track_all = np.array(z_track_all)
  
  # Mark start and end points on 3D globe
  start_pos_vec = GeographicCoordinateConverter.spherical_to_cartesian(lat[ 0] * CONVERTER.RAD_PER_DEG, lon[ 0] * CONVERTER.RAD_PER_DEG, r_earth * 1.02)
  end_pos_vec   = GeographicCoordinateConverter.spherical_to_cartesian(lat[-1] * CONVERTER.RAD_PER_DEG, lon[-1] * CONVERTER.RAD_PER_DEG, r_earth * 1.02)
  ax_3d.scatter([start_pos_vec[0]], [start_pos_vec[1]], [start_pos_vec[2]], s=400, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=4)  # type: ignore
  ax_3d.scatter([end_pos_vec[0]], [end_pos_vec[1]], [end_pos_vec[2]], s=400, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=4)  # type: ignore
  # Mark tracker locations on 3D globe (red dots) and show field of view
  if include_tracker_on_body and trackers is not None:
    for tracker in trackers:
      # Tracker position is stored in radians
      tracker_lat_rad = tracker.position.latitude
      tracker_lon_rad = tracker.position.longitude
      tracker_pos_vec = GeographicCoordinateConverter.spherical_to_cartesian(tracker_lat_rad, tracker_lon_rad, r_earth * 1.02)
      ax_3d.scatter([tracker_pos_vec[0]], [tracker_pos_vec[1]], [tracker_pos_vec[2]], s=200, c='red', marker='o', edgecolors='darkred', linewidths=2, zorder=5)  # type: ignore

      # Draw field of view hemisphere (use tracker's max range)
      tracker_lat_deg = tracker_lat_rad * CONVERTER.DEG_PER_RAD
      tracker_lon_deg = tracker_lon_rad * CONVERTER.DEG_PER_RAD
      fov_radius_m = tracker.performance.constraints.range.max
      x_fov, y_fov, z_fov = _create_tracker_fov_hemisphere(tracker_lat_deg, tracker_lon_deg, r_earth, fov_radius_m, resolution=30)
      ax_3d.plot_surface(x_fov, y_fov, z_fov, color='red', alpha=0.2, edgecolor='none', zorder=2)  # type: ignore

      # Draw FOV ground track circle on Earth's surface
      lat_circle, lon_circle = _calculate_hemisphere_ground_track(tracker_lat_deg, tracker_lon_deg, r_earth, fov_radius_m, num_points=100)
      circle_pos_vec = GeographicCoordinateConverter.spherical_to_cartesian(lat_circle * CONVERTER.RAD_PER_DEG, lon_circle * CONVERTER.RAD_PER_DEG, r_earth * 1.01)
      ax_3d.plot(circle_pos_vec[0], circle_pos_vec[1], circle_pos_vec[2], 'r--', linewidth=2, zorder=3)  # type: ignore

  # Set 3D axis properties
  ax_3d.set_xlabel('X [m]')
  ax_3d.set_ylabel('Y [m]')
  ax_3d.set_zlabel('Z [m]')  # type: ignore
  ax_3d.set_box_aspect([1, 1, 1])  # type: ignore
  
  # Set equal axis limits for sphere
  limit = r_earth * 1.8
  ax_3d.set_xlim([-limit, limit])
  ax_3d.set_ylim([-limit, limit])
  ax_3d.set_zlim([-limit, limit])  # type: ignore
  
  min_limit = -limit
  max_limit = limit
  
  # Add ground track shadow projections (darker gray)
  shadow_color = 'dimgray'
  shadow_alpha = 0.5
  shadow_lw = 0.5
  ax_3d.plot(x_track_all, y_track_all, np.full_like(z_track_all, min_limit), color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  ax_3d.plot(x_track_all, np.full_like(y_track_all, max_limit), z_track_all, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  ax_3d.plot(np.full_like(x_track_all, min_limit), y_track_all, z_track_all, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  
  # Add Earth projection shadows (filled circles on planes - lighter gray)
  r_disk = np.linspace(0, r_earth, 2)
  t_disk = np.linspace(0, 2*np.pi, 60)
  R_disk, T_disk = np.meshgrid(r_disk, t_disk)
  U_disk = R_disk * np.cos(T_disk)
  V_disk = R_disk * np.sin(T_disk)
  earth_shadow_alpha = 0.1
  
  ax_3d.plot_surface(U_disk, V_disk, np.full_like(U_disk, min_limit), color='gray', alpha=earth_shadow_alpha, shade=False)  # type: ignore
  ax_3d.plot_surface(U_disk, np.full_like(U_disk, max_limit), V_disk, color='gray', alpha=earth_shadow_alpha, shade=False)  # type: ignore
  ax_3d.plot_surface(np.full_like(U_disk, min_limit), U_disk, V_disk, color='gray', alpha=earth_shadow_alpha, shade=False)  # type: ignore
  
  # Set pane colors to white
  ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  
  # Set view angle for good visualization
  ax_3d.view_init(elev=20, azim=-60)  # type: ignore
  
  # ========== RIGHT SUBPLOT: 2D Map Projection ==========
  ax_2d = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
  ax_2d.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # type: ignore
  ax_2d.add_feature(cfeature.COASTLINE)  # type: ignore
  ax_2d.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)  # type: ignore
  gl = ax_2d.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')  # type: ignore
  gl.top_labels = False
  gl.right_labels = False
  
  # Plot each segment on 2D map
  plot_kwargs = {'transform': ccrs.PlateCarree()}
  for lat_seg, lon_seg in zip(lat_segments, lon_segments):
    ax_2d.plot(lon_seg, lat_seg, 'b-', linewidth=1.5, **plot_kwargs)
  
  # Mark start and end points on 2D map
  ax_2d.scatter([lon[ 0]], [lat[ 0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=5, label='Initial', **plot_kwargs)
  ax_2d.scatter([lon[-1]], [lat[-1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=5, label='Final', **plot_kwargs)

  # Mark tracker locations on 2D map (red dots) and show FOV ground tracks
  if include_tracker_on_body and trackers is not None:
    for tracker in trackers:
      # Tracker position is stored in radians, convert to degrees for plotting
      tracker_lat_deg = tracker.position.latitude * CONVERTER.DEG_PER_RAD
      tracker_lon_deg = tracker.position.longitude * CONVERTER.DEG_PER_RAD
      ax_2d.scatter([tracker_lon_deg], [tracker_lat_deg], s=200, c='red', marker='o', edgecolors='darkred', linewidths=2, zorder=6, **plot_kwargs)

      # Draw FOV ground track circle (use tracker's max range)
      fov_radius_m = tracker.performance.constraints.range.max
      r_earth = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
      lat_circle, lon_circle = _calculate_hemisphere_ground_track(tracker_lat_deg, tracker_lon_deg, r_earth, fov_radius_m, num_points=100)
      ax_2d.plot(lon_circle, lat_circle, 'r--', linewidth=2, zorder=5, **plot_kwargs)

  # Build info text for bottom of figure (consistent with 3D plots)
  info_text = "Frame: IAU_EARTH (Body-Fixed)"
  if epoch_dt_utc is not None:
    start_utc  = epoch_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time   = epoch_dt_utc + timedelta(seconds=time_s[-1])
    end_utc    = end_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    info_text += f"  |  Initial: {start_utc}  |  Final: {end_utc}"
  
  # Add info text as figure text at bottom (consistent with 3D plots)
  fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=11, color='black', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))
  
  # Set title using suptitle (will be overwritten by caller, but provides default)
  fig.suptitle(title_text, fontsize=16)
  
  plt.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))  # Leave space at bottom for info text and top for title
  return fig
