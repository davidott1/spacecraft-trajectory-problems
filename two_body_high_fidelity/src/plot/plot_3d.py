"""
3D trajectory plotting functions.

This module contains functions for plotting 3D trajectories in various
reference frames (J2000 inertial, IAU_EARTH body-fixed, heliocentric).
"""
import numpy             as np
import matplotlib.pyplot as plt
import spiceypy          as spice

from datetime          import datetime, timedelta
from typing            import Optional
from matplotlib.figure import Figure
from matplotlib.lines  import Line2D

from src.plot.utility          import get_equal_limits
from src.model.constants       import CONVERTER, SOLARSYSTEMCONSTANTS
from src.model.frame_converter import FrameConverter
from src.model.time_converter  import utc_to_et
from src.model.orbit_converter import GeographicCoordinateConverter, OrbitConverter
from src.schemas.propagation   import PropagationResult
from src.schemas.state         import TrackerStation


def project_to_bounds(origin, direction, ax):
  """
  Finds the intersection of a vector with the plot boundaries.
  """
  x_lim = ax.get_xlim()
  y_lim = ax.get_ylim()
  z_lim = ax.get_zlim()
  
  # 1. Determine which wall (min or max) the vector is pointing toward for each axis
  target_x = x_lim[1] if direction[0] >= 0 else x_lim[0]
  target_y = y_lim[1] if direction[1] >= 0 else y_lim[0]
  target_z = z_lim[1] if direction[2] >= 0 else z_lim[0]
  
  # 2. Calculate the scalar 't' required to hit that wall
  epsilon = 1e-9
  t_x = (target_x - origin[0]) / (direction[0] if abs(direction[0]) > epsilon else epsilon)
  t_y = (target_y - origin[1]) / (direction[1] if abs(direction[1]) > epsilon else epsilon)
  t_z = (target_z - origin[2]) / (direction[2] if abs(direction[2]) > epsilon else epsilon)
  
  # 3. The smallest positive 't' is the wall we hit first
  # We filter for positive t only (forward direction)
  ts = [t for t in [t_x, t_y, t_z] if t > 0]
  if not ts:
      return origin # Should not happen if direction is valid
  t_final = min(ts)
  
  # 4. Calculate the intersection point
  intersection = origin + direction * t_final
  
  return intersection


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


def plot_3d_trajectories(
  result : PropagationResult,
  epoch  : Optional[datetime] = None,
  frame  : str = "J2000",
) -> Figure:
  """
  Plot 3D position and velocity trajectories in a 1x2 grid.
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_delta_time'.
    epoch : datetime, optional
      Reference epoch (start time) for labeling.
    frame : str
      Reference frame name (default "J2000").
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the 3D plots.
  """
  fig = plt.figure(figsize=(18,10))
  
  # Extract state vectors
  posvel_vec = result.state
  pos_x, pos_y, pos_z = posvel_vec[0, :], posvel_vec[1, :], posvel_vec[2, :]
  vel_x, vel_y, vel_z = posvel_vec[3, :], posvel_vec[4, :], posvel_vec[5, :]
  
  # Build info string with frame and time if epoch is provided
  plot_delta_time = result.time_grid.deltas
  info_text = f"Frame: {frame}"
  if epoch is not None and plot_delta_time is not None:
    start_utc = epoch.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time  = epoch + timedelta(seconds=plot_delta_time[-1])
    end_utc   = end_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    info_text += f"  |  Initial: {start_utc}  |  Final: {end_utc}"
  
  # Plot 3D position trajectory
  ax1 = fig.add_subplot(121, projection='3d')
  
  # Add Earth
  u       = np.linspace(0, 2 * np.pi, 24)
  v       = np.linspace(0, np.pi, 12)
  r_eq    = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
  r_pol   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.POLAR
  x_earth = r_eq * np.outer(np.cos(u), np.sin(v))
  y_earth = r_eq * np.outer(np.sin(u), np.sin(v))
  z_earth = r_pol * np.outer(np.ones(np.size(u)), np.cos(v))
  ax1.plot_wireframe(x_earth, y_earth, z_earth, color='black', linewidth=0.5, alpha=1.0) # type: ignore

  ax1.plot(pos_x, pos_y, pos_z, 'b-', linewidth=2.0)
  ax1.scatter([pos_x[ 0]], [pos_y[ 0]], [pos_z[ 0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2)  # type: ignore
  ax1.scatter([pos_x[-1]], [pos_y[-1]], [pos_z[-1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2)  # type: ignore
  ax1.set_xlabel('Pos-X [m]')
  ax1.set_ylabel('Pos-Y [m]')
  ax1.set_zlabel('Pos-Z [m]') # type: ignore
  ax1.grid(True)
  ax1.set_box_aspect([1,1,1]) # type: ignore

  # Set pane colors to white
  ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore

  min_limit, max_limit = get_equal_limits(ax1, buffer_fraction=0.45)
  
  ax1.set_xlim([min_limit, max_limit]) # type: ignore
  ax1.set_ylim([min_limit, max_limit]) # type: ignore
  ax1.set_zlim([min_limit, max_limit]) # type: ignore

  # Add position trajectory shadows (projections onto planes)
  shadow_color = 'gray'
  shadow_alpha = 0.75
  shadow_lw    = 0.5
  # XY plane shadow (z = min_limit)
  ax1.plot(pos_x, pos_y, np.full_like(pos_z, min_limit), color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  # XZ plane shadow (y = max_limit)
  ax1.plot(pos_x, np.full_like(pos_y, max_limit), pos_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  # YZ plane shadow (x = min_limit)
  ax1.plot(np.full_like(pos_x, min_limit), pos_y, pos_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)

  # Add Earth projection shadows (filled circles on planes)
  r_disk = np.linspace(0, r_eq, 2)
  t_disk = np.linspace(0, 2*np.pi, 60)
  R_disk, T_disk = np.meshgrid(r_disk, t_disk)
  U_disk = R_disk * np.cos(T_disk)
  V_disk = R_disk * np.sin(T_disk)
  earth_shadow_alpha = 0.1

  # XY plane (z = min_limit)
  ax1.plot_surface(U_disk, V_disk, np.full_like(U_disk, min_limit), color='black', alpha=earth_shadow_alpha, shade=False)  # type: ignore 
  # XZ plane (y = max_limit)
  ax1.plot_surface(U_disk, np.full_like(U_disk, max_limit), V_disk, color='black', alpha=earth_shadow_alpha, shade=False)  # type: ignore
  # YZ plane (x = min_limit)
  ax1.plot_surface(np.full_like(U_disk, min_limit), U_disk, V_disk, color='black', alpha=earth_shadow_alpha, shade=False)  # type: ignore

  # Add sun direction markers on box walls (only if we have epoch and are in J2000 frame)
  if epoch is not None and frame == "J2000" and plot_delta_time is not None:
    try:
      # Get start and end times
      epoch_et_start = utc_to_et(epoch)
      epoch_et_end = epoch_et_start + plot_delta_time[-1]
      
      # Get sun position at start time (returns in km)
      sun_pos_start_km, _ = spice.spkpos('SUN', epoch_et_start, 'J2000', 'NONE', 'EARTH')
      sun_dir_start = sun_pos_start_km / np.linalg.norm(sun_pos_start_km)
      
      # Get sun position at end time (returns in km)
      sun_pos_end_km, _ = spice.spkpos('SUN', epoch_et_end, 'J2000', 'NONE', 'EARTH')
      sun_dir_end = sun_pos_end_km / np.linalg.norm(sun_pos_end_km)
      
      # Get moon position at start time (returns in km)
      moon_pos_start_km, _ = spice.spkpos('MOON', epoch_et_start, 'J2000', 'NONE', 'EARTH')
      moon_dir_start = moon_pos_start_km / np.linalg.norm(moon_pos_start_km)
      
      # Get moon position at end time (returns in km)
      moon_pos_end_km, _ = spice.spkpos('MOON', epoch_et_end, 'J2000', 'NONE', 'EARTH')
      moon_dir_end = moon_pos_end_km / np.linalg.norm(moon_pos_end_km)
      
      origin = np.array([0, 0, 0])  # Earth center
      z_floor = min_limit

      # --- SUN MARKERS ---
      sun_marker_pos_start = project_to_bounds(origin, sun_dir_start, ax1)
      ax1.scatter([sun_marker_pos_start[0]], [sun_marker_pos_start[1]], [sun_marker_pos_start[2]], s=600, marker=r'$\odot_{\text{o}}$', color='gold', zorder=10, label='Sun (Initial)')  # type: ignore
      ax1.plot([sun_marker_pos_start[0], sun_marker_pos_start[0]], [sun_marker_pos_start[1], sun_marker_pos_start[1]], [sun_marker_pos_start[2], z_floor], color='gold', linestyle=':', linewidth=2, alpha=1.0)  # type: ignore
      ax1.scatter([sun_marker_pos_start[0]], [sun_marker_pos_start[1]], [z_floor], color='gold', marker='.', s=20, alpha=1.0)  # type: ignore

      sun_marker_pos_end = project_to_bounds(origin, sun_dir_end, ax1)
      ax1.scatter([sun_marker_pos_end[0]], [sun_marker_pos_end[1]], [sun_marker_pos_end[2]], s=600, marker=r'$\odot_{\text{f}}$', color='gold', zorder=10, label='Sun (Final)')  # type: ignore
      ax1.plot([sun_marker_pos_end[0], sun_marker_pos_end[0]], [sun_marker_pos_end[1], sun_marker_pos_end[1]], [sun_marker_pos_end[2], z_floor], color='gold', linestyle=':', linewidth=2, alpha=1.0)  # type: ignore
      ax1.scatter([sun_marker_pos_end[0]], [sun_marker_pos_end[1]], [z_floor], color='gold', marker='.', s=20, alpha=1.0)  # type: ignore

      # --- MOON MARKERS ---
      moon_marker_pos_start = project_to_bounds(origin, moon_dir_start, ax1)
      ax1.scatter([moon_marker_pos_start[0]], [moon_marker_pos_start[1]], [moon_marker_pos_start[2]], s=600, marker=r'$☾_{\text{o}}$', color='gray', zorder=10, label='Moon (Initial)')  # type: ignore
      ax1.plot([moon_marker_pos_start[0], moon_marker_pos_start[0]], [moon_marker_pos_start[1], moon_marker_pos_start[1]], [moon_marker_pos_start[2], z_floor], color='gray', linestyle=':', linewidth=2, alpha=0.5)  # type: ignore
      ax1.scatter([moon_marker_pos_start[0]], [moon_marker_pos_start[1]], [z_floor], color='gray', marker='.', s=20, alpha=0.5)  # type: ignore

      moon_marker_pos_end = project_to_bounds(origin, moon_dir_end, ax1)
      ax1.scatter([moon_marker_pos_end[0]], [moon_marker_pos_end[1]], [moon_marker_pos_end[2]], s=600, marker=r'$☾_{\text{f}}$', color='gray', zorder=10, label='Moon (Final)')  # type: ignore
      ax1.plot([moon_marker_pos_end[0], moon_marker_pos_end[0]], [moon_marker_pos_end[1], moon_marker_pos_end[1]], [moon_marker_pos_end[2], z_floor], color='gray', linestyle=':', linewidth=2, alpha=0.5)  # type: ignore
      ax1.scatter([moon_marker_pos_end[0]], [moon_marker_pos_end[1]], [z_floor], color='gray', marker='.', s=20, alpha=0.5)  # type: ignore
      
    except Exception as e:
      # If SPICE kernels aren't loaded or other error, silently skip sun arrow
      pass

  # Plot 3D velocity trajectory
  ax2 = fig.add_subplot(122, projection='3d')
  ax2.plot(vel_x, vel_y, vel_z, 'b-', linewidth=2.0)
  ax2.scatter([vel_x[ 0]], [vel_y[ 0]], [vel_z[ 0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2) # type: ignore
  ax2.scatter([vel_x[-1]], [vel_y[-1]], [vel_z[-1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2) # type: ignore
  ax2.set_xlabel('Vel-X [m/s]')
  ax2.set_ylabel('Vel-Y [m/s]')
  ax2.set_zlabel('Vel-Z [m/s]') # type: ignore
  ax2.grid(True)
  ax2.set_box_aspect([1,1,1]) # type: ignore

  # Set pane colors to white
  ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore

  min_limit_vel, max_limit_vel = get_equal_limits(ax2, buffer_fraction=0.45)
  
  ax2.set_xlim([min_limit_vel, max_limit_vel])  # type: ignore
  ax2.set_ylim([min_limit_vel, max_limit_vel])  # type: ignore
  ax2.set_zlim([min_limit_vel, max_limit_vel])  # type: ignore

  # Add velocity trajectory shadows (projections onto planes)
  # XY plane shadow (z = min_limit_vel)
  ax2.plot(vel_x, vel_y, np.full_like(vel_z, min_limit_vel), color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  # XZ plane shadow (y = max_limit_vel)
  ax2.plot(vel_x, np.full_like(vel_y, max_limit_vel), vel_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  # YZ plane shadow (x = min_limit_vel)
  ax2.plot(np.full_like(vel_x, min_limit_vel), vel_y, vel_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)

  # Create custom legend handles (lines only, no markers)
  legend_handles = [
    Line2D([0], [0], color='black', linewidth=1.5, label='Earth'),
    Line2D([0], [0], color='b', linewidth=2.0, label='Spacecraft'),
    Line2D([0], [0], color='gold', linewidth=1.5, label='Sun'),
    Line2D([0], [0], color='gray', linewidth=1.5, label='Moon'),
  ]
  leg = fig.legend(handles=legend_handles, loc='upper right', fontsize=11, framealpha=0.9)
  leg.get_frame().set_edgecolor('black')

  # Add info text as figure text
  fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=11, color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

  plt.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))  # Leave space at bottom for info text and top for legend
  return fig


def plot_3d_trajectories_body_fixed(
  result                  : PropagationResult,
  epoch_dt_utc            : Optional[datetime] = None,
  trackers                : Optional[list['TrackerStation']] = None,
  include_tracker_on_body : bool = False,
) -> Figure:
  """
  Plot 3D position and velocity trajectories in body-fixed (IAU_EARTH) frame.
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_delta_time'.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the 3D plots.
  """
  fig = plt.figure(figsize=(18, 10))
  
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
  
  # Transform each position and velocity to body-fixed frame
  iau_earth_pos_vec = np.zeros((3, n_points))
  iau_earth_vel_vec = np.zeros((3, n_points))
  for i in range(n_points):
    epoch_et_i = epoch_et + time_s[i]
    rot_mat_j2000_to_iau_earth = FrameConverter.j2000_to_iau_earth(epoch_et_i)
    iau_earth_pos_vec[:, i] = rot_mat_j2000_to_iau_earth @ j2000_pos_vec[:, i]
    iau_earth_vel_vec[:, i] = rot_mat_j2000_to_iau_earth @ j2000_vel_vec[:, i]
  
  pos_x, pos_y, pos_z = iau_earth_pos_vec[0, :], iau_earth_pos_vec[1, :], iau_earth_pos_vec[2, :]
  vel_x, vel_y, vel_z = iau_earth_vel_vec[0, :], iau_earth_vel_vec[1, :], iau_earth_vel_vec[2, :]
  
  # Build info string
  info_text = "Frame: IAU_EARTH (Body-Fixed)"
  if epoch_dt_utc is not None:
    start_time_iso_utc = epoch_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time_dt_utc    = epoch_dt_utc + timedelta(seconds=time_s[-1])
    end_time_iso_utc   = end_time_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    info_text += f"  |  Initial: {start_time_iso_utc}  |  Final: {end_time_iso_utc}"
  
  # Plot 3D position trajectory
  ax1 = fig.add_subplot(121, projection='3d')
  
  # Add Earth wireframe ellipsoid
  u       = np.linspace(0, 2 * np.pi, 24)
  v       = np.linspace(0, np.pi, 12)
  r_eq    = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
  r_pol   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.POLAR
  x_earth = r_eq * np.outer(np.cos(u), np.sin(v))
  y_earth = r_eq * np.outer(np.sin(u), np.sin(v))
  z_earth = r_pol * np.outer(np.ones(np.size(u)), np.cos(v))
  ax1.plot_wireframe(x_earth, y_earth, z_earth, color='black', linewidth=0.5, alpha=1.0)  # type: ignore
  
  ax1.plot(pos_x, pos_y, pos_z, 'b-', linewidth=2.0)
  ax1.scatter([pos_x[ 0]], [pos_y[ 0]], [pos_z[ 0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2)  # type: ignore
  ax1.scatter([pos_x[-1]], [pos_y[-1]], [pos_z[-1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2)  # type: ignore

  # Mark tracker locations on 3D body-fixed plot (red dots) and show field of view
  if include_tracker_on_body and trackers is not None:
    for tracker in trackers:
      # Tracker position is stored in radians
      tracker_lat_rad = tracker.position.latitude
      tracker_lon_rad = tracker.position.longitude
      # Use Earth's equatorial radius for consistency
      tracker_pos_vec = GeographicCoordinateConverter.spherical_to_cartesian(tracker_lat_rad, tracker_lon_rad, r_eq * 1.02)
      ax1.scatter([tracker_pos_vec[0]], [tracker_pos_vec[1]], [tracker_pos_vec[2]], s=200, c='red', marker='o', edgecolors='darkred', linewidths=2, zorder=5)  # type: ignore

      # Draw field of view hemisphere (use tracker's max range)
      tracker_lat_deg = tracker_lat_rad * CONVERTER.DEG_PER_RAD
      tracker_lon_deg = tracker_lon_rad * CONVERTER.DEG_PER_RAD
      fov_radius_m = tracker.performance.constraints.range.max
      x_fov, y_fov, z_fov = _create_tracker_fov_hemisphere(tracker_lat_deg, tracker_lon_deg, r_eq, fov_radius_m, resolution=30)
      ax1.plot_surface(x_fov, y_fov, z_fov, color='red', alpha=0.2, edgecolor='none', zorder=2)  # type: ignore

  ax1.set_xlabel('Pos-X [m]')
  ax1.set_ylabel('Pos-Y [m]')
  ax1.set_zlabel('Pos-Z [m]') # type: ignore
  ax1.grid(True)
  ax1.set_box_aspect([1,1,1]) # type: ignore

  # Set pane colors to white
  ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore

  min_limit, max_limit = get_equal_limits(ax1, buffer_fraction=0.45)
  
  ax1.set_xlim((min_limit, max_limit))
  ax1.set_ylim((min_limit, max_limit))
  ax1.set_zlim((min_limit, max_limit))  # type: ignore

  # Add position trajectory shadows
  shadow_color = 'gray'
  shadow_alpha = 0.75
  shadow_lw    = 0.5
  ax1.plot(pos_x, pos_y, np.full_like(pos_z, min_limit), color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  ax1.plot(pos_x, np.full_like(pos_y, max_limit), pos_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  ax1.plot(np.full_like(pos_x, min_limit), pos_y, pos_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)

  # Add Earth projection shadows (filled circles on planes)
  r_disk = np.linspace(0, r_eq, 2)
  t_disk = np.linspace(0, 2*np.pi, 60)
  R_disk, T_disk = np.meshgrid(r_disk, t_disk)
  U_disk = R_disk * np.cos(T_disk)
  V_disk = R_disk * np.sin(T_disk)
  earth_shadow_alpha = 0.1

  ax1.plot_surface(U_disk, V_disk, np.full_like(U_disk, min_limit), color='black', alpha=earth_shadow_alpha, shade=False)  # type: ignore
  ax1.plot_surface(U_disk, np.full_like(U_disk, max_limit), V_disk, color='black', alpha=earth_shadow_alpha, shade=False)  # type: ignore
  ax1.plot_surface(np.full_like(U_disk, min_limit), U_disk, V_disk, color='black', alpha=earth_shadow_alpha, shade=False)  # type: ignore

  # Plot 3D velocity trajectory
  ax2 = fig.add_subplot(122, projection='3d')
  ax2.plot(vel_x, vel_y, vel_z, 'b-', linewidth=2.0)
  ax2.scatter([vel_x[ 0]], [vel_y[ 0]], [vel_z[ 0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2)  # type: ignore
  ax2.scatter([vel_x[-1]], [vel_y[-1]], [vel_z[-1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2)  # type: ignore
  ax2.set_xlabel('Vel-X [m/s]')
  ax2.set_ylabel('Vel-Y [m/s]')
  ax2.set_zlabel('Vel-Z [m/s]') # type: ignore
  ax2.grid(True)
  ax2.set_box_aspect([1,1,1]) # type: ignore

  # Set pane colors to white
  ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore
  ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # type: ignore

  min_limit_vel, max_limit_vel = get_equal_limits(ax2, buffer_fraction=0.45)
  
  ax2.set_xlim((min_limit_vel, max_limit_vel))
  ax2.set_ylim((min_limit_vel, max_limit_vel))
  ax2.set_zlim((min_limit_vel, max_limit_vel)) # type: ignore

  # Add velocity trajectory shadows
  ax2.plot(vel_x, vel_y, np.full_like(vel_z, min_limit_vel), color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  ax2.plot(vel_x, np.full_like(vel_y, max_limit_vel), vel_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  ax2.plot(np.full_like(vel_x, min_limit_vel), vel_y, vel_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)

  # Create custom legend handles (lines only, no markers)
  legend_handles = [
    Line2D([0], [0], color='black', linewidth=1.5, label='Earth'),
    Line2D([0], [0], color='b', linewidth=2.0, label='Spacecraft'),
  ]
  leg = fig.legend(handles=legend_handles, loc='upper right', fontsize=11, framealpha=0.9)
  leg.get_frame().set_edgecolor('black')

  # Add info text as figure text
  fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=11, color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

  plt.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
  return fig


def plot_3d_trajectory_sun_centered(
  result : PropagationResult,
  epoch  : Optional[datetime] = None,
) -> Figure:
  """
  Plot 3D position trajectory with Moon trajectory in J2000 Earth-centered frame.
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_delta_time'.
    epoch : datetime, optional
      Reference epoch (start time) for labeling and Moon position computation.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the 3D plot.
  """
  fig = plt.figure(figsize=(18, 10))
  
  # Extract state vectors
  posvel_vec = result.state
  time_s     = result.time_grid.deltas
  
  # Build info string
  info_text = "Frame: J2000 - Sun-Centered"
  if epoch is not None and time_s is not None:
    start_utc  = epoch.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time   = epoch + timedelta(seconds=time_s[-1])
    end_utc    = end_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    info_text += f"  |  Initial: {start_utc}  |  Final: {end_utc}"
  
  # Create subplots in a 1x2 grid
  ax_sun_dup = fig.add_subplot(121, projection='3d')
  ax_sun     = fig.add_subplot(122, projection='3d')
  
  # Pre-calculate Moon and Sun data if epoch is available
  moon_pos        = None
  n_moon_points   = 0
  moon_orbit_full = None
  
  sun_pos        = None
  n_sun_points   = 0
  sun_orbit_full = None
  
  # Heliocentric data
  earth_orbit_helio     = None
  earth_pos_helio_init  = None
  earth_pos_helio_final = None
  earth_pos_helio       = None
  sc_pos_helio          = None
  moon_pos_helio        = None
  
  if epoch is not None:
    try:
      epoch_et_start = utc_to_et(epoch)
      
      # --- HELIOCENTRIC EARTH ORBIT ---
      # Get Earth state relative to Sun at start
      earth_state_sun, _   = spice.spkezr('EARTH', epoch_et_start, 'J2000', 'NONE', 'SUN')
      earth_pos_helio_init = np.array(earth_state_sun)[0:3] * CONVERTER.M_PER_KM  # m
      earth_vel_helio_init = np.array(earth_state_sun)[3:6] * CONVERTER.M_PER_KM  # m/s
      
      # Get Earth state relative to Sun at end
      epoch_et_end           = epoch_et_start + time_s[-1]
      earth_state_sun_end, _ = spice.spkezr('EARTH', epoch_et_end, 'J2000', 'NONE', 'SUN')
      earth_pos_helio_final  = np.array(earth_state_sun_end)[0:3] * CONVERTER.M_PER_KM  # m

      # Calculate full orbit based on initial osculating elements
      try:
        mu_sun = SOLARSYSTEMCONSTANTS.SUN.GP
      except AttributeError:
        mu_sun = 1.32712440018e20

      earth_coe = OrbitConverter.pv_to_coe(earth_pos_helio_init, earth_vel_helio_init, mu_sun)
      
      n_orbit_points    = 200
      ta_vals           = np.linspace(0, 2*np.pi, n_orbit_points)
      earth_orbit_helio = np.zeros((3, n_orbit_points))
      
      for i, ta in enumerate(ta_vals):
        earth_coe.ta = ta
        r_vec, _ = OrbitConverter.coe_to_pv(earth_coe, mu_sun)
        earth_orbit_helio[:, i] = r_vec
      
      # --- SPACECRAFT & MOON HELIOCENTRIC TRAJECTORY ---
      # Calculate spacecraft position relative to Sun
      # Downsample for performance
      stride          = max(1, len(time_s) // 500)
      indices         = range(0, len(time_s), stride)
      sc_pos_helio    = np.zeros((3, len(indices)))
      moon_pos_helio  = np.zeros((3, len(indices)))
      earth_pos_helio = np.zeros((3, len(indices)))
      
      for i, idx in enumerate(indices):
        t  = time_s[idx]
        et = epoch_et_start + t
        # Earth relative to Sun
        pos_earth_sun_km, _ = spice.spkpos('EARTH', et, 'J2000', 'NONE', 'SUN')
        pos_earth_sun_m = np.array(pos_earth_sun_km) * 1000.0
        earth_pos_helio[:, i] = pos_earth_sun_m
        
        # SC relative to Earth (from result)
        pos_sc_earth_m = posvel_vec[0:3, idx]
        
        # SC relative to Sun
        sc_pos_helio[:, i] = pos_earth_sun_m + pos_sc_earth_m

        # Moon relative to Earth
        pos_moon_earth_km, _ = spice.spkpos('MOON', et, 'J2000', 'NONE', 'EARTH')
        pos_moon_earth_m = np.array(pos_moon_earth_km) * 1000.0

        # Moon relative to Sun
        moon_pos_helio[:, i] = pos_earth_sun_m + pos_moon_earth_m

      # --- MOON ---
      # Moon trajectory during simulation
      n_moon_points = min(len(time_s), 500)  # Limit Moon points for performance
      moon_time_indices = np.linspace(0, len(time_s) - 1, n_moon_points, dtype=int)
      
      moon_pos = np.zeros((3, n_moon_points))
      for i, idx in enumerate(moon_time_indices):
        epoch_et_i = epoch_et_start + time_s[idx]
        # Get Moon position relative to Earth in J2000 (returns in km)
        moon_pos_km, _ = spice.spkpos('MOON', epoch_et_i, 'J2000', 'NONE', 'EARTH')
        moon_pos[:, i] = np.array(moon_pos_km) * 1000.0  # Convert to meters

      # Full approximate Moon orbit (Keplerian ellipse based on initial state)
      # Get initial state of Moon (km, km/s)
      moon_state_km, _ = spice.spkezr('MOON', epoch_et_start, 'J2000', 'NONE', 'EARTH')
      moon_state_arr = np.array(moon_state_km)
      moon_pos_init = moon_state_arr[0:3] * CONVERTER.M_PER_KM  # m
      moon_vel_init = moon_state_arr[3:6] * CONVERTER.M_PER_KM  # m/s
      
      # Convert to orbital elements
      moon_coe = OrbitConverter.pv_to_coe(moon_pos_init, moon_vel_init, SOLARSYSTEMCONSTANTS.EARTH.GP)
      
      # Generate full orbit points
      n_orbit_points  = 200
      ta_vals         = np.linspace(0, 2*np.pi, n_orbit_points)
      moon_orbit_full = np.zeros((3, n_orbit_points))
      
      for i, ta in enumerate(ta_vals):
        # Update TA in COE
        moon_coe.ta = ta

        # Convert back to PV
        r_vec, _ = OrbitConverter.coe_to_pv(moon_coe, SOLARSYSTEMCONSTANTS.EARTH.GP)
        moon_orbit_full[:, i] = r_vec

      # --- SUN ---
      # 1. Sun trajectory during simulation
      n_sun_points     = min(len(time_s), 500)
      sun_time_indices = np.linspace(0, len(time_s) - 1, n_sun_points, dtype=int)
      
      sun_pos = np.zeros((3, n_sun_points))
      for i, idx in enumerate(sun_time_indices):
        epoch_et_i = epoch_et_start + time_s[idx]
        sun_pos_km, _ = spice.spkpos('SUN', epoch_et_i, 'J2000', 'NONE', 'EARTH')
        sun_pos[:, i] = np.array(sun_pos_km) * 1000.0
        
      # 2. Full approximate Sun orbit
      # Use Sun's GP for relative motion (mu_sun + mu_earth approx mu_sun)
      sun_state_km, _ = spice.spkezr('SUN', epoch_et_start, 'J2000', 'NONE', 'EARTH')
      sun_pos_init    = np.array(sun_state_km)[0:3] * 1000.0
      sun_vel_init    = np.array(sun_state_km)[3:6] * 1000.0
      
      sun_coe = OrbitConverter.pv_to_coe(sun_pos_init, sun_vel_init, mu_sun)
      
      sun_orbit_full = np.zeros((3, n_orbit_points))
      for i, ta in enumerate(ta_vals):
        sun_coe.ta = ta
        r_vec, _ = OrbitConverter.coe_to_pv(sun_coe, mu_sun)
        sun_orbit_full[:, i] = r_vec
      
    except Exception:
      # If SPICE kernels aren't loaded or other error, skip Moon/Sun trajectory
      pass

  # Plot Heliocentric view (ax_sun and ax_sun_dup)
  for ax in [ax_sun, ax_sun_dup]:
    if ax == ax_sun:
      ax.set_title("Magnification")
    
    # Plot Sun (Only on middle plot)
    if ax == ax_sun_dup:
      # Sun wireframe
      r_sun = 696340000.0 # meters
      u_sun = np.linspace(0, 2 * np.pi, 24)
      v_sun = np.linspace(0, np.pi, 12)
      x_sun = r_sun * np.outer(np.cos(u_sun), np.sin(v_sun))
      y_sun = r_sun * np.outer(np.sin(u_sun), np.sin(v_sun))
      z_sun = r_sun * np.outer(np.ones(np.size(u_sun)), np.cos(v_sun))
      ax.plot_wireframe(x_sun, y_sun, z_sun, color='orange', linewidth=0.5, alpha=0.8) # type: ignore
    
    if earth_orbit_helio is not None:
      # Plot Orbit (Only on middle plot)
      if ax == ax_sun_dup:
        ax.plot(earth_orbit_helio[0, :], earth_orbit_helio[1, :], earth_orbit_helio[2, :], color='black', linestyle='--', linewidth=1, alpha=0.6, label='Earth Orbit')
      
      # Plot Earth Trajectory
      if earth_pos_helio is not None:
        ax.plot(earth_pos_helio[0, :], earth_pos_helio[1, :], earth_pos_helio[2, :], color='black', linewidth=2.0, alpha=0.8, label='Earth')
      
      # Plot Earth Initial
      if earth_pos_helio_init is not None:
        ax.scatter([earth_pos_helio_init[0]], [earth_pos_helio_init[1]], [earth_pos_helio_init[2]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='black', linewidths=2, zorder=5)  # type: ignore
                    
      # Plot Earth Final
      if earth_pos_helio_final is not None:
        ax.scatter([earth_pos_helio_final[0]], [earth_pos_helio_final[1]], [earth_pos_helio_final[2]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='black', linewidths=2, zorder=5)  # type: ignore
      
      if ax == ax_sun_dup and moon_pos_helio is not None:
        # Plot Moon Trajectory
        ax.plot(moon_pos_helio[0, :], moon_pos_helio[1, :], moon_pos_helio[2, :], color='gray', linewidth=1.0, alpha=0.8, label='Moon')
        # Add Moon markers
        ax.scatter([moon_pos_helio[0,  0]], [moon_pos_helio[1,  0]], [moon_pos_helio[2,  0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='gray', linewidths=2, zorder=8)  # type: ignore
        ax.scatter([moon_pos_helio[0, -1]], [moon_pos_helio[1, -1]], [moon_pos_helio[2, -1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='gray', linewidths=2, zorder=8)  # type: ignore

      if sc_pos_helio is not None:
        # Plot Spacecraft Trajectory
        ax.plot(sc_pos_helio[0, :], sc_pos_helio[1, :], sc_pos_helio[2, :], color='b', linewidth=1.5, label='Spacecraft')
        ax.scatter([sc_pos_helio[0,  0]], [sc_pos_helio[1,  0]], [sc_pos_helio[2,  0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=10)  # type: ignore
        ax.scatter([sc_pos_helio[0, -1]], [sc_pos_helio[1, -1]], [sc_pos_helio[2, -1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=10)  # type: ignore

      ax.set_xlabel('Pos-X [m]')
      ax.set_ylabel('Pos-Y [m]')
      ax.set_zlabel('Pos-Z [m]')  # type: ignore
      ax.grid(True)
      
      # Custom Legend
      if ax == ax_sun_dup:
        legend_handles_sun = []
        legend_handles_sun.append(Line2D([0], [0], color='orange',                 linewidth=1.5, label='Sun'        ))
        legend_handles_sun.append(Line2D([0], [0], color='black' , linestyle='--', linewidth=1  , label='Earth Orbit'))
        legend_handles_sun.append(Line2D([0], [0], color='black' ,                 linewidth=2.0, label='Earth'      ))
        legend_handles_sun.extend([
          Line2D([0], [0], color='gray', linewidth=1, label='Moon'),
          Line2D([0], [0], color='b', linewidth=1.5, label='Spacecraft'),
        ])

        leg_sun = ax.legend(handles=legend_handles_sun, loc='upper right', fontsize=10, framealpha=0.9)
        leg_sun.get_frame().set_edgecolor('black')
      
      # Set pane colors to white
      ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # type: ignore
      ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # type: ignore
      ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # type: ignore
      
      # Set limits for heliocentric plot (Variable limits with equal scaling)
      # Collect all data points
      xs_sun = []
      ys_sun = []
      zs_sun = []
      
      # For bottom-left plot (ax_sun), center on spacecraft trajectory only
      if ax == ax_sun:
        if sc_pos_helio is not None:
          xs_sun.extend(sc_pos_helio[0, :])
          ys_sun.extend(sc_pos_helio[1, :])
          zs_sun.extend(sc_pos_helio[2, :])
      else:
        # Include Sun at origin only for middle plot
        if ax == ax_sun_dup:
          xs_sun.append(0.0)
          ys_sun.append(0.0)
          zs_sun.append(0.0)
        
        if earth_orbit_helio is not None:
          # Include orbit line only for middle plot
          if ax == ax_sun_dup:
            xs_sun.extend(earth_orbit_helio[0, :])
            ys_sun.extend(earth_orbit_helio[1, :])
            zs_sun.extend(earth_orbit_helio[2, :])
          
          # Include Earth trajectory segment for limits
          if earth_pos_helio is not None:
            xs_sun.extend(earth_pos_helio[0, :])
            ys_sun.extend(earth_pos_helio[1, :])
            zs_sun.extend(earth_pos_helio[2, :])
          
          # Include Earth positions for both plots (since markers are plotted)
          if earth_pos_helio_init is not None:
            xs_sun.append(earth_pos_helio_init[0])
            ys_sun.append(earth_pos_helio_init[1])
            zs_sun.append(earth_pos_helio_init[2])
          
          if earth_pos_helio_final is not None:
            xs_sun.append(earth_pos_helio_final[0])
            ys_sun.append(earth_pos_helio_final[1])
            zs_sun.append(earth_pos_helio_final[2])
            
        if sc_pos_helio is not None:
          xs_sun.extend(sc_pos_helio[0, :])
          ys_sun.extend(sc_pos_helio[1, :])
          zs_sun.extend(sc_pos_helio[2, :])

        if moon_pos_helio is not None:
          xs_sun.extend(moon_pos_helio[0, :])
          ys_sun.extend(moon_pos_helio[1, :])
          zs_sun.extend(moon_pos_helio[2, :])

      # Convert to numpy arrays for min/max
      if len(xs_sun) > 0:
        all_x_sun = np.array(xs_sun)
        all_y_sun = np.array(ys_sun)
        all_z_sun = np.array(zs_sun)
        
        x_min, x_max = np.min(all_x_sun), np.max(all_x_sun)
        y_min, y_max = np.min(all_y_sun), np.max(all_y_sun)
        z_min, z_max = np.min(all_z_sun), np.max(all_z_sun)
        
        # Add buffer
        buffer = 0.1
        dx = x_max - x_min
        dy = y_max - y_min
        dz = z_max - z_min
        
        # Avoid zero range
        if dx == 0: dx = 1.0
        if dy == 0: dy = 1.0
        if dz == 0: dz = 1.0
        
        ax.set_xlim((x_min - buffer*dx, x_max + buffer*dx))
        ax.set_ylim((y_min - buffer*dy, y_max + buffer*dy))
        ax.set_zlim((z_min - buffer*dz, z_max + buffer*dz)) # type: ignore
        
        # Set box aspect to match data range ratios (maintains equal scale)
        ax.set_box_aspect((float(dx), float(dy), float(dz))) # type: ignore

  # Add info text
  fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=10, color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

  plt.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
  return fig


def plot_3d_error(
  result_ref  : PropagationResult,
  result_comp : PropagationResult,
  title      : str = "Position and Velocity Error",
) -> Figure:
  """
  Plot 3D position and velocity error trajectories in a 1x2 grid.
  
  Input:
  ------
    result_ref : PropagationResult
      Reference result (e.g., SGP4).
    result_comp : PropagationResult
      Comparison result (e.g., high-fidelity).
    title : str
      Plot title.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the 3D error plots.
  """
  fig = plt.figure(figsize=(18,10))
  
  # Interpolate comparison result to reference time points
  from scipy.interpolate import interp1d

  time_ref   = result_ref.time_grid.deltas
  time_comp  = result_comp.time_grid.deltas
  state_comp = result_comp.state
  
  # Interpolate each state component
  state_comp_interp = np.zeros((6, len(time_ref)))
  for i in range(6):
    interpolator = interp1d(time_comp, state_comp[i, :], kind='cubic', fill_value='extrapolate') # type: ignore
    state_comp_interp[i, :] = interpolator(time_ref)
  
  # Calculate errors (comparison - reference)
  state_ref = result_ref.state
  pos_error = state_comp_interp[0:3, :] - state_ref[0:3, :]
  vel_error = state_comp_interp[3:6, :] - state_ref[3:6, :]
  
  # Plot 3D position error
  ax1 = fig.add_subplot(121, projection='3d')
  ax1.plot(pos_error[0, :], pos_error[1, :], pos_error[2, :], 'b-', linewidth=1)
  ax1.scatter([pos_error[0,  0]], [pos_error[1,  0]], [pos_error[2,  0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2, label='Initial') # type: ignore
  ax1.scatter([pos_error[0, -1]], [pos_error[1, -1]], [pos_error[2, -1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2, label='Final'  ) # type: ignore
  ax1.set_xlabel('Error X [m]')
  ax1.set_ylabel('Error Y [m]')
  ax1.set_zlabel('Error Z [m]') # type: ignore
  ax1.set_title('Position Error')
  ax1.grid(True)
  leg1 = ax1.legend()
  leg1.get_frame().set_edgecolor('black')
  
  # Plot 3D velocity error
  ax2 = fig.add_subplot(122, projection='3d')
  ax2.plot(vel_error[0, :], vel_error[1, :], vel_error[2, :], 'b-', linewidth=1)
  ax2.scatter([vel_error[0,  0]], [vel_error[1,  0]], [vel_error[2,  0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2, label='Initial') # type: ignore
  ax2.scatter([vel_error[0, -1]], [vel_error[1, -1]], [vel_error[2, -1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2, label='Final') # type: ignore
  ax2.set_xlabel('Error Vx [m/s]')
  ax2.set_ylabel('Error Vy [m/s]')
  ax2.set_zlabel('Error Vz [m/s]') # type: ignore
  ax2.set_title('Velocity Error')
  ax2.grid(True)
  leg2 = ax2.legend()
  leg2.get_frame().set_edgecolor('black')
  
  fig.suptitle(title, fontsize=16)
  plt.tight_layout()
  return fig
