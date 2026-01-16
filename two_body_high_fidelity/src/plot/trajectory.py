import datetime
import warnings

import matplotlib.pyplot as plt
import numpy             as np
import cartopy.crs       as ccrs
import cartopy.feature   as cfeature
import spiceypy          as spice

from datetime          import timedelta
from pathlib           import Path
from typing            import Optional
from matplotlib.figure import Figure
from matplotlib.lines  import Line2D

from src.plot.utility                  import get_equal_limits, add_utc_time_axis
from src.orbit_determination.topocentric import compute_topocentric_coordinates
from src.model.constants               import CONVERTER, SOLARSYSTEMCONSTANTS
from src.model.frame_converter         import FrameConverter
from src.model.time_converter          import utc_to_et
from src.model.orbit_converter         import GeographicCoordinateConverter, OrbitConverter
from src.schemas.propagation           import PropagationResult
from src.schemas.state                 import TrackerStation, TopocentricCoordinates


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


def plot_3d_trajectories(
  result : PropagationResult,
  epoch  : Optional[datetime.datetime] = None,
  frame  : str = "J2000",
) -> Figure:
  """
  Plot 3D position and velocity trajectories in a 1x2 grid.
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_time_s'.
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
  plot_time_s = result.plot_time_s
  info_text = f"Frame: {frame}"
  if epoch is not None and plot_time_s is not None:
    start_utc = epoch.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time  = epoch + timedelta(seconds=plot_time_s[-1])
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
  if epoch is not None and frame == "J2000" and plot_time_s is not None:
    try:
      # Get start and end times
      epoch_et_start = utc_to_et(epoch)
      epoch_et_end = epoch_et_start + plot_time_s[-1]
      
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


def plot_time_series(
  result : PropagationResult,
  epoch  : Optional[datetime.datetime] = None,
) -> Figure:
  """
  Plot position and velocity components vs time in a 3-column grid.
  
  Input:
  ------
    result : PropagationResult
      Propagation result.
    epoch : datetime, optional
      Reference epoch for UTC time axis.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the time series plots.
  """
  
  # Create figure (wider to accommodate 3 columns)
  fig = plt.figure(figsize=(24, 10))
  
  # Extract data
  time   = result.plot_time_s
  states = result.state
  pos_x, pos_y, pos_z = states[0, :], states[1, :], states[2, :]
  vel_x, vel_y, vel_z = states[3, :], states[4, :], states[5, :]
  coe = result.coe
  mee = result.mee
  
  # Calculate magnitudes
  pos_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
  vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
  
  # LEFT COLUMN: Position and Velocity
  # Plot position vs time (spans rows 0-2, column 0)
  ax_pos = plt.subplot2grid((6, 3), (0, 0), rowspan=3)
  ax_pos.plot(time, pos_x, 'r-', label='X', linewidth=1.5)
  ax_pos.plot(time, pos_y, 'g-', label='Y', linewidth=1.5)
  ax_pos.plot(time, pos_z, 'b-', label='Z', linewidth=1.5)
  ax_pos.plot(time, pos_mag, 'k-', label='Magnitude', linewidth=2)
  ax_pos.tick_params(labelbottom=False)
  ax_pos.set_ylabel('Position\n[m]')
  ax_pos.legend()
  ax_pos.grid(True)
  ax_pos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot velocity vs time (spans rows 3-5, column 0)
  ax_vel = plt.subplot2grid((6, 3), (3, 0), rowspan=3, sharex=ax_pos)
  ax_vel.plot(time, vel_x, 'r-', label='X', linewidth=1.5)
  ax_vel.plot(time, vel_y, 'g-', label='Y', linewidth=1.5)
  ax_vel.plot(time, vel_z, 'b-', label='Z', linewidth=1.5)
  ax_vel.plot(time, vel_mag, 'k-', label='Magnitude', linewidth=2)
  ax_vel.set_xlabel('Time\n[s]')
  ax_vel.set_ylabel('Velocity\n[m/s]')
  ax_vel.legend()
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # MIDDLE COLUMN: Classical Orbital Elements
  # Plot sma vs time (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 3), (0, 1), sharex=ax_pos)
  ax_sma.plot(time, coe.sma, 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot ecc vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 3), (1, 1), sharex=ax_pos)
  ax_ecc.plot(time, coe.ecc, 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('ECC\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inc vs time (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 3), (2, 1), sharex=ax_pos)
  ax_inc.plot(time, coe.inc * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('INC\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot raan vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 3), (3, 1), sharex=ax_pos)
  ax_raan.plot(time, coe.raan * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN\n[deg]')
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot aop vs time (row 4, column 1)
  ax_aop = plt.subplot2grid((6, 3), (4, 1), sharex=ax_pos)
  aop_unwrapped = np.unwrap(coe.aop) * CONVERTER.DEG_PER_RAD
  ax_aop.plot(time, aop_unwrapped, 'b-', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('AOP\n[deg]')
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot ta, ea, ma vs time (row 5, column 1)
  ax_anom = plt.subplot2grid((6, 3), (5, 1), sharex=ax_pos)
  ax_anom.plot(time, coe.ta * CONVERTER.DEG_PER_RAD, 'r-', label='TA', linewidth=1.5)
  ax_anom.plot(time, coe.ea * CONVERTER.DEG_PER_RAD, 'g-', label='EA', linewidth=1.5)
  ax_anom.plot(time, coe.ma * CONVERTER.DEG_PER_RAD, 'b-', label='MA', linewidth=1.5)
  ax_anom.set_xlabel('Time\n[s]')
  ax_anom.set_ylabel('ANOMALY\n[deg]')
  ax_anom.legend(fontsize=8)
  ax_anom.grid(True)
  ax_anom.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RIGHT COLUMN: Modified Equinoctial Elements
  # Plot p (semi-latus rectum) vs time (row 0, column 2)
  ax_p = plt.subplot2grid((6, 3), (0, 2), sharex=ax_pos)
  ax_p.plot(time, mee.p, 'b-', linewidth=1.5)
  ax_p.tick_params(labelbottom=False)
  ax_p.set_ylabel('p\n[m]')
  ax_p.grid(True)
  ax_p.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot f vs time (row 1, column 2)
  ax_f = plt.subplot2grid((6, 3), (1, 2), sharex=ax_pos)
  ax_f.plot(time, mee.f, 'b-', linewidth=1.5)
  ax_f.tick_params(labelbottom=False)
  ax_f.set_ylabel('f\n[-]')
  ax_f.grid(True)
  ax_f.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot g vs time (row 2, column 2)
  ax_g = plt.subplot2grid((6, 3), (2, 2), sharex=ax_pos)
  ax_g.plot(time, mee.g, 'b-', linewidth=1.5)
  ax_g.tick_params(labelbottom=False)
  ax_g.set_ylabel('g\n[-]')
  ax_g.grid(True)
  ax_g.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot h vs time (row 3, column 2)
  ax_h = plt.subplot2grid((6, 3), (3, 2), sharex=ax_pos)
  ax_h.plot(time, mee.h, 'b-', linewidth=1.5)
  ax_h.tick_params(labelbottom=False)
  ax_h.set_ylabel('h\n[-]')
  ax_h.grid(True)
  ax_h.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot k vs time (row 4, column 2)
  ax_k = plt.subplot2grid((6, 3), (4, 2), sharex=ax_pos)
  ax_k.plot(time, mee.k, 'b-', linewidth=1.5)
  ax_k.tick_params(labelbottom=False)
  ax_k.set_ylabel('k\n[-]')
  ax_k.grid(True)
  ax_k.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot L (true longitude) vs time (row 5, column 2)
  ax_L = plt.subplot2grid((6, 3), (5, 2), sharex=ax_pos)
  ax_L.plot(time, mee.L * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_L.set_xlabel('Time\n[s]')
  ax_L.set_ylabel('L\n[deg]')
  ax_L.grid(True)
  ax_L.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Add UTC time axis if epoch is provided
  if epoch is not None:
    # Only add UTC time to top row axes
    top_row_axes = [ax_pos, ax_sma, ax_p]
    max_time = time[-1]
    
    for ax in top_row_axes:
      add_utc_time_axis(ax, epoch, max_time)

  # Align y-axis labels for middle column (COE)
  fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_aop, ax_anom])
  
  # Align y-axis labels for right column (MEE)
  fig.align_ylabels([ax_p, ax_f, ax_g, ax_h, ax_k, ax_L])
  
  # Align y-axis labels for left column
  fig.align_ylabels([ax_pos, ax_vel])

  plt.subplots_adjust(hspace=0.17, wspace=0.25)
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
  
  time_ref   = result_ref.plot_time_s
  time_comp  = result_comp.time
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


def plot_time_series_error(
  result_ref  : PropagationResult, 
  result_comp : PropagationResult, 
  epoch       : Optional[datetime.datetime] = None, 
  title       : str                         = "Time Series Error", 
  use_ric     : bool                        = True,
) -> Figure:
  """
  Create time series error plots between reference and comparison trajectories.
  
  Input:
  ------
    result_ref : PropagationResult
      Reference result with 'plot_time_s' and 'state'/'coe'/'mee'.
    result_comp : PropagationResult
      Comparison result with 'plot_time_s' and 'state'/'coe'/'mee'.
    epoch : datetime, optional
      Reference epoch for time axis.
    title : str
      Title for the figure.
    use_ric : bool
      If True, transform to RIC frame. If False, use XYZ inertial frame.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the time series error plots.
  """
  # Use plot_time_s for both datasets
  time_ref  = result_ref.plot_time_s
  time_comp = result_comp.plot_time_s
  
  state_ref  = result_ref.state
  state_comp = result_comp.state
  coe_ref    = result_ref.coe
  coe_comp   = result_comp.coe
  mee_ref    = result_ref.mee
  mee_comp   = result_comp.mee
  
  # Verify time grids match (use allclose for floating-point comparison)
  if len(time_ref) != len(time_comp) or not np.allclose(time_ref, time_comp, rtol=1e-9, atol=1e-9):
    raise ValueError(
      f"Time grids don't match! "
      f"Reference has {len(time_ref)} points from {time_ref[0]:.1f} to {time_ref[-1]:.1f} s, "
      f"Comparison has {len(time_comp)} points from {time_comp[0]:.1f} to {time_comp[-1]:.1f} s. "
      f"Both datasets must use the same time grid for error comparison."
    )
  
  # Create figure with subplots matching the grid structure
  fig = plt.figure(figsize=(24, 10))
  
  time = time_ref
  
  if use_ric:
    # Compute RIC frame errors
    pos_error_ric = np.zeros((3, len(time_ref)))
    vel_error_ric = np.zeros((3, len(time_ref)))
    
    for i in range(len(time_ref)):
      # Reference position and velocity
      ref_pos = state_ref[0:3, i]
      ref_vel = state_ref[3:6, i]
      
      # Rotation matrix from inertial to RIC
      R_inertial_to_ric = FrameConverter.xyz_to_ric(ref_pos, ref_vel)
      
      # Compute errors in inertial frame
      pos_error_inertial = state_comp[0:3, i] - state_ref[0:3, i]
      vel_error_inertial = state_comp[3:6, i] - state_ref[3:6, i]
      
      # Transform errors to RIC frame
      pos_error_ric[:, i] = R_inertial_to_ric @ pos_error_inertial
      vel_error_ric[:, i] = R_inertial_to_ric @ vel_error_inertial
    
    pos_error = pos_error_ric
    vel_error = vel_error_ric
    pos_labels = ['Radial', 'In-track', 'Cross-track']
    vel_labels = ['Radial', 'In-track', 'Cross-track']
    pos_ylabel = 'Position Error (RIC)\n[m]'
    vel_ylabel = 'Velocity Error (RIC)\n[m/s]'
  else:
    # Use XYZ inertial frame errors
    pos_error = state_comp[0:3, :] - state_ref[0:3, :]
    vel_error = state_comp[3:6, :] - state_ref[3:6, :]
    pos_labels = ['X', 'Y', 'Z']
    vel_labels = ['X', 'Y', 'Z']
    pos_ylabel = 'Position Error (XYZ)\n[m]'
    vel_ylabel = 'Velocity Error (XYZ)\n[m/s]'
  
  # Calculate error magnitudes
  pos_error_mag = np.linalg.norm(pos_error, axis=0)
  vel_error_mag = np.linalg.norm(vel_error, axis=0)
  
  # LEFT COLUMN: Position and Velocity Errors
  # Plot position error vs time (spans rows 0-2, column 0)
  ax_pos = plt.subplot2grid((6, 3), (0, 0), rowspan=3)
  ax_pos.plot(time, pos_error[0, :], 'r-', label=pos_labels[0], linewidth=1.5)
  ax_pos.plot(time, pos_error[1, :], 'g-', label=pos_labels[1], linewidth=1.5)
  ax_pos.plot(time, pos_error[2, :], 'b-', label=pos_labels[2], linewidth=1.5)
  ax_pos.plot(time, pos_error_mag, 'k-', label='Magnitude', linewidth=2)
  ax_pos.tick_params(labelbottom=False)
  ax_pos.set_ylabel(pos_ylabel)
  ax_pos.legend()
  ax_pos.grid(True)
  ax_pos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot velocity error vs time (spans rows 3-5, column 0)
  ax_vel = plt.subplot2grid((6, 3), (3, 0), rowspan=3, sharex=ax_pos)
  ax_vel.plot(time, vel_error[0, :], 'r-', label=vel_labels[0], linewidth=1.5)
  ax_vel.plot(time, vel_error[1, :], 'g-', label=vel_labels[1], linewidth=1.5)
  ax_vel.plot(time, vel_error[2, :], 'b-', label=vel_labels[2], linewidth=1.5)
  ax_vel.plot(time, vel_error_mag, 'k-', label='Magnitude', linewidth=2)
  ax_vel.set_xlabel('Time\n[s]')
  ax_vel.set_ylabel(vel_ylabel)
  ax_vel.legend()
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # MIDDLE COLUMN: Classical Orbital Elements Errors
  # Plot sma error vs time (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 3), (0, 1), sharex=ax_pos)
  if coe_ref.sma is not None and coe_comp.sma is not None:
    sma_error = coe_comp.sma - coe_ref.sma
    ax_sma.plot(time, sma_error, 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA Error\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot eccentricity error vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 3), (1, 1), sharex=ax_pos)
  if coe_ref.ecc is not None and coe_comp.ecc is not None:
    ecc_error = coe_comp.ecc - coe_ref.ecc
    ax_ecc.plot(time, ecc_error, 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('ECC Error\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inclination error (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 3), (2, 1), sharex=ax_pos)
  if coe_ref.inc is not None and coe_comp.inc is not None:
    inc_error = (coe_comp.inc - coe_ref.inc) * CONVERTER.DEG_PER_RAD
    ax_inc.plot(time, inc_error, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('INC Error\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot RAAN error vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 3), (3, 1), sharex=ax_pos)
  if coe_ref.raan is not None and coe_comp.raan is not None:
    # Handle angle wrapping for RAAN
    raan_error_rad = np.arctan2(np.sin(coe_comp.raan - coe_ref.raan), 
                   np.cos(coe_comp.raan - coe_ref.raan))
    raan_error = raan_error_rad * CONVERTER.DEG_PER_RAD
    ax_raan.plot(time, raan_error, 'b-', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN Error\n[deg]')
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot argument of periapsis error (row 4, column 1)
  ax_aop = plt.subplot2grid((6, 3), (4, 1), sharex=ax_pos)
  if coe_ref.aop is not None and coe_comp.aop is not None:
    # Handle angle wrapping for AOP
    aop_error_rad = np.arctan2(np.sin(coe_comp.aop - coe_ref.aop), 
                   np.cos(coe_comp.aop - coe_ref.aop))
    aop_error = aop_error_rad * CONVERTER.DEG_PER_RAD
    ax_aop.plot(time, aop_error, 'b-', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('AOP Error\n[deg]')
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot true anomaly error vs time (row 5, column 1)
  ax_ta = plt.subplot2grid((6, 3), (5, 1), sharex=ax_pos)
  if coe_ref.ta is not None and coe_comp.ta is not None:
    # Handle angle wrapping for TA
    ta_error_rad = np.arctan2(np.sin(coe_comp.ta - coe_ref.ta), 
                   np.cos(coe_comp.ta - coe_ref.ta))
    ta_error = ta_error_rad * CONVERTER.DEG_PER_RAD
    ax_ta.plot(time, ta_error, 'b-', linewidth=1.5)
  ax_ta.set_xlabel('Time\n[s]')
  ax_ta.set_ylabel('TA Error\n[deg]')
  ax_ta.grid(True)
  ax_ta.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RIGHT COLUMN: Modified Equinoctial Elements Errors
  # Plot p error vs time (row 0, column 2)
  ax_p = plt.subplot2grid((6, 3), (0, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.p is not None and mee_comp.p is not None:
    p_error = mee_comp.p - mee_ref.p
    ax_p.plot(time, p_error, 'b-', linewidth=1.5)
  ax_p.tick_params(labelbottom=False)
  ax_p.set_ylabel('p Error\n[m]')
  ax_p.grid(True)
  ax_p.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot f error vs time (row 1, column 2)
  ax_f = plt.subplot2grid((6, 3), (1, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.f is not None and mee_comp.f is not None:
    f_error = mee_comp.f - mee_ref.f
    ax_f.plot(time, f_error, 'b-', linewidth=1.5)
  ax_f.tick_params(labelbottom=False)
  ax_f.set_ylabel('f Error\n[-]')
  ax_f.grid(True)
  ax_f.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot g error vs time (row 2, column 2)
  ax_g = plt.subplot2grid((6, 3), (2, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.g is not None and mee_comp.g is not None:
    g_error = mee_comp.g - mee_ref.g
    ax_g.plot(time, g_error, 'b-', linewidth=1.5)
  ax_g.tick_params(labelbottom=False)
  ax_g.set_ylabel('g Error\n[-]')
  ax_g.grid(True)
  ax_g.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot h error vs time (row 3, column 2)
  ax_h = plt.subplot2grid((6, 3), (3, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.h is not None and mee_comp.h is not None:
    h_error = mee_comp.h - mee_ref.h
    ax_h.plot(time, h_error, 'b-', linewidth=1.5)
  ax_h.tick_params(labelbottom=False)
  ax_h.set_ylabel('h Error\n[-]')
  ax_h.grid(True)
  ax_h.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot k error vs time (row 4, column 2)
  ax_k = plt.subplot2grid((6, 3), (4, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.k is not None and mee_comp.k is not None:
    k_error = mee_comp.k - mee_ref.k
    ax_k.plot(time, k_error, 'b-', linewidth=1.5)
  ax_k.tick_params(labelbottom=False)
  ax_k.set_ylabel('k Error\n[-]')
  ax_k.grid(True)
  ax_k.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot L error vs time (row 5, column 2)
  ax_L = plt.subplot2grid((6, 3), (5, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.L is not None and mee_comp.L is not None:
    # Handle angle wrapping for L (true longitude)
    L_error_rad = np.arctan2(np.sin(mee_comp.L - mee_ref.L), 
                   np.cos(mee_comp.L - mee_ref.L))
    L_error = L_error_rad * CONVERTER.DEG_PER_RAD
    ax_L.plot(time, L_error, 'b-', linewidth=1.5)
  ax_L.set_xlabel('Time\n[s]')
  ax_L.set_ylabel('L Error\n[deg]')
  ax_L.grid(True)
  ax_L.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Add UTC time axis if epoch is not None:
  if epoch is not None:
    # Only add UTC time to top row axes
    top_row_axes = [ax_pos, ax_sma, ax_p]
    max_time = time[-1]
    
    for ax in top_row_axes:
      add_utc_time_axis(ax, epoch, max_time)

  # Align y-axis labels
  fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_aop, ax_ta])
  fig.align_ylabels([ax_p, ax_f, ax_g, ax_h, ax_k, ax_L])
  fig.align_ylabels([ax_pos, ax_vel])
  
  fig.suptitle(title, fontsize=16)
  plt.subplots_adjust(hspace=0.17, wspace=0.2)
  return fig


def plot_3d_trajectories_body_fixed(
  result                  : PropagationResult,
  epoch_dt_utc            : Optional[datetime.datetime] = None,
  trackers                : Optional[list['TrackerStation']] = None,
  include_tracker_on_body : bool = False,
) -> Figure:
  """
  Plot 3D position and velocity trajectories in body-fixed (IAU_EARTH) frame.
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_time_s'.
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
  time_s        = result.plot_time_s
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
      # Tracker position is stored in radians, convert to degrees for _latlon_to_xyz
      tracker_lat_deg = tracker.position.latitude * CONVERTER.DEG_PER_RAD
      tracker_lon_deg = tracker.position.longitude * CONVERTER.DEG_PER_RAD
      # Use Earth's equatorial radius for consistency
      x_tracker, y_tracker, z_tracker = _latlon_to_xyz(tracker_lat_deg, tracker_lon_deg, r_eq * 1.02)
      ax1.scatter([x_tracker], [y_tracker], [z_tracker], s=200, c='red', marker='o', edgecolors='darkred', linewidths=2, zorder=5)  # type: ignore

      # Draw field of view hemisphere (use tracker's max range)
      fov_radius_m = tracker.performance.range.max
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


def plot_time_series(
  result : PropagationResult,
  epoch  : Optional[datetime.datetime] = None,
) -> Figure:
  """
  Plot position and velocity components vs time in a 3-column grid.
  
  Input:
  ------
    result : PropagationResult
      Propagation result.
    epoch : datetime, optional
      Reference epoch for UTC time axis.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the time series plots.
  """
  
  # Create figure (wider to accommodate 3 columns)
  fig = plt.figure(figsize=(24, 10))
  
  # Extract data
  time   = result.plot_time_s
  states = result.state
  pos_x, pos_y, pos_z = states[0, :], states[1, :], states[2, :]
  vel_x, vel_y, vel_z = states[3, :], states[4, :], states[5, :]
  coe = result.coe
  mee = result.mee
  
  # Calculate magnitudes
  pos_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
  vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
  
  # LEFT COLUMN: Position and Velocity
  # Plot position vs time (spans rows 0-2, column 0)
  ax_pos = plt.subplot2grid((6, 3), (0, 0), rowspan=3)
  ax_pos.plot(time, pos_x, 'r-', label='X', linewidth=1.5)
  ax_pos.plot(time, pos_y, 'g-', label='Y', linewidth=1.5)
  ax_pos.plot(time, pos_z, 'b-', label='Z', linewidth=1.5)
  ax_pos.plot(time, pos_mag, 'k-', label='Magnitude', linewidth=2)
  ax_pos.tick_params(labelbottom=False)
  ax_pos.set_ylabel('Position\n[m]')
  ax_pos.legend()
  ax_pos.grid(True)
  ax_pos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot velocity vs time (spans rows 3-5, column 0)
  ax_vel = plt.subplot2grid((6, 3), (3, 0), rowspan=3, sharex=ax_pos)
  ax_vel.plot(time, vel_x, 'r-', label='X', linewidth=1.5)
  ax_vel.plot(time, vel_y, 'g-', label='Y', linewidth=1.5)
  ax_vel.plot(time, vel_z, 'b-', label='Z', linewidth=1.5)
  ax_vel.plot(time, vel_mag, 'k-', label='Magnitude', linewidth=2)
  ax_vel.set_xlabel('Time\n[s]')
  ax_vel.set_ylabel('Velocity\n[m/s]')
  ax_vel.legend()
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # MIDDLE COLUMN: Classical Orbital Elements
  # Plot sma vs time (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 3), (0, 1), sharex=ax_pos)
  ax_sma.plot(time, coe.sma, 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot ecc vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 3), (1, 1), sharex=ax_pos)
  ax_ecc.plot(time, coe.ecc, 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('ECC\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inc vs time (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 3), (2, 1), sharex=ax_pos)
  ax_inc.plot(time, coe.inc * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('INC\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot raan vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 3), (3, 1), sharex=ax_pos)
  ax_raan.plot(time, coe.raan * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN\n[deg]')
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot aop vs time (row 4, column 1)
  ax_aop = plt.subplot2grid((6, 3), (4, 1), sharex=ax_pos)
  aop_unwrapped = np.unwrap(coe.aop) * CONVERTER.DEG_PER_RAD
  ax_aop.plot(time, aop_unwrapped, 'b-', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('AOP\n[deg]')
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot ta, ea, ma vs time (row 5, column 1)
  ax_anom = plt.subplot2grid((6, 3), (5, 1), sharex=ax_pos)
  ax_anom.plot(time, coe.ta * CONVERTER.DEG_PER_RAD, 'r-', label='TA', linewidth=1.5)
  ax_anom.plot(time, coe.ea * CONVERTER.DEG_PER_RAD, 'g-', label='EA', linewidth=1.5)
  ax_anom.plot(time, coe.ma * CONVERTER.DEG_PER_RAD, 'b-', label='MA', linewidth=1.5)
  ax_anom.set_xlabel('Time\n[s]')
  ax_anom.set_ylabel('ANOMALY\n[deg]')
  ax_anom.legend(fontsize=8)
  ax_anom.grid(True)
  ax_anom.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RIGHT COLUMN: Modified Equinoctial Elements
  # Plot p (semi-latus rectum) vs time (row 0, column 2)
  ax_p = plt.subplot2grid((6, 3), (0, 2), sharex=ax_pos)
  ax_p.plot(time, mee.p, 'b-', linewidth=1.5)
  ax_p.tick_params(labelbottom=False)
  ax_p.set_ylabel('p\n[m]')
  ax_p.grid(True)
  ax_p.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot f vs time (row 1, column 2)
  ax_f = plt.subplot2grid((6, 3), (1, 2), sharex=ax_pos)
  ax_f.plot(time, mee.f, 'b-', linewidth=1.5)
  ax_f.tick_params(labelbottom=False)
  ax_f.set_ylabel('f\n[-]')
  ax_f.grid(True)
  ax_f.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot g vs time (row 2, column 2)
  ax_g = plt.subplot2grid((6, 3), (2, 2), sharex=ax_pos)
  ax_g.plot(time, mee.g, 'b-', linewidth=1.5)
  ax_g.tick_params(labelbottom=False)
  ax_g.set_ylabel('g\n[-]')
  ax_g.grid(True)
  ax_g.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot h vs time (row 3, column 2)
  ax_h = plt.subplot2grid((6, 3), (3, 2), sharex=ax_pos)
  ax_h.plot(time, mee.h, 'b-', linewidth=1.5)
  ax_h.tick_params(labelbottom=False)
  ax_h.set_ylabel('h\n[-]')
  ax_h.grid(True)
  ax_h.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot k vs time (row 4, column 2)
  ax_k = plt.subplot2grid((6, 3), (4, 2), sharex=ax_pos)
  ax_k.plot(time, mee.k, 'b-', linewidth=1.5)
  ax_k.tick_params(labelbottom=False)
  ax_k.set_ylabel('k\n[-]')
  ax_k.grid(True)
  ax_k.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot L (true longitude) vs time (row 5, column 2)
  ax_L = plt.subplot2grid((6, 3), (5, 2), sharex=ax_pos)
  ax_L.plot(time, mee.L * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_L.set_xlabel('Time\n[s]')
  ax_L.set_ylabel('L\n[deg]')
  ax_L.grid(True)
  ax_L.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Add UTC time axis if epoch is provided
  if epoch is not None:
    # Only add UTC time to top row axes
    top_row_axes = [ax_pos, ax_sma, ax_p]
    max_time = time[-1]
    
    for ax in top_row_axes:
      add_utc_time_axis(ax, epoch, max_time)

  # Align y-axis labels for middle column (COE)
  fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_aop, ax_anom])
  
  # Align y-axis labels for right column (MEE)
  fig.align_ylabels([ax_p, ax_f, ax_g, ax_h, ax_k, ax_L])
  
  # Align y-axis labels for left column
  fig.align_ylabels([ax_pos, ax_vel])

  plt.subplots_adjust(hspace=0.17, wspace=0.25)
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
  
  time_ref   = result_ref.plot_time_s
  time_comp  = result_comp.time
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


def plot_time_series_error(
  result_ref  : PropagationResult, 
  result_comp : PropagationResult, 
  epoch       : Optional[datetime.datetime] = None, 
  title       : str                         = "Time Series Error", 
  use_ric     : bool                        = True,
) -> Figure:
  """
  Create time series error plots between reference and comparison trajectories.
  
  Input:
  ------
    result_ref : PropagationResult
      Reference result with 'plot_time_s' and 'state'/'coe'/'mee'.
    result_comp : PropagationResult
      Comparison result with 'plot_time_s' and 'state'/'coe'/'mee'.
    epoch : datetime, optional
      Reference epoch for time axis.
    title : str
      Title for the figure.
    use_ric : bool
      If True, transform to RIC frame. If False, use XYZ inertial frame.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the time series error plots.
  """
  # Use plot_time_s for both datasets
  time_ref  = result_ref.plot_time_s
  time_comp = result_comp.plot_time_s
  
  state_ref  = result_ref.state
  state_comp = result_comp.state
  coe_ref    = result_ref.coe
  coe_comp   = result_comp.coe
  mee_ref    = result_ref.mee
  mee_comp   = result_comp.mee
  
  # Verify time grids match (use allclose for floating-point comparison)
  if len(time_ref) != len(time_comp) or not np.allclose(time_ref, time_comp, rtol=1e-9, atol=1e-9):
    raise ValueError(
      f"Time grids don't match! "
      f"Reference has {len(time_ref)} points from {time_ref[0]:.1f} to {time_ref[-1]:.1f} s, "
      f"Comparison has {len(time_comp)} points from {time_comp[0]:.1f} to {time_comp[-1]:.1f} s. "
      f"Both datasets must use the same time grid for error comparison."
    )
  
  # Create figure with subplots matching the grid structure
  fig = plt.figure(figsize=(24, 10))
  
  time = time_ref
  
  if use_ric:
    # Compute RIC frame errors
    pos_error_ric = np.zeros((3, len(time_ref)))
    vel_error_ric = np.zeros((3, len(time_ref)))
    
    for i in range(len(time_ref)):
      # Reference position and velocity
      ref_pos = state_ref[0:3, i]
      ref_vel = state_ref[3:6, i]
      
      # Rotation matrix from inertial to RIC
      R_inertial_to_ric = FrameConverter.xyz_to_ric(ref_pos, ref_vel)
      
      # Compute errors in inertial frame
      pos_error_inertial = state_comp[0:3, i] - state_ref[0:3, i]
      vel_error_inertial = state_comp[3:6, i] - state_ref[3:6, i]
      
      # Transform errors to RIC frame
      pos_error_ric[:, i] = R_inertial_to_ric @ pos_error_inertial
      vel_error_ric[:, i] = R_inertial_to_ric @ vel_error_inertial
    
    pos_error = pos_error_ric
    vel_error = vel_error_ric
    pos_labels = ['Radial', 'In-track', 'Cross-track']
    vel_labels = ['Radial', 'In-track', 'Cross-track']
    pos_ylabel = 'Position Error (RIC)\n[m]'
    vel_ylabel = 'Velocity Error (RIC)\n[m/s]'
  else:
    # Use XYZ inertial frame errors
    pos_error = state_comp[0:3, :] - state_ref[0:3, :]
    vel_error = state_comp[3:6, :] - state_ref[3:6, :]
    pos_labels = ['X', 'Y', 'Z']
    vel_labels = ['X', 'Y', 'Z']
    pos_ylabel = 'Position Error (XYZ)\n[m]'
    vel_ylabel = 'Velocity Error (XYZ)\n[m/s]'
  
  # Calculate error magnitudes
  pos_error_mag = np.linalg.norm(pos_error, axis=0)
  vel_error_mag = np.linalg.norm(vel_error, axis=0)
  
  # LEFT COLUMN: Position and Velocity Errors
  # Plot position error vs time (spans rows 0-2, column 0)
  ax_pos = plt.subplot2grid((6, 3), (0, 0), rowspan=3)
  ax_pos.plot(time, pos_error[0, :], 'r-', label=pos_labels[0], linewidth=1.5)
  ax_pos.plot(time, pos_error[1, :], 'g-', label=pos_labels[1], linewidth=1.5)
  ax_pos.plot(time, pos_error[2, :], 'b-', label=pos_labels[2], linewidth=1.5)
  ax_pos.plot(time, pos_error_mag, 'k-', label='Magnitude', linewidth=2)
  ax_pos.tick_params(labelbottom=False)
  ax_pos.set_ylabel(pos_ylabel)
  ax_pos.legend()
  ax_pos.grid(True)
  ax_pos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot velocity error vs time (spans rows 3-5, column 0)
  ax_vel = plt.subplot2grid((6, 3), (3, 0), rowspan=3, sharex=ax_pos)
  ax_vel.plot(time, vel_error[0, :], 'r-', label=vel_labels[0], linewidth=1.5)
  ax_vel.plot(time, vel_error[1, :], 'g-', label=vel_labels[1], linewidth=1.5)
  ax_vel.plot(time, vel_error[2, :], 'b-', label=vel_labels[2], linewidth=1.5)
  ax_vel.plot(time, vel_error_mag, 'k-', label='Magnitude', linewidth=2)
  ax_vel.set_xlabel('Time\n[s]')
  ax_vel.set_ylabel(vel_ylabel)
  ax_vel.legend()
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # MIDDLE COLUMN: Classical Orbital Elements Errors
  # Plot sma error vs time (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 3), (0, 1), sharex=ax_pos)
  if coe_ref.sma is not None and coe_comp.sma is not None:
    sma_error = coe_comp.sma - coe_ref.sma
    ax_sma.plot(time, sma_error, 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA Error\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot eccentricity error vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 3), (1, 1), sharex=ax_pos)
  if coe_ref.ecc is not None and coe_comp.ecc is not None:
    ecc_error = coe_comp.ecc - coe_ref.ecc
    ax_ecc.plot(time, ecc_error, 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('ECC Error\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inclination error (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 3), (2, 1), sharex=ax_pos)
  if coe_ref.inc is not None and coe_comp.inc is not None:
    inc_error = (coe_comp.inc - coe_ref.inc) * CONVERTER.DEG_PER_RAD
    ax_inc.plot(time, inc_error, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('INC Error\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot RAAN error vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 3), (3, 1), sharex=ax_pos)
  if coe_ref.raan is not None and coe_comp.raan is not None:
    # Handle angle wrapping for RAAN
    raan_error_rad = np.arctan2(np.sin(coe_comp.raan - coe_ref.raan), 
                   np.cos(coe_comp.raan - coe_ref.raan))
    raan_error = raan_error_rad * CONVERTER.DEG_PER_RAD
    ax_raan.plot(time, raan_error, 'b-', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN Error\n[deg]')
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot argument of periapsis error (row 4, column 1)
  ax_aop = plt.subplot2grid((6, 3), (4, 1), sharex=ax_pos)
  if coe_ref.aop is not None and coe_comp.aop is not None:
    # Handle angle wrapping for AOP
    aop_error_rad = np.arctan2(np.sin(coe_comp.aop - coe_ref.aop), 
                   np.cos(coe_comp.aop - coe_ref.aop))
    aop_error = aop_error_rad * CONVERTER.DEG_PER_RAD
    ax_aop.plot(time, aop_error, 'b-', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('AOP Error\n[deg]')
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot true anomaly error vs time (row 5, column 1)
  ax_ta = plt.subplot2grid((6, 3), (5, 1), sharex=ax_pos)
  if coe_ref.ta is not None and coe_comp.ta is not None:
    # Handle angle wrapping for TA
    ta_error_rad = np.arctan2(np.sin(coe_comp.ta - coe_ref.ta), 
                   np.cos(coe_comp.ta - coe_ref.ta))
    ta_error = ta_error_rad * CONVERTER.DEG_PER_RAD
    ax_ta.plot(time, ta_error, 'b-', linewidth=1.5)
  ax_ta.set_xlabel('Time\n[s]')
  ax_ta.set_ylabel('TA Error\n[deg]')
  ax_ta.grid(True)
  ax_ta.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RIGHT COLUMN: Modified Equinoctial Elements Errors
  # Plot p error vs time (row 0, column 2)
  ax_p = plt.subplot2grid((6, 3), (0, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.p is not None and mee_comp.p is not None:
    p_error = mee_comp.p - mee_ref.p
    ax_p.plot(time, p_error, 'b-', linewidth=1.5)
  ax_p.tick_params(labelbottom=False)
  ax_p.set_ylabel('p Error\n[m]')
  ax_p.grid(True)
  ax_p.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot f error vs time (row 1, column 2)
  ax_f = plt.subplot2grid((6, 3), (1, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.f is not None and mee_comp.f is not None:
    f_error = mee_comp.f - mee_ref.f
    ax_f.plot(time, f_error, 'b-', linewidth=1.5)
  ax_f.tick_params(labelbottom=False)
  ax_f.set_ylabel('f Error\n[-]')
  ax_f.grid(True)
  ax_f.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot g error vs time (row 2, column 2)
  ax_g = plt.subplot2grid((6, 3), (2, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.g is not None and mee_comp.g is not None:
    g_error = mee_comp.g - mee_ref.g
    ax_g.plot(time, g_error, 'b-', linewidth=1.5)
  ax_g.tick_params(labelbottom=False)
  ax_g.set_ylabel('g Error\n[-]')
  ax_g.grid(True)
  ax_g.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot h error vs time (row 3, column 2)
  ax_h = plt.subplot2grid((6, 3), (3, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.h is not None and mee_comp.h is not None:
    h_error = mee_comp.h - mee_ref.h
    ax_h.plot(time, h_error, 'b-', linewidth=1.5)
  ax_h.tick_params(labelbottom=False)
  ax_h.set_ylabel('h Error\n[-]')
  ax_h.grid(True)
  ax_h.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot k error vs time (row 4, column 2)
  ax_k = plt.subplot2grid((6, 3), (4, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.k is not None and mee_comp.k is not None:
    k_error = mee_comp.k - mee_ref.k
    ax_k.plot(time, k_error, 'b-', linewidth=1.5)
  ax_k.tick_params(labelbottom=False)
  ax_k.set_ylabel('k Error\n[-]')
  ax_k.grid(True)
  ax_k.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot L error vs time (row 5, column 2)
  ax_L = plt.subplot2grid((6, 3), (5, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.L is not None and mee_comp.L is not None:
    # Handle angle wrapping for L (true longitude)
    L_error_rad = np.arctan2(np.sin(mee_comp.L - mee_ref.L), 
                   np.cos(mee_comp.L - mee_ref.L))
    L_error = L_error_rad * CONVERTER.DEG_PER_RAD
    ax_L.plot(time, L_error, 'b-', linewidth=1.5)
  ax_L.set_xlabel('Time\n[s]')
  ax_L.set_ylabel('L Error\n[deg]')
  ax_L.grid(True)
  ax_L.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Add UTC time axis if epoch is not None:
  if epoch is not None:
    # Only add UTC time to top row axes
    top_row_axes = [ax_pos, ax_sma, ax_p]
    max_time = time[-1]
    
    for ax in top_row_axes:
      add_utc_time_axis(ax, epoch, max_time)

  # Align y-axis labels
  fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_aop, ax_ta])
  fig.align_ylabels([ax_p, ax_f, ax_g, ax_h, ax_k, ax_L])
  fig.align_ylabels([ax_pos, ax_vel])
  
  fig.suptitle(title, fontsize=16)
  plt.subplots_adjust(hspace=0.17, wspace=0.2)
  return fig


def plot_3d_trajectories_body_fixed(
  result                  : PropagationResult,
  epoch_dt_utc            : Optional[datetime.datetime] = None,
  trackers                : Optional[list['TrackerStation']] = None,
  include_tracker_on_body : bool = False,
) -> Figure:
  """
  Plot 3D position and velocity trajectories in body-fixed (IAU_EARTH) frame.
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_time_s'.
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
  time_s        = result.plot_time_s
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
      # Tracker position is stored in radians, convert to degrees for _latlon_to_xyz
      tracker_lat_deg = tracker.position.latitude * CONVERTER.DEG_PER_RAD
      tracker_lon_deg = tracker.position.longitude * CONVERTER.DEG_PER_RAD
      # Use Earth's equatorial radius for consistency
      x_tracker, y_tracker, z_tracker = _latlon_to_xyz(tracker_lat_deg, tracker_lon_deg, r_eq * 1.02)
      ax1.scatter([x_tracker], [y_tracker], [z_tracker], s=200, c='red', marker='o', edgecolors='darkred', linewidths=2, zorder=5)  # type: ignore

      # Draw field of view hemisphere (use tracker's max range)
      fov_radius_m = tracker.performance.range.max
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


def _latlon_to_xyz(lat_deg, lon_deg, radius):
  """
  Convert latitude/longitude (degrees) to 3D Cartesian coordinates on a sphere.
  
  Input:
  ------
    lat_deg : array-like
      Latitude in degrees
    lon_deg : array-like
      Longitude in degrees
    radius : float
      Sphere radius
      
  Output:
  -------
    x, y, z : arrays
      Cartesian coordinates
  """
  lat_rad = np.deg2rad(lat_deg)
  lon_rad = np.deg2rad(lon_deg)
  x = radius * np.cos(lat_rad) * np.cos(lon_rad)
  y = radius * np.cos(lat_rad) * np.sin(lon_rad)
  z = radius * np.sin(lat_rad)
  return x, y, z


def _create_tracker_fov_hemisphere(tracker_lat_deg, tracker_lon_deg, earth_radius_m, fov_radius_m, resolution=30):
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
    x, y, z : 2D arrays
      Cartesian coordinates of the hemisphere surface
  """
  # Get the tracker position on Earth's surface
  x_tracker, y_tracker, z_tracker = _latlon_to_xyz(tracker_lat_deg, tracker_lon_deg, earth_radius_m)

  # Calculate the outward normal vector (from Earth center to tracker)
  normal = np.array([x_tracker, y_tracker, z_tracker])
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
  x_fov = rotated_points[0].reshape(x_sphere.shape) + x_tracker
  y_fov = rotated_points[1].reshape(y_sphere.shape) + y_tracker
  z_fov = rotated_points[2].reshape(z_sphere.shape) + z_tracker

  return x_fov, y_fov, z_fov


def _calculate_hemisphere_ground_track(tracker_lat_deg, tracker_lon_deg, earth_radius_m, fov_radius_m, num_points=100):
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
    lat_circle : array
      Latitudes of the circle points in degrees
    lon_circle : array
      Longitudes of the circle points in degrees
  """
  # Calculate the angular radius of the FOV circle on Earth's surface
  # This is the angle subtended at Earth's center by the FOV radius
  angular_radius_rad = fov_radius_m / earth_radius_m

  # Convert tracker position to radians
  tracker_lat_rad = np.deg2rad(tracker_lat_deg)
  tracker_lon_rad = np.deg2rad(tracker_lon_deg)

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

    lat_circle.append(np.rad2deg(lat_rad))
    lon_circle.append(np.rad2deg(lon_rad))

  # Convert to arrays and ensure longitude wraps properly
  lat_circle = np.array(lat_circle)
  lon_circle = np.array(lon_circle)

  # Wrap longitude to [-180, 180]
  lon_circle = np.mod(lon_circle + 180, 360) - 180

  return lat_circle, lon_circle


def plot_ground_track(
  result                  : PropagationResult,
  epoch_dt_utc            : Optional[datetime.datetime] = None,
  title_text              : str = "Ground Track",
  trackers                : Optional[list['TrackerStation']] = None,
  include_tracker_on_body : bool = False,
) -> Figure:
  """
  Plot ground track with 3D globe (left) and 2D map projection (right).
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_time_s'.
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
  time_s        = result.plot_time_s
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
    x_coast, y_coast, z_coast = _latlon_to_xyz(lat_coast, lon_coast, r_earth * 1.001)  # Slightly above surface
    ax_3d.plot(x_coast, y_coast, z_coast, 'k-', linewidth=0.5, alpha=0.8, zorder=2)
  
  # Draw latitude/longitude grid lines
  grid_alpha = 0.2
  grid_color = 'gray'
  # Latitude lines (every 30 degrees)
  for lat_line in [-60, -30, 0, 30, 60]:
    lon_grid = np.linspace(-180, 180, 100)
    lat_grid = np.full_like(lon_grid, lat_line)
    x_grid, y_grid, z_grid = _latlon_to_xyz(lat_grid, lon_grid, r_earth * 1.002)
    ax_3d.plot(x_grid, y_grid, z_grid, color=grid_color, linewidth=0.3, alpha=grid_alpha, zorder=1)
  # Longitude lines (every 30 degrees)
  for lon_line in range(-180, 180, 30):
    lat_grid = np.linspace(-90, 90, 50)
    lon_grid = np.full_like(lat_grid, lon_line)
    x_grid, y_grid, z_grid = _latlon_to_xyz(lat_grid, lon_grid, r_earth * 1.002)
    ax_3d.plot(x_grid, y_grid, z_grid, color=grid_color, linewidth=0.3, alpha=grid_alpha, zorder=1)
  
  # Plot ground track on 3D sphere (projected to surface)
  x_track_all = []
  y_track_all = []
  z_track_all = []
  for lat_seg, lon_seg in zip(lat_segments, lon_segments):
    x_track, y_track, z_track = _latlon_to_xyz(lat_seg, lon_seg, r_earth * 1.01)  # Slightly above surface
    ax_3d.plot(x_track, y_track, z_track, 'b-', linewidth=2.0, zorder=3)
    x_track_all.extend(x_track)
    y_track_all.extend(y_track)
    z_track_all.extend(z_track)
  x_track_all = np.array(x_track_all)
  y_track_all = np.array(y_track_all)
  z_track_all = np.array(z_track_all)
  
  # Mark start and end points on 3D globe
  x_start, y_start, z_start = _latlon_to_xyz(lat[0], lon[0], r_earth * 1.02)
  x_end, y_end, z_end = _latlon_to_xyz(lat[-1], lon[-1], r_earth * 1.02)
  ax_3d.scatter([x_start], [y_start], [z_start], s=400, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=4)  # type: ignore
  ax_3d.scatter([x_end], [y_end], [z_end], s=400, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=4)  # type: ignore

  # Mark tracker locations on 3D globe (red dots) and show field of view
  if include_tracker_on_body and trackers is not None:
    for tracker in trackers:
      # Tracker position is stored in radians, convert to degrees for _latlon_to_xyz
      tracker_lat_deg = tracker.position.latitude * CONVERTER.DEG_PER_RAD
      tracker_lon_deg = tracker.position.longitude * CONVERTER.DEG_PER_RAD
      x_tracker, y_tracker, z_tracker = _latlon_to_xyz(tracker_lat_deg, tracker_lon_deg, r_earth * 1.02)
      ax_3d.scatter([x_tracker], [y_tracker], [z_tracker], s=200, c='red', marker='o', edgecolors='darkred', linewidths=2, zorder=5)  # type: ignore

      # Draw field of view hemisphere (use tracker's max range)
      fov_radius_m = tracker.performance.range.max
      x_fov, y_fov, z_fov = _create_tracker_fov_hemisphere(tracker_lat_deg, tracker_lon_deg, r_earth, fov_radius_m, resolution=30)
      ax_3d.plot_surface(x_fov, y_fov, z_fov, color='red', alpha=0.2, edgecolor='none', zorder=2)  # type: ignore

      # Draw FOV ground track circle on Earth's surface
      lat_circle, lon_circle = _calculate_hemisphere_ground_track(tracker_lat_deg, tracker_lon_deg, r_earth, fov_radius_m, num_points=100)
      x_circle, y_circle, z_circle = _latlon_to_xyz(lat_circle, lon_circle, r_earth * 1.01)
      ax_3d.plot(x_circle, y_circle, z_circle, 'r--', linewidth=2, zorder=3)  # type: ignore

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
      fov_radius_m = tracker.performance.range.max
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


def plot_3d_trajectory_sun_centered(
  result : PropagationResult,
  epoch  : Optional[datetime.datetime] = None,
) -> Figure:
  """
  Plot 3D position trajectory with Moon trajectory in J2000 Earth-centered frame.
  
  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_time_s'.
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
  time_s     = result.plot_time_s
  
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

def generate_error_plots(
  result_jpl_horizons_ephemeris    : Optional[PropagationResult],
  result_high_fidelity_propagation : PropagationResult,
  result_sgp4_propagation          : Optional[PropagationResult],
  time_o_dt                        : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool,
  compare_tle                      : bool,
  object_name                      : str = "object",
  object_name_display              : str = "Object",
) -> dict:
  """
  Generate and save error comparison plots.
  
  Returns:
  --------
    dict : Dictionary with error plot filenames organized by comparison type.
  """
  error_files = {
    'hf_vs_horizons': [],
    'hf_vs_sgp4': [],
    'sgp4_vs_horizons': [],
  }
  
  # If neither comparison is requested, do nothing
  if not (compare_jpl_horizons or compare_tle):
    return error_files

  # Define availability flags
  has_horizons      = result_jpl_horizons_ephemeris is not None and result_jpl_horizons_ephemeris.success
  has_high_fidelity = result_high_fidelity_propagation.success
  has_sgp4          = result_sgp4_propagation is not None and result_sgp4_propagation.success

  # Check for pre-computed ephemeris-time data
  has_hf_at_ephem   = has_high_fidelity and result_high_fidelity_propagation.at_ephem_times is not None
  has_sgp4_at_ephem = has_sgp4 and result_sgp4_propagation.at_ephem_times is not None

  # Lowercase name for filenames
  name_lower = object_name.lower()

  # High-Fidelity Relative To JPL Horizons (compare at ephemeris times)
  if compare_jpl_horizons and has_horizons and has_hf_at_ephem:
    # Build result dict for comparison using pre-computed at_ephem_times data
    # Note: plot_time_series_error expects dicts or objects. 
    # at_ephem_times is a dict in PropagationResult.
    hf_at_ephem = result_high_fidelity_propagation.at_ephem_times
    
    fig_err_ts = plot_time_series_error(
      result_ref  = result_jpl_horizons_ephemeris,
      result_comp = hf_at_ephem,
      epoch       = time_o_dt,
      use_ric     = True,
    )
    title = f'Error Time Series: High-Fidelity vs JPL Horizons - {object_name_display}'
    fig_err_ts.suptitle(title, fontsize=14)
    filename = f'error_timeseries_high_fidelity_rel_jpl_horizons_{name_lower}.png'
    fig_err_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    error_files['hf_vs_horizons'].append(filename)
    plt.close(fig_err_ts)

  # High-Fidelity Relative To SGP4 (compare at equal grid times)
  if compare_tle and has_high_fidelity and has_sgp4:
    # Use equal grid data (main plot_time_s, state, coe)
    fig_err_ts = plot_time_series_error(
      result_ref  = result_sgp4_propagation,
      result_comp = result_high_fidelity_propagation,
      epoch       = time_o_dt,
      use_ric     = True,
    )
    title = f'Error Time Series: High-Fidelity vs SGP4 - {object_name_display}'
    fig_err_ts.suptitle(title, fontsize=14)
    filename = f'error_timeseries_high_fidelity_rel_sgp4_{name_lower}.png'
    fig_err_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    error_files['hf_vs_sgp4'].append(filename)
    plt.close(fig_err_ts)

  # SGP4 Relative To JPL Horizons (compare at ephemeris times)
  if compare_jpl_horizons and compare_tle and has_horizons and has_sgp4_at_ephem:
    # Build result dict for comparison using pre-computed at_ephem_times data
    sgp4_at_ephem = result_sgp4_propagation.at_ephem_times
    
    fig_err_ts = plot_time_series_error(
      result_ref  = result_jpl_horizons_ephemeris,
      result_comp = sgp4_at_ephem,
      epoch       = time_o_dt,
      use_ric     = True,
    )
    title = f'RIC Errors: SGP4 vs JPL Horizons - {object_name_display}'
    fig_err_ts.suptitle(title, fontsize=14)
    filename = f'error_timeseries_sgp4_rel_jpl_horizons_{name_lower}.png'
    fig_err_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    error_files['sgp4_vs_horizons'].append(filename)
    plt.close(fig_err_ts)
  
  return error_files


def plot_skyplot(
  result       : PropagationResult,
  tracker      : TrackerStation,
  epoch_dt_utc : Optional[datetime.datetime] = None,
  title_text   : str = "Skyplot",
) -> Figure:
  """
  Plot a skyplot (polar plot of azimuth vs elevation) from a ground station.

  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_time_s'.
    tracker : TrackerStation
      Ground tracking station with latitude, longitude, altitude.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.
    title_text : str
      Base title for the plot.

  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the skyplot and time-series plots.
  """
  fig = plt.figure(figsize=(20, 10))
  
  # Compute topocentric coordinates
  topo = compute_topocentric_coordinates(result, tracker, epoch_dt_utc)
  
  # Convert to degrees for display
  az_deg  = topo.azimuth   * CONVERTER.DEG_PER_RAD
  el_deg  = topo.elevation * CONVERTER.DEG_PER_RAD
  time_s  = result.plot_time_s

  # Normalize azimuth to -180° to +180° range
  az_deg = ((az_deg + 180.0) % 360.0) - 180.0

  # For polar plot: radius = 90 - elevation (so zenith is at center)
  # theta = azimuth (convert negative angles to positive for polar plotting: 0 to 2π)
  radius = 90.0 - el_deg
  theta  = np.deg2rad(np.where(az_deg >= 0, az_deg, az_deg + 360.0))

  # Create grid: polar skyplot on left, three time-series plots on right
  ax = fig.add_subplot(1, 2, 1, projection='polar')
  
  # Configure polar plot for skyplot convention
  ax.set_theta_zero_location('N')  # North at top
  ax.set_theta_direction(-1)       # Clockwise
  ax.set_rlim(0, 90)               # 0 to 90 degrees from zenith
  ax.set_rticks([0, 15, 30, 45, 60, 75, 90])
  ax.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'])  # Elevation labels

  # Add gray shaded regions for elevation and azimuth constraints
  if tracker.performance:

    # Elevation constraints (circular boundaries)
    if tracker.performance.elevation:
      el_min_deg = tracker.performance.elevation.min * CONVERTER.DEG_PER_RAD
      el_max_deg = tracker.performance.elevation.max * CONVERTER.DEG_PER_RAD

      # Convert elevation to radius (radius = 90 - elevation)
      radius_max_constraint = 90.0 - el_min_deg  # Outer boundary (low elevation limit)
      radius_min_constraint = 90.0 - el_max_deg  # Inner boundary (high elevation limit)

      # Create full circle for constraint visualization
      theta_circle = np.linspace(0, 2 * np.pi, 360)

      # Shade region outside valid elevation range (below minimum elevation)
      if el_min_deg > 0:
        # Fill from outer edge (90 deg from zenith = horizon) to minimum elevation
        ax.fill_between(theta_circle, radius_max_constraint, 90,
                        color='gray', alpha=0.1, label=f'Below Min El ({el_min_deg:.0f}°)')

      # Shade region above maximum elevation (if max < 90 deg)
      if el_max_deg < 90:
        # Fill from zenith (0 deg from zenith) to maximum elevation
        ax.fill_between(theta_circle, 0, radius_min_constraint,
                        color='gray', alpha=0.1, label=f'Above Max El ({el_max_deg:.0f}°)')

    # Azimuth constraints (wedge-shaped boundaries)
    if tracker.performance.azimuth:
      # Azimuth values are already normalized to -180° to +180° in the loader
      az_min_deg = tracker.performance.azimuth.min * CONVERTER.DEG_PER_RAD
      az_max_deg = tracker.performance.azimuth.max * CONVERTER.DEG_PER_RAD

      # Convert azimuth to theta (polar angle in radians)
      # Note: polar plots use 0-2π, but our azimuth is in -180° to +180°
      # Convert negative angles to positive for polar plotting
      az_min_rad = np.deg2rad(az_min_deg if az_min_deg >= 0 else az_min_deg + 360.0)
      az_max_rad = np.deg2rad(az_max_deg if az_max_deg >= 0 else az_max_deg + 360.0)

      # Check if the valid azimuth range wraps around (e.g., 120° to -120° crosses ±180°)
      wraps_around = az_max_deg < az_min_deg

      if not wraps_around:
        # Normal case: valid range doesn't cross ±180°
        # Shade two wedges: [0, az_min] and [az_max, 2π]

        # Shade before minimum azimuth (0° to az_min)
        if az_min_rad > 0:
          theta_wedge_1 = np.linspace(0, az_min_rad, 100)
          radius_full = np.full_like(theta_wedge_1, 90)
          ax.fill_between(theta_wedge_1, 0, radius_full,
                          color='gray', alpha=0.1, label=f'Az < {az_min_deg:.0f}°')

        # Shade after maximum azimuth (az_max to 360°)
        if az_max_rad < 2 * np.pi:
          theta_wedge_2 = np.linspace(az_max_rad, 2 * np.pi, 100)
          radius_full = np.full_like(theta_wedge_2, 90)
          ax.fill_between(theta_wedge_2, 0, radius_full,
                          color='gray', alpha=0.1, label=f'Az > {az_max_deg:.0f}°')
      else:
        # Wraps around ±180°: valid range is [az_min, 180] + [-180, az_max]
        # Shade the middle wedge [az_max, az_min]
        theta_wedge = np.linspace(az_max_rad, az_min_rad, 100)
        radius_full = np.full_like(theta_wedge, 90)
        ax.fill_between(theta_wedge, 0, radius_full,
                        color='gray', alpha=0.1,
                        label=f'Az {az_max_deg:.0f}° to {az_min_deg:.0f}° (Invalid)')

  # Split track where satellite goes below horizon
  visible_mask = el_deg >= 0

  # Check which points satisfy tracker performance constraints
  constraint_valid_mask = np.ones(len(el_deg), dtype=bool)

  if tracker.performance:
    # Check elevation constraints
    if tracker.performance.elevation:
      el_min_deg = tracker.performance.elevation.min * CONVERTER.DEG_PER_RAD
      el_max_deg = tracker.performance.elevation.max * CONVERTER.DEG_PER_RAD
      constraint_valid_mask &= (el_deg >= el_min_deg) & (el_deg <= el_max_deg)

    # Check azimuth constraints
    if tracker.performance.azimuth:
      az_min_deg = tracker.performance.azimuth.min * CONVERTER.DEG_PER_RAD
      az_max_deg = tracker.performance.azimuth.max * CONVERTER.DEG_PER_RAD

      # Check if this covers the full circle (e.g., -180 to 180 or 0 to 360)
      az_range_deg = az_max_deg - az_min_deg
      if abs(az_range_deg - 360.0) < 1e-6:
        # Full circle coverage - all azimuths are valid
        pass  # Don't modify constraint_valid_mask
      elif az_max_deg < az_min_deg:
        # Wraps around: valid range is [az_min, 360] OR [0, az_max]
        constraint_valid_mask &= (az_deg >= az_min_deg) | (az_deg <= az_max_deg)
      else:
        # Normal case: valid range is [az_min, az_max]
        constraint_valid_mask &= (az_deg >= az_min_deg) & (az_deg <= az_max_deg)

  # Get range values
  range_m = topo.range

  # Check range constraints
  if tracker.performance and tracker.performance.range:
    range_min_m = tracker.performance.range.min
    range_max_m = tracker.performance.range.max
    constraint_valid_mask &= (range_m >= range_min_m) & (range_m <= range_max_m)
  
  # Compute marker sizes based on VISIBLE range only (closer = larger, further = smaller)
  visible_range = range_m[visible_mask]
  if len(visible_range) > 0:
    range_min = np.min(visible_range)
    range_max = np.max(visible_range)
  else:
    range_min = np.min(range_m)
    range_max = np.max(range_m)
  
  marker_size_min = 2
  marker_size_max = 100
  # Linear scale: closer (smaller range) -> larger marker
  if range_max > range_min:
    marker_sizes = marker_size_max - (range_m - range_min) / (range_max - range_min) * (marker_size_max - marker_size_min)
  else:
    marker_sizes = np.full_like(range_m, (marker_size_min + marker_size_max) / 2)
  
  # Find segments where satellite is visible
  segment_starts = []
  segment_ends = []
  in_segment = False
  
  for i in range(len(visible_mask)):
    if visible_mask[i] and not in_segment:
      segment_starts.append(i)
      in_segment = True
    elif not visible_mask[i] and in_segment:
      segment_ends.append(i)
      in_segment = False
  if in_segment:
    segment_ends.append(len(visible_mask))
  
  # Plot each visible segment
  for seg_start, seg_end in zip(segment_starts, segment_ends):
    seg_theta = theta[seg_start:seg_end]
    seg_radius = radius[seg_start:seg_end]
    seg_time = time_s[seg_start:seg_end]
    seg_marker_sizes = marker_sizes[seg_start:seg_end]
    seg_range = range_m[seg_start:seg_end]
    seg_constraint_valid = constraint_valid_mask[seg_start:seg_end]
    seg_el_deg = el_deg[seg_start:seg_end]
    seg_az_deg = az_deg[seg_start:seg_end]

    # Plot trajectory
    if len(seg_time) > 0:

      # Plot line segments based on point types:
      # - Gray solid line between gray (not valid) points
      # - Gray dashed line between gray and blue (transition) points
      # - Blue solid line between blue (valid) points
      for i in range(len(seg_theta) - 1):
        if seg_constraint_valid[i] and seg_constraint_valid[i+1]:
          # Both points valid (blue) - blue solid line
          ax.plot(seg_theta[i:i+2], seg_radius[i:i+2], 'b-', linewidth=2.0, alpha=0.8)
        elif not seg_constraint_valid[i] and not seg_constraint_valid[i+1]:
          # Both points not valid (gray) - gray solid line
          ax.plot(seg_theta[i:i+2], seg_radius[i:i+2], color='gray', linestyle='-', linewidth=1.5, alpha=0.8)
        else:
          # Transition between gray and blue - gray dashed line
          ax.plot(seg_theta[i:i+2], seg_radius[i:i+2], color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

      # Plot visible points with gray (do not connect)
      ax.scatter(seg_theta, seg_radius, c='gray', s=seg_marker_sizes, alpha=0.6)

      # Plot valid portions with blue markers
      if np.any(seg_constraint_valid):
        # Plot blue scatter points for valid portions
        valid_indices = np.where(seg_constraint_valid)[0]
        if len(valid_indices) > 0:
          ax.scatter(seg_theta[valid_indices], seg_radius[valid_indices],
                    c='blue', s=seg_marker_sizes[valid_indices], alpha=1.0)
      
      # Add entry marker if this is a true entry (satellite rose above horizon during propagation)
      # A true entry means there was a point before this segment that was below horizon
      is_true_entry = seg_start > 0
      if is_true_entry:
        ax.scatter([seg_theta[0]], [seg_radius[0]], s=120, marker='s', facecolors='none', 
                  edgecolors='black', linewidths=2, zorder=10)
        if epoch_dt_utc is not None:
          entry_dt = epoch_dt_utc + timedelta(seconds=seg_time[0])
          entry_label = entry_dt.strftime('%Y-%m-%d %H:%M:%S')
          ax.annotate(f'Entry\n{entry_label}', (seg_theta[0], seg_radius[0]), 
                     textcoords='offset points', xytext=(8, 8), fontsize=8,
                     color='black', fontweight='normal',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8))
      
      # Add exit marker if this is a true exit (satellite set below horizon during propagation)
      # A true exit means there's a point after this segment that's below horizon
      is_true_exit = seg_end < len(visible_mask)
      if is_true_exit:
        ax.scatter([seg_theta[-1]], [seg_radius[-1]], s=120, marker='s', facecolors='none',
                  edgecolors='black', linewidths=2, zorder=10)
        if epoch_dt_utc is not None:
          exit_dt = epoch_dt_utc + timedelta(seconds=seg_time[-1])
          exit_label = exit_dt.strftime('%Y-%m-%d %H:%M:%S')
          ax.annotate(f'Exit\n{exit_label}', (seg_theta[-1], seg_radius[-1]),
                     textcoords='offset points', xytext=(8, -12), fontsize=8,
                     color='black', fontweight='normal',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8))
  
  # Add initial marker at the first point of the entire propagation (if visible)
  if visible_mask[0]:
    ax.scatter([theta[0]], [radius[0]], s=120, marker='s', facecolors='none', 
              edgecolors='black', linewidths=2, zorder=10)
    if epoch_dt_utc is not None:
      initial_dt = epoch_dt_utc + timedelta(seconds=time_s[0])
      initial_label = initial_dt.strftime('%Y-%m-%d %H:%M:%S')
      ax.annotate(f'Initial\n{initial_label}', (theta[0], radius[0]), 
                 textcoords='offset points', xytext=(8, 8), fontsize=8,
                 color='black', fontweight='normal',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8))
  
  # Add final marker at the last point of the entire propagation (if visible)
  if visible_mask[-1]:
    ax.scatter([theta[-1]], [radius[-1]], s=120, marker='s', facecolors='none',
              edgecolors='black', linewidths=2, zorder=10)
    if epoch_dt_utc is not None:
      final_dt = epoch_dt_utc + timedelta(seconds=time_s[-1])
      final_label = final_dt.strftime('%Y-%m-%d %H:%M:%S')
      ax.annotate(f'Final\n{final_label}', (theta[-1], radius[-1]),
                 textcoords='offset points', xytext=(8, -12), fontsize=8,
                 color='black', fontweight='normal',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8))
  
  # Find and mark maximum elevation with UTC time
  max_el_idx = np.argmax(el_deg)
  if visible_mask[max_el_idx]:
    ax.scatter([theta[max_el_idx]], [radius[max_el_idx]], s=marker_size_max * 3.0, marker='s', 
              facecolors='none', edgecolors='black', linewidths=2, zorder=11,
              label=f'Max Elevation ({el_deg[max_el_idx]:.1f}°)')
    if epoch_dt_utc is not None:
      tca_dt = epoch_dt_utc + timedelta(seconds=time_s[max_el_idx])
      tca_label = tca_dt.strftime('%Y-%m-%d %H:%M:%S')
      ax.annotate(f'Max Elevation\n{tca_label}', (theta[max_el_idx], radius[max_el_idx]),
                 textcoords='offset points', xytext=(12, 13), fontsize=8,
                 color='black', fontweight='normal',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))
  
  # Find and mark minimum range (closest approach) with UTC time
  visible_indices = np.where(visible_mask)[0]
  if len(visible_indices) > 0:
    visible_ranges = range_m[visible_indices]
    min_range_visible_idx = np.argmin(visible_ranges)
    min_range_idx = visible_indices[min_range_visible_idx]
    min_range_km = range_m[min_range_idx] / 1000.0
    ax.scatter([theta[min_range_idx]], [radius[min_range_idx]], s=marker_size_max * 3.0, marker='s', 
              facecolors='none', edgecolors='black', linewidths=2, zorder=11,
              label=f'Min Range ({min_range_km:.0f} km)')
    if epoch_dt_utc is not None:
      min_range_dt = epoch_dt_utc + timedelta(seconds=time_s[min_range_idx])
      min_range_label = min_range_dt.strftime('%Y-%m-%d %H:%M:%S')
      ax.annotate(f'Min Range\n{min_range_label}', (theta[min_range_idx], radius[min_range_idx]),
                 textcoords='offset points', xytext=(13, -27), fontsize=8,
                 color='black', fontweight='normal',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))
  
  # Add cardinal direction labels (set ticks first to avoid warning)
  ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
  ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
  
  # Build info text
  tracker_lat_deg = tracker.position.latitude  * CONVERTER.DEG_PER_RAD
  tracker_lon_deg = tracker.position.longitude * CONVERTER.DEG_PER_RAD
  info_text = f"Station: {tracker.name}  |  Lat: {tracker_lat_deg:.4f}°  |  Lon: {tracker_lon_deg:.4f}°  |  Alt: {tracker.position.altitude:.1f} m"
  
  if epoch_dt_utc is not None:
    start_time_iso_utc = epoch_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time_dt_utc    = epoch_dt_utc + timedelta(seconds=time_s[-1])
    end_time_iso_utc   = end_time_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    info_text += f"\nInitial: {start_time_iso_utc}  |  Final: {end_time_iso_utc}"
  
  # Add visibility statistics
  n_visible = np.sum(visible_mask)
  n_total = len(visible_mask)
  visibility_pct = 100.0 * n_visible / n_total if n_total > 0 else 0.0
  max_elevation = np.max(el_deg)
  
  # Compute time step between points
  if len(time_s) > 1:
    dt_step = time_s[1] - time_s[0]
  else:
    dt_step = 0.0
  
  info_text += f"\nVisibility: {visibility_pct:.1f}%  |  Max Elevation: {max_elevation:.2f}°  |  Δt: {dt_step:.1f} s"
  
  # Add info text below plot
  fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=10, color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))
  
  # Add range-size legend box at bottom left of figure
  range_mid = (range_min + range_max) / 2.0
  range_min_km = range_min / 1000.0
  range_max_km = range_max / 1000.0
  range_mid_km = range_mid / 1000.0

  # Create a separate axes for the legend at bottom left
  legend_ax = fig.add_axes([0.05, 0.02, 0.15, 0.09])  # [left, bottom, width, height]
  legend_ax.set_xlim(0, 10)
  legend_ax.set_ylim(0, 10)
  legend_ax.axis('off')

  # Add border box FIRST (draw background before foreground elements)
  legend_ax.add_patch(plt.Rectangle((0, 0), 10, 10, fill=True, facecolor='white',
                                     edgecolor='black', alpha=1.0, linewidth=1, zorder=0))

  # Draw dots and range labels horizontally (closer = larger dot)
  legend_ax.scatter([1], [6], s=marker_size_max, c='blue', alpha=0.8, zorder=2)
  legend_ax.text(1, 4.5, f'{range_min_km:.0f} km\nmin', ha='center', va='top', fontsize=8, zorder=2)

  legend_ax.scatter([5], [6], s=(marker_size_min + marker_size_max) / 2, c='blue', alpha=0.8, zorder=2)
  legend_ax.text(5, 4.5, f'{range_mid_km:.0f} km\nmid', ha='center', va='top', fontsize=8, zorder=2)

  legend_ax.scatter([9], [6], s=marker_size_min, c='blue', alpha=0.8, zorder=2)
  legend_ax.text(9, 4.5, f'{range_max_km:.0f} km\nmax', ha='center', va='top', fontsize=8, zorder=2)

  legend_ax.text(5, 8.5, 'Range', ha='center', fontsize=9, fontweight='bold', zorder=2)

  ax.set_title(title_text, fontsize=14, pad=20)

  # ============================================
  # Time-series plots on the right column
  # ============================================

  # Convert time to hours for better readability
  time_hrs = time_s / 3600.0

  # Create three subplots in the right column (stacked vertically)
  ax_range = fig.add_subplot(3, 2, 2)
  ax_az    = fig.add_subplot(3, 2, 4)
  ax_el    = fig.add_subplot(3, 2, 6)

  # Plot Range vs Time
  range_km = topo.range / 1000.0

  # Thin black line for entire solution (not in legend)
  ax_range.plot(time_hrs, range_km, 'k-', linewidth=0.5, alpha=0.8)
  ax_range.set_ylabel('Range [km]', fontsize=11)
  ax_range.grid(True, alpha=0.3)
  ax_range.set_title('Truth Measurements vs Time', fontsize=12)

  # Find segments for visible portions
  visible_segment_starts = []
  visible_segment_ends = []
  in_visible_segment = False

  for i in range(len(visible_mask)):
    if visible_mask[i] and not in_visible_segment:
      visible_segment_starts.append(i)
      in_visible_segment = True
    elif not visible_mask[i] and in_visible_segment:
      visible_segment_ends.append(i)
      in_visible_segment = False
  if in_visible_segment:
    visible_segment_ends.append(len(visible_mask))

  # Plot gray lines and markers for each visible segment
  gray_plotted = False
  for seg_start, seg_end in zip(visible_segment_starts, visible_segment_ends):
    label = 'Above Horizon' if not gray_plotted else None
    ax_range.plot(time_hrs[seg_start:seg_end], range_km[seg_start:seg_end],
                  color='gray', linewidth=2.5, alpha=0.6, marker='o', markersize=4, label=label)
    gray_plotted = True

  # Find segments for valid portions (same segment logic)
  valid_segment_starts = []
  valid_segment_ends = []
  in_valid_segment = False

  for i in range(len(constraint_valid_mask)):
    if constraint_valid_mask[i] and not in_valid_segment:
      valid_segment_starts.append(i)
      in_valid_segment = True
    elif not constraint_valid_mask[i] and in_valid_segment:
      valid_segment_ends.append(i)
      in_valid_segment = False
  if in_valid_segment:
    valid_segment_ends.append(len(constraint_valid_mask))

  # Plot blue lines for each valid segment (do not connect across segments)
  blue_plotted = False
  for seg_start, seg_end in zip(valid_segment_starts, valid_segment_ends):
    label = 'Trackable' if not blue_plotted else None
    ax_range.plot(time_hrs[seg_start:seg_end], range_km[seg_start:seg_end],
                  'b-', linewidth=3.5, alpha=0.8, label=label)
    blue_plotted = True

  # Add horizontal lines for range constraints (if they exist)
  if tracker.performance and tracker.performance.range:
    range_min_km = tracker.performance.range.min / 1000.0
    range_max_km = tracker.performance.range.max / 1000.0
    if range_min_km > 0:
      ax_range.axhline(y=range_min_km, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Min Range ({range_min_km:.0f} km)')
    if range_max_km < np.inf:
      ax_range.axhline(y=range_max_km, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Max Range ({range_max_km:.0f} km)')

  ax_range.legend(loc='best', fontsize=9)
  # Hide x-axis tick labels for range plot
  ax_range.set_xticklabels([])

  # Plot Azimuth vs Time
  # Thin black line for entire solution (not in legend)
  ax_az.plot(time_hrs, az_deg, 'k-', linewidth=0.5, alpha=0.8)
  ax_az.set_ylabel('Azimuth [deg]', fontsize=11)
  ax_az.grid(True, alpha=0.3)

  # Plot gray lines and markers for each visible segment
  gray_plotted = False
  for seg_start, seg_end in zip(visible_segment_starts, visible_segment_ends):
    label = 'Above Horizon' if not gray_plotted else None
    ax_az.plot(time_hrs[seg_start:seg_end], az_deg[seg_start:seg_end],
               color='gray', linewidth=2.5, alpha=0.6, marker='o', markersize=4, label=label)
    gray_plotted = True

  # Plot blue lines for each valid segment (do not connect across segments)
  blue_plotted = False
  for seg_start, seg_end in zip(valid_segment_starts, valid_segment_ends):
    label = 'Trackable' if not blue_plotted else None
    ax_az.plot(time_hrs[seg_start:seg_end], az_deg[seg_start:seg_end],
               'b-', linewidth=3.5, alpha=0.8, label=label)
    blue_plotted = True

  # Add horizontal lines for azimuth constraints (if they exist)
  if tracker.performance and tracker.performance.azimuth:
    az_min_constraint = tracker.performance.azimuth.min * CONVERTER.DEG_PER_RAD
    az_max_constraint = tracker.performance.azimuth.max * CONVERTER.DEG_PER_RAD
    ax_az.axhline(y=az_min_constraint, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Min Az ({az_min_constraint:.0f}°)')
    ax_az.axhline(y=az_max_constraint, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Max Az ({az_max_constraint:.0f}°)')

  ax_az.legend(loc='best', fontsize=9)
  # Hide x-axis tick labels for azimuth plot
  ax_az.set_xticklabels([])

  # Plot Elevation vs Time
  # Convert time_s to UTC datetime for x-axis
  if epoch_dt_utc is not None:
    time_utc = [epoch_dt_utc + timedelta(seconds=float(t)) for t in time_s]
  else:
    # Fallback to hours if no epoch provided
    time_utc = time_hrs

  # Thin black line for entire solution (not in legend)
  if epoch_dt_utc is not None:
    ax_el.plot(time_utc, el_deg, 'k-', linewidth=0.5, alpha=0.8)
  else:
    ax_el.plot(time_hrs, el_deg, 'k-', linewidth=0.5, alpha=0.8)

  ax_el.set_ylabel('Elevation [deg]', fontsize=11)
  ax_el.set_xlabel('UTC Time', fontsize=11)
  ax_el.grid(True, alpha=0.3)

  # Plot gray lines and markers for each visible segment
  gray_plotted = False
  for seg_start, seg_end in zip(visible_segment_starts, visible_segment_ends):
    label = 'Above Horizon' if not gray_plotted else None
    if epoch_dt_utc is not None:
      ax_el.plot(time_utc[seg_start:seg_end], el_deg[seg_start:seg_end],
                 color='gray', linewidth=2.5, alpha=0.6, marker='o', markersize=4, label=label)
    else:
      ax_el.plot(time_hrs[seg_start:seg_end], el_deg[seg_start:seg_end],
                 color='gray', linewidth=2.5, alpha=0.6, marker='o', markersize=4, label=label)
    gray_plotted = True

  # Plot blue lines for each valid segment (do not connect across segments)
  blue_plotted = False
  for seg_start, seg_end in zip(valid_segment_starts, valid_segment_ends):
    label = 'Trackable' if not blue_plotted else None
    if epoch_dt_utc is not None:
      ax_el.plot(time_utc[seg_start:seg_end], el_deg[seg_start:seg_end],
                 'b-', linewidth=3.5, alpha=0.8, label=label)
    else:
      ax_el.plot(time_hrs[seg_start:seg_end], el_deg[seg_start:seg_end],
                 'b-', linewidth=3.5, alpha=0.8, label=label)
    blue_plotted = True

  # Add horizontal line at elevation = 0 (horizon)
  ax_el.axhline(y=0, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label='Horizon')

  # Add horizontal lines for elevation constraints (if they exist)
  if tracker.performance and tracker.performance.elevation:
    el_min_constraint = tracker.performance.elevation.min * CONVERTER.DEG_PER_RAD
    el_max_constraint = tracker.performance.elevation.max * CONVERTER.DEG_PER_RAD
    if el_min_constraint > 0:
      ax_el.axhline(y=el_min_constraint, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Min El ({el_min_constraint:.0f}°)')
    if el_max_constraint < 90:
      ax_el.axhline(y=el_max_constraint, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Max El ({el_max_constraint:.0f}°)')

  # Rotate x-axis labels for better readability
  if epoch_dt_utc is not None:
    ax_el.tick_params(axis='x', rotation=45)
    # Format x-axis to show time nicely
    import matplotlib.dates as mdates
    ax_el.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=45)

  ax_el.legend(loc='best', fontsize=9)

  # Adjust layout
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    plt.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))

  return fig


def generate_3d_and_time_series_plots(
  result_jpl_horizons_ephemeris    : Optional[PropagationResult],
  result_high_fidelity_propagation : PropagationResult,
  result_sgp4_propagation          : Optional[PropagationResult],
  time_o_dt                        : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool,
  compare_tle                      : bool,
  object_name                      : str = "object",
  object_name_display              : str = "Object",
  trackers                         : Optional[list['TrackerStation']] = None,
  include_tracker_on_body          : bool = False,
) -> dict:
  """
  Generate and save 3D trajectory and time series plots.
  
  Input:
  ------
    result_horizons : PropagationResult | None
      Horizons ephemeris result.
    result_high_fidelity : PropagationResult
      High-fidelity propagation result.
    result_sgp4 : PropagationResult | None
      SGP4 propagation result.
    time_o_dt : datetime
      Simulation start time (for plot labels).
    figures_folderpath : Path
      Directory to save plots.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    compare_tle : bool
      Flag to enable comparison with TLE/SGP4.
    object_name : str
      Sanitized name of the object for filenames.
    object_name_display : str
      Original name of the object for plot titles.
      
  Output:
  -------
    dict : Dictionary with plot filenames organized by source type.
  """
  plot_files = {
    'jpl_horizons': {},
    'high_fidelity': {},
    'sgp4': {},
  }

  # Lowercase name for filenames
  name_lower = object_name.lower()

  # Horizons plots
  if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success:
    fig1 = plot_3d_trajectories(result_jpl_horizons_ephemeris, epoch=time_o_dt, frame="J2000")
    fig1.suptitle(f'3D Inertial - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'3d_j2000_earth_centered_jpl_horizons_{name_lower}.png'
    fig1.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['jpl_horizons']['3d_inertial'] = filename
    plt.close(fig1)

    fig2 = plot_time_series(result_jpl_horizons_ephemeris, epoch=time_o_dt)
    fig2.suptitle(f'Time Series - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'timeseries_jpl_horizons_{name_lower}.png'
    fig2.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['jpl_horizons']['time_series'] = filename
    plt.close(fig2)

    # Body-fixed 3D plot for Horizons
    fig_ef = plot_3d_trajectories_body_fixed(result_jpl_horizons_ephemeris, epoch_dt_utc=time_o_dt, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    fig_ef.suptitle(f'3D Body-Fixed - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'3d_iau_earth_jpl_horizons_{name_lower}.png'
    fig_ef.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['jpl_horizons']['3d_body_fixed'] = filename
    plt.close(fig_ef)

    # Ground track plot for Horizons
    gt_title = f'Ground Track - {object_name_display} - JPL Horizons'
    fig_gt = plot_ground_track(result_jpl_horizons_ephemeris, epoch_dt_utc=time_o_dt, title_text=gt_title, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    filename = f'groundtrack_jpl_horizons_{name_lower}.png'
    fig_gt.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['jpl_horizons']['ground_track'] = filename
    plt.close(fig_gt)
  
  # High-fidelity plots
  if result_high_fidelity_propagation.success:
    fig3 = plot_3d_trajectories(result_high_fidelity_propagation, epoch=time_o_dt, frame="J2000")
    fig3.suptitle(f'3D Inertial - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_j2000_earth_centered_high_fidelity_{name_lower}.png'
    fig3.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['3d_inertial'] = filename
    plt.close(fig3)

    fig4 = plot_time_series(result_high_fidelity_propagation, epoch=time_o_dt)
    fig4.suptitle(f'Time Series - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'timeseries_high_fidelity_{name_lower}.png'
    fig4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['time_series'] = filename
    plt.close(fig4)

    # Body-fixed 3D plot
    fig_ef = plot_3d_trajectories_body_fixed(result_high_fidelity_propagation, epoch_dt_utc=time_o_dt, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    fig_ef.suptitle(f'3D Body-Fixed - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_iau_earth_high_fidelity_{name_lower}.png'
    fig_ef.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['3d_body_fixed'] = filename
    plt.close(fig_ef)

    # Ground track plot
    gt_title = f'Ground Track - {object_name_display} - High-Fidelity'
    fig_gt = plot_ground_track(result_high_fidelity_propagation, epoch_dt_utc=time_o_dt, title_text=gt_title, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    filename = f'groundtrack_high_fidelity_{name_lower}.png'
    fig_gt.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['ground_track'] = filename
    plt.close(fig_gt)

    # 3D plot Sun-centered trajectory
    fig_moon = plot_3d_trajectory_sun_centered(result_high_fidelity_propagation, epoch=time_o_dt)
    fig_moon.suptitle(f'3D Inertial - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_j2000_sun_centered_high_fidelity_{name_lower}.png'
    fig_moon.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['high_fidelity']['3d_sun_centered'] = filename
    plt.close(fig_moon)
  
  # SGP4 plots
  if compare_tle and result_sgp4_propagation and result_sgp4_propagation.success:
    # 3D trajectory plot
    fig_sgp4_3d = plot_3d_trajectories(result_sgp4_propagation, epoch=time_o_dt, frame="J2000")
    fig_sgp4_3d.suptitle(f'3D Inertial - {object_name_display} - SGP4', fontsize=16)
    filename = f'3d_j2000_earth_centered_sgp4_{name_lower}.png'
    fig_sgp4_3d.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['sgp4']['3d_inertial'] = filename
    plt.close(fig_sgp4_3d)
    
    # Time series plot
    fig_sgp4_ts = plot_time_series(result_sgp4_propagation, epoch=time_o_dt)
    fig_sgp4_ts.suptitle(f'Time Series - {object_name_display} - SGP4', fontsize=16)
    filename = f'timeseries_sgp4_{name_lower}.png'
    fig_sgp4_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['sgp4']['time_series'] = filename
    plt.close(fig_sgp4_ts)

    # Ground track plot for SGP4
    gt_title = f'Ground Track - {object_name_display} - SGP4'
    fig_gt_sgp4 = plot_ground_track(result_sgp4_propagation, epoch_dt_utc=time_o_dt, title_text=gt_title, trackers=trackers, include_tracker_on_body=include_tracker_on_body)
    filename = f'groundtrack_sgp4_{name_lower}.png'
    fig_gt_sgp4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    plot_files['sgp4']['ground_track'] = filename
    plt.close(fig_gt_sgp4)

  return plot_files

def generate_plots(
  result_jpl_horizons_ephemeris    : Optional[PropagationResult],
  result_high_fidelity_propagation : PropagationResult,
  result_sgp4_propagation          : Optional[PropagationResult],
  time_o_dt                        : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool = False,
  compare_tle                      : bool = False,
  object_name                      : str  = "object",
  object_name_display              : str  = "Object",
  trackers                         : Optional[list['TrackerStation']] = None,
  include_tracker_on_body          : bool = False,
) -> None:
  """
  Generate and save all simulation plots.

  Input:
  ------
    result_horizons : PropagationResult | None
      Horizons ephemeris result.
    result_high_fidelity : PropagationResult
      High-fidelity propagation result.
    result_sgp4 : PropagationResult | None
      SGP4 propagation result.
    time_o_dt : datetime
      Simulation start time (for plot labels).
    figures_folderpath : Path
      Directory to save plots.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    compare_tle : bool
      Flag to enable comparison with TLE/SGP4.
    object_name : str
      Sanitized name of the object for filenames.
    object_name_display : str
      Original name of the object for plot titles.
    tracker : TrackerStation | None
      Tracker station object with normalized azimuth values.
    include_tracker_on_body : bool
      Flag to show tracker location on ground track and 3D body-fixed plots.

  Output:
  -------
    None
  """
  title = "Generate and Save Plots"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))

  print()
  print("  Progress")

  # Generate 3D and time series plots
  print("    Generate 3D-trajectory, time-series, and groundtrack plots")
  plot_files = generate_3d_and_time_series_plots(
    result_jpl_horizons_ephemeris    = result_jpl_horizons_ephemeris,
    result_high_fidelity_propagation = result_high_fidelity_propagation,
    result_sgp4_propagation          = result_sgp4_propagation,
    time_o_dt                        = time_o_dt,
    figures_folderpath               = figures_folderpath,
    compare_jpl_horizons             = compare_jpl_horizons,
    compare_tle                      = compare_tle,
    object_name                      = object_name,
    object_name_display              = object_name_display,
    trackers                         = trackers,
    include_tracker_on_body          = include_tracker_on_body,
  )

  # Generate error plots only if a comparison was requested
  error_files = {'hf_vs_horizons': [], 'hf_vs_sgp4': [], 'sgp4_vs_horizons': []}
  if compare_jpl_horizons or compare_tle:
    print("    Generate error plots")
    error_files = generate_error_plots(
      result_jpl_horizons_ephemeris    = result_jpl_horizons_ephemeris,
      result_high_fidelity_propagation = result_high_fidelity_propagation,
      result_sgp4_propagation          = result_sgp4_propagation,
      time_o_dt                        = time_o_dt,
      figures_folderpath               = figures_folderpath,
      compare_jpl_horizons             = compare_jpl_horizons,
      compare_tle                      = compare_tle,
      object_name                      = object_name,
      object_name_display              = object_name_display,
    )

  # Generate skyplots for each tracker
  skyplot_files = {}  # Dict of tracker_name -> list of filenames
  if trackers is not None and len(trackers) > 0:
    print("    Generate skyplots")

    name_lower = object_name.lower().replace(' ', '_').replace('-', '_')

    for tracker in trackers:
      try:
        tracker_name_sanitized = tracker.name.lower().replace(' ', '_').replace('-', '_')

        # Collect filenames for this tracker
        filenames = []

        # Generate skyplot for high-fidelity propagation
        if result_high_fidelity_propagation.success:
          skyplot_title = f'Skyplot - {object_name_display} - High-Fidelity - {tracker.name}'
          fig_skyplot = plot_skyplot(
            result       = result_high_fidelity_propagation,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = skyplot_title,
          )
          filename = f'skyplot_{tracker_name_sanitized}_high_fidelity_{name_lower}.png'
          fig_skyplot.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
          plt.close(fig_skyplot)
          filenames.append(filename)

        # Generate skyplot for SGP4 if available
        if compare_tle and result_sgp4_propagation and result_sgp4_propagation.success:
          skyplot_title = f'Skyplot - {object_name_display} - SGP4 - {tracker.name}'
          fig_skyplot_sgp4 = plot_skyplot(
            result       = result_sgp4_propagation,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = skyplot_title,
          )
          filename = f'skyplot_{tracker_name_sanitized}_sgp4_{name_lower}.png'
          fig_skyplot_sgp4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
          plt.close(fig_skyplot_sgp4)
          filenames.append(filename)

        # Generate skyplot for JPL Horizons if available
        if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.success:
          skyplot_title = f'Skyplot - {object_name_display} - JPL Horizons - {tracker.name}'
          fig_skyplot_horizons = plot_skyplot(
            result       = result_jpl_horizons_ephemeris,
            tracker      = tracker,
            epoch_dt_utc = time_o_dt,
            title_text   = skyplot_title,
          )
          filename = f'skyplot_{tracker_name_sanitized}_jpl_horizons_{name_lower}.png'
          fig_skyplot_horizons.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
          plt.close(fig_skyplot_horizons)
          filenames.append(filename)

        skyplot_files[tracker.name] = filenames

      except Exception as e:
        print(f"      [WARNING] Failed to generate skyplot for {tracker.name}: {e}")

  print()
  print("  Summary")
  print(f"    Figure Folderpath : {figures_folderpath}")
  print()
  
  # Print 3D-Trajectory, Time-Series, and Groundtrack Plots
  print("    3D-Trajectory, Time-Series, and Groundtrack Plots")
  
  if plot_files['jpl_horizons']:
    print("      JPL-Horizons-Ephemeris Plots")
    if '3d_inertial' in plot_files['jpl_horizons']:
      print(f"        3D Inertial    : <figures_folderpath>/{plot_files['jpl_horizons']['3d_inertial']}")
    if 'time_series' in plot_files['jpl_horizons']:
      print(f"        Time Series    : <figures_folderpath>/{plot_files['jpl_horizons']['time_series']}")
    if '3d_body_fixed' in plot_files['jpl_horizons']:
      print(f"        3D Body-Fixed  : <figures_folderpath>/{plot_files['jpl_horizons']['3d_body_fixed']}")
    if 'ground_track' in plot_files['jpl_horizons']:
      print(f"        Ground Track   : <figures_folderpath>/{plot_files['jpl_horizons']['ground_track']}")
  
  if plot_files['high_fidelity']:
    print("      High-Fidelity-Model Plots")
    if '3d_inertial' in plot_files['high_fidelity']:
      print(f"        3D Inertial    : <figures_folderpath>/{plot_files['high_fidelity']['3d_inertial']}")
    if 'time_series' in plot_files['high_fidelity']:
      print(f"        Time Series    : <figures_folderpath>/{plot_files['high_fidelity']['time_series']}")
    if '3d_body_fixed' in plot_files['high_fidelity']:
      print(f"        3D Body-Fixed  : <figures_folderpath>/{plot_files['high_fidelity']['3d_body_fixed']}")
    if 'ground_track' in plot_files['high_fidelity']:
      print(f"        Ground Track   : <figures_folderpath>/{plot_files['high_fidelity']['ground_track']}")
    if '3d_sun_centered' in plot_files['high_fidelity']:
      print(f"        3D Sun-Centered: <figures_folderpath>/{plot_files['high_fidelity']['3d_sun_centered']}")
  
  if plot_files['sgp4']:
    print("      SGP4-Model Plots")
    if '3d_inertial' in plot_files['sgp4']:
      print(f"        3D Inertial    : <figures_folderpath>/{plot_files['sgp4']['3d_inertial']}")
    if 'time_series' in plot_files['sgp4']:
      print(f"        Time Series    : <figures_folderpath>/{plot_files['sgp4']['time_series']}")
    if 'ground_track' in plot_files['sgp4']:
      print(f"        Ground Track   : <figures_folderpath>/{plot_files['sgp4']['ground_track']}")
  
  # Print Error Plots
  has_error_plots = any(error_files[k] for k in error_files)
  if has_error_plots:
    print()
    print("    Error Plots")
    if error_files['hf_vs_horizons']:
      print("      High-Fidelity Relative To JPL Horizons")
      for filename in error_files['hf_vs_horizons']:
        print(f"        Time-Series Error : <figures_folderpath>/{filename}")
    if error_files['hf_vs_sgp4']:
      print("      High-Fidelity Relative To SGP4")
      for filename in error_files['hf_vs_sgp4']:
        print(f"        Time-Series Error : <figures_folderpath>/{filename}")
    if error_files['sgp4_vs_horizons']:
      print("      SGP4 Relative To JPL Horizons")
      for filename in error_files['sgp4_vs_horizons']:
        print(f"        Time-Series Error : <figures_folderpath>/{filename}")
  
  # Print Skyplots
  if skyplot_files:
    print()
    print("    Skyplots")
    for tracker_name, filenames in skyplot_files.items():
      print(f"      Tracker {tracker_name}")
      for filename in filenames:
        print(f"        <figures_folderpath>/{filename}")