import datetime

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

from src.plot.utility          import get_equal_limits, add_utc_time_axis
from src.model.constants       import CONVERTER, SOLARSYSTEMCONSTANTS
from src.model.frame_converter import FrameConverter
from src.model.time_converter  import utc_to_et
from src.model.orbit_converter import GeographicCoordinateConverter, OrbitConverter


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
  result : dict,
  epoch  : Optional[datetime.datetime] = None,
  frame  : str = "J2000",
) -> Figure:
  """
  Plot 3D position and velocity trajectories in a 1x2 grid.
  
  Input:
  ------
    result : dict
      Propagation result dictionary containing 'state' (6xN array) and 'plot_time_s'.
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
  posvel_vec = result['state']
  pos_x, pos_y, pos_z = posvel_vec[0, :], posvel_vec[1, :], posvel_vec[2, :]
  vel_x, vel_y, vel_z = posvel_vec[3, :], posvel_vec[4, :], posvel_vec[5, :]
  
  # Build info string with frame and time if epoch is provided
  info_text = f"Frame: {frame}"
  if epoch is not None and 'plot_time_s' in result:
    start_utc = epoch.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time  = epoch + timedelta(seconds=result['plot_time_s'][-1])
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
  if epoch is not None and frame == "J2000" and 'plot_time_s' in result:
    try:
      # Get start and end times
      epoch_et_start = utc_to_et(epoch)
      epoch_et_end = epoch_et_start + result['plot_time_s'][-1]
      
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
  result : dict,
  epoch  : Optional[datetime.datetime] = None,
) -> Figure:
  """
  Plot position and velocity components vs time in a 3-column grid.
  
  Input:
  ------
    result : dict
      Propagation result dictionary containing 'plot_time_s', 'state', 'coe', and 'mee'.
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
  time = result['plot_time_s']
  states = result['state']
  pos_x, pos_y, pos_z = states[0, :], states[1, :], states[2, :]
  vel_x, vel_y, vel_z = states[3, :], states[4, :], states[5, :]
  coe = result['coe']
  mee = result['mee']
  
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
  ax_sma.plot(time, coe['sma'], 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot ecc vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 3), (1, 1), sharex=ax_pos)
  ax_ecc.plot(time, coe['ecc'], 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('ECC\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inc vs time (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 3), (2, 1), sharex=ax_pos)
  ax_inc.plot(time, coe['inc'] * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('INC\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot raan vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 3), (3, 1), sharex=ax_pos)
  ax_raan.plot(time, coe['raan'] * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN\n[deg]')
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot aop vs time (row 4, column 1)
  ax_aop = plt.subplot2grid((6, 3), (4, 1), sharex=ax_pos)
  aop_unwrapped = np.unwrap(coe['aop']) * CONVERTER.DEG_PER_RAD
  ax_aop.plot(time, aop_unwrapped, 'b-', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('AOP\n[deg]')
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot ta, ea, ma vs time (row 5, column 1)
  ax_anom = plt.subplot2grid((6, 3), (5, 1), sharex=ax_pos)
  ax_anom.plot(time, coe['ta'] * CONVERTER.DEG_PER_RAD, 'r-', label='TA', linewidth=1.5)
  ax_anom.plot(time, coe['ea'] * CONVERTER.DEG_PER_RAD, 'g-', label='EA', linewidth=1.5)
  ax_anom.plot(time, coe['ma'] * CONVERTER.DEG_PER_RAD, 'b-', label='MA', linewidth=1.5)
  ax_anom.set_xlabel('Time\n[s]')
  ax_anom.set_ylabel('ANOMALY\n[deg]')
  ax_anom.legend(fontsize=8)
  ax_anom.grid(True)
  ax_anom.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RIGHT COLUMN: Modified Equinoctial Elements
  # Plot p (semi-latus rectum) vs time (row 0, column 2)
  ax_p = plt.subplot2grid((6, 3), (0, 2), sharex=ax_pos)
  ax_p.plot(time, mee['p'], 'b-', linewidth=1.5)
  ax_p.tick_params(labelbottom=False)
  ax_p.set_ylabel('p\n[m]')
  ax_p.grid(True)
  ax_p.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot f vs time (row 1, column 2)
  ax_f = plt.subplot2grid((6, 3), (1, 2), sharex=ax_pos)
  ax_f.plot(time, mee['f'], 'b-', linewidth=1.5)
  ax_f.tick_params(labelbottom=False)
  ax_f.set_ylabel('f\n[-]')
  ax_f.grid(True)
  ax_f.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot g vs time (row 2, column 2)
  ax_g = plt.subplot2grid((6, 3), (2, 2), sharex=ax_pos)
  ax_g.plot(time, mee['g'], 'b-', linewidth=1.5)
  ax_g.tick_params(labelbottom=False)
  ax_g.set_ylabel('g\n[-]')
  ax_g.grid(True)
  ax_g.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot h vs time (row 3, column 2)
  ax_h = plt.subplot2grid((6, 3), (3, 2), sharex=ax_pos)
  ax_h.plot(time, mee['h'], 'b-', linewidth=1.5)
  ax_h.tick_params(labelbottom=False)
  ax_h.set_ylabel('h\n[-]')
  ax_h.grid(True)
  ax_h.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot k vs time (row 4, column 2)
  ax_k = plt.subplot2grid((6, 3), (4, 2), sharex=ax_pos)
  ax_k.plot(time, mee['k'], 'b-', linewidth=1.5)
  ax_k.tick_params(labelbottom=False)
  ax_k.set_ylabel('k\n[-]')
  ax_k.grid(True)
  ax_k.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot L (true longitude) vs time (row 5, column 2)
  ax_L = plt.subplot2grid((6, 3), (5, 2), sharex=ax_pos)
  ax_L.plot(time, mee['L'] * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
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
  result_ref : dict,
  result_comp : dict,
  title      : str = "Position and Velocity Error",
) -> Figure:
  """
  Plot 3D position and velocity error trajectories in a 1x2 grid.
  
  Input:
  ------
    result_ref : dict
      Reference result dictionary (e.g., SGP4).
    result_comp : dict
      Comparison result dictionary (e.g., high-fidelity).
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
  
  time_ref   = result_ref['delta_time']
  time_comp  = result_comp['time']
  state_comp = result_comp['state']
  
  # Interpolate each state component
  state_comp_interp = np.zeros((6, len(time_ref)))
  for i in range(6):
    interpolator = interp1d(time_comp, state_comp[i, :], kind='cubic', fill_value='extrapolate') # type: ignore
    state_comp_interp[i, :] = interpolator(time_ref)
  
  # Calculate errors (comparison - reference)
  state_ref = result_ref['state']
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
  result_ref  : dict, 
  result_comp : dict, 
  epoch       : Optional[datetime.datetime] = None, 
  title       : str                         = "Time Series Error", 
  use_ric     : bool                        = True,
) -> Figure:
  """
  Create time series error plots between reference and comparison trajectories.
  
  Input:
  ------
    result_ref : dict
      Reference result dictionary with 'plot_time_s' and 'state'/'coe'/'mee'.
    result_comp : dict
      Comparison result dictionary with 'plot_time_s' and 'state'/'coe'/'mee'.
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
  time_ref  = result_ref['plot_time_s']
  time_comp = result_comp['plot_time_s']
  
  state_ref  = result_ref['state']
  state_comp = result_comp['state']
  coe_ref    = result_ref['coe']
  coe_comp   = result_comp['coe']
  mee_ref    = result_ref['mee']
  mee_comp   = result_comp['mee']
  
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
  if 'sma' in coe_ref and 'sma' in coe_comp:
    sma_error = coe_comp['sma'] - coe_ref['sma']
    ax_sma.plot(time, sma_error, 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA Error\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot eccentricity error vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 3), (1, 1), sharex=ax_pos)
  if 'ecc' in coe_ref and 'ecc' in coe_comp:
    ecc_error = coe_comp['ecc'] - coe_ref['ecc']
    ax_ecc.plot(time, ecc_error, 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('ECC Error\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inclination error (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 3), (2, 1), sharex=ax_pos)
  if 'inc' in coe_ref and 'inc' in coe_comp:
    inc_error = (coe_comp['inc'] - coe_ref['inc']) * CONVERTER.DEG_PER_RAD
    ax_inc.plot(time, inc_error, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('INC Error\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot RAAN error vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 3), (3, 1), sharex=ax_pos)
  if 'raan' in coe_ref and 'raan' in coe_comp:
    # Handle angle wrapping for RAAN
    raan_error_rad = np.arctan2(np.sin(coe_comp['raan'] - coe_ref['raan']), 
                   np.cos(coe_comp['raan'] - coe_ref['raan']))
    raan_error = raan_error_rad * CONVERTER.DEG_PER_RAD
    ax_raan.plot(time, raan_error, 'b-', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN Error\n[deg]')
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot argument of periapsis error (row 4, column 1)
  ax_aop = plt.subplot2grid((6, 3), (4, 1), sharex=ax_pos)
  if 'aop' in coe_ref and 'aop' in coe_comp:
    # Handle angle wrapping for AOP
    aop_error_rad = np.arctan2(np.sin(coe_comp['aop'] - coe_ref['aop']), 
                   np.cos(coe_comp['aop'] - coe_ref['aop']))
    aop_error = aop_error_rad * CONVERTER.DEG_PER_RAD
    ax_aop.plot(time, aop_error, 'b-', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('AOP Error\n[deg]')
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot true anomaly error vs time (row 5, column 1)
  ax_ta = plt.subplot2grid((6, 3), (5, 1), sharex=ax_pos)
  if 'ta' in coe_ref and 'ta' in coe_comp:
    # Handle angle wrapping for TA
    ta_error_rad = np.arctan2(np.sin(coe_comp['ta'] - coe_ref['ta']), 
                   np.cos(coe_comp['ta'] - coe_ref['ta']))
    ta_error = ta_error_rad * CONVERTER.DEG_PER_RAD
    ax_ta.plot(time, ta_error, 'b-', linewidth=1.5)
  ax_ta.set_xlabel('Time\n[s]')
  ax_ta.set_ylabel('TA Error\n[deg]')
  ax_ta.grid(True)
  ax_ta.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RIGHT COLUMN: Modified Equinoctial Elements Errors
  # Plot p error vs time (row 0, column 2)
  ax_p = plt.subplot2grid((6, 3), (0, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and 'p' in mee_ref and 'p' in mee_comp:
    p_error = mee_comp['p'] - mee_ref['p']
    ax_p.plot(time, p_error, 'b-', linewidth=1.5)
  ax_p.tick_params(labelbottom=False)
  ax_p.set_ylabel('p Error\n[m]')
  ax_p.grid(True)
  ax_p.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot f error vs time (row 1, column 2)
  ax_f = plt.subplot2grid((6, 3), (1, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and 'f' in mee_ref and 'f' in mee_comp:
    f_error = mee_comp['f'] - mee_ref['f']
    ax_f.plot(time, f_error, 'b-', linewidth=1.5)
  ax_f.tick_params(labelbottom=False)
  ax_f.set_ylabel('f Error\n[-]')
  ax_f.grid(True)
  ax_f.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot g error vs time (row 2, column 2)
  ax_g = plt.subplot2grid((6, 3), (2, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and 'g' in mee_ref and 'g' in mee_comp:
    g_error = mee_comp['g'] - mee_ref['g']
    ax_g.plot(time, g_error, 'b-', linewidth=1.5)
  ax_g.tick_params(labelbottom=False)
  ax_g.set_ylabel('g Error\n[-]')
  ax_g.grid(True)
  ax_g.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot h error vs time (row 3, column 2)
  ax_h = plt.subplot2grid((6, 3), (3, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and 'h' in mee_ref and 'h' in mee_comp:
    h_error = mee_comp['h'] - mee_ref['h']
    ax_h.plot(time, h_error, 'b-', linewidth=1.5)
  ax_h.tick_params(labelbottom=False)
  ax_h.set_ylabel('h Error\n[-]')
  ax_h.grid(True)
  ax_h.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot k error vs time (row 4, column 2)
  ax_k = plt.subplot2grid((6, 3), (4, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and 'k' in mee_ref and 'k' in mee_comp:
    k_error = mee_comp['k'] - mee_ref['k']
    ax_k.plot(time, k_error, 'b-', linewidth=1.5)
  ax_k.tick_params(labelbottom=False)
  ax_k.set_ylabel('k Error\n[-]')
  ax_k.grid(True)
  ax_k.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot L error vs time (row 5, column 2)
  ax_L = plt.subplot2grid((6, 3), (5, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and 'L' in mee_ref and 'L' in mee_comp:
    # Handle angle wrapping for L (true longitude)
    L_error_rad = np.arctan2(np.sin(mee_comp['L'] - mee_ref['L']), 
                   np.cos(mee_comp['L'] - mee_ref['L']))
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
  result       : dict,
  epoch_dt_utc : Optional[datetime.datetime] = None,
) -> Figure:
  """
  Plot 3D position and velocity trajectories in body-fixed (IAU_EARTH) frame.
  
  Input:
  ------
    result : dict
      Propagation result dictionary containing 'state' (6xN array) and 'plot_time_s'.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the 3D plots.
  """
  fig = plt.figure(figsize=(18, 10))
  
  # Extract J2000 state vectors
  j2000_state   = result['state']
  j2000_pos_vec = j2000_state[0:3, :]
  j2000_vel_vec = j2000_state[3:6, :]
  time_s        = result['plot_time_s']
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


def plot_ground_track(
  result       : dict,
  epoch_dt_utc : Optional[datetime.datetime] = None,
  title_text   : str = "Ground Track",
) -> Figure:
  """
  Plot ground track (latitude vs longitude) on a 2D map projection.
  
  Input:
  ------
    result : dict
      Propagation result dictionary containing 'state' (6xN array) and 'plot_time_s'.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.
    title_text : str
      Base title for the plot.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the ground track plot.
  """
  fig = plt.figure(figsize=(14, 8))
  ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
  ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  # type: ignore
  ax.add_feature(cfeature.COASTLINE)  # type: ignore
  ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)  # type: ignore
  gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')  # type: ignore
  gl.top_labels = False
  gl.right_labels = False
  
  # Extract J2000 state vectors
  j2000_state   = result['state']
  j2000_pos_vec = j2000_state[0:3, :]
  time_s        = result['plot_time_s']
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
  lat = geo_coords['latitude']  * CONVERTER.DEG_PER_RAD
  lon = geo_coords['longitude'] * CONVERTER.DEG_PER_RAD
  
  # Handle longitude wrapping for plotting
  # Split trajectory at discontinuities (where lon jumps by more than 180 deg)
  lon_diff = np.abs(np.diff(lon))
  split_indices = np.where(lon_diff > 180)[0] + 1
  
  # Split into segments
  lat_segments = np.split(lat, split_indices)
  lon_segments = np.split(lon, split_indices)
  
  # Plot each segment
  plot_kwargs = {'transform': ccrs.PlateCarree()}

  for lat_seg, lon_seg in zip(lat_segments, lon_segments):
    ax.plot(lon_seg, lat_seg, 'b-', linewidth=1.5, **plot_kwargs)
  
  # Mark start and end points
  ax.scatter([lon[ 0]], [lat[ 0]], s=600, marker=r'$\blacksquare_{\text{o}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=5, label='Initial', **plot_kwargs)
  ax.scatter([lon[-1]], [lat[-1]], s=600, marker=r'$\blacksquare_{\text{f}}$', facecolors='white', edgecolors='b', linewidths=2, zorder=5, label='Final', **plot_kwargs)
  
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
  result : dict,
  epoch  : Optional[datetime.datetime] = None,
) -> Figure:
  """
  Plot 3D position trajectory with Moon trajectory in J2000 Earth-centered frame.
  
  Input:
  ------
    result : dict
      Propagation result dictionary containing 'state' (6xN array) and 'plot_time_s'.
    epoch : datetime, optional
      Reference epoch (start time) for labeling and Moon position computation.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the 3D plot.
  """
  fig = plt.figure(figsize=(18, 10))
  
  # Extract state vectors
  posvel_vec = result['state']
  time_s     = result['plot_time_s']
  
  # Build info string
  info_text = "Frame: J2000 - Sun-Centered"
  if epoch is not None and 'plot_time_s' in result:
    start_utc  = epoch.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time   = epoch + timedelta(seconds=result['plot_time_s'][-1])
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
  result_jpl_horizons_ephemeris    : Optional[dict],
  result_high_fidelity_propagation : dict,
  result_sgp4_propagation          : Optional[dict],
  time_o_dt                        : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool,
  compare_tle                      : bool,
  object_name                      : str = "object",
  object_name_display              : str = "Object",
) -> None:
  """
  Generate and save error comparison plots.
  """
  # If neither comparison is requested, do nothing
  if not (compare_jpl_horizons or compare_tle):
    return
  
  print("\n  Generate Error Plots")

  # Define availability flags
  has_horizons      = result_jpl_horizons_ephemeris is not None and result_jpl_horizons_ephemeris.get('success', False)
  has_high_fidelity = result_high_fidelity_propagation.get('success', False)
  has_sgp4          = result_sgp4_propagation is not None and result_sgp4_propagation.get('success', False)
  
  # Check for pre-computed ephemeris-time data
  has_hf_at_ephem   = has_high_fidelity and 'at_ephem_times' in result_high_fidelity_propagation
  has_sgp4_at_ephem = has_sgp4 and 'at_ephem_times' in result_sgp4_propagation
  
  # Lowercase name for filenames
  name_lower = object_name.lower()

  # High-Fidelity Relative To JPL Horizons (compare at ephemeris times)
  if compare_jpl_horizons and has_horizons and has_hf_at_ephem:
    print("    High-Fidelity Relative To JPL Horizons (at ephemeris times)")
    
    # Build result dict for comparison using pre-computed at_ephem_times data
    hf_at_ephem = {
      'plot_time_s' : result_high_fidelity_propagation['at_ephem_times']['plot_time_s'],
      'state'       : result_high_fidelity_propagation['at_ephem_times']['state'],
      'coe'         : result_high_fidelity_propagation['at_ephem_times']['coe'],
      'mee'         : result_high_fidelity_propagation['at_ephem_times']['mee'],
    }
    
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
    print(f"      Time-Series Error : <figures_folderpath>/{filename}")
    plt.close(fig_err_ts)

  # High-Fidelity Relative To SGP4 (compare at equal grid times)
  if compare_tle and has_high_fidelity and has_sgp4:
    print("    High-Fidelity Relative To SGP4 (at equal grid times)")
    
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
    print(f"      Time-Series Error : <figures_folderpath>/{filename}")
    plt.close(fig_err_ts)

  # SGP4 Relative To JPL Horizons (compare at ephemeris times)
  if compare_jpl_horizons and compare_tle and has_horizons and has_sgp4_at_ephem:
    print("    SGP4 Relative To JPL Horizons (at ephemeris times)")
    
    # Build result dict for comparison using pre-computed at_ephem_times data
    sgp4_at_ephem = {
      'plot_time_s' : result_sgp4_propagation['at_ephem_times']['plot_time_s'],
      'state'       : result_sgp4_propagation['at_ephem_times']['state'],
      'coe'         : result_sgp4_propagation['at_ephem_times']['coe'],
      'mee'         : result_sgp4_propagation['at_ephem_times']['mee'],
    }
    
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
    print(f"      Time-Series Error : <figures_folderpath>/{filename}")
    plt.close(fig_err_ts)
  


def generate_3d_and_time_series_plots(
  result_jpl_horizons_ephemeris    : Optional[dict],
  result_high_fidelity_propagation : dict,
  result_sgp4_propagation          : Optional[dict],
  time_o_dt                        : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool,
  compare_tle                      : bool,
  object_name                      : str = "object",
  object_name_display              : str = "Object",
) -> None:
  """
  Generate and save 3D trajectory and time series plots.
  
  Input:
  ------
    result_horizons : dict | None
      Horizons ephemeris result.
    result_high_fidelity : dict
      High-fidelity propagation result.
    result_sgp4 : dict | None
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
    None
  """
  print("  Generate 3D-Trajectory and Time-Series Plots")
  
  # Lowercase name for filenames
  name_lower = object_name.lower()

  # Horizons plots
  if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success'):
    print("    JPL-Horizons-Ephemeris Plots")

    fig1 = plot_3d_trajectories(result_jpl_horizons_ephemeris, epoch=time_o_dt, frame="J2000")
    fig1.suptitle(f'3D Inertial - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'3d_j2000_earth_centered_jpl_horizons_{name_lower}.png'
    fig1.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Inertial    : <figures_folderpath>/{filename}")
    plt.close(fig1)
    
    fig2 = plot_time_series(result_jpl_horizons_ephemeris, epoch=time_o_dt)
    fig2.suptitle(f'Time Series - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'timeseries_jpl_horizons_{name_lower}.png'
    fig2.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Time Series    : <figures_folderpath>/{filename}")
    plt.close(fig2)
    
    # Body-fixed 3D plot for Horizons
    fig_ef = plot_3d_trajectories_body_fixed(result_jpl_horizons_ephemeris, epoch_dt_utc=time_o_dt)
    fig_ef.suptitle(f'3D Body-Fixed - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'3d_iau_earth_jpl_horizons_{name_lower}.png'
    fig_ef.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Body-Fixed  : <figures_folderpath>/{filename}")
    plt.close(fig_ef)
    
    # Ground track plot for Horizons
    gt_title = f'Ground Track - {object_name_display} - JPL Horizons'
    fig_gt = plot_ground_track(result_jpl_horizons_ephemeris, epoch_dt_utc=time_o_dt, title_text=gt_title)
    filename = f'groundtrack_jpl_horizons_{name_lower}.png'
    fig_gt.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Ground Track   : <figures_folderpath>/{filename}")
    plt.close(fig_gt)
  
  # High-fidelity plots
  if result_high_fidelity_propagation.get('success'):
    print("    High-Fidelity-Model Plots")

    fig3 = plot_3d_trajectories(result_high_fidelity_propagation, epoch=time_o_dt, frame="J2000")
    fig3.suptitle(f'3D Inertial - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_j2000_earth_centered_high_fidelity_{name_lower}.png'
    fig3.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Inertial    : <figures_folderpath>/{filename}")
    plt.close(fig3)
    
    fig4 = plot_time_series(result_high_fidelity_propagation, epoch=time_o_dt)
    fig4.suptitle(f'Time Series - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'timeseries_high_fidelity_{name_lower}.png'
    fig4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Time Series    : <figures_folderpath>/{filename}")
    plt.close(fig4)
    
    # Body-fixed 3D plot
    fig_ef = plot_3d_trajectories_body_fixed(result_high_fidelity_propagation, epoch_dt_utc=time_o_dt)
    fig_ef.suptitle(f'3D Body-Fixed - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_iau_earth_high_fidelity_{name_lower}.png'
    fig_ef.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Body-Fixed  : <figures_folderpath>/{filename}")
    plt.close(fig_ef)
    
    # Ground track plot
    gt_title = f'Ground Track - {object_name_display} - High-Fidelity'
    fig_gt = plot_ground_track(result_high_fidelity_propagation, epoch_dt_utc=time_o_dt, title_text=gt_title)
    filename = f'groundtrack_high_fidelity_{name_lower}.png'
    fig_gt.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Ground Track   : <figures_folderpath>/{filename}")
    plt.close(fig_gt)
    
    # 3D plot Sun-centered trajectory
    fig_moon = plot_3d_trajectory_sun_centered(result_high_fidelity_propagation, epoch=time_o_dt)
    fig_moon.suptitle(f'3D Inertial - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_j2000_sun_centered_high_fidelity_{name_lower}.png'
    fig_moon.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Inertial    : <figures_folderpath>/{filename}")
    plt.close(fig_moon)
  
  # SGP4 plots
  if compare_tle and result_sgp4_propagation and result_sgp4_propagation.get('success'):
    print("    SGP4-Model Plots")
    
    # 3D trajectory plot
    fig_sgp4_3d = plot_3d_trajectories(result_sgp4_propagation, epoch=time_o_dt, frame="J2000")
    fig_sgp4_3d.suptitle(f'3D Inertial - {object_name_display} - SGP4', fontsize=16)
    filename = f'3d_j2000_earth_centered_sgp4_{name_lower}.png'
    fig_sgp4_3d.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Inertial    : <figures_folderpath>/{filename}")
    plt.close(fig_sgp4_3d)
    
    # Time series plot
    fig_sgp4_ts = plot_time_series(result_sgp4_propagation, epoch=time_o_dt)
    fig_sgp4_ts.suptitle(f'Time Series - {object_name_display} - SGP4', fontsize=16)
    filename = f'timeseries_sgp4_{name_lower}.png'
    fig_sgp4_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Time Series    : <figures_folderpath>/{filename}")
    plt.close(fig_sgp4_ts)

    # Ground track plot for SGP4
    gt_title = f'Ground Track - {object_name_display} - SGP4'
    fig_gt_sgp4 = plot_ground_track(result_sgp4_propagation, epoch_dt_utc=time_o_dt, title_text=gt_title)
    filename = f'groundtrack_sgp4_{name_lower}.png'
    fig_gt_sgp4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Ground Track   : <figures_folderpath>/{filename}")
    plt.close(fig_gt_sgp4)

def generate_plots(
  result_jpl_horizons_ephemeris    : Optional[dict],
  result_high_fidelity_propagation : dict,
  result_sgp4_propagation          : Optional[dict],
  time_o_dt                        : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool = False,
  compare_tle                      : bool = False,
  object_name                      : str  = "object",
  object_name_display              : str  = "Object",
) -> None:
  """
  Generate and save all simulation plots.
  
  Input:
  ------
    result_horizons : dict | None
      Horizons ephemeris result.
    result_high_fidelity : dict
      High-fidelity propagation result.
    result_sgp4 : dict | None
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
    None
  """
  print("\nGenerate and Save Plots")
  print(f"  Figure Folderpath : {figures_folderpath}\n")
  
  # Generate 3D and time series plots
  generate_3d_and_time_series_plots(
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
    
  # Generate error plots only if a comparison was requested
  if compare_jpl_horizons or compare_tle:
    generate_error_plots(
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