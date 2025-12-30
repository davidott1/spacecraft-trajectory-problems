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
  
  # Add Earth wireframe ellipsoid
  u       = np.linspace(0, 2 * np.pi, 24)
  v       = np.linspace(0, np.pi, 12)
  r_eq    = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
  r_pol   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.POLAR
  x_earth = r_eq * np.outer(np.cos(u), np.sin(v))
  y_earth = r_eq * np.outer(np.sin(u), np.sin(v))
  z_earth = r_pol * np.outer(np.ones(np.size(u)), np.cos(v))
  ax1.plot_wireframe(x_earth, y_earth, z_earth, color='black', linewidth=0.5, alpha=1.0) # type: ignore
  
  ax1.plot(pos_x, pos_y, pos_z, 'b-', linewidth=2.0)
  ax1.scatter([pos_x[0]], [pos_y[0]], [pos_z[0]], s=100, marker='>', facecolors='white', edgecolors='b', linewidths=2) # type: ignore
  ax1.scatter([pos_x[-1]], [pos_y[-1]], [pos_z[-1]], s=100, marker='s', facecolors='white', edgecolors='b', linewidths=2) # type: ignore
  ax1.set_xlabel('Pos-X [m]')
  ax1.set_ylabel('Pos-Y [m]')
  ax1.set_zlabel('Pos-Z [m]') # type: ignore
  ax1.grid(True)
  ax1.set_box_aspect([1,1,1]) # type: ignore

  # Set pane colors to white
  ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

  min_limit, max_limit = get_equal_limits(ax1, buffer_fraction=0.25)
  
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
  ax1.plot_surface(U_disk, V_disk, np.full_like(U_disk, min_limit), color='black', alpha=earth_shadow_alpha, shade=False)

  # XZ plane (y = max_limit)
  ax1.plot_surface(U_disk, np.full_like(U_disk, max_limit), V_disk, color='black', alpha=earth_shadow_alpha, shade=False)

  # YZ plane (x = min_limit)
  ax1.plot_surface(np.full_like(U_disk, min_limit), U_disk, V_disk, color='black', alpha=earth_shadow_alpha, shade=False)

  # Add sun direction arrows (only if we have epoch and are in J2000 frame)
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
      
      # Scale arrows to be visible in the plot (30% of axis range)
      arrow_length = 0.3 * (max_limit - min_limit)
      origin = np.array([0, 0, 0])  # Earth center
      
      # Line start points (half length from origin) and end points (full length)
      line_start_initial = origin + 0.5 * arrow_length * sun_dir_start
      line_end_initial = origin + arrow_length * sun_dir_start
      
      line_start_final = origin + 0.5 * arrow_length * sun_dir_end
      line_end_final = origin + arrow_length * sun_dir_end
      
      # Draw 3D line for initial sun direction (gold color)
      ax1.plot([line_start_initial[0], line_end_initial[0]], 
               [line_start_initial[1], line_end_initial[1]], 
               [line_start_initial[2], line_end_initial[2]],
               color='gold', linewidth=2.5, alpha=0.9)
      
      # Draw line for final sun direction (orange color)
      ax1.plot([line_start_final[0], line_end_final[0]], 
               [line_start_final[1], line_end_final[1]], 
               [line_start_final[2], line_end_final[2]],
               color='orange', linewidth=2.5, alpha=0.9)
      
      # Add triangle marker at initial sun direction tip
      ax1.scatter([line_end_initial[0]], [line_end_initial[1]], [line_end_initial[2]], 
                  s=150, marker='>', color='gold', edgecolors='gold', linewidths=1.5, 
                  zorder=10, label='Sun (Initial)')
      
      # Add square marker at final sun direction tip
      ax1.scatter([line_end_final[0]], [line_end_final[1]], [line_end_final[2]], 
                  s=150, marker='s', color='orange', edgecolors='orange', linewidths=1.5, 
                  zorder=10, label='Sun (Final)')
      
      # Project sun vectors onto planes (like trajectory shadows)
      sun_shadow_alpha = 0.6
      sun_shadow_lw = 1.5
      sun_shadow_color_initial = 'darkgray'   # Lighter gray for initial
      sun_shadow_color_final   = 'dimgray'    # Darker gray for final
      
      # XY plane shadows (z = min_limit)
      ax1.plot([line_start_initial[0], line_end_initial[0]], 
               [line_start_initial[1], line_end_initial[1]], 
               [min_limit, min_limit],
               color=sun_shadow_color_initial, linewidth=sun_shadow_lw, alpha=sun_shadow_alpha)
      ax1.plot([line_start_final[0], line_end_final[0]], 
               [line_start_final[1], line_end_final[1]], 
               [min_limit, min_limit],
               color=sun_shadow_color_final, linewidth=sun_shadow_lw, alpha=sun_shadow_alpha)
      
      # XZ plane shadows (y = max_limit)
      ax1.plot([line_start_initial[0], line_end_initial[0]], 
               [max_limit, max_limit], 
               [line_start_initial[2], line_end_initial[2]],
               color=sun_shadow_color_initial, linewidth=sun_shadow_lw, alpha=sun_shadow_alpha)
      ax1.plot([line_start_final[0], line_end_final[0]], 
               [max_limit, max_limit], 
               [line_start_final[2], line_end_final[2]],
               color=sun_shadow_color_final, linewidth=sun_shadow_lw, alpha=sun_shadow_alpha)
      
      # YZ plane shadows (x = min_limit)
      ax1.plot([min_limit, min_limit], 
               [line_start_initial[1], line_end_initial[1]], 
               [line_start_initial[2], line_end_initial[2]],
               color=sun_shadow_color_initial, linewidth=sun_shadow_lw, alpha=sun_shadow_alpha)
      ax1.plot([min_limit, min_limit], 
               [line_start_final[1], line_end_final[1]], 
               [line_start_final[2], line_end_final[2]],
               color=sun_shadow_color_final, linewidth=sun_shadow_lw, alpha=sun_shadow_alpha)
      
    except Exception as e:
      # If SPICE kernels aren't loaded or other error, silently skip sun arrow
      pass

  # Plot 3D velocity trajectory
  ax2 = fig.add_subplot(122, projection='3d')
  ax2.plot(vel_x, vel_y, vel_z, 'r-', linewidth=2.0)
  ax2.scatter([vel_x[0]], [vel_y[0]], [vel_z[0]], s=100, marker='>', facecolors='white', edgecolors='r', linewidths=2) # type: ignore
  ax2.scatter([vel_x[-1]], [vel_y[-1]], [vel_z[-1]], s=100, marker='s', facecolors='white', edgecolors='r', linewidths=2) # type: ignore
  ax2.set_xlabel('Vel-X [m/s]')
  ax2.set_ylabel('Vel-Y [m/s]')
  ax2.set_zlabel('Vel-Z [m/s]') # type: ignore
  ax2.grid(True)
  ax2.set_box_aspect([1,1,1]) # type: ignore

  # Set pane colors to white
  ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

  min_limit_vel, max_limit_vel = get_equal_limits(ax2, buffer_fraction=0.25)
  
  ax2.set_xlim([min_limit_vel, max_limit_vel]) # type: ignore
  ax2.set_ylim([min_limit_vel, max_limit_vel]) # type: ignore
  ax2.set_zlim([min_limit_vel, max_limit_vel]) # type: ignore

  # Add velocity trajectory shadows (projections onto planes)
  # XY plane shadow (z = min_limit_vel)
  ax2.plot(vel_x, vel_y, np.full_like(vel_z, min_limit_vel), color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  # XZ plane shadow (y = max_limit_vel)
  ax2.plot(vel_x, np.full_like(vel_y, max_limit_vel), vel_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  # YZ plane shadow (x = min_limit_vel)
  ax2.plot(np.full_like(vel_x, min_limit_vel), vel_y, vel_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)

  # Create custom legend handles with black edges
  legend_handles = [
    Line2D([0], [0], marker='>', color='w', markerfacecolor='white', markeredgecolor='black', 
           markersize=10, markeredgewidth=2, linestyle='None', label='Initial'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', 
           markersize=10, markeredgewidth=2, linestyle='None', label='Final'),
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
  Plot position and velocity components vs time in a 2x1 grid.
  
  Input:
  ------
    result : dict
      Propagation result dictionary containing 'plot_time_s', 'state', and 'coe'.
    epoch : datetime, optional
      Reference epoch for UTC time axis.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the time series plots.
  """
  
  # Create figure
  fig = plt.figure(figsize=(18,10))
  
  # Extract data
  time = result['plot_time_s']
  states = result['state']
  pos_x, pos_y, pos_z = states[0, :], states[1, :], states[2, :]
  vel_x, vel_y, vel_z = states[3, :], states[4, :], states[5, :]
  coe = result['coe']
  
  # Calculate magnitudes
  pos_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
  vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
  
  # Plot position vs time (spans rows 0-2, column 0)
  ax_pos = plt.subplot2grid((6, 2), (0, 0), rowspan=3)
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
  ax_vel = plt.subplot2grid((6, 2), (3, 0), rowspan=3, sharex=ax_pos)
  ax_vel.plot(time, vel_x, 'r-', label='X', linewidth=1.5)
  ax_vel.plot(time, vel_y, 'g-', label='Y', linewidth=1.5)
  ax_vel.plot(time, vel_z, 'b-', label='Z', linewidth=1.5)
  ax_vel.plot(time, vel_mag, 'k-', label='Magnitude', linewidth=2)
  ax_vel.set_xlabel('Time\n[s]')
  ax_vel.set_ylabel('Velocity\n[m/s]')
  ax_vel.legend()
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot sma vs time (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 2), (0, 1), sharex=ax_pos)
  ax_sma.plot(time, coe['sma'], 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('Semi-Major Axis\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot ecc vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 2), (1, 1), sharex=ax_pos)
  ax_ecc.plot(time, coe['ecc'], 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('Eccentricity\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inc vs time (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 2), (2, 1), sharex=ax_pos)
  ax_inc.plot(time, coe['inc'] * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('Inclination\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot raan vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 2), (3, 1), sharex=ax_pos)
  ax_raan.plot(time, coe['raan'] * CONVERTER.DEG_PER_RAD, 'b-', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN\n[deg]')
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot aop vs time (row 4, column 1)
  ax_aop = plt.subplot2grid((6, 2), (4, 1), sharex=ax_pos)
  aop_unwrapped = np.unwrap(coe['aop']) * CONVERTER.DEG_PER_RAD
  ax_aop.plot(time, aop_unwrapped, 'b-', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('Argument of Periapsis\n[deg]')
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot ta, ea, ma vs time (row 5, column 1)
  ax_anom = plt.subplot2grid((6, 2), (5, 1), sharex=ax_pos)
  ax_anom.plot(time, coe['ta'] * CONVERTER.DEG_PER_RAD, 'r-', label='TA', linewidth=1.5)
  ax_anom.plot(time, coe['ea'] * CONVERTER.DEG_PER_RAD, 'g-', label='EA', linewidth=1.5)
  ax_anom.plot(time, coe['ma'] * CONVERTER.DEG_PER_RAD, 'b-', label='MA', linewidth=1.5)
  ax_anom.set_xlabel('Time\n[s]')
  ax_anom.set_ylabel('Anomaly\n[deg]')
  ax_anom.legend()
  ax_anom.grid(True)
  ax_anom.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Add UTC time axis if epoch is provided
  if epoch is not None:
    # Only add UTC time to top row axes
    top_row_axes = [ax_pos, ax_sma]
    max_time = time[-1]
    
    for ax in top_row_axes:
      add_utc_time_axis(ax, epoch, max_time)

  # Align y-axis labels for right column
  fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_aop, ax_anom])
  
  # Align y-axis labels for left column
  fig.align_ylabels([ax_pos, ax_vel])

  plt.subplots_adjust(hspace=0.17, wspace=0.2)
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
  ax1.scatter([pos_error[0, 0]], [pos_error[1, 0]], [pos_error[2, 0]], 
              s=100, marker='>', facecolors='white', edgecolors='b', linewidths=2, label='Initial') # type: ignore
  ax1.scatter([pos_error[0, -1]], [pos_error[1, -1]], [pos_error[2, -1]], 
              s=100, marker='s', facecolors='white', edgecolors='b', linewidths=2, label='Final') # type: ignore
  ax1.set_xlabel('Error X [m]')
  ax1.set_ylabel('Error Y [m]')
  ax1.set_zlabel('Error Z [m]') # type: ignore
  ax1.set_title('Position Error')
  ax1.grid(True)
  leg1 = ax1.legend()
  leg1.get_frame().set_edgecolor('black')
  
  # Plot 3D velocity error
  ax2 = fig.add_subplot(122, projection='3d')
  ax2.plot(vel_error[0, :], vel_error[1, :], vel_error[2, :], 'r-', linewidth=1)
  ax2.scatter([vel_error[0, 0]], [vel_error[1, 0]], [vel_error[2, 0]], 
              s=100, marker='>', facecolors='white', edgecolors='r', linewidths=2, label='Initial') # type: ignore
  ax2.scatter([vel_error[0, -1]], [vel_error[1, -1]], [vel_error[2, -1]], 
              s=100, marker='s', facecolors='white', edgecolors='r', linewidths=2, label='Final') # type: ignore
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
      Reference result dictionary with 'plot_time_s' and 'state'/'coe'.
    result_comp : dict
      Comparison result dictionary with 'plot_time_s' and 'state'/'coe'.
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
  
  # Verify time grids match (use allclose for floating-point comparison)
  if len(time_ref) != len(time_comp) or not np.allclose(time_ref, time_comp, rtol=1e-9, atol=1e-9):
    raise ValueError(
      f"Time grids don't match! "
      f"Reference has {len(time_ref)} points from {time_ref[0]:.1f} to {time_ref[-1]:.1f} s, "
      f"Comparison has {len(time_comp)} points from {time_comp[0]:.1f} to {time_comp[-1]:.1f} s. "
      f"Both datasets must use the same time grid for error comparison."
    )
  
  # Create figure with subplots matching the grid structure
  fig = plt.figure(figsize=(18, 10))
  
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
  
  # LEFT SIDE: Position and Velocity Errors
  # Plot position error vs time (spans rows 0-2, column 0)
  ax_pos = plt.subplot2grid((6, 2), (0, 0), rowspan=3)
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
  ax_vel = plt.subplot2grid((6, 2), (3, 0), rowspan=3, sharex=ax_pos)
  ax_vel.plot(time, vel_error[0, :], 'r-', label=vel_labels[0], linewidth=1.5)
  ax_vel.plot(time, vel_error[1, :], 'g-', label=vel_labels[1], linewidth=1.5)
  ax_vel.plot(time, vel_error[2, :], 'b-', label=vel_labels[2], linewidth=1.5)
  ax_vel.plot(time, vel_error_mag, 'k-', label='Magnitude', linewidth=2)
  ax_vel.set_xlabel('Time\n[s]')
  ax_vel.set_ylabel(vel_ylabel)
  ax_vel.legend()
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot sma error vs time (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 2), (0, 1), sharex=ax_pos)
  if 'sma' in coe_ref and 'sma' in coe_comp:
    sma_error = coe_comp['sma'] - coe_ref['sma']
    ax_sma.plot(time, sma_error, 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA Error\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot eccentricity error vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 2), (1, 1), sharex=ax_pos)
  if 'ecc' in coe_ref and 'ecc' in coe_comp:
    ecc_error = coe_comp['ecc'] - coe_ref['ecc']
    ax_ecc.plot(time, ecc_error, 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('Eccentricity Error\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inclination error (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 2), (2, 1), sharex=ax_pos)
  if 'inc' in coe_ref and 'inc' in coe_comp:
    inc_error = (coe_comp['inc'] - coe_ref['inc']) * CONVERTER.DEG_PER_RAD
    ax_inc.plot(time, inc_error, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('Inclination Error\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot RAAN error vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 2), (3, 1), sharex=ax_pos)
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
  ax_aop = plt.subplot2grid((6, 2), (4, 1), sharex=ax_pos)
  if 'aop' in coe_ref and 'aop' in coe_comp:
    # Handle angle wrapping for AOP
    aop_error_rad = np.arctan2(np.sin(coe_comp['aop'] - coe_ref['aop']), 
                   np.cos(coe_comp['aop'] - coe_ref['aop']))
    aop_error = aop_error_rad * CONVERTER.DEG_PER_RAD
    ax_aop.plot(time, aop_error, 'b-', linewidth=1.5)
  ax_aop.tick_params(labelbottom=False)
  ax_aop.set_ylabel('Argument of\nPeriapsis Error\n[deg]')
  ax_aop.grid(True)
  ax_aop.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot true anomaly error vs time (row 5, column 1)
  ax_ta = plt.subplot2grid((6, 2), (5, 1), sharex=ax_pos)
  if 'ta' in coe_ref and 'ta' in coe_comp:
    # Handle angle wrapping for TA
    ta_error_rad = np.arctan2(np.sin(coe_comp['ta'] - coe_ref['ta']), 
                   np.cos(coe_comp['ta'] - coe_ref['ta']))
    ta_error = ta_error_rad * CONVERTER.DEG_PER_RAD
    ax_ta.plot(time, ta_error, 'b-', linewidth=1.5)
  ax_ta.set_xlabel('Time\n[s]')
  ax_ta.set_ylabel('True Anomaly\nError\n[deg]')
  ax_ta.grid(True)
  ax_ta.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Add UTC time axis if epoch is not None
  if epoch is not None:
    add_utc_time_axis(ax_pos, epoch, time_ref[-1])

  # Align y-axis labels
  fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_aop, ax_ta])
  fig.align_ylabels([ax_pos, ax_vel])
  
  fig.suptitle(title, fontsize=16)
  plt.subplots_adjust(hspace=0.17, wspace=0.2)
  return fig


def plot_3d_trajectories_earth_fixed(
  result       : dict,
  epoch_dt_utc : Optional[datetime.datetime] = None,
) -> Figure:
  """
  Plot 3D position and velocity trajectories in Earth-fixed (IAU_EARTH) frame.
  
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
  
  # Transform each position and velocity to Earth-fixed frame
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
  info_text = "Frame: IAU_EARTH (Earth-Fixed)"
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
  ax1.plot_wireframe(x_earth, y_earth, z_earth, color='black', linewidth=0.5, alpha=1.0)
  
  ax1.plot(pos_x, pos_y, pos_z, 'b-', linewidth=2.0)
  ax1.scatter([pos_x[0]], [pos_y[0]], [pos_z[0]], s=100, marker='>', facecolors='white', edgecolors='b', linewidths=2)
  ax1.scatter([pos_x[-1]], [pos_y[-1]], [pos_z[-1]], s=100, marker='s', facecolors='white', edgecolors='b', linewidths=2)
  ax1.set_xlabel('Pos-X [m]')
  ax1.set_ylabel('Pos-Y [m]')
  ax1.set_zlabel('Pos-Z [m]') # type: ignore
  ax1.grid(True)
  ax1.set_box_aspect([1,1,1]) # type: ignore

  # Set pane colors to white
  ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

  min_limit, max_limit = get_equal_limits(ax1, buffer_fraction=0.25)
  
  ax1.set_xlim([min_limit, max_limit])
  ax1.set_ylim([min_limit, max_limit])
  ax1.set_zlim([min_limit, max_limit])

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

  ax1.plot_surface(U_disk, V_disk, np.full_like(U_disk, min_limit), color='black', alpha=earth_shadow_alpha, shade=False)
  ax1.plot_surface(U_disk, np.full_like(U_disk, max_limit), V_disk, color='black', alpha=earth_shadow_alpha, shade=False)
  ax1.plot_surface(np.full_like(U_disk, min_limit), U_disk, V_disk, color='black', alpha=earth_shadow_alpha, shade=False)

  # Plot 3D velocity trajectory
  ax2 = fig.add_subplot(122, projection='3d')
  ax2.plot(vel_x, vel_y, vel_z, 'r-', linewidth=2.0)
  ax2.scatter([vel_x[0]], [vel_y[0]], [vel_z[0]], s=100, marker='>', facecolors='white', edgecolors='r', linewidths=2) # type: ignore
  ax2.scatter([vel_x[-1]], [vel_y[-1]], [vel_z[-1]], s=100, marker='s', facecolors='white', edgecolors='r', linewidths=2) # type: ignore
  ax2.set_xlabel('Vel-X [m/s]')
  ax2.set_ylabel('Vel-Y [m/s]')
  ax2.set_zlabel('Vel-Z [m/s]') # type: ignore
  ax2.grid(True)
  ax2.set_box_aspect([1,1,1]) # type: ignore

  # Set pane colors to white
  ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # type: ignore

  min_limit_vel, max_limit_vel = get_equal_limits(ax2, buffer_fraction=0.25)
  
  ax2.set_xlim((min_limit_vel, max_limit_vel))
  ax2.set_ylim((min_limit_vel, max_limit_vel))
  ax2.set_zlim((min_limit_vel, max_limit_vel)) # type: ignore

  # Add velocity trajectory shadows
  ax2.plot(vel_x, vel_y, np.full_like(vel_z, min_limit_vel), color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  ax2.plot(vel_x, np.full_like(vel_y, max_limit_vel), vel_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)
  ax2.plot(np.full_like(vel_x, min_limit_vel), vel_y, vel_z, color=shadow_color, alpha=shadow_alpha, linewidth=shadow_lw)

  # Legend
  legend_handles = [
    Line2D([0], [0], marker='>', color='w', markerfacecolor='white', markeredgecolor='black', 
           markersize=10, markeredgewidth=2, linestyle='None', label='Initial'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', 
           markersize=10, markeredgewidth=2, linestyle='None', label='Final'),
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
  ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
  ax.add_feature(cfeature.COASTLINE)
  ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
  gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
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
  
  # Transform each position to Earth-fixed frame
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
  ax.scatter([lon[0]], [lat[0]], s=100, marker='>', facecolors='white', edgecolors='b', linewidths=2, zorder=5, label='Initial', **plot_kwargs)
  ax.scatter([lon[-1]], [lat[-1]], s=100, marker='s', facecolors='white', edgecolors='b', linewidths=2, zorder=5, label='Final', **plot_kwargs)
  
  # Legend
  leg = ax.legend(loc='upper right')
  leg.get_frame().set_edgecolor('black')
  
  # Build info text for bottom of figure (consistent with 3D plots)
  info_text = "Frame: IAU_EARTH (Earth-Fixed)"
  if epoch_dt_utc is not None:
    start_utc = epoch_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time  = epoch_dt_utc + timedelta(seconds=time_s[-1])
    end_utc   = end_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    info_text += f"  |  Initial: {start_utc}  |  Final: {end_utc}"
  
  # Add info text as figure text at bottom (consistent with 3D plots)
  fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=11, color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))
  
  # Set title using suptitle (will be overwritten by caller, but provides default)
  fig.suptitle(title_text, fontsize=16)
  
  plt.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))  # Leave space at bottom for info text and top for title
  return fig


def plot_3d_trajectory_with_moon(
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
  fig = plt.figure(figsize=(28, 10))
  
  # Extract state vectors
  posvel_vec = result['state']
  pos_x, pos_y, pos_z = posvel_vec[0, :], posvel_vec[1, :], posvel_vec[2, :]
  time_s = result['plot_time_s']
  
  # Build info string
  info_text = "Frame: J2000 (Earth-Centered)"
  if epoch is not None and 'plot_time_s' in result:
    start_utc = epoch.strftime('%Y-%m-%d %H:%M:%S UTC')
    end_time  = epoch + timedelta(seconds=result['plot_time_s'][-1])
    end_utc   = end_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    info_text += f"  |  Initial: {start_utc}  |  Final: {end_utc}"
  
  # Create subplots
  ax_sun = fig.add_subplot(131, projection='3d')
  ax1 = fig.add_subplot(132, projection='3d')
  ax2 = fig.add_subplot(133, projection='3d')
  axes = [ax1, ax2]
  
  # Pre-calculate Earth wireframe ellipsoid
  u       = np.linspace(0, 2 * np.pi, 24)
  v       = np.linspace(0, np.pi, 12)
  r_eq    = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
  r_pol   = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.POLAR
  x_earth = r_eq * np.outer(np.cos(u), np.sin(v))
  y_earth = r_eq * np.outer(np.sin(u), np.sin(v))
  z_earth = r_pol * np.outer(np.ones(np.size(u)), np.cos(v))
  
  # Pre-calculate Moon and Sun data if epoch is available
  moon_pos = None
  n_moon_points = 0
  moon_orbit_full = None
  x_moon_sphere, y_moon_sphere, z_moon_sphere = None, None, None
  
  sun_pos = None
  n_sun_points = 0
  sun_orbit_full = None
  
  # Heliocentric data
  earth_orbit_helio = None
  earth_pos_helio_init = None
  earth_pos_helio_final = None
  sc_pos_helio = None
  moon_pos_helio = None
  
  if epoch is not None:
    try:
      epoch_et_start = utc_to_et(epoch)
      
      # --- HELIOCENTRIC EARTH ORBIT ---
      # Get Earth state relative to Sun at start
      earth_state_sun, _ = spice.spkezr('EARTH', epoch_et_start, 'J2000', 'NONE', 'SUN')
      earth_pos_helio_init = earth_state_sun[0:3] * 1000.0 # m
      earth_vel_helio_init = earth_state_sun[3:6] * 1000.0 # m/s
      
      # Get Earth state relative to Sun at end
      epoch_et_end = epoch_et_start + time_s[-1]
      earth_state_sun_end, _ = spice.spkezr('EARTH', epoch_et_end, 'J2000', 'NONE', 'SUN')
      earth_pos_helio_final = earth_state_sun_end[0:3] * 1000.0 # m

      # Calculate full orbit based on initial osculating elements
      try:
          mu_sun = SOLARSYSTEMCONSTANTS.SUN.GP
      except AttributeError:
          mu_sun = 1.32712440018e20

      earth_coe = OrbitConverter.pv_to_coe(earth_pos_helio_init, earth_vel_helio_init, mu_sun)
      
      n_orbit_points = 200
      ta_vals = np.linspace(0, 2*np.pi, n_orbit_points)
      earth_orbit_helio = np.zeros((3, n_orbit_points))
      
      for i, ta in enumerate(ta_vals):
        earth_coe['ta'] = ta
        r_vec, _ = OrbitConverter.coe_to_pv(earth_coe, mu_sun)
        earth_orbit_helio[:, i] = r_vec
      
      # --- SPACECRAFT & MOON HELIOCENTRIC TRAJECTORY ---
      # Calculate spacecraft position relative to Sun
      # Downsample for performance
      stride = max(1, len(time_s) // 500)
      indices = range(0, len(time_s), stride)
      sc_pos_helio = np.zeros((3, len(indices)))
      moon_pos_helio = np.zeros((3, len(indices)))
      
      for i, idx in enumerate(indices):
          t = time_s[idx]
          et = epoch_et_start + t
          # Earth relative to Sun
          pos_earth_sun_km, _ = spice.spkpos('EARTH', et, 'J2000', 'NONE', 'SUN')
          pos_earth_sun_m = np.array(pos_earth_sun_km) * 1000.0
          
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
      # 1. Moon trajectory during simulation
      n_moon_points = min(len(time_s), 500)  # Limit Moon points for performance
      moon_time_indices = np.linspace(0, len(time_s) - 1, n_moon_points, dtype=int)
      
      moon_pos = np.zeros((3, n_moon_points))
      for i, idx in enumerate(moon_time_indices):
        epoch_et_i = epoch_et_start + time_s[idx]
        # Get Moon position relative to Earth in J2000 (returns in km)
        moon_pos_km, _ = spice.spkpos('MOON', epoch_et_i, 'J2000', 'NONE', 'EARTH')
        moon_pos[:, i] = np.array(moon_pos_km) * 1000.0  # Convert to meters
      
      # Moon sphere at final position
      moon_radius = 1.7374e6  # Moon radius in meters
      u_moon = np.linspace(0, 2 * np.pi, 12)
      v_moon = np.linspace(0, np.pi, 8)
      x_moon_sphere = moon_radius * np.outer(np.cos(u_moon), np.sin(v_moon)) + moon_pos[0, -1]
      y_moon_sphere = moon_radius * np.outer(np.sin(u_moon), np.sin(v_moon)) + moon_pos[1, -1]
      z_moon_sphere = moon_radius * np.outer(np.ones(np.size(u_moon)), np.cos(v_moon)) + moon_pos[2, -1]

      # 2. Full approximate Moon orbit (Keplerian ellipse based on initial state)
      # Get initial state of Moon (km, km/s)
      moon_state_km, _ = spice.spkezr('MOON', epoch_et_start, 'J2000', 'NONE', 'EARTH')
      moon_pos_init = moon_state_km[0:3] * 1000.0 # m
      moon_vel_init = moon_state_km[3:6] * 1000.0 # m/s
      
      # Convert to orbital elements
      moon_coe = OrbitConverter.pv_to_coe(moon_pos_init, moon_vel_init, SOLARSYSTEMCONSTANTS.EARTH.GP)
      
      # Generate full orbit points
      n_orbit_points = 200
      ta_vals = np.linspace(0, 2*np.pi, n_orbit_points)
      moon_orbit_full = np.zeros((3, n_orbit_points))
      
      for i, ta in enumerate(ta_vals):
        # Update TA in COE
        moon_coe['ta'] = ta
        # Convert back to PV
        r_vec, _ = OrbitConverter.coe_to_pv(moon_coe, SOLARSYSTEMCONSTANTS.EARTH.GP)
        moon_orbit_full[:, i] = r_vec

      # --- SUN ---
      # 1. Sun trajectory during simulation
      n_sun_points = min(len(time_s), 500)
      sun_time_indices = np.linspace(0, len(time_s) - 1, n_sun_points, dtype=int)
      
      sun_pos = np.zeros((3, n_sun_points))
      for i, idx in enumerate(sun_time_indices):
        epoch_et_i = epoch_et_start + time_s[idx]
        sun_pos_km, _ = spice.spkpos('SUN', epoch_et_i, 'J2000', 'NONE', 'EARTH')
        sun_pos[:, i] = np.array(sun_pos_km) * 1000.0
        
      # 2. Full approximate Sun orbit
      # Use Sun's GP for relative motion (mu_sun + mu_earth approx mu_sun)
      sun_state_km, _ = spice.spkezr('SUN', epoch_et_start, 'J2000', 'NONE', 'EARTH')
      sun_pos_init = sun_state_km[0:3] * 1000.0
      sun_vel_init = sun_state_km[3:6] * 1000.0
      
      sun_coe = OrbitConverter.pv_to_coe(sun_pos_init, sun_vel_init, mu_sun)
      
      sun_orbit_full = np.zeros((3, n_orbit_points))
      for i, ta in enumerate(ta_vals):
        sun_coe['ta'] = ta
        r_vec, _ = OrbitConverter.coe_to_pv(sun_coe, mu_sun)
        sun_orbit_full[:, i] = r_vec
      
    except Exception:
      # If SPICE kernels aren't loaded or other error, skip Moon/Sun trajectory
      pass

  # Plot Heliocentric view (ax_sun)
  ax_sun.set_title("Heliocentric J2000 (Sun-Centered)")
  
  # Plot Sun
  ax_sun.scatter([0], [0], [0], s=300, color='orange', edgecolors='orange', label='Sun')
  
  if earth_orbit_helio is not None:
      # Plot Orbit
      ax_sun.plot(earth_orbit_helio[0, :], earth_orbit_helio[1, :], earth_orbit_helio[2, :],
                  color='black', linestyle='--', linewidth=1, alpha=0.6, label='Earth Orbit')
      
      # Plot Earth Initial
      ax_sun.scatter([earth_pos_helio_init[0]], [earth_pos_helio_init[1]], [earth_pos_helio_init[2]],
                     color='black', s=100, marker='>', label='Earth (Initial)')
                     
      # Plot Earth Final
      if earth_pos_helio_final is not None:
          ax_sun.scatter([earth_pos_helio_final[0]], [earth_pos_helio_final[1]], [earth_pos_helio_final[2]],
                         color='black', s=100, marker='s', label='Earth (Final)')
  
  if moon_pos_helio is not None:
      # Plot Moon Trajectory
      ax_sun.plot(moon_pos_helio[0, :], moon_pos_helio[1, :], moon_pos_helio[2, :],
                  color='gray', linewidth=1.0, alpha=0.8, label='Moon')

  if sc_pos_helio is not None:
      # Plot Spacecraft Trajectory
      ax_sun.plot(sc_pos_helio[0, :], sc_pos_helio[1, :], sc_pos_helio[2, :],
                  color='b', linewidth=1.5, label='Spacecraft')
      
      # Plot Spacecraft Initial
      ax_sun.scatter([sc_pos_helio[0, 0]], [sc_pos_helio[1, 0]], [sc_pos_helio[2, 0]],
                     s=80, marker='>', facecolors='white', edgecolors='b', zorder=10)
      
      # Plot Spacecraft Final
      ax_sun.scatter([sc_pos_helio[0, -1]], [sc_pos_helio[1, -1]], [sc_pos_helio[2, -1]],
                     s=80, marker='s', facecolors='white', edgecolors='b', zorder=10)
  
  ax_sun.set_xlabel('X [m]')
  ax_sun.set_ylabel('Y [m]')
  ax_sun.set_zlabel('Z [m]') # type: ignore
  ax_sun.grid(True)
  
  # Custom Legend for ax_sun
  legend_handles_sun = [
      Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markeredgecolor='orange', markersize=10, linestyle='None', label='Sun'),
      Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Earth Orbit'),
      Line2D([0], [0], color='gray', linewidth=1, label='Moon'),
      Line2D([0], [0], color='b', linewidth=1.5, label='Spacecraft'),
      Line2D([0], [0], marker='>', color='w', markerfacecolor='white', markeredgecolor='black', 
             markersize=10, markeredgewidth=2, linestyle='None', label='Initial'),
      Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', 
             markersize=10, markeredgewidth=2, linestyle='None', label='Final'),
  ]
  leg_sun = ax_sun.legend(handles=legend_handles_sun, loc='upper right', fontsize=10, framealpha=0.9)
  leg_sun.get_frame().set_edgecolor('black')
  
  # Set pane colors to white
  ax_sun.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax_sun.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax_sun.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  
  # Set limits for heliocentric plot (Variable limits with equal scaling)
  # Collect all data points
  xs_sun = [0.0] # Sun at origin
  ys_sun = [0.0]
  zs_sun = [0.0]
  
  if earth_orbit_helio is not None:
      xs_sun.extend(earth_orbit_helio[0, :])
      ys_sun.extend(earth_orbit_helio[1, :])
      zs_sun.extend(earth_orbit_helio[2, :])
      
  if sc_pos_helio is not None:
      xs_sun.extend(sc_pos_helio[0, :])
      ys_sun.extend(sc_pos_helio[1, :])
      zs_sun.extend(sc_pos_helio[2, :])

  if moon_pos_helio is not None:
      xs_sun.extend(moon_pos_helio[0, :])
      ys_sun.extend(moon_pos_helio[1, :])
      zs_sun.extend(moon_pos_helio[2, :])

  # Convert to numpy arrays for min/max
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
  
  ax_sun.set_xlim([x_min - buffer*dx, x_max + buffer*dx])
  ax_sun.set_ylim([y_min - buffer*dy, y_max + buffer*dy])
  ax_sun.set_zlim([z_min - buffer*dz, z_max + buffer*dz]) # type: ignore
  
  # Set box aspect to match data range ratios (maintains equal scale)
  ax_sun.set_box_aspect((dx, dy, dz)) # type: ignore

  # Plot on both axes
  for ax in axes:
    # Add Earth wireframe ellipsoid
    ax.plot_wireframe(x_earth, y_earth, z_earth, color='blue', linewidth=0.5, alpha=0.6)
    
    # Plot spacecraft trajectory
    ax.plot(pos_x, pos_y, pos_z, 'b-', linewidth=2.0, label='Spacecraft')
    ax.scatter([pos_x[0]], [pos_y[0]], [pos_z[0]], s=100, marker='>', facecolors='white', edgecolors='b', linewidths=2, zorder=5)
    ax.scatter([pos_x[-1]], [pos_y[-1]], [pos_z[-1]], s=100, marker='s', facecolors='white', edgecolors='b', linewidths=2, zorder=5)
    
    if moon_pos is not None:
      # Plot Moon trajectory segment
      ax.plot(moon_pos[0, :], moon_pos[1, :], moon_pos[2, :], 
              color='gray', linewidth=2.0, alpha=0.8, label='Moon (Sim)')
      
      # Add Moon position markers (initial and final)
      ax.scatter([moon_pos[0, 0]], [moon_pos[1, 0]], [moon_pos[2, 0]], 
                 s=100, marker='>', facecolors='white', edgecolors='gray', linewidths=2, zorder=5)
      ax.scatter([moon_pos[0, -1]], [moon_pos[1, -1]], [moon_pos[2, -1]], 
                 s=100, marker='s', facecolors='white', edgecolors='gray', linewidths=2, zorder=5)
      
      # Add Moon sphere at final position
      if x_moon_sphere is not None:
        ax.plot_wireframe(x_moon_sphere, y_moon_sphere, z_moon_sphere, color='gray', linewidth=0.3, alpha=0.5)
      
      # Plot full orbit as a thin dashed line
      if moon_orbit_full is not None:
        ax.plot(moon_orbit_full[0, :], moon_orbit_full[1, :], moon_orbit_full[2, :],
                color='gray', linewidth=1.0, linestyle='--', alpha=0.5, label='Moon Orbit (Approx)')
    
    # Plot Sun only on left subplot (ax1)
    if ax == ax1 and sun_pos is not None:
      ax.plot(sun_pos[0, :], sun_pos[1, :], sun_pos[2, :], 
              color='orange', linewidth=2.0, alpha=0.8, label='Sun (Sim)')
      
      # Add Sun position markers
      ax.scatter([sun_pos[0, 0]], [sun_pos[1, 0]], [sun_pos[2, 0]], 
                 s=100, marker='>', facecolors='white', edgecolors='orange', linewidths=2, zorder=5)
      ax.scatter([sun_pos[0, -1]], [sun_pos[1, -1]], [sun_pos[2, -1]], 
                 s=100, marker='s', facecolors='white', edgecolors='orange', linewidths=2, zorder=5)
      
      # Plot full Sun orbit
      if sun_orbit_full is not None:
        ax.plot(sun_orbit_full[0, :], sun_orbit_full[1, :], sun_orbit_full[2, :],
                color='orange', linewidth=1.0, linestyle='--', alpha=0.5, label='Sun Orbit (Approx)')
    
    ax.set_xlabel('Pos-X [m]')
    ax.set_ylabel('Pos-Y [m]')
    ax.set_zlabel('Pos-Z [m]')  # type: ignore
    ax.grid(True)

    # Set pane colors to white
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

       # Configure limits and aspect ratio
    z_bottom = 0.0
    
    if ax == ax1:
      # Left subplot: Independent limits but equal scale (non-cubic box)
      # Collect all data points to determine limits
      xs = [x_earth.flatten(), pos_x]
      ys = [y_earth.flatten(), pos_y]
      zs = [z_earth.flatten(), pos_z]
      
      if moon_pos is not None:
        xs.append(moon_pos[0, :])
        ys.append(moon_pos[1, :])
        zs.append(moon_pos[2, :])
        if moon_orbit_full is not None:
          xs.append(moon_orbit_full[0, :])
          ys.append(moon_orbit_full[1, :])
          zs.append(moon_orbit_full[2, :])
          
      if sun_pos is not None:
        xs.append(sun_pos[0, :])
        ys.append(sun_pos[1, :])
        zs.append(sun_pos[2, :])
        if sun_orbit_full is not None:
          xs.append(sun_orbit_full[0, :])
          ys.append(sun_orbit_full[1, :])
          zs.append(sun_orbit_full[2, :])
      
      all_x = np.concatenate(xs)
      all_y = np.concatenate(ys)
      all_z = np.concatenate(zs)
      
      x_min, x_max = np.min(all_x), np.max(all_x)
      y_min, y_max = np.min(all_y), np.max(all_y)
      z_min, z_max = np.min(all_z), np.max(all_z)
      
      # Add buffer
      buffer = 0.1
      dx = x_max - x_min
      dy = y_max - y_min
      dz = z_max - z_min
      
      # Avoid zero range
      if dx == 0: dx = 1.0
      if dy == 0: dy = 1.0
      if dz == 0: dz = 1.0
      
      ax.set_xlim([x_min - buffer*dx, x_max + buffer*dx])
      ax.set_ylim([y_min - buffer*dy, y_max + buffer*dy])
      ax.set_zlim([z_min - buffer*dz, z_max + buffer*dz])
      
      # Set box aspect to match data range ratios (maintains equal scale)
      ax.set_box_aspect((dx, dy, dz)) # type: ignore
      
      z_bottom = z_min - buffer*dz
      
    else:
      # Right subplot: Equal limits (Cubic box)
      ax.set_box_aspect([1, 1, 1]) # type: ignore
      min_limit, max_limit = get_equal_limits(ax, buffer_fraction=0.1)
      ax.set_xlim([min_limit, max_limit])
      ax.set_ylim([min_limit, max_limit])
      ax.set_zlim([min_limit, max_limit])
      z_bottom = min_limit

    # Add trajectory shadow on XY plane
    shadow_alpha = 0.4
    shadow_lw    = 0.5
    ax.plot(pos_x, pos_y, np.full_like(pos_z, z_bottom), color='lightblue', alpha=shadow_alpha, linewidth=shadow_lw)
    
    # Moon trajectory shadow
    if moon_pos is not None:
      ax.plot(moon_pos[0, :], moon_pos[1, :], np.full(n_moon_points, z_bottom), 
              color='lightgray', alpha=shadow_alpha, linewidth=shadow_lw)
    
    # Sun trajectory shadow (only ax1)
    if ax == ax1 and sun_pos is not None:
      ax.plot(sun_pos[0, :], sun_pos[1, :], np.full(n_sun_points, z_bottom), 
              color='orange', alpha=shadow_alpha, linewidth=shadow_lw)

    # Create custom legend
    legend_handles = [
      Line2D([0], [0], color='b', linewidth=2, label='Spacecraft'),
      Line2D([0], [0], color='gray', linewidth=2, label='Moon (Sim)'),
      Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Moon Orbit (Approx)'),
      Line2D([0], [0], marker='>', color='w', markerfacecolor='white', markeredgecolor='black', 
             markersize=10, markeredgewidth=2, linestyle='None', label='Initial'),
      Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', 
             markersize=10, markeredgewidth=2, linestyle='None', label='Final'),
    ]
    
    # Add Sun to legend for ax1
    if ax == ax1 and sun_pos is not None:
      legend_handles.insert(1, Line2D([0], [0], color='orange', linewidth=2, label='Sun (Sim)'))
      legend_handles.insert(2, Line2D([0], [0], color='orange', linewidth=1, linestyle='--', label='Sun Orbit (Approx)'))
      
    leg = ax.legend(handles=legend_handles, loc='upper left', fontsize=10, framealpha=0.9)
    leg.get_frame().set_edgecolor('black')

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
    }
    
    fig_err_ts = plot_time_series_error(
      result_ref  = result_jpl_horizons_ephemeris,
      result_comp = hf_at_ephem,
      epoch       = time_o_dt,
      use_ric     = True,
    )
    title = f'RIC Errors: High-Fidelity vs JPL Horizons - {object_name_display}'
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
    title = f'RIC Errors: High-Fidelity vs SGP4 - {object_name_display}'
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
    filename = f'3d_j2000_jpl_horizons_{name_lower}.png'
    fig1.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Inertial    : <figures_folderpath>/{filename}")
    plt.close(fig1)
    
    fig2 = plot_time_series(result_jpl_horizons_ephemeris, epoch=time_o_dt)
    fig2.suptitle(f'Time Series - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'timeseries_jpl_horizons_{name_lower}.png'
    fig2.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Time Series    : <figures_folderpath>/{filename}")
    plt.close(fig2)
    
    # Earth-fixed 3D plot for Horizons
    fig_ef = plot_3d_trajectories_earth_fixed(result_jpl_horizons_ephemeris, epoch_dt_utc=time_o_dt)
    fig_ef.suptitle(f'3D Earth-Fixed - {object_name_display} - JPL Horizons', fontsize=16)
    filename = f'3d_iau_earth_jpl_horizons_{name_lower}.png'
    fig_ef.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Earth-Fixed : <figures_folderpath>/{filename}")
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
    filename = f'3d_j2000_high_fidelity_{name_lower}.png'
    fig3.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Inertial    : <figures_folderpath>/{filename}")
    plt.close(fig3)
    
    fig4 = plot_time_series(result_high_fidelity_propagation, epoch=time_o_dt)
    fig4.suptitle(f'Time Series - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'timeseries_high_fidelity_{name_lower}.png'
    fig4.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Time Series    : <figures_folderpath>/{filename}")
    plt.close(fig4)
    
    # Earth-fixed 3D plot
    fig_ef = plot_3d_trajectories_earth_fixed(result_high_fidelity_propagation, epoch_dt_utc=time_o_dt)
    fig_ef.suptitle(f'3D Earth-Fixed - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_iau_earth_high_fidelity_{name_lower}.png'
    fig_ef.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D Earth-Fixed : <figures_folderpath>/{filename}")
    plt.close(fig_ef)
    
    # Ground track plot
    gt_title = f'Ground Track - {object_name_display} - High-Fidelity'
    fig_gt = plot_ground_track(result_high_fidelity_propagation, epoch_dt_utc=time_o_dt, title_text=gt_title)
    filename = f'groundtrack_high_fidelity_{name_lower}.png'
    fig_gt.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Ground Track   : <figures_folderpath>/{filename}")
    plt.close(fig_gt)
    
    # 3D plot with Moon trajectory
    fig_moon = plot_3d_trajectory_with_moon(result_high_fidelity_propagation, epoch=time_o_dt)
    fig_moon.suptitle(f'3D with Moon - {object_name_display} - High-Fidelity', fontsize=16)
    filename = f'3d_j2000_with_moon_high_fidelity_{name_lower}.png'
    fig_moon.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D with Moon   : <figures_folderpath>/{filename}")
    plt.close(fig_moon)
  
  # SGP4 plots
  if compare_tle and result_sgp4_propagation and result_sgp4_propagation.get('success'):
    print("    SGP4-Model Plots")
    
    # 3D trajectory plot
    fig_sgp4_3d = plot_3d_trajectories(result_sgp4_propagation, epoch=time_o_dt, frame="J2000")
    fig_sgp4_3d.suptitle(f'3D Inertial - {object_name_display} - SGP4', fontsize=16)
    filename = f'3d_j2000_sgp4_{name_lower}.png'
    fig_sgp4_3d.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      3D          : <figures_folderpath>/{filename}")
    plt.close(fig_sgp4_3d)
    
    # Time series plot
    fig_sgp4_ts = plot_time_series(result_sgp4_propagation, epoch=time_o_dt)
    fig_sgp4_ts.suptitle(f'Time Series - {object_name_display} - SGP4', fontsize=16)
    filename = f'timeseries_sgp4_{name_lower}.png'
    fig_sgp4_ts.savefig(figures_folderpath / filename, dpi=300, bbox_inches='tight')
    print(f"      Time Series : <figures_folderpath>/{filename}")
    plt.close(fig_sgp4_ts)

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