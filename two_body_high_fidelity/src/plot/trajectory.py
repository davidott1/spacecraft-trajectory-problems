import matplotlib.pyplot as plt
import numpy             as np
import datetime

from datetime          import timedelta
from pathlib           import Path
from typing            import Optional
from matplotlib.figure import Figure
from matplotlib.lines  import Line2D

from src.plot.utility          import get_equal_limits, add_utc_time_axis, add_stats
from src.model.constants       import CONVERTER, PHYSICALCONSTANTS
from src.model.frame_converter import FrameConverter


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
    matplotlib.figure.Figure
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
    info_text += f"  |  Start: {start_utc}  |  End: {end_utc}"
  
  # Plot 3D position trajectory
  ax1 = fig.add_subplot(121, projection='3d')
  
  # Add Earth ellipsoid
  u       = np.linspace(0, 2 * np.pi, 50)
  v       = np.linspace(0, np.pi, 50)
  r_eq    = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR
  r_pol   = PHYSICALCONSTANTS.EARTH.RADIUS.POLAR
  x_earth = r_eq * np.outer(np.cos(u), np.sin(v))
  y_earth = r_eq * np.outer(np.sin(u), np.sin(v))
  z_earth = r_pol * np.outer(np.ones(np.size(u)), np.cos(v))
  ax1.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3, edgecolor='none') # type: ignore
  
  ax1.plot(pos_x, pos_y, pos_z, 'b-', linewidth=1)
  ax1.scatter([pos_x[0]], [pos_y[0]], [pos_z[0]], s=100, marker='>', facecolors='white', edgecolors='b', linewidths=2) # type: ignore
  ax1.scatter([pos_x[-1]], [pos_y[-1]], [pos_z[-1]], s=100, marker='s', facecolors='white', edgecolors='b', linewidths=2) # type: ignore
  ax1.set_xlabel('Pos-X [m]')
  ax1.set_ylabel('Pos-Y [m]')
  ax1.set_zlabel('Pos-Z [m]') # type: ignore
  ax1.grid(True)
  ax1.set_box_aspect([1,1,1]) # type: ignore
  min_limit, max_limit = get_equal_limits(ax1)
  ax1.set_xlim([min_limit, max_limit]) # type: ignore
  ax1.set_ylim([min_limit, max_limit]) # type: ignore
  ax1.set_zlim([min_limit, max_limit]) # type: ignore

  # Plot 3D velocity trajectory
  ax2 = fig.add_subplot(122, projection='3d')
  ax2.plot(vel_x, vel_y, vel_z, 'r-', linewidth=1)
  ax2.scatter([vel_x[0]], [vel_y[0]], [vel_z[0]], s=100, marker='>', facecolors='white', edgecolors='r', linewidths=2) # type: ignore
  ax2.scatter([vel_x[-1]], [vel_y[-1]], [vel_z[-1]], s=100, marker='s', facecolors='white', edgecolors='r', linewidths=2) # type: ignore
  ax2.set_xlabel('Vel-X [m/s]')
  ax2.set_ylabel('Vel-Y [m/s]')
  ax2.set_zlabel('Vel-Z [m/s]') # type: ignore
  ax2.grid(True)
  ax2.set_box_aspect([1,1,1]) # type: ignore
  min_limit, max_limit = get_equal_limits(ax2)
  ax2.set_xlim([min_limit, max_limit]) # type: ignore
  ax2.set_ylim([min_limit, max_limit]) # type: ignore
  ax2.set_zlim([min_limit, max_limit]) # type: ignore

  # Create custom legend handles with black edges
  legend_handles = [
    Line2D([0], [0], marker='>', color='w', markerfacecolor='white', markeredgecolor='black', 
           markersize=10, markeredgewidth=2, linestyle='None', label='Start'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='black', 
           markersize=10, markeredgewidth=2, linestyle='None', label='End'),
  ]
  fig.legend(handles=legend_handles, loc='upper right', fontsize=11, framealpha=0.9)

  # Add info text as figure text
  fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=11, color='black',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

  plt.tight_layout(rect=[0, 0.06, 1, 0.95])  # Leave space at bottom for info text and top for legend
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
    matplotlib.figure.Figure
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

  # Plot argp vs time (row 4, column 1)
  ax_argp = plt.subplot2grid((6, 2), (4, 1), sharex=ax_pos)
  argp_unwrapped = np.unwrap(coe['argp']) * CONVERTER.DEG_PER_RAD
  ax_argp.plot(time, argp_unwrapped, 'b-', linewidth=1.5)
  ax_argp.tick_params(labelbottom=False)
  ax_argp.set_ylabel('Argument of Perigee\n[deg]')
  ax_argp.grid(True)
  ax_argp.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

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
  fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_argp, ax_anom])
  
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
    matplotlib.figure.Figure
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
              s=100, marker='>', facecolors='white', edgecolors='b', linewidths=2, label='Start') # type: ignore
  ax1.scatter([pos_error[0, -1]], [pos_error[1, -1]], [pos_error[2, -1]], 
              s=100, marker='s', facecolors='white', edgecolors='b', linewidths=2, label='End') # type: ignore
  ax1.set_xlabel('Error X [m]')
  ax1.set_ylabel('Error Y [m]')
  ax1.set_zlabel('Error Z [m]') # type: ignore
  ax1.set_title('Position Error')
  ax1.grid(True)
  ax1.legend()
  
  # Plot 3D velocity error
  ax2 = fig.add_subplot(122, projection='3d')
  ax2.plot(vel_error[0, :], vel_error[1, :], vel_error[2, :], 'r-', linewidth=1)
  ax2.scatter([vel_error[0, 0]], [vel_error[1, 0]], [vel_error[2, 0]], 
              s=100, marker='>', facecolors='white', edgecolors='r', linewidths=2, label='Start') # type: ignore
  ax2.scatter([vel_error[0, -1]], [vel_error[1, -1]], [vel_error[2, -1]], 
              s=100, marker='s', facecolors='white', edgecolors='r', linewidths=2, label='End') # type: ignore
  ax2.set_xlabel('Error Vx [m/s]')
  ax2.set_ylabel('Error Vy [m/s]')
  ax2.set_zlabel('Error Vz [m/s]') # type: ignore
  ax2.set_title('Velocity Error')
  ax2.grid(True)
  ax2.legend()
  
  fig.suptitle(title, fontsize=16)
  plt.tight_layout()
  return fig


def plot_time_series_error(
  result_ref  : dict, 
  result_comp : dict, 
  epoch       : Optional[datetime.datetime] = None, 
  title       : str                         = "Time Series Error", 
  use_ric     : bool                        = True
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
    matplotlib.figure.Figure
      Figure object containing the time series error plots.
  """
  # Use plot_time_s for both datasets
  time_ref  = result_ref['plot_time_s']
  time_comp = result_comp['plot_time_s']
  
  # Check if time grids match
  if not np.array_equal(time_ref, time_comp):
    raise ValueError(
      f"Time grids don't match! "
      f"Reference has {len(time_ref)} points from {time_ref[0]:.1f} to {time_ref[-1]:.1f} s, "
      f"Comparison has {len(time_comp)} points from {time_comp[0]:.1f} to {time_comp[-1]:.1f} s. "
      f"Both datasets must use the same time grid for error comparison."
    )
  
  state_ref  = result_ref['state']
  state_comp = result_comp['state']
  coe_comp   = result_comp['coe']
  
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
  ax_pos.set_ylabel('Position\n[m]')
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
  ax_vel.set_ylabel('Velocity\n[m/s]')
  ax_vel.legend()
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot sma error vs time (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 2), (0, 1), sharex=ax_pos)
  if 'sma' in result_ref['coe'] and 'sma' in coe_comp:
    sma_error = result_ref['coe']['sma'] - coe_comp['sma']
    ax_sma.plot(time, sma_error, 'b-', linewidth=1.5)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA Error\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot eccentricity error vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 2), (1, 1), sharex=ax_pos)
  if 'ecc' in result_ref['coe'] and 'ecc' in coe_comp:
    ecc_error = result_ref['coe']['ecc'] - coe_comp['ecc']
    ax_ecc.plot(time, ecc_error, 'b-', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('Eccentricity Error\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inclination error (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 2), (2, 1), sharex=ax_pos)
  if 'inc' in result_ref['coe'] and 'inc' in coe_comp:
    inc_error = (result_ref['coe']['inc'] - coe_comp['inc']) * CONVERTER.DEG_PER_RAD
    ax_inc.plot(time, inc_error, 'b-', linewidth=1.5)
  ax_inc.tick_params(labelbottom=False)
  ax_inc.set_ylabel('Inclination Error\n[deg]')
  ax_inc.grid(True)
  ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot RAAN error vs time (row 3, column 1)
  ax_raan = plt.subplot2grid((6, 2), (3, 1), sharex=ax_pos)
  if 'raan' in result_ref['coe'] and 'raan' in coe_comp:
    # Handle angle wrapping for RAAN
    raan_error_rad = np.arctan2(np.sin(result_ref['coe']['raan'] - coe_comp['raan']), 
                   np.cos(result_ref['coe']['raan'] - coe_comp['raan']))
    raan_error = raan_error_rad * CONVERTER.DEG_PER_RAD
    ax_raan.plot(time, raan_error, 'b-', linewidth=1.5)
  ax_raan.tick_params(labelbottom=False)
  ax_raan.set_ylabel('RAAN Error\n[deg]')
  ax_raan.grid(True)
  ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot argument of perigee error (row 4, column 1)
  ax_argp = plt.subplot2grid((6, 2), (4, 1), sharex=ax_pos)
  if 'argp' in result_ref['coe'] and 'argp' in coe_comp:
    # Handle angle wrapping for ArgP
    argp_error_rad = np.arctan2(np.sin(result_ref['coe']['argp'] - coe_comp['argp']), 
                   np.cos(result_ref['coe']['argp'] - coe_comp['argp']))
    argp_error = argp_error_rad * CONVERTER.DEG_PER_RAD
    ax_argp.plot(time, argp_error, 'b-', linewidth=1.5)
  ax_argp.tick_params(labelbottom=False)
  ax_argp.set_ylabel('Argument of\nPerigee Error\n[deg]')
  ax_argp.grid(True)
  ax_argp.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot true anomaly error vs time (row 5, column 1)
  ax_ta = plt.subplot2grid((6, 2), (5, 1), sharex=ax_pos)
  if 'ta' in result_ref['coe'] and 'ta' in coe_comp:
    # Handle angle wrapping for TA
    ta_error_rad = np.arctan2(np.sin(result_ref['coe']['ta'] - coe_comp['ta']), 
                   np.cos(result_ref['coe']['ta'] - coe_comp['ta']))
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
  fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_argp, ax_ta])
  fig.align_ylabels([ax_pos, ax_vel])
  
  fig.suptitle(title, fontsize=16)
  plt.subplots_adjust(hspace=0.17, wspace=0.2)
  return fig


def plot_true_longitude_error(
  result_ref  : dict, 
  result_comp : dict, 
  epoch       : Optional[datetime.datetime] = None
) -> Figure:
  """
  Create position/velocity error plots in the RIC frame (reference = result_ref).
  
  Input:
  ------
    result_ref : dict
      Reference result dictionary.
    result_comp : dict
      Comparison result dictionary.
    epoch : datetime, optional
      Reference epoch for UTC time axis.
      
  Output:
  -------
    matplotlib.figure.Figure
      Figure object containing the RIC frame error plots.
  """
  time_ref  = result_ref['plot_time_s']
  time_comp = result_comp['plot_time_s']
  if not np.array_equal(time_ref, time_comp):
    raise ValueError(
      f"Time grids don't match! "
      f"Reference has {len(time_ref)} points from {time_ref[0]:.1f} to {time_ref[-1]:.1f} s, "
      f"Comparison has {len(time_comp)} points from {time_comp[0]:.1f} to {time_comp[-1]:.1f} s. "
      f"Both datasets must use the same time grid for error comparison."
    )

  state_ref  = result_ref['state']
  state_comp = result_comp['state']
  ric_delta_pos = np.zeros((3, len(time_ref)))
  ric_delta_vel = np.zeros((3, len(time_ref)))

  for i in range(len(time_ref)):
    xyz_ref_pos = state_ref[0:3, i]
    xyz_ref_vel = state_ref[3:6, i]
    
    rot_mat_xyz_to_ric = FrameConverter.xyz_to_ric(xyz_ref_pos, xyz_ref_vel)
    
    xyz_delta_pos_vec = state_comp[0:3, i] - xyz_ref_pos
    xyz_delta_vel_vec = state_comp[3:6, i] - xyz_ref_vel

    ric_delta_pos[:, i] = rot_mat_xyz_to_ric @ xyz_delta_pos_vec
    ric_delta_vel[:, i] = rot_mat_xyz_to_ric @ xyz_delta_vel_vec

  ric_delta_pos_mag = np.linalg.norm(ric_delta_pos, axis=0)
  ric_delta_vel_mag = np.linalg.norm(ric_delta_vel, axis=0)

  fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
  ax_pos.plot(time_ref, ric_delta_pos[0], 'r-', label='Radial')
  ax_pos.plot(time_ref, ric_delta_pos[1], 'g-', label='In-track')
  ax_pos.plot(time_ref, ric_delta_pos[2], 'b-', label='Cross-track')
  ax_pos.plot(time_ref, ric_delta_pos_mag, 'k-', linewidth=2, label='Magnitude')
  ax_pos.set_ylabel('Position Error [m]')
  ax_pos.grid(True, alpha=0.3)
  ax_pos.legend()

  ax_vel.plot(time_ref, ric_delta_vel[0], 'r-', label='Radial')
  ax_vel.plot(time_ref, ric_delta_vel[1], 'g-', label='In-track')
  ax_vel.plot(time_ref, ric_delta_vel[2], 'b-', label='Cross-track')
  ax_vel.plot(time_ref, ric_delta_vel_mag, 'k-', linewidth=2, label='Magnitude')
  ax_vel.set_ylabel('Velocity Error [m/s]')
  ax_vel.set_xlabel('Time [s]')
  ax_vel.grid(True, alpha=0.3)
  ax_vel.legend()

  add_stats(ax_pos, ric_delta_pos_mag, 'Pos [m]')
  add_stats(ax_vel, ric_delta_vel_mag, 'Vel [m/s]')

  if epoch is not None:
    add_utc_time_axis(ax_pos, epoch, time_ref[-1])

  fig.suptitle('RIC Frame Error (Reference = Horizons)', fontsize=14)
  fig.tight_layout()

  return fig


def generate_error_plots(
  result_jpl_horizons_ephemeris    : Optional[dict],
  result_high_fidelity_propagation : dict,
  result_sgp4_propagation          : Optional[dict],
  desired_time_o_dt                : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool,
  compare_tle                      : bool,
  object_name                      : str = "object",
) -> None:
  """
  Generate and save error comparison plots.
  
  Input:
  ------
    result_horizons : dict | None
      Horizons ephemeris result.
    result_high_fidelity : dict
      High-fidelity propagation result.
    result_sgp4 : dict | None
      SGP4 propagation result.
    desired_time_o_dt : datetime
      Simulation start time (for plot labels).
    figures_folderpath : Path
      Directory to save plots.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    compare_tle : bool
      Flag to enable comparison with TLE/SGP4.
    object_name : str
      Name of the object for plot titles and filenames.
      
  Output:
  -------
    None
  """
  # Only proceed if we have something to compare against (Horizons) and comparison is requested
  if not compare_jpl_horizons or not (result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success')):
    return

  print("\n  Generate Error Plots")
  
  # Lowercase name for filenames
  name_lower = object_name.lower()

  # Create error comparison plots if both Horizons and high-fidelity are available
  if result_high_fidelity_propagation.get('success'):
    print("    High-Fidelity Model Relative to JPL Horizons")
    
    # Position and velocity error plots
    fig_err_3d = plot_3d_error(result_jpl_horizons_ephemeris, result_high_fidelity_propagation)
    fig_err_3d.suptitle(f'{object_name} Orbit Error: Horizons vs High-Fidelity', fontsize=16)
    fig_err_3d.savefig(figures_folderpath / f'error_3d_{name_lower}_high_fidelity.png', dpi=300, bbox_inches='tight')
    print(f"      3D Error          : <figures_folderpath>/error_3d_{name_lower}_high_fidelity.png")
    
    # Time series error plots
    fig_err_ts = plot_time_series_error(result_jpl_horizons_ephemeris, result_high_fidelity_propagation, epoch=desired_time_o_dt)
    fig_err_ts.suptitle(f'{object_name} RIC Position/Velocity Errors: Horizons vs High-Fidelity', fontsize=16)
    fig_err_ts.savefig(figures_folderpath / f'error_timeseries_{name_lower}_high_fidelity.png', dpi=300, bbox_inches='tight')
    print(f"      Time-Series Error : <figures_folderpath>/error_timeseries_{name_lower}_high_fidelity.png")

  # Error plots comparing SGP4 to Horizons
  if compare_tle and result_sgp4_propagation and result_sgp4_propagation.get('success'):
    print("    SGP4 Model Relative to JPL Horizons")

    # 3D error plot
    fig_sgp4_err_3d = plot_3d_error(result_jpl_horizons_ephemeris, result_sgp4_propagation)
    fig_sgp4_err_3d.suptitle(f'{object_name} Orbit Error: Horizons vs SGP4', fontsize=16)
    fig_sgp4_err_3d.savefig(figures_folderpath / f'error_3d_{name_lower}_sgp4.png', dpi=300, bbox_inches='tight')
    print(f"      3D Error          : <figures_folderpath>/error_3d_{name_lower}_sgp4.png")
    
    # Time series error plot (RIC frame)
    fig_sgp4_err_ts = plot_time_series_error(result_jpl_horizons_ephemeris, result_sgp4_propagation, epoch=desired_time_o_dt, use_ric=False)
    fig_sgp4_err_ts.suptitle(f'{object_name} XYZ Position/Velocity Errors: Horizons vs SGP4', fontsize=16)
    fig_sgp4_err_ts.savefig(figures_folderpath / f'error_timeseries_{name_lower}_sgp4.png', dpi=300, bbox_inches='tight')
    print(f"      Time-Series Error : <figures_folderpath>/error_timeseries_{name_lower}_sgp4.png")


def generate_3d_and_time_series_plots(
  result_jpl_horizons_ephemeris    : Optional[dict],
  result_high_fidelity_propagation : dict,
  result_sgp4_propagation          : Optional[dict],
  desired_time_o_dt                : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool,
  compare_tle                      : bool,
  object_name                      : str = "object",
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
    desired_time_o_dt : datetime
      Simulation start time (for plot labels).
    figures_folderpath : Path
      Directory to save plots.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    compare_tle : bool
      Flag to enable comparison with TLE/SGP4.
    object_name : str
      Name of the object for plot titles and filenames.
      
  Output:
  -------
    None
  """
  print("  Generate 3D-Trajectory and Time-Series Plots")
  
  # Lowercase name for filenames
  name_lower = object_name.lower()

  # Horizons plots (first)
  if compare_jpl_horizons and result_jpl_horizons_ephemeris and result_jpl_horizons_ephemeris.get('success'):
    print("    JPL-Horizons-Ephemeris Plots")

    fig1 = plot_3d_trajectories(result_jpl_horizons_ephemeris, epoch=desired_time_o_dt, frame="J2000")
    fig1.suptitle(f'{object_name} Orbit - JPL Horizons - 3D', fontsize=16)
    fig1.savefig(figures_folderpath / f'3d_{name_lower}_jpl_horizons.png', dpi=300, bbox_inches='tight')
    print(f"      3D          : <figures_folderpath>/3d_{name_lower}_jpl_horizons.png")
    
    fig2 = plot_time_series(result_jpl_horizons_ephemeris, epoch=desired_time_o_dt)
    fig2.suptitle(f'{object_name} Orbit - JPL Horizons - Time Series', fontsize=16)
    fig2.savefig(figures_folderpath / f'timeseries_{name_lower}_jpl_horizons.png', dpi=300, bbox_inches='tight')
    print(f"      Time Series : <figures_folderpath>/timeseries_{name_lower}_jpl_horizons.png")
  
  # High-fidelity plots (second)
  if result_high_fidelity_propagation.get('success'):
    print("    High-Fidelity-Model Plots")

    fig3 = plot_3d_trajectories(result_high_fidelity_propagation, epoch=desired_time_o_dt, frame="J2000")
    fig3.suptitle(f'{object_name} Orbit - High-Fidelity Model - 3D', fontsize=16)
    fig3.savefig(figures_folderpath / f'3d_{name_lower}_high_fidelity.png', dpi=300, bbox_inches='tight')
    print(f"      3D          : <figures_folderpath>/3d_{name_lower}_high_fidelity.png")
    
    fig4 = plot_time_series(result_high_fidelity_propagation, epoch=desired_time_o_dt)
    fig4.suptitle(f'{object_name} Orbit - High-Fidelity Model - Time Series', fontsize=16)
    fig4.savefig(figures_folderpath / f'timeseries_{name_lower}_high_fidelity.png', dpi=300, bbox_inches='tight')
    print(f"      Time Series : <figures_folderpath>/timeseries_{name_lower}_high_fidelity.png")
  
  # SGP4 at Horizons time points plots
  if compare_tle and result_sgp4_propagation and result_sgp4_propagation.get('success'):
    print("    SGP4-Model Plots")
    
    # 3D trajectory plot
    fig_sgp4_hz_3d = plot_3d_trajectories(result_sgp4_propagation, epoch=desired_time_o_dt, frame="J2000")
    fig_sgp4_hz_3d.suptitle(f'{object_name} Orbit - SGP4 Model - 3D', fontsize=16)
    fig_sgp4_hz_3d.savefig(figures_folderpath / f'3d_{name_lower}_sgp4.png', dpi=300, bbox_inches='tight')
    print(f"      3D          : <figures_folderpath>/3d_{name_lower}_sgp4.png")
    
    # Time series plot
    fig_sgp4_hz_ts = plot_time_series(result_sgp4_propagation, epoch=desired_time_o_dt)
    fig_sgp4_hz_ts.suptitle(f'{object_name} Orbit - SGP4 Model - Time Series', fontsize=16)
    fig_sgp4_hz_ts.savefig(figures_folderpath / f'timeseries_{name_lower}_sgp4.png', dpi=300, bbox_inches='tight')
    print(f"      Time Series : <figures_folderpath>/timeseries_{name_lower}_sgp4.png")


def generate_plots(
  result_jpl_horizons_ephemeris    : Optional[dict],
  result_high_fidelity_propagation : dict,
  result_sgp4_propagation          : Optional[dict],
  desired_time_o_dt                : datetime.datetime,
  figures_folderpath               : Path,
  compare_jpl_horizons             : bool = False,
  compare_tle                      : bool = False,
  object_name                      : str  = "object",
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
    desired_time_o_dt : datetime
      Simulation start time (for plot labels).
    figures_folderpath : Path
      Directory to save plots.
    compare_jpl_horizons : bool
      Flag to enable comparison with Horizons.
    compare_tle : bool
      Flag to enable comparison with TLE/SGP4.
    object_name : str
      Name of the object for plot titles and filenames.
      
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
    desired_time_o_dt                = desired_time_o_dt,
    figures_folderpath               = figures_folderpath,
    compare_jpl_horizons             = compare_jpl_horizons,
    compare_tle                      = compare_tle,
    object_name                      = object_name,
  )
    
  # Generate error plots
  generate_error_plots(
    result_jpl_horizons_ephemeris    = result_jpl_horizons_ephemeris,
    result_high_fidelity_propagation = result_high_fidelity_propagation,
    result_sgp4_propagation          = result_sgp4_propagation,
    desired_time_o_dt                = desired_time_o_dt,
    figures_folderpath               = figures_folderpath,
    compare_jpl_horizons             = compare_jpl_horizons,
    compare_tle                      = compare_tle,
    object_name                      = object_name,
  )