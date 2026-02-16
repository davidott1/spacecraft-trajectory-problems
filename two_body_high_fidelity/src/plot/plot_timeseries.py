"""
Time series plotting functions.

This module contains functions for plotting time series of state vectors,
orbital elements, and error comparisons.
"""
import numpy             as np
import matplotlib.pyplot as plt

from datetime          import datetime, timedelta
from typing            import Optional
from matplotlib.figure import Figure

from src.plot.utility          import add_utc_time_axis
from src.model.constants       import CONVERTER
from src.model.frame_converter import FrameConverter
from src.schemas.propagation   import PropagationResult


def plot_time_series(
  result : PropagationResult,
  epoch  : Optional[datetime] = None,
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
  time   = result.time_grid.grid.relative_initial
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


def plot_time_series_error(
  result_ref     : PropagationResult, 
  result_comp    : PropagationResult, 
  epoch          : Optional[datetime] = None, 
  title          : str                = "Time Series Error", 
  use_ric        : bool               = True,
  result_smoother: Optional[PropagationResult] = None,
) -> Figure:
  """
  Create time series error plots between reference and comparison trajectories.
  
  Input:
  ------
    result_ref : PropagationResult
      Reference result with 'plot_delta_time' and 'state'/'coe'/'mee'.
    result_comp : PropagationResult
      Comparison result (filter) with 'plot_delta_time' and 'state'/'coe'/'mee'.
    epoch : datetime, optional
      Reference epoch for time axis.
    title : str
      Title for the figure.
    use_ric : bool
      If True, transform to RIC frame. If False, use XYZ inertial frame.
    result_smoother : PropagationResult, optional
      Smoother result with 'plot_delta_time' and 'state'/'coe'/'mee'.
      If provided, smoother errors are overlaid with dashed lines.
      
  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the time series error plots.
  """
  # Use time_grid.grid.relative_initial for both datasets
  time_ref  = result_ref.time_grid.grid.relative_initial
  time_comp = result_comp.time_grid.grid.relative_initial
  
  state_ref  = result_ref.state
  state_comp = result_comp.state
  coe_ref    = result_ref.coe
  coe_comp   = result_comp.coe
  mee_ref    = result_ref.mee
  mee_comp   = result_comp.mee
  
  # Smoother data (if provided)
  has_smoother = result_smoother is not None
  if has_smoother:
    time_smoother = result_smoother.time_grid.grid.relative_initial
    state_smoother = result_smoother.state
    coe_smoother = result_smoother.coe
    mee_smoother = result_smoother.mee
    
    # Verify smoother time grid matches reference
    if len(time_ref) != len(time_smoother) or not np.allclose(time_ref, time_smoother, rtol=1e-9, atol=1e-9):
      raise ValueError(
        f"Smoother time grid doesn't match reference! "
        f"Reference has {len(time_ref)} points, smoother has {len(time_smoother)} points."
      )
  
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
    # Compute RIC frame errors for filter
    pos_error_ric = np.zeros((3, len(time_ref)))
    vel_error_ric = np.zeros((3, len(time_ref)))
    
    # Compute RIC frame errors for smoother (if provided)
    if has_smoother:
      pos_error_smoother_ric = np.zeros((3, len(time_ref)))
      vel_error_smoother_ric = np.zeros((3, len(time_ref)))
    
    for i in range(len(time_ref)):
      # Reference position and velocity
      ref_pos = state_ref[0:3, i]
      ref_vel = state_ref[3:6, i]
      
      # Rotation matrix from inertial to RIC
      R_inertial_to_ric = FrameConverter.xyz_to_ric(ref_pos, ref_vel)
      
      # Compute filter errors in inertial frame
      pos_error_inertial = state_comp[0:3, i] - state_ref[0:3, i]
      vel_error_inertial = state_comp[3:6, i] - state_ref[3:6, i]
      
      # Transform filter errors to RIC frame
      pos_error_ric[:, i] = R_inertial_to_ric @ pos_error_inertial
      vel_error_ric[:, i] = R_inertial_to_ric @ vel_error_inertial
      
      # Compute smoother errors in RIC frame (if provided)
      if has_smoother:
        pos_error_smoother_inertial = state_smoother[0:3, i] - state_ref[0:3, i]
        vel_error_smoother_inertial = state_smoother[3:6, i] - state_ref[3:6, i]
        pos_error_smoother_ric[:, i] = R_inertial_to_ric @ pos_error_smoother_inertial
        vel_error_smoother_ric[:, i] = R_inertial_to_ric @ vel_error_smoother_inertial
    
    pos_error = pos_error_ric
    vel_error = vel_error_ric
    if has_smoother:
      pos_error_smoother = pos_error_smoother_ric
      vel_error_smoother = vel_error_smoother_ric
    pos_labels = ['Radial', 'In-track', 'Cross-track']
    vel_labels = ['Radial', 'In-track', 'Cross-track']
    pos_ylabel = 'Position Error (RIC)\n[m]'
    vel_ylabel = 'Velocity Error (RIC)\n[m/s]'
  else:
    # Use XYZ inertial frame errors
    pos_error = state_comp[0:3, :] - state_ref[0:3, :]
    vel_error = state_comp[3:6, :] - state_ref[3:6, :]
    if has_smoother:
      pos_error_smoother = state_smoother[0:3, :] - state_ref[0:3, :]
      vel_error_smoother = state_smoother[3:6, :] - state_ref[3:6, :]
    pos_labels = ['X', 'Y', 'Z']
    vel_labels = ['X', 'Y', 'Z']
    pos_ylabel = 'Position Error (XYZ)\n[m]'
    vel_ylabel = 'Velocity Error (XYZ)\n[m/s]'
  
  # Calculate error magnitudes
  pos_error_mag = np.linalg.norm(pos_error, axis=0)
  vel_error_mag = np.linalg.norm(vel_error, axis=0)
  if has_smoother:
    pos_error_smoother_mag = np.linalg.norm(pos_error_smoother, axis=0)
    vel_error_smoother_mag = np.linalg.norm(vel_error_smoother, axis=0)
  
  # LEFT COLUMN: Position and Velocity Errors
  # Plot position error vs time (spans rows 0-2, column 0)
  ax_pos = plt.subplot2grid((6, 3), (0, 0), rowspan=3)
  # Filter errors (solid lines)
  filter_label_suffix = ' (Filter)' if has_smoother else ''
  ax_pos.plot(time, pos_error[0, :], 'r-', label=pos_labels[0] + filter_label_suffix, linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  ax_pos.plot(time, pos_error[1, :], 'g-', label=pos_labels[1] + filter_label_suffix, linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  ax_pos.plot(time, pos_error[2, :], 'b-', label=pos_labels[2] + filter_label_suffix, linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  ax_pos.plot(time, pos_error_mag, 'k-', label='Magnitude' + filter_label_suffix, linewidth=2, alpha=0.7 if has_smoother else 1.0)
  # Smoother errors (dashed lines)
  if has_smoother:
    ax_pos.plot(time, pos_error_smoother[0, :], 'r--', label=pos_labels[0] + ' (Smoother)', linewidth=1.5)
    ax_pos.plot(time, pos_error_smoother[1, :], 'g--', label=pos_labels[1] + ' (Smoother)', linewidth=1.5)
    ax_pos.plot(time, pos_error_smoother[2, :], 'b--', label=pos_labels[2] + ' (Smoother)', linewidth=1.5)
    ax_pos.plot(time, pos_error_smoother_mag, 'k--', label='Magnitude (Smoother)', linewidth=2)
  ax_pos.tick_params(labelbottom=False)
  ax_pos.set_ylabel(pos_ylabel)
  ax_pos.legend(fontsize=7, ncol=2 if has_smoother else 1)
  ax_pos.grid(True)
  ax_pos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot velocity error vs time (spans rows 3-5, column 0)
  ax_vel = plt.subplot2grid((6, 3), (3, 0), rowspan=3, sharex=ax_pos)
  # Filter errors (solid lines)
  ax_vel.plot(time, vel_error[0, :], 'r-', label=vel_labels[0] + filter_label_suffix, linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  ax_vel.plot(time, vel_error[1, :], 'g-', label=vel_labels[1] + filter_label_suffix, linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  ax_vel.plot(time, vel_error[2, :], 'b-', label=vel_labels[2] + filter_label_suffix, linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  ax_vel.plot(time, vel_error_mag, 'k-', label='Magnitude' + filter_label_suffix, linewidth=2, alpha=0.7 if has_smoother else 1.0)
  # Smoother errors (dashed lines)
  if has_smoother:
    ax_vel.plot(time, vel_error_smoother[0, :], 'r--', label=vel_labels[0] + ' (Smoother)', linewidth=1.5)
    ax_vel.plot(time, vel_error_smoother[1, :], 'g--', label=vel_labels[1] + ' (Smoother)', linewidth=1.5)
    ax_vel.plot(time, vel_error_smoother[2, :], 'b--', label=vel_labels[2] + ' (Smoother)', linewidth=1.5)
    ax_vel.plot(time, vel_error_smoother_mag, 'k--', label='Magnitude (Smoother)', linewidth=2)
  ax_vel.set_xlabel('Time\n[s]')
  ax_vel.set_ylabel(vel_ylabel)
  ax_vel.legend(fontsize=7, ncol=2 if has_smoother else 1)
  ax_vel.grid(True)
  ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # MIDDLE COLUMN: Classical Orbital Elements Errors
  # Plot sma error vs time (row 0, column 1)
  ax_sma = plt.subplot2grid((6, 3), (0, 1), sharex=ax_pos)
  if coe_ref.sma is not None and coe_comp.sma is not None:
    sma_error = coe_comp.sma - coe_ref.sma
    ax_sma.plot(time, sma_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0, label='Filter' if has_smoother else None)
  if has_smoother and coe_ref.sma is not None and coe_smoother.sma is not None:
    sma_error_smoother = coe_smoother.sma - coe_ref.sma
    ax_sma.plot(time, sma_error_smoother, 'b--', linewidth=1.5, label='Smoother')
    ax_sma.legend(fontsize=7)
  ax_sma.tick_params(labelbottom=False)
  ax_sma.set_ylabel('SMA Error\n[m]')
  ax_sma.grid(True)
  ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot eccentricity error vs time (row 1, column 1)
  ax_ecc = plt.subplot2grid((6, 3), (1, 1), sharex=ax_pos)
  if coe_ref.ecc is not None and coe_comp.ecc is not None:
    ecc_error = coe_comp.ecc - coe_ref.ecc
    ax_ecc.plot(time, ecc_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and coe_ref.ecc is not None and coe_smoother.ecc is not None:
    ecc_error_smoother = coe_smoother.ecc - coe_ref.ecc
    ax_ecc.plot(time, ecc_error_smoother, 'b--', linewidth=1.5)
  ax_ecc.tick_params(labelbottom=False)
  ax_ecc.set_ylabel('ECC Error\n[-]')
  ax_ecc.grid(True)
  ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot inclination error (row 2, column 1)
  ax_inc = plt.subplot2grid((6, 3), (2, 1), sharex=ax_pos)
  if coe_ref.inc is not None and coe_comp.inc is not None:
    inc_error = (coe_comp.inc - coe_ref.inc) * CONVERTER.DEG_PER_RAD
    ax_inc.plot(time, inc_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and coe_ref.inc is not None and coe_smoother.inc is not None:
    inc_error_smoother = (coe_smoother.inc - coe_ref.inc) * CONVERTER.DEG_PER_RAD
    ax_inc.plot(time, inc_error_smoother, 'b--', linewidth=1.5)
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
    ax_raan.plot(time, raan_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and coe_ref.raan is not None and coe_smoother.raan is not None:
    raan_error_smoother_rad = np.arctan2(np.sin(coe_smoother.raan - coe_ref.raan), 
                   np.cos(coe_smoother.raan - coe_ref.raan))
    raan_error_smoother = raan_error_smoother_rad * CONVERTER.DEG_PER_RAD
    ax_raan.plot(time, raan_error_smoother, 'b--', linewidth=1.5)
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
    ax_aop.plot(time, aop_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and coe_ref.aop is not None and coe_smoother.aop is not None:
    aop_error_smoother_rad = np.arctan2(np.sin(coe_smoother.aop - coe_ref.aop), 
                   np.cos(coe_smoother.aop - coe_ref.aop))
    aop_error_smoother = aop_error_smoother_rad * CONVERTER.DEG_PER_RAD
    ax_aop.plot(time, aop_error_smoother, 'b--', linewidth=1.5)
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
    ax_ta.plot(time, ta_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and coe_ref.ta is not None and coe_smoother.ta is not None:
    ta_error_smoother_rad = np.arctan2(np.sin(coe_smoother.ta - coe_ref.ta), 
                   np.cos(coe_smoother.ta - coe_ref.ta))
    ta_error_smoother = ta_error_smoother_rad * CONVERTER.DEG_PER_RAD
    ax_ta.plot(time, ta_error_smoother, 'b--', linewidth=1.5)
  ax_ta.set_xlabel('Time\n[s]')
  ax_ta.set_ylabel('TA Error\n[deg]')
  ax_ta.grid(True)
  ax_ta.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # RIGHT COLUMN: Modified Equinoctial Elements Errors
  # Plot p error vs time (row 0, column 2)
  ax_p = plt.subplot2grid((6, 3), (0, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.p is not None and mee_comp.p is not None:
    p_error = mee_comp.p - mee_ref.p
    ax_p.plot(time, p_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0, label='Filter' if has_smoother else None)
  if has_smoother and mee_ref is not None and mee_smoother is not None and mee_ref.p is not None and mee_smoother.p is not None:
    p_error_smoother = mee_smoother.p - mee_ref.p
    ax_p.plot(time, p_error_smoother, 'b--', linewidth=1.5, label='Smoother')
    ax_p.legend(fontsize=7)
  ax_p.tick_params(labelbottom=False)
  ax_p.set_ylabel('p Error\n[m]')
  ax_p.grid(True)
  ax_p.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot f error vs time (row 1, column 2)
  ax_f = plt.subplot2grid((6, 3), (1, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.f is not None and mee_comp.f is not None:
    f_error = mee_comp.f - mee_ref.f
    ax_f.plot(time, f_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and mee_ref is not None and mee_smoother is not None and mee_ref.f is not None and mee_smoother.f is not None:
    f_error_smoother = mee_smoother.f - mee_ref.f
    ax_f.plot(time, f_error_smoother, 'b--', linewidth=1.5)
  ax_f.tick_params(labelbottom=False)
  ax_f.set_ylabel('f Error\n[-]')
  ax_f.grid(True)
  ax_f.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot g error vs time (row 2, column 2)
  ax_g = plt.subplot2grid((6, 3), (2, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.g is not None and mee_comp.g is not None:
    g_error = mee_comp.g - mee_ref.g
    ax_g.plot(time, g_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and mee_ref is not None and mee_smoother is not None and mee_ref.g is not None and mee_smoother.g is not None:
    g_error_smoother = mee_smoother.g - mee_ref.g
    ax_g.plot(time, g_error_smoother, 'b--', linewidth=1.5)
  ax_g.tick_params(labelbottom=False)
  ax_g.set_ylabel('g Error\n[-]')
  ax_g.grid(True)
  ax_g.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot h error vs time (row 3, column 2)
  ax_h = plt.subplot2grid((6, 3), (3, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.h is not None and mee_comp.h is not None:
    h_error = mee_comp.h - mee_ref.h
    ax_h.plot(time, h_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and mee_ref is not None and mee_smoother is not None and mee_ref.h is not None and mee_smoother.h is not None:
    h_error_smoother = mee_smoother.h - mee_ref.h
    ax_h.plot(time, h_error_smoother, 'b--', linewidth=1.5)
  ax_h.tick_params(labelbottom=False)
  ax_h.set_ylabel('h Error\n[-]')
  ax_h.grid(True)
  ax_h.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

  # Plot k error vs time (row 4, column 2)
  ax_k = plt.subplot2grid((6, 3), (4, 2), sharex=ax_pos)
  if mee_ref is not None and mee_comp is not None and mee_ref.k is not None and mee_comp.k is not None:
    k_error = mee_comp.k - mee_ref.k
    ax_k.plot(time, k_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and mee_ref is not None and mee_smoother is not None and mee_ref.k is not None and mee_smoother.k is not None:
    k_error_smoother = mee_smoother.k - mee_ref.k
    ax_k.plot(time, k_error_smoother, 'b--', linewidth=1.5)
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
    ax_L.plot(time, L_error, 'b-', linewidth=1.5, alpha=0.7 if has_smoother else 1.0)
  if has_smoother and mee_ref is not None and mee_smoother is not None and mee_ref.L is not None and mee_smoother.L is not None:
    L_error_smoother_rad = np.arctan2(np.sin(mee_smoother.L - mee_ref.L), 
                   np.cos(mee_smoother.L - mee_ref.L))
    L_error_smoother = L_error_smoother_rad * CONVERTER.DEG_PER_RAD
    ax_L.plot(time, L_error_smoother, 'b--', linewidth=1.5)
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
