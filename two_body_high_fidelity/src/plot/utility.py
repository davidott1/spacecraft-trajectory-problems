import numpy as np

from datetime import datetime, timedelta
from typing   import Any


def get_equal_limits(
  ax              : Any,
  buffer_fraction : float = 0.0,
) -> tuple[float, float]:
  """
  Get equal limits for a 3D plot to ensure aspect ratio is preserved.
  
  Input:
  ------
    ax : matplotlib.axes._subplots.Axes3DSubplot
      The 3D axes object.
    buffer_fraction : float
      Fraction of the range to add as buffer on each side (default 0.0).
      E.g., 0.25 adds 25% buffer to each side.
    
  Output:
  -------
    min_limit : float
      The minimum limit for all axes.
    max_limit : float
      The maximum limit for all axes.
  """
  x_limits   = ax.get_xlim3d()
  y_limits   = ax.get_ylim3d()
  z_limits   = ax.get_zlim3d()
  all_limits = np.array([x_limits, y_limits, z_limits])
  min_limit  = np.min(all_limits[:, 0])
  max_limit  = np.max(all_limits[:, 1])
  
  # Apply buffer if specified
  if buffer_fraction > 0:
    range_limit = max_limit - min_limit
    buffer      = buffer_fraction * range_limit
    min_limit  -= buffer
    max_limit  += buffer
  
  return min_limit, max_limit


def add_utc_time_axis(
  ax       : Any,
  epoch    : datetime,
  max_time : float,
) -> None:
  """
  Add a secondary x-axis with UTC time labels.
  
  Input:
  ------
    ax : matplotlib.axes.Axes
      The axes object to add the secondary axis to.
    epoch : datetime
      The epoch time corresponding to t=0.
    max_time : float
      The maximum time value on the primary x-axis.
      
  Output:
  -------
    None
  """
  ax2 = ax.twiny()
  ax2.set_xlim(ax.get_xlim())
  
  tick_positions_sec = ax.get_xticks()
  valid_ticks = tick_positions_sec[(tick_positions_sec >= 0) & (tick_positions_sec <= max_time)]
  
  if len(valid_ticks) < 2:
    valid_ticks = np.linspace(0, max_time, 7)
    
  utc_times = [epoch + timedelta(seconds=float(t)) for t in valid_ticks]
  time_labels = [t.strftime('%m/%d %H:%M') for t in utc_times]
  
  ax2.set_xticks(valid_ticks)
  ax2.set_xticklabels(time_labels, rotation=45, ha='left', fontsize=8)
  ax2.set_xlabel('UTC Time', fontsize=9)
  ax2.xaxis.set_label_position('top')
  ax2.xaxis.tick_top()


def add_stats(
  ax    : Any,
  data  : np.ndarray,
  label : str,
) -> None:
  """
  Add statistics text box to a plot axis.
  
  Input:
  ------
    ax : matplotlib.axes.Axes
      The axes object to add statistics to.
    data : np.ndarray
      Data array to compute statistics from.
    label : str
      Label prefix for statistics text.
      
  Output:
  -------
    None
  """
  stats = (
    f'{label} Mean : {np.mean(data):.3f}\n'
    f'{label} RMS  : {np.sqrt(np.mean(data**2)):.3f}\n'
    f'{label} Max  : {np.max(np.abs(data)):.3f}'
  )
  ax.text(
    0.02, 0.95,
    stats,
    transform         = ax.transAxes,
    fontsize          = 9,
    verticalalignment = 'top',
    bbox              = dict(boxstyle='round', facecolor='white', alpha=0.8),
  )