import numpy as np
from typing import Any
from datetime import datetime, timedelta

def get_equal_limits(
  ax : Any,
) -> tuple[float, float]:
  """
  Get equal limits for a 3D plot to ensure aspect ratio is preserved.
  
  Input:
  ------
  ax : matplotlib.axes._subplots.Axes3DSubplot
    The 3D axes object.
    
  Output:
  -------
  tuple[float, float]
    The minimum and maximum limits for all axes.
  """
  x_limits   = ax.get_xlim3d()
  y_limits   = ax.get_ylim3d()
  z_limits   = ax.get_zlim3d()
  all_limits = np.array([x_limits, y_limits, z_limits])
  min_limit  = np.min(all_limits[:, 0])
  max_limit  = np.max(all_limits[:, 1])
  return min_limit, max_limit


def add_utc_time_axis(
  ax    : Any,
  epoch : datetime,
) -> None:
  """
  Add a secondary x-axis at the top showing UTC time.
  
  Input:
  ------
  ax : matplotlib.axes.Axes
    The axes object to add the secondary axis to.
  epoch : datetime
    The reference epoch corresponding to t=0.
  """
  ax_top = ax.twiny()
  ax_top.set_xlim(ax.get_xlim())
  
  ticks = ax.get_xticks()
  # Filter ticks to be within the current view limits
  x_min, x_max = ax.get_xlim()
  valid = ticks[(ticks >= x_min) & (ticks <= x_max)]
  
  if len(valid) < 2:
    valid = np.linspace(x_min, x_max, 7)
    
  labels = [(epoch + timedelta(seconds=float(t))).strftime('%m/%d %H:%M') for t in valid]
  
  ax_top.set_xticks(valid)
  ax_top.set_xticklabels(labels, rotation=45, ha='left', fontsize=8)
  ax_top.set_xlabel('UTC Time', fontsize=9)
  ax_top.xaxis.set_label_position('top')
  ax_top.xaxis.tick_top()