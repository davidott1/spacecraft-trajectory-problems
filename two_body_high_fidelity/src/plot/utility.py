import numpy as np

from typing import Any

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

