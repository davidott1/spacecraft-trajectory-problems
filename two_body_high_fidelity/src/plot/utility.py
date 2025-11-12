import numpy as np

def get_equal_limits(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    all_limits = np.array([x_limits, y_limits, z_limits])
    min_limit = np.min(all_limits[:, 0])
    max_limit = np.max(all_limits[:, 1])
    return min_limit, max_limit