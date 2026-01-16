"""
Plot module for trajectory visualization.

This module provides functions for visualizing spacecraft trajectories
in various coordinate frames and formats.

Modules:
--------
- plot_3d: 3D trajectory visualizations (inertial, body-fixed, heliocentric)
- plot_timeseries: Time series plots of state vectors and orbital elements
- plot_groundtrack: Ground track visualizations on 2D maps and 3D globes
- plot_skyplot: Skyplot (azimuth/elevation) visualizations from ground stations
- plot_generator: High-level functions for generating and saving multiple plots
- utility: Shared utility functions for plotting
"""

# Re-export main functions for convenience
from src.plot.plot_3d import (
    plot_3d_trajectories,
    plot_3d_trajectories_body_fixed,
    plot_3d_trajectory_sun_centered,
    plot_3d_error,
)

from src.plot.plot_timeseries import (
    plot_time_series,
    plot_time_series_error,
)

from src.plot.plot_groundtrack import (
    plot_ground_track,
)

from src.plot.plot_skyplot import (
    plot_skyplot,
    plot_pass_timeseries,
)

from src.plot.plot_generator import (
    generate_plots,
    generate_error_plots,
    generate_3d_and_time_series_plots,
)

__all__ = [
    # 3D plots
    'plot_3d_trajectories',
    'plot_3d_trajectories_body_fixed',
    'plot_3d_trajectory_sun_centered',
    'plot_3d_error',
    # Time series
    'plot_time_series',
    'plot_time_series_error',
    # Ground track
    'plot_ground_track',
    # Skyplot
    'plot_skyplot',
    'plot_pass_timeseries',
    # Generators
    'generate_plots',
    'generate_error_plots',
    'generate_3d_and_time_series_plots',
]
