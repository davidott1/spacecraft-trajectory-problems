"""
Skyplot plotting functions.

This module contains functions for plotting skyplots (polar plots of azimuth
vs elevation) from ground tracking stations.
"""
import warnings

import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from scipy.stats         import norm

from datetime          import datetime, timedelta
from typing            import Optional
from matplotlib.figure import Figure

from src.model.constants                   import CONVERTER
from src.orbit_determination.topocentric   import compute_topocentric_coordinates_with_rates
from src.schemas.propagation               import PropagationResult
from src.schemas.state                     import TrackerStation
from src.schemas.measurement               import SimulatedMeasurements


def _plot_hist_and_gaussian(ax, data, limits=None):
  """
  Helper for plotting histogram + gaussian on an axis.
  """
  if data is None or len(data) < 2 or not np.isfinite(data).all():
    return
  
  # Horizontal Histogram
  ax.hist(data, bins=30, orientation='horizontal', density=True, color='b', alpha=0.3)
  
  # Gaussian Fit
  try:
    mu, std = norm.fit(data)
    if std > 0:
      if limits:
        ymin, ymax = limits
      else:
        ymin, ymax = min(data), max(data)
        
      y_pts = np.linspace(ymin, ymax, 200)
      p = norm.pdf(y_pts, mu, std)
      ax.plot(p, y_pts, 'b-', linewidth=1.5, alpha=0.8)
  except Exception:
    pass

  if limits:
    ax.set_ylim(limits)
    
  ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
  ax.set_yticklabels([])
  ax.set_xticks([])
  ax.grid(True, alpha=0.3)


def plot_skyplot(
  result       : PropagationResult,
  tracker      : TrackerStation,
  epoch_dt_utc : Optional[datetime] = None,
  title_text   : str = "Skyplot",
  measurements : Optional[SimulatedMeasurements] = None,
) -> Figure:
  """
  Plot a skyplot (polar plot of azimuth vs elevation) from a ground station.

  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_delta_time'.
    tracker : TrackerStation
      Ground tracking station with latitude, longitude, altitude.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.
    title_text : str
      Base title for the plot.
    measurements : SimulatedMeasurements, optional
      Simulated measurements with truth data.
      If provided, uses truth from measurements.
      If None, computes topocentric coordinates from propagation result.

  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the skyplot and time-series plots.
  """
  fig = plt.figure(figsize=(30, 10))

  # If measurements are provided, use them; otherwise compute topocentric coordinates
  if measurements is not None:
    # Use truth from SimulatedMeasurements
    topo_truth = measurements.truth
  else:
    # Compute topocentric coordinates with rates (no measurements)
    topo_truth = compute_topocentric_coordinates_with_rates(result, tracker, epoch_dt_utc)
  
  # Convert truth to degrees for display
  az_deg_truth  = topo_truth.azimuth   * CONVERTER.DEG_PER_RAD
  el_deg_truth  = topo_truth.elevation * CONVERTER.DEG_PER_RAD

  # Get time from appropriate source
  if hasattr(topo_truth, 'delta_time_epoch'):
    delta_time_epoch = topo_truth.delta_time_epoch  # From TopocentricState (SimulatedMeasurements)
  else:
    delta_time_epoch = result.time_grid.deltas  # From PropagationResult (compute_topocentric_coordinates_with_rates)

  range_m_truth = topo_truth.range

  # Get rates from truth (convert angular rates to deg/s)
  az_dot_deg_truth  = topo_truth.azimuth_dot   * CONVERTER.DEG_PER_RAD if topo_truth.azimuth_dot is not None else None
  el_dot_deg_truth  = topo_truth.elevation_dot * CONVERTER.DEG_PER_RAD if topo_truth.elevation_dot is not None else None
  rng_dot_truth     = topo_truth.range_dot  # m/s

  # Normalize truth azimuth to -180° to +180° range
  az_deg_truth = ((az_deg_truth + 180.0) % 360.0) - 180.0

  # For polar plot: radius = 90 - elevation (so zenith is at center)
  # theta = azimuth (convert negative angles to positive for polar plotting: 0 to 2π)
  radius_truth = 90.0 - el_deg_truth
  theta_truth  = np.where(az_deg_truth >= 0, az_deg_truth, az_deg_truth + 360.0) * CONVERTER.RAD_PER_DEG

  # Use truth for visibility and constraint checking
  az_deg = az_deg_truth
  el_deg = el_deg_truth
  range_m = range_m_truth
  radius = radius_truth
  theta = theta_truth

  # Create grid: polar skyplot on left, three time-series plots in middle, three rate plots on right
  ax = fig.add_subplot(1, 3, 1, projection='polar')
  
  # Configure polar plot for skyplot convention
  ax.set_theta_zero_location('N')  # North at top
  ax.set_theta_direction(-1)       # Clockwise
  ax.set_rlim(0, 90)               # 0 to 90 degrees from zenith
  ax.set_rticks([0, 15, 30, 45, 60, 75, 90])
  ax.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'])  # Elevation labels

  # Add gray shaded regions for elevation and azimuth constraints
  if tracker.performance:

    # Elevation constraints (circular boundaries)
    if tracker.performance.constraints.elevation:
      el_min_deg = tracker.performance.constraints.elevation.min * CONVERTER.DEG_PER_RAD
      el_max_deg = tracker.performance.constraints.elevation.max * CONVERTER.DEG_PER_RAD

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
    if tracker.performance.constraints.azimuth:
      # Azimuth values are already normalized to -180° to +180° in the loader
      az_min_deg = tracker.performance.constraints.azimuth.min * CONVERTER.DEG_PER_RAD
      az_max_deg = tracker.performance.constraints.azimuth.max * CONVERTER.DEG_PER_RAD

      # Check if this covers the full circle (e.g., -180 to 180 or 0 to 360)
      az_range_deg = az_max_deg - az_min_deg
      is_full_circle = abs(az_range_deg - 360.0) < 1e-6

      # Only add shading if not covering full circle
      if not is_full_circle:
        # Convert azimuth to theta (polar angle in radians)
        # Note: polar plots use 0-2π, but our azimuth is in -180° to +180°
        # Convert negative angles to positive for polar plotting
        az_min_rad = (az_min_deg if az_min_deg >= 0 else az_min_deg + 360.0) * CONVERTER.RAD_PER_DEG
        az_max_rad = (az_max_deg if az_max_deg >= 0 else az_max_deg + 360.0) * CONVERTER.RAD_PER_DEG

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
      # If is_full_circle, don't shade anything (all azimuths valid)

  # Split track where satellite goes below horizon
  visible_mask = el_deg >= 0

  # Check which points satisfy tracker performance constraints
  constraint_valid_mask = np.ones(len(el_deg), dtype=bool)

  if tracker.performance:
    # Check elevation constraints
    if tracker.performance.constraints.elevation:
      el_min_deg = tracker.performance.constraints.elevation.min * CONVERTER.DEG_PER_RAD
      el_max_deg = tracker.performance.constraints.elevation.max * CONVERTER.DEG_PER_RAD
      constraint_valid_mask &= (el_deg >= el_min_deg) & (el_deg <= el_max_deg)

    # Check azimuth constraints
    if tracker.performance.constraints.azimuth:
      az_min_deg = tracker.performance.constraints.azimuth.min * CONVERTER.DEG_PER_RAD
      az_max_deg = tracker.performance.constraints.azimuth.max * CONVERTER.DEG_PER_RAD

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
  range_m = topo_truth.range

  # Check range constraints
  if tracker.performance and tracker.performance.constraints.range:
    range_min_m = tracker.performance.constraints.range.min
    range_max_m = tracker.performance.constraints.range.max
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
    marker_sizes_truth = marker_size_max - (range_m - range_min) / (range_max - range_min) * (marker_size_max - marker_size_min)
  else:
    marker_sizes_truth = np.full_like(range_m, (marker_size_min + marker_size_max) / 2)
  
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
    # Truth data
    seg_theta_truth          = theta_truth[seg_start:seg_end]
    seg_radius_truth         = radius_truth[seg_start:seg_end]
    seg_time                 = delta_time_epoch[seg_start:seg_end]
    seg_marker_sizes_truth   = marker_sizes_truth[seg_start:seg_end]
    seg_range                = range_m[seg_start:seg_end]
    seg_constraint_valid     = constraint_valid_mask[seg_start:seg_end]
    seg_el_deg               = el_deg[seg_start:seg_end]
    seg_az_deg               = az_deg[seg_start:seg_end]

    # Plot trajectory
    if len(seg_time) > 0:

      # Plot TRUTH line segments (blue)
      # - Gray solid line between gray (not valid) points
      # - Gray dashed line between gray and blue (transition) points
      # - Blue solid line between blue (valid) points
      for i in range(len(seg_theta_truth) - 1):
        if seg_constraint_valid[i] and seg_constraint_valid[i+1]:
          # Both points valid (blue) - blue solid line
          ax.plot(seg_theta_truth[i:i+2], seg_radius_truth[i:i+2], 'b-', linewidth=2.0, alpha=0.8)
        elif not seg_constraint_valid[i] and not seg_constraint_valid[i+1]:
          # Both points not valid (gray) - gray solid line
          ax.plot(seg_theta_truth[i:i+2], seg_radius_truth[i:i+2], color='gray', linestyle='-', linewidth=1.5, alpha=0.8)
        else:
          # Transition between gray and blue - gray dashed line
          ax.plot(seg_theta_truth[i:i+2], seg_radius_truth[i:i+2], color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

      # Plot visible truth points with gray (do not connect)
      ax.scatter(seg_theta_truth, seg_radius_truth, c='gray', s=seg_marker_sizes_truth, alpha=0.6)

      # Plot valid portions with blue markers (truth)
      if np.any(seg_constraint_valid):
        # Plot blue scatter points for valid portions
        valid_indices = np.where(seg_constraint_valid)[0]
        if len(valid_indices) > 0:
          ax.scatter(seg_theta_truth[valid_indices], seg_radius_truth[valid_indices],
                    c='blue', s=seg_marker_sizes_truth[valid_indices], alpha=1.0, label='Truth' if seg_start == segment_starts[0] else '')
      
      # Add entry marker if this is a true entry (satellite rose above horizon during propagation)
      # A true entry means there was a point before this segment that was below horizon
      is_true_entry = seg_start > 0
      if is_true_entry:
        ax.scatter([seg_theta_truth[0]], [seg_radius_truth[0]], s=120, marker='s', facecolors='none',
                  edgecolors='black', linewidths=2, zorder=10)
        if epoch_dt_utc is not None:
          entry_dt = epoch_dt_utc + timedelta(seconds=seg_time[0])
          entry_label = entry_dt.strftime('%Y-%m-%d %H:%M:%S')
          ax.annotate(f'Entry\n{entry_label}', (seg_theta_truth[0], seg_radius_truth[0]),
                     textcoords='offset points', xytext=(8, 8), fontsize=8,
                     color='black', fontweight='normal',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8))

      # Add exit marker if this is a true exit (satellite set below horizon during propagation)
      # A true exit means there's a point after this segment that's below horizon
      is_true_exit = seg_end < len(visible_mask)
      if is_true_exit:
        ax.scatter([seg_theta_truth[-1]], [seg_radius_truth[-1]], s=120, marker='s', facecolors='none',
                  edgecolors='black', linewidths=2, zorder=10)
        if epoch_dt_utc is not None:
          exit_dt = epoch_dt_utc + timedelta(seconds=seg_time[-1])
          exit_label = exit_dt.strftime('%Y-%m-%d %H:%M:%S')
          ax.annotate(f'Exit\n{exit_label}', (seg_theta_truth[-1], seg_radius_truth[-1]),
                     textcoords='offset points', xytext=(8, -12), fontsize=8,
                     color='black', fontweight='normal',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8))
  
  # Add initial marker at the first point of the entire propagation (if visible)
  if visible_mask[0]:
    ax.scatter([theta[0]], [radius[0]], s=120, marker='s', facecolors='none', 
              edgecolors='black', linewidths=2, zorder=10)
    if epoch_dt_utc is not None:
      initial_dt = epoch_dt_utc + timedelta(seconds=delta_time_epoch[0])
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
      final_dt = epoch_dt_utc + timedelta(seconds=delta_time_epoch[-1])
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
      tca_dt = epoch_dt_utc + timedelta(seconds=delta_time_epoch[max_el_idx])
      tca_label = tca_dt.strftime('%Y-%m-%d %H:%M:%S')
      ax.annotate(f'Max Elevation\n{tca_label}', (theta[max_el_idx], radius[max_el_idx]),
                 textcoords='offset points', xytext=(12, 13), fontsize=8,
                 color='black', fontweight='normal',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))
  
  # Find and mark minimum range (closest approach) with UTC time
  visible_indices = np.where(visible_mask)[0]
  min_range_km = None  # Initialize to None for later use in info text
  if len(visible_indices) > 0:
    visible_ranges = range_m[visible_indices]
    min_range_visible_idx = np.argmin(visible_ranges)
    min_range_idx = visible_indices[min_range_visible_idx]
    min_range_km = range_m[min_range_idx] / 1000.0
    ax.scatter([theta[min_range_idx]], [radius[min_range_idx]], s=marker_size_max * 3.0, marker='s',
              facecolors='none', edgecolors='black', linewidths=2, zorder=11,
              label=f'Min Range ({min_range_km:.0f} km)')
    if epoch_dt_utc is not None:
      min_range_dt = epoch_dt_utc + timedelta(seconds=delta_time_epoch[min_range_idx])
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
    end_time_dt_utc    = epoch_dt_utc + timedelta(seconds=delta_time_epoch[-1])
    end_time_iso_utc   = end_time_dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
    info_text += f"\nInitial: {start_time_iso_utc}  |  Final: {end_time_iso_utc}"
  
  # Add visibility statistics
  n_visible = np.sum(visible_mask)
  n_total = len(visible_mask)
  visibility_pct = 100.0 * n_visible / n_total if n_total > 0 else 0.0
  max_elevation = np.max(el_deg)
  
  # Compute time step between points
  if len(delta_time_epoch) > 1:
    dt_step = delta_time_epoch[1] - delta_time_epoch[0]
  else:
    dt_step = 0.0
  
  # Build visibility line with Min Range if available
  visibility_line = f"\nVisibility: {visibility_pct:.1f}%  |  Max Elevation: {max_elevation:.2f}°"
  if min_range_km is not None:
    visibility_line += f"  |  Min Range: {min_range_km:.0f} km"
  visibility_line += f"  |  Δt: {dt_step:.1f} s"
  info_text += visibility_line
  
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
  # Time-series plots in the middle column
  # ============================================

  # Convert time to hours for better readability
  time_hrs = delta_time_epoch / 3600.0

  # Create three subplots in the middle column (stacked vertically)
  ax_range = fig.add_subplot(3, 3, 2)
  ax_az    = fig.add_subplot(3, 3, 5)
  ax_el    = fig.add_subplot(3, 3, 8)

  # Plot Range vs Time
  range_km = topo_truth.range / 1000.0

  # Thin black line for entire solution (not in legend)
  ax_range.plot(time_hrs, range_km, 'k-', linewidth=0.5, alpha=0.8)
  ax_range.set_ylabel('Range [km]', fontsize=11)
  ax_range.yaxis.set_label_coords(-0.06, 0.5)
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
  if tracker.performance and tracker.performance.constraints.range:
    range_min_km = tracker.performance.constraints.range.min / 1000.0
    range_max_km = tracker.performance.constraints.range.max / 1000.0
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
  ax_az.yaxis.set_label_coords(-0.06, 0.5)
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
  if tracker.performance and tracker.performance.constraints.azimuth:
    az_min_constraint = tracker.performance.constraints.azimuth.min * CONVERTER.DEG_PER_RAD
    az_max_constraint = tracker.performance.constraints.azimuth.max * CONVERTER.DEG_PER_RAD
    ax_az.axhline(y=az_min_constraint, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Min Az ({az_min_constraint:.0f}°)')
    ax_az.axhline(y=az_max_constraint, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Max Az ({az_max_constraint:.0f}°)')

  ax_az.legend(loc='best', fontsize=9)
  # Hide x-axis tick labels for azimuth plot
  ax_az.set_xticklabels([])

  # Plot Elevation vs Time
  # Convert delta_time_epoch to UTC datetime for x-axis
  if epoch_dt_utc is not None:
    time_utc = [epoch_dt_utc + timedelta(seconds=float(t)) for t in delta_time_epoch]

  # Thin black line for entire solution (not in legend)
  if epoch_dt_utc is not None:
    ax_el.plot(time_utc, el_deg, 'k-', linewidth=0.5, alpha=0.8)
  else:
    ax_el.plot(time_hrs, el_deg, 'k-', linewidth=0.5, alpha=0.8)

  ax_el.set_ylabel('Elevation [deg]', fontsize=11)
  ax_el.yaxis.set_label_coords(-0.06, 0.5)
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
  if tracker.performance and tracker.performance.constraints.elevation:
    el_min_constraint = tracker.performance.constraints.elevation.min * CONVERTER.DEG_PER_RAD
    el_max_constraint = tracker.performance.constraints.elevation.max * CONVERTER.DEG_PER_RAD
    if el_min_constraint > 0:
      ax_el.axhline(y=el_min_constraint, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Min El ({el_min_constraint:.0f}°)')
    if el_max_constraint < 90:
      ax_el.axhline(y=el_max_constraint, color='k', linestyle='--', linewidth=1.0, alpha=0.5, label=f'Max El ({el_max_constraint:.0f}°)')

  # Rotate x-axis labels for better readability
  if epoch_dt_utc is not None:
    ax_el.tick_params(axis='x', rotation=45)
    # Format x-axis to show time nicely
    ax_el.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

  ax_el.legend(loc='best', fontsize=9)

  # ============================================
  # Rate plots in the right column
  # ============================================

  # Create three subplots in the right column (stacked vertically)
  ax_rng_dot = fig.add_subplot(3, 3, 3)
  ax_az_dot  = fig.add_subplot(3, 3, 6)
  ax_el_dot  = fig.add_subplot(3, 3, 9)

  # Plot Range Rate vs Time
  if rng_dot_truth is not None:
    rng_dot_km = rng_dot_truth / 1000.0  # Convert to km/s

    # Thin black line for entire solution
    ax_rng_dot.plot(time_hrs, rng_dot_km, 'k-', linewidth=0.5, alpha=0.8)
    ax_rng_dot.set_ylabel('Range Rate [km/s]', fontsize=11)
    ax_rng_dot.yaxis.set_label_coords(-0.06, 0.5)
    ax_rng_dot.grid(True, alpha=0.3)
    ax_rng_dot.set_title('Truth Measurement Rates vs Time', fontsize=12)

    # Plot gray lines and markers for each visible segment
    gray_plotted = False
    for seg_start, seg_end in zip(visible_segment_starts, visible_segment_ends):
      label = 'Above Horizon' if not gray_plotted else None
      ax_rng_dot.plot(time_hrs[seg_start:seg_end], rng_dot_km[seg_start:seg_end],
                      color='gray', linewidth=2.5, alpha=0.6, marker='o', markersize=4, label=label)
      gray_plotted = True

    # Plot blue lines for each valid segment
    blue_plotted = False
    for seg_start, seg_end in zip(valid_segment_starts, valid_segment_ends):
      label = 'Trackable' if not blue_plotted else None
      ax_rng_dot.plot(time_hrs[seg_start:seg_end], rng_dot_km[seg_start:seg_end],
                      'b-', linewidth=3.5, alpha=0.8, label=label)
      blue_plotted = True

    if gray_plotted or blue_plotted:
      ax_rng_dot.legend(loc='best', fontsize=9)
  ax_rng_dot.set_xticklabels([])

  # Plot Azimuth Rate vs Time
  if az_dot_deg_truth is not None:
    # Thin black line for entire solution
    ax_az_dot.plot(time_hrs, az_dot_deg_truth, 'k-', linewidth=0.5, alpha=0.8)
    ax_az_dot.set_ylabel('Azimuth Rate [deg/s]', fontsize=11)
    ax_az_dot.yaxis.set_label_coords(-0.06, 0.5)
    ax_az_dot.grid(True, alpha=0.3)

    # Plot gray lines and markers for each visible segment
    gray_plotted = False
    for seg_start, seg_end in zip(visible_segment_starts, visible_segment_ends):
      label = 'Above Horizon' if not gray_plotted else None
      ax_az_dot.plot(time_hrs[seg_start:seg_end], az_dot_deg_truth[seg_start:seg_end],
                     color='gray', linewidth=2.5, alpha=0.6, marker='o', markersize=4, label=label)
      gray_plotted = True

    # Plot blue lines for each valid segment
    blue_plotted = False
    for seg_start, seg_end in zip(valid_segment_starts, valid_segment_ends):
      label = 'Trackable' if not blue_plotted else None
      ax_az_dot.plot(time_hrs[seg_start:seg_end], az_dot_deg_truth[seg_start:seg_end],
                     'b-', linewidth=3.5, alpha=0.8, label=label)
      blue_plotted = True

    if gray_plotted or blue_plotted:
      ax_az_dot.legend(loc='best', fontsize=9)
  ax_az_dot.set_xticklabels([])

  # Plot Elevation Rate vs Time
  if el_dot_deg_truth is not None:
    # Thin black line for entire solution
    if epoch_dt_utc is not None:
      ax_el_dot.plot(time_utc, el_dot_deg_truth, 'k-', linewidth=0.5, alpha=0.8)
    else:
      ax_el_dot.plot(time_hrs, el_dot_deg_truth, 'k-', linewidth=0.5, alpha=0.8)

    ax_el_dot.set_ylabel('Elevation Rate [deg/s]', fontsize=11)
    ax_el_dot.yaxis.set_label_coords(-0.06, 0.5)
    ax_el_dot.set_xlabel('UTC Time', fontsize=11)
    ax_el_dot.grid(True, alpha=0.3)

    # Plot gray lines and markers for each visible segment
    gray_plotted = False
    for seg_start, seg_end in zip(visible_segment_starts, visible_segment_ends):
      label = 'Above Horizon' if not gray_plotted else None
      if epoch_dt_utc is not None:
        ax_el_dot.plot(time_utc[seg_start:seg_end], el_dot_deg_truth[seg_start:seg_end],
                       color='gray', linewidth=2.5, alpha=0.6, marker='o', markersize=4, label=label)
      else:
        ax_el_dot.plot(time_hrs[seg_start:seg_end], el_dot_deg_truth[seg_start:seg_end],
                       color='gray', linewidth=2.5, alpha=0.6, marker='o', markersize=4, label=label)
      gray_plotted = True

    # Plot blue lines for each valid segment
    blue_plotted = False
    for seg_start, seg_end in zip(valid_segment_starts, valid_segment_ends):
      label = 'Trackable' if not blue_plotted else None
      if epoch_dt_utc is not None:
        ax_el_dot.plot(time_utc[seg_start:seg_end], el_dot_deg_truth[seg_start:seg_end],
                       'b-', linewidth=3.5, alpha=0.8, label=label)
      else:
        ax_el_dot.plot(time_hrs[seg_start:seg_end], el_dot_deg_truth[seg_start:seg_end],
                       'b-', linewidth=3.5, alpha=0.8, label=label)
      blue_plotted = True

    # Rotate x-axis labels for better readability
    if epoch_dt_utc is not None:
      ax_el_dot.tick_params(axis='x', rotation=45)
      ax_el_dot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    if gray_plotted or blue_plotted:
      ax_el_dot.legend(loc='best', fontsize=9)

  # Adjust layout
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    plt.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))

  return fig


def plot_pass_timeseries(
  result       : PropagationResult,
  tracker      : TrackerStation,
  epoch_dt_utc : Optional[datetime] = None,
  title_text   : str = "Pass Time Series",
) -> Optional[Figure]:
  """
  Plot magnified time-series of trackable passes showing range, azimuth, elevation and their rates.
  
  Creates a grid with 3 rows (range, azimuth, elevation) and (num_passes × 2) columns.
  Layout groups measurements and rates:
    Row 0: Range [Pass 1, 2, 3...] | Range Rate [Pass 1, 2, 3...]
    Row 1: Azimuth [Pass 1, 2, 3...] | Azimuth Rate [Pass 1, 2, 3...]
    Row 2: Elevation [Pass 1, 2, 3...] | Elevation Rate [Pass 1, 2, 3...]

  Input:
  ------
    result : PropagationResult
      Propagation result containing 'state' (6xN array) and 'plot_delta_time'.
    tracker : TrackerStation
      Ground tracking station with latitude, longitude, altitude.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time conversion to ET.
    title_text : str
      Base title for the plot.

  Output:
  -------
    fig : matplotlib.figure.Figure | None
      Figure object containing the pass time-series plots, or None if no trackable passes.
  """
  # Compute topocentric coordinates with rates
  topo = compute_topocentric_coordinates_with_rates(result, tracker, epoch_dt_utc)
  
  # Convert to degrees for display
  az_deg  = topo.azimuth   * CONVERTER.DEG_PER_RAD
  el_deg  = topo.elevation * CONVERTER.DEG_PER_RAD
  delta_time_epoch  = result.time_grid.deltas
  
  # Get rates (convert angular rates to deg/s)
  az_dot_deg  = topo.azimuth_dot   * CONVERTER.DEG_PER_RAD if topo.azimuth_dot is not None else None
  el_dot_deg  = topo.elevation_dot * CONVERTER.DEG_PER_RAD if topo.elevation_dot is not None else None
  rng_dot     = topo.range_dot  # m/s

  # Normalize azimuth to -180° to +180° range
  az_deg = ((az_deg + 180.0) % 360.0) - 180.0

  # Get range values
  range_m  = topo.range
  range_km = range_m / 1000.0

  # Convert rates
  rng_dot_km = rng_dot / 1000.0 if rng_dot is not None else None

  # Build constraint valid mask
  constraint_valid_mask = np.ones(len(el_deg), dtype=bool)

  if tracker.performance:
    # Check elevation constraints
    if tracker.performance.constraints.elevation:
      el_min_deg = tracker.performance.constraints.elevation.min * CONVERTER.DEG_PER_RAD
      el_max_deg = tracker.performance.constraints.elevation.max * CONVERTER.DEG_PER_RAD
      constraint_valid_mask &= (el_deg >= el_min_deg) & (el_deg <= el_max_deg)

    # Check azimuth constraints
    if tracker.performance.constraints.azimuth:
      az_min_deg = tracker.performance.constraints.azimuth.min * CONVERTER.DEG_PER_RAD
      az_max_deg = tracker.performance.constraints.azimuth.max * CONVERTER.DEG_PER_RAD

      az_range_deg = az_max_deg - az_min_deg
      if abs(az_range_deg - 360.0) < 1e-6:
        pass  # Full circle - all valid
      elif az_max_deg < az_min_deg:
        constraint_valid_mask &= (az_deg >= az_min_deg) | (az_deg <= az_max_deg)
      else:
        constraint_valid_mask &= (az_deg >= az_min_deg) & (az_deg <= az_max_deg)

    # Check range constraints
    if tracker.performance.constraints.range:
      range_min_m = tracker.performance.constraints.range.min
      range_max_m = tracker.performance.constraints.range.max
      constraint_valid_mask &= (range_m >= range_min_m) & (range_m <= range_max_m)

  # Find trackable (valid) segments/passes
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

  num_passes = len(valid_segment_starts)
  
  # If no trackable passes, return None
  if num_passes == 0:
    return None

  # Compute y-axis limits across all passes for consistent scaling
  all_range_km = []
  all_az_deg = []
  all_el_deg = []
  all_rng_dot_km = []
  all_az_dot_deg = []
  all_el_dot_deg = []

  for seg_start, seg_end in zip(valid_segment_starts, valid_segment_ends):
    all_range_km.extend(range_km[seg_start:seg_end])
    all_az_deg.extend(az_deg[seg_start:seg_end])
    all_el_deg.extend(el_deg[seg_start:seg_end])
    if rng_dot_km is not None:
      all_rng_dot_km.extend(rng_dot_km[seg_start:seg_end])
    if az_dot_deg is not None:
      all_az_dot_deg.extend(az_dot_deg[seg_start:seg_end])
    if el_dot_deg is not None:
      all_el_dot_deg.extend(el_dot_deg[seg_start:seg_end])

  # Compute limits with small padding
  def get_limits(data, pad_frac=0.05):
    if len(data) == 0:
      return (0, 1)
    dmin, dmax = min(data), max(data)
    pad = (dmax - dmin) * pad_frac if dmax != dmin else 0.1
    return (dmin - pad, dmax + pad)

  range_ylim     = get_limits(all_range_km)
  az_ylim        = get_limits(all_az_deg)
  el_ylim        = get_limits(all_el_deg)
  rng_dot_ylim   = get_limits(all_rng_dot_km)
  az_dot_ylim    = get_limits(all_az_dot_deg)
  el_dot_ylim    = get_limits(all_el_dot_deg)

  # Create figure: 6 rows × num_passes columns
  # Layout: Row 0=Range, Row 1=Range Rate, Row 2=Az, Row 3=Az Rate, Row 4=El, Row 5=El Rate
  num_rows = 6
  fig_width = max(3.5 * num_passes, 10)  # At least 3.5 inches per column, min 10
  fig_height = 12
  fig, axes = plt.subplots(num_rows, num_passes, figsize=(fig_width, fig_height), squeeze=False)

  # Convert time to UTC datetime if epoch provided
  if epoch_dt_utc is not None:
    time_utc = [epoch_dt_utc + timedelta(seconds=float(t)) for t in delta_time_epoch]

  # Y-label alignment coordinate
  ylabel_x = -0.15

  # Row configuration: (data_key, rate_data_key, ylabel, rate_ylabel, ylim, rate_ylim)
  row_configs = [
    ('range',     'rng_dot',  'Range [km]',     'Range Rate [km/s]',     range_ylim,   rng_dot_ylim),
    ('az',        'az_dot',   'Azimuth [deg]',  'Azimuth Rate [deg/s]',  az_ylim,      az_dot_ylim),
    ('el',        'el_dot',   'Elevation [deg]','Elevation Rate [deg/s]',el_ylim,      el_dot_ylim),
  ]

  # Plot each pass
  for pass_idx, (seg_start, seg_end) in enumerate(zip(valid_segment_starts, valid_segment_ends)):
    col = pass_idx

    # Extract segment data
    seg_time_s    = delta_time_epoch[seg_start:seg_end]
    seg_range_km  = range_km[seg_start:seg_end]
    seg_az_deg    = az_deg[seg_start:seg_end]
    seg_el_deg    = el_deg[seg_start:seg_end]
    
    seg_rng_dot_km = rng_dot_km[seg_start:seg_end] if rng_dot_km is not None else None
    seg_az_dot_deg = az_dot_deg[seg_start:seg_end] if az_dot_deg is not None else None
    seg_el_dot_deg = el_dot_deg[seg_start:seg_end] if el_dot_deg is not None else None

    # Time axis for this segment
    if epoch_dt_utc is not None:
      seg_time = time_utc[seg_start:seg_end]
    else:
      seg_time = seg_time_s / 3600.0  # hours

    # Pass label for title (only on row 0)
    pass_label = f"Pass {pass_idx + 1}"
    if epoch_dt_utc is not None and len(seg_time) > 0:
      pass_start_str = seg_time[0].strftime('%H:%M:%S')
      pass_end_str   = seg_time[-1].strftime('%H:%M:%S')
      pass_label = f"Pass {pass_idx + 1}\n{pass_start_str} - {pass_end_str}"

    # Data arrays for each measurement type
    seg_data = {
      'range': seg_range_km,
      'az': seg_az_deg,
      'el': seg_el_deg,
      'rng_dot': seg_rng_dot_km,
      'az_dot': seg_az_dot_deg,
      'el_dot': seg_el_dot_deg,
    }

    # Plot each row pair (measurement + rate)
    for meas_idx, (meas_key, rate_key, ylabel, rate_ylabel, ylim, rate_ylim) in enumerate(row_configs):
      row_meas = meas_idx * 2      # 0, 2, 4
      row_rate = meas_idx * 2 + 1  # 1, 3, 5

      # ---- Measurement row ----
      ax_meas = axes[row_meas, col]
      ax_meas.plot(seg_time, seg_data[meas_key], 'b-', linewidth=2.0)
      ax_meas.set_ylim(ylim)
      if col == 0:
        ax_meas.set_ylabel(ylabel, fontsize=10)
        ax_meas.yaxis.set_label_coords(ylabel_x, 0.5)
      else:
        ax_meas.set_yticklabels([])
      ax_meas.grid(True, alpha=0.3)
      ax_meas.set_xticklabels([])
      if row_meas == 0:
        ax_meas.set_title(pass_label, fontsize=10)

      # ---- Rate row ----
      ax_rate = axes[row_rate, col]
      if seg_data[rate_key] is not None:
        ax_rate.plot(seg_time, seg_data[rate_key], 'b-', linewidth=2.0)
      ax_rate.set_ylim(rate_ylim)
      if col == 0:
        ax_rate.set_ylabel(rate_ylabel, fontsize=10)
        ax_rate.yaxis.set_label_coords(ylabel_x, 0.5)
      else:
        ax_rate.set_yticklabels([])
      ax_rate.grid(True, alpha=0.3)
      ax_rate.set_xticklabels([])

    # Bottom row (el_dot) gets x-axis labels
    ax_bottom = axes[5, col]
    if epoch_dt_utc is not None:
      ax_bottom.tick_params(axis='x', rotation=45)
      ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax_bottom.set_xlabel('UTC Time', fontsize=10)

  # Add overall title
  fig.suptitle(title_text, fontsize=14, y=0.98)

  # Adjust layout with tighter spacing
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

  return fig


def plot_error_skyplot(
  measurements : SimulatedMeasurements,
  epoch_dt_utc : Optional[datetime] = None,
  title_text   : str = "Measurement Error",
) -> Optional[Figure]:
  """
  Plot error time-series (measured - truth) for range, range_rate, az, az_rate, el, el_rate.
  
  Creates a grid with 6 rows × num_passes columns:
    Row 0: Range Error
    Row 1: Range Rate Error
    Row 2: Azimuth Error
    Row 3: Azimuth Rate Error
    Row 4: Elevation Error
    Row 5: Elevation Rate Error
  
  Each column is a trackable pass (based on tracker constraints applied to truth).

  Input:
  ------
    measurements : SimulatedMeasurements
      Simulated measurements containing both truth and measured (noisy) values.
    epoch_dt_utc : datetime, optional
      Reference epoch (start time) for time axis formatting.
    title_text : str
      Title for the plot.

  Output:
  -------
    fig : matplotlib.figure.Figure | None
      Figure object containing the error plots, or None if no trackable passes.
  """
  # Get truth and measured from SimulatedMeasurements
  topo_truth = measurements.truth
  topo_meas  = measurements.measured
  tracker    = measurements.tracker
  delta_time_epoch = topo_truth.delta_time_epoch

  # Convert to degrees for display
  az_meas_deg  = topo_meas.azimuth   * CONVERTER.DEG_PER_RAD
  el_meas_deg  = topo_meas.elevation * CONVERTER.DEG_PER_RAD
  az_truth_deg = topo_truth.azimuth   * CONVERTER.DEG_PER_RAD
  el_truth_deg = topo_truth.elevation * CONVERTER.DEG_PER_RAD

  # Normalize azimuth to -180° to +180° range
  az_meas_deg  = ((az_meas_deg + 180.0) % 360.0) - 180.0
  az_truth_deg = ((az_truth_deg + 180.0) % 360.0) - 180.0

  # Get range values
  range_meas_m  = topo_meas.range
  range_truth_m = topo_truth.range
  range_meas_km  = range_meas_m / 1000.0
  range_truth_km = range_truth_m / 1000.0

  # Get rates (convert angular rates to deg/s)
  az_dot_meas_deg  = topo_meas.azimuth_dot   * CONVERTER.DEG_PER_RAD if topo_meas.azimuth_dot is not None else None
  el_dot_meas_deg  = topo_meas.elevation_dot * CONVERTER.DEG_PER_RAD if topo_meas.elevation_dot is not None else None
  rng_dot_meas     = topo_meas.range_dot  # m/s
  az_dot_truth_deg = topo_truth.azimuth_dot   * CONVERTER.DEG_PER_RAD if topo_truth.azimuth_dot is not None else None
  el_dot_truth_deg = topo_truth.elevation_dot * CONVERTER.DEG_PER_RAD if topo_truth.elevation_dot is not None else None
  rng_dot_truth    = topo_truth.range_dot  # m/s

  # Convert rate to km/s
  rng_dot_meas_km  = rng_dot_meas / 1000.0 if rng_dot_meas is not None else None
  rng_dot_truth_km = rng_dot_truth / 1000.0 if rng_dot_truth is not None else None

  # Compute errors (measurement - truth)
  range_err_km   = range_meas_km - range_truth_km
  az_err_deg     = az_meas_deg - az_truth_deg
  el_err_deg     = el_meas_deg - el_truth_deg

  # Handle azimuth wraparound (if error > 180, it wrapped)
  az_err_deg = np.where(az_err_deg > 180.0, az_err_deg - 360.0, az_err_deg)
  az_err_deg = np.where(az_err_deg < -180.0, az_err_deg + 360.0, az_err_deg)

  # Compute rate errors
  if rng_dot_meas_km is not None and rng_dot_truth_km is not None:
    rng_dot_err_km = rng_dot_meas_km - rng_dot_truth_km
  else:
    rng_dot_err_km = None
  if az_dot_meas_deg is not None and az_dot_truth_deg is not None:
    az_dot_err_deg = az_dot_meas_deg - az_dot_truth_deg
  else:
    az_dot_err_deg = None
  if el_dot_meas_deg is not None and el_dot_truth_deg is not None:
    el_dot_err_deg = el_dot_meas_deg - el_dot_truth_deg
  else:
    el_dot_err_deg = None

  # Build visibility mask based on elevation (above horizon)
  # For error skyplots, we only check elevation to determine visibility,
  # not range/azimuth constraints which may filter out high-altitude satellites
  constraint_valid_mask = el_truth_deg > 0.0  # Above horizon

  # Optionally apply elevation min constraint if specified
  if tracker.performance and tracker.performance.constraints and tracker.performance.constraints.elevation:
    el_min_deg = tracker.performance.constraints.elevation.min * CONVERTER.DEG_PER_RAD
    constraint_valid_mask &= (el_truth_deg >= el_min_deg)

  # Find trackable (valid) segments/passes
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

  num_passes = len(valid_segment_starts)
  
  # If no trackable passes, return None
  if num_passes == 0:
    return None

  # Compute y-axis limits across all passes for consistent scaling
  all_range_err = []
  all_az_err = []
  all_el_err = []
  all_rng_dot_err = []
  all_az_dot_err = []
  all_el_dot_err = []

  for seg_start, seg_end in zip(valid_segment_starts, valid_segment_ends):
    all_range_err.extend(range_err_km[seg_start:seg_end])
    all_az_err.extend(az_err_deg[seg_start:seg_end])
    all_el_err.extend(el_err_deg[seg_start:seg_end])
    if rng_dot_err_km is not None:
      all_rng_dot_err.extend(rng_dot_err_km[seg_start:seg_end])
    if az_dot_err_deg is not None:
      all_az_dot_err.extend(az_dot_err_deg[seg_start:seg_end])
    if el_dot_err_deg is not None:
      all_el_dot_err.extend(el_dot_err_deg[seg_start:seg_end])

  # Compute limits with small padding
  def get_limits(data, pad_frac=0.05):
    if len(data) == 0:
      return (-1, 1)
    dmin, dmax = min(data), max(data)
    pad = (dmax - dmin) * pad_frac if dmax != dmin else abs(dmax) * 0.1 + 0.001
    return (dmin - pad, dmax + pad)

  # Helper for plotting histogram + gaussian
  # (Refactored to module level _plot_hist_and_gaussian)

  range_err_ylim   = get_limits(all_range_err)
  az_err_ylim      = get_limits(all_az_err)
  el_err_ylim      = get_limits(all_el_err)
  rng_dot_err_ylim = get_limits(all_rng_dot_err)
  az_dot_err_ylim  = get_limits(all_az_dot_err)
  el_dot_err_ylim  = get_limits(all_el_dot_err)

  # Create figure: 6 rows × (2 * num_passes) columns
  num_rows = 6
  num_cols = 2 * num_passes
  
  # Width ratios: Time Series (3) vs Histogram (1)
  width_ratios = []
  for _ in range(num_passes):
    width_ratios.extend([3, 1])

  fig_width = max((3.5 + 1.5) * num_passes, 14)
  fig_height = 12
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False, gridspec_kw={'width_ratios': width_ratios})

  # Convert time to UTC datetime if epoch provided
  if epoch_dt_utc is not None:
    time_utc = [epoch_dt_utc + timedelta(seconds=float(t)) for t in delta_time_epoch]

  # Y-label alignment coordinate
  ylabel_x = -0.11

  # Row configuration: (error_data, ylabel, ylim)
  row_configs = [
    ('range_err',     'rng_dot_err',  'Range Error\n[km]',       'Range Rate Error\n[km/s]',     range_err_ylim,   rng_dot_err_ylim),
    ('az_err',        'az_dot_err',   'Azimuth Error\n[deg]',    'Azimuth Rate Error\n[deg/s]',  az_err_ylim,      az_dot_err_ylim),
    ('el_err',        'el_dot_err',   'Elevation Error\n[deg]',  'Elevation Rate Error\n[deg/s]',el_err_ylim,      el_dot_err_ylim),
  ]

  # Plot each pass
  for pass_idx, (seg_start, seg_end) in enumerate(zip(valid_segment_starts, valid_segment_ends)):
    col_ts   = pass_idx * 2
    col_hist = pass_idx * 2 + 1

    # Extract segment error data
    seg_time_s      = delta_time_epoch[seg_start:seg_end]
    seg_range_err   = range_err_km[seg_start:seg_end]
    seg_az_err      = az_err_deg[seg_start:seg_end]
    seg_el_err      = el_err_deg[seg_start:seg_end]
    seg_rng_dot_err = rng_dot_err_km[seg_start:seg_end] if rng_dot_err_km is not None else None
    seg_az_dot_err  = az_dot_err_deg[seg_start:seg_end] if az_dot_err_deg is not None else None
    seg_el_dot_err  = el_dot_err_deg[seg_start:seg_end] if el_dot_err_deg is not None else None

    # Time axis for this segment
    if epoch_dt_utc is not None:
      seg_time = time_utc[seg_start:seg_end]
    else:
      seg_time = seg_time_s / 3600.0  # hours

    # Pass label for title (only on row 0)
    pass_label = f"Pass {pass_idx + 1}"
    if epoch_dt_utc is not None and len(seg_time) > 0:
      pass_start_str = seg_time[0].strftime('%H:%M:%S')
      pass_end_str   = seg_time[-1].strftime('%H:%M:%S')
      pass_label = f"Pass {pass_idx + 1}\n{pass_start_str} - {pass_end_str}"

    # Data arrays for each error type
    seg_data = {
      'range_err': seg_range_err,
      'az_err': seg_az_err,
      'el_err': seg_el_err,
      'rng_dot_err': seg_rng_dot_err,
      'az_dot_err': seg_az_dot_err,
      'el_dot_err': seg_el_dot_err,
    }

    # Plot each row pair (measurement error + rate error)
    for meas_idx, (meas_key, rate_key, ylabel, rate_ylabel, ylim, rate_ylim) in enumerate(row_configs):
      row_meas = meas_idx * 2      # 0, 2, 4
      row_rate = meas_idx * 2 + 1  # 1, 3, 5

      # ---- Measurement error row ----
      # Time Series
      ax_meas = axes[row_meas, col_ts]
      ax_meas.plot(seg_time, seg_data[meas_key], 'b-', linewidth=2.0)
      ax_meas.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
      ax_meas.set_ylim(ylim)
      if col_ts == 0:
        ax_meas.set_ylabel(ylabel, fontsize=10)
        ax_meas.yaxis.set_label_coords(ylabel_x, 0.5)
      else:
        # If not first pass column, maybe hide Y labels? 
        # But here col_ts is variable. It is 0, 2, 4...
        # Only show Y label on absolute column 0
        if col_ts > 0:
           ax_meas.set_yticklabels([])

      ax_meas.grid(True, alpha=0.3)
      ax_meas.set_xticklabels([])
      if row_meas == 0:
        ax_meas.set_title(pass_label, fontsize=10)
      
      # Histogram
      ax_hist_meas = axes[row_meas, col_hist]
      _plot_hist_and_gaussian(ax_hist_meas, seg_data[meas_key], ylim)

      # ---- Rate error row ----
      # Time Series
      ax_rate = axes[row_rate, col_ts]
      if seg_data[rate_key] is not None:
        ax_rate.plot(seg_time, seg_data[rate_key], 'b-', linewidth=2.0)
      ax_rate.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
      ax_rate.set_ylim(rate_ylim)
      if col_ts == 0:
        ax_rate.set_ylabel(rate_ylabel, fontsize=10)
        ax_rate.yaxis.set_label_coords(ylabel_x, 0.5)
      else:
        if col_ts > 0:
           ax_rate.set_yticklabels([])

      ax_rate.grid(True, alpha=0.3)
      ax_rate.set_xticklabels([])

      # Histogram
      ax_hist_rate = axes[row_rate, col_hist]
      _plot_hist_and_gaussian(ax_hist_rate, seg_data[rate_key], rate_ylim)

    # Bottom row (el_dot_err) gets x-axis labels (only on TS column)
    ax_bottom = axes[5, col_ts]
    if epoch_dt_utc is not None:
      ax_bottom.tick_params(axis='x', rotation=45)
      ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax_bottom.set_xlabel('UTC Time', fontsize=10)

  # Add overall title
  fig.suptitle(title_text, fontsize=14, y=0.98)

  # Adjust layout with tighter spacing
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

  return fig


def plot_measurement_errors(
  measurements : SimulatedMeasurements,
  tracker      : TrackerStation,
  epoch_dt_utc : Optional[datetime] = None,
  title_text   : str = "Measurement Errors",
) -> Figure:
  """
  Plot measurement errors (measured - truth) for all 6 measurement types vs time.

  Creates a 3x2 grid of subplots:
    - Range error vs time
    - Range rate error vs time
    - Azimuth error vs time
    - Azimuth rate error vs time
    - Elevation error vs time
    - Elevation rate error vs time

  Input:
  ------
    measurements : SimulatedMeasurements
      Simulated measurements containing truth and measured data.
    tracker : TrackerStation
      Ground tracking station.
    epoch_dt_utc : datetime, optional
      Reference epoch for time conversion.
    title_text : str
      Title for the plot.

  Output:
  -------
    fig : matplotlib.figure.Figure
      Figure object containing the error plots.
  """
  fig = plt.figure(figsize=(16, 12))

  # Extract truth and measured data
  truth = measurements.truth
  measured = measurements.measured

  # Get time array
  if hasattr(truth, 'delta_time_epoch'):
    delta_time_epoch = truth.delta_time_epoch
  else:
    # Should not happen with SimulatedMeasurements, but fallback
    delta_time_epoch = np.arange(len(truth.azimuth)) * 10.0  # Dummy time

  # Convert to hours or UTC datetime
  if epoch_dt_utc is not None:
    time_utc = [epoch_dt_utc + timedelta(seconds=float(t)) for t in delta_time_epoch]
    time_x = time_utc
    xlabel = 'UTC Time'
  else:
    time_hrs = delta_time_epoch / 3600.0
    time_x = time_hrs
    xlabel = 'Time [hours]'

  # Compute errors (measured - truth)
  range_error = measured.range - truth.range  # meters
  range_rate_error = measured.range_dot - truth.range_dot if measured.range_dot is not None else None  # m/s
  azimuth_error = measured.azimuth - truth.azimuth  # radians
  azimuth_rate_error = measured.azimuth_dot - truth.azimuth_dot if measured.azimuth_dot is not None else None  # rad/s
  elevation_error = measured.elevation - truth.elevation  # radians
  elevation_rate_error = measured.elevation_dot - truth.elevation_dot if measured.elevation_dot is not None else None  # rad/s

  # Convert angular errors to degrees
  azimuth_error_deg = azimuth_error * CONVERTER.DEG_PER_RAD
  elevation_error_deg = elevation_error * CONVERTER.DEG_PER_RAD
  azimuth_rate_error_deg_per_s = azimuth_rate_error * CONVERTER.DEG_PER_RAD if azimuth_rate_error is not None else None
  elevation_rate_error_deg_per_s = elevation_rate_error * CONVERTER.DEG_PER_RAD if elevation_rate_error is not None else None

  # Create 3x5 subplot grid (TS, Hist, Spacer, TS, Hist)
  # Row 0: Range (0,1), Range Rate (3,4)
  # Row 1: Azimuth (0,1), Azimuth Rate (3,4)
  # Row 2: Elevation (0,1), Elevation Rate (3,4)
  # Spacer is column 2
  
  fig, axes = plt.subplots(3, 5, figsize=(20, 12), gridspec_kw={'width_ratios': [3, 1, 0.4, 3, 1]})

  # Helper to set limits
  def get_data_limits(data, pad_frac=0.05):
    if data is None or len(data) == 0: return (-1, 1)
    dmin, dmax = np.min(data), np.max(data)
    pad = (dmax - dmin) * pad_frac if dmax != dmin else abs(dmax) * 0.1 + 0.001
    return dmin - pad, dmax + pad

  # Calculate limits for consistent scaling between TS and Hist
  range_ylim     = get_data_limits(range_error)
  range_rate_ylim= get_data_limits(range_rate_error) if range_rate_error is not None else None
  az_ylim        = get_data_limits(azimuth_error_deg)
  az_rate_ylim   = get_data_limits(azimuth_rate_error_deg_per_s) if azimuth_rate_error_deg_per_s is not None else None
  el_ylim        = get_data_limits(elevation_error_deg)
  el_rate_ylim   = get_data_limits(elevation_rate_error_deg_per_s) if elevation_rate_error_deg_per_s is not None else None

  # Plot Config: (TS_Col, Hist_Col, Row, Data, Label, YLim, Title)
  # Note: Right graphs are now at columns 3 and 4
  plot_defs = [
    # Row 0: Range, Range Rate
    (0, 1, 0, range_error, 'Range Error [m]', range_ylim, 'Range Error vs Time'),
    (3, 4, 0, range_rate_error, 'Range Rate Error [m/s]', range_rate_ylim, 'Range Rate Error vs Time'),
    # Row 1: Azimuth, Azimuth Rate
    (0, 1, 1, azimuth_error_deg, 'Azimuth Error [deg]', az_ylim, 'Azimuth Error vs Time'),
    (3, 4, 1, azimuth_rate_error_deg_per_s, 'Azimuth Rate Error [deg/s]', az_rate_ylim, 'Azimuth Rate Error vs Time'),
    # Row 2: Elevation, Elevation Rate
    (0, 1, 2, elevation_error_deg, 'Elevation Error [deg]', el_ylim, 'Elevation Error vs Time'),
    (3, 4, 2, elevation_rate_error_deg_per_s, 'Elevation Rate Error [deg/s]', el_rate_ylim, 'Elevation Rate Error vs Time'),
  ]
  
  # Hide all axes in spacer column (col 2)
  for r in range(3):
    axes[r, 2].axis('off')

  for ts_col, hist_col, row, data, ylabel, ylim, title in plot_defs:
    # Time Series
    ax_ts = axes[row, ts_col]
    if data is not None:
      ax_ts.plot(time_x, data, 'b-', linewidth=1.5, alpha=0.7)
      if ylim: ax_ts.set_ylim(ylim)
    
    ax_ts.axhline(y=0, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
    ax_ts.set_ylabel(ylabel, fontsize=11)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_title(title, fontsize=12)
    
    if epoch_dt_utc is not None:
      ax_ts.tick_params(axis='x', rotation=45)
      ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    else:
      # Only show label on bottom row if not date
      if row != 2:
        ax_ts.set_xticklabels([])
    
    if row == 2:
      ax_ts.set_xlabel(xlabel, fontsize=11)

    # Histogram
    ax_hist = axes[row, hist_col]
    if data is not None:
      _plot_hist_and_gaussian(ax_hist, data, ylim)
    else:
      ax_hist.axis('off')

  # Add overall title
  fig.suptitle(title_text, fontsize=14, y=0.995)

  # Adjust layout
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

  return fig
