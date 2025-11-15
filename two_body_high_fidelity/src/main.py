import matplotlib.pyplot as plt
import numpy             as np
import spiceypy          as spice

from pathlib         import Path
from scipy.integrate import solve_ivp

from src.plot.trajectory             import plot_3d_trajectories, plot_time_series, plot_3d_error, plot_time_series_error
from src.propagation.propagator      import propagate_state_numerical_integration
from src.propagation.tle_propagator  import propagate_tle
from src.propagation.horizons_loader import load_horizons_ephemeris
from src.model.dynamics              import Acceleration, OrbitConverter
from src.model.constants             import PHYSICALCONSTANTS, CONVERTER

def main():
  """
  Propagate ISS orbit using high-fidelity dynamics model.
  Initial state derived from TLE, then propagated with detailed force models.
  Compare with SGP4 and JPL Horizons ephemeris.
  """
  #### INPUT ####

  # ISS TLE (example - update with current TLE)
  tle_line1_iss = "1 25544U 98067A   25274.11280702  .00018412  00000-0  33478-3 0  9995"
  tle_line2_iss = "2 25544  51.6324 142.0598 0001038 182.7689 177.3294 15.49574764531593"

  # Define exact propagation time span: Oct 1, 2025 00:00:00 to Oct 2, 2025 00:00:00 UTC
  from datetime import datetime, timedelta
  from sgp4.api import Satrec
  
  # Parse TLE epoch
  satellite = Satrec.twoline2rv(tle_line1_iss, tle_line2_iss)
  tle_epoch_jd = satellite.jdsatepoch + satellite.jdsatepochF
  tle_epoch_dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=tle_epoch_jd - 2451545.0)
  
  # Target propagation start/end times
  target_start_dt = datetime(2025, 10, 1, 0, 0, 0)
  target_end_dt   = datetime(2025, 10, 2, 0, 0, 0)
  delta_time      = (target_end_dt - target_start_dt).total_seconds()
  
  # Integration time bounds (seconds from TLE epoch)
  integ_time_o = (target_start_dt - tle_epoch_dt).total_seconds()
  integ_time_f = integ_time_o + delta_time
  delta_integ_time = integ_time_f - integ_time_o
  
  print(f"\nPropagation Time Span:")
  print(f"  TLE epoch:     {tle_epoch_dt.isoformat()} UTC")
  print(f"  Target start:  {target_start_dt.isoformat()} UTC")
  print(f"  Target end:    {target_end_dt.isoformat()} UTC")
  print(f"  Time offset from TLE epoch: {integ_time_o/3600:.2f} hours ({integ_time_o:.1f} seconds)")
  print(f"  Propagation duration: {delta_integ_time/3600:.2f} hours")
  
  # ISS properties (approximate)
  mass      = 420000.0    # ISS mass [kg] (approximate)
  cd        = 2.2         # drag coefficient [-]
  area_drag = 1000.0      # cross-sectional area [m²] (approximate)
  
  # Output directory for figures
  output_dir = Path('./output/figures')
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # Horizons ephemeris file
  horizons_file = Path('/Users/davidottesen/github/spacecraft-trajectory-problems/data/ephems/horizons_ephem_25544_iss_20251001_20251008_1m.csv')
  
  #### END INPUT ####

  print("\n" + "="*60)
  print("ISS Orbit Propagation")
  print("="*60)
  
  # Step 1: Get initial state from TLE using SGP4
  print("\nStep 1: Converting TLE to initial Cartesian state...")
  print(f"  TLE Line 1: {tle_line1_iss}")
  print(f"  TLE Line 2: {tle_line2_iss}")
  
  # Propagate TLE to target start time to get position/velocity
  result_tle_initial = propagate_tle(
    tle_line1  = tle_line1_iss,
    tle_line2  = tle_line2_iss,
    time_o     = integ_time_o,
    time_f     = integ_time_o,  # Just get initial state at target start time
    num_points = 1,
    to_j2000   = True,    # Convert from TEME to J2000
  )
  if not result_tle_initial['success']:
    raise RuntimeError(f"Failed to get initial state from TLE: {result_tle_initial['message']}")
  
  initial_state = result_tle_initial['state'][:, 0]
  
  print(f"  Initial position (Oct 1 00:00 UTC): [{initial_state[0]/1e3:.3f}, {initial_state[1]/1e3:.3f}, {initial_state[2]/1e3:.3f}] km")
  print(f"  Initial velocity (Oct 1 00:00 UTC): [{initial_state[3]/1e3:.3f}, {initial_state[4]/1e3:.3f}, {initial_state[5]/1e3:.3f}] km/s")
  
  # Step 2: Set up high-fidelity dynamics model
  print("\nStep 2: Setting up high-fidelity dynamics model ...")
  print(f"  Including: Two-body gravity, J2, J3, J4, Atmospheric drag, Third-body (Sun/Moon)")
  
  # Convert target_start_dt to ET seconds for SPICE
  # SPICE needs ET (Ephemeris Time), not UTC
  # Use spiceypy to do the proper conversion
  
  # Load leap seconds kernel first (minimal kernel set for time conversion)
  lsk_path = Path('/Users/davidottesen/github/spacecraft-trajectory-problems/data/spice_kernels/naif0012.tls')
  spice.furnsh(str(lsk_path))
  
  # Convert UTC datetime to ET seconds past J2000
  utc_str = target_start_dt.strftime('%Y-%m-%dT%H:%M:%S')
  et_j2000_time_o = spice.str2et(utc_str)
  
  print(f"  Target start UTC: {utc_str}")
  print(f"  Target start ET seconds from J2000: {et_j2000_time_o:.3f}")
  
  # Clear kernels (they'll be reloaded in Acceleration init)
  spice.kclear()
  
  acceleration = Acceleration(
    gp                      = PHYSICALCONSTANTS.EARTH.GP,
    et_j2000_time_o         = et_j2000_time_o,
    time_o                  = integ_time_o,
    j2                      = PHYSICALCONSTANTS.EARTH.J2,
    j3                      = PHYSICALCONSTANTS.EARTH.J3,
    j4                      = PHYSICALCONSTANTS.EARTH.J4,
    pos_ref                 = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR,
    mass                    = mass,
    enable_drag             = True,
    cd                      = cd,
    area_drag               = area_drag,
    enable_third_body       = True,
    third_body_use_spice    = True,
    third_body_bodies       = ['SUN', 'MOON'],
    spice_kernel_folderpath = '/Users/davidottesen/github/spacecraft-trajectory-problems/data/spice_kernels',
  )
  
  # Step 3: Propagate with high-fidelity model
  print("\nStep 3: Propagating orbit with high-fidelity model using numerical integration ...")
  print(f"  Time span: {target_start_dt} to {target_end_dt} UTC ({delta_time/3600:.1f} hours)")
  result_hifi = propagate_state_numerical_integration(
    initial_state       = initial_state,
    time_o              = integ_time_o,
    time_f              = integ_time_f,
    dynamics            = acceleration,
    method              = 'DOP853',
    rtol                = 1e-12,
    atol                = 1e-12,
    get_coe_time_series = True,
    gp                  = PHYSICALCONSTANTS.EARTH.GP,
  )

  if result_hifi['success']:
    print(f"  ✓ Propagation successful!")
    print(f"  Number of time steps: {len(result_hifi['time'])}")
    
    # Store integration time (seconds from TLE epoch)
    result_hifi['integ_time_s'] = result_hifi['time']
    print(f"  Integration time range (from TLE epoch): {result_hifi['integ_time_s'][0]:.1f} to {result_hifi['integ_time_s'][-1]:.1f} seconds")
    
    # Create plotting time array (seconds from target start time)
    result_hifi['plot_time_s'] = result_hifi['time'] - integ_time_o
    print(f"  Plotting time range (from Oct 1 00:00): {result_hifi['plot_time_s'][0]:.1f} to {result_hifi['plot_time_s'][-1]:.1f} seconds")
  else:
    print(f"  ✗ Propagation failed: {result_hifi['message']}")
    return result_hifi
  
  # Step 4: Propagating with SGP4 for comparison
  print("\nStep 4: Propagating with SGP4 for comparison...")
  result_sgp4 = propagate_tle(
    tle_line1  = tle_line1_iss,
    tle_line2  = tle_line2_iss,
    time_o     = integ_time_o,
    time_f     = integ_time_f,
    num_points = 1000,
    to_j2000   = True,
  )
  
  if result_sgp4['success']:
    # Store integration time (seconds from TLE epoch)
    result_sgp4['integ_time_s'] = result_sgp4['time']
    print(f"  SGP4 integration time (from TLE epoch): [{result_sgp4['integ_time_s'][0]:.1f}, {result_sgp4['integ_time_s'][-1]:.1f}] seconds")
    
    # Create plotting time array (seconds from target start time)
    result_sgp4['plot_time_s'] = result_sgp4['time'] - integ_time_o
    print(f"  SGP4 plotting time (from Oct 1 00:00): [{result_sgp4['plot_time_s'][0]:.1f}, {result_sgp4['plot_time_s'][-1]:.1f}] seconds")
    print(f"  ✓ SGP4 propagation successful!")
  else:
    print(f"  ✗ SGP4 propagation failed: {result_sgp4['message']}")
  
  # # Step 5: Load Horizons ephemeris
  # print("\nStep 5: Loading JPL Horizons ephemeris...")
  # print(f"  File path: {horizons_file}")
  # print(f"  File exists: {horizons_file.exists()}")
  
  # result_horizons = load_horizons_ephemeris(
  #   filepath = str(horizons_file),
  #   time_o   = 0.0,
  #   time_f   = 86400.0,  # 24 hours worth of data
  # )
  
  # if result_horizons['success']:
  #   print(f"  ✓ Horizons ephemeris loaded!")
  #   print(f"  Horizons file epoch: {result_horizons['epoch'].isoformat()} UTC")
    
  #   # Calculate offset to align Horizons to Oct 1 00:00
  #   horizons_offset = (target_start_dt - result_horizons['epoch']).total_seconds()
  #   print(f"  Time offset to align to Oct 1 00:00: {horizons_offset/3600:.2f} hours")
    
  #   # Find the index closest to our target start time
  #   target_idx = np.argmin(np.abs(result_horizons['time'] - horizons_offset))
  #   actual_offset = result_horizons['time'][target_idx]
    
  #   # Extract 24 hours of data starting from target_idx
  #   end_idx = min(target_idx + 1441, result_horizons['state'].shape[1])  # 1441 points to include 24:00
    
  #   # Slice and re-baseline the time
  #   result_horizons['time'] = result_horizons['time'][target_idx:end_idx] - actual_offset
  #   result_horizons['state'] = result_horizons['state'][:, target_idx:end_idx]
  #   result_horizons['final_state'] = result_horizons['state'][:, -1]
  #   result_horizons['epoch'] = target_start_dt
    
  #   print(f"  Aligned data:")
  #   print(f"    Number of points: {len(result_horizons['time'])}")
  #   print(f"    Time span: {result_horizons['time'][0]:.1f} to {result_horizons['time'][-1]:.1f} seconds")
    
  #   # Compute COEs for Horizons data
  #   num_points = result_horizons['state'].shape[1]
  #   result_horizons['coe'] = {
  #     'sma'  : np.zeros(num_points),
  #     'ecc'  : np.zeros(num_points),
  #     'inc'  : np.zeros(num_points),
  #     'raan' : np.zeros(num_points),
  #     'argp' : np.zeros(num_points),
  #     'ma'   : np.zeros(num_points),
  #     'ta'   : np.zeros(num_points),
  #     'ea'   : np.zeros(num_points),
  #   }
    
  #   for i in range(num_points):
  #     coe = OrbitConverter.pv_to_coe(
  #       result_horizons['state'][0:3, i],
  #       result_horizons['state'][3:6, i],
  #       PHYSICALCONSTANTS.EARTH.GP
  #     )
  #     for key in result_horizons['coe'].keys():
  #       if coe[key] is not None:
  #         result_horizons['coe'][key][i] = coe[key]
  # else:
  #   print(f"  ✗ Horizons loading failed: {result_horizons['message']}")
  #   result_horizons = None
  
  # Step 6: Display results and create plots
  print("\n" + "="*60)
  print("Results Summary")
  print("="*60)
  
  print("\nFinal time ranges for plotting:")
  print(f"  High-fidelity: {result_hifi['plot_time_s'][0]:.1f} to {result_hifi['plot_time_s'][-1]:.1f} seconds")
  if result_sgp4['success']:
    print(f"  SGP4:          {result_sgp4['plot_time_s'][0]:.1f} to {result_sgp4['plot_time_s'][-1]:.1f} seconds")
  # if result_horizons and result_horizons['success']:
  #   print(f"  Horizons:      {result_horizons['time'][0]:.1f} to {result_horizons['time'][-1]:.1f} seconds")
  
  # Print final orbital elements (high-fidelity)
  final_alt_km = (np.linalg.norm(result_hifi['state'][0:3, -1]) - PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR) / 1e3
  print(f"\nFinal altitude (high-fidelity): {final_alt_km:.2f} km")
  print(f"Final semi-major axis: {result_hifi['coe']['sma'][-1]/1e3:.2f} km")
  print(f"Final eccentricity: {result_hifi['coe']['ecc'][-1]:.6f}")
  print(f"Final inclination: {np.rad2deg(result_hifi['coe']['inc'][-1]):.4f}°")
  
  # Create plots
  print("\nGenerating and saving plots...")
  
  # Debug: Check what epoch we're passing
  print(f"\nDEBUG - Epoch being passed to plots: {target_start_dt.isoformat()} UTC")
  print(f"DEBUG - First time value in result_hifi['plot_time_s']: {result_hifi['plot_time_s'][0]} seconds")
  print(f"DEBUG - Expected first UTC time: {(target_start_dt + timedelta(seconds=result_hifi['plot_time_s'][0])).isoformat()}")
  
  # High-fidelity plots
  fig1 = plot_3d_trajectories(result_hifi)
  fig1.suptitle('ISS Orbit - High-Fidelity Propagation', fontsize=16)
  fig1.savefig(output_dir / 'iss_hifi_3d.png', dpi=300, bbox_inches='tight')
  print(f"  Saved: {output_dir / 'iss_hifi_3d.png'}")
  
  fig2 = plot_time_series(result_hifi, epoch=target_start_dt)
  fig2.suptitle('ISS Orbit - High-Fidelity Time Series', fontsize=16)
  fig2.savefig(output_dir / 'iss_hifi_timeseries.png', dpi=300, bbox_inches='tight')
  print(f"  Saved: {output_dir / 'iss_hifi_timeseries.png'}")
  
  # SGP4 plots
  if result_sgp4['success']:
    fig3 = plot_3d_trajectories(result_sgp4)
    fig3.suptitle('ISS Orbit - SGP4 Propagation', fontsize=16)
    fig3.savefig(output_dir / 'iss_sgp4_3d.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'iss_sgp4_3d.png'}")
    
    fig4 = plot_time_series(result_sgp4, epoch=target_start_dt)
    fig4.suptitle('ISS Orbit - SGP4 Time Series', fontsize=16)
    fig4.savefig(output_dir / 'iss_sgp4_timeseries.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'iss_sgp4_timeseries.png'}")
  # # Horizons plots
  # if result_horizons and result_horizons['success']:
  #   fig5 = plot_3d_trajectories(result_horizons)
  #   fig5.suptitle('ISS Orbit - JPL Horizons Ephemeris', fontsize=16)
  #   fig5.savefig(output_dir / 'iss_horizons_3d.png', dpi=300, bbox_inches='tight')
  #   print(f"  Saved: {output_dir / 'iss_horizons_3d.png'}")
    
  #   fig6 = plot_time_series(result_horizons, epoch=target_start_dt)
  #   fig6.suptitle('ISS Orbit - JPL Horizons Time Series', fontsize=16)
  #   fig6.savefig(output_dir / 'iss_horizons_timeseries.png', dpi=300, bbox_inches='tight')
  #   print(f"  Saved: {output_dir / 'iss_horizons_timeseries.png'}")
  
  print(f"\nAll figures saved to: {output_dir}")
  plt.show()
  
  return result_hifi


if __name__ == "__main__":
  main()