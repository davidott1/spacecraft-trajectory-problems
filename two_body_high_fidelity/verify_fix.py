
import sys
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.input.loader import load_files, get_horizons_ephemeris, unload_files
from src.propagation.propagator import run_propagations
from src.input.configuration import build_config
from src.input.cli import parse_command_line_arguments
from src.propagation.state_initializer import get_initial_state

def main():
    # 1. Setup Arguments (Mimic lageos2.yaml)
    # Mock CLI args if none provided
    old_argv = sys.argv
    if len(sys.argv) <= 1:
        print("Using Mock Arguments (LAGEOS-2)...")
        sys.argv = ["main.py", "--config", "lageos2.yaml", "--auto-download"]
    
    try:
        args = parse_command_line_arguments()
    finally:
        sys.argv = old_argv

    # 2. Build Config
    config = build_config(
        args.initial_state_norad_id,
        args.initial_state_filename,
        args.timespan,
        args.include_drag,
        args.compare_tle,
        args.compare_jpl_horizons,
        args.third_bodies,
        args.gravity_harmonics_coefficients,
        args.include_srp,
        args.include_relativity,
        args.include_solid_tides,
        args.include_ocean_tides,
        args.auto_download,
        args.initial_state_source,
        args.gravity_harmonics_degree_order,
        args.gravity_model_filename,
        args.atol,
        args.rtol,
        args.include_tracker_skyplots,
        args.tracker_filename,
        args.tracker_filepath,
        args.include_tracker_on_body,
        # args.include_orbit_determination, # REMOVED: build_config doesn't accept this argument in this version
    )

    # Updated Gravity Config for higher fidelity test
    config.gravity.spherical_harmonics.degree = 70
    config.gravity.spherical_harmonics.order  = 70

    # Override GP to match DE440 (Horizons standard)
    de440_gp = 3.98600435420e+14
    config.gravity.gp = de440_gp
    config.gravity.spherical_harmonics.gp = de440_gp
    
    # 3. Load Dependencies
    spherical_harmonics_model, trackers = load_files(
        spice_kernels_folderpath  = config.output_paths.spice_kernels_folderpath,
        lsk_filepath              = config.output_paths.lsk_filepath,
        gravity_model_folderpath  = config.gravity.folderpath,
        gravity_model_filename    = config.gravity.filename,
        gravity_model_degree      = config.gravity.spherical_harmonics.degree,
        gravity_model_order       = config.gravity.spherical_harmonics.order,
        gravity_coefficient_names = config.gravity.spherical_harmonics.coefficients if config.gravity.spherical_harmonics.enabled else None,
        tracker_filepath          = config.output_paths.tracker_filepath,
    )
    
    if spherical_harmonics_model:
        config.gravity.spherical_harmonics.model = spherical_harmonics_model
        config.gravity.spherical_harmonics.gp     = spherical_harmonics_model.gp
        config.gravity.spherical_harmonics.radius = spherical_harmonics_model.radius

    # 4. Get Horizons
    print("Loading Horizons...")
    result_jpl_horizons_ephemeris = get_horizons_ephemeris(
      jpl_horizons_folderpath = config.output_paths.jpl_horizons_folderpath,
      desired_time_o_dt       = config.time_o_dt,
      desired_time_f_dt       = config.time_f_dt,
      norad_id                = config.initial_state.norad_id,
      object_name             = config.object_name,
      auto_download           = config.auto_download,
    )
    
    # 5. Get Initial State
    initial_state = get_initial_state(
        tle_line_1                    = getattr(config, 'tle_line_1', None),
        tle_line_2                    = getattr(config, 'tle_line_2', None),
        time_o_dt                     = config.time_o_dt,
        result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
        initial_state_source          = config.initial_state.source,
    )
    
    print(f"Initial State Source: {config.initial_state.source}")
    print(f"Initial State: {initial_state}")
    
    # 6. Run Propagation
    print("Running Propagation...")
    
    # helper to unpack initial state
    initial_state_vector = initial_state[0]
    initial_state_epoch  = initial_state[1]

    # Update configuration start time to match the actual loaded initial state epoch
    # This ensures we don't start the simulation at 00:00:00 with the state from 00:00:50
    config.time_o_dt = initial_state_epoch
    config.propagation_config.time_o_dt = initial_state_epoch

    hf_result, _ = run_propagations(
        initial_state                 = initial_state_vector,
        propagation_config            = config.propagation_config,
        spacecraft                    = config.spacecraft,
        compare_tle                   = False,
        compare_jpl_horizons          = True,
        result_jpl_horizons_ephemeris = result_jpl_horizons_ephemeris,
        tle_line_1                    = getattr(config, 'tle_line_1', None),
        tle_line_2                    = getattr(config, 'tle_line_2', None),
        two_body_gravity_model        = config.gravity,
    )
    
    # 7. Check Error at first ephemeris point
    if hf_result.at_ephem_times and hf_result.at_ephem_times.success:
        # Get first point
        prop_state_0 = hf_result.at_ephem_times.state[:, 0]
        horizons_state_0 = result_jpl_horizons_ephemeris.state[:, 0]
        
        diff = prop_state_0 - horizons_state_0
        pos_diff = np.linalg.norm(diff[:3])
        
        print(f"\n--- Validation Results ---")
        print(f"First Horizons Point Time (UTC): {result_jpl_horizons_ephemeris.time_grid.initial}")
        print(f"Simulation Start Time (UTC): {config.time_o_dt}")
        offset = (result_jpl_horizons_ephemeris.time_grid.initial - config.time_o_dt).total_seconds()
        print(f"Time Offset (s): {offset}")
        print(f"Propagated Pos at T0: {prop_state_0[:3]}")
        print(f"Horizons Pos at T0: {horizons_state_0[:3]}")
        print(f"Position Error at T0: {pos_diff:.4f} m")
        
        # Verify 12 hrs
        idx_end = -1
        prop_state_end = hf_result.at_ephem_times.state[:, idx_end]
        horizons_state_end = result_jpl_horizons_ephemeris.state[:, idx_end]
        diff_end = np.linalg.norm((prop_state_end - horizons_state_end)[:3])
        print(f"Position Error at End: {diff_end:.4f} m")
        
    else:
        print("Propagation failed or interpolation failed.")

    unload_files()

if __name__ == "__main__":
    main()
