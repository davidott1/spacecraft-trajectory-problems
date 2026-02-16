"""
Plot ephemeris data for any satellite by NORAD ID

Usage:
  python -m src.plot.ephem --initial-state-norad-id <norad_id> --timespan <start_time> <end_time>

Examples:
  python -m src.plot.ephem --initial-state-norad-id 25544 --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00
  python -m src.plot.ephem --initial-state-norad-id 39166 --timespan 2025-10-01 2025-10-02
"""

import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.input.loader import get_horizons_ephemeris, load_files, unload_files
from src.plot.plot_timeseries import plot_time_series
from src.input.cli import parse_time


def parse_command_line_args():
    """Parse command line arguments for ephemeris plotting."""
    parser = argparse.ArgumentParser(
        description='Plot ephemeris time series for a satellite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.plot.ephem --initial-state-norad-id 25544 --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00
  python -m src.plot.ephem --initial-state-norad-id 39166 --timespan 2025-10-01 2025-10-02

Time format (UTC assumed):
  YYYY-MM-DD
  YYYY-MM-DD HH:MM
  YYYY-MM-DD HH:MM:SS
  YYYY-MM-DDTHH:MM
  YYYY-MM-DDTHH:MM:SS
        """
    )

    parser.add_argument(
        '--initial-state-norad-id',
        type=str,
        required=True,
        help='NORAD catalog ID of the satellite'
    )

    parser.add_argument(
        '--timespan',
        nargs=2,
        required=True,
        metavar=('START_TIME', 'END_TIME'),
        help='Start and end times for ephemeris (UTC)'
    )

    args = parser.parse_args()

    # Parse timespan
    try:
        start_time = parse_time(args.timespan[0])
        end_time = parse_time(args.timespan[1])
    except ValueError as e:
        print(f"Error parsing timespan: {e}")
        sys.exit(1)

    if end_time <= start_time:
        print("Error: End time must be after start time")
        sys.exit(1)

    return args.initial_state_norad_id, start_time, end_time


def main():
    """Main function to plot ephemeris data."""
    # Parse command-line arguments
    norad_id, start_time, end_time = parse_command_line_args()

    print("\n" + "=" * 60)
    print("  JPL Horizons Ephemeris Plotter")
    print("=" * 60)
    print(f"\n  NORAD ID   : {norad_id}")
    print(f"  Start Time : {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  End Time   : {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()

    # Setup paths with timestamp folder
    output_base = PROJECT_ROOT / 'output'
    data_base = PROJECT_ROOT / 'data'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output = output_base / timestamp

    jpl_horizons_folderpath = data_base / 'ephems'
    spice_kernels_folderpath = data_base / 'spice_kernels'
    lsk_filepath = spice_kernels_folderpath / 'naif0012.tls'
    figures_folderpath = timestamped_output / 'figures'

    # Ensure directories exist
    figures_folderpath.mkdir(parents=True, exist_ok=True)

    # Load SPICE kernels (required for time conversions)
    print("Loading SPICE kernels...")
    try:
        load_files(
            spice_kernels_folderpath=spice_kernels_folderpath,
            lsk_filepath=lsk_filepath,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load SPICE kernels: {e}")
        print("Please ensure SPICE kernels are downloaded.")
        sys.exit(1)

    # Get JPL Horizons ephemeris
    print("\nLoading JPL Horizons ephemeris...")
    result = get_horizons_ephemeris(
        jpl_horizons_folderpath=jpl_horizons_folderpath,
        desired_time_o_dt=start_time,
        desired_time_f_dt=end_time,
        norad_id=norad_id,
        object_name=f"norad_{norad_id}",
        auto_download=True,
    )

    if result is None or not result.success:
        print(f"\n[ERROR] Failed to load ephemeris: {result.message if result else 'Unknown error'}")
        unload_files()
        sys.exit(1)

    print("\n[SUCCESS] Ephemeris loaded successfully")
    print(f"  Grid Points : {len(result.time.grid.relative_initial)}")

    # Generate time series plot
    print("\nGenerating time series plot...")
    fig = plot_time_series(result, epoch=start_time)
    fig.suptitle(f'Time Series - NORAD {norad_id} - JPL Horizons', fontsize=16)

    # Save plot
    filename = f'timeseries_cart_coe_mee_jpl_horizons_norad_{norad_id}.png'
    filepath = figures_folderpath / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')

    print(f"\n[SUCCESS] Plot saved:")
    print(f"  {filepath}")
    print("\nDisplaying interactive plot (close window to exit)...")

    # Show interactive plot
    plt.show()

    # Cleanup
    unload_files()
    print()


if __name__ == "__main__":
    main()
