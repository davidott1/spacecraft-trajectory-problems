"""
GEO Validation Using NASA HORIZONS Ephemeris

Uses high-precision ephemeris from NASA's HORIZONS system for validation.
Compares:
  1. TLE/SDP4 propagation
  2. High-fidelity with SPICE third-body
  3. High-fidelity with Analytical third-body
  4. High-fidelity without third-body

Against HORIZONS "truth" ephemeris.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from astropy.table import Table
from astropy.time import Time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import TIMEVALUES, CONVERTER
from model.dynamics import TwoBodyDynamics, PHYSICALCONSTANTS
from model.coordinate_system_converter import CoordinateSystemConverter
from main import propagate_orbit # type: ignore
from tle_propagator import propagate_tle


def load_horizons_csv(filepath):
    """
    Load HORIZONS ephemeris from CSV file (downloaded via astroquery)
    
    Returns:
    --------
    dict with 'time' (seconds from epoch), 'state' (6xN array in m, m/s)
    """
    # Read CSV file
    data = Table.read(filepath, format='csv')
    
    # Extract times (convert to seconds from first epoch)
    times = Time(data['datetime_jd'], format='jd')
    epoch = times[0]
    time_sec = (times - epoch).sec
    
    # Extract state vectors (convert km, km/s to m, m/s)
    x = data['x'].data * 1000.0
    y = data['y'].data * 1000.0
    z = data['z'].data * 1000.0
    vx = data['vx'].data * 1000.0
    vy = data['vy'].data * 1000.0
    vz = data['vz'].data * 1000.0
    
    state = np.vstack([x, y, z, vx, vy, vz])
    
    return {
        'time': time_sec,
        'state': state,
        'epoch_jd': epoch.jd,
    }


def validate_with_horizons(horizons_file, tle_file, propagation_days=7):
    """
    Validate propagators against HORIZONS ephemeris
    
    Parameters:
    -----------
    horizons_file : str or Path
        Path to HORIZONS ephemeris CSV file
    tle_file : str or Path
        Path to TLE file (two lines)
    propagation_days : float
        Number of days to propagate
    """
    print("="*80)
    print(" GEO VALIDATION WITH NASA HORIZONS EPHEMERIS")
    print("="*80)
    
    # Load HORIZONS data
    print(f"\nLoading HORIZONS ephemeris from: {horizons_file}")
    horizons_data = load_horizons_csv(horizons_file)
    
    print(f"HORIZONS epoch (JD): {horizons_data['epoch_jd']}")
    print(f"Number of HORIZONS points: {len(horizons_data['time'])}")
    print(f"HORIZONS time span: {horizons_data['time'][0]/TIMEVALUES.ONE_DAY:.2f} to {horizons_data['time'][-1]/TIMEVALUES.ONE_DAY:.2f} days")
    
    # Load TLE
    print(f"\nLoading TLE from: {tle_file}")
    with open(tle_file, 'r') as f:
        lines = f.readlines()
    tle_line1 = lines[0].strip()
    tle_line2 = lines[1].strip()
    print(f"Line 1: {tle_line1}")
    print(f"Line 2: {tle_line2}")
    
    # Use first HORIZONS state as initial condition
    initial_state = horizons_data['state'][:, 0]
    time_o = 0.0
    time_f = min(propagation_days * TIMEVALUES.ONE_DAY, horizons_data['time'][-1])
    num_points = len(horizons_data['time'])
    
    # Spacecraft properties (GEO)
    cd = 0.0
    area = 0.0
    mass = 1.0
    
    # Test 1: TLE/SDP4 propagation
    print("\n" + "-"*80)
    print("Test 1: TLE/SDP4 Propagation")
    print("-"*80)
    
    result_tle = propagate_tle(
        tle_line1=tle_line1,
        tle_line2=tle_line2,
        time_o=time_o,
        time_f=time_f,
        num_points=num_points,
        disable_drag=True,
        to_j2000=True,
    )
    
    if result_tle['success']:
        print("✓ SDP4 propagation successful")
    else:
        print(f"✗ SDP4 propagation failed: {result_tle['message']}")
        result_tle = None
    
    # Test configurations for high-fidelity
    configs = {
        'SPICE 3rd-body': {
            'enable_third_body': True,
            'third_body_use_spice': True,
        },
        'Analytical 3rd-body': {
            'enable_third_body': True,
            'third_body_use_spice': False,
        },
        'No 3rd-body': {
            'enable_third_body': False,
        },
    }
    
    results = {}
    spice_kernels_folderpath = Path(__file__).parent.parent.parent.parent / 'data' / 'spice_kernels'
    
    for name, config in configs.items():
        print(f"\n{'-'*80}")
        print(f"Test: High-Fidelity {name}")
        print(f"{'-'*80}")
        
        dynamics = TwoBodyDynamics(
            gp=PHYSICALCONSTANTS.EARTH.GP,
            time_o=time_o,
            j_2=PHYSICALCONSTANTS.EARTH.J_2,
            j_3=0.0,
            j_4=0.0,
            pos_ref=PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR,
            cd=cd, area=area, mass=mass,
            spice_kernel_dir=str(spice_kernels_folderpath) if config.get('third_body_use_spice') else None,
            third_body_bodies=['SUN', 'MOON'] if config['enable_third_body'] else [],
            **config
        )
        
        result = propagate_orbit(
            initial_state=initial_state,
            time_o=time_o,
            time_f=time_f,
            dynamics=dynamics,
            num_points=num_points,
        )
        
        if result['success']:
            print(f"✓ Propagation successful")
            results[name] = result
        else:
            print(f"✗ Propagation failed: {result['message']}")
    
    # Calculate errors vs HORIZONS
    print("\n" + "="*80)
    print(" RESULTS VS HORIZONS (TRUTH)")
    print("="*80)
    
    errors = {}
    
    # TLE error
    if result_tle:
        pos_error_tle = np.linalg.norm(
            result_tle['state'][0:3, :] - horizons_data['state'][0:3, :], 
            axis=0
        ) / 1000.0  # km
        errors['TLE/SDP4'] = pos_error_tle
    
    # High-fidelity errors
    for name, result in results.items():
        pos_error = np.linalg.norm(
            result['state'][0:3, :] - horizons_data['state'][0:3, :], 
            axis=0
        ) / 1000.0  # km
        errors[name] = pos_error
    
    # Print statistics
    for name, error in errors.items():
        print(f"\n{name}:")
        print(f"  Initial error: {error[0]:.6f} km")
        print(f"  Final error:   {error[-1]:.2f} km")
        print(f"  Mean error:    {np.mean(error):.2f} km")
        print(f"  RMS error:     {np.sqrt(np.mean(error**2)):.2f} km")
        print(f"  Max error:     {np.max(error):.2f} km")
    
    # Plotting
    print("\nGenerating plots...")
    
    # Figure 1: Error plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GEO Validation vs NASA HORIZONS Ephemeris', fontsize=16, fontweight='bold')
    
    time_days = horizons_data['time'] / TIMEVALUES.ONE_DAY
    
    # Plot 1: Position errors (log scale)
    for name, error in errors.items():
        linestyle = '-' if name == 'TLE/SDP4' else '--'
        linewidth = 2.5 if name == 'TLE/SDP4' else 1.5
        axes[0, 0].semilogy(time_days, error, label=name, linewidth=linewidth, linestyle=linestyle)
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Position Error (km)')
    axes[0, 0].set_title('Position Error vs HORIZONS (Log Scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, which='both')
    
    # Plot 2: Error growth rate (linear scale)
    for name, error in errors.items():
        linestyle = '-' if name == 'TLE/SDP4' else '--'
        linewidth = 2.5 if name == 'TLE/SDP4' else 1.5
        axes[0, 1].plot(time_days, error, label=name, linewidth=linewidth, linestyle=linestyle)
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Position Error (km)')
    axes[0, 1].set_title('Position Error vs HORIZONS (Linear Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error difference (High-fidelity vs TLE)
    if 'TLE/SDP4' in errors:
        for name, error in errors.items():
            if name != 'TLE/SDP4':
                error_diff = error - errors['TLE/SDP4']
                axes[1, 0].plot(time_days, error_diff, label=f'{name} - TLE', linewidth=2)
        axes[1, 0].set_xlabel('Time (days)')
        axes[1, 0].set_ylabel('Error Difference (km)')
        axes[1, 0].set_title('High-Fidelity Error - TLE Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # Plot 4: Summary
    axes[1, 1].axis('off')
    summary_lines = ["VALIDATION SUMMARY", "="*40, ""]
    summary_lines.append(f"Reference: NASA HORIZONS")
    summary_lines.append(f"Propagation: {time_days[-1]:.2f} days")
    summary_lines.append("")
    summary_lines.append("Final Position Errors:")
    
    # Sort by final error
    sorted_errors = sorted(errors.items(), key=lambda x: x[1][-1])
    for name, error in sorted_errors:
        summary_lines.append(f"  {name:20s}: {error[-1]:7.2f} km")
    
    summary_lines.append("")
    summary_lines.append("CONCLUSION:")
    best_name, best_error = sorted_errors[0]
    summary_lines.append(f"Best model: {best_name}")
    summary_lines.append(f"Final error: {best_error[-1]:.2f} km")
    
    if 'TLE/SDP4' in errors:
        tle_error = errors['TLE/SDP4'][-1]
        summary_lines.append("")
        summary_lines.append(f"TLE error: {tle_error:.2f} km")
        summary_lines.append(f"Improvement: {tle_error - best_error[-1]:.2f} km")
        summary_lines.append(f"             ({(1 - best_error[-1]/tle_error)*100:.1f}% better)")
    
    summary_text = "\n".join(summary_lines)
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    
    # Figure 2: Position and Velocity Time Series
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Position and Velocity Time Series', fontsize=16, fontweight='bold')
    
    # HORIZONS reference
    horizons_pos = horizons_data['state'][0:3, :] / 1000.0  # km
    horizons_vel = horizons_data['state'][3:6, :] / 1000.0  # km/s
    
    # Position components
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes2[0, i].plot(time_days, horizons_pos[i, :], 'k-', label='HORIZONS', linewidth=2)
        
        if result_tle:
            axes2[0, i].plot(time_days, result_tle['state'][i, :]/1000.0, 
                           label='TLE/SDP4', linewidth=1.5, linestyle='-', alpha=0.7)
        
        for name, result in results.items():
            linestyle = '--' if 'SPICE' in name else '-.' if 'Analytical' in name else ':'
            axes2[0, i].plot(time_days, result['state'][i, :]/1000.0, 
                           label=name, linewidth=1.5, linestyle=linestyle, alpha=0.7)
        
        axes2[0, i].set_xlabel('Time (days)')
        axes2[0, i].set_ylabel(f'{label} Position (km)')
        axes2[0, i].set_title(f'{label} Position vs Time')
        axes2[0, i].legend(fontsize=8)
        axes2[0, i].grid(True, alpha=0.3)
    
    # Velocity components
    for i, label in enumerate(['VX', 'VY', 'VZ']):
        axes2[1, i].plot(time_days, horizons_vel[i, :], 'k-', label='HORIZONS', linewidth=2)
        
        if result_tle:
            axes2[1, i].plot(time_days, result_tle['state'][3+i, :]/1000.0, 
                           label='TLE/SDP4', linewidth=1.5, linestyle='-', alpha=0.7)
        
        for name, result in results.items():
            linestyle = '--' if 'SPICE' in name else '-.' if 'Analytical' in name else ':'
            axes2[1, i].plot(time_days, result['state'][3+i, :]/1000.0, 
                           label=name, linewidth=1.5, linestyle=linestyle, alpha=0.7)
        
        axes2[1, i].set_xlabel('Time (days)')
        axes2[1, i].set_ylabel(f'{label[1:]} Velocity (km/s)')
        axes2[1, i].set_title(f'{label} Velocity vs Time')
        axes2[1, i].legend(fontsize=8)
        axes2[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Figure 3: Orbital Elements
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 12))
    fig3.suptitle('Orbital Elements Time Series', fontsize=16, fontweight='bold')
    
    # Convert HORIZONS states to orbital elements
    converter = CoordinateSystemConverter(PHYSICALCONSTANTS.EARTH.GP)
    horizons_coe = {
        'sma': np.zeros(len(time_days)),
        'ecc': np.zeros(len(time_days)),
        'inc': np.zeros(len(time_days)),
        'raan': np.zeros(len(time_days)),
        'argp': np.zeros(len(time_days)),
        'ta': np.zeros(len(time_days)),
    }
    
    for i in range(len(time_days)):
        coe = converter.rv2coe(horizons_data['state'][0:3, i], horizons_data['state'][3:6, i])
        for key in horizons_coe.keys():
            horizons_coe[key][i] = coe[key]
    
    # Convert TLE states to orbital elements
    if result_tle:
        tle_coe = {
            'sma': np.zeros(len(time_days)),
            'ecc': np.zeros(len(time_days)),
            'inc': np.zeros(len(time_days)),
            'raan': np.zeros(len(time_days)),
            'argp': np.zeros(len(time_days)),
            'ta': np.zeros(len(time_days)),
        }
        for i in range(len(time_days)):
            coe = converter.rv2coe(result_tle['state'][0:3, i], result_tle['state'][3:6, i])
            for key in tle_coe.keys():
                tle_coe[key][i] = coe[key]
    
    # Convert high-fidelity states to orbital elements
    hf_coes = {}
    for name, result in results.items():
        hf_coes[name] = {
            'sma': np.zeros(len(time_days)),
            'ecc': np.zeros(len(time_days)),
            'inc': np.zeros(len(time_days)),
            'raan': np.zeros(len(time_days)),
            'argp': np.zeros(len(time_days)),
            'ta': np.zeros(len(time_days)),
        }
        for i in range(len(time_days)):
            coe = converter.rv2coe(result['state'][0:3, i], result['state'][3:6, i])
            for key in hf_coes[name].keys():
                hf_coes[name][key][i] = coe[key]
    
    # Plot orbital elements
    elements = [
        ('sma', 'Semi-major Axis (km)', 1000.0),
        ('ecc', 'Eccentricity', 1.0),
        ('inc', 'Inclination (deg)', np.degrees(1.0)),
        ('raan', 'RAAN (deg)', np.degrees(1.0)),
        ('argp', 'Arg of Perigee (deg)', np.degrees(1.0)),
        ('ta', 'True Anomaly (deg)', np.degrees(1.0)),
    ]
    
    for idx, (key, label, scale) in enumerate(elements):
        ax = axes3[idx // 2, idx % 2]
        
        # HORIZONS
        ax.plot(time_days, horizons_coe[key] * scale, 'k-', label='HORIZONS', linewidth=2)
        
        # TLE
        if result_tle:
            ax.plot(time_days, tle_coe[key] * scale, 
                   label='TLE/SDP4', linewidth=1.5, linestyle='-', alpha=0.7)
        
        # High-fidelity
        for name in hf_coes.keys():
            linestyle = '--' if 'SPICE' in name else '-.' if 'Analytical' in name else ':'
            ax.plot(time_days, hf_coes[name][key] * scale, 
                   label=name, linewidth=1.5, linestyle=linestyle, alpha=0.7)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.show()
    
    return {
        'horizons': horizons_data,
        'tle': result_tle,
        'results': results,
        'errors': errors,
    }


if __name__ == "__main__":
    # Use downloaded data
    data_dir = Path(__file__).parent.parent.parent.parent / 'data' / 'ephems'
    horizons_file = data_dir / 'horizons_goes76.csv'
    tle_file = data_dir / 'tle_goes76.txt'
    
    if not horizons_file.exists() or not tle_file.exists():
        print(f"Data files not found!")
        print(f"HORIZONS: {horizons_file}")
        print(f"TLE:      {tle_file}")
        print("\nRun download_horizons_data.py first to download the data:")
        print("  python -m src.validation.download_horizons_data")
    else:
        validate_with_horizons(horizons_file, tle_file, propagation_days=7)
