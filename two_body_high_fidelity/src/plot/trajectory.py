import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from plot.utility import get_equal_limits
from constants import CONVERTER
from model.dynamics import PHYSICALCONSTANTS

def plot_3d_trajectories(
    result: dict,
):
    """
    Plot 3D position and velocity trajectories in a 1x2 grid.
    """
    fig = plt.figure(figsize=(18,10))
    
    # Extract state vectors
    states = result['state']
    pos_x, pos_y, pos_z = states[0, :], states[1, :], states[2, :]
    vel_x, vel_y, vel_z = states[3, :], states[4, :], states[5, :]
    
    # Plot 3D position trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Add Earth ellipsoid
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    r_eq = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR
    r_pol = PHYSICALCONSTANTS.EARTH.RADIUS.POLAR
    x_earth = r_eq * np.outer(np.cos(u), np.sin(v))
    y_earth = r_eq * np.outer(np.sin(u), np.sin(v))
    z_earth = r_pol * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='lightblue', alpha=0.3, edgecolor='none') # type: ignore
    
    ax1.plot(pos_x, pos_y, pos_z, 'b-', linewidth=1)
    ax1.scatter([pos_x[0]], [pos_y[0]], [pos_z[0]], s=100, marker='>', facecolors='white', edgecolors='b', linewidths=2, label='Start') # type: ignore
    ax1.scatter([pos_x[-1]], [pos_y[-1]], [pos_z[-1]], s=100, marker='s', facecolors='white', edgecolors='b', linewidths=2, label='End') # type: ignore
    ax1.set_xlabel('Pos-X [m]')
    ax1.set_ylabel('Pos-Y [m]')
    ax1.set_zlabel('Pos-Z [m]') # type: ignore
    ax1.grid(True)
    ax1.set_box_aspect([1,1,1]) # type: ignore
    min_limit, max_limit = get_equal_limits(ax1)
    ax1.set_xlim([min_limit, max_limit]) # type: ignore
    ax1.set_ylim([min_limit, max_limit]) # type: ignore
    ax1.set_zlim([min_limit, max_limit]) # type: ignore

    # Plot 3D velocity trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(vel_x, vel_y, vel_z, 'r-', linewidth=1)
    ax2.scatter([vel_x[0]], [vel_y[0]], [vel_z[0]], s=100, marker='>', facecolors='white', edgecolors='r', linewidths=2, label='Start') # type: ignore
    ax2.scatter([vel_x[-1]], [vel_y[-1]], [vel_z[-1]], s=100, marker='s', facecolors='white', edgecolors='r', linewidths=2, label='End') # type: ignore
    ax2.set_xlabel('Vel-X [m/s]')
    ax2.set_ylabel('Vel-Y [m/s]')
    ax2.set_zlabel('Vel-Z [m/s]') # type: ignore
    ax2.grid(True)
    ax2.set_box_aspect([1,1,1]) # type: ignore
    min_limit, max_limit = get_equal_limits(ax2)
    ax2.set_xlim([min_limit, max_limit]) # type: ignore
    ax2.set_ylim([min_limit, max_limit]) # type: ignore
    ax2.set_zlim([min_limit, max_limit]) # type: ignore

    plt.tight_layout()
    return fig


def plot_time_series(
    result: dict,
):
    """
    Plot position and velocity components vs time in a 2x1 grid.
    """
    fig = plt.figure(figsize=(18,10))
    
    # Extract data
    time = result['time']
    states = result['state']
    pos_x, pos_y, pos_z = states[0, :], states[1, :], states[2, :]
    vel_x, vel_y, vel_z = states[3, :], states[4, :], states[5, :]
    
    # Calculate magnitudes
    pos_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
    # Plot position vs time (spans rows 0-2, column 0)
    ax_pos = plt.subplot2grid((6, 2), (0, 0), rowspan=3)
    ax_pos.plot(time, pos_x, 'r-', label='X', linewidth=1.5)
    ax_pos.plot(time, pos_y, 'g-', label='Y', linewidth=1.5)
    ax_pos.plot(time, pos_z, 'b-', label='Z', linewidth=1.5)
    ax_pos.plot(time, pos_mag, 'k-', label='Magnitude', linewidth=2)
    ax_pos.set_xticklabels([])
    ax_pos.set_ylabel('Position\n[m]')
    ax_pos.legend()
    ax_pos.grid(True)
    ax_pos.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot velocity vs time (spans rows 3-5, column 0)
    ax_vel = plt.subplot2grid((6, 2), (3, 0), rowspan=3, sharex=ax_pos)
    ax_vel.plot(time, vel_x, 'r-', label='X', linewidth=1.5)
    ax_vel.plot(time, vel_y, 'g-', label='Y', linewidth=1.5)
    ax_vel.plot(time, vel_z, 'b-', label='Z', linewidth=1.5)
    ax_vel.plot(time, vel_mag, 'k-', label='Magnitude', linewidth=2)
    ax_vel.set_xlabel('Time\n[s]')
    ax_vel.set_ylabel('Velocity\n[m/s]')
    ax_vel.legend()
    ax_vel.grid(True)
    ax_vel.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot sma vs time (row 0, column 1)
    ax_sma = plt.subplot2grid((6, 2), (0, 1), sharex=ax_pos)
    ax_sma.plot(time, result['coe']['sma'], 'b-', linewidth=1.5)
    ax_sma.set_xticklabels([])
    ax_sma.set_ylabel('Semi-Major Axis\n[m]')
    ax_sma.grid(True)
    ax_sma.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot ecc vs time (row 1, column 1)
    ax_ecc = plt.subplot2grid((6, 2), (1, 1), sharex=ax_pos)
    ax_ecc.plot(time, result['coe']['ecc'], 'b-', linewidth=1.5)
    ax_ecc.set_xticklabels([])
    ax_ecc.set_ylabel('Eccentricity\n[-]')
    ax_ecc.grid(True)
    ax_ecc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot inc vs time (row 2, column 1)
    ax_inc = plt.subplot2grid((6, 2), (2, 1), sharex=ax_pos)
    ax_inc.plot(time, result['coe']['inc'] * CONVERTER.RAD2DEG, 'b-', linewidth=1.5)
    ax_inc.set_xticklabels([])
    ax_inc.set_ylabel('Inclination\n[deg]')
    ax_inc.grid(True)
    ax_inc.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot raan vs time (row 3, column 1)
    ax_raan = plt.subplot2grid((6, 2), (3, 1), sharex=ax_pos)
    ax_raan.plot(time, result['coe']['raan'] * CONVERTER.RAD2DEG, 'b-', linewidth=1.5)
    ax_raan.set_xticklabels([])
    ax_raan.set_ylabel('RAAN\n[deg]')
    ax_raan.grid(True)
    ax_raan.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot argp vs time (row 4, column 1)
    ax_argp = plt.subplot2grid((6, 2), (4, 1), sharex=ax_pos)
    argp_unwrapped = np.unwrap(result['coe']['argp']) * CONVERTER.RAD2DEG
    ax_argp.plot(time, argp_unwrapped, 'b-', linewidth=1.5)
    ax_argp.set_xticklabels([])
    ax_argp.set_ylabel('Argument of Perigee\n[deg]')
    ax_argp.grid(True)
    ax_argp.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Plot ta, ea, ma vs time (row 5, column 1)
    ax_anom = plt.subplot2grid((6, 2), (5, 1), sharex=ax_pos)
    ax_anom.plot(time, result['coe']['ta'] * CONVERTER.RAD2DEG, 'r-', label='TA', linewidth=1.5)
    ax_anom.plot(time, result['coe']['ea'] * CONVERTER.RAD2DEG, 'g-', label='EA', linewidth=1.5)
    ax_anom.plot(time, result['coe']['ma'] * CONVERTER.RAD2DEG, 'b-', label='MA', linewidth=1.5)
    ax_anom.set_xlabel('Time\n[s]')
    ax_anom.set_ylabel('Anomaly\n[deg]')
    ax_anom.legend()
    ax_anom.grid(True)
    ax_anom.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Align y-axis labels for right column
    fig.align_ylabels([ax_sma, ax_ecc, ax_inc, ax_raan, ax_argp, ax_anom])
    
    # Align y-axis labels for left column
    fig.align_ylabels([ax_pos, ax_vel])

    plt.subplots_adjust(hspace=0.17, wspace=0.2)
    return fig