import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from plot.utility import get_equal_limits

def plot_3d_trajectories(
    result: dict,
):
    """
    Plot 3D position and velocity trajectories in a 1x2 grid.
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Extract state vectors
    states = result['state']
    pos_x, pos_y, pos_z = states[0, :], states[1, :], states[2, :]
    vel_x, vel_y, vel_z = states[3, :], states[4, :], states[5, :]
    
    # Plot 3D position trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(pos_x, pos_y, pos_z, 'b-', linewidth=1)
    ax1.scatter([pos_x[0]], [pos_y[0]], [pos_z[0]], c='b', s=100, marker='>', label='Start')
    ax1.scatter([pos_x[-1]], [pos_y[-1]], [pos_z[-1]], c='b', s=100, marker='s', label='End')
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
    ax2.scatter([vel_x[0]], [vel_y[0]], [vel_z[0]], c='r', s=100, marker='>', label='Start')
    ax2.scatter([vel_x[-1]], [vel_y[-1]], [vel_z[-1]], c='r', s=100, marker='s', label='End')
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


def plot_pos_vel_time_series(
    result: dict,
):
    """
    Plot position and velocity components vs time in a 2x1 grid.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    # Extract data
    time = result['time']
    states = result['state']
    pos_x, pos_y, pos_z = states[0, :], states[1, :], states[2, :]
    vel_x, vel_y, vel_z = states[3, :], states[4, :], states[5, :]
    
    # Calculate magnitudes
    pos_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
    # Plot position vs time
    ax[0].plot(time, pos_x, 'r-', label='X', linewidth=1.5)
    ax[0].plot(time, pos_y, 'g-', label='Y', linewidth=1.5)
    ax[0].plot(time, pos_z, 'b-', label='Z', linewidth=1.5)
    ax[0].plot(time, pos_mag, 'k--', label='Magnitude', linewidth=2)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Position [m]')
    ax[0].legend()
    ax[0].grid(True)

    # Plot velocity vs time
    ax[1].plot(time, vel_x, 'r-', label='X', linewidth=1.5)
    ax[1].plot(time, vel_y, 'g-', label='Y', linewidth=1.5)
    ax[1].plot(time, vel_z, 'b-', label='Z', linewidth=1.5)
    ax[1].plot(time, vel_mag, 'k--', label='Magnitude', linewidth=2)
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Velocity [m/s]')
    ax[1].grid(True)

    plt.tight_layout()
    return fig