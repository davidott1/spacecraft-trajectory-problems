"""
One-body thrust estimation problem main program.

Example usage:
    python main.py
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
np.random.seed(42)

N_STEPS = 1000
N_MEAS = 1000

@dataclass
class Trajectory:
    n_steps: int = 0 # number of time steps
    time: np.ndarray = field(default_factory=lambda: np.array([])) # time array [s]
    position: np.ndarray = field(default_factory=lambda: np.array([])) # position array [m]
    velocity: np.ndarray = field(default_factory=lambda: np.array([])) # velocity array [m/s]
    acceleration: np.ndarray = field(default_factory=lambda: np.array([])) # acceleration array [m/s^2]
    thrust_acceleration: np.ndarray = field(default_factory=lambda: np.array([])) # thrust acceleration array [m/s^2]

    def __post_init__(self):
        if self.n_steps > 0:
            if len(self.time) == 0:
                self.time = np.zeros(self.n_steps)
            if len(self.position) == 0:
                self.position = np.zeros(self.n_steps)
            if len(self.velocity) == 0:
                self.velocity = np.zeros(self.n_steps)
            if len(self.acceleration) == 0:
                self.acceleration = np.zeros(self.n_steps)
            if len(self.thrust_acceleration) == 0:
                self.thrust_acceleration = np.zeros(self.n_steps)

@dataclass
class Measurements:
    n_meas: int = 0 # number of measurements
    time: np.ndarray = field(default_factory=lambda: np.array([])) # time array [s]
    range: np.ndarray = field(default_factory=lambda: np.array([])) # range array [m]
    range_rate: np.ndarray = field(default_factory=lambda: np.array([])) # range rate array [m/s]

    def __post_init__(self):
        if self.n_meas > 0:
            if len(self.time) == 0:
                self.time = np.zeros(self.n_meas)
            if len(self.range) == 0:
                self.range = np.zeros(self.n_meas)
            if len(self.range_rate) == 0:
                self.range_rate = np.zeros(self.n_meas)

@dataclass
class Body:
    trajectory: Trajectory
    measurements: Measurements

class SimulatePathAndMeasurements:
    def __init__(
            self,
            n_steps: int,
            n_meas: int,
        ):
        self.body = Body(
            trajectory=Trajectory(n_steps=n_steps),
            measurements=Measurements(n_meas=n_meas)
        )

    def dynamics(
            self,
            time,
            state,
            params,
        ):
        delta_time_01 = params.get('delta_time_01', 0.0)
        delta_time_12 = params.get('delta_time_12', 0.0)
        delta_time_23 = params.get('delta_time_23', 0.0)
        pos = state[0]
        vel = state[1]

        acc = 0.0

        if time <= delta_time_01:
            thrust_acc_mag = params.get('thrust_acc_mag_01', 0.0)
            thrust_acc_dir = params.get('thrust_acc_dir_01', 0.0)
        elif time <= delta_time_01 + delta_time_12:
            thrust_acc_mag = params.get('thrust_acc_mag_12', 0.0)
            thrust_acc_dir = params.get('thrust_acc_dir_12', 0.0)
        else:
            thrust_acc_mag = params.get('thrust_acc_mag_23', 0.0)
            thrust_acc_dir = params.get('thrust_acc_dir_23', 0.0)
        thrust_acc = thrust_acc_mag * thrust_acc_dir

        dpos__dtime = vel
        dvel__dtime = acc + thrust_acc

        dstate__dtime = np.array([dpos__dtime, dvel__dtime])

        return dstate__dtime

    def shooting_function(self, thrust_mag_array, params):
        time_o = params.get('time_o', 0.0)
        time_f = params.get('time_f', 0.0)
        pos_o = params.get('pos_o', 0.0)
        vel_o = params.get('vel_o', 0.0)
        initial_state = np.array([pos_o, vel_o])
        thrust_mag = thrust_mag_array[0]
        params['thrust_acc_mag_01'] = thrust_mag
        params['thrust_acc_mag_23'] = thrust_mag
        solution = solve_ivp(
            fun=lambda time, state: self.dynamics(time, state, params),
            t_span=(time_o, time_f),
            y0=initial_state,
            t_eval=self.body.trajectory.time,
            method='RK45',
            rtol=1e-12,
            atol=1e-12,
        )
        pos_f = solution.y[0, -1]
        position_error = pos_f - 1.0 # target position
        return position_error

    def create_path(self):

        # Time parameters
        time_o = 0.0
        time_f = 10.0
        self.body.trajectory.time = np.linspace(time_o, time_f, self.body.trajectory.n_steps)

        # Thrust acceleration parameters (on-off-on)
        thrust_acc_mag = 0.2  # m/s^2
        delta_time_01 = 2.0  # s, thrust-acc on
        delta_time_12 = 6.0  # s, thrust-acc off
        delta_time_23 = time_f - time_o - delta_time_01 - delta_time_12  # s, thrust-acc on, unused

        # Initialize propagation
        pos_o = 0.0  # m
        vel_o = 0.0  # m/s
        state_o = np.array([pos_o, vel_o])

        # Initial guess for thrust magnitude
        thrust_mag_guess = 0.2
        params = {
            'pos_o': pos_o,
            'vel_o': vel_o,
            'time_o': time_o,
            'time_f': time_f,
            'delta_time_01': delta_time_01,
            'delta_time_12': delta_time_12,
            'delta_time_23': delta_time_23,
            'thrust_acc_mag_01': thrust_mag_guess,
            'thrust_acc_dir_01': 1.0,
            'thrust_acc_mag_12': 0.0,
            'thrust_acc_dir_12': 0.0,
            'thrust_acc_mag_23': thrust_mag_guess,
            'thrust_acc_dir_23': -1.0,
        }
        
        # Solve for thrust magnitude that achieves target final conditions
        result = fsolve(
            self.shooting_function,
            thrust_mag_guess,
            args=(params,),
            xtol=1e-12,
        )
        optimal_thrust_mag = result[0]
        
        # Generate final trajectory with optimal thrust
        params_final = {
            'pos_o': pos_o,
            'vel_o': vel_o,
            'time_o': time_o,
            'time_f': time_f,
            'delta_time_01': delta_time_01,
            'delta_time_12': delta_time_12,
            'delta_time_23': delta_time_23,
            'thrust_acc_mag_01': optimal_thrust_mag,
            'thrust_acc_dir_01': 1.0,
            'thrust_acc_mag_12': 0.0,
            'thrust_acc_dir_12': 0.0,
            'thrust_acc_mag_23': optimal_thrust_mag,
            'thrust_acc_dir_23': -1.0,
        }
        solution_final = solve_ivp(
            fun=lambda time, state: self.dynamics(time, state, params_final),
            t_span=(time_o, time_f),
            y0=state_o,
            t_eval=self.body.trajectory.time,
            method='RK45',
            rtol=1e-12,
            atol=1e-12,
        )
        
        # Store results
        self.body.trajectory.position = solution_final.y[0]
        self.body.trajectory.velocity = solution_final.y[1]
        for i, t in enumerate(self.body.trajectory.time):
            if t <= delta_time_01:
                self.body.trajectory.thrust_acceleration[i] = optimal_thrust_mag  # thrust on
            elif t <= delta_time_01 + delta_time_12:
                self.body.trajectory.thrust_acceleration[i] = 0.0  # thrust off
            else:
                self.body.trajectory.thrust_acceleration[i] = -optimal_thrust_mag  # thrust on

        # Print summary
        print( "  Trajectory")
        print(f"    Time             : {    self.body.trajectory.time[0]:8.1e} s    to {    self.body.trajectory.time[-1]:8.1e} s   {self.body.trajectory.n_steps} steps")
        print(f"    Position         : {self.body.trajectory.position[0]:8.1e} m    to {self.body.trajectory.position[-1]:8.1e} m")
        print(f"    Velocity         : {self.body.trajectory.velocity[0]:8.1e} m/s  to {self.body.trajectory.velocity[-1]:8.1e} m/s")
        print(f"    Thrust Acc Mag   : {   optimal_thrust_mag:8.1e} m/s²")

    def create_measurements(self):
        self.body.measurements.time = np.linspace(self.body.trajectory.time[0], self.body.trajectory.time[-1], self.body.measurements.n_meas)
        measurement_range_noise_std = 5 / 100  # m
        measurement_range_rate_noise_std = 1 / 100  # m/s
        self.body.measurements.range      = self.body.trajectory.position + np.random.normal(0, measurement_range_noise_std     , self.body.trajectory.n_steps)
        self.body.measurements.range_rate = self.body.trajectory.velocity + np.random.normal(0, measurement_range_rate_noise_std, self.body.trajectory.n_steps)

        # Randomly thin measurements
        indices = np.sort(np.random.choice(self.body.trajectory.n_steps, int(0.1*self.body.measurements.n_meas), replace=False))
        self.body.measurements.time       = self.body.measurements.time[indices]
        self.body.measurements.range      = self.body.measurements.range[indices]
        self.body.measurements.range_rate = self.body.measurements.range_rate[indices]

        # Print summary
        print( "  Measurements")
        print( "    Range")
        print(f"      Value          : {self.body.measurements.range[0]:8.1e} m    to {self.body.measurements.range[-1]:8.1e} m")
        print(f"      Std Dev        : {self.body.measurements.range[0]:8.1e} m    to {self.body.measurements.range[-1]:8.1e} m")
        print(f"    Range Rate")
        print(f"      Value          : {self.body.measurements.range_rate[0]:8.1e} m/s  to {self.body.measurements.range_rate[-1]:8.1e} m/s")
        print(f"      Std Dev        : {self.body.measurements.range_rate[0]:8.1e} m/s  to {self.body.measurements.range_rate[-1]:8.1e} m/s")

class ThrustEstimator:
    def __init__(self, simulator):
        self.simulator = simulator
        self.approx_thrust_acceleration_true = np.zeros(self.simulator.body.trajectory.n_steps)

    def _get_approx_thrust_acceleration_true(self):
        for idx in range(self.simulator.body.trajectory.n_steps-1):
            delta_vel = self.simulator.body.trajectory.velocity[idx+1] - self.simulator.body.trajectory.velocity[idx]
            delta_time = self.simulator.body.trajectory.time[idx+1] - self.simulator.body.trajectory.time[idx]
            self.approx_thrust_acceleration_true[idx] = delta_vel / delta_time

    def estimate_thrust(self):
        self._get_approx_thrust_acceleration_true()


class PlotResults:
    def __init__(self, simulator, thrust_estimator):
        self.simulator = simulator
        self.thrust_estimator = thrust_estimator

    def plot_all(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(self.simulator.body.trajectory.time, self.simulator.body.trajectory.position, label='Position [m]', color=mcolors.TABLEAU_COLORS['tab:blue'], linewidth=4)
        axs[0].plot(self.simulator.body.measurements.time, self.simulator.body.measurements.range, 'o', label='Range Measurements [m]', color=mcolors.TABLEAU_COLORS['tab:blue'], markersize=4, alpha=0.5)
        axs[0].set_ylabel('Position [m]')
        axs[0].tick_params(axis='x', length=0)
        axs[0].grid()

        axs[1].plot(self.simulator.body.trajectory.time, self.simulator.body.trajectory.velocity, label='Velocity [m/s]', color=mcolors.TABLEAU_COLORS['tab:orange'], linewidth=4)
        axs[1].plot(self.simulator.body.measurements.time, self.simulator.body.measurements.range_rate, 'o', label='Range Rate Measurements [m/s]', color=mcolors.TABLEAU_COLORS['tab:orange'], markersize=4, alpha=0.5)
        axs[1].set_ylabel('Velocity [m/s]')
        axs[1].tick_params(axis='x', length=0)
        axs[1].grid()

        axs[2].plot(self.simulator.body.trajectory.time, self.simulator.body.trajectory.thrust_acceleration, label='Thrust Acceleration [m/s²]', color=mcolors.TABLEAU_COLORS['tab:green'], linewidth=4)
        axs[2].plot(self.simulator.body.trajectory.time[:-1], self.thrust_estimator.approx_thrust_acceleration_true[:-1], '-', label='Approx Thrust Acceleration [m/s²]', color=mcolors.TABLEAU_COLORS['tab:green'], linewidth=12, alpha=0.5)

        axs[2].set_xlabel('Time [s]')
        axs[2].set_ylabel('Thrust Acceleration [m/s²]')
        axs[2].grid()
        
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

def main():
    
    # Start thrust estimation program
    print(2*"\n"+"=========================")
    print(       "THRUST ESTIMATION PROGRAM")
    print(       "=========================")

    # Simulate path and measurements
    print("\nSIMULATE PATH AND MEASUREMENTS")
    simulator = SimulatePathAndMeasurements(n_steps=N_STEPS, n_meas=N_MEAS)
    simulator.create_path()
    simulator.create_measurements()
    
    # Run sequential filter
    print("\nRUN SEQUENTIAL FILTER")
    print("  (Not implemented)")

    # Run smoother
    print("\nRUN SMOOTHER")
    print("  (Not implemented)")

    # Approximate thrust profile
    print("\nAPPROXIMATE THRUST PROFILE")
    thrust_estimator = ThrustEstimator(simulator)
    thrust_estimator.estimate_thrust()

    # Plot results
    print("\nPLOT RESULTS") 
    plotter = PlotResults(simulator, thrust_estimator)
    plotter.plot_all()

    # End thrust estimation program
    print("\nTHRUST ESTIMATION PROGRAM COMPLETE")
    return True

if __name__ == '__main__':
    main()

