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
from scipy.signal import find_peaks
np.random.seed(42)

N_STEPS = 1000
N_MEAS = 1000

PCT_THIN = 0.5

MEASUREMENT_RANGE_NOISE_STD      = 0.100  # m
MEASUREMENT_RANGE_RATE_NOISE_STD = 0.010  # m/s
PROCESS_NOISE_STD                = 0.15*0.0625  # m/s^2

DELTA_TIME_03 = 10.0  # total time [s]
DELTA_TIME_01 =  2.0  # thrust-acc on [s]
DELTA_TIME_12 =  6.0  # thrust-acc off [s]
DELTA_TIME_23 = DELTA_TIME_03 - DELTA_TIME_01 - DELTA_TIME_12 # thrust-acc on [s]

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
        delta_time_01  = DELTA_TIME_01  # s, thrust-acc on
        delta_time_12  = DELTA_TIME_12  # s, thrust-acc off
        delta_time_23  = DELTA_TIME_23  # s, thrust-acc on, unused

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
        self.body.measurements.time       = np.linspace(self.body.trajectory.time[0], self.body.trajectory.time[-1], self.body.measurements.n_meas)
        measurement_range_noise_std       = MEASUREMENT_RANGE_NOISE_STD  # m
        measurement_range_rate_noise_std  = MEASUREMENT_RANGE_RATE_NOISE_STD  # m/s
        self.body.measurements.range      = self.body.trajectory.position + np.random.normal(0, measurement_range_noise_std     , self.body.trajectory.n_steps)
        self.body.measurements.range_rate = self.body.trajectory.velocity + np.random.normal(0, measurement_range_rate_noise_std, self.body.trajectory.n_steps)

        # Randomly thin measurements
        indices = np.sort(np.random.choice(self.body.trajectory.n_steps, int(PCT_THIN*self.body.measurements.n_meas), replace=False))
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

class SequentialFilter:
    def __init__(self, simulator):
        self.simulator = simulator
        # Extend state to include acceleration and jerk [pos, vel, acc, jerk]
        self.state_est_hat = np.array([0.0, 0.0, 0.0, 0.0]) # initial state estimate [pos, vel, acc, jerk]
        self.P = np.eye(4) # initial covariance estimate
        self.process_noise_std = PROCESS_NOISE_STD # std dev of process noise (unmodeled snap) [m/s^4]

        # Store results
        self.time = []
        self.pos  = []
        self.vel  = []
        self.acc  = []
        self.jerk = []  # Add jerk storage
        self.x_hat_history = []
        self.P_history = []
        self.x_hat_minus_history = []
        self.P_minus_history = []
        self.F_history = []

    def run(self):
        print("  Run Sequential Filter")
        measurements = self.simulator.body.measurements
    
        # Use first measurement to initialize state
        # Estimate initial acceleration and jerk as 0
        self.x_hat = np.array([measurements.range[0], measurements.range_rate[0], 0.0, 0.0])
        self.time.append(measurements.time[0])
        self.pos.append(self.x_hat[0])
        self.vel.append(self.x_hat[1])
        self.acc.append(self.x_hat[2])
        self.jerk.append(self.x_hat[3])
        self.x_hat_history.append(self.x_hat)
        self.P_history.append(self.P)

        # No prediction for first step, so add placeholders
        self.x_hat_minus_history.append(self.x_hat)
        self.P_minus_history.append(self.P)
        self.F_history.append(np.eye(4))

        measurement_range_noise_std      = MEASUREMENT_RANGE_NOISE_STD
        measurement_range_rate_noise_std = MEASUREMENT_RANGE_RATE_NOISE_STD
        R = np.diag([measurement_range_noise_std**2, measurement_range_rate_noise_std**2]) # measurement noise covariance
        # Update H to map from 4D state to 2D measurements
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]) # measurement matrix

        for k in range(1, len(measurements.time)):

            # Prediction
            dt = measurements.time[k] - measurements.time[k-1] # time step
            # Update F for constant jerk model
            F = np.array([
                [1.0, dt, 0.5*dt**2, (1/6)*dt**3],
                [0.0, 1.0, dt,      0.5*dt**2],
                [0.0, 0.0, 1.0,     dt],
                [0.0, 0.0, 0.0,     1.0]
            ]) # state transition matrix
            
            # Process noise affects jerk directly (snap model)
            G = np.array([[(1/6)*dt**3], [0.5*dt**2], [dt], [1.0]]) # process noise gain matrix
            Q = G @ G.T * self.process_noise_std**2 # process noise covariance

            x_hat_minus = F @ self.x_hat
            P_minus = F @ self.P @ F.T + Q

            # Update
            z = np.array([measurements.range[k], measurements.range_rate[k]]) # measurement vector
            y = z - H @ x_hat_minus # measurement residual
            S = H @ P_minus @ H.T + R # innovation covariance
            K = P_minus @ H.T @ np.linalg.inv(S) # gain

            self.x_hat = x_hat_minus + K @ y # updated state estimate
            self.P = (np.eye(4) - K @ H) @ P_minus # updated covariance estimate

            # Store results
            self.time.append(measurements.time[k])
            self.pos.append(self.x_hat[0])
            self.vel.append(self.x_hat[1])
            self.acc.append(self.x_hat[2])
            self.jerk.append(self.x_hat[3])
            self.x_hat_history.append(self.x_hat)
            self.P_history.append(self.P)
            self.x_hat_minus_history.append(x_hat_minus)
            self.P_minus_history.append(P_minus)
            self.F_history.append(F)


class Smoother:
    def __init__(
            self,
            sequential_filter: SequentialFilter,
        ):
        self.filter = sequential_filter
        self.time = []
        self.pos = []
        self.vel = []
        self.acc = []
        self.jerk = []  # Add jerk storage

    def run(self):
        print("  Run Smoother (RTS)")
        if not self.filter.time:
            print("  Filter has not been run. Skipping smoother.")
            return

        num_meas = len(self.filter.time)
        x_hat_s = [np.zeros_like(x) for x in self.filter.x_hat_history]
        P_s = [np.zeros_like(p) for p in self.filter.P_history]

        x_hat_s[-1] = self.filter.x_hat_history[-1]
        P_s[-1] = self.filter.P_history[-1]

        for k in range(num_meas - 2, -1, -1):
            # Smoother gain
            C_k = self.filter.P_history[k] @ self.filter.F_history[k+1].T @ np.linalg.inv(self.filter.P_minus_history[k+1])
            
            # Smoothed state and covariance
            x_hat_s[k] = self.filter.x_hat_history[k] + C_k @ (x_hat_s[k+1] - self.filter.x_hat_minus_history[k+1])
            P_s[k] = self.filter.P_history[k] + C_k @ (P_s[k+1] - self.filter.P_minus_history[k+1]) @ C_k.T

        self.time = self.filter.time
        self.pos = [x[0] for x in x_hat_s]
        self.vel = [x[1] for x in x_hat_s]
        self.acc = [x[2] for x in x_hat_s]
        self.jerk = [x[3] for x in x_hat_s]  # Extract jerk estimates


class ThrustEstimator:
    def __init__(
            self, 
            simulator: SimulatePathAndMeasurements, 
            sequential_filter: SequentialFilter,
            smoother: Smoother,
        ):
        self.simulator = simulator
        self.sequential_filter = sequential_filter
        self.smoother = smoother
        self.approx_thrust_acceleration_true = np.zeros(self.simulator.body.trajectory.n_steps)
        self.approx_thrust_acceleration_filter = np.zeros(int(PCT_THIN*self.simulator.body.measurements.n_meas))
        self.approx_thrust_acceleration_smoother = np.zeros(int(PCT_THIN*self.simulator.body.measurements.n_meas))
        # No need for the _get methods since we directly estimate acceleration now

    def _get_approx_thrust_acceleration_true(self):
        for idx in range(self.simulator.body.trajectory.n_steps-1):
            delta_vel = self.simulator.body.trajectory.velocity[idx+1] - self.simulator.body.trajectory.velocity[idx]
            delta_time = self.simulator.body.trajectory.time[idx+1] - self.simulator.body.trajectory.time[idx]
            self.approx_thrust_acceleration_true[idx] = delta_vel / delta_time

    def estimate_thrust(self):
        print("  Approximate thrust acceleration using finite difference of true velocity")
        self._get_approx_thrust_acceleration_true()
        # No need to compute from velocity differences since we have direct acceleration estimates
        # Just copy the filter and smoother acceleration estimates
        self.approx_thrust_acceleration_filter = np.array(self.sequential_filter.acc)
        self.approx_thrust_acceleration_smoother = np.array(self.smoother.acc)


class ThrustProfileOptimizer:
    def __init__(
            self,
            smoother: Smoother,
            thrust_magnitude: float = 0.0625,  # m/s^2
            total_time: float = 10.0,  # seconds
        ):
        self.smoother = smoother
        self.thrust_magnitude = thrust_magnitude
        self.total_time = total_time
        self.delta_time_01 = None  # First segment duration (to be optimized)
        self.delta_time_12 = None  # Second segment duration (to be optimized)
        self.delta_time_23 = None  # Third segment duration (computed from total)
        self.optimized_profile = None  # Will store the optimized profile

    def generate_thrust_profile(self, delta_time_01, delta_time_12, time_points):
        """Generate thrust profile for given switching times at specified time points."""
        profile = np.zeros_like(time_points)
        
        # First segment: positive thrust
        mask_segment_1 = time_points <= delta_time_01
        profile[mask_segment_1] = self.thrust_magnitude
        
        # Second segment: zero thrust
        mask_segment_2 = (time_points > delta_time_01) & (time_points <= delta_time_01 + delta_time_12)
        profile[mask_segment_2] = 0.0
        
        # Third segment: negative thrust
        mask_segment_3 = time_points > delta_time_01 + delta_time_12
        profile[mask_segment_3] = -self.thrust_magnitude
        
        return profile

    def generate_smooth_thrust_profile(self, delta_time_01, delta_time_12, time_points):
        """Generate a smooth thrust profile using tanh for smooth transitions."""
        k = 50  # Steepness factor for tanh, determines how sharp the transition is
        
        # Transition from +A to 0 at t1
        term1 = (self.thrust_magnitude / 2) * (1 - np.tanh(k * (time_points - delta_time_01)))
        
        # Transition from 0 to -A at t2
        t2 = delta_time_01 + delta_time_12
        term2 = (-self.thrust_magnitude / 2) * (1 + np.tanh(k * (time_points - t2)))
        
        # The initial state is +A, so we start with that and subtract the transitions
        profile = self.thrust_magnitude + (term1 - self.thrust_magnitude) + term2
        
        return profile

    def objective_function(self, delta_time_01, delta_time_12):
        """Compute squared error between thrust profile and smoother acceleration."""
        # Check bounds
        if delta_time_01 < 0 or delta_time_12 < 0 or delta_time_01 + delta_time_12 > self.total_time:
            return float('inf')  # Invalid configuration
        
        # Generate thrust profile at smoother time points
        model_thrust = self.generate_smooth_thrust_profile(delta_time_01, delta_time_12, np.array(self.smoother.time))
        
        # Compute squared error
        squared_error = np.sum((model_thrust - np.array(self.smoother.acc)) ** 2)

        # Add a small penalty for being out of bounds to guide the optimizer
        if delta_time_01 + delta_time_12 > self.total_time:
            squared_error += 1e6 * (delta_time_01 + delta_time_12 - self.total_time)**2

        return squared_error

    def objective_function_for_minimize(self, params):
        """Wrapper for scipy.optimize.minimize compatibility."""
        delta_time_01, delta_time_12 = params
        return self.objective_function(delta_time_01, delta_time_12)

    def grid_search(self):
        """Find optimal switching times using grid search."""
        # Define grid resolution
        num_points = 100
        dt01_values = np.linspace(0, self.total_time, num_points)
        dt12_values = np.linspace(0, self.total_time, num_points)
        
        # Initialize best values
        best_error = float('inf')
        best_dt01 = 0
        best_dt12 = 0
        
        # Track progress
        total_evaluations = num_points * num_points
        evaluations_done = 0
        
        print(f"  Starting grid search with {total_evaluations} evaluations...")
        
        # Grid search
        for dt01 in dt01_values:
            for dt12 in dt12_values:
                # Skip invalid combinations
                if dt01 + dt12 > self.total_time:
                    continue
                
                # Compute error
                error = self.objective_function(dt01, dt12)
                
                # Update if better
                if error < best_error:
                    best_error = error
                    best_dt01 = dt01
                    best_dt12 = dt12
                
                # Show progress occasionally
                evaluations_done += 1
                if evaluations_done % 1000 == 0 or evaluations_done == total_evaluations:
                    print(f"    Progress: {evaluations_done}/{total_evaluations} ({100*evaluations_done/total_evaluations:.1f}%)")
                    print(f"    Current best: dt01={best_dt01:.4f}, dt12={best_dt12:.4f}, error={best_error:.6f}")
        
        # Refine best solution with finer grid around the best point
        print(f"  Refining solution around best point...")
        dt01_range = 0.5  # Range around best point
        dt12_range = 0.5
        dt01_refine = np.linspace(max(0, best_dt01 - dt01_range), min(self.total_time, best_dt01 + dt01_range), num_points)
        dt12_refine = np.linspace(max(0, best_dt12 - dt12_range), min(self.total_time, best_dt12 + dt12_range), num_points)
        
        for dt01 in dt01_refine:
            for dt12 in dt12_refine:
                # Skip invalid combinations
                if dt01 + dt12 > self.total_time:
                    continue
                
                # Compute error
                error = self.objective_function(dt01, dt12)
                
                # Update if better
                if error < best_error:
                    best_error = error
                    best_dt01 = dt01
                    best_dt12 = dt12
        
        return best_dt01, best_dt12, best_error

    def refine_with_minimize(self, initial_guess, initial_error):
        """Refine the solution using scipy.optimize.minimize with Powell."""
        from scipy.optimize import minimize

        dt01_initial, dt12_initial = initial_guess
        print(f"  Refining solution with minimize(method='Powell')...")
        print(f"    Initial guess: dt01={dt01_initial:.4f}, dt12={dt12_initial:.4f}, error={initial_error:.6f}")

        # Define bounds for Powell
        bounds = [(0, self.total_time), (0, self.total_time)]

        try:
            result = minimize(
                self.objective_function_for_minimize,
                [dt01_initial, dt12_initial],
                method='Powell',
                bounds=bounds,
                options={'disp': True, 'maxiter': 1000, 'xtol': 1e-6, 'ftol': 1e-6}
            )

            if result.success:
                dt01, dt12 = result.x
                error = result.fun
                print(f"    Powell result: dt01={dt01:.4f}, dt12={dt12:.4f}, error={error:.6f}, iterations={result.nit}")
                return dt01, dt12, error
            else:
                print(f"    Powell optimization failed: {result.message}")
                return dt01_initial, dt12_initial, self.objective_function(dt01_initial, dt12_initial)
        except Exception as e:
            print(f"    Powell error: {str(e)}")
            return dt01_initial, dt12_initial, self.objective_function(dt01_initial, dt12_initial)

    def optimize(self):
        """Find optimal switching times using two-step approach:
           1. Grid search for robust initial guess
           2. Minimize for precise refinement
        """
        print("  Optimizing thrust profile (two-step approach)...")
        
        # Step 1: Grid search for initial guess
        print("  STEP 1: Grid search for initial guess")
        best_dt01, best_dt12, best_error = self.grid_search()

        # best_dt01, best_dt12 = 4.0, 4.0  # Uncomment to test Powell only
        
        # Step 2: Refine with minimize
        print("\n  STEP 2: Refine with Powell optimizer")
        refined_dt01, refined_dt12, refined_error = self.refine_with_minimize([best_dt01, best_dt12], best_error)
        
        # Use the best result (grid search or refined)
        if refined_error < best_error:
            self.delta_time_01 = refined_dt01
            self.delta_time_12 = refined_dt12
            final_error = refined_error
            print("  Using refined solution (better than grid search)")
        else:
            self.delta_time_01 = best_dt01
            self.delta_time_12 = best_dt12
            final_error = best_error
            print("  Using grid search solution (Powell did not improve)")
        
        self.delta_time_23 = self.total_time - self.delta_time_01 - self.delta_time_12
        
        # Generate final profile for plotting
        time_points = np.linspace(0, self.total_time, 1000)
        self.optimized_profile = {
            'time': time_points,
            'thrust': self.generate_thrust_profile(self.delta_time_01, self.delta_time_12, time_points)
        }
        
        print(f"\n  Optimization complete.")
        print(f"    delta_time_01    : {self.delta_time_01:.4f} s  (True value: {DELTA_TIME_01:.4f} s)")
        print(f"    delta_time_12    : {self.delta_time_12:.4f} s  (True value: {DELTA_TIME_12:.4f} s)")
        print(f"    delta_time_23    : {self.delta_time_23:.4f} s  (True value: {DELTA_TIME_23:.4f} s)")
        print(f"    Thrust magnitude : {self.thrust_magnitude:.4f} m/s²")
        print(f"    Final error      : {final_error:.6f}")
        
        return {
            'delta_time_01': self.delta_time_01,
            'delta_time_12': self.delta_time_12,
            'delta_time_23': self.delta_time_23,
            'error': final_error
        }


class ThrustProfileOptimizer2:
    """
    Second optimizer variant.

    Differences from ThrustProfileOptimizer:
    - Fixed segment times (t0, t1, t2, t3) provided as input (default: 0,2,8,10)
    - Fixed thrust direction pattern: +1, 0, -1 (provided as input)
    - Initial thrust magnitude guesses per segment (default: 0.05, 0.0, 0.05)
    - Inputs are the smoother solutions (pos, vel, acc, jerk) as arrays (not the smoother object)

    (No optimization logic added yet — just structure and profile generation.)
    """
    def __init__(
            self,
            smoother_pos,
            smoother_vel,
            smoother_acc,
            smoother_jerk,
            times=(0.0, 2.0, 8.0, 10.0),              # (t0, t1, t2, t3)
            thrust_acc_dir=(1.0, 0.0, -1.0),           # segment directions
            thrust_mag_init_guess=(0.05, 0.0, 0.05),   # initial magnitudes per segment
        ):
        # Store smoother solutions
        self.smoother_pos = np.array(smoother_pos)
        self.smoother_vel = np.array(smoother_vel)
        self.smoother_acc = np.array(smoother_acc)
        self.smoother_jerk = np.array(smoother_jerk)

        # Store segmentation
        if len(times) != 4:
            raise ValueError("times must be a tuple/list of length 4: (t0, t1, t2, t3)")
        self.t0, self.t1, self.t2, self.t3 = times
        if not (self.t0 < self.t1 < self.t2 < self.t3):
            raise ValueError("Require t0 < t1 < t2 < t3")

        # Store thrust directions and initial magnitudes
        if len(thrust_acc_dir) != 3:
            raise ValueError("thrust_acc_dir must have 3 entries (one per segment)")
        if len(thrust_mag_init_guess) != 3:
            raise ValueError("thrust_mag_init_guess must have 3 entries (one per segment)")
        self.thrust_acc_dir = np.array(thrust_acc_dir, dtype=float)
        self.thrust_mag_init = np.array(thrust_mag_init_guess, dtype=float)

        # Derived storage
        self.segment_times = (self.t0, self.t1, self.t2, self.t3)
        self.num_segments = 3

        # Placeholder for future optimization outputs
        self.opt_thrust_magnitudes = self.thrust_mag_init.copy()
        self.generated_profile = None  # dict with 'time' and 'thrust' when generated

    def generate_piecewise_profile(self, time_points):
        """
        Generate a piecewise-constant thrust acceleration profile using
        current (possibly still initial) magnitudes and fixed directions.
        """
        time_points = np.array(time_points)
        thrust = np.zeros_like(time_points)

        seg_bounds = [(self.t0, self.t1), (self.t1, self.t2), (self.t2, self.t3)]
        for i, (t_start, t_end) in enumerate(seg_bounds):
            mask = (time_points >= t_start) & (time_points <= t_end)
            thrust[mask] = self.opt_thrust_magnitudes[i] * self.thrust_acc_dir[i]

        self.generated_profile = {
            'time': time_points,
            'thrust': thrust
        }
        return self.generated_profile

    def objective_function(self, time_points):
        """
        Compute residual between generated thrust acceleration profile and smoother acceleration.
        Returns an array: (thrust_profile - smoother_acc_resampled)
        """
        time_points = np.array(time_points)
        profile = self.generate_piecewise_profile(time_points)
        thrust_acc = profile['thrust']
        smoother_acc = self.smoother_acc

        # If lengths differ, resample smoother acceleration uniformly to match
        if len(smoother_acc) != len(thrust_acc):
            orig_param = np.linspace(0.0, 1.0, len(smoother_acc))
            new_param = np.linspace(0.0, 1.0, len(thrust_acc))
            smoother_acc_resampled = np.interp(new_param, orig_param, smoother_acc)
        else:
            smoother_acc_resampled = smoother_acc

        residual = thrust_acc - smoother_acc_resampled
        return float(np.sum(residual**2))

    def _objective_magnitudes(self, magnitudes, time_points):
        """
        Internal objective: sum of squared residuals between a thrust profile
        built from 'magnitudes' (applied with fixed directions) and the smoother acceleration.
        Does NOT mutate stored optimal magnitudes during optimization iterations.
        """
        magnitudes = np.array(magnitudes, dtype=float)
        time_points = np.array(time_points)

        # Build thrust profile directly (avoid side effects)
        thrust = np.zeros_like(time_points)
        seg_bounds = [(self.t0, self.t1), (self.t1, self.t2), (self.t2, self.t3)]
        for i, (t_start, t_end) in enumerate(seg_bounds):
            mask = (time_points >= t_start) & (time_points <= t_end)
            thrust[mask] = magnitudes[i] * self.thrust_acc_dir[i]

        # Resample smoother acceleration if needed
        smoother_acc = self.smoother_acc
        if len(smoother_acc) != len(thrust):
            orig_param = np.linspace(0.0, 1.0, len(smoother_acc))
            new_param = np.linspace(0.0, 1.0, len(thrust))
            smoother_acc_resampled = np.interp(new_param, orig_param, smoother_acc)
        else:
            smoother_acc_resampled = smoother_acc

        residual = thrust - smoother_acc_resampled
        return float(np.sum(residual**2))

    def minimize_magnitudes(self, time_points=None, bounds=None):
        """
        Minimize objective over thrust acceleration magnitudes ONLY.
        Fixed:
          - segment times (t0,t1,t2,t3)
          - thrust directions (thrust_acc_dir)
        Initial guess: self.thrust_mag_init
        """
        from scipy.optimize import minimize

        if time_points is None:
            # Use uniform time grid matching smoother acceleration length
            time_points = np.linspace(self.t0, self.t3, len(self.smoother_acc))
        else:
            time_points = np.array(time_points, dtype=float)

        if bounds is None:
            # Non-negative magnitudes (direction supplies sign)
            bounds = [(0.0, None)] * self.num_segments

        x0 = self.thrust_mag_init.copy()

        result = minimize(
            lambda m: self._objective_magnitudes(m, time_points),
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            self.opt_thrust_magnitudes = result.x.astype(float)
            # Regenerate and store profile with optimized magnitudes
            self.generated_profile = self.generate_piecewise_profile(time_points)
            final_error = result.fun
        else:
            final_error = self._objective_magnitudes(self.opt_thrust_magnitudes, time_points)

        return {
            'success': bool(result.success),
            'message': result.message,
            'opt_thrust_magnitudes': self.opt_thrust_magnitudes,
            'final_error': final_error
        }

    def summary(self):
        print("ThrustProfileOptimizer2 Summary:")
        print(f"  Segment times : t0={self.t0}, t1={self.t1}, t2={self.t2}, t3={self.t3}")
        print(f"  Directions    : {self.thrust_acc_dir}")
        print(f"  Init magnitudes: {self.thrust_mag_init}")
        print(f"  Current magnitudes: {self.opt_thrust_magnitudes}")


# Update PlotResults class to include the optimizer
class PlotResults:
    def __init__(
            self,
            simulator: SimulatePathAndMeasurements,
            thrust_estimator: ThrustEstimator,
            sequential_filter: SequentialFilter,
            smoother: Smoother,
            optimizer: ThrustProfileOptimizer = None,
            optimizer2: 'ThrustProfileOptimizer2' = None,  # NEW
    ):
        self.simulator = simulator
        self.thrust_estimator = thrust_estimator
        self.sequential_filter = sequential_filter
        self.smoother = smoother
        self.optimizer = optimizer
        self.optimizer2 = optimizer2  # NEW

    def plot_all(self):
        print("  Plot trajectory, measurements, and thrust acceleration")
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # Position plot
        axs[0].plot(self.simulator.body.trajectory.time, self.simulator.body.trajectory.position, label='Position [m]', color=mcolors.TABLEAU_COLORS['tab:blue'], linewidth=4)
        axs[0].plot(self.simulator.body.measurements.time, self.simulator.body.measurements.range, 'o', label='Range Measurements [m]', color=mcolors.TABLEAU_COLORS['tab:blue'], markersize=4, alpha=0.5)
        axs[0].plot(self.sequential_filter.time, self.sequential_filter.pos, '--', label='Filtered Position [m]', color='red', linewidth=2)
        axs[0].plot(self.smoother.time, self.smoother.pos, '-.', label='Smoothed Position [m]', color='purple', linewidth=2)
        axs[0].set_ylabel('Position [m]')
        axs[0].tick_params(axis='x', length=0)
        axs[0].grid()
        axs[0].legend()

        # Velocity plot
        axs[1].plot(self.simulator.body.trajectory.time, self.simulator.body.trajectory.velocity, label='True Velocity [m/s]', color=mcolors.TABLEAU_COLORS['tab:orange'], linewidth=4)
        axs[1].plot(self.simulator.body.measurements.time, self.simulator.body.measurements.range_rate, 'o', label='Range Rate Measurements [m/s]', color=mcolors.TABLEAU_COLORS['tab:orange'], markersize=4, alpha=0.5)
        axs[1].plot(self.sequential_filter.time, self.sequential_filter.vel, '--', label='Filtered Velocity [m/s]', color='red', linewidth=2)
        axs[1].plot(self.smoother.time, self.smoother.vel, '-.', label='Smoothed Velocity [m/s]', color='purple', linewidth=2)
        axs[1].set_ylabel('Velocity [m/s]')
        axs[1].set_ylabel('Velocity [m/s]')
        axs[1].tick_params(axis='x', length=0)
        axs[1].grid()
        axs[1].legend()

        # Acceleration plot
        axs[2].plot(self.simulator.body.trajectory.time, self.simulator.body.trajectory.thrust_acceleration, label='True Thrust Accel [m/s²]', color=mcolors.TABLEAU_COLORS['tab:green'], linewidth=4)
        axs[2].plot(self.simulator.body.trajectory.time[:-1], self.thrust_estimator.approx_thrust_acceleration_true[:-1], '-', label='Approx Thrust Accel (True) [m/s²]', color=mcolors.TABLEAU_COLORS['tab:green'], linewidth=12, alpha=0.5)
        axs[2].plot(self.sequential_filter.time, self.sequential_filter.acc, '--', label='Estimated Accel (Filter) [m/s²]', color='red', linewidth=2)
        axs[2].plot(self.smoother.time, self.smoother.acc, '-.', label='Estimated Accel (Smoother) [m/s²]', color='purple', linewidth=2)
        
        # Add optimized thrust profile
        if self.optimizer and hasattr(self.optimizer, 'optimized_profile') and self.optimizer.optimized_profile:
            axs[2].plot(self.optimizer.optimized_profile['time'], self.optimizer.optimized_profile['thrust'], 
                     '-', label='Optimized Thrust Profile [m/s²]', color='black', linewidth=2)
            
            # Add vertical lines at the switching times
            if hasattr(self.optimizer, 'delta_time_01') and self.optimizer.delta_time_01 is not None:
                axs[2].axvline(x=self.optimizer.delta_time_01, color='black', linestyle='--', alpha=0.5)
                axs[2].axvline(x=self.optimizer.delta_time_01 + self.optimizer.delta_time_12, color='black', linestyle='--', alpha=0.5)
        
        axs[2].set_ylabel('Acceleration [m/s²]')
        axs[2].tick_params(axis='x', length=0)
        axs[2].grid()
        axs[2].legend()

        # Jerk plot
        axs[3].plot(self.smoother.time, self.smoother.jerk, '-.', label='Smoothed Jerk [m/s³]', color='cyan', linewidth=2)
        axs[3].set_xlabel('Time [s]')
        axs[3].set_ylabel('Jerk [m/s³]')
        axs[3].grid()
        axs[3].legend()
        
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        # plt.show()

    def plot_optimizer2(self):
        """
        Plot ThrustProfileOptimizer2 result in a separate figure.
        """
        if self.optimizer2 is None or self.optimizer2.generated_profile is None:
            print("  Optimizer2 profile not available to plot.")
            return
        print("  Plot Optimizer2 thrust profile (separate figure)")
        plt.figure(figsize=(10,4))
        prof = self.optimizer2.generated_profile
        plt.plot(prof['time'], prof['thrust'], label='Optimizer2 Thrust [m/s²]', color='black', linewidth=3)
        plt.plot(self.smoother.time, self.smoother.acc, label='Smoothed Acc [m/s²]', color='purple', alpha=0.6)
        # Segment boundaries
        for x in [self.optimizer2.t1, self.optimizer2.t2]:
            plt.axvline(x=x, color='gray', linestyle='--', alpha=0.7)
        plt.title("ThrustProfileOptimizer2 Result")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [m/s²]")
        mag_str = ", ".join(f"{m:.4f}" for m in self.optimizer2.opt_thrust_magnitudes)
        plt.text(0.02, 0.95, f"Magnitudes: {mag_str}", transform=plt.gca().transAxes, fontsize=9, va='top')
        plt.grid()
        plt.legend()
        plt.tight_layout()
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
    print("\nSEQUENTIAL FILTER")
    sequential_filter = SequentialFilter(simulator)
    sequential_filter.run()

    # Run smoother
    print("\nSMOOTHER")
    smoother = Smoother(sequential_filter)
    smoother.run()

    # Helper to report all significant |jerk| peaks
    def report_jerk_peaks(time_arr, jerk_arr, label, rel_height=0.70, min_separation_sec=0.25):
        time_arr = np.asarray(time_arr, dtype=float)
        jerk_arr = np.asarray(jerk_arr, dtype=float)
        if len(jerk_arr) == 0:
            return
        abs_jerk = np.abs(jerk_arr)
        max_abs = abs_jerk.max()
        if max_abs == 0.0:
            return
        # Height threshold relative to global max
        height_thresh = rel_height * max_abs

        # Convert desired temporal separation into sample count
        if len(time_arr) > 1:
            dt_med = np.median(np.diff(time_arr))
            distance = max(1, int(np.round(min_separation_sec / dt_med)))
        else:
            distance = 1

        peaks, props = find_peaks(abs_jerk, height=height_thresh, distance=distance)

        print(f"\n{label} |jerk| peak detection")
        print(f"  Global max |jerk| = {max_abs:.6e}")
        print(f"  Threshold  (rel {rel_height:.2f}) = {height_thresh:.6e}")
        if len(peaks) == 0:
            print("  No peaks above threshold.")
            return
        for i, idx in enumerate(peaks):
            print(f"  Peak {i+1:02d}: t = {time_arr[idx]:.6f} s  jerk = {jerk_arr[idx]:+.6e} m/s^3  |jerk| = {abs_jerk[idx]:.6e}")

    # Existing single max reports (keep)
    if sequential_filter.jerk:
        filt_jerk_arr = np.array(sequential_filter.jerk, dtype=float)
        idx_f = int(np.argmax(np.abs(filt_jerk_arr)))
        print("\nMAX ABS JERK (Filter)")
        print(f"  Time: {sequential_filter.time[idx_f]:.6f} s  "
              f"Jerk: {filt_jerk_arr[idx_f]:.6e} m/s^3  "
              f"|jerk|: {abs(filt_jerk_arr[idx_f]):.6e}")
        # NEW: all significant peaks
        report_jerk_peaks(sequential_filter.time, sequential_filter.jerk, "FILTER")

    if smoother.jerk:
        sm_jerk_arr = np.array(smoother.jerk, dtype=float)
        idx_s = int(np.argmax(np.abs(sm_jerk_arr)))
        print("\nMAX ABS JERK (Smoother)")
        print(f"  Time: {smoother.time[idx_s]:.6f} s  "
              f"Jerk: {sm_jerk_arr[idx_s]:.6e} m/s^3  "
              f"|jerk|: {abs(sm_jerk_arr[idx_s]):.6e}")
        # NEW: all significant peaks (will capture those near ~2 s and ~8 s)
        report_jerk_peaks(smoother.time, smoother.jerk, "SMOOTHER")

    # Approximate thrust profile
    print("\nAPPROXIMATE THRUST PROFILE")
    thrust_estimator = ThrustEstimator(simulator, sequential_filter, smoother)
    thrust_estimator.estimate_thrust()
    
    # Optimize thrust profile
    print("\nOPTIMIZE THRUST PROFILE")
    optimizer = ThrustProfileOptimizer(smoother, thrust_magnitude=0.0625)
    optimal_params = optimizer.optimize()

    # NEW: Optimize magnitudes only with ThrustProfileOptimizer2
    print("\nOPTIMIZE THRUST MAGNITUDES (Optimizer2)")
    optimizer2 = ThrustProfileOptimizer2(
        smoother_pos=smoother.pos,
        smoother_vel=smoother.vel,
        smoother_acc=smoother.acc,
        smoother_jerk=smoother.jerk,
        times=(0.0, 2.0, 8.0, 10.0),
        thrust_acc_dir=(1.0, 0.0, -1.0),
        thrust_mag_init_guess=(0.05, 0.0, 0.05),
    )
    opt2_result = optimizer2.minimize_magnitudes()
    print(f"  Optimizer2 result: success={opt2_result['success']}, "
          f"magnitudes={opt2_result['opt_thrust_magnitudes']}, "
          f"final_error={opt2_result['final_error']:.6e}")

    # Plot results
    print("\nPLOT RESULTS") 
    print("\nPLOT RESULTS") 
    plotter = PlotResults(simulator, thrust_estimator, sequential_filter, smoother, optimizer, optimizer2)
    plotter.plot_all()
    plotter.plot_optimizer2()

    # End thrust estimation program
    print("\nTHRUST ESTIMATION PROGRAM COMPLETE\n\n")
    return True

if __name__ == '__main__':
    main()

