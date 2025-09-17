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

PCT_THIN = 0.5

MEASUREMENT_RANGE_NOISE_STD      = 0.050  # m
MEASUREMENT_RANGE_RATE_NOISE_STD = 0.001  # m/s
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
        # Extend state to include acceleration [pos, vel, acc]
        self.state_est_hat = np.array([0.0, 0.0, 0.0]) # initial state estimate [pos, vel, acc] [m, m/s, m/s^2]
        self.P = np.eye(3) # initial covariance estimate [m^2, (m/s)^2, (m/s^2)^2]
        self.process_noise_std = PROCESS_NOISE_STD # std dev of process noise (unmodeled jerk) [m/s^3]

        # Store results
        self.time = []
        self.pos  = []
        self.vel  = []
        self.acc  = []  # Add acceleration storage
        self.x_hat_history = []
        self.P_history = []
        self.x_hat_minus_history = []
        self.P_minus_history = []
        self.F_history = []

    def run(self):
        print("  Run Sequential Filter")
        measurements = self.simulator.body.measurements
    
        # Use first measurement to initialize state
        # Estimate initial acceleration as 0
        self.x_hat = np.array([measurements.range[0], measurements.range_rate[0], 0.0])
        self.time.append(measurements.time[0])
        self.pos.append(self.x_hat[0])
        self.vel.append(self.x_hat[1])
        self.acc.append(self.x_hat[2])
        self.x_hat_history.append(self.x_hat)
        self.P_history.append(self.P)

        # No prediction for first step, so add placeholders
        self.x_hat_minus_history.append(self.x_hat)
        self.P_minus_history.append(self.P)
        self.F_history.append(np.eye(3))

        measurement_range_noise_std      = MEASUREMENT_RANGE_NOISE_STD
        measurement_range_rate_noise_std = MEASUREMENT_RANGE_RATE_NOISE_STD
        R = np.diag([measurement_range_noise_std**2, measurement_range_rate_noise_std**2]) # measurement noise covariance
        # Update H to map from 3D state to 2D measurements
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # measurement matrix

        for k in range(1, len(measurements.time)):

            # Prediction
            dt = measurements.time[k] - measurements.time[k-1] # time step
            # Update F for constant acceleration model
            F = np.array([
                [1.0, dt, 0.5*dt**2],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0]
            ]) # state transition matrix
            
            # Process noise affects acceleration directly (jerk model)
            G = np.array([[0.5*dt**3], [dt**2], [dt]]) # process noise gain matrix
            Q = G @ G.T * self.process_noise_std**2 # process noise covariance

            x_hat_minus = F @ self.x_hat
            P_minus = F @ self.P @ F.T + Q

            # Update
            z = np.array([measurements.range[k], measurements.range_rate[k]]) # measurement vector
            y = z - H @ x_hat_minus # measurement residual
            S = H @ P_minus @ H.T + R # innovation covariance
            K = P_minus @ H.T @ np.linalg.inv(S) # gain

            self.x_hat = x_hat_minus + K @ y # updated state estimate
            self.P = (np.eye(3) - K @ H) @ P_minus # updated covariance estimate

            # Store results
            self.time.append(measurements.time[k])
            self.pos.append(self.x_hat[0])
            self.vel.append(self.x_hat[1])
            self.acc.append(self.x_hat[2])  # Store acceleration estimate
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
        self.acc = []  # Add acceleration storage

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
        self.acc = [x[2] for x in x_hat_s]  # Extract acceleration estimates


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
        model_thrust = self.generate_smooth_thrust_profile(delta_time_01, delta_time_12, self.smoother.time)
        
        # Compute squared error
        squared_error = np.sum((model_thrust - self.smoother.acc) ** 2)

        # Add a small penalty for being out of bounds to guide the optimizer
        if delta_time_01 + delta_time_12 > self.total_time:
            squared_error += 1e6 * (delta_time_01 + delta_time_12 - self.total_time)**2

        return squared_error

    def residuals_function(self, params):
        """Compute residuals for least-squares optimization."""
        delta_time_01, delta_time_12 = params
        
        # Generate thrust profile at smoother time points
        model_thrust = self.generate_thrust_profile(delta_time_01, delta_time_12, self.smoother.time)
        
        # Compute residuals
        residuals = model_thrust - self.smoother.acc
        return residuals

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

    def _refine_with_nelder_mead(self, initial_guess):
        """Refine the solution using Nelder-Mead optimization. (UNUSED)"""
        from scipy.optimize import minimize
        
        dt01_initial, dt12_initial = initial_guess
        
        print(f"  Refining solution with Nelder-Mead optimization...")
        print(f"    Initial guess: dt01={dt01_initial:.4f}, dt12={dt12_initial:.4f}")
        
        # Define penalized objective function used by methods that don't support constraints
        def penalized_objective(params):
            dt01, dt12 = params
            
            # Apply soft bounds
            dt01 = max(0, min(dt01, self.total_time))
            dt12 = max(0, min(dt12, self.total_time))
            
            error = self.objective_function(dt01, dt12)
            # # Add penalty if constraint is violated
            # if dt01 + dt12 > self.total_time:
            #     penalty = 1e6 * ((dt01 + dt12) - self.total_time)**2
            #     return error + penalty
            return error
        
        # Run Nelder-Mead optimization
        try:
            result = minimize(
                penalized_objective,
                [dt01_initial, dt12_initial],
                method='Powell', # 'Nelder-Mead' 'Powell'
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'disp': True,
                    'adaptive': True  # Use adaptive parameters
                }
            )
            
            if result.success:
                dt01, dt12 = result.x
                # Enforce bounds
                dt01 = max(0, min(dt01, self.total_time))
                dt12 = max(0, min(dt12, self.total_time - dt01))
                error = self.objective_function(dt01, dt12)
                
                print(f"    Nelder-Mead result: dt01={dt01:.4f}, dt12={dt12:.4f}, error={error:.6f}, iterations={result.nit}")
                return dt01, dt12, error
            else:
                print(f"    Nelder-Mead optimization failed: {result.message}")
                return dt01_initial, dt12_initial, self.objective_function(dt01_initial, dt12_initial)
        except Exception as e:
            print(f"    Nelder-Mead error: {str(e)}")
            return dt01_initial, dt12_initial, self.objective_function(dt01_initial, dt12_initial)

    def _refine_with_lm(self, initial_guess):
        """Refine the solution using Levenberg-Marquardt optimization. (UNUSED)"""
        from scipy.optimize import least_squares
        
        dt01_initial, dt12_initial = initial_guess
        
        print(f"  Refining solution with Levenberg-Marquardt (LM) optimization...")
        print(f"    Initial guess: dt01={dt01_initial:.4f}, dt12={dt12_initial:.4f}")
        
        # Define bounds for the variables
        bounds = ([0, 0], [self.total_time, self.total_time])
        
        # Average time step between measurements, used for finite difference step
        avg_dt = np.mean(np.diff(self.smoother.time))

        # Run Levenberg-Marquardt optimization
        try:
            result = least_squares(
                self.residuals_function,
                [dt01_initial, dt12_initial],
                method='lm',
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=5000,
                diff_step=10.0*avg_dt,  # Use a larger step for finite differencing
            )
            if result.success:
                dt01, dt12 = result.x
                
                # Enforce constraints after optimization
                if dt01 + dt12 > self.total_time:
                    # A simple strategy if constraint is violated: scale down
                    scale = self.total_time / (dt01 + dt12)
                    dt01 *= scale
                    dt12 *= scale
                
                dt01 = max(0, dt01)
                dt12 = max(0, dt12)

                error = self.objective_function(dt01, dt12)
                
                print(f"    LM result: dt01={dt01:.4f}, dt12={dt12:.4f}, error={error:.6f}, iterations={result.nfev}")
                return dt01, dt12, error
            else:
                print(f"    LM optimization failed: {result.message}")
                return dt01_initial, dt12_initial, self.objective_function(dt01_initial, dt12_initial)
        except Exception as e:
            print(f"    LM error: {str(e)}")
            return dt01_initial, dt12_initial, self.objective_function(dt01_initial, dt12_initial)

    def _refine_with_differential_evolution(self, initial_guess):
        """Refine the solution using differential evolution. (UNUSED)"""
        from scipy.optimize import differential_evolution, LinearConstraint

        print(f"  Refining solution with Differential Evolution...")
        
        # Bounds for the variables [dt01, dt12]
        bounds = [(0, self.total_time), (0, self.total_time)]

        # Constraint: dt01 + dt12 <= total_time
        constraint = LinearConstraint([1, 1], -np.inf, self.total_time)

        try:
            result = differential_evolution(
                self.objective_function_for_minimize,
                bounds,
                constraints=(constraint),
                seed=42,
                disp=True,
                polish=True,
            )

            if result.success:
                dt01, dt12 = result.x
                error = result.fun
                print(f"    Differential Evolution result: dt01={dt01:.4f}, dt12={dt12:.4f}, error={error:.6f}, iterations={result.nfev}")
                return dt01, dt12, error
            else:
                print(f"    Differential Evolution optimization failed: {result.message}")
                return initial_guess[0], initial_guess[1], self.objective_function(initial_guess[0], initial_guess[1])
        except Exception as e:
            print(f"    Differential Evolution error: {str(e)}")
            return initial_guess[0], initial_guess[1], self.objective_function(initial_guess[0], initial_guess[1])

    def _refine_with_cobyla(self, initial_guess):
        """Refine the solution using scipy.optimize.minimize with COBYLA. (UNUSED)"""
        from scipy.optimize import minimize

        dt01_initial, dt12_initial = initial_guess
        print(f"  Refining solution with minimize(method='COBYLA')...")
        print(f"    Initial guess: dt01={dt01_initial:.4f}, dt12={dt12_initial:.4f}")

        # Define constraints for COBYLA. 'ineq' means c(x) >= 0.
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},  # dt01 >= 0
            {'type': 'ineq', 'fun': lambda x: x[1]},  # dt12 >= 0
            {'type': 'ineq', 'fun': lambda x: self.total_time - x[0] - x[1]}  # dt01 + dt12 <= total_time
        ]

        try:
            result = minimize(
                self.objective_function_for_minimize,
                [dt01_initial, dt12_initial],
                method='COBYLA',
                constraints=constraints,
                options={'disp': True, 'maxiter': 5000, 'rhobeg': 0.1}
            )

            if result.success:
                dt01, dt12 = result.x
                error = result.fun
                print(f"    COBYLA result: dt01={dt01:.4f}, dt12={dt12:.4f}, error={error:.6f}, iterations={result.nfev}")
                return dt01, dt12, error
            else:
                print(f"    COBYLA optimization failed: {result.message}")
                return dt01_initial, dt12_initial, self.objective_function(dt01_initial, dt12_initial)
        except Exception as e:
            print(f"    COBYLA error: {str(e)}")
            return dt01_initial, dt12_initial, self.objective_function(dt01_initial, dt12_initial)

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
                options={'disp': True, 'maxiter': 5000, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            breakpoint()
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
        best_dt01, best_dt12 = 3,3
        
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


# Update PlotResults class to include the optimizer
class PlotResults:
    def __init__(
            self,
            simulator: SimulatePathAndMeasurements,
            thrust_estimator: ThrustEstimator,
            sequential_filter: SequentialFilter,
            smoother: Smoother,
            optimizer: ThrustProfileOptimizer = None,
    ):
        self.simulator = simulator
        self.thrust_estimator = thrust_estimator
        self.sequential_filter = sequential_filter
        self.smoother = smoother
        self.optimizer = optimizer

    def plot_all(self):
        print("  Plot trajectory, measurements, and thrust acceleration")
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

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
        if self.optimizer and hasattr(self.optimizer, 'optimized_profile'):
            axs[2].plot(self.optimizer.optimized_profile['time'], self.optimizer.optimized_profile['thrust'], 
                     '-', label='Optimized Thrust Profile [m/s²]', color='black', linewidth=2)
            
            # Add vertical lines at the switching times
            if hasattr(self.optimizer, 'delta_time_01'):
                axs[2].axvline(x=self.optimizer.delta_time_01, color='black', linestyle='--', alpha=0.5)
                axs[2].axvline(x=self.optimizer.delta_time_01 + self.optimizer.delta_time_12, color='black', linestyle='--', alpha=0.5)
        
        axs[2].set_xlabel('Time [s]')
        axs[2].set_ylabel('Acceleration [m/s²]')
        axs[2].grid()
        axs[2].legend()
        
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
    print("\nSEQUENTIAL FILTER")
    sequential_filter = SequentialFilter(simulator)
    sequential_filter.run()

    # Run smoother
    print("\nSMOOTHER")
    smoother = Smoother(sequential_filter)
    smoother.run()

    # Approximate thrust profile
    print("\nAPPROXIMATE THRUST PROFILE")
    thrust_estimator = ThrustEstimator(simulator, sequential_filter, smoother)
    thrust_estimator.estimate_thrust()
    
    # Optimize thrust profile
    print("\nOPTIMIZE THRUST PROFILE")
    optimizer = ThrustProfileOptimizer(smoother, thrust_magnitude=0.0625)
    optimal_params = optimizer.optimize()

    # Plot results
    print("\nPLOT RESULTS") 
    plotter = PlotResults(simulator, thrust_estimator, sequential_filter, smoother, optimizer)
    plotter.plot_all()

    # End thrust estimation program
    print("\nTHRUST ESTIMATION PROGRAM COMPLETE\n\n")
    return True

if __name__ == '__main__':
    main()

