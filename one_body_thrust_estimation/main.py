# Imports
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Trajectory:
    n_steps: int = 0 # number of time steps
    _time: Optional[np.ndarray] = None # time array [s]
    _position: Optional[np.ndarray] = None # position array [m]
    _velocity: Optional[np.ndarray] = None # velocity array [m/s]
    _acceleration: Optional[np.ndarray] = None # acceleration array [m/s^2]
    _thrust_acceleration: Optional[np.ndarray] = None # thrust acceleration array [m/s^2]

    def _validate_and_update(self, new_array, field_name):
        """Validate new array and update n_steps if needed"""
        if new_array is not None:
            new_length = len(new_array)
            
            # Get lengths of existing arrays
            existing_arrays = {}
            if self._time is not None and field_name != 'time':
                existing_arrays['time'] = len(self._time)
            if self._position is not None and field_name != 'position':
                existing_arrays['position'] = len(self._position)
            if self._velocity is not None and field_name != 'velocity':
                existing_arrays['velocity'] = len(self._velocity)
            if self._acceleration is not None and field_name != 'acceleration':
                existing_arrays['acceleration'] = len(self._acceleration)
            if self._thrust_acceleration is not None and field_name != 'thrust_acceleration':
                existing_arrays['thrust_acceleration'] = len(self._thrust_acceleration)
            
            # Check consistency with existing arrays
            if existing_arrays:
                existing_lengths = list(existing_arrays.values())
                if not all(length == new_length for length in existing_lengths):
                    raise ValueError(f"Array length mismatch: {field_name} has length {new_length}, "
                                   f"but existing arrays have lengths {existing_arrays}")
            
            # Update n_steps
            self.n_steps = new_length

    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, value):
        self._validate_and_update(value, 'time')
        self._time = value

    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._validate_and_update(value, 'position')
        self._position = value

    @property
    def velocity(self):
        return self._velocity
    
    @velocity.setter
    def velocity(self, value):
        self._validate_and_update(value, 'velocity')
        self._velocity = value

    @property
    def acceleration(self):
        return self._acceleration
    
    @acceleration.setter
    def acceleration(self, value):
        self._validate_and_update(value, 'acceleration')
        self._acceleration = value

    @property
    def thrust_acceleration(self):
        return self._thrust_acceleration
    
    @thrust_acceleration.setter
    def thrust_acceleration(self, value):
        self._validate_and_update(value, 'thrust_acceleration')
        self._thrust_acceleration = value

    def __post_init__(self):
        # Collect all provided arrays and their lengths
        provided_arrays = {}
        if self._time is not None:
            provided_arrays['time'] = len(self._time)
        if self._position is not None:
            provided_arrays['position'] = len(self._position)
        if self._velocity is not None:
            provided_arrays['velocity'] = len(self._velocity)
        if self._acceleration is not None:
            provided_arrays['acceleration'] = len(self._acceleration)
        if self._thrust_acceleration is not None:
            provided_arrays['thrust_acceleration'] = len(self._thrust_acceleration)
        
        # Check consistency
        if provided_arrays:
            array_lengths = list(provided_arrays.values())
            # Check all provided arrays have the same length
            if not all(length == array_lengths[0] for length in array_lengths):
                raise ValueError(f"Inconsistent array lengths: {provided_arrays}")
            
            # Check if n_steps matches first provided array length if n_steps > 0. 
            # Already guaranteed all lengths are equal.
            if self.n_steps > 0 and self.n_steps != array_lengths[0]:
                raise ValueError(f"n_steps ({self.n_steps}) doesn't match array length ({array_lengths[0]})")
            
            # Update n_steps to match provided arrays
            self.n_steps = array_lengths[0]
        
        # Initialize only None arrays
        if self.n_steps > 0:
            if self._time is None:
                self._time = np.zeros(self.n_steps)
            if self._position is None:
                self._position = np.zeros(self.n_steps)
            if self._velocity is None:
                self._velocity = np.zeros(self.n_steps)
            if self._acceleration is None:
                self._acceleration = np.zeros(self.n_steps)
            if self._thrust_acceleration is None:
                self._thrust_acceleration = np.zeros(self.n_steps)
        else:
            # Handle zero case - only set None arrays to empty
            if self._time is None:
                self._time = np.array([])
            if self._position is None:
                self._position = np.array([])
            if self._velocity is None:
                self._velocity = np.array([])
            if self._acceleration is None:
                self._acceleration = np.array([])
            if self._thrust_acceleration is None:
                self._thrust_acceleration = np.array([])


class SimulatePathAndMeasurements:
    def __init__(self):
        self.body = Trajectory()

    def create_path(self):
        # Time parameters
        n_steps = 100
        time_o = 0.0
        time_f = 10.0
        self.body.time = np.linspace(time_o, time_f, n_steps)

        # Thrust acceleration parameters (on-off-on)
        thrust_acc_mag = 0.2  # m/s^2
        delta_time_01 = 2.0  # s, thrust-acc on
        delta_time_12 = 6.0  # s, thrust-acc off
        delta_time_23 = time_f - time_o - delta_time_01 - delta_time_12  # s, thrust-acc on, unused
        
        # Create thrust profile
        breakpoint()
        for i, t in enumerate(self.body.time):
            if t <= delta_time_01:
                self.body.acceleration[i] = thrust_acc_mag  # thrust on
            elif t <= delta_time_01 + delta_time_12:
                self.body.acceleration[i] = 0.0  # thrust off
            else:
                self.body.acceleration[i] = -thrust_acc_mag  # thrust on
        
        # Integrate to get velocity and position
        for i in range(1, n_steps):
            self.body.velocity[i] = self.body.velocity[i-1] + self.body.acceleration[i-1] * dt
            self.body.position[i] = self.body.position[i-1] + self.body.velocity[i-1] * dt
        
        # Scale position to go from 0 to 1 meter
        self.body.position = self.body.position / np.max(self.body.position)

        print(f"Path created: {self.body.n_steps} time steps from 0 to {self.body.duration}s")
        print(f"Position range: {np.min(self.body.position):.3f} to {np.max(self.body.position):.3f} meters")

    def create_measurements(self):
        pass


def main():
    
    # Start thrust estimation program
    print(2*"\n"+"==========================")
    print(       "THRUST ESTIMATION PROGRAM")
    print(       "==========================")

    # Simulate path and measurements
    print("\nSIMULATE PATH AND MEASUREMENTS")
    simulator = SimulatePathAndMeasurements()
    simulator.create_path()
    simulator.create_measurements()

    # Run sequential filter
    print("\nRUN SEQUENTIAL FILTER")

    # Run smoother
    print("\nRUN SMOOTHER")

    # Approximate thrust profile
    print("\nAPPROXIMATE THRUST PROFILE")

    # End thrust estimation program
    print("\nTHRUST ESTIMATION PROGRAM COMPLETE")
    return True

if __name__ == '__main__':
    main()

