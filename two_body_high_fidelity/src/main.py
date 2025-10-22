import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from constants import CONVERTER, TIMEVALUES
from model.dynamics import TwoBodyDynamics, PHYSICALCONSTANTS, EquationsOfMotion
from plot.trajectory import plot_3d_trajectories, plot_time_series
from model.coordinate_system_converter import CoordinateSystemConverter
from initialization import initial_guess

def propagate_orbit(
    initial_state       : np.ndarray,
    time_o              : float,
    time_f              : float,
    dynamics            : TwoBodyDynamics,
    method              : str = 'DOP853',
    rtol                : float = 1e-12,
    atol                : float = 1e-12,
    get_coe_time_series : bool = False,
) -> dict:
    """
    Propagate an orbit from initial cartesian state.
    """
    # Time span for integration
    time_span = (time_o, time_f)

    # Solve initial value problem
    solution = solve_ivp(
        fun          = EquationsOfMotion(dynamics).state_time_derivative,
        t_span       = time_span,
        y0           = initial_state,
        method       = method,
        rtol         = rtol,
        atol         = atol,
        dense_output = True,
    )
    
    # Convert all states to classical orbital elements
    num_steps = solution.y.shape[1]
    coe_time_series = {
        'sma'  : np.zeros(num_steps),
        'ecc'  : np.zeros(num_steps),
        'inc'  : np.zeros(num_steps),
        'raan' : np.zeros(num_steps),
        'argp' : np.zeros(num_steps),
        'ma'   : np.zeros(num_steps),
        'ta'   : np.zeros(num_steps),
        'ea'   : np.zeros(num_steps),
    }
    if get_coe_time_series:
        coord_sys_converter = CoordinateSystemConverter(dynamics.gp)
        for i in range(num_steps):
            pos = solution.y[0:3, i]
            vel = solution.y[3:6, i]
            coe = coord_sys_converter.rv2coe(pos, vel)
            
            for key in coe_time_series.keys():
                coe_time_series[key][i] = coe[key]
    
    return {
        'success'     : solution.success,
        'message'     : solution.message,
        'time'        : solution.t,
        'state'       : solution.y,
        'final_state' : solution.y[:, -1],
        'coe'         : coe_time_series,
    }

def main():
    """
    Main function to set up and propagate a spacecraft orbit for one day.
    """
    #### INPUT ####

    # Time
    time_o = 0.0                          # initial time [s]
    time_f = time_o + 10 * TIMEVALUES.ONE_DAY  # final time [s]

    # Initial state
    igs = 'circular'               # initial guess selection
    alt = 500e3                    # altitude [m]
    inc = 45.0 * CONVERTER.DEG2RAD # inclination [rad]

    #### INPUT ####

    # Initial state
    #   initial_guess_selection : 'circular'
    initial_state = initial_guess.get_initial_state(
        initial_guess_selection = igs,
        altitude                = alt,
        inclination             = inc,
    )
    
    # Set up dynamics model for Earth with J2 perturbation
    earth_dynamics = TwoBodyDynamics(
        gp      = PHYSICALCONSTANTS.EARTH.GP,
        time_o  = time_o,
        j_2     = PHYSICALCONSTANTS.EARTH.J_2,
        j_3     = PHYSICALCONSTANTS.EARTH.J_3,
        j_4     = PHYSICALCONSTANTS.EARTH.J_4,
        pos_ref = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR
    )
    
    # Propagate the orbit
    result = propagate_orbit(
        initial_state       = initial_state,
        time_o              = time_o,
        time_f              = time_f,
        dynamics            = earth_dynamics,
        get_coe_time_series = True,
    )

    # Display results
    if result['success']:
        print(f"\nPropagation successful!")
        print(f"Status: {result['message']}")
        print(f"Number of time steps: {len(result['time'])}")
        
        final_state = result['final_state']
        print(f"\nFinal State (Inertial Frame):")
        print(f"  Position: [{final_state[0]:.3f}, {final_state[1]:.3f}, {final_state[2]:.3f}] m")
        print(f"  Velocity: [{final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}] m/s")
        
        # Compute orbital parameters
        r_initial = np.linalg.norm(initial_state[0:3])
        r_final   = np.linalg.norm(final_state[0:3])
        v_initial = np.linalg.norm(initial_state[3:6])
        v_final   = np.linalg.norm(final_state[3:6])
        
        print(f"\nOrbital Magnitude Comparison:")
        print(f"  Initial radius: {r_initial:.3f} m")
        print(f"  Final radius:   {r_final:.3f} m")
        print(f"  Difference:     {r_final - r_initial:.3f} m")
        print(f"\n  Initial speed:  {v_initial:.3f} m/s")
        print(f"  Final speed:    {v_final:.3f} m/s")
        print(f"  Difference:     {v_final - v_initial:.3f} m/s")
        
        # Create plots
        print("\nGenerating plots...")
        plot_3d_trajectories(result)
        plot_time_series(result)
        plt.show()
    else:
        print(f"\nPropagation failed!")
        print(f"Status: {result['message']}")
    
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()