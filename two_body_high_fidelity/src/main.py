import numpy as np
from scipy.integrate import solve_ivp
from constants import Converter
from model.dynamics import TwoBodyDynamics, PHYSICALCONSTANTS, EquationsOfMotion


def propagate_orbit(
    initial_state: np.ndarray,
    propagation_time: float,
    dynamics: TwoBodyDynamics,
    method: str = 'DOP853',
    rtol: float = 1e-12,
    atol: float = 1e-12
) -> dict:
    """
    Propagate an orbit from initial cartesian state.
    
    Args:
        initial_state: Initial state [x, y, z, vx, vy, vz] in inertial frame [m, m/s]
        propagation_time: Time to propagate [s]
        dynamics: TwoBodyDynamics object with gravitational parameters
        method: Integration method (default: 'DOP853' - 8th order Runge-Kutta)
        rtol: Relative tolerance for integrator
        atol: Absolute tolerance for integrator
        
    Returns:
        Dictionary containing:
            - 'success': Boolean indicating success
            - 'message': Status message
            - 'time': Time array [s]
            - 'state': State history [6 x n] array
            - 'final_state': Final state vector [6]
    """
    # Time span for integration
    t_span = (0.0, propagation_time)
    
    # Solve the initial value problem
    solution = solve_ivp(
        fun=EquationsOfMotion(dynamics).state_time_derivative,
        t_span=t_span,
        y0=initial_state,
        method=method,
        args=(dynamics,),
        rtol=rtol,
        atol=atol,
        dense_output=True
    )
    
    return {
        'success': solution.success,
        'message': solution.message,
        'time': solution.t,
        'state': solution.y,
        'final_state': solution.y[:, -1]
    }


def main():
    """
    Main function to set up and propagate a spacecraft orbit for one day.
    """
    # Define propagation time: 1 day in seconds
    ONE_DAY = 1 * 24 * 60 * 60  # [s]
    
    # Set up dynamics model for Earth with J2 perturbation
    earth_dynamics = TwoBodyDynamics(
        gp     = PHYSICALCONSTANTS.EARTH.GP,
        time_o = 0.0,
        j_2    = PHYSICALCONSTANTS.EARTH.J_2,
        j_3    = PHYSICALCONSTANTS.EARTH.J_3,
        j_4    = PHYSICALCONSTANTS.EARTH.J_4,
        pos_ref= PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR
    )
    
    # Define initial cartesian state in inertial frame
    altitude  = 500e3  # [m]
    pos_mag_o = PHYSICALCONSTANTS.EARTH.RADIUS.EQUATOR + altitude
    vel_mag_o = np.sqrt(PHYSICALCONSTANTS.EARTH.GP / pos_mag_o)

    # Initial state
    initial_state = np.array([
        pos_mag_o, # x [m]
        0.0,       # y [m]
        0.0,       # z [m]
        0.0,       # vx [m/s]
        vel_mag_o, # vy [m/s]
        0.0        # vz [m/s]
    ])
    
    # Propagate the orbit
    result = propagate_orbit(
        initial_state    = initial_state,
        propagation_time = ONE_DAY,
        dynamics         = earth_dynamics
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
        r_final = np.linalg.norm(final_state[0:3])
        v_initial = np.linalg.norm(initial_state[3:6])
        v_final = np.linalg.norm(final_state[3:6])
        
        print(f"\nOrbital Magnitude Comparison:")
        print(f"  Initial radius: {r_initial:.3f} m")
        print(f"  Final radius:   {r_final:.3f} m")
        print(f"  Difference:     {r_final - r_initial:.3f} m")
        print(f"\n  Initial speed:  {v_initial:.3f} m/s")
        print(f"  Final speed:    {v_final:.3f} m/s")
        print(f"  Difference:     {v_final - v_initial:.3f} m/s")
    else:
        print(f"\nPropagation failed!")
        print(f"Status: {result['message']}")
    
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()