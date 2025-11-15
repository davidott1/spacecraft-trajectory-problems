"""
Orbit Propagator
================

Numerical integration of spacecraft equations of motion.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing          import Optional

from src.model.dynamics  import GeneralStateEquationsOfMotion, OrbitConverter, Acceleration
from src.model.constants import PHYSICALCONSTANTS

def propagate_state_numerical_integration(
    initial_state       : np.ndarray,
    time_o              : float,
    time_f              : float,
    dynamics            : Acceleration,
    method              : str           = 'DOP853', # DOP853 RK45
    rtol                : float         = 1e-12,
    atol                : float         = 1e-12,
    get_coe_time_series : bool          = False,
    num_points          : Optional[int] = None,
    gp                  : float         = PHYSICALCONSTANTS.EARTH.GP,
) -> dict:
    """
    Propagate an orbit from initial cartesian state.
    
    Parameters:
    -----------
    initial_state : np.ndarray
        Initial state vector [pos, vel] in meters and m/s
    time_o : float
        Initial time [s]
    time_f : float
        Final time [s]
    dynamics : Acceleration
        Acceleration model containing all force models
    method : str
        Integration method for scipy.solve_ivp (default: 'DOP853')
    rtol : float
        Relative tolerance for integration
    atol : float
        Absolute tolerance for integration
    get_coe_time_series : bool
        If True, convert states to classical orbital elements
    num_points : int, optional
        Number of output points. If None, uses adaptive timesteps from solver.
        If specified, solution is evaluated at uniformly spaced times.
    gp : float, optional
        Gravitational parameter for orbital element conversion [m³/s²]
        If None, uses dynamics.gravity.two_body.gp
    
    Returns:
    --------
    dict : Dictionary containing:
        - success : bool - Integration success flag
        - message : str - Status message
        - time : np.ndarray - Time array [s]
        - state : np.ndarray - State history [6 x N]
        - final_state : np.ndarray - Final state vector
        - coe : dict - Classical orbital elements time series (if requested)
    """
    # Time span for integration
    time_span = (time_o, time_f)

    # Solve initial value problem
    solution = solve_ivp(
      fun          = GeneralStateEquationsOfMotion(dynamics).state_time_derivative,
      t_span       = time_span,
      y0           = initial_state,
      method       = method,
      rtol         = rtol,
      atol         = atol,
      dense_output = True,
    )
    
    # If num_points is specified, evaluate solution at uniform time grid
    if num_points is not None:
      t_eval     = np.linspace(time_o, time_f, num_points)
      y_eval     = solution.sol(t_eval)
      solution.t = t_eval
      solution.y = y_eval
    
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
      for i in range(num_steps):
        pos = solution.y[0:3, i]
        vel = solution.y[3:6, i]
        
        coe = OrbitConverter.pv_to_coe(pos, vel, gp)
        for key in coe_time_series.keys():
          if coe[key] is not None:
            coe_time_series[key][i] = coe[key]
    
    return {
        'success' : solution.success,
        'message' : solution.message,
        'time'    : solution.t,
        'state'   : solution.y,
        'state_f' : solution.y[:, -1],
        'coe'     : coe_time_series,
    }
