from astropy.units.quantity import Quantity
import astropy.units as u
import numpy as np
from typing import List, Tuple
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import matplotlib.pyplot as plt

from rocket_dynamics import rocket_dynamics_2d_indirect

def process_input(
    delta_time_value_input: float,
    delta_time_unit_input: str,
    pos_vec_o_value_input: List[float],
    pos_vec_o_unit_input: str,
    vel_vec_o_value_input: List[float],
    vel_vec_o_unit_input: str,
    mass_o_value_input: float,
    mass_o_unit_input: str,
    grav_acc_const_value_input: float,
    grav_acc_const_unit_input: str,
    grav_acc_sea_level_value_input: float,
    grav_acc_sea_level_unit_input: str,
    spec_imp_value_input: float,
    spec_imp_unit_input: str,
    thrust_vec_value_input: List[float],
    thrust_vec_unit_input: str,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]:

    if delta_time_unit_input in ("second", "seconds", "sec", "secs", "s"):
        delta_time_unit = u.s  # type: ignore
    else:
        delta_time_unit = u.s  # type: ignore
    delta_time = delta_time_value_input * delta_time_unit

    if pos_vec_o_unit_input in ("meter", "meters", "m"):
        pos_vec_o_unit = u.m  # type: ignore
    else:
        pos_vec_o_unit = u.m  # type: ignore
    pos_vec_o = pos_vec_o_value_input * pos_vec_o_unit

    if vel_vec_o_unit_input in ("meter/second", "m/s"):
        vel_vec_o_unit = u.m/u.s # type: ignore
    else:
        vel_vec_o_unit = u.m/u.s # type: ignore
    vel_vec_o = vel_vec_o_value_input * vel_vec_o_unit

    if mass_o_unit_input in ("kg", "kilogram", "kilograms"):
        mass_o_unit = u.kg  # type: ignore
    else:
        mass_o_unit = u.kg  # type: ignore
    mass_o = mass_o_value_input * mass_o_unit

    if grav_acc_const_unit_input in ("m/s^2", "m/s2", "m/s^2", "m/s2"):
        grav_acc_const_unit = u.m/u.s**2  # type: ignore
    else:
        grav_acc_const_unit = u.m/u.s**2  # type: ignore
    grav_acc_const = grav_acc_const_value_input * grav_acc_const_unit
    
    if grav_acc_sea_level_unit_input in ("m/s^2", "m/s2", "m/s^2", "m/s2"):
        grav_acc_sea_level_unit = u.m/u.s**2  # type: ignore
    else:
        grav_acc_sea_level_unit = u.m/u.s**2  # type: ignore
    grav_acc_sea_level = grav_acc_sea_level_value_input * grav_acc_sea_level_unit

    if spec_imp_unit_input in ("s", "sec", "seconds"):
        spec_imp_unit = u.s  # type: ignore
    else:
        spec_imp_unit = u.s  # type: ignore    
    spec_imp = spec_imp_value_input * spec_imp_unit

    if thrust_vec_unit_input in ("N", "newton", "newtons"):
        thrust_vec_unit = u.N  # type: ignore
    else:
        thrust_vec_unit = u.N  # type: ignore
    thrust_vec = thrust_vec_value_input * thrust_vec_unit

    return (
        delta_time,
        pos_vec_o,
        vel_vec_o,
        mass_o,
        grav_acc_const,
        grav_acc_sea_level,
        spec_imp,
        thrust_vec,
    )


def propagate_rocket_trajectory_2d(
    delta_time         : Quantity,
    time_steps         : int,
    pos_vec_o          : Quantity,
    vel_vec_o          : Quantity,
    mass_o             : Quantity,
    copos_vec_o        : Quantity,
    covel_vec_o        : Quantity,
    comass_o           : Quantity,
    thrust_vec         : Quantity,
    spec_imp           : Quantity,
    grav_acc_const     : Quantity,
    grav_acc_sea_level : Quantity,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, OdeResult]:
    """
    Simulate and plot 2D rocket trajectory to a specified altitude with no vertical velocity.
    """
    time_span        = (0, delta_time.value)
    time_eval        = np.linspace(0, delta_time.value, time_steps)
    state_o          = np.hstack([pos_vec_o.value, vel_vec_o.value, mass_o.value, copos_vec_o.value, covel_vec_o.value, comass_o.value])
    exhaust_velocity = spec_imp * grav_acc_sea_level
    thrust_mag       = np.linalg.norm(thrust_vec)
    params           = (thrust_vec.value, exhaust_velocity.value, grav_acc_const.value)

    soln = solve_ivp(
        rocket_dynamics_2d_indirect,
        time_span, 
        state_o,
        args   = params, 
        t_eval = time_eval, 
        method = 'RK45',
    )

    pos_vec_f   = soln.y[0:2,-1] *   pos_vec_o.unit
    vel_vec_f   = soln.y[2:4,-1] *   vel_vec_o.unit
    mass_f      = soln.y[  4,-1] *      mass_o.unit
    copos_vec_f = soln.y[5:7,-1] * copos_vec_o.unit
    covel_vec_f = soln.y[7:9,-1] * covel_vec_o.unit
    comass_f    = soln.y[  9,-1] *    comass_o.unit

    return (
        pos_vec_f,
        vel_vec_f,
        mass_f,
        copos_vec_f,
        covel_vec_f,
        comass_f,
        soln_init,
    )

def plot_rocket_trajectory(
    soln_init: OdeResult,
):
    """
    Plot the rocket trajectory based on the solution from the ODE solver.
    """
    # Extract position data
    pos_x_t_init = soln_init.y[0, :] * u.m  # type: ignore
    pos_y_t_init = soln_init.y[1, :] * u.m  # type: ignore
    
    # Create a plot 
    plt.figure(figsize=(10, 6))
    plt.plot(pos_x_t_init.value, pos_y_t_init.value, color='black')
    plt.title('Rocket Trajectory Simulation')
    plt.xlabel('Pos-X [m]')
    plt.ylabel('Pos-Y [m]')
    plt.grid()
    plt.axis('equal')

# Main
if __name__ == "__main__":
    # Input
    delta_time_value         = 1.0
    delta_time_unit          = "sec"
    time_steps               = 100
    pos_vec_o_value          = [1.0, 0.0]
    pos_vec_o_unit           = "m"
    vel_vec_o_value          = [1.0, 0.0]
    vel_vec_o_unit           = "m"
    mass_o_value             = 1000.0
    mass_o_unit              = "kg"
    grav_acc_const_value     = 9.81
    grav_acc_const_unit      = "m/s^2"
    grav_acc_sea_level_value = 9.81
    grav_acc_sea_level_unit  = "m/s^2"
    spec_imp_value           = 300.0
    spec_imp_unit            = "s"
    thrust_vec_value         = [0.0, 20000.0]
    thrust_vec_unit          = "N"

    # Process input
    (
        delta_time,
        pos_vec_o,
        vel_vec_o,
        mass_o,
        grav_acc_const,
        grav_acc_sea_level,
        spec_imp,
        thrust_vec,
    ) = process_input(
        delta_time_value,
        delta_time_unit,
        pos_vec_o_value,
        pos_vec_o_unit,
        vel_vec_o_value,
        vel_vec_o_unit,
        mass_o_value,
        mass_o_unit,
        grav_acc_const_value,
        grav_acc_const_unit,
        grav_acc_sea_level_value,
        grav_acc_sea_level_unit,
        spec_imp_value,
        spec_imp_unit,
        thrust_vec_value,
        thrust_vec_unit,
    )

    # Guess costate vectors
    copos_vec_o = pos_vec_o * 0.0 * u.m     # type: ignore
    covel_vec_o = vel_vec_o * 0.0 * u.m/u.s # type: ignore
    comass_o    =    mass_o * 0.0 * u.kg    # type: ignore

    # Simulate rocket trajectory
    (
        pos_vec_f,
        vel_vec_f,
        mass_f,
        copos_vec_f,
        covel_vec_f,
        comass_f,
        soln_init,
    ) = propagate_rocket_trajectory_2d(
        delta_time         = delta_time,
        time_steps         = time_steps,
        pos_vec_o          = pos_vec_o,
        vel_vec_o          = vel_vec_o,
        mass_o             = mass_o,
        copos_vec_o        = copos_vec_o,
        covel_vec_o        = covel_vec_o,
        comass_o           = comass_o,
        thrust_vec         = thrust_vec,
        spec_imp           = spec_imp,
        grav_acc_const     = grav_acc_const,
        grav_acc_sea_level = grav_acc_sea_level,
    )
    
    # Plot the trajectory
    plot_rocket_trajectory(soln_init)

    # Show the plot
    plt.show()



