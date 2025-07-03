from astropy.units.quantity import Quantity
import astropy.units as u
import numpy as np
from typing import List, Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from rocket_dynamics import rocket_dynamics

def process_input(
    delta_time_value_input: float,
    delta_time_unit_input: str,
    pos_o_value_input: float,
    pos_o_unit_input: str,
    vel_o_value_input: float,
    vel_o_unit_input: str,
    mass_o_value_input: float,
    mass_o_unit_input: str,
    grav_acc_const_value_input: float,
    grav_acc_const_unit_input: str,
    grav_acc_sea_level_value_input: float,
    grav_acc_sea_level_unit_input: str,
    spec_imp_value_input: float,
    spec_imp_unit_input: str,
    thrust_value_input: float,
    thrust_unit_input: str,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]:

    if delta_time_unit_input in ("second", "seconds", "sec", "secs", "s"):
        delta_time_unit = u.s  # type: ignore
    else:
        delta_time_unit = u.s  # type: ignore
    delta_time = delta_time_value_input * delta_time_unit

    if pos_o_unit_input in ("meter", "meters", "m"):
        pos_vec_o_unit = u.m  # type: ignore
    else:
        pos_vec_o_unit = u.m  # type: ignore
    pos_o = pos_o_value_input * pos_vec_o_unit

    if vel_o_unit_input in ("meter/second", "m/s"):
        vel_o_unit = u.m/u.s # type: ignore
    else:
        vel_o_unit = u.m/u.s # type: ignore
    vel_o = vel_o_value_input * vel_o_unit

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

    if thrust_unit_input in ("N", "newton", "newtons"):
        thrust_unit = u.N  # type: ignore
    else:
        thrust_unit = u.N  # type: ignore
    thrust = thrust_value_input * thrust_unit

    return (
        delta_time,
        pos_o,
        vel_o,
        mass_o,
        grav_acc_const,
        grav_acc_sea_level,
        spec_imp,
        thrust,
    )


def simulate_rocket_trajectory(
    delta_time         : Quantity,
    time_steps         : int,
    pos_o              : Quantity,
    vel_o              : Quantity,
    mass_o             : Quantity,
    thrust             : Quantity,
    spec_imp           : Quantity,
    grav_acc_const     : Quantity,
    grav_acc_sea_level : Quantity,
) -> None:
    
    time_span = (0, delta_time.value)
    time_eval = np.linspace(0, delta_time.value, time_steps)
    state_o   = np.array([pos_o.value, vel_o.value, mass_o.value], dtype=float)
    params    = (thrust.value, spec_imp.value, grav_acc_const.value, grav_acc_sea_level.value)

    sol = solve_ivp(
        rocket_dynamics,
        time_span, 
        state_o,
        args=params, 
        time_eval=time_eval, 
        method='RK45',
    )

    breakpoint()



# Main
if __name__ == "__main__":
    # Input
    delta_time_value         = 1.0
    delta_time_unit          = "sec"
    time_steps               = 100
    pos_o_value              = 1.0
    pos_o_unit               = "m"
    vel_o_value              = 0.0
    vel_o_unit               = "m"
    mass_o_value             = 1000.0
    mass_o_unit              = "kg"
    grav_acc_const_value     = 9.81
    grav_acc_const_unit      = "m/s^2"
    grav_acc_sea_level_value = 9.81
    grav_acc_sea_level_unit  = "m/s^2"
    spec_imp_value           = 300.0
    spec_imp_unit            = "s"
    thrust_value             = 20000.0
    thrust_unit              = "N"

    # Process Input
    (
        delta_time,
        pos_o,
        vel_o,
        mass_o,
        grav_acc_const,
        grav_acc_sea_level,
        spec_imp,
        thrust,
    ) = process_input(
        delta_time_value,
        delta_time_unit,
        pos_o_value,
        pos_o_unit,
        vel_o_value,
        vel_o_unit,
        mass_o_value,
        mass_o_unit,
        grav_acc_const_value,
        grav_acc_const_unit,
        grav_acc_sea_level_value,
        grav_acc_sea_level_unit,
        spec_imp_value,
        spec_imp_unit,
        thrust_value,
        thrust_unit,
    )

    # Simulate
    simulate_rocket_trajectory(
        delta_time         = delta_time,
        time_steps         = time_steps,
        pos_o              = pos_o,
        vel_o              = vel_o,
        mass_o             = mass_o,
        thrust             = thrust,
        spec_imp           = spec_imp,
        grav_acc_const     = grav_acc_const,
        grav_acc_sea_level = grav_acc_sea_level,
    )

    # Output
    pass


