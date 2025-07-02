from astropy.units.quantity import Quantity
import numpy as np
from typing import List, Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from rocket_simulation import rocket_dynamics

def process_input(
    delta_time_value_input: float,
    delta_time_unit_input: str,
    pos_o_value_input: float,
    pos_o_unit_input: str,
    vel_o_value_input: float,
    vel_o_unit_input: str,
    mass_o_value_input: float,
    mass_o_unit_input: str,
) -> Tuple[Quantity, Quantity, Quantity, Quantity]:

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

    return (
        delta_time,
        pos_o,
        vel_o,
        mass_o,
    )


def simulate_rocket_trajectory(
    delta_time     : Quantity,
    pos_o          : Quantity,
    vel_o          : Quantity,
    mass_o         : Quantity,
    thrust         : float,
    spec_imp       : float,
    grav_acc_const : float,
    total_time     : float,
) -> None:
    
    t_span = (0, 20)  # Time span for the integration (start, end)

    time_steps = int(total_time / delta_time.value)
    state_o = np.array([pos_o.value, vel_o.value, mass_o.value], dtype=float)


    rocket_dynamics(time, state, thrust=thrust, spec_imp=spec_imp, grav_acc_const=grav_acc_const)

    sol = solve_ivp(
        rocket_dynamics,
        t_span, 
        state_o,
        args=params, 
        t_eval=t_eval, 
        method='RK45
    )



# Main
if __name__ == "__main__":
    # Input
    delta_time_value = 1.0
    delta_time_unit  = "sec"
    pos_o_value      = 1.0
    pos_o_unit       = "m"
    vel_o_value      = 0.0
    vel_o_unit       = "m"
    mass_o_value     = 0.0
    mass_o_unit      = "kg"

    # Process Input
    (
        delta_time,
        pos_o,
        vel_o,
        mass_o,
    ) = process_input(
        delta_time_value,
        delta_time_unit,
        pos_o_value,
        pos_o_unit,
        vel_o_value,
        vel_o_unit,
        mass_o_value,
        mass_o_unit,
    )

    # Simulate
    simulate_rocket_trajectory(
        delta_time=delta_time,
        pos_o=pos_o,
        vel_o=vel_o,
        mass_o=mass_o,
    )

    # Output
    pass


# # Main execution
# if __name__ == "__main__":
#     # Example parameters for simulation
#     delta_time_value = 1.0
#     delta_time_unit = "sec"
#     pos_vec_o_value = [0.0, 0.0, 0.0]
#     pos_vec_o_unit = "m"
#     mass_o_value = 1000.0  # Initial mass in kg
#     mass_o_unit = "kg"
#     thrust = 15000.0  # Thrust in Newtons
#     spec_imp = 300.0  # Specific impulse in seconds

#     # Process Input
#     delta_time, pos_vec_o, mass_o = process_input(
#         delta_time_value,
#         delta_time_unit,
#         pos_vec_o_value,
#         pos_vec_o_unit,
#         mass_o_value,
#         mass_o_unit,
#     )

#     # Simulate
#     simulate_rocket_trajectory(
#         delta_time=delta_time,
#         pos_vec_o=pos_vec_o,
#         mass_o=mass_o,
#         thrust=thrust,
#         spec_imp=spec_imp,
#     )