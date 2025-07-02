# Import Libraries
import astropy.units as u  # type: ignore
from astropy.units.quantity import Quantity
from typing import List, Tuple
import math
import numpy as np

# Functions
def process_input(
    delta_time_value_input: float,
    delta_time_unit_input: str,
    pos_vec_o_value_input: List[float],
    pos_vec_o_unit_input: str,
    mass_o_value_input: float,
    mass_o_unit_input: str,
) -> Tuple[Quantity, Quantity, Quantity]:

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

    if mass_o_unit_input in ("kg", "kilogram", "kilograms"):
        mass_o_unit = u.kg  # type: ignore
    else:
        mass_o_unit = u.kg  # type: ignore
    mass_o = mass_o_value_input * mass_o_unit

    return (
        delta_time,
        pos_vec_o,
        mass_o,
    )


def rocket_dynamics(
    time: float,
    state: np.ndarray,
    **kwargs,
) -> np.ndarray:

    pos  = state[0]
    vel  = state[1]
    mass = state[2]

    spec_imp           = kwargs.get('spec_imp', 0.0)
    grav_acc_const     = kwargs.get('grav_acc_const', 9.81)
    grav_acc_sea_level = kwargs.get('grav_acc_sea_level', 9.81)
    thrust             = kwargs.get('thrust', 0.0)

    exhaust_velocity = spec_imp * grav_acc_sea_level

    grav_acc   = -grav_acc_const
    thrust_acc = thrust / mass

    dpos__dtime  = vel
    dvel__dtime  = grav_acc + thrust_acc # + drag_acc
    dmass__dtime = -thrust / exhaust_velocity

    dstate__dtime    = np.zeros(3, dtype=float)
    dstate__dtime[0] = dpos__dtime
    dstate__dtime[1] = dvel__dtime
    dstate__dtime[2] = dmass__dtime

    return dstate__dtime


def simulate_rocket_trajectory(
    delta_time: Quantity,
    pos_vec_o: Quantity,
    mass_o: Quantity,
) -> None:
    # rocket_dynamics(time, state)
    pass


# Main
if __name__ == "__main__":
    # Input
    delta_time_value = 1.0
    delta_time_unit = "sec"
    pos_vec_o_value = [0.0, 0.0, 0.0]
    pos_vec_o_unit = "m"
    mass_o_value = 0.0
    mass_o_unit = "kg"

    # Process Input
    (
        delta_time,
        pos_vec_o,
        mass_o,
    ) = process_input(
        delta_time_value,
        delta_time_unit,
        pos_vec_o_value,
        pos_vec_o_unit,
        mass_o_value,
        mass_o_unit,
    )

    # Simulate
    simulate_rocket_trajectory(
        delta_time=delta_time,
        pos_vec_o=pos_vec_o,
        mass_o=mass_o,
    )

    # Output
    pass
