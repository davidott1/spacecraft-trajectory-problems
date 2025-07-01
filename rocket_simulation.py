# Import Libraries
import astropy.units as u  # type: ignore
from astropy.units.quantity import Quantity
from typing import List, Tuple
import math

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
    state: List[float],
) -> float:
    
    # Unpack state: state = [vel, mass]
    vel = state[0]
    mass = state[1]

    spec_imp = 0.0
    grav_acc_const = 9.81
    exhaust_velocity = spec_imp * grav_acc_const
    dmass__dtime = 0.0
    
    drag_acc = 0.0

    flight_path_angle = 90.0 * math.pi / 180.0
    grav_acc = -grav_acc_const * math.sin(flight_path_angle)

    dvel__dtime = exhaust_velocity * dmass__dtime + drag_acc + grav_acc

    dstate__dtime[0] = dvel__dtime
    dstate__dtime[1] = dmass__dtime

    return dstate__dtime


def simulate_rocket_trajectory(
    delta_time: Quantity,
    pos_vec_o: Quantity,
    mass_o: Quantity,
) -> None:
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
