from astropy.units.quantity import Quantity
import astropy.units as u
import numpy as np
from typing import List, Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from rocket_dynamics import rocket_dynamics_1d, rocket_dynamics_2d

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


def simulate_rocket_trajectory_1d(
    delta_time         : Quantity,
    time_steps         : int,
    pos_vec_o          : Quantity,
    vel_vec_o          : Quantity,
    mass_o             : Quantity,
    thrust_vec         : Quantity,
    spec_imp           : Quantity,
    grav_acc_const     : Quantity,
    grav_acc_sea_level : Quantity,
) -> None:
    """Simulate and plot 1D rocket trajectory."""
    time_span        = (0, delta_time.value)
    time_eval        = np.linspace(0, delta_time.value, time_steps)
    state_o          = np.array([pos_vec_o[0].value, vel_vec_o[0].value, mass_o.value], dtype=float)
    exhaust_velocity = spec_imp * grav_acc_sea_level
    thrust_mag       = np.linalg.norm(thrust_vec)
    params           = (thrust_mag.value, exhaust_velocity.value, grav_acc_const.value) # type: ignore

    soln = solve_ivp(
        rocket_dynamics_1d,
        time_span, 
        state_o,
        args   = params, 
        t_eval = time_eval, 
        method = 'RK45',
    )

    pos_vec_f = [soln.y[0,-1], 0.0] * pos_vec_o.unit
    vel_vec_f = [soln.y[1,-1], 0.0] * vel_vec_o.unit
    mass_f    = soln.y[2,-1] * mass_o.unit

    print("Simulation Results: 1D Rocket Trajectory")
    print(f"  Success              : {soln.success}")
    print(f"  Message              : {soln.message}")
    print(f"  Time                 :")
    print(f"    Steps              : {time_steps:>12d}")
    print(f"    Initial            : {0.0:>12.6f} s")
    print(f"    Final              : {delta_time.to_value('s'):>12.6f} s")
    print(f"  State                :")
    print(f"    Initial            :")
    print(f"      Position         : {pos_vec_o[0].to_value('m'):>12.6f} m")
    print(f"      Velocity         : {vel_vec_o[0].to_value('m/s'):>12.6f} m/s")
    print(f"      Mass             : {mass_o.to_value('kg'):>12.6f} kg")
    print(f"    Final State        :")
    print(f"      Position         : {pos_vec_f[0].to_value('m'):>12.6f} m")
    print(f"      Velocity         : {vel_vec_f[0].to_value('m/s'):>12.6f} m/s")
    print(f"      Mass             : {mass_f.to_value('kg'):>12.6f} kg")
    print(f"  System Parameters    :")
    print(f"    Thrust             : {thrust_mag.to_value('N'):>12.6f} N") # type: ignore
    print(f"    Grav Acc Const     : {grav_acc_const.to_value('m/s^2'):>12.6f} m/s^2")
    print(f"    Specific Impulse   : {spec_imp.to_value('s'):>12.6f} s")
    print(f"    Grav Acc Sea Level : {grav_acc_sea_level.to_value('m/s^2'):>12.6f} m/s^2")
    print(f"    Exhaust Velocity   : {exhaust_velocity.to_value('m/s'):>12.6f} m/s")

    print("Plotting the results ...")
    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    fig.suptitle('Rocket Trajectory Simulation (1D)')
    fig.align_ylabels(ax)
    ax[0].set_ylabel('Position [m]')
    ax[0].grid()
    ax[0].tick_params(labelbottom=False)
    ax[1].set_ylabel('Velocity [m/s]')
    ax[1].grid()
    ax[1].tick_params(labelbottom=False)
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Mass [kg]')
    ax[2].grid()
    ax[0].plot(soln.t, soln.y[0], label='Pos', color='orange')
    ax[1].plot(soln.t, soln.y[1], label='Vel', color='orange')
    ax[2].plot(soln.t, soln.y[2], label='Mass', color='black')
    plt.show()


def simulate_rocket_trajectory_2d(
    delta_time         : Quantity,
    time_steps         : int,
    pos_vec_o          : Quantity,
    vel_vec_o          : Quantity,
    mass_o             : Quantity,
    thrust_vec         : Quantity,
    spec_imp           : Quantity,
    grav_acc_const     : Quantity,
    grav_acc_sea_level : Quantity,
) -> None:
    """Simulate and plot 2D rocket trajectory."""
    time_span        = (0, delta_time.value)
    time_eval        = np.linspace(0, delta_time.value, time_steps)
    state_o          = np.hstack([pos_vec_o.value, vel_vec_o.value, mass_o.value])
    exhaust_velocity = spec_imp * grav_acc_sea_level
    thrust_mag       = np.linalg.norm(thrust_vec)
    params           = (thrust_vec.value, exhaust_velocity.value, grav_acc_const.value)

    soln = solve_ivp(
        rocket_dynamics_2d,
        time_span, 
        state_o,
        args   = params, 
        t_eval = time_eval, 
        method = 'RK45',
    )

    pos_vec_f = soln.y[0:2,-1] * pos_vec_o.unit
    vel_vec_f = soln.y[2:4,-1] * vel_vec_o.unit
    mass_f    = soln.y[4,-1] * mass_o.unit

    print("\nSimulation Results: 2D Rocket Trajectory")
    print(f"  Success              : {soln.success}")
    print(f"  Message              : {soln.message}")
    print(f"  Time                 :")
    print(f"    Steps              : {time_steps:>12d}")
    print(f"    Initial            : {0.0:>12.6f} s")
    print(f"    Final              : {delta_time.to_value('s'):>12.6f} s")
    print(f"  State                :")
    print(f"    Initial            :")
    print(f"      Position         : {pos_vec_o[0].to_value('m'):>12.6f} m")
    print(f"      Velocity         : {vel_vec_o[0].to_value('m/s'):>12.6f} m/s")
    print(f"      Mass             : {mass_o.to_value('kg'):>12.6f} kg")
    print(f"    Final State        :")
    print(f"      Position         : {pos_vec_f[0].to_value('m'):>12.6f} m")
    print(f"      Velocity         : {vel_vec_f[0].to_value('m/s'):>12.6f} m/s")
    print(f"      Mass             : {mass_f.to_value('kg'):>12.6f} kg")
    print(f"  System Parameters    :")
    print(f"    Thrust             : {thrust_mag:>12.6f} N")
    print(f"    Grav Acc Const     : {grav_acc_const.to_value('m/s^2'):>12.6f} m/s^2")
    print(f"    Specific Impulse   : {spec_imp.to_value('s'):>12.6f} s")
    print(f"    Grav Acc Sea Level : {grav_acc_sea_level.to_value('m/s^2'):>12.6f} m/s^2")
    print(f"    Exhaust Velocity   : {exhaust_velocity.to_value('m/s'):>12.6f} m/s")

    print("Plotting the results ...")
    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    fig.suptitle('Rocket Trajectory Simulation (2D)')
    fig.align_ylabels(ax)
    ax[0].set_ylabel('Position [m]')
    ax[0].grid()
    ax[0].tick_params(labelbottom=False)
    ax[1].set_ylabel('Velocity [m/s]')
    ax[1].grid()
    ax[1].tick_params(labelbottom=False)
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Mass [kg]')
    ax[2].grid()
    ax[0].plot(soln.t, soln.y[0], label='Pos X', color='blue')
    ax[0].plot(soln.t, soln.y[1], label='Pos Y', color='orange')
    ax[1].plot(soln.t, soln.y[2], label='Vel X', color='blue')
    ax[1].plot(soln.t, soln.y[3], label='Vel Y', color='orange')
    ax[2].plot(soln.t, soln.y[4], label='Mass', color='black')
    ax[0].legend()
    ax[1].legend()

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].set_xlabel('Position X [m]')
    ax[0].set_ylabel('Position Y [m]')
    ax[0].grid()
    ax[1].set_xlabel('Velocity X [m/s]')
    ax[1].set_ylabel('Velocity Y [m/s]')
    ax[1].grid()
    ax[0].plot(soln.y[0,0], soln.y[1,0], color='black', marker='>', markersize=5)
    ax[0].plot(soln.y[0], soln.y[1], color='black')
    ax[0].plot(soln.y[0,-1], soln.y[1,-1], color='black', marker='s', markersize=5)
    ax[1].plot(soln.y[2,0], soln.y[3,0], color='red', marker='>', markersize=5)
    ax[1].plot(soln.y[2], soln.y[3], color='red')
    ax[1].plot(soln.y[2,-1], soln.y[3,-1], color='red', marker='s', markersize=5)
    plt.show()

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

    # Simulate
    simulate_rocket_trajectory_1d(
        delta_time         = delta_time,
        time_steps         = time_steps,
        pos_vec_o          = pos_vec_o,
        vel_vec_o          = vel_vec_o,
        mass_o             = mass_o,
        thrust_vec         = thrust_vec,
        spec_imp           = spec_imp,
        grav_acc_const     = grav_acc_const,
        grav_acc_sea_level = grav_acc_sea_level,
    )

    simulate_rocket_trajectory_2d(
        delta_time         = delta_time,
        time_steps         = time_steps,
        pos_vec_o          = pos_vec_o,
        vel_vec_o          = vel_vec_o,
        mass_o             = mass_o,
        thrust_vec         = thrust_vec,
        spec_imp           = spec_imp,
        grav_acc_const     = grav_acc_const,
        grav_acc_sea_level = grav_acc_sea_level,
    )




