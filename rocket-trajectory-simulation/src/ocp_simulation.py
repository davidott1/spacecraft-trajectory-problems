from astropy.units.quantity import Quantity
import astropy.units as u
import numpy as np
from typing import List, Tuple
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import copy
from functools import partial
from scipy.optimize import root

from rocket_dynamics import rocket_dynamics_2d_indirect, forcefreedynamics_2d_minenergy_indirect

def process_input(
    delta_time_value_input: float,
    delta_time_unit_input: str,
    pos_vec_o_value_input: List[float],
    pos_vec_o_unit_input: str,
    vel_vec_o_value_input: List[float],
    vel_vec_o_unit_input: str,
    mass_o_value_input: float,
    mass_o_unit_input: str,
    pos_y_f_value_input: float,
    pos_y_f_unit_input: str,
    vel_x_f_value_input: float,
    vel_x_f_unit_input: str,
    vel_y_f_value_input: float,
    vel_y_f_unit_input: str,
    grav_acc_const_value_input: float,
    grav_acc_const_unit_input: str,
    grav_acc_sea_level_value_input: float,
    grav_acc_sea_level_unit_input: str,
    spec_imp_value_input: float,
    spec_imp_unit_input: str,
    thrust_max_value_input: float,
    thrust_max_unit_input: str,
) -> Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]:

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

    if pos_y_f_unit_input in ("meter", "meters", "m"):
        pos_y_f_unit = u.m  # type: ignore
    else:
        pos_y_f_unit = u.m  # type: ignore
    pos_y_f = pos_y_f_value_input * pos_y_f_unit

    if vel_x_f_unit_input in ("meter/second", "m/s"):
        vel_x_f_unit = u.m/u.s  # type: ignore
    else:
        vel_x_f_unit = u.m/u.s  # type: ignore
    vel_x_f = vel_x_f_value_input * vel_x_f_unit

    if vel_y_f_unit_input in ("meter/second", "m/s"):
        vel_y_f_unit = u.m/u.s  # type: ignore
    else:
        vel_y_f_unit = u.m/u.s  # type: ignore
    vel_y_f = vel_y_f_value_input * vel_y_f_unit

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

    if thrust_max_unit_input in ("N", "newton", "newtons"):
        thrust_max_unit = u.N  # type: ignore
    else:
        thrust_max_unit = u.N  # type: ignore
    thrust_vec = thrust_max_value_input * thrust_max_unit

    exhaust_velocity = spec_imp * grav_acc_sea_level

    return (
        delta_time,
        pos_vec_o,
        vel_vec_o,
        mass_o,
        pos_y_f,
        vel_x_f,
        vel_y_f,
        grav_acc_const,
        grav_acc_sea_level,
        spec_imp,
        thrust_vec,
        exhaust_velocity,
    )


def plot_rocket_trajectory(
    soln_init: OdeResult,
):
    
    pos_x_t_init = soln_init.y[0, :] * u.m  # type: ignore
    pos_y_t_init = soln_init.y[1, :] * u.m  # type: ignore
    
    plt.figure(figsize=(10, 6))
    # plt.xlim(0, np.max(pos_x_t_init.value) * 1.1)
    # plt.ylim(0, np.max(pos_y_t_init.value) * 1.1)
    plt.title('Rocket Trajectory Simulation')
    plt.xlabel('Pos-X [m]')
    plt.ylabel('Pos-Y [m]')
    plt.grid()
    plt.axis('equal')
    plt.plot(pos_x_t_init.value, pos_y_t_init.value, color='black')
    plt.plot(pos_x_t_init[0].value, pos_y_t_init[0].value, marker='>', color='white', markeredgecolor='black', markersize=10)
    plt.plot(pos_x_t_init[-1].value, pos_y_t_init[-1].value, marker='s', color='white', markeredgecolor='black', markersize=10)
   


def objective_function(
        decision_states: np.ndarray,
        time_span,
        integration_posvel_o, 
        integration_posvel_f,
    ):
    
    # Combine known initial states with the guessed initial costates
    integration_state_o = np.concatenate((integration_posvel_o, decision_states))
    
    # Integrate the dynamics from t0 to tf
    soln = solve_ivp(
        forcefreedynamics_2d_minenergy_indirect,
        time_span,
        integration_state_o,
        dense_output=True,
        method='DOP853',
    )
    
    # Get the state vector at the final time tf
    integration_state_f = soln.sol(time_span[1])[:4]
    
    # Calculate the error for root finder
    error = integration_state_f - integration_posvel_f
    
    return error


# Time interval
t_o = 0
t_f = 10

# Boundary Conditions (initial and final states) [rx, ry, vx, vy]
integration_posvel_o = np.array([0, 0, 1, 2])
integration_posvel_f = np.array([10, 15, 0, 0])

# Initial guess decision state
decistion_state_initguess = np.zeros(4)

# Use the root finder to find the correct initial costates
soln_root = root(
    objective_function, 
    decistion_state_initguess, 
    args=([t_o, t_f], integration_posvel_o, integration_posvel_f),
    method='hybr',
)

# Check if the solver was successful
if not soln_root.success:
    raise RuntimeError(f"Root finder failed to converge: {soln_root.message}")

# The optimal initial costates found by the solver
decistion_state_finalsoln = soln_root.x
print(f"Optimal Initial Costates:\n{decistion_state_finalsoln}")

# 4. Get the Full Optimal Trajectory
# ------------------------------------

# Combine the true initial states and the optimal initial costates
integration_state_finalsoln = np.concatenate((integration_posvel_o, decistion_state_finalsoln))

# Perform one final integration to get the full, dense solution
time_eval = np.linspace(t_o, t_f, 200)
soln_finalsoln = \
    solve_ivp(
        forcefreedynamics_2d_minenergy_indirect, 
        [t_o, t_f], 
        integration_state_finalsoln, 
        dense_output=True, 
        t_eval=time_eval,
    )

# Extract state and control trajectories
time = soln_finalsoln.t
states = soln_finalsoln.y
rx, ry, vx, vy, copos_x, copos_y, covel_x, covel_y = states

# Optimal control inputs
thrust_acc_x = -covel_x
thrust_acc_y = -covel_y

# 5. Plot the Results
# ---------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Optimal Trajectory and Control', fontsize=16)

# Plot Position (rx, ry)
axs[0].plot(time, rx, label='$r_x(t)$')
axs[0].plot(time, ry, label='$r_y(t)$', linestyle='--')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid(True)

# Plot Velocity (vx, vy)
axs[1].plot(time, vx, label='$v_x(t)$')
axs[1].plot(time, vy, label='$v_y(t)$', linestyle='--')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid(True)

# Plot Control Input (thrust acceleration)
axs[2].plot(time, thrust_acc_x, label='$thrust\_acc_x(t)$')
axs[2].plot(time, thrust_acc_y, label='$thrust\_acc_y(t)$', linestyle='--')
axs[2].set_ylabel('Thrust Acceleration')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot the 2D trajectory path
plt.figure(figsize=(8, 6))
plt.plot(rx, ry, label='Trajectory Path')
plt.scatter([integration_posvel_o[0]], [integration_posvel_o[1]], color='green', s=100, zorder=5, label='Start')
plt.scatter([integration_posvel_f[0]], [integration_posvel_f[1]], color='red', s=100, zorder=5, label='End')
plt.title('Optimal 2D Trajectory')
plt.xlabel('$r_x$')
plt.ylabel('$r_y$')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


# # Main
# if __name__ == "__main__":
#     # Input
#     delta_time_value, delta_time_unit = 10.0, "s"
#     time_steps                        = 100

#     pos_vec_o_value, pos_vec_o_unit = [0.0, 0.0], "m"
#     vel_vec_o_value, vel_vec_o_unit = [0.0, 0.0], "m/s"
#     mass_o_value   , mass_o_unit    =     1000.0, "kg" # 1000.0, "kg" | 549054, "kg"

#     pos_y_f_value, pos_y_f_unit = 100.0, "m"
#     vel_x_f_value, vel_x_f_unit =  10.0, "m/s"
#     vel_y_f_value, vel_y_f_unit =   0.0, "m/s"

#     grav_acc_const_value    , grav_acc_const_unit     =             9.81, "m/s^2"
#     grav_acc_sea_level_value, grav_acc_sea_level_unit =             9.81, "m/s^2"
#     spec_imp_value          , spec_imp_unit           =           300.0 , "s"
#     thrust_max_value        , thrust_max_unit         = 1.0*9.81*1000.0 , "N" # 1.0*9.81*1000.0 , "N" | 7607000.0, "N"

#     # Guess costate vectors
#     copos_vec_o = [ 0.0e-3,  1.0e-3] * u.m     # type: ignore
#     covel_vec_o = [ 0.0e-6, -1.0e-6] * u.m/u.s # type: ignore

#     # Process input
#     (
#         delta_time,
#         pos_vec_o,
#         vel_vec_o,
#         mass_o,
#         pos_y_f,
#         vel_x_f,
#         vel_y_f,
#         grav_acc_const,
#         grav_acc_sea_level,
#         spec_imp,
#         thrust_max,
#         exhaust_velocity,
#     ) = \
#         process_input(
#             delta_time_value,
#             delta_time_unit,
#             pos_vec_o_value,
#             pos_vec_o_unit,
#             vel_vec_o_value,
#             vel_vec_o_unit,
#             mass_o_value,
#             mass_o_unit,
#             pos_y_f_value,
#             pos_y_f_unit,
#             vel_x_f_value,
#             vel_x_f_unit,
#             vel_y_f_value,
#             vel_y_f_unit,
#             grav_acc_const_value,
#             grav_acc_const_unit,
#             grav_acc_sea_level_value,
#             grav_acc_sea_level_unit,
#             spec_imp_value,
#             spec_imp_unit,
#             thrust_max_value,
#             thrust_max_unit,
#         )


#     # # Plot the trajectory
#     # plot_rocket_trajectory(soln_init)

#     # Show the plot
#     plt.show()



# def propagate_rocket_trajectory_2d(
#     delta_time         : Quantity,
#     time_steps         : int,
#     pos_vec_o          : Quantity,
#     vel_vec_o          : Quantity,
#     mass_o             : Quantity,
#     copos_vec_o        : Quantity,
#     covel_vec_o        : Quantity,
#     comass_o           : Quantity,
#     thrust_max         : Quantity,
#     exhaust_velocity   : Quantity,
#     grav_acc_const     : Quantity,
# ) -> Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, Quantity, OdeResult]:
#     """
#     Simulate and plot 2D rocket trajectory to a specified altitude with no vertical velocity.
#     """
#     time_span = (0, delta_time.value)
#     time_eval = np.linspace(0, delta_time.value, time_steps)
#     state_o   = np.hstack([pos_vec_o.value, vel_vec_o.value, mass_o.value, copos_vec_o.value, covel_vec_o.value, comass_o.value])s
#     params    = (thrust_max.value, exhaust_velocity.value, grav_acc_const.value)

#     soln = solve_ivp(
#         rocket_dynamics_2d_indirect,
#         time_span, 
#         state_o,
#         args   = params, 
#         t_eval = time_eval, 
#         method = 'RK45',
#         rtol   = 1.0e-9,  # Decreased relative tolerance (more strict)
#         atol   = 1.0e-12,  # Decreased absolute tolerance (more strict)
#     )
#     soln_init = copy.deepcopy(soln)

#     pos_vec_f    = soln.y[0:2,-1] *   pos_vec_o.unit
#     vel_vec_f    = soln.y[2:4,-1] *   vel_vec_o.unit
#     mass_f       = soln.y[  4,-1] *      mass_o.unit
#     copos_vec_f  = soln.y[5:7,-1] * copos_vec_o.unit
#     covel_vec_f  = soln.y[7:9,-1] * covel_vec_o.unit
#     comass_f     = soln.y[  9,-1] *    comass_o.unit
#     thrust_ang_f = np.arctan2(-covel_vec_f[1], -covel_vec_f[0])
#     ham_f        = copos_vec_f[0]*vel_vec_f[0] + copos_vec_f[1]*vel_vec_f[1] \
#                    + thrust_max/mass_f * (covel_vec_f[0]*np.cos(thrust_ang_f) + covel_vec_f[1]*np.sin(thrust_ang_f)) \
#                    - covel_vec_f[1]*grav_acc_const \
#                    - thrust_max*comass_f/exhaust_velocity

#     return (
#         pos_vec_f,
#         vel_vec_f,
#         mass_f,
#         copos_vec_f,
#         covel_vec_f,
#         comass_f,
#         ham_f,
#         soln_init,
#     )
 

# def boundary_conditions_rocket_trajectory_2d(
#     state_o          : np.ndarray, #List[Quantity],
#     state_f          : np.ndarray, #List[Quantity],
#     thrust_max       : float, #Quantity,
#     exhaust_velocity : float, #Quantity,
#     grav_acc_const   : float, #Quantity,
# ):

#     pos_x_o   = state_o[0]
#     pos_y_o   = state_o[1]
#     vel_x_o   = state_o[2]
#     vel_y_o   = state_o[3]
#     mass_o    = state_o[4]
#     copos_x_o = state_o[5]
#     copos_y_o = state_o[6]
#     covel_x_o = state_o[7]
#     covel_y_o = state_o[8]
#     comass_o  = state_o[9]

#     pos_x_f   = state_f[0]
#     pos_y_f   = state_f[1]
#     vel_x_f   = state_f[2]
#     vel_y_f   = state_f[3]
#     mass_f    = state_f[4]
#     copos_x_f = state_f[5]
#     copos_y_f = state_f[6]
#     covel_x_f = state_f[7]
#     covel_y_f = state_f[8]
#     comass_f  = state_f[9]

#     # thrust_ang_o = np.arctan2(-covel_y_o, -covel_x_o)
#     # ham_o        = copos_x_o*vel_x_o + copos_y_o*vel_y_o \
#     #                + thrust_max/mass_o * (covel_x_o*np.cos(thrust_ang_o) + covel_y_o*np.sin(thrust_ang_o)) \
#     #                - covel_y_o*grav_acc_const \
#     #                - thrust_max*comass_o/exhaust_velocity

#     # thrust_ang_f = np.arctan2(-covel_y_f, -covel_x_f)
#     # ham_f        = copos_x_f*vel_x_f + copos_y_f*vel_y_f \
#     #                + thrust_max/mass_f * (covel_x_f*np.cos(thrust_ang_f) + covel_y_f*np.sin(thrust_ang_f)) \
#     #                - covel_y_f*grav_acc_const \
#     #                - thrust_max*comass_f/exhaust_velocity

#     pos_x_o_bc   =    0.0
#     pos_y_o_bc   =    0.0
#     vel_x_o_bc   =    0.0
#     vel_y_o_bc   =    0.0
#     mass_o_bc    = 1000.0
#     # copos_x_o_bc =  free(unknown) x
#     # copos_y_o_bc =  free(unknown) x
#     # covel_x_o_bc =  free(unknown) x
#     # covel_y_o_bc =  free(unknown) x
#     # comass_o_bc  =  free(unknown) x
#     # ham_o_bc     =    0.0 # free t0

#     # pos_x_f_bc   =  free
#     pos_y_f_bc   =  100.0 # x
#     vel_x_f_bc   =   10.0 # x
#     vel_y_f_bc   =    0.0 # x
#     # mass_f_bc    =  free
#     copos_x_f_bc =    0.0 # x
#     # copos_y_f_bc =  free
#     # covel_x_f_bc =  free
#     # covel_y_f_bc =  free
#     comass_f_bc  =   -1.0 # x
#     # ham_f_bc     =    0.0 # free tf

#     # decision variables = 10 + 1
#     return np.array([
#         pos_x_o        - pos_x_o_bc,     #   pos_x_o -    0.0
#         pos_y_o        - pos_y_o_bc,     #   pos_y_o -    0.0
#         vel_x_o        - vel_x_o_bc,     #   vel_x_o -    0.0
#         vel_y_o        - vel_y_o_bc,     #   vel_y_o -    0.0
#         mass_o         - mass_o_bc,      #    mass_o - 1000.0
#         # copos_x_o      - copos_x_o_bc, # free
#         # copos_y_o      - copos_y_o_bc, # free
#         # covel_x_o      - covel_x_o_bc, # free
#         # covel_y_o      - covel_y_o_bc, # free
#         # comass_o       - comass_o_bc,  # free
#         # ham_o          - ham_o_bc,     #   ham_o -    0.0
#         # pos_x_f        - pos_x_f_bc,   # free
#         pos_y_f        - pos_y_f_bc,     #   pos_y_f -  100.0
#         vel_x_f        - vel_x_f_bc,     #   vel_x_f -   10.0
#         vel_y_f        - vel_y_f_bc,     #   vel_y_f -    0.0
#         # mass_f         - mass_f_bc,    # free
#         copos_x_f      - copos_x_f_bc,   # copos_x_f -    0.0
#         # copos_y_f      - copos_y_f_bc, # free
#         # covel_x_f      - covel_x_f_bc, # free
#         # covel_y_f      - covel_y_f_bc, # free
#         comass_f       - comass_f_bc,    #  comass_f +    1.0
#         # ham_f          - ham_f_bc,     #     ham_f -    0.0
#     ])

# def boundary_conditions_rocket_trajectory_2d(
#     dec_state          : List[Quantity],
#     pos_vec_o          : Quantity,
#     vel_vec_o          : Quantity,
#     mass_o             : Quantity,
#     copos_x_o          : Quantity,
#     thrust_max         : Quantity,
#     spec_imp           : Quantity,
#     grav_acc_const     : Quantity,
#     grav_acc_sea_level : Quantity,
# ):
    
#     delta_time = dec_state[0]
#     copos_y_o  = dec_state[1]
#     covel_x_o  = dec_state[2]
#     covel_y_o  = dec_state[3]
#     comass_o   = dec_state[4]

#     copos_vec_o[0] = copos_x_o
#     copos_vec_o[1] = copos_y_o
#     covel_vec_o[0] = covel_x_o
#     covel_vec_o[1] = covel_y_o

#     (
#         pos_vec_f,
#         vel_vec_f,
#         mass_f,
#         copos_vec_f,
#         covel_vec_f,
#         comass_f,
#         ham_f,
#         soln_init,
#     ) = propagate_rocket_trajectory_2d(
#         delta_time         = delta_time,
#         time_steps         = time_steps,
#         pos_vec_o          = pos_vec_o,
#         vel_vec_o          = vel_vec_o,
#         mass_o             = mass_o,
#         copos_vec_o        = copos_vec_o,
#         covel_vec_o        = covel_vec_o,
#         comass_o           = comass_o,
#         thrust_max         = thrust_max,
#         spec_imp           = spec_imp,
#         grav_acc_const     = grav_acc_const,
#         grav_acc_sea_level = grav_acc_sea_level,
#     )

#     return np.array([
#         pos_vec_f[1]   - 100.0,
#         vel_vec_f[0]   -  10.0,
#         vel_vec_f[1]   -   0.0,
#         comass_f       +   1.0,
#         ham_f          -   0.0,
#     ])