import numpy as np

def rocket_dynamics_1d(
    time             : float,
    state            : np.ndarray,
    thrust           : float = 0.0,
    exhaust_velocity : float = 300.0,
    grav_acc_const   : float = 9.81,
) -> np.ndarray:

    pos  = state[0]
    vel  = state[1]
    mass = state[2]

    grav_acc   = -grav_acc_const
    thrust_acc = thrust / mass

    dpos__dtime  = vel
    dvel__dtime  = grav_acc + thrust_acc
    dmass__dtime = -thrust / exhaust_velocity

    dstate__dtime    = np.zeros(3, dtype=float)
    dstate__dtime[0] = dpos__dtime
    dstate__dtime[1] = dvel__dtime
    dstate__dtime[2] = dmass__dtime

    return dstate__dtime


def rocket_dynamics_2d(
    time             : float,
    state            : np.ndarray,
    thrust_vec       : np.ndarray = np.array([0.0,1.0]),
    exhaust_velocity : float      = 300.0,
    grav_acc_const   : float      = 9.81,
):

    pos_vec = state[0:2]
    vel_vec = state[2:4]
    mass    = state[4  ]

    grav_acc_vec   = np.array([0.0,-grav_acc_const])
    thrust_acc_vec = thrust_vec / mass
    thrust_mag     = np.linalg.norm(thrust_vec)

    dpos_vec__dtime = vel_vec
    dvel_vec__dtime = grav_acc_vec + thrust_acc_vec
    dmass__dtime    = -thrust_mag / exhaust_velocity

    dstate__dtime      = np.zeros(5, dtype=float)
    dstate__dtime[0:2] = dpos_vec__dtime
    dstate__dtime[2:4] = dvel_vec__dtime
    dstate__dtime[4  ] = dmass__dtime

    return dstate__dtime


def rocket_dynamics_2d_indirect(
    time             : float,
    state            : np.ndarray,
    thrust_mag_max   : float = 0.0,
    exhaust_velocity : float = 300.0,
    grav_acc_const   : float = 9.81,
):

    pos_x   = state[0]
    pos_y   = state[1]
    vel_x   = state[2]
    vel_y   = state[3]
    mass    = state[4]
    copos_x = state[5]
    copos_y = state[6]
    covel_x = state[7]
    covel_y = state[8]
    comass  = state[9]

    grav_acc_x   = 0.0
    grav_acc_y   = -grav_acc_const

    thrust_ang  = np.arctan2(-covel_y, -covel_x)
    switch_func = 1.0/mass * (covel_x*np.cos(thrust_ang) + covel_y*np.sin(thrust_ang)) - comass/exhaust_velocity
    if switch_func > 0.0:
        thrust_mag = thrust_mag_max
    elif switch_func < 0.0:
        thrust_mag = 0.0
    else:
        thrust_mag = 0.0 # indeterminate

    thrust_acc_x  = thrust_mag/mass * np.cos(thrust_ang)
    thrust_acc_y  = thrust_mag/mass * np.sin(thrust_ang)

    dpos_x__dtime   = vel_x
    dpos_y__dtime   = vel_y
    dvel_x__dtime   = grav_acc_x + thrust_acc_x
    dvel_y__dtime   = grav_acc_y + thrust_acc_y
    dmass__dtime    = -thrust_mag / exhaust_velocity
    dcopos_x__dtime = 0.0
    dcopos_y__dtime = 0.0
    dcovel_x__dtime = -copos_x
    dcovel_y__dtime = -copos_y
    dcomass__dtime  = thrust_mag/mass**2 * (covel_x*np.cos(thrust_ang) + covel_y*np.sin(thrust_ang)) - covel_y*grav_acc_const/mass

    dstate__dtime    = np.zeros(10, dtype=float)
    dstate__dtime[0] = dpos_x__dtime
    dstate__dtime[1] = dpos_y__dtime
    dstate__dtime[2] = dvel_x__dtime
    dstate__dtime[3] = dvel_y__dtime
    dstate__dtime[4] = dmass__dtime
    dstate__dtime[5] = dcopos_x__dtime
    dstate__dtime[6] = dcopos_y__dtime
    dstate__dtime[7] = dcovel_x__dtime
    dstate__dtime[8] = dcovel_y__dtime
    dstate__dtime[9] = dcomass__dtime

    return dstate__dtime


def forcefreedynamics_2d_minenergy_indirect(
    time  : float,
    state : np.ndarray,
):

    pos_x, pos_y, vel_x, vel_y, copos_x, copos_y, covel_x, covel_y = state[:8]

    thrust_acc_x = -covel_x
    thrust_acc_y = -covel_y

    dpos_x__dtime   = vel_x
    dpos_y__dtime   = vel_y
    dvel_x__dtime   = thrust_acc_x
    dvel_y__dtime   = thrust_acc_y
    dcopos_x__dtime = 0.0
    dcopos_y__dtime = 0.0
    dcovel_x__dtime = -copos_x
    dcovel_y__dtime = -copos_y

    return [
        dpos_x__dtime, dpos_y__dtime, 
        dvel_x__dtime, dvel_y__dtime,
        dcopos_x__dtime, dcopos_y__dtime, 
        dcovel_x__dtime, dcovel_y__dtime
    ]