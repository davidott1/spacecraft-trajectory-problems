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