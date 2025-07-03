import numpy as np

def rocket_dynamics(
    time: float,
    state: np.ndarray,
    thrust: float = 0.0,
    spec_imp: float = 0.0,
    grav_acc_const: float  = 9.81,
    grav_acc_sea_level: float = 9.81,
) -> np.ndarray:

    pos  = state[0]
    vel  = state[1]
    mass = state[2]

    exhaust_velocity = spec_imp * grav_acc_sea_level

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