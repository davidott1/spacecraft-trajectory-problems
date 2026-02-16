"""
Optimization Schemas
====================

Dataclasses for trajectory optimization configuration and results.
"""

import numpy as np

from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Optional, List

from src.schemas.propagation import PropagationResult, Time


@dataclass
class LunarTransferConfig:
  """
  Configuration for Earth-to-Moon patched conic transfer optimization.

  Attributes:
    leo_altitude_m            : LEO altitude above Earth surface [m]
    llo_altitude_m            : LLO altitude above Moon surface [m]
    departure_epoch           : Earliest departure epoch (UTC)
    max_transfer_time_s       : Maximum transfer time from departure to Moon SOI [s]
    dv1_search_bounds_m_s     : Search bounds for ΔV₁ magnitude [m/s]
    departure_search_window_s : Search window for departure time [s] from departure_epoch
    n_departure_candidates    : Number of departure time candidates in grid search
    llo_coast_orbits          : Number of LLO orbits to propagate after insertion
    atol                      : Absolute tolerance for numerical integration
    rtol                      : Relative tolerance for numerical integration
  """
  leo_altitude_m            : float               = 200_000.0
  llo_altitude_m            : float               = 100_000.0
  departure_epoch           : Optional[datetime]  = None
  max_transfer_time_s       : float               = 7.0 * 86400.0
  dv1_search_bounds_m_s     : tuple[float, float] = (2800.0, 3400.0)
  departure_search_window_s : float               = 30.0 * 86400.0
  n_departure_candidates    : int                 = 720
  llo_coast_orbits          : int                 = 3
  atol                      : float               = 1e-12
  rtol                      : float               = 1e-12


@dataclass
class TransferLeg:
  """
  Single leg of a patched conic transfer trajectory.

  State is always expressed relative to central_body in the J2000 frame.

  Attributes:
    name            : Leg identifier ('earth_departure', 'lunar_arrival', 'llo_coast')
    central_body    : Central gravitational body for this leg ('EARTH' or 'MOON')
    j2000_state_vec : State array centered on central_body, J2000 frame, shape (6, N) [m, m/s]
    time_grid       : Time for this leg
  """
  name            : str
  central_body    : str
  j2000_state_vec : np.ndarray
  time_grid       : Time


@dataclass
class LunarTransferResult:
  """
  Result of lunar transfer optimization.

  Attributes:
    success               : Whether optimization succeeded
    message               : Status or error message

    delta_vel_vec_1       : ΔV₁ vector in J2000 [m/s]
    delta_vel_vec_2       : ΔV₂ vector in Moon-centered frame [m/s]
    delta_vel_mag_1       : ΔV₁ magnitude [m/s]
    delta_vel_mag_2       : ΔV₂ magnitude [m/s]
    delta_vel_total       : Total ΔV [m/s]

    departure_epoch       : Optimal departure epoch (UTC)
    arrival_epoch         : Epoch at Moon periapsis (UTC)
    transfer_time_s       : Transfer time from departure to Moon periapsis [s]

    soi_crossing_et       : Ephemeris time at Moon SOI crossing [s past J2000]
    soi_state_earth_j2000 : State at SOI crossing in Earth-centered J2000 (6,)
    soi_state_moon        : State at SOI crossing in Moon-centered frame (6,)
    v_infinity_mag        : Hyperbolic excess velocity at Moon SOI [m/s]

    periapsis_radius_m    : Periapsis radius at Moon [m]
    periapsis_altitude_m  : Periapsis altitude above Moon surface [m]

    earth_departure_leg   : TransferLeg for Earth-centered departure phase
    lunar_arrival_leg     : TransferLeg for Moon-centered arrival phase
    llo_coast_leg         : TransferLeg for LLO coast phase

    combined_trajectory   : Full trajectory as PropagationResult in Earth-centered J2000
  """
  success               : bool
  message               : str = ""

  # Delta-V
  delta_vel_vec_1       : Optional[np.ndarray] = None
  delta_vel_vec_2       : Optional[np.ndarray] = None
  delta_vel_mag_1       : float                = 0.0
  delta_vel_mag_2       : float                = 0.0
  delta_vel_total       : float                = 0.0

  # Timing
  departure_epoch       : Optional[datetime]   = None
  arrival_epoch         : Optional[datetime]   = None
  transfer_time_s       : float                = 0.0

  # SOI crossing
  soi_crossing_et       : float                = 0.0
  soi_state_earth_j2000 : Optional[np.ndarray] = None
  soi_state_moon        : Optional[np.ndarray] = None
  v_infinity_mag        : float                = 0.0

  # Periapsis at Moon
  periapsis_radius_m    : float                = 0.0
  periapsis_altitude_m  : float                = 0.0

  # Trajectory legs
  earth_departure_leg   : Optional[TransferLeg]        = None
  lunar_arrival_leg     : Optional[TransferLeg]        = None
  llo_coast_leg         : Optional[TransferLeg]        = None

  # Combined trajectory in Earth-centered J2000
  combined_trajectory   : Optional[PropagationResult]  = None
