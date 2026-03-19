"""
Deluxe Ephemeris Writer
=======================

Writes a detailed ephemeris CSV with segment and node rows.
Segment rows contain the propagated trajectory state at each timestep.
Node rows mark impulsive maneuver locations between propagation arcs.
"""

import csv
import numpy as np

from pathlib  import Path
from datetime import datetime, timedelta
from typing   import Optional

from src.schemas.propagation   import PropagationResult
from src.schemas.optimization  import DecisionState
from src.model.orbit_converter import OrbitConverter
from src.model.constants       import SOLARSYSTEMCONSTANTS, CONVERTER


# CSV column headers
DELUXE_EPHEMERIS_COLUMNS = [
  'type',
  'time_utc',
  'time_et_s',
  'pos_x_m', 'pos_y_m', 'pos_z_m',
  'vel_x_m_s', 'vel_y_m_s', 'vel_z_m_s',
  'sma_m', 'ecc', 'inc_deg', 'raan_deg', 'aop_deg', 'ta_deg',
  'imp_mnvr_x_m_s', 'imp_mnvr_y_m_s', 'imp_mnvr_z_m_s',
  'imp_mnvr_frame',
]


def write_deluxe_ephemeris(
  output_filepath : Path,
  result          : PropagationResult,
  decision_state  : Optional[DecisionState] = None,
  gp              : float = SOLARSYSTEMCONSTANTS.EARTH.GP,
) -> None:
  """
  Write deluxe ephemeris CSV.

  Segment rows contain propagated state at each timestep.
  Node rows mark impulsive maneuver boundaries.

  Input:
  ------
    output_filepath : Path
      Path to write the CSV file.
    result : PropagationResult
      Propagated trajectory result.
    decision_state : DecisionState | None
      Decision state with maneuver information (optional).
    gp : float
      Gravitational parameter for COE computation [m^3/s^2].
  """
  if not result.success or result.state is None:
    return

  output_filepath.parent.mkdir(parents=True, exist_ok=True)

  # Build maneuver time lookup (UTC datetime -> maneuver data)
  maneuver_lookup = {}
  if decision_state is not None and decision_state.maneuvers is not None:
    for m in decision_state.maneuvers:
      maneuver_lookup[m.time_dt] = m

  n_points = result.state.shape[1]

  with open(output_filepath, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(DELUXE_EPHEMERIS_COLUMNS)

    for i in range(n_points):
      state = result.state[:, i]
      pos   = state[0:3]
      vel   = state[3:6]

      # Time
      if result.time and result.time.grid:
        time_utc = result.time.grid.utc[i]
        time_et  = result.time.grid.et[i]
      else:
        time_utc = None
        time_et  = 0.0

      time_utc_str = time_utc.strftime('%Y-%m-%dT%H:%M:%S.%f') if time_utc else ''

      # COE
      try:
        coe = OrbitConverter.pv_to_coe(pos_vec=pos, vel_vec=vel, gp=gp)
        sma  = coe.sma
        ecc  = coe.ecc
        inc  = coe.inc  * CONVERTER.DEG_PER_RAD if coe.inc  is not None else ''
        raan = coe.raan * CONVERTER.DEG_PER_RAD if coe.raan is not None else ''
        aop  = coe.aop  * CONVERTER.DEG_PER_RAD if coe.aop  is not None else ''
        ta   = coe.ta   * CONVERTER.DEG_PER_RAD if coe.ta   is not None else ''
      except Exception:
        sma = ecc = inc = raan = aop = ta = ''

      # Check if this timestep coincides with a maneuver node
      is_node = False
      mnvr_dv = ['', '', '']
      mnvr_frame = ''

      if time_utc is not None:
        for m_time, m in maneuver_lookup.items():
          if abs((time_utc - m_time).total_seconds()) < 0.5:
            is_node = True
            mnvr_dv = [m.delta_vel_vec[0], m.delta_vel_vec[1], m.delta_vel_vec[2]]
            mnvr_frame = m.frame
            break

      row_type = 'node' if is_node else 'segment'

      writer.writerow([
        row_type,
        time_utc_str,
        f'{time_et:.6f}',
        f'{pos[0]:.12e}', f'{pos[1]:.12e}', f'{pos[2]:.12e}',
        f'{vel[0]:.12e}', f'{vel[1]:.12e}', f'{vel[2]:.12e}',
        f'{sma:.12e}' if isinstance(sma, float) else sma,
        f'{ecc:.12e}' if isinstance(ecc, float) else ecc,
        f'{inc:.12e}' if isinstance(inc, float) else inc,
        f'{raan:.12e}' if isinstance(raan, float) else raan,
        f'{aop:.12e}' if isinstance(aop, float) else aop,
        f'{ta:.12e}' if isinstance(ta, float) else ta,
        f'{mnvr_dv[0]:.12e}' if isinstance(mnvr_dv[0], float) else mnvr_dv[0],
        f'{mnvr_dv[1]:.12e}' if isinstance(mnvr_dv[1], float) else mnvr_dv[1],
        f'{mnvr_dv[2]:.12e}' if isinstance(mnvr_dv[2], float) else mnvr_dv[2],
        mnvr_frame,
      ])
