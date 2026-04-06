"""
Initial Guess Module
====================

Generates initial trajectory guesses for the maneuver optimizer using
low-fidelity analytical methods.

Currently implements a patched conic grid search for circular-orbit-to-
circular-orbit transfers in the Earth-Moon system.

Algorithm (PatchedConicGridSearch):
-----------------------------------
  1. Compute Hohmann transfer estimates as baseline
  2. Grid search over departure times and ΔV₁ magnitudes
  3. For each candidate:
     a. Propagate parking orbit to departure time (two-body)
     b. Apply ΔV₁ prograde burn
     c. Propagate transfer orbit under Earth two-body gravity
     d. Detect Moon SOI crossing using SPICE ephemeris
     e. Transform to Moon-centered frame at SOI boundary
     f. Propagate under Moon two-body gravity to periapsis
     g. Compute ΔV₂ for circularization at periapsis
  4. Refine best candidate with scipy.optimize.minimize
  5. Build output trajectory with Nodes and Segments

Classes:
--------
  PatchedConicGridSearch : 2-D grid search (departure time × ΔV₁) for
                           circular-to-circular transfers via patched conics.

Usage:
------
  from src.optimization.initial_guess import PatchedConicGridSearch
  from src.schemas.optimization import OptimizationProblem, OptimizationConfig
  from src.schemas.optimization import DecisionState, Objective, BoundaryCondition, Constraint

  problem = OptimizationProblem(
    objective           = Objective(quantity='delta_v_total', nodes=[0, 2]),
    decision_state      = DecisionState(epoch=datetime(2025, 10, 1)),
    constraints         = Constraint(final=[BoundaryCondition(node=2, quantity='altitude', target=100_000.0)]),
    optimization_config = OptimizationConfig(method='nelder-mead'),
  )

  search = PatchedConicGridSearch(problem, initial_state_j2000)
  result = search.solve()
"""
import numpy as np

from datetime        import datetime, timedelta
from scipy.optimize  import minimize
from typing          import Optional

from src.model.constants         import SOLARSYSTEMCONSTANTS, NAIFIDS, CONVERTER
from src.model.time_converter    import utc_to_et, et_to_utc
from src.model.orbital_mechanics import compute_circular_velocity
from src.schemas.time            import TimeStructure
from src.schemas.optimization    import OptimizationProblem, OptimizationConfig, OptimizationResult, Segment, Node, Trajectory

from src.model.orbital_mechanics           import compute_hohmann_velocities, compute_soi_radius
from src.model.frame_and_vector_converter  import BodyVectorConverter
from src.propagation.analytical_propagator import (
  propagate_to_soi,
  propagate_to_periapsis,
  propagate_two_body,
)


class PatchedConicGridSearch:
  """
  Patched conic grid search for circular-to-circular orbit transfers.

  Finds the minimum-ΔV transfer between two circular orbits in a two-body
  system with one SOI patch point. Currently configured for Earth-Moon
  transfers (circular LEO → circular LLO).

  The search grids over departure time and ΔV₁ magnitude, evaluates each
  candidate using patched conic propagation (Earth two-body → SOI → Moon
  two-body), and returns the best transfer found.

  Assumptions:
    - Patched conic approximation (two-body only, no perturbations)
    - Impulsive maneuvers (instantaneous ΔV)
    - Circular departure orbit (ΔV₁ applied prograde)
    - Circular target orbit (ΔV₂ computed for circularization)
    - Moon SOI as the single patch point
    - Moon position from SPICE ephemeris (DE440)

  Attributes:
    problem          : OptimizationProblem
    initial_state    : np.ndarray (6,) - Initial state in J2000 [m, m/s]
    radius_leo       : float - Departure orbit radius [m]
    radius_llo       : float - Target orbit radius [m]
    vel_mag_circ_leo : float - Circular velocity at departure orbit [m/s]
    vel_mag_circ_llo : float - Circular velocity at target orbit [m/s]
    soi_moon         : float - Moon's sphere of influence radius [m]
    hohmann          : dict  - Hohmann transfer estimates
  """

  # Central body constants
  EARTH_GP     = SOLARSYSTEMCONSTANTS.EARTH.GP
  EARTH_RADIUS = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
  MOON_GP      = SOLARSYSTEMCONSTANTS.MOON.GP
  MOON_RADIUS  = SOLARSYSTEMCONSTANTS.MOON.RADIUS.EQUATOR
  MOON_SMA     = SOLARSYSTEMCONSTANTS.MOON.SMA

  def __init__(
    self,
    problem                                : OptimizationProblem,
    initial_state                          : np.ndarray,
    leo_altitude__m                        : float = 200_000.0,
    llo_altitude__m                        : float = 100_000.0,
    max_transfer_time__s                   : float = 7.0 * 86400.0,
    delta_vel_mag_1_search_bounds__m_per_s : tuple = (2800.0, 3400.0),
    departure_search_window__s             : float = 30.0 * 86400.0,
    n_departure_candidates                 : int   = 720,
    llo_coast_orbits                       : int   = 3,
  ):
    """
    Initialize the patched conic grid search.

    Input:
    ------
      problem : OptimizationProblem
        Optimization problem definition (objective, constraints, solver settings).
        The departure epoch is taken from problem.decision_state.epoch.
      initial_state : np.ndarray (6,)
        Initial spacecraft state in Earth-centered J2000 [m, m/s].
        Should be on a circular orbit at the departure epoch.
      leo_altitude__m : float
        Departure orbit altitude above Earth surface [m].
      llo_altitude__m : float
        Target orbit altitude above Moon surface [m].
      max_transfer_time__s : float
        Maximum transfer time from departure to Moon SOI [s].
      delta_vel_mag_1_search_bounds__m_per_s : tuple
        Search bounds for ΔV₁ magnitude [m/s].
      departure_search_window__s : float
        Search window for departure time [s] from departure epoch.
      n_departure_candidates : int
        Number of departure time candidates in grid search.
      llo_coast_orbits : int
        Number of target orbits to propagate after insertion.
    """
    self.problem       = problem
    self.config        = problem.optimization_config
    self.initial_state = initial_state.copy()

    # Problem-specific parameters
    self.leo_altitude__m            = leo_altitude__m
    self.llo_altitude__m            = llo_altitude__m
    self.max_transfer_time__s       = max_transfer_time__s
    self.delta_vel_mag_1_search_bounds__m_per_s = delta_vel_mag_1_search_bounds__m_per_s
    self.departure_search_window__s = departure_search_window__s
    self.n_departure_candidates    = n_departure_candidates
    self.llo_coast_orbits          = llo_coast_orbits

    # Departure epoch from decision state
    self.departure_epoch = problem.decision_state.epoch

    # Derived orbital radii
    self.radius_leo = self.EARTH_RADIUS + leo_altitude__m
    self.radius_llo = self.MOON_RADIUS  + llo_altitude__m

    # Circular velocities
    self.vel_mag_circ_leo = compute_circular_velocity(self.radius_leo, self.EARTH_GP)
    self.vel_mag_circ_llo = compute_circular_velocity(self.radius_llo, self.MOON_GP)

    # Moon's sphere of influence
    self.soi_moon = compute_soi_radius(self.MOON_SMA, self.EARTH_GP, self.MOON_GP)

    # Hohmann transfer estimates (Earth-only, to Moon orbit distance)
    self.hohmann = compute_hohmann_velocities(self.radius_leo, self.MOON_SMA, self.EARTH_GP)


  def solve(self) -> OptimizationResult:
    """
    Find optimal patched conic transfer (convenience wrapper).

    Runs grid_search(), then refine(), then builds and returns the full result.

    Output:
    -------
      result : OptimizationResult
        Transfer solution with trajectory, ΔV breakdown, and timing.
    """
    t0_et = utc_to_et(self.departure_epoch)

    # Verify SPICE is loaded
    try:
      BodyVectorConverter.get_body_state(NAIFIDS.MOON, t0_et, NAIFIDS.EARTH)
    except Exception as e:
      return OptimizationResult(
        success = False,
        message = f"SPICE kernels not loaded. Call load_files() first. Error: {e}",
      )

    # Print header
    title = "Patched Conic Grid Search (Circular-to-Circular Transfer)"
    print("\n" + "-" * len(title))
    print(title)
    print("-" * len(title))

    self._print_configuration()

    # Phase 1: Grid search
    print()
    print("  Progress")
    grid_result = self.grid_search()

    if grid_result is None:
      return OptimizationResult(
        success = False,
        message = "No valid transfer found. The Moon may not be accessible from "
                  "this orbit plane within the search window.",
      )

    # Phase 2: Local refinement
    refined_result = self.refine(grid_result)

    # Phase 3: Build full trajectory
    print("    Phase 3: Building trajectory")
    t_depart_et_final = refined_result['t_depart_et']
    result = self._build_result(t_depart_et_final, refined_result['eval'])

    # Print summary
    self._print_summary(result, refined_result['eval'])

    return result


  def grid_search(self) -> Optional[dict]:
    """
    Phase 1: Grid search over departure times and ΔV₁ magnitudes.

    Scans n_departure_candidates evenly spaced departure times across the
    search window, with multiple ΔV₁ multipliers at each time.

    Output:
    -------
      best : dict | None
        Best candidate found, or None if no valid transfer exists.
        Keys:
          'eval'            : dict from _evaluate_transfer
          'delta_vel_total' : float - total ΔV [m/s]
          'offset__s'       : float - departure time offset from epoch [s]
          'delta_vel_mag_1' : float - ΔV₁ magnitude [m/s]
    """
    t0_et = utc_to_et(self.departure_epoch)

    print("    Phase 1: Grid search over departure times")

    search_window           = self.departure_search_window__s
    n_candidates             = self.n_departure_candidates
    delta_vel_mag_1_initial  = self.hohmann['delta_vel_mag_o']

    best_result              = None
    best_delta_vel_total     = np.inf
    best_offset              = 0.0
    best_delta_vel_mag_1     = delta_vel_mag_1_initial
    n_soi_crossings = 0

    departure_offsets = np.linspace(0, search_window, n_candidates)

    for i, dt_offset in enumerate(departure_offsets):
      # Progress indicator
      if (i + 1) % max(1, n_candidates // 10) == 0:
        pct = 100 * (i + 1) / n_candidates
        print(f"      {pct:5.1f}%  ({n_soi_crossings} SOI crossings found)")

      t_depart_et = t0_et + dt_offset

      # Try a range of ΔV₁ values around Hohmann estimate
      for delta_vel_mag_1_mult in [0.98, 1.0, 1.02, 1.05]:
        delta_vel_mag_1_candidate = delta_vel_mag_1_initial * delta_vel_mag_1_mult
        result = self._evaluate_transfer(t_depart_et, delta_vel_mag_1_candidate)

        if result is not None:
          n_soi_crossings += 1
          if result['delta_vel_total'] < best_delta_vel_total:
            best_delta_vel_total = result['delta_vel_total']
            best_result          = result
            best_offset          = dt_offset
            best_delta_vel_mag_1 = delta_vel_mag_1_candidate

    print(f"      Grid search complete: {n_soi_crossings} total SOI crossings")

    if best_result is None:
      print("      [ERROR] No valid transfer found in search window")
      print()
      print("  Suggestions:")
      print("    - Expand the search window (departure_search_window__s)")
      print("    - Check that the departure orbit plane passes near the Moon")
      print("    - Try a different departure epoch")
      return None

    return {
      'eval'            : best_result,
      'delta_vel_total' : best_delta_vel_total,
      'offset__s'       : best_offset,
      'delta_vel_mag_1' : best_delta_vel_mag_1,
    }


  def refine(self, candidate: dict) -> dict:
    """
    Phase 2: Local optimization around a grid search candidate.

    Uses scipy.optimize.minimize to refine the departure time offset and
    ΔV₁ magnitude from the best grid search result.

    Input:
    ------
      candidate : dict
        Output from grid_search().

    Output:
    -------
      refined : dict
        Refined result with keys:
          'eval'            : dict from _evaluate_transfer
          'delta_vel_total' : float
          'offset__s'       : float
          'delta_vel_mag_1' : float
          't_depart_et'     : float
    """
    t0_et = utc_to_et(self.departure_epoch)
    search_window = self.departure_search_window__s

    best_offset          = candidate['offset__s']
    best_delta_vel_mag_1 = candidate['delta_vel_mag_1']
    best_delta_vel_total = candidate['delta_vel_total']
    best_eval            = candidate['eval']

    departure_dt_best = self.departure_epoch + timedelta(seconds=best_offset)
    print(f"    Phase 2: Refining best solution")
    print(f"      Best grid result: ΔV_total = {best_delta_vel_total:.2f} m/s "
          f"at offset = {best_offset/86400:.2f} days "
          f"({departure_dt_best.strftime('%Y-%m-%d %H:%M')} UTC)")

    def objective(x):
      dt_offset_opt, delta_vel_mag_1_opt = x
      # Clamp to valid range
      dt_offset_opt       = max(0, min(search_window, dt_offset_opt))
      delta_vel_mag_1_opt = max(self.delta_vel_mag_1_search_bounds__m_per_s[0],
                                min(self.delta_vel_mag_1_search_bounds__m_per_s[1], delta_vel_mag_1_opt))
      t_depart_et_opt = t0_et + dt_offset_opt
      result_opt = self._evaluate_transfer(t_depart_et_opt, delta_vel_mag_1_opt)
      if result_opt is None:
        return 1e10
      return result_opt['delta_vel_total']

    x0 = [best_offset, best_delta_vel_mag_1]
    opt = minimize(
      objective, x0,
      method  = self.config.method,
      options = {'xatol': self.config.xatol, 'fatol': self.config.fatol,
                 'maxiter': self.config.maxiter, 'adaptive': True},
    )

    # Evaluate final solution
    final_offset, final_delta_vel_mag_1 = opt.x
    final_offset          = max(0, min(search_window, final_offset))
    final_delta_vel_mag_1 = max(self.delta_vel_mag_1_search_bounds__m_per_s[0],
                                min(self.delta_vel_mag_1_search_bounds__m_per_s[1], final_delta_vel_mag_1))
    t_depart_et_final = t0_et + final_offset
    final_eval = self._evaluate_transfer(t_depart_et_final, final_delta_vel_mag_1)

    if final_eval is None or final_eval['delta_vel_total'] >= best_delta_vel_total:
      # Refinement didn't improve; use grid search result
      final_eval        = best_eval
      final_offset      = best_offset
      t_depart_et_final = t0_et + final_offset

    departure_dt_final = self.departure_epoch + timedelta(seconds=final_offset)
    print(f"      Optimized: ΔV_total = {final_eval['delta_vel_total']:.2f} m/s "
          f"at offset = {final_offset/86400:.2f} days "
          f"({departure_dt_final.strftime('%Y-%m-%d %H:%M')} UTC)")

    return {
      'eval'            : final_eval,
      'delta_vel_total' : final_eval['delta_vel_total'],
      'offset__s'       : final_offset,
      'delta_vel_mag_1' : final_delta_vel_mag_1,
      't_depart_et'     : t_depart_et_final,
    }


  def _print_configuration(self):
    """Print search configuration and Hohmann estimates."""
    print()
    print("  Configuration")
    print(f"    LEO Altitude         : {self.leo_altitude__m/1000:.1f} km")
    print(f"    LLO Altitude         : {self.llo_altitude__m/1000:.1f} km")
    print(f"    LEO Radius           : {self.radius_leo/1000:.1f} km")
    print(f"    LLO Radius           : {self.radius_llo/1000:.1f} km")
    print(f"    V_circ LEO           : {self.vel_mag_circ_leo:.2f} m/s")
    print(f"    V_circ LLO           : {self.vel_mag_circ_llo:.2f} m/s")
    print(f"    Moon SOI Radius      : {self.soi_moon/1000:.1f} km")
    print(f"    Departure Epoch      : {self.departure_epoch}")
    print(f"    Search Window        : {self.departure_search_window__s/86400:.1f} days")
    print(f"    Max Transfer Time    : {self.max_transfer_time__s/86400:.1f} days")

    print()
    print("  Hohmann Transfer Estimates")
    print(f"    ΔV₁ (departure)      : {self.hohmann['delta_vel_mag_o']:.2f} m/s")
    print(f"    ΔV₂ (arrival)        : {self.hohmann['delta_vel_mag_f']:.2f} m/s")
    print(f"    Total ΔV             : {self.hohmann['delta_vel_total']:.2f} m/s")
    print(f"    Transfer Time        : {self.hohmann['delta_time_of']/86400:.2f} days")
    print(f"    Transfer SMA         : {self.hohmann['sma_of']/1000:.1f} km")


  def _evaluate_transfer(
    self,
    t_depart_et     : float,
    delta_vel_mag_1 : float,
  ) -> Optional[dict]:
    """
    Evaluate a single transfer attempt.

    Input:
    ------
      t_depart_et : float
        Departure ephemeris time [s past J2000].
      delta_vel_mag_1 : float
        ΔV₁ magnitude [m/s] (prograde tangential burn).

    Output:
    -------
      result : dict | None
        Transfer evaluation results, or None if transfer fails.
        Keys:
          'delta_vel_mag_1'    : float
          'delta_vel_vec_1'    : np.ndarray (3,)
          'delta_vel_mag_2'    : float
          'delta_vel_vec_2'    : np.ndarray (3,)
          'delta_vel_total'    : float
          'soi_result'         : dict (from propagate_to_soi)
          'peri_result'        : dict (from propagate_to_periapsis)
          'state_soi_moon'     : np.ndarray (6,)
          'v_infinity'         : float
          't_depart_et'        : float
          't_soi'              : float
          'state_at_departure' : np.ndarray (6,)
          'state_post_burn'    : np.ndarray (6,)
          'radius_peri'        : float
    """
    # Propagate parking orbit to departure time
    t0_et = utc_to_et(self.departure_epoch)
    dt_coast = t_depart_et - t0_et

    if abs(dt_coast) > 1.0:
      coast_result = propagate_two_body(
        state0   = self.initial_state,
        t0_et    = t0_et,
        tf_et    = t_depart_et,
        gp       = SOLARSYSTEMCONSTANTS.EARTH.GP,
        n_points = 2,
      )
      if not coast_result['success']:
        return None
      state_at_departure = coast_result['states'][:, -1]
    else:
      state_at_departure = self.initial_state.copy()

    # Apply ΔV₁ (prograde tangential burn)
    vel_vec         = state_at_departure[3:6]
    vel_hat         = vel_vec / np.linalg.norm(vel_vec)
    delta_vel_vec_1 = delta_vel_mag_1 * vel_hat

    state_post_burn      = state_at_departure.copy()
    state_post_burn[3:6] += delta_vel_vec_1

    # Propagate transfer orbit to Moon SOI
    soi_result = propagate_to_soi(
      state_o           = state_post_burn,
      time_et_o         = t_depart_et,
      gp                = self.EARTH_GP,
      naif_id_secondary = NAIFIDS.MOON,
      naif_id_primary   = NAIFIDS.EARTH,
      soi_radius        = self.soi_moon,
      max_time__s       = self.max_transfer_time__s,
      rtol              = self.config.rtol,
      atol              = self.config.atol,
    )

    if not soi_result['success']:
      return None

    # Transform to Moon-centered frame at SOI crossing
    t_soi           = soi_result['soi_time_et']
    state_soi_earth = soi_result['soi_state']
    state_soi_moon  = BodyVectorConverter.j2000_xyz__rel_earth_to_rel_moon(state_soi_earth, t_soi)

    v_infinity = np.linalg.norm(state_soi_moon[3:6])

    # Propagate Moon-centered orbit to periapsis
    peri_result = propagate_to_periapsis(
      state0     = state_soi_moon,
      t0_et      = t_soi,
      gp         = self.MOON_GP,
      max_time__s = 3.0 * 86400.0,  # Max 3 days around Moon
      rtol       = self.config.rtol,
      atol       = self.config.atol,
    )

    if not peri_result['success']:
      return None

    # Check periapsis altitude
    radius_peri = peri_result['periapsis_radius']
    if radius_peri < self.MOON_RADIUS:
      return None  # Lunar impact

    # ΔV₂ for circularization at periapsis
    vel_mag_peri         = peri_result['periapsis_velocity']
    vel_mag_circ_at_peri = compute_circular_velocity(radius_peri, self.MOON_GP)
    delta_vel_mag_2 = abs(vel_mag_peri - vel_mag_circ_at_peri)

    # ΔV₂ direction: retrograde at periapsis (opposite to velocity)
    vel_peri_vec = peri_result['periapsis_state'][3:6]
    vel_peri_hat = vel_peri_vec / np.linalg.norm(vel_peri_vec)
    delta_vel_vec_2 = -delta_vel_mag_2 * vel_peri_hat  # retrograde for capture

    delta_vel_total = delta_vel_mag_1 + delta_vel_mag_2

    return {
      'delta_vel_mag_1'    : delta_vel_mag_1,
      'delta_vel_vec_1'    : delta_vel_vec_1,
      'delta_vel_mag_2'    : delta_vel_mag_2,
      'delta_vel_vec_2'    : delta_vel_vec_2,
      'delta_vel_total'    : delta_vel_total,
      'soi_result'         : soi_result,
      'peri_result'        : peri_result,
      'state_soi_moon'     : state_soi_moon,
      'v_infinity'         : v_infinity,
      't_depart_et'        : t_depart_et,
      't_soi'              : t_soi,
      'state_at_departure' : state_at_departure,
      'state_post_burn'    : state_post_burn,
      'radius_peri'        : radius_peri,
    }


  def _build_result(
    self,
    t_depart_et : float,
    eval_result : dict,
  ) -> OptimizationResult:
    """
    Build OptimizationResult from evaluation output.

    Constructs the full trajectory by:
    1. Propagating Earth departure leg (post-burn to SOI)
    2. Propagating Moon arrival leg (SOI to periapsis)
    3. Propagating LLO coast (post-insertion circular orbit)
    4. Creating Node objects at junction points
    5. Assembling Trajectory from nodes and segments

    Input:
    ------
      t_depart_et : float
        Departure ET.
      eval_result : dict
        Transfer evaluation from _evaluate_transfer.

    Output:
    -------
      result : OptimizationResult
        Complete transfer solution.
    """
    t_soi  = eval_result['t_soi']
    radius_peri = eval_result['radius_peri']

    departure_dt = et_to_utc(t_depart_et)

    # --------------------------------------------------
    # Earth departure leg (transfer orbit: departure → SOI)
    # --------------------------------------------------
    earth_times  = eval_result['soi_result']['trajectory_times']
    earth_states = eval_result['soi_result']['trajectory_states']

    earth_leg = Segment(
      name            = 'earth_departure',
      central_body    = 'EARTH',
      j2000_state_vec = earth_states,
      time            = TimeStructure(
        initial                = et_to_utc(earth_times[0]),
        grid_relative_initial  = earth_times - earth_times[0],
      ),
    )

    # --------------------------------------------------
    # Lunar arrival leg (SOI → periapsis)
    # --------------------------------------------------
    moon_times  = eval_result['peri_result']['trajectory_times']
    moon_states = eval_result['peri_result']['trajectory_states']

    lunar_leg = Segment(
      name            = 'lunar_arrival',
      central_body    = 'MOON',
      j2000_state_vec = moon_states,
      time            = TimeStructure(
        initial                = et_to_utc(moon_times[0]),
        grid_relative_initial  = moon_times - moon_times[0],
      ),
    )

    # --------------------------------------------------
    # LLO coast (circular orbit after insertion)
    # --------------------------------------------------
    t_insertion = eval_result['peri_result']['periapsis_time_et']
    state_peri_moon = eval_result['peri_result']['periapsis_state']

    # Apply ΔV₂ to circularize
    vel_peri_vec       = state_peri_moon[3:6]
    vel_peri_mag       = np.linalg.norm(vel_peri_vec)
    vel_peri_hat       = vel_peri_vec / vel_peri_mag
    vel_mag_circ_at_peri = compute_circular_velocity(radius_peri, self.MOON_GP)

    state_llo_moon = state_peri_moon.copy()
    state_llo_moon[3:6] = vel_mag_circ_at_peri * vel_peri_hat  # Circular at same direction

    # LLO orbital period
    llo_period      = 2.0 * np.pi * np.sqrt(radius_peri**3 / self.MOON_GP)
    llo_coast_time  = self.llo_coast_orbits * llo_period
    n_llo_points    = max(500, int(llo_coast_time / 10))

    llo_prop = propagate_two_body(
      state0   = state_llo_moon,
      t0_et    = t_insertion,
      tf_et    = t_insertion + llo_coast_time,
      gp       = self.MOON_GP,
      n_points = n_llo_points,
      rtol     = self.config.rtol,
      atol     = self.config.atol,
    )

    llo_times  = llo_prop['times']
    llo_states = llo_prop['states']

    llo_leg = Segment(
      name            = 'llo_coast',
      central_body    = 'MOON',
      j2000_state_vec = llo_states,
      time            = TimeStructure(
        initial                = et_to_utc(llo_times[0]),
        grid_relative_initial  = llo_times - llo_times[0],
      ),
    )

    # --------------------------------------------------
    # Create nodes at junction points
    # --------------------------------------------------
    node_departure = Node(
      name            = 'departure_burn',
      central_body    = 'EARTH',
      time_mns        = TimeStructure(initial=departure_dt, grid_relative_initial=np.array([0.0])),
      time_pls        = TimeStructure(initial=departure_dt, grid_relative_initial=np.array([0.0])),
      j2000_state_vec = eval_result['state_post_burn'],
    )

    soi_dt = et_to_utc(t_soi)
    node_soi = Node(
      name            = 'soi_crossing',
      central_body    = 'MOON',
      time_mns        = TimeStructure(initial=soi_dt, grid_relative_initial=np.array([0.0])),
      time_pls        = TimeStructure(initial=soi_dt, grid_relative_initial=np.array([0.0])),
      j2000_state_vec = eval_result['soi_result']['soi_state'],
    )

    periapsis_dt = et_to_utc(t_insertion)
    node_periapsis = Node(
      name            = 'llo_insertion',
      central_body    = 'MOON',
      time_mns        = TimeStructure(initial=periapsis_dt, grid_relative_initial=np.array([0.0])),
      time_pls        = TimeStructure(initial=periapsis_dt, grid_relative_initial=np.array([0.0])),
      j2000_state_vec = state_peri_moon,
    )

    # --------------------------------------------------
    # Assemble trajectory
    # --------------------------------------------------
    trajectory = Trajectory(elements=[
      node_departure, earth_leg, node_soi, lunar_leg, node_periapsis, llo_leg,
    ])

    # --------------------------------------------------
    # Build result
    # --------------------------------------------------
    return OptimizationResult(
      success         = True,
      message         = "Patched conic transfer solution found",
      trajectory      = trajectory,
      objective_value = eval_result['delta_vel_total'],
    )


  def _print_summary(
    self,
    result      : OptimizationResult,
    eval_result : dict,
  ) -> None:
    """
    Print formatted summary of the transfer solution.
    """
    if not result.success:
      print(f"\n  [ERROR] {result.message}")
      return

    radius_peri        = eval_result['radius_peri']
    t_soi              = eval_result['t_soi']
    t_depart_et        = eval_result['t_depart_et']
    t_insertion        = eval_result['peri_result']['periapsis_time_et']
    periapsis_altitude = radius_peri - self.MOON_RADIUS
    transfer_time      = t_insertion - t_depart_et
    departure_dt       = et_to_utc(t_depart_et)
    arrival_dt         = et_to_utc(t_insertion)

    print()
    print("  Summary")

    # Delta-V breakdown
    print(f"    Delta-V")
    print(f"      ΔV₁ (LEO departure)  : {eval_result['delta_vel_mag_1']:12.4f} m/s")
    print(f"      ΔV₂ (LLO insertion)  : {eval_result['delta_vel_mag_2']:12.4f} m/s")
    print(f"      Total ΔV             : {eval_result['delta_vel_total']:12.4f} m/s")

    # Timing
    print(f"    Timing")
    print(f"      Departure Epoch      : {departure_dt}")
    print(f"      Arrival Epoch        : {arrival_dt}")
    print(f"      Transfer Time        : {transfer_time/86400:.4f} days")
    print(f"                           : {transfer_time/3600:.2f} hours")

    # SOI crossing
    soi_dt = et_to_utc(t_soi)
    print(f"    Moon SOI Crossing")
    print(f"      Time                 : {soi_dt}")
    print(f"      V∞ (hyperbolic excess): {eval_result['v_infinity']:.2f} m/s")

    # Periapsis
    print(f"    Moon Periapsis")
    print(f"      Radius               : {radius_peri/1000:.2f} km")
    print(f"      Altitude             : {periapsis_altitude/1000:.2f} km")

    # LLO
    llo_period = 2.0 * np.pi * np.sqrt(radius_peri**3 / self.MOON_GP)
    print(f"    LLO (post-insertion)")
    print(f"      V_circ               : {compute_circular_velocity(radius_peri, self.MOON_GP):.2f} m/s")
    print(f"      Period               : {llo_period/60:.1f} min")

    # ΔV₁ vector
    delta_vel_vec_1 = eval_result['delta_vel_vec_1']
    print(f"    ΔV₁ Vector (J2000)")
    print(f"      [{delta_vel_vec_1[0]:>12.4f}, {delta_vel_vec_1[1]:>12.4f}, {delta_vel_vec_1[2]:>12.4f}] m/s")

    # ΔV₂ vector
    delta_vel_vec_2 = eval_result['delta_vel_vec_2']
    print(f"    ΔV₂ Vector (Moon-centered)")
    print(f"      [{delta_vel_vec_2[0]:>12.4f}, {delta_vel_vec_2[1]:>12.4f}, {delta_vel_vec_2[2]:>12.4f}] m/s")

    # Trajectory statistics
    if result.trajectory is not None:
      segments = result.trajectory.segments
      total_pts = sum(seg.j2000_state_vec.shape[1] for seg in segments)
      print(f"    Trajectory")
      print(f"      Total Points         : {total_pts}")
      for seg in segments:
        print(f"      {seg.name:<22s}: {seg.j2000_state_vec.shape[1]} points")
