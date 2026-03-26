"""
Maneuver Optimizer
==================

General-purpose optimizer for impulsive maneuver plans. Adjusts decision
variables (maneuver times, ΔV components, initial state) to minimize total
ΔV while satisfying trajectory constraints.

Uses scipy.optimize.minimize with the method specified in OptimizationConfig.

Key Functions:
--------------
  optimize_maneuver_plan  - Top-level entry point
  pack_decision_vector    - DecisionState → flat array
  unpack_decision_vector  - flat array → DecisionState updates
  evaluate_objective      - Objective function for scipy

Units:
------
  Position : meters [m]
  Velocity : meters per second [m/s]
  Time     : seconds [s]
"""

import numpy as np

from copy     import deepcopy
from datetime import datetime, timedelta
from typing   import Optional, List, Tuple

from scipy.optimize import minimize

from src.model.constants  import SOLARSYSTEMCONSTANTS
from src.model.dynamics   import AccelerationSTMDot
from src.propagation.propagator import propagate_with_maneuvers
from src.schemas.optimization   import DecisionState, OptimizationConfig, OptimizationResult
from src.schemas.propagation    import PropagationConfig


# --------------------------------------------------------------------------
# Decision Vector Packing / Unpacking
# --------------------------------------------------------------------------

def pack_decision_vector(
  decision_state : DecisionState,
) -> np.ndarray:
  """
  Pack variable components from a DecisionState into a flat 1-D array.

  The order is:
    [epoch_offset_s, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
     maneuver_0_time_offset_s, maneuver_0_dv_x, maneuver_0_dv_y, maneuver_0_dv_z,
     maneuver_1_time_offset_s, ...]

  Only components whose variable flag is True are included.
  Epoch and maneuver times are stored as offsets in seconds from their
  nominal values so the optimizer works in natural units.

  Input:
  ------
    decision_state : DecisionState

  Output:
  -------
    x : np.ndarray, shape (n_variables,)
  """
  parts = []

  if decision_state.variable_epoch:
    parts.append(0.0)  # offset from current epoch

  for i in range(3):
    if decision_state.variable_position[i]:
      parts.append(decision_state.position[i])

  for i in range(3):
    if decision_state.variable_velocity[i]:
      parts.append(decision_state.velocity[i])

  if decision_state.maneuvers is not None:
    for j, m in enumerate(decision_state.maneuvers):
      if j < len(decision_state.variable_maneuver_time) and decision_state.variable_maneuver_time[j]:
        parts.append(0.0)  # offset from current maneuver time

      if j < len(decision_state.variable_maneuver_delta_v):
        dv_flags = decision_state.variable_maneuver_delta_v[j]
        for k in range(3):
          if dv_flags[k]:
            parts.append(m.delta_vel_vec[k])

  return np.array(parts, dtype=float)


def unpack_decision_vector(
  x              : np.ndarray,
  decision_state : DecisionState,
) -> None:
  """
  Unpack a flat decision vector back into a DecisionState (in-place).

  Input:
  ------
    x              : np.ndarray, shape (n_variables,)
    decision_state : DecisionState (modified in-place)
  """
  idx = 0

  if decision_state.variable_epoch:
    offset_s = x[idx]; idx += 1
    decision_state.epoch = decision_state.epoch + timedelta(seconds=float(offset_s))

  for i in range(3):
    if decision_state.variable_position[i]:
      decision_state.position[i] = x[idx]; idx += 1

  for i in range(3):
    if decision_state.variable_velocity[i]:
      decision_state.velocity[i] = x[idx]; idx += 1

  if decision_state.maneuvers is not None:
    for j, m in enumerate(decision_state.maneuvers):
      if j < len(decision_state.variable_maneuver_time) and decision_state.variable_maneuver_time[j]:
        offset_s = x[idx]; idx += 1
        m.time_dt = m.time_dt + timedelta(seconds=float(offset_s))

      if j < len(decision_state.variable_maneuver_delta_v):
        dv_flags = decision_state.variable_maneuver_delta_v[j]
        for k in range(3):
          if dv_flags[k]:
            m.delta_vel_vec[k] = x[idx]; idx += 1


# --------------------------------------------------------------------------
# Objective Function
# --------------------------------------------------------------------------

def evaluate_objective(
  x                : np.ndarray,
  decision_state_0 : DecisionState,
  dynamics         : AccelerationSTMDot,
  propagation_config : PropagationConfig,
  config           : OptimizationConfig,
) -> float:
  """
  Objective function for scipy.optimize.minimize.

  1. Deep-copies the baseline DecisionState
  2. Unpacks x into the copy
  3. Propagates the trajectory with maneuvers
  4. Returns total ΔV magnitude (sum of all burns)

  If propagation fails, returns a large penalty value.

  Input:
  ------
    x                  : Decision vector
    decision_state_0   : Baseline DecisionState (not modified)
    dynamics           : Acceleration model
    propagation_config : Time span and integration settings
    config             : Optimizer settings (tolerances, penalty weight)

  Output:
  -------
    cost : float
      Total ΔV [m/s], or penalty if propagation fails.
  """
  ds = deepcopy(decision_state_0)
  unpack_decision_vector(x, ds)

  # Reconstruct initial state
  initial_state = np.concatenate([ds.position, ds.velocity])

  # Check maneuver time ordering: burns must be after epoch and in sequence
  if ds.maneuvers and len(ds.maneuvers) > 0:
    prev_time = ds.epoch
    for m in ds.maneuvers:
      if m.time_dt <= prev_time:
        return config.penalty_weight  # bad ordering
      prev_time = m.time_dt
    if ds.maneuvers[-1].time_dt >= propagation_config.time_f_dt:
      return config.penalty_weight  # burn after end time

  # Propagate
  try:
    result = propagate_with_maneuvers(
      initial_state = initial_state,
      initial_dt    = ds.epoch,
      final_dt      = propagation_config.time_f_dt,
      dynamics      = dynamics,
      maneuvers     = list(ds.maneuvers) if ds.maneuvers else [],
      method        = 'DOP853',
      rtol          = config.rtol,
      atol          = config.atol,
    )
  except Exception:
    return config.penalty_weight

  if not result.success:
    return config.penalty_weight

  # Objective: total ΔV
  total_dv = sum(m.mag() for m in ds.maneuvers) if ds.maneuvers else 0.0

  return total_dv


# --------------------------------------------------------------------------
# Top-Level Optimizer
# --------------------------------------------------------------------------

def optimize_maneuver_plan(
  decision_state     : DecisionState,
  dynamics           : AccelerationSTMDot,
  propagation_config : PropagationConfig,
  optimization_config : OptimizationConfig = None,
  verbose            : bool = True,
) -> OptimizationResult:
  """
  Optimize a maneuver plan by adjusting variable quantities to minimize total ΔV.

  Input:
  ------
    decision_state      : DecisionState with initial guess and variable flags
    dynamics            : High-fidelity acceleration model
    propagation_config  : Time span and integration tolerances
    optimization_config : Solver settings (optional, uses defaults)
    verbose             : Print progress during optimization

  Output:
  -------
    result : OptimizationResult
      Contains success flag, optimized trajectory info, final ΔV, iteration counts.
      On success, decision_state is updated in-place with optimized values.
  """
  if optimization_config is None:
    optimization_config = OptimizationConfig()

  if not decision_state.has_any_variable():
    return OptimizationResult(
      success  = False,
      message  = "No variable quantities to optimize.",
    )

  # Save baseline for the objective function (deep copy so modifications don't affect it)
  decision_state_baseline = deepcopy(decision_state)

  # Pack initial guess
  x0 = pack_decision_vector(decision_state)

  if verbose:
    print(f"    Method:    {optimization_config.method}")
    print(f"    Variables: {len(x0)}")
    print(f"    Max iter:  {optimization_config.maxiter}")
    initial_dv = sum(m.mag() for m in decision_state.maneuvers) if decision_state.maneuvers else 0.0
    print(f"    Initial ΔV: {initial_dv:.4f} m/s")

  # Iteration counter for progress reporting
  eval_count = [0]

  def callback(xk):
    eval_count[0] += 1
    if verbose and eval_count[0] % 10 == 0:
      ds_temp = deepcopy(decision_state_baseline)
      unpack_decision_vector(xk, ds_temp)
      current_dv = sum(m.mag() for m in ds_temp.maneuvers) if ds_temp.maneuvers else 0.0
      print(f"      Iteration {eval_count[0]:4d}:  ΔV = {current_dv:.4f} m/s")

  # Build scipy options
  options = {
    'maxiter': optimization_config.maxiter,
    'xatol':   optimization_config.xatol,
    'fatol':   optimization_config.fatol,
    'adaptive': True,
  }

  method = optimization_config.method.lower()

  # For methods that don't support xatol/fatol, adjust options
  if method in ('cobyla', 'powell', 'bfgs', 'l-bfgs-b'):
    options = {
      'maxiter': optimization_config.maxiter,
    }

  # Run optimizer
  scipy_result = minimize(
    fun      = evaluate_objective,
    x0       = x0,
    args     = (decision_state_baseline, dynamics, propagation_config, optimization_config),
    method   = method,
    callback = callback,
    options  = options,
  )

  # Unpack optimized values into the original decision_state (in-place update)
  unpack_decision_vector(scipy_result.x, decision_state)

  final_dv = sum(m.mag() for m in decision_state.maneuvers) if decision_state.maneuvers else 0.0

  if verbose:
    print(f"    Final ΔV:   {final_dv:.4f} m/s")
    print(f"    Iterations: {scipy_result.nit}")
    print(f"    Func evals: {scipy_result.nfev}")
    print(f"    Success:    {scipy_result.success}")
    print(f"    Message:    {scipy_result.message}")

  return OptimizationResult(
    success          = scipy_result.success,
    message          = scipy_result.message,
    objective_value  = final_dv,
    n_iterations     = scipy_result.nit,
    n_function_evals = scipy_result.nfev,
  )
