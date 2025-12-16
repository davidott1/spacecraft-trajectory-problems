"""
Validation Metrics Module
=========================

Utilities for computing validation metrics and comparing propagation results.

Notes:
------
Energy and angular momentum conservation tests are only valid for 
**conservative force models** (gravity, J2/J3/J4 zonal harmonics, third-body gravity).

Non-conservative forces (atmospheric drag, SRP) will cause energy and angular 
momentum to drift over time - this is physically correct behavior:
- Drag removes energy (orbit decays)
- SRP can add or remove energy depending on geometry
"""
import numpy as np

from typing import Optional


def compute_position_error_statistics(
  pos_error : np.ndarray,
) -> dict:
  """
  Compute statistics for position error.
  
  Input:
  ------
    pos_error : np.ndarray
      Position error array, shape (3, N) or (N,) for magnitude.
      
  Output:
  -------
    stats : dict
      Dictionary containing mean, rms, max, and std of errors.
  """
  if pos_error.ndim == 2:
    # 3xN array, compute magnitude
    error_mag = np.linalg.norm(pos_error, axis=0)
  else:
    error_mag = np.abs(pos_error)
  
  return {
    'mean' : np.mean(error_mag),
    'rms'  : np.sqrt(np.mean(error_mag**2)),
    'max'  : np.max(error_mag),
    'std'  : np.std(error_mag),
    'min'  : np.min(error_mag),
  }


def compute_velocity_error_statistics(
  vel_error : np.ndarray,
) -> dict:
  """
  Compute statistics for velocity error.
  
  Input:
  ------
    vel_error : np.ndarray
      Velocity error array, shape (3, N) or (N,) for magnitude.
      
  Output:
  -------
    stats : dict
      Dictionary containing mean, rms, max, and std of errors.
  """
  return compute_position_error_statistics(vel_error)  # Same calculation


def compute_energy_drift(
  states : np.ndarray,
  gp     : float,
) -> np.ndarray:
  """
  Compute specific orbital energy at each time step.
  
  Input:
  ------
    states : np.ndarray
      State vectors, shape (6, N).
    gp : float
      Gravitational parameter [m³/s²].
      
  Output:
  -------
    energy : np.ndarray
      Specific energy at each time step [m²/s²].
  """
  pos_vec = states[0:3, :]
  vel_vec = states[3:6, :]
  
  pos_mag = np.linalg.norm(pos_vec, axis=0)
  vel_mag = np.linalg.norm(vel_vec, axis=0)
  
  return 0.5 * vel_mag**2 - gp / pos_mag


def compute_angular_momentum_drift(
  states : np.ndarray,
) -> np.ndarray:
  """
  Compute angular momentum magnitude at each time step.
  
  Input:
  ------
    states : np.ndarray
      State vectors, shape (6, N).
      
  Output:
  -------
    h_mag : np.ndarray
      Angular momentum magnitude at each time step [m²/s].
  """
  pos_vec = states[0:3, :]
  vel_vec = states[3:6, :]
  
  ang_mom = np.cross(pos_vec.T, vel_vec.T).T
  
  return np.linalg.norm(ang_mom, axis=0)


def compute_rms_error(
  reference : np.ndarray,
  computed  : np.ndarray,
) -> float:
  """
  Compute RMS error between reference and computed values.
  
  Input:
  ------
    reference : np.ndarray
      Reference values.
    computed : np.ndarray
      Computed values.
      
  Output:
  -------
    rms : float
      RMS error.
  """
  diff = computed - reference
  return np.sqrt(np.mean(diff**2))


class PropagatorValidator:
  """
  Comprehensive validation suite for orbital propagator.
  
  Notes:
  ------
  Energy and angular momentum conservation tests should only be used
  with conservative force models (gravity only, no drag/SRP).
  For non-conservative forces, use comparison with reference trajectories instead.
  """
  
  def __init__(
    self,
    propagator_func,
    gp : float,
  ):
    """
    Initialize validator.
    
    Input:
    ------
      propagator_func : callable
        Function that propagates state: propagator_func(state_0, t_span, **kwargs) -> states, times
      gp : float
        Gravitational parameter [m³/s²].
    """
    self.propagator_func = propagator_func
    self.gp              = gp
  
  def test_energy_conservation(
    self,
    initial_state : np.ndarray,
    time_span     : tuple,
    tolerance     : float = 1e-10,
  ) -> dict:
    """
    Test energy conservation for conservative force model.
    
    WARNING: This test is only valid for conservative forces (gravity, J2, third-body).
    Non-conservative forces (drag, SRP) will cause energy drift by design.
    
    Input:
    ------
      initial_state : np.ndarray
        Initial state vector [pos, vel].
      time_span : tuple
        (t0, tf) time span for propagation.
      tolerance : float
        Acceptable relative energy drift.
        
    Output:
    -------
      result : dict
        Dictionary with 'passed', 'energy_drift', 'message'.
    """
    states, times = self.propagator_func(initial_state, time_span)
    
    energy = compute_energy_drift(states, self.gp)
    
    energy_o  = energy[0]
    rel_drift = np.abs((energy - energy_o) / energy_o)
    max_drift = np.max(rel_drift)
    
    passed = max_drift < tolerance
    
    return {
      'passed'       : passed,
      'max_drift'    : max_drift,
      'tolerance'    : tolerance,
      'energy_array' : energy,
      'message'      : f"Max relative energy drift: {max_drift:.2e} (tolerance: {tolerance:.2e})",
    }
  
  def test_angular_momentum_conservation(
    self,
    initial_state : np.ndarray,
    time_span     : tuple,
    tolerance     : float = 1e-10,
  ) -> dict:
    """
    Test angular momentum conservation.
    
    WARNING: This test is only valid for central-force models (two-body gravity).
    J2 and other non-spherical perturbations can cause angular momentum direction to change.
    Non-conservative forces (drag, SRP) will also cause angular momentum drift.
    
    Input:
    ------
      initial_state : np.ndarray
        Initial state vector [pos, vel].
      time_span : tuple
        (t0, tf) time span for propagation.
      tolerance : float
        Acceptable relative angular momentum drift.
        
    Output:
    -------
      result : dict
        Dictionary with 'passed', 'h_drift', 'message'.
    """
    states, times = self.propagator_func(initial_state, time_span)
    
    ang_mom_mag = compute_angular_momentum_drift(states)
    
    ang_mom_o = ang_mom_mag[0]
    rel_drift = np.abs((ang_mom_mag - ang_mom_o) / ang_mom_o)
    max_drift = np.max(rel_drift)
    
    passed = max_drift < tolerance
    
    return {
      'passed'        : passed,
      'max_drift'     : max_drift,
      'tolerance'     : tolerance,
      'ang_mom_array' : ang_mom_mag,
      'message'       : f"Max relative angular momentum drift: {max_drift:.2e} (tolerance: {tolerance:.2e})",
    }
  
  def compare_with_reference(
    self,
    computed_states  : np.ndarray,
    reference_states : np.ndarray,
    times            : np.ndarray,
  ) -> dict:
    """
    Compare computed trajectory with reference.
    
    This method works for any force model (conservative or non-conservative).
    
    Input:
    ------
      computed_states : np.ndarray
        Computed state vectors, shape (6, N).
      reference_states : np.ndarray
        Reference state vectors, shape (6, N).
      times : np.ndarray
        Time array.
        
    Output:
    -------
      result : dict
        Dictionary with position and velocity error statistics.
    """
    pos_error = computed_states[0:3, :] - reference_states[0:3, :]
    vel_error = computed_states[3:6, :] - reference_states[3:6, :]
    
    return {
      'position_error_stats' : compute_position_error_statistics(pos_error),
      'velocity_error_stats' : compute_velocity_error_statistics(vel_error),
      'pos_error'            : pos_error,
      'vel_error'            : vel_error,
      'times'                : times,
    }
