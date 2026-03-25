"""
Optimization Schemas
====================

Dataclasses for trajectory optimization configuration and results.
"""

import numpy as np

from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Optional, List, Union

from src.schemas.time        import TimeStructure
from src.schemas.spacecraft  import ManeuversConfig


@dataclass
class DecisionState:
  """
  Decision state for trajectory optimization.

  Holds the initial state (position, velocity, epoch) and maneuver plan,
  along with per-component flags indicating which quantities are variable
  (optimizable) vs. fixed.

  All defaults are fixed (False). The optimizer only adjusts components
  whose corresponding variable flag is True.

  Attributes:
    position  : Initial position vector [m], shape (3,)
    velocity  : Initial velocity vector [m/s], shape (3,)
    epoch     : Initial epoch (UTC)
    maneuvers : Maneuver plan (list-like container of ImpulsiveManeuver)

    variable_position          : Per-component variable flags for position [pos_x, pos_y, pos_z]
    variable_velocity          : Per-component variable flags for velocity [vel_x, vel_y, vel_z]
    variable_epoch             : Whether epoch is variable
    variable_maneuver_time     : Per-maneuver variable flags for burn time
    variable_maneuver_delta_v  : Per-maneuver per-component variable flags for ΔV [dvx, dvy, dvz]
  """
  # State
  position  : np.ndarray              = field(default_factory=lambda: np.zeros(3))
  velocity  : np.ndarray              = field(default_factory=lambda: np.zeros(3))
  epoch     : Optional[datetime]      = None
  maneuvers : Optional[ManeuversConfig] = None

  # Variable vs. fixed
  variable_position          : np.ndarray  = field(default_factory=lambda: np.array([False, False, False]))
  variable_velocity          : np.ndarray  = field(default_factory=lambda: np.array([False, False, False]))
  variable_epoch             : bool        = False
  variable_maneuver_time     : List[bool]          = field(default_factory=list)
  variable_maneuver_delta_v  : List[np.ndarray]    = field(default_factory=list)

  def has_any_variable(self) -> bool:
    """Return True if any component is marked as variable."""
    if np.any(self.variable_position):
      return True
    if np.any(self.variable_velocity):
      return True
    if self.variable_epoch:
      return True
    if any(self.variable_maneuver_time):
      return True
    if any(np.any(dv) for dv in self.variable_maneuver_delta_v):
      return True
    return False

  @property
  def n_variables(self) -> int:
    """Count the total number of variable (optimizable) components."""
    count = int(np.sum(self.variable_position))
    count += int(np.sum(self.variable_velocity))
    if self.variable_epoch:
      count += 1
    count += sum(1 for t in self.variable_maneuver_time if t)
    count += sum(int(np.sum(dv)) for dv in self.variable_maneuver_delta_v)
    return count


@dataclass
class Segment:
  """
  Continuous trajectory segment.

  State is always expressed relative to central_body in the J2000 frame.

  Attributes:
    name            : Segment identifier (e.g. 'earth_departure', 'lunar_arrival')
    central_body    : Central gravitational body for this segment ('EARTH', 'MOON', etc.)
    j2000_state_vec : State array centered on central_body, J2000 frame, shape (6, N) [m, m/s]
    time            : TimeStructure for this segment
  """
  name            : str
  central_body    : str
  j2000_state_vec : np.ndarray
  time            : TimeStructure


@dataclass
class Node:
  """
  Discrete event point on a trajectory (e.g. maneuver, SOI crossing, periapsis).

  Attributes:
    name              : Node identifier (e.g. 'departure_burn', 'soi_crossing')
    central_body      : Central gravitational body at this node
    time_mns          : TimeStructure immediately before the node (minus side)
    time_pls          : TimeStructure immediately after the node (plus side)
    j2000_state_vec   : State vector at this node, J2000 frame, shape (6,) [m, m/s]
  """
  name            : str
  central_body    : str
  time_mns        : TimeStructure
  time_pls        : TimeStructure
  j2000_state_vec : np.ndarray


@dataclass
class Objective:
  """
  Optimization objective definition.

  Specifies which quantity to minimize and at which nodes.

  Attributes:
    quantity : What to minimize ('delta_v_total')
    nodes    : List of node indices whose contributions are summed
  """
  quantity : str        = 'delta_v_total'
  nodes    : List[int]  = field(default_factory=list)


@dataclass
class BoundaryCondition:
  """
  Equality or inequality constraint at a specific node.

  For equality:   quantity(node) == target
  For inequality: quantity(node) <= target  (upper bound)
                  quantity(node) >= target  (lower bound)

  Attributes:
    node     : Node index where this condition applies
    quantity : Physical quantity to constrain ('altitude', 'velocity_mag', etc.)
    target   : Target value [SI units]
    type     : Constraint type ('equality', 'upper_bound', 'lower_bound')
  """
  node     : int
  quantity : str
  target   : float
  type     : str   = 'equality'


@dataclass
class Constraint:
  """
  Collection of constraints on the trajectory.

  Attributes:
    initial       : Boundary conditions at the first node
    final         : Boundary conditions at the last node
    intermediate  : Boundary conditions at interior nodes (future use)
  """
  initial       : List[BoundaryCondition] = field(default_factory=list)
  final         : List[BoundaryCondition] = field(default_factory=list)
  intermediate  : List[BoundaryCondition] = field(default_factory=list)


@dataclass
class OptimizationConfig:
  """
  Solver settings for trajectory optimization.

  Attributes:
    method         : Optimization method ('nelder-mead', 'cobyla', etc.)
    maxiter        : Maximum number of optimizer iterations
    xatol          : Absolute tolerance on decision variables
    fatol          : Absolute tolerance on objective function value
    penalty_weight : Penalty weight for constraint violations
    box_bounds     : Per-variable [min, max] bounds, shape (n_variables, 2)
    atol           : Absolute tolerance for numerical integration
    rtol           : Relative tolerance for numerical integration
  """
  method         : str            = 'nelder-mead'
  maxiter        : int            = 300
  xatol          : float          = 1.0
  fatol          : float          = 0.1
  penalty_weight : float          = 1e6
  box_bounds     : Optional[np.ndarray] = None
  atol           : float          = 1e-12
  rtol           : float          = 1e-12


@dataclass
class Trajectory:
  """
  Ordered sequence of nodes and segments forming a complete trajectory.

  Elements alternate: Node, Segment, Node, Segment, ..., Node.
  Each element has an order index for sequencing.

  Attributes:
    elements : Ordered list of Nodes and Segments
  """
  elements : List[Union[Node, Segment]] = field(default_factory=list)

  @property
  def nodes(self) -> List[Node]:
    return [e for e in self.elements if isinstance(e, Node)]

  @property
  def segments(self) -> List[Segment]:
    return [e for e in self.elements if isinstance(e, Segment)]

  @property
  def n_nodes(self) -> int:
    return len(self.nodes)

  @property
  def n_segments(self) -> int:
    return len(self.segments)


@dataclass
class OptimizationProblem:
  """
  Complete trajectory optimization problem definition.

  Attributes:
    objective           : What to minimize
    decision_state      : Variables and initial guess
    constraints         : Boundary conditions on the trajectory
    optimization_config : Solver settings
    trajectory          : Result trajectory (populated after solving)
  """
  objective           : Objective
  decision_state      : DecisionState
  constraints         : Constraint
  optimization_config : OptimizationConfig
  trajectory          : Optional[Trajectory] = None


@dataclass
class OptimizationResult:
  """
  Result of trajectory optimization.

  Attributes:
    success               : Whether optimization succeeded
    message               : Status or error message
    trajectory            : Solved trajectory (nodes + segments)
    objective_value       : Final objective function value
    constraint_violations : Residual for each constraint (0 = satisfied)
    n_iterations          : Number of optimizer iterations
    n_function_evals      : Number of objective function evaluations
  """
  success               : bool
  message               : str                    = ""
  trajectory            : Optional[Trajectory]   = None
  objective_value       : float                  = 0.0
  constraint_violations : List[float]            = field(default_factory=list)
  n_iterations          : int                    = 0
  n_function_evals      : int                    = 0
