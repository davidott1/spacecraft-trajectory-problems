"""
Gravity Field Module
====================

Handles loading and computing spherical harmonic gravity accelerations
using standard gravity field coefficient files (ICGEM format).

Supported formats:
- ICGEM .gfc files (EGM2008, EGM96, etc.)

References:
- Montenbruck & Gill, "Satellite Orbits", Chapter 3.2
- EGM2008: Pavlis et al. (2012)
"""

import numpy as np

from pathlib import Path
from typing  import Optional, Tuple, Set

from src.model.constants       import SOLARSYSTEMCONSTANTS
from src.model.frame_converter import FrameConverter


class GravityFieldCoefficients:
  """
  Container for spherical harmonic gravity field coefficients.
  """
  
  def __init__(
    self,
    max_degree : int,
    max_order  : int,
  ):
    """
    Initialize coefficient arrays.
    
    Input:
    ------
      max_degree : int
        Maximum degree of expansion.
      max_order : int
        Maximum order of expansion.
    """
    self.max_degree = max_degree
    self.max_order  = min(max_order, max_degree)
    
    # Allocate coefficient arrays (n x m)
    #   - C[n,m] and S[n,m] where n=degree, m=order
    self.C = np.zeros((max_degree + 1, max_order + 1))
    self.S = np.zeros((max_degree + 1, max_order + 1))
    
    # C(0,0) = 1 for normalized coefficients (represents point mass)
    self.C[0, 0] = 1.0
  
  def set_coefficient(
    self,
    degree : int,
    order  : int,
    Cnm    : float,
    Snm    : float,
  ) -> None:
    """
    Set a single coefficient pair.
    
    Input:
    ------
      degree : int
        Degree (n)
      order : int
        Order (m)
      Cnm : float
        C coefficient
      Snm : float
        S coefficient
    """
    if degree <= self.max_degree and order <= self.max_order:
      self.C[degree, order] = Cnm
      self.S[degree, order] = Snm


def load_icgem_file(
  filepath : Path,
  degree   : int,
  order    : int,
) -> tuple:
  """
  Load gravity field coefficients from ICGEM format file.
  
  Output:
  -------
    coeffs : GravityFieldCoefficients
      Loaded coefficients.
    gp : float
      Gravitational parameter [m³/s²].
    radius : float
      Reference radius [m].
  """
  # Default values (will be overwritten from file header)
  gp     = SOLARSYSTEMCONSTANTS.EARTH.GP
  radius = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
  
  # Open file
  with open(filepath, 'r') as f:
    # Read Header
    for line in f:
      line = line.strip()
      if not line:
        continue
        
      if line.startswith('earth_gravity_constant'):
        parts   = line.split()
        val_str = parts[1].replace('D', 'E').replace('d', 'e')
        gp      = float(val_str)
      elif line.startswith('radius'):
        parts   = line.split()
        val_str = parts[1].replace('D', 'E').replace('d', 'e')
        radius  = float(val_str)
      elif line.startswith('end_of_head'):
        break
    
    # Initialize coefficients container
    coeffs = GravityFieldCoefficients(degree, order)
    
    # Read Coefficients (continue reading from current file position)
    for line in f:
      line = line.strip()
      
      if not line or line.startswith('#'):
        continue
      
      parts = line.split()
      if len(parts) < 4:
        continue
      
      if parts[0] in ['gfc', 'gcf']:
        try:
          n_degree = int(parts[1]) # n
          m_order  = int(parts[2]) # m
          
          # Handle 'D' or 'E' exponents (e.g. 1.0D-06)
          cnm_str = parts[3].replace('D', 'E').replace('d', 'e')
          Cnm     = float(cnm_str)
          if len(parts) > 4:
            snm_str = parts[4].replace('D', 'E').replace('d', 'e')
            Snm     = float(snm_str)
          else:
            Snm = 0.0
          
          if n_degree <= degree and m_order <= order:
            coeffs.set_coefficient(n_degree, m_order, Cnm, Snm)
        except (ValueError, IndexError):
          continue
  
  return coeffs, gp, radius


def _parse_coefficient_name(name: str) -> Optional[tuple]:
  """
  Parse a coefficient name like 'J2', 'C22', 'S22' into (degree, order, type).
  
  Only supports single-digit degree (2-9) and order (0-9).
  
  Input:
  ------
    name : str
      Coefficient name (e.g., 'J2', 'J3', 'C21', 'S22', 'C33').
      
  Output:
  -------
    result : tuple | None
      (degree, order, coeff_type) where coeff_type is 'J', 'C', or 'S'.
      Returns None if parsing fails.
  """
  # Normalize name: uppercase and strip whitespace
  name = name.upper().strip()
  
  # Parse based on prefix
  if name.startswith('J'):
    # Zonal harmonic: J2, J3, ..., J9
    # Zonal harmonics always have order = 0
    # Only support single-digit degree (J2-J9)
    nums = name[1:]
    if len(nums) != 1:
      return None
    try:
      degree = int(nums)
      order  = 0
      return (degree, order, 'J')
    except ValueError:
      return None
  elif name.startswith('C'):
    # Tesseral/sectorial C coefficient: C21, C22, C31, ..., C99
    # Only support 2-digit format (single-digit degree and order)
    nums = name[1:]
    if len(nums) != 2:
      return None
    try:
      degree = int(nums[0])
      order  = int(nums[1])
      return (degree, order, 'C')
    except ValueError:
      return None
  elif name.startswith('S'):
    # Tesseral/sectorial S coefficient: S21, S22, S31, ..., S99
    # Only support 2-digit format (single-digit degree and order)
    nums = name[1:]
    if len(nums) != 2:
      return None
    try:
      degree = int(nums[0])
      order  = int(nums[1])
      return (degree, order, 'S')
    except ValueError:
      return None
  
  return None


def _get_required_degrees_orders(coefficient_names: list) -> tuple:
  """
  Determine the maximum degree and order needed for the given coefficient list.
  
  Input:
  ------
    coefficient_names : list
      List of coefficient names (e.g., ['J2', 'J3', 'C22', 'S22']).
      
  Output:
  -------
    result : tuple
      (max_degree, max_order, parsed_coefficients)
      where parsed_coefficients is a list of (degree, order, type) tuples.
  """
  max_degree = 0
  max_order = 0
  parsed = []
  
  for name in coefficient_names:
    result = _parse_coefficient_name(name)
    if result is not None:
      degree, order, coeff_type = result
      max_degree = max(max_degree, degree)
      max_order = max(max_order, order)
      parsed.append(result)
  
  return max_degree, max_order, parsed


class SphericalHarmonicsGravity:
  """
  Compute gravitational acceleration using spherical harmonics expansion.
  
  Notation:
    n : degree (zonal index)
    m : order (sectorial index)
  
  Uses Pines' algorithm as described in Montenbruck & Gill for stable computation.
  Coefficients are assumed to be fully normalized.
  """
  
  def __init__(
    self,
    coefficients        : GravityFieldCoefficients,
    degree              : Optional[int] = None,
    order               : Optional[int] = None,
    radius              : Optional[float] = None,
    gp                  : Optional[float] = None,
    active_coefficients : Optional[Set[tuple]] = None,
  ):
    """
    Initialize spherical harmonics gravity model.
    
    Input:
    ------
      coefficients : GravityFieldCoefficients
        Container with C and S coefficient arrays.
      degree : int, optional
        Maximum degree to use (defaults to coefficients.max_degree).
      order : int, optional
        Maximum order to use (defaults to coefficients.max_order).
      radius : float, optional
        Reference radius [m].
      gp : float, optional
        Gravitational parameter [m³/s²].
      active_coefficients : Set[tuple], optional
        Set of (degree, order, type) tuples specifying which coefficients
        are active. If None, all coefficients up to degree/order are used.
        Type is 'J' for zonal, 'C' for cosine tesseral, 'S' for sine tesseral.
    """
    self.coefficients        = coefficients
    self.degree              = degree if degree is not None else coefficients.max_degree
    self.order               = order  if order  is not None else coefficients.max_order
    self.radius              = radius
    self.gp                  = gp
    self.active_coefficients = active_coefficients
    
    # If active_coefficients is specified, create a mask
    if active_coefficients is not None:
      self._create_coefficient_mask()
    else:
      self._C_mask = None
      self._S_mask = None
    
    # Precompute normalization factors
    self._precompute_normalization()
  
  def _create_coefficient_mask(self) -> None:
    """
    Create masks for C and S coefficients based on active_coefficients set.
    Only coefficients in the active set will be used in computation.
    """

    # Initialize masks assuming all False or not used
    self._C_mask = np.zeros((self.degree + 1, self.order + 1), dtype=bool)
    self._S_mask = np.zeros((self.degree + 1, self.order + 1), dtype=bool)
    
    # Set True for active coefficients
    for degree, order, coeff_type in self.active_coefficients:
      if degree <= self.degree and order <= self.order:
        if coeff_type == 'J':
          order = 0
          self._C_mask[degree, order] = True
          # self._S_mask[degree, order] = False  # S not used for zonal
        elif coeff_type == 'C':
          self._C_mask[degree, order] = True
        elif coeff_type == 'S':
          self._S_mask[degree, order] = True
  
  def _precompute_normalization(self) -> None:
    """
    Precompute normalization factors for Legendre recursion.
    These convert between normalized and unnormalized associated Legendre functions.
    """
    n_max = self.degree + 2  # Need extra for derivatives
    m_max = self.order  + 2
    
    # Anm factors for vertical recursion: P(n,m) from P(n-1,m) and P(n-2,m)
    self.anm = np.zeros((n_max, m_max))
    self.bnm = np.zeros((n_max, m_max))
    
    for n_degree in range(2, n_max):
      for m_order in range(min(n_degree, m_max)):
        self.anm[n_degree, m_order] = np.sqrt((4*n_degree*n_degree - 1) / (n_degree*n_degree - m_order*m_order))
        self.bnm[n_degree, m_order] = np.sqrt((2*n_degree + 1) * (n_degree - 1 - m_order) * (n_degree - 1 + m_order) / 
                                      ((2*n_degree - 3) * (n_degree*n_degree - m_order*m_order)))
    
    # Diagonal recursion factors
    self.dnm = np.zeros(n_max)
    for m_order in range(1, n_max):
      if m_order == 1:
        self.dnm[m_order] = np.sqrt(3.0)
      else:
        self.dnm[m_order] = np.sqrt((2.0 * m_order + 1.0) / (2.0 * m_order))
  
  def compute(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute gravitational acceleration from spherical harmonics.
    """
    # Transform to body-fixed frame
    try:
      rot_mat = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      rot_mat = np.eye(3)
    
    bf_pos = rot_mat @ j2000_pos_vec
    
    # Use Pines algorithm
    bf_acc = self._compute_acceleration_pines(bf_pos)
    
    # Transform back to J2000
    j2000_acc = rot_mat.T @ bf_acc
    
    return j2000_acc

  def _compute_acceleration_pines(
    self,
    pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute acceleration using Pines' non-singular algorithm.
    
    Based on Montenbruck & Gill, "Satellite Orbits", Section 3.2.5
    This avoids singularities at the poles.
    """
    x, y, z = pos_vec
    r = np.linalg.norm(pos_vec)
    
    # Normalized coordinates
    s = x / r  # cos(lon) * cos(lat)
    t = y / r  # sin(lon) * cos(lat)
    u = z / r  # sin(lat)
    
    Re = self.radius
    gp = self.gp
    
    n_max = self.degree
    m_max = self.order
    
    # Initialize Legendre functions (fully normalized)
    V = np.zeros((n_max + 3, m_max + 3))
    W = np.zeros((n_max + 3, m_max + 3))
    
    # Seed values
    rho = Re / r
    V[0, 0] = rho
    W[0, 0] = 0.0
    
    # First compute along m=0 (zonal)
    V[1, 0] = u * np.sqrt(3.0) * rho * V[0, 0]
    
    for n_degree in range(2, n_max + 2):
      V[n_degree, 0] = (self.anm[n_degree, 0] * u * V[n_degree-1, 0]) * rho - (self.bnm[n_degree, 0] * V[n_degree-2, 0]) * rho * rho
    
    # Now compute sectorial (m=n) and tesseral (m<n) 
    for m_order in range(1, m_max + 2):
      # Sectorial: n=m
      if m_order <= n_max + 1:
        V[m_order, m_order] = self.dnm[m_order] * (s * V[m_order-1, m_order-1] - t * W[m_order-1, m_order-1]) * rho
        W[m_order, m_order] = self.dnm[m_order] * (s * W[m_order-1, m_order-1] + t * V[m_order-1, m_order-1]) * rho
      
      # First tesseral: n=m+1
      if m_order + 1 <= n_max + 1:
        fac = np.sqrt(2*m_order + 3)
        V[m_order+1, m_order] = fac * u * V[m_order, m_order] * rho
        W[m_order+1, m_order] = fac * u * W[m_order, m_order] * rho
      
      # Remaining tesseral for this m
      for n_degree in range(m_order + 2, n_max + 2):
        V[n_degree, m_order] = (self.anm[n_degree, m_order] * u * V[n_degree-1, m_order]) * rho - (self.bnm[n_degree, m_order] * V[n_degree-2, m_order]) * rho * rho
        W[n_degree, m_order] = (self.anm[n_degree, m_order] * u * W[n_degree-1, m_order]) * rho - (self.bnm[n_degree, m_order] * W[n_degree-2, m_order]) * rho * rho
    
    # Compute acceleration partials
    ax = 0.0
    ay = 0.0
    az = 0.0
    
    # Sum contributions from degree 2 onwards
    for n_degree in range(2, n_max + 1):
      norm_fix = np.sqrt((2.0 * n_degree + 1.0) / (2.0 * n_degree + 3.0))

      for m_order in range(0, min(n_degree + 1, m_max + 1)):
        # Get coefficients, applying mask if set
        Cnm = self.coefficients.C[n_degree, m_order]
        Snm = self.coefficients.S[n_degree, m_order]
        
        # Apply coefficient mask if active_coefficients was specified
        if self._C_mask is not None:
          if not self._C_mask[n_degree, m_order]:
            Cnm = 0.0
        if self._S_mask is not None:
          if not self._S_mask[n_degree, m_order]:
            Snm = 0.0
        
        # Skip if both coefficients are zero
        if Cnm == 0.0 and Snm == 0.0:
          continue
        
        # Factors for derivative computation
        if m_order == 0:
          # m = 0 case (zonal terms)
          # Following Montenbruck & Gill eq. 3.33
          fac1 = np.sqrt((n_degree + 1) * (n_degree + 2) / 2.0)
          fac2 = (n_degree + 1)
          
          ax -= Cnm * fac1 * V[n_degree+1, 1] * norm_fix
          ay -= Cnm * fac1 * W[n_degree+1, 1] * norm_fix
          az -= Cnm * fac2 * V[n_degree+1, 0] * norm_fix
        else:
          # m > 0 case (tesseral and sectorial terms)
          # Factor for m-1 term
          fac1 = np.sqrt((n_degree - m_order + 1) * (n_degree - m_order + 2))
          # Factor for m+1 term  
          fac2 = np.sqrt((n_degree + m_order + 1) * (n_degree + m_order + 2))
          # Factor for z derivative
          fac3 = np.sqrt((n_degree - m_order + 1) * (n_degree + m_order + 1))
          
          if m_order == 1:
            fac1 = np.sqrt((n_degree) * (n_degree + 1))
          
          ax += 0.5 * (
            -fac2 * (Cnm * V[n_degree+1, m_order+1] + Snm * W[n_degree+1, m_order+1])
            + fac1 * (Cnm * V[n_degree+1, m_order-1] + Snm * W[n_degree+1, m_order-1])
          ) * norm_fix
          ay += 0.5 * (
            -fac2 * (Cnm * W[n_degree+1, m_order+1] - Snm * V[n_degree+1, m_order+1])
            - fac1 * (Cnm * W[n_degree+1, m_order-1] - Snm * V[n_degree+1, m_order-1])
          ) * norm_fix
          az -= fac3 * (Cnm * V[n_degree+1, m_order] + Snm * W[n_degree+1, m_order]) * norm_fix
    
    # Scale by GM/Re^2 (Pines formulation scaling)
    scale = gp / (Re * Re)
    acc_vec = scale * np.array([ax, ay, az])
    
    # Add Two-Body Point Mass term explicitly (n=0)
    acc_vec += -gp * pos_vec / r**3
    
    return acc_vec


def load_gravity_field(
  filepath : Path,
  degree   : int,
  order    : int,
) -> SphericalHarmonicsGravity:
  """
  Convenience function to load gravity field and create model.
  
  Input:
  ------
    filepath : Path
      Path to coefficient file.
    degree : int
      Maximum degree.
    order : int
      Maximum order.
  
  Output:
  -------
    model : SphericalHarmonicsGravity
      Ready-to-use gravity model.
  """
  coefficients, gp, radius = load_icgem_file(filepath, degree, order)
  return SphericalHarmonicsGravity(
    coefficients = coefficients,
    degree       = degree,
    order        = order,
    radius       = radius,
    gp           = gp,
  )


def create_gravity_model_from_coefficients(
  coefficient_names : list,
  gp                : Optional[float] = None,
  radius            : Optional[float] = None,
  gravity_file_path : Optional[Path]  = None,
) -> SphericalHarmonicsGravity:
  """
  Create a gravity model using only specific named coefficients (e.g., 'J2', 'C22', 'S22').
  
  If a gravity file path is provided, coefficient values are read from that file.
  Otherwise, falls back to hardcoded Earth constants from SOLARSYSTEMCONSTANTS.
  
  Input:
  ------
    coefficient_names : list
      List of coefficient names (e.g., ['J2', 'J3', 'C22', 'S22']).
    gp : float, optional
      Gravitational parameter [m³/s²]. Defaults to value from file or Earth's GP.
    radius : float, optional
      Reference radius [m]. Defaults to value from file or Earth's equatorial radius.
    gravity_file_path : Path, optional
      Path to gravity coefficient file (e.g., EGM2008.gfc). If provided, 
      coefficients are read from this file.
  
  Output:
  -------
    model : SphericalHarmonicsGravity
      Gravity model with only the specified coefficients active.
  """
  # Parse coefficient names and determine required degree/order
  max_degree, max_order, parsed_coefficients = _get_required_degrees_orders(coefficient_names)
  
  if max_degree < 2:
    max_degree = 2  # Minimum for harmonics
  
  # Load from file if provided
  if gravity_file_path is not None and gravity_file_path.exists():
    # Load coefficients from file
    file_coeffs, file_gp, file_radius = load_icgem_file(gravity_file_path, max_degree, max_order)
    
    # Use file values if not explicitly provided
    if gp is None:
      gp = file_gp
    if radius is None:
      radius = file_radius
    
    # Create coefficient container with values from file
    coeffs = file_coeffs
    
  else:
    # Fall back to hardcoded constants
    if gp is None:
      gp = SOLARSYSTEMCONSTANTS.EARTH.GP
    if radius is None:
      radius = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR
    
    # Create coefficient container
    coeffs = GravityFieldCoefficients(max_degree, max_order)
    
    # Map coefficient names to values from constants (fallback)
    coefficient_values = {
      ('J', 2): SOLARSYSTEMCONSTANTS.EARTH.J2,
      ('J', 3): SOLARSYSTEMCONSTANTS.EARTH.J3,
      ('J', 4): SOLARSYSTEMCONSTANTS.EARTH.J4,
      ('J', 5): SOLARSYSTEMCONSTANTS.EARTH.J5,
      ('J', 6): SOLARSYSTEMCONSTANTS.EARTH.J6,
      ('C', 2, 1): SOLARSYSTEMCONSTANTS.EARTH.C21,
      ('S', 2, 1): SOLARSYSTEMCONSTANTS.EARTH.S21,
      ('C', 2, 2): SOLARSYSTEMCONSTANTS.EARTH.C22,
      ('S', 2, 2): SOLARSYSTEMCONSTANTS.EARTH.S22,
      ('C', 3, 1): SOLARSYSTEMCONSTANTS.EARTH.C31,
      ('S', 3, 1): SOLARSYSTEMCONSTANTS.EARTH.S31,
      ('C', 3, 2): SOLARSYSTEMCONSTANTS.EARTH.C32,
      ('S', 3, 2): SOLARSYSTEMCONSTANTS.EARTH.S32,
      ('C', 3, 3): SOLARSYSTEMCONSTANTS.EARTH.C33,
      ('S', 3, 3): SOLARSYSTEMCONSTANTS.EARTH.S33,
    }
    
    # Set coefficients from hardcoded values
    for degree, order, coeff_type in parsed_coefficients:
      if coeff_type == 'J':
        key = ('J', degree)
        if key in coefficient_values:
          Cn0 = -coefficient_values[key]
          coeffs.set_coefficient(degree, 0, Cn0, 0.0)
      elif coeff_type == 'C':
        key = ('C', degree, order)
        if key in coefficient_values:
          current_S = coeffs.S[degree, order] if degree <= max_degree and order <= max_order else 0.0
          coeffs.set_coefficient(degree, order, coefficient_values[key], current_S)
      elif coeff_type == 'S':
        key = ('S', degree, order)
        if key in coefficient_values:
          current_C = coeffs.C[degree, order] if degree <= max_degree and order <= max_order else 0.0
          coeffs.set_coefficient(degree, order, current_C, coefficient_values[key])
  
  # Build active set (same logic for both file and fallback)
  active_coefficients = set()
  for degree, order, coeff_type in parsed_coefficients:
    if coeff_type == 'J':
      active_coefficients.add((degree, 0, 'J'))
    elif coeff_type == 'C':
      active_coefficients.add((degree, order, 'C'))
    elif coeff_type == 'S':
      active_coefficients.add((degree, order, 'S'))
  
  return SphericalHarmonicsGravity(
    coefficients        = coeffs,
    degree              = max_degree,
    order               = max_order,
    radius              = radius,
    gp                  = gp,
    active_coefficients = active_coefficients,
  )