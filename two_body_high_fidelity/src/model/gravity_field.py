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
from typing  import Optional, Set

from src.model.constants       import SOLARSYSTEMCONSTANTS
from src.model.frame_and_vector_converter import FrameConverter


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
    n_max = self.degree + 4  # Increased for curvature (Jacobian) computation
    m_max = self.order  + 4
    
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

  def jacobian_approx(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
    eps           : float = 1.0e-6,
  ) -> np.ndarray:
    """
    Numerical Jacobian of spherical harmonics acceleration.

    Computes ∂a/∂r using central differences.

    Input:
    ------
      time_et : float
        Current Ephemeris Time (ET) [s]
      j2000_pos_vec : np.ndarray
        Position vector [m] in inertial frame (J2000)
      eps : float
        Relative step size for finite differences (default 1e-6).

    Output:
    -------
      daccvec__dposvec : np.ndarray (3x3)
        Jacobian matrix ∂a/∂r in J2000 frame
    """
    pos_vec = np.array(j2000_pos_vec, dtype=float)
    daccvec__dposvec = np.zeros((3, 3))

    # Compute rotation once to avoid repeated SPICE calls
    try:
      rot_mat = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      rot_mat = np.eye(3)
    rot_T = rot_mat.T

    for i in range(3):
      step = eps * max(1.0, abs(pos_vec[i]))
      if step == 0.0:
        step = eps

      pos_plus = pos_vec.copy()
      pos_minus = pos_vec.copy()
      pos_plus[i] += step
      pos_minus[i] -= step

      bf_plus = rot_mat @ pos_plus
      bf_minus = rot_mat @ pos_minus

      acc_plus = rot_T @ self._compute_acceleration_pines(bf_plus)
      acc_minus = rot_T @ self._compute_acceleration_pines(bf_minus)

      daccvec__dposvec[:, i] = (acc_plus - acc_minus) / (2.0 * step)

    return daccvec__dposvec

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

  def jacobian(
    self,
    time_et       : float,
    j2000_pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute gravitational Jacobian (Gravity Gradient Matrix) using spherical harmonics.
    
    Input:
    ------
      time_et : float
        Ephemeris time [s]
      j2000_pos_vec : np.ndarray
        Position vector in J2000 frame [m]
        
    Output:
    -------
      j2000_jac : np.ndarray (3, 3)
        3x3 Jacobian matrix (da/dr) in J2000 frame [1/s^2]
    """
    # Transform to body-fixed frame
    try:
      rot_mat = FrameConverter.j2000_to_iau_earth(time_et)
    except Exception:
      rot_mat = np.eye(3)
    
    bf_pos = rot_mat @ j2000_pos_vec
    
    # Use Pines algorithm for Jacobian (Vines)
    bf_jac = self._compute_jacobian_pines(bf_pos)
    
    # Transform back to J2000: J_inertial = R^T * J_body * R
    j2000_jac = rot_mat.T @ bf_jac @ rot_mat
    
    return j2000_jac

  def _compute_jacobian_pines(
    self,
    pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute Gravity Gradient Matrix (Jacobian) using Pines' algorithm.
    The partial derivatives of the acceleration vector.
    """
    x, y, z = pos_vec
    r = np.linalg.norm(pos_vec)
    r_sq = r * r
    
    # Normalized coordinates
    s = x / r
    t = y / r
    u = z / r
    
    Re = self.radius
    gp = self.gp
    
    n_max = self.degree
    m_max = self.order
    
    # Needs recursion up to n_max + 2 for second derivatives
    V = np.zeros((n_max + 4, m_max + 4))
    W = np.zeros((n_max + 4, m_max + 4))
    
    # Seed values
    rho = Re / r
    V[0, 0] = rho
    W[0, 0] = 0.0
    
    # -------------------------------------------------------------------------
    # Recursion (same as acceleration, but extended depth)
    # -------------------------------------------------------------------------
    
    # Zonal (m=0)
    V[1, 0] = u * np.sqrt(3.0) * rho * V[0, 0]
    for n in range(2, n_max + 3):
      V[n, 0] = (self.anm[n, 0] * u * V[n-1, 0]) * rho - (self.bnm[n, 0] * V[n-2, 0]) * rho * rho

    # Sectorial/Tesseral
    for m in range(1, m_max + 3):
      # Sectorial n=m
      if m <= n_max + 2:
        V[m, m] = self.dnm[m] * (s * V[m-1, m-1] - t * W[m-1, m-1]) * rho
        W[m, m] = self.dnm[m] * (s * W[m-1, m-1] + t * V[m-1, m-1]) * rho
      
      # First tesseral n=m+1
      if m + 1 <= n_max + 2:
        fac = np.sqrt(2*m + 3)
        V[m+1, m] = fac * u * V[m, m] * rho
        W[m+1, m] = fac * u * W[m, m] * rho
      
      # Remaining tesseral
      for n in range(m + 2, n_max + 3):
        V[n, m] = (self.anm[n, m] * u * V[n-1, m]) * rho - (self.bnm[n, m] * V[n-2, m]) * rho * rho

    # -------------------------------------------------------------------------
    # Gradient Computation
    # -------------------------------------------------------------------------
    # Initialize partials
    j_xx = j_yy = j_zz = j_xy = j_xz = j_yz = 0.0
    
    for n in range(2, n_max + 1):
      # Normalization correction for first derivative (acceleration)
      nf1 = np.sqrt((2.0 * n + 1.0) / (2.0 * n + 3.0))
      # Normalization correction for second derivative (gradient)
      nf2 = np.sqrt((2.0 * n + 3.0) / (2.0 * n + 5.0))
      
      # Combined normalization
      norm_fix = nf1 * nf2

      for m in range(0, min(n + 1, m_max + 1)):
        Cnm = self.coefficients.C[n, m]
        Snm = self.coefficients.S[n, m]
        
        # Apply masks
        if self._C_mask is not None and not self._C_mask[n, m]: Cnm = 0.0
        if self._S_mask is not None and not self._S_mask[n, m]: Snm = 0.0
        if Cnm == 0.0 and Snm == 0.0: continue

        # Helper terms for m-2, m, m+2
        # C_alpha: V/W terms for m-2
        # C_beta:  V/W terms for m
        # C_gamma: V/W terms for m+2

        # Factors for recursions n->n+1->n+2
        # We process based on the 9 components of Hessian, exploiting symmetry
        # da_x / dx
        #   involves m-2, m, m+2
        # da_z / dz
        #   involves m, m
        
        # Recalculate factors for n, m
        # fac1_1: sqrt((n-m+1)(n-m+2)) corresponding to m-1 step
        # fac2_1: sqrt((n+m+1)(n+m+2)) corresponding to m+1 step
        # fac3_1: sqrt((n-m+1)(n+m+1)) corresponding to z step

        # --- Base Factors (Level 1: n -> n+1) ---
        f_m_minus_1 = 0.0
        if m > 0:
           # valid for m-1
           f_m_minus_1 = np.sqrt((n - m + 1) * (n - m + 2))
           if m == 1: f_m_minus_1 = np.sqrt(n * (n + 1)) # special m=1 case
        
        f_m_plus_1  = np.sqrt((n + m + 1) * (n + m + 2))
        f_z         = np.sqrt((n - m + 1) * (n + m + 1))
        
        # --- Level 2 Factors (Level 2: n+1 -> n+2) ---
        # We need these evaluated at the new orders (m-1, m, m+1)
        
        # For term m-2 (path: m -> m-1 -> m-2)
        f_m_minus_2 = 0.0 # from m-1 down to m-2
        if m > 1:
           # parent is m-1, child is m-2
           # factor formula using n_new = n+1, m_new = m-1
           # sqrt((n' - m' + 1)(n' - m' + 2)) -> sqrt((n+1 - (m-1) + 1)(...))
           # sqrt((n - m + 3)(n - m + 4))
           if m == 2:
             f_m_minus_2 = np.sqrt((n + 1) * (n + 2))
           else:
             f_m_minus_2 = np.sqrt((n - m + 3) * (n - m + 4))

        # For term m (path: m -> m-1 -> m)
        # parent m-1, child m
        # factor sqrt((n' + m' + 1)(n' + m' + 2)) where n'=n+1, m'=m-1
        # sqrt((n+1 + m-1 + 1)(...)) -> sqrt((n+m+1)(n+m+2))
        # Note: this equals f_m_plus_1!
        
        # For term m (path: m -> m+1 -> m)
        # parent m+1, child m
        # factor sqrt((n' - m' + 1)(n' - m' + 2)) where n'=n+1, m'=m+1
        # sqrt((n+1 - (m+1) + 1)(...)) -> sqrt((n-m+1)(n-m+2))
        # Note: this equals f_m_minus_1!

        # For term m+2 (path: m -> m+1 -> m+2)
        # parent m+1, child m+2
        f_m_plus_2  = np.sqrt((n + m + 3) * (n + m + 4))

        # --- Term Combinations ---
        # 1. Term (m-2)
        # Coeff: 0.25 * f_m_minus_1 * f_m_minus_2
        val_m_minus_2_C = 0.0
        val_m_minus_2_S = 0.0
        if m >= 2:
            val_m_minus_2_C = V[n+2, m-2]
            val_m_minus_2_S = W[n+2, m-2]
        term_m_minus_2_C = (Cnm * val_m_minus_2_C + Snm * val_m_minus_2_S)
        term_m_minus_2_S = (Cnm * val_m_minus_2_S - Snm * val_m_minus_2_C)
        
        c_m_minus_2 = 0.25 * f_m_minus_1 * f_m_minus_2

        # 2. Term (m+2)
        # Coeff: 0.25 * f_m_plus_1 * f_m_plus_2
        val_m_plus_2_C = V[n+2, m+2]
        val_m_plus_2_S = W[n+2, m+2]
        term_m_plus_2_C = (Cnm * val_m_plus_2_C + Snm * val_m_plus_2_S)
        term_m_plus_2_S = (Cnm * val_m_plus_2_S - Snm * val_m_plus_2_C)
        
        c_m_plus_2 = 0.25 * f_m_plus_1 * f_m_plus_2
        
        # 3. Term (m) - from x/y mixed
        # Coeff: 0.25 * (f_m_minus_1 * f_m_plus_1 + f_m_plus_1 * f_m_minus_1) ... wait check structure
        # Actually for partial_xx: 0.5 * d/dx ( +fac1 * V... - fac2 * V... )
        # d/dx V(n,m) ~ +V(n,m-1) - V(n,m+1)
        # So we get cross terms.
        
        val_m_C = V[n+2, m]
        val_m_S = W[n+2, m]
        term_m_C = (Cnm * val_m_C + Snm * val_m_S)
        term_m_S = (Cnm * val_m_S - Snm * val_m_C)

        # Coefficient for central term (m) in xx/yy
        # derived from -fac2 * ( +fac1 * V(m) ) + fac1 * ( -fac2 * V(m) )
        # = -2 * fac1 * fac2 * V(m) * 0.25 = -0.5 * fac1 * fac2
        c_m_center = -0.5 * f_m_minus_1 * f_m_plus_1
        
        # --- Hessian Components ---

        # d(ax)/dx
        # = Term(m-2) + Term(m+2) + Term(m center)
        # (+ c_m_minus_2 * V_m-2  + c_m_plus_2 * V_m+2  + c_m_center * V_m)
        j_xx += (c_m_minus_2 * term_m_minus_2_C + c_m_plus_2 * term_m_plus_2_C + c_m_center * term_m_C) * norm_fix

        # d(ay)/dy
        # = - Term(m-2) - Term(m+2) + Term(m center) 
        # (signs flip for m-2 and m+2 due to W/V swapping in d/dy)
        # d/dy V(m) ~ W(m-1) + W(m+1)
        # d/dy W(m) ~ -V(m-1) - V(m+1)
        # Combining leads to negations for the m+/-2 terms relative to xx
        j_yy += (-c_m_minus_2 * term_m_minus_2_C - c_m_plus_2 * term_m_plus_2_C + c_m_center * term_m_C) * norm_fix

        # d(ax)/dy = d(ay)/dx (xy)
        # Involves S terms for m +/- 2
        # c_m_minus_2 * TermS(m-2) - c_m_plus_2 * TermS(m+2)
        # The center term cancels out for xy
        j_xy += (c_m_minus_2 * term_m_minus_2_S - c_m_plus_2 * term_m_plus_2_S) * norm_fix

        # d(az)/dz
        # d/dz V(n,m) = -fac3 * V(n+1,m)
        # Second deriv: (-fac3_1) * (-fac3_2) * V(n+2,m)
        # fac3_2 for n+1->n+2, order m->m
        # sqrt((n+1-m+1)(n+1+m+1)) = sqrt((n-m+2)(n+m+2))
        f_z_2 = np.sqrt((n - m + 2) * (n + m + 2))
        # d(az)/dz
        # d/dz V(n,m) = -fac3 * V(n+1,m)
        # Second deriv: (-fac3_1) * (-fac3_2) * V(n+2,m)
        # fac3_2 for n+1->n+2, order m->m
        # sqrt((n+1-m+1)(n+1+m+1)) = sqrt((n-m+2)(n+m+2))
        f_z_2 = np.sqrt((n - m + 2) * (n + m + 2))
        c_zz = f_z * f_z_2
        
        term_zz = (c_zz * term_m_C) * norm_fix
        j_zz += term_zz

        # Special handling for Zonal Harmonics (m=0)
        # For m=0, V depends only on z and r, so V is cylindrically symmetric about z.
        # This implies d2V/dx2 = d2V/dy2.
        # From Laplace: d2V/dx2 + d2V/dy2 + d2V/dz2 = 0
        # 2 * d2V/dx2 = -d2V/dz2  =>  d2V/dx2 = -0.5 * d2V/dz2
        # This is more robust than the general recursion which fails for m=0 due to c_center=0.
        if m == 0:
            j_xx += -0.5 * term_zz
            j_yy += -0.5 * term_zz
            # j_xy contribution is zero for zonals
        else:
            # Standard Pines recursion for m > 0

            # d(ax)/dx
            # = Term(m-2) + Term(m+2) + Term(m center)
            j_xx += (c_m_minus_2 * term_m_minus_2_C + c_m_plus_2 * term_m_plus_2_C + c_m_center * term_m_C) * norm_fix

            # d(ay)/dy
            # = - Term(m-2) - Term(m+2) + Term(m center) 
            # (signs flip for m-2 and m+2 due to W/V swapping in d/dy)
            j_yy += (-c_m_minus_2 * term_m_minus_2_C - c_m_plus_2 * term_m_plus_2_C + c_m_center * term_m_C) * norm_fix

            # d(ax)/dy = d(ay)/dx (xy)
            # Involves S terms for m +/- 2
            # c_m_minus_2 * TermS(m-2) - c_m_plus_2 * TermS(m+2)
            # The center term cancels out for xy
            j_xy += (c_m_minus_2 * term_m_minus_2_S - c_m_plus_2 * term_m_plus_2_S) * norm_fix

        if m > 0:
            # Standard recursion for xz, yz also works for m=0 if bounds are careful
            # But let's reuse logic below
            pass

        # d(ax)/dz (xz)
        # d/dz ( dx )
        # dx terms: +0.5 * f1 * V(m-1) - 0.5 * f2 * V(m+1)
        # apply d/dz: * -f_z_next
        # ... derived coeff c_xz_1, c_xz_2
        
        c_xz_1 = 0.5 * f_z * np.sqrt((n - m + 2) * (n - m + 3))
        c_xz_2 = -0.5 * f_z * np.sqrt((n + m + 2) * (n + m + 3))
        
        val_m_minus_1_C = 0.0
        val_m_minus_1_S = 0.0
        if m >= 1:
           val_m_minus_1_C = V[n+2, m-1]
           val_m_minus_1_S = W[n+2, m-1]
        
        val_m_plus_1_C = V[n+2, m+1]
        val_m_plus_1_S = W[n+2, m+1]

        t_m_minus_1_C = (Cnm * val_m_minus_1_C + Snm * val_m_minus_1_S)
        t_m_minus_1_S = (Cnm * val_m_minus_1_S - Snm * val_m_minus_1_C)
        t_m_plus_1_C  = (Cnm * val_m_plus_1_C + Snm * val_m_plus_1_S)
        t_m_plus_1_S  = (Cnm * val_m_plus_1_S - Snm * val_m_plus_1_C)

        j_xz += (c_xz_1 * t_m_minus_1_C + c_xz_2 * t_m_plus_1_C) * norm_fix
        j_yz += (c_xz_1 * t_m_minus_1_S + c_xz_2 * t_m_plus_1_S) * norm_fix # d(ay)/dz like d(ax)/dz with W/V swap (S term)

    # Scale
    scale = gp / (Re**3)
    
    # Point mass contribution (n=0)
    # G_pm = -mu/r^3 * I + 3*mu/r^5 * r*r^T
    # The loop handled n >= 2.
    # We must add point mass G explicitly.
    
    J = np.zeros((3, 3))
    J[0, 0] = j_xx * scale
    J[0, 1] = j_xy * scale
    J[0, 2] = j_xz * scale
    J[1, 1] = j_yy * scale
    J[1, 2] = j_yz * scale
    J[2, 2] = j_zz * scale
    
    # Symmetrize
    J[1, 0] = J[0, 1]
    J[2, 0] = J[0, 2]
    J[2, 1] = J[1, 2]
    
    # Point mass terms
    pm_term = -gp / r_sq / r
    pm_term3 = 3.0 * gp / (r_sq * r_sq * r)
    
    I_mat = np.eye(3)
    rr_T = np.outer(pos_vec, pos_vec)
    
    J_pm = pm_term * I_mat + pm_term3 * rr_T
    
    return J + J_pm

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