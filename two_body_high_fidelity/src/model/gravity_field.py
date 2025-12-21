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
from typing  import Optional, Tuple

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
    gp         : float,
    radius     : float,
  ):
    """
    Initialize coefficient arrays.
    
    Input:
    ------
      max_degree : int
        Maximum degree of expansion.
      max_order : int
        Maximum order of expansion.
      gp : float
        Gravitational parameter [m³/s²].
      radius : float
        Reference radius [m].
    """
    self.max_degree = max_degree
    self.max_order  = min(max_order, max_degree)
    self.gp         = gp
    self.radius     = radius
    
    # Allocate coefficient arrays (n x m)
    # C[n,m] and S[n,m] where n=degree, m=order
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
  filepath   : Path,
  max_degree : int,
  max_order  : int,
) -> GravityFieldCoefficients:
  """
  Load gravity field coefficients from ICGEM format file.
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
        parts = line.split()
        # Handle 'D' exponents in header values if present
        val_str = parts[1].replace('D', 'E').replace('d', 'e')
        gp = float(val_str)
      elif line.startswith('radius'):
        parts   = line.split()
        val_str = parts[1].replace('D', 'E').replace('d', 'e')
        radius  = float(val_str)
      elif line.startswith('end_of_head'):
        break
    
    # Initialize coefficients container
    coeffs = GravityFieldCoefficients(max_degree, max_order, gp, radius)
    
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
          degree = int(parts[1]) # n
          order  = int(parts[2]) # m
          
          # Handle 'D' or 'E' exponents (e.g. 1.0D-06)
          cnm_str = parts[3].replace('D', 'E').replace('d', 'e')
          Cnm     = float(cnm_str)
          if len(parts) > 4:
            snm_str = parts[4].replace('D', 'E').replace('d', 'e')
            Snm     = float(snm_str)
          else:
            Snm = 0.0
          
          if degree <= max_degree and order <= max_order:
            coeffs.set_coefficient(degree, order, Cnm, Snm)
        except (ValueError, IndexError):
          continue
  
  return coeffs


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
    coefficients : GravityFieldCoefficients,
    use_degree   : Optional[int] = None,
    use_order    : Optional[int] = None,
  ):
    """
    Initialize spherical harmonics gravity model.
    """
    self.coeffs     = coefficients
    self.max_degree = use_degree if use_degree else coefficients.max_degree
    self.max_order  = use_order if use_order else coefficients.max_order
    self.max_order  = min(self.max_order, self.max_degree)
    
    # Precompute normalization factors
    self._precompute_normalization()
  
  def _precompute_normalization(self) -> None:
    """
    Precompute normalization factors for Legendre recursion.
    These convert between normalized and unnormalized associated Legendre functions.
    """
    n_max = self.max_degree + 2  # Need extra for derivatives
    m_max = self.max_order  + 2
    
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
        # Special case for m=1 due to Kronecker delta in normalization definition.
        # The normalization factor has k=1 for m=0 and k=2 for m>0.
        # The transition from m=0 to m=1 introduces an extra sqrt(2) factor.
        # General formula gives sqrt(1.5); correct value is sqrt(1.5 * 2) = sqrt(3).
        self.dnm[m_order] = np.sqrt(3.0)
      else:
        # General case for m >= 2
        # Derived from fully normalized recursion: P_mm = sqrt((2m+1)/2m) * sin(theta) * P_{m-1,m-1}
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
  
  def _compute_acceleration_direct(
    self,
    pos_vec : np.ndarray,
  ) -> np.ndarray:
    """
    Compute acceleration using direct analytical formulas.
    
    More accurate for low-degree terms (J2, J3, etc.)
    """
    x, y, z = pos_vec
    r = np.linalg.norm(pos_vec)
    
    gp = self.coeffs.gp
    Re = self.coeffs.radius
    
    # Start with point mass
    acc = -gp * pos_vec / r**3
    
    # J2 term (degree 2, order 0)
    if self.max_degree >= 2:
      # J2 = -sqrt(5) * C20 for fully normalized coefficients
      J2 = -np.sqrt(5.0) * self.coeffs.C[2, 0]
      
      r2 = r * r
      z2 = z * z
      factor = 1.5 * J2 * gp * Re**2 / (r2 * r2 * r)
      
      acc[0] += factor * x * (5.0 * z2 / r2 - 1.0)
      acc[1] += factor * y * (5.0 * z2 / r2 - 1.0)
      acc[2] += factor * z * (5.0 * z2 / r2 - 3.0)
    
    return acc

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
    
    Re = self.coeffs.radius
    gp = self.coeffs.gp
    
    n_max = self.max_degree
    m_max = self.max_order
    
    # Initialize Legendre functions (fully normalized)
    # V(n,m) and W(n,m) are related to normalized ALFs
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
    # Note: n=0 is point mass (handled separately), n=1 usually zero for Earth-centered
    for n_degree in range(2, n_max + 1):
      # Normalization correction factor for derivative terms
      # This factor sqrt((2n+1)/(2n+3)) is needed because the derivative of a normalized
      # Legendre function of degree n involves normalized functions of degree n+1.
      norm_fix = np.sqrt((2.0 * n_degree + 1.0) / (2.0 * n_degree + 3.0))

      for m_order in range(0, min(n_degree + 1, m_max + 1)):
        Cnm = self.coeffs.C[n_degree, m_order]
        Snm = self.coeffs.S[n_degree, m_order]
        
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
          
          # Special case for m=1: different kronecker delta
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
    # This avoids issues with normalization factors for the monopole term
    acc_vec += -gp * pos_vec / r**3
    
    return acc_vec


def load_gravity_field(
  filepath           : Path,
  max_gravity_degree : int,
  max_gravity_order  : int,
) -> SphericalHarmonicsGravity:
  """
  Convenience function to load gravity field and create model.
  
  Input:
  ------
    filepath : Path
      Path to coefficient file.
    max_gravity_degree : int
      Maximum degree.
    max_gravity_order : int
      Maximum order.
  
  Output:
  -------
    model : SphericalHarmonicsGravity
      Ready-to-use gravity model.
  """
  coeffs = load_icgem_file(filepath, max_gravity_degree, max_gravity_order)
  return SphericalHarmonicsGravity(coeffs, max_gravity_degree, max_gravity_order)