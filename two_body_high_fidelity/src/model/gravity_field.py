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
    n   : int,
    m   : int,
    Cnm : float,
    Snm : float,
  ) -> None:
    """
    Set a single coefficient pair.
    """
    if n <= self.max_degree and m <= self.max_order:
      self.C[n, m] = Cnm
      self.S[n, m] = Snm


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
        parts = line.split()
        val_str = parts[1].replace('D', 'E').replace('d', 'e')
        radius = float(val_str)
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
          n   = int(parts[1])
          m   = int(parts[2])
          # Handle 'D' exponents in coefficients (e.g. 1.0D-06)
          cnm_str = parts[3].replace('D', 'E').replace('d', 'e')
          Cnm = float(cnm_str)
          
          if len(parts) > 4:
            snm_str = parts[4].replace('D', 'E').replace('d', 'e')
            Snm = float(snm_str)
          else:
            Snm = 0.0
          
          if n <= max_degree and m <= max_order:
            coeffs.set_coefficient(n, m, Cnm, Snm)
        except (ValueError, IndexError):
          continue
  
  return coeffs


class SphericalHarmonicsGravity:
  """
  Compute gravitational acceleration using spherical harmonics expansion.
  
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
    m_max = self.max_order + 2
    
    # Anm factors for vertical recursion: P(n,m) from P(n-1,m) and P(n-2,m)
    self.anm = np.zeros((n_max, m_max))
    self.bnm = np.zeros((n_max, m_max))
    
    for n in range(2, n_max):
      for m in range(min(n, m_max)):
        self.anm[n, m] = np.sqrt((4*n*n - 1) / (n*n - m*m))
        self.bnm[n, m] = np.sqrt((2*n + 1) * (n - 1 - m) * (n - 1 + m) / 
                                  ((2*n - 3) * (n*n - m*m)))
    
    # Diagonal recursion factors
    self.dnm = np.zeros(n_max)
    for m in range(1, n_max):
      if m == 1:
        self.dnm[m] = np.sqrt(3.0)
      else:
        self.dnm[m] = np.sqrt((2.0 * m + 1.0) / (2.0 * m))
  
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
    
    for n in range(2, n_max + 2):
      V[n, 0] = (self.anm[n, 0] * u * V[n-1, 0]) * rho - (self.bnm[n, 0] * V[n-2, 0]) * rho * rho
    
    # Now compute sectorial (m=n) and tesseral (m<n) 
    for m in range(1, m_max + 2):
      # Sectorial: n=m
      if m <= n_max + 1:
        V[m, m] = self.dnm[m] * (s * V[m-1, m-1] - t * W[m-1, m-1]) * rho
        W[m, m] = self.dnm[m] * (s * W[m-1, m-1] + t * V[m-1, m-1]) * rho
      
      # First tesseral: n=m+1
      if m + 1 <= n_max + 1:
        fac = np.sqrt(2*m + 3)
        V[m+1, m] = fac * u * V[m, m] * rho
        W[m+1, m] = fac * u * W[m, m] * rho
      
      # Remaining tesseral for this m
      for n in range(m + 2, n_max + 2):
        V[n, m] = (self.anm[n, m] * u * V[n-1, m]) * rho - (self.bnm[n, m] * V[n-2, m]) * rho * rho
        W[n, m] = (self.anm[n, m] * u * W[n-1, m]) * rho - (self.bnm[n, m] * W[n-2, m]) * rho * rho
    
    # Compute acceleration partials
    ax = 0.0
    ay = 0.0
    az = 0.0
    
    # Sum contributions from degree 2 onwards
    # Note: n=0 is point mass (handled separately), n=1 usually zero for Earth-centered
    for n in range(2, n_max + 1):
      # Normalization correction factor for derivative terms
      # This factor sqrt((2n+1)/(2n+3)) is needed because the derivative of a normalized
      # Legendre function of degree n involves normalized functions of degree n+1.
      norm_fix = np.sqrt((2.0 * n + 1.0) / (2.0 * n + 3.0))

      for m in range(0, min(n + 1, m_max + 1)):
        Cnm = self.coeffs.C[n, m]
        Snm = self.coeffs.S[n, m]
        
        # Factors for derivative computation
        if m == 0:
          # m = 0 case (zonal terms)
          # Following Montenbruck & Gill eq. 3.33
          fac1 = np.sqrt((n + 1) * (n + 2) / 2.0)
          fac2 = (n + 1)
          
          ax -= Cnm * fac1 * V[n+1, 1] * norm_fix
          ay -= Cnm * fac1 * W[n+1, 1] * norm_fix
          az -= Cnm * fac2 * V[n+1, 0] * norm_fix
        else:
          # m > 0 case (tesseral and sectorial terms)
          # Factor for m-1 term
          fac1 = np.sqrt((n - m + 1) * (n - m + 2))
          # Factor for m+1 term  
          fac2 = np.sqrt((n + m + 1) * (n + m + 2))
          # Factor for z derivative
          fac3 = np.sqrt((n - m + 1) * (n + m + 1))
          
          # Special case for m=1: different kronecker delta
          if m == 1:
            fac1 = np.sqrt((n) * (n + 1))
          
          ax += 0.5 * (
            -fac2 * (Cnm * V[n+1, m+1] + Snm * W[n+1, m+1])
            + fac1 * (Cnm * V[n+1, m-1] + Snm * W[n+1, m-1])
          ) * norm_fix
          ay += 0.5 * (
            -fac2 * (Cnm * W[n+1, m+1] - Snm * V[n+1, m+1])
            - fac1 * (Cnm * W[n+1, m-1] - Snm * V[n+1, m-1])
          ) * norm_fix
          az -= fac3 * (Cnm * V[n+1, m] + Snm * W[n+1, m]) * norm_fix
    
    # Scale by GM/Re^2 (Pines formulation scaling)
    scale = gp / (Re * Re)
    acc_vec = scale * np.array([ax, ay, az])
    
    # Add Two-Body Point Mass term explicitly (n=0)
    # This avoids issues with normalization factors for the monopole term
    acc_vec += -gp * pos_vec / r**3
    
    return acc_vec


def load_gravity_field(
  filepath   : Path,
  max_degree : int,
  max_order  : int,
) -> SphericalHarmonicsGravity:
  """
  Convenience function to load gravity field and create model.
  
  Input:
  ------
    filepath : Path
      Path to coefficient file.
    max_degree : int
      Maximum degree.
    max_order : int
      Maximum order.
  
  Output:
  -------
    model : SphericalHarmonicsGravity
      Ready-to-use gravity model.
  """
  coeffs = load_icgem_file(filepath, max_degree, max_order)
  return SphericalHarmonicsGravity(coeffs, max_degree, max_order)
