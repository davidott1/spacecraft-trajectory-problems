"""
Test script to verify J2 Jacobian against numerical finite differences.
"""
import numpy as np
import spiceypy as spice
from src.model.dynamics import TwoBodyGravity
from src.model.constants import SOLARSYSTEMCONSTANTS

# Load SPICE kernels for frame transformations
spice.furnsh('src/verification/fixtures/spice_kernels/naif0012.tls')
spice.furnsh('src/verification/fixtures/spice_kernels/pck00010.tpc')

# Create TwoBody with J2
earth = SOLARSYSTEMCONSTANTS.EARTH
two_body = TwoBodyGravity(
    gp=earth.GP,
    j2=earth.J2,
    pos_ref=earth.RADIUS.EQUATOR,
)

# Test position (LEO) - use non-zero y to exercise all terms
pos = np.array([5000e3, 4000e3, 3000e3])  # General position
time_et = 0.0

# Analytical Jacobian
jac_analytical = two_body.oblate_j2_jacobian(time_et, pos)

# Numerical Jacobian via finite differences
eps = 1.0  # 1 meter perturbation
jac_numerical = np.zeros((3, 3))
for i in range(3):
    pos_plus = pos.copy()
    pos_minus = pos.copy()
    pos_plus[i] += eps
    pos_minus[i] -= eps
    acc_plus = two_body.oblate_j2(time_et, pos_plus)
    acc_minus = two_body.oblate_j2(time_et, pos_minus)
    jac_numerical[:, i] = (acc_plus - acc_minus) / (2 * eps)

print('Analytical J2 Jacobian:')
print(jac_analytical)
print()
print('Numerical J2 Jacobian:')
print(jac_numerical)
print()
print('Relative error (should be small):')
rel_err = np.abs(jac_analytical - jac_numerical) / (np.abs(jac_numerical) + 1e-20)
print(rel_err)
print()
print('Max relative error:', np.max(rel_err))
print()
print('Absolute error:')
abs_err = np.abs(jac_analytical - jac_numerical)
print(abs_err)
print()
print('Max absolute error:', np.max(abs_err))
