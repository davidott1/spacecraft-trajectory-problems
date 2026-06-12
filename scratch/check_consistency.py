#!/usr/bin/env python3
"""Diagnostic script to check EKF filter vs smoother values."""

import pickle
import numpy as np
import os

# Find latest output
output_dirs = sorted([d for d in os.listdir('output') if os.path.isdir(f'output/{d}')])
latest = output_dirs[-1]
print(f"Checking: output/{latest}")

# Load EKF results
with open(f'output/{latest}/ekf_results.pkl', 'rb') as f:
    ekf = pickle.load(f)

# Check sizes
print(f"\nFilter states shape: {ekf['filter_states'].shape}")
print(f"Smoother states shape: {ekf['smoother_states'].shape}")

# Check state differences
state_diff = ekf['filter_states'] - ekf['smoother_states']
print(f"\nState difference stats:")
print(f"  Position diff (km): min={np.min(state_diff[:3,:]):.2e}, max={np.max(state_diff[:3,:]):.2e}")
print(f"  Velocity diff (km/s): min={np.min(state_diff[3:,:]):.2e}, max={np.max(state_diff[3:,:]):.2e}")

# Check covariance differences
P_f = ekf['filter_covariances']
P_s = ekf['smoother_covariances']
P_diff = P_f - P_s

# Check diagonals at a few points
for idx, name in [(0, 'first'), (P_f.shape[2]//2, 'middle'), (-1, 'last')]:
    print(f"\n{name.capitalize()} point (idx={idx}):")
    print(f"  P_filter diag: {np.diag(P_f[:,:,idx])}")
    print(f"  P_smooth diag: {np.diag(P_s[:,:,idx])}")
    print(f"  P_diff diag:   {np.diag(P_diff[:,:,idx])}")

# Check if they're nearly identical
if np.allclose(P_f, P_s):
    print("\n*** WARNING: Filter and smoother covariances are nearly identical! ***")

if np.allclose(ekf['filter_states'], ekf['smoother_states']):
    print("\n*** WARNING: Filter and smoother states are nearly identical! ***")
