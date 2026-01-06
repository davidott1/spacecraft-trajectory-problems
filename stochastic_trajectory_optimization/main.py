import numpy as np
from scipy.optimize import minimize

# --- CONSTANTS ---
MU = 398600.44
R1, R2 = 6778.0, 42164.0
V1_LEO = np.sqrt(MU/R1)

# --- MONTE CARLO SETUP ---
N = 10000
np.random.seed(42)
# 1% execution error (1-sigma)
ERR_SAMPLES = np.random.normal(1.0, 0.01, N)

def get_mcc(dv1_actual, target_r2):
    """Calculates the MCC required to fix apogee error from dv1."""
    v_per = V1_LEO + dv1_actual
    # Energy: E = v^2/2 - mu/r
    energy = (v_per**2) / 2 - MU / R1
    # Semi-major axis: a = -mu/(2E)
    a = -MU / (2 * energy)
    # Apogee: r_a = 2a - r_p
    r_a_actual = 2 * a - R1
    # Simplified MCC: 0.01 km/s cost for every 10km of apogee miss
    return np.abs(target_r2 - r_a_actual) * 0.001

def get_dv2_circularize(dv1_actual, target_r2):
    """Calculates ΔV2 required to circularize at target_r2 after MCC correction."""
    # After MCC, we're on a transfer orbit from R1 to target_r2
    # At apogee (target_r2), velocity on transfer orbit:
    a_transfer = (R1 + target_r2) / 2
    v_apogee_transfer = np.sqrt(MU * (2/target_r2 - 1/a_transfer))
    # Circular velocity at target_r2:
    v_circular = np.sqrt(MU / target_r2)
    # ΔV2 to circularize:
    return v_circular - v_apogee_transfer

def evaluate_strategy(dv_plan):
    """Evaluates a FIXED plan against N random realizations."""
    dv1_nom = dv_plan[0]  # Only ΔV1 is optimized; ΔV2 is determined by physics
    dv1_realizations = dv1_nom * ERR_SAMPLES
    mccs = get_mcc(dv1_realizations, R2)
    # ΔV2 is always what's needed to circularize at R2 (after MCC fixes apogee)
    dv2_required = get_dv2_circularize(dv1_realizations, R2)
    # Total DV = |Noisy Burn 1| + |MCC| + |ΔV2 to circularize|
    total_dvs = np.abs(dv1_realizations) + mccs + np.abs(dv2_required)
    return np.mean(total_dvs)

# 1. NOMINAL HOHMANN SOLUTION
# v_peri = sqrt(mu/r1 * 2*r2/(r1+r2))
dv1_hohmann = np.sqrt(MU/R1) * (np.sqrt(2*R2/(R1+R2)) - 1)
dv2_hohmann = np.sqrt(MU/R2) * (1 - np.sqrt(2*R1/(R1+R2)))
hohmann_plan = [dv1_hohmann, dv2_hohmann]

# 2. REACTIVE SOLUTION (Evaluate the Hohmann plan with noise)
dv_reactive = evaluate_strategy([dv1_hohmann])

# 3. ROBUST SOLUTION (Minimize the expected value)
res_robust = minimize(evaluate_strategy, [dv1_hohmann], method='Nelder-Mead',
                      options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
dv_robust = res_robust.fun
dv1_robust = res_robust.x[0]
dv2_robust = get_dv2_circularize(dv1_robust, R2)

# --- PRINT RESULTS ---
print("=== Stochastic Trajectory Optimization Results ===")
print(f"\nNominal Hohmann Transfer:")
print(f"  ΔV1: {dv1_hohmann:.4f} km/s")
print(f"  ΔV2: {dv2_hohmann:.4f} km/s")
print(f"  Total ΔV (deterministic): {dv1_hohmann + dv2_hohmann:.4f} km/s")

print(f"\nReactive Strategy (Hohmann + MCC correction):")
print(f"  Expected Total ΔV: {dv_reactive:.4f} km/s")

print(f"\nRobust Strategy (Optimized for uncertainty):")
print(f"  Optimized ΔV1: {dv1_robust:.4f} km/s")
print(f"  Corresponding ΔV2: {dv2_robust:.4f} km/s")
print(f"  Expected Total ΔV: {dv_robust:.4f} km/s")