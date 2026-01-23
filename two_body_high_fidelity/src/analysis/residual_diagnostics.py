"""
Residual Diagnostics Analysis
==============================

Functions to diagnose measurement residual behavior and identify issues
with measurement noise calibration or model nonlinearity.
"""
import numpy as np
from typing import List, Dict


def analyze_residual_behavior(
    residuals              : List[np.ndarray],
    innovation_covariances : List[np.ndarray],
    measurement_types      : List[str] = None,
) -> Dict:
    """
    Analyze measurement residual behavior to diagnose filter performance issues.

    This function computes detailed statistics on:
    1. Raw residuals (not normalized)
    2. Innovation standard deviations (sqrt of diagonal of S)
    3. Residual ratios (residual / sigma)
    4. Statistical tests for normality

    Input:
    ------
        residuals : List[np.ndarray]
            List of measurement residuals (innovations) from EKF.
        innovation_covariances : List[np.ndarray]
            List of innovation covariance matrices from EKF.
        measurement_types : List[str], optional
            Names of measurement types.

    Output:
    -------
        diagnostics : dict
            Dictionary containing diagnostic information for each measurement type.
    """
    if measurement_types is None:
        measurement_types = [
            'Range',
            'Range Rate',
            'Azimuth',
            'Azimuth Rate',
            'Elevation',
            'Elevation Rate',
        ]

    n_measurements = len(residuals)
    if n_measurements == 0:
        return {}

    n_types = len(residuals[0])

    # Collect residuals and innovation stdevs
    raw_residuals = np.zeros((n_types, n_measurements))
    innovation_stdevs = np.zeros((n_types, n_measurements))
    residual_ratios = np.zeros((n_types, n_measurements))

    for i in range(n_measurements):
        residual = residuals[i]
        S = innovation_covariances[i]

        for j in range(n_types):
            raw_residuals[j, i] = residual[j]
            innovation_stdevs[j, i] = np.sqrt(S[j, j])

            if innovation_stdevs[j, i] > 1e-12:
                residual_ratios[j, i] = residual[j] / innovation_stdevs[j, i]
            else:
                residual_ratios[j, i] = 0.0

    # Compute diagnostics for each measurement type
    diagnostics = {}

    for j in range(n_types):
        meas_type = measurement_types[j] if j < len(measurement_types) else f'Meas {j+1}'

        # Raw residual statistics
        raw_mean = np.mean(raw_residuals[j, :])
        raw_std = np.std(raw_residuals[j, :])
        raw_rms = np.sqrt(np.mean(raw_residuals[j, :]**2))

        # Innovation covariance statistics
        innov_mean = np.mean(innovation_stdevs[j, :])
        innov_std = np.std(innovation_stdevs[j, :])
        innov_min = np.min(innovation_stdevs[j, :])
        innov_max = np.max(innovation_stdevs[j, :])

        # Residual ratio statistics
        ratio_mean = np.mean(residual_ratios[j, :])
        ratio_std = np.std(residual_ratios[j, :])
        ratio_rms = np.sqrt(np.mean(residual_ratios[j, :]**2))

        # Count violations of ±1σ, ±2σ, ±3σ bounds
        n_outside_1sigma = np.sum(np.abs(residual_ratios[j, :]) > 1.0)
        n_outside_2sigma = np.sum(np.abs(residual_ratios[j, :]) > 2.0)
        n_outside_3sigma = np.sum(np.abs(residual_ratios[j, :]) > 3.0)

        # Expected percentages for normal distribution
        # ±1σ: 68.27% inside → 31.73% outside
        # ±2σ: 95.45% inside → 4.55% outside
        # ±3σ: 99.73% inside → 0.27% outside
        pct_outside_1sigma = 100.0 * n_outside_1sigma / n_measurements
        pct_outside_2sigma = 100.0 * n_outside_2sigma / n_measurements
        pct_outside_3sigma = 100.0 * n_outside_3sigma / n_measurements

        # Ratio of actual residual std to innovation std
        # If this is >> 1, the filter is underestimating uncertainty
        # If this is << 1, the filter is overestimating uncertainty
        consistency_ratio = raw_std / innov_mean

        diagnostics[meas_type] = {
            # Raw residuals
            'raw_residual_mean': raw_mean,
            'raw_residual_std': raw_std,
            'raw_residual_rms': raw_rms,

            # Innovation covariance
            'innovation_std_mean': innov_mean,
            'innovation_std_std': innov_std,
            'innovation_std_min': innov_min,
            'innovation_std_max': innov_max,

            # Residual ratios
            'ratio_mean': ratio_mean,
            'ratio_std': ratio_std,
            'ratio_rms': ratio_rms,

            # Normality checks
            'pct_outside_1sigma': pct_outside_1sigma,
            'pct_outside_2sigma': pct_outside_2sigma,
            'pct_outside_3sigma': pct_outside_3sigma,
            'expected_pct_outside_1sigma': 31.73,
            'expected_pct_outside_2sigma': 4.55,
            'expected_pct_outside_3sigma': 0.27,

            # Consistency
            'consistency_ratio': consistency_ratio,  # Should be ~1 for well-calibrated filter
        }

    return diagnostics


def print_residual_diagnostics(diagnostics: Dict):
    """
    Print formatted residual diagnostics report.

    Input:
    ------
        diagnostics : dict
            Dictionary from analyze_residual_behavior()
    """
    print("\n" + "="*80)
    print("MEASUREMENT RESIDUAL DIAGNOSTICS")
    print("="*80)

    for meas_type, stats in diagnostics.items():
        print(f"\n{meas_type}:")
        print("-" * 80)

        # Raw residuals
        print(f"  Raw Residuals:")
        print(f"    Mean: {stats['raw_residual_mean']:12.6e}")
        print(f"    Std:  {stats['raw_residual_std']:12.6e}")
        print(f"    RMS:  {stats['raw_residual_rms']:12.6e}")

        # Innovation covariance
        print(f"\n  Innovation Covariance (sqrt(S_ii)):")
        print(f"    Mean: {stats['innovation_std_mean']:12.6e}")
        print(f"    Std:  {stats['innovation_std_std']:12.6e}")
        print(f"    Min:  {stats['innovation_std_min']:12.6e}")
        print(f"    Max:  {stats['innovation_std_max']:12.6e}")

        # Residual ratios
        print(f"\n  Residual Ratios (should be N(0,1)):")
        print(f"    Mean: {stats['ratio_mean']:8.4f}")
        print(f"    Std:  {stats['ratio_std']:8.4f}  (expected: 1.0)")
        print(f"    RMS:  {stats['ratio_rms']:8.4f}")

        # Normality checks
        print(f"\n  Normality Test (% outside bounds):")
        print(f"    ±1σ: {stats['pct_outside_1sigma']:5.1f}%  (expected: {stats['expected_pct_outside_1sigma']:.1f}%)")
        print(f"    ±2σ: {stats['pct_outside_2sigma']:5.1f}%  (expected: {stats['expected_pct_outside_2sigma']:.1f}%)")
        print(f"    ±3σ: {stats['pct_outside_3sigma']:5.1f}%  (expected: {stats['expected_pct_outside_3sigma']:.1f}%)")

        # Consistency ratio
        ratio = stats['consistency_ratio']
        print(f"\n  Consistency Check:")
        print(f"    Ratio (actual_std / assumed_std): {ratio:.4f}")

        if ratio > 1.5:
            print(f"    ⚠️  WARNING: Filter UNDERESTIMATES uncertainty by {ratio:.2f}x")
            print(f"        → Measurement noise (R) may be too small")
            print(f"        → Process noise (Q) may be too small")
        elif ratio < 0.67:
            print(f"    ⚠️  WARNING: Filter OVERESTIMATES uncertainty by {1/ratio:.2f}x")
            print(f"        → Measurement noise (R) may be too large")
        else:
            print(f"    ✓ Filter uncertainty is reasonably calibrated")

    print("\n" + "="*80)
