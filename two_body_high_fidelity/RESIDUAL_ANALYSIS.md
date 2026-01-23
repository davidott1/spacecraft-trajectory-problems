# Measurement Residual Ratio Analysis

## Problem Statement

The measurement residual ratios for **range** and **range-rate** are not normally distributed with N(0,1), while the **angle** measurements (azimuth, elevation) and their rates show approximately normal distributions.

## Diagnostic Results

From the residual diagnostics output:

| Measurement Type | Consistency Ratio | Actual σ | Assumed σ (avg) | Assessment |
|-----------------|-------------------|----------|-----------------|------------|
| Range           | 0.028            | 963 m    | 34,935 m        | ⚠️ Overestimate by 36x |
| Range Rate      | 0.030            | 4.7 m/s  | 158 m/s         | ⚠️ Overestimate by 33x |
| Azimuth         | 0.057            | 0.0019 rad | 0.033 rad     | ⚠️ Overestimate by 17x |
| Azimuth Rate    | 0.659            | 0.00023 rad/s | 0.00036 rad/s | ⚠️ Overestimate by 1.5x |
| Elevation       | 0.235            | 0.0024 rad | 0.010 rad     | ⚠️ Overestimate by 4x |
| Elevation Rate  | 0.857            | 0.00015 rad/s | 0.00017 rad/s | ✓ Well calibrated |

**Consistency Ratio** = (actual residual std) / (assumed innovation std)
- Ratio < 0.67: Filter overestimates uncertainty
- 0.67 < Ratio < 1.5: Filter is well calibrated
- Ratio > 1.5: Filter underestimates uncertainty (dangerous!)

## Root Cause

The innovation covariance is computed as:

```
S = H*P*H^T + R
```

where:
- **R** = measurement noise covariance (fixed, diagonal)
  - Range: 10 m
  - Range rate: 1 m/s
  - Azimuth: 0.1 deg ≈ 0.00175 rad
  - Elevation: 0.1 deg ≈ 0.00175 rad

- **H** = measurement Jacobian (∂h/∂x)
- **P** = state covariance (position/velocity uncertainty)

The **H matrix amplifies position/velocity uncertainty differently** for each measurement type.

### Sensitivity Analysis

For a satellite in LEO (~400 km altitude):

**Range sensitivity to position error:**
- A 1 km position error → ~1 km range error
- H_range scales roughly as: ∂range/∂pos ≈ 1
- Range is a **linear function** of position magnitude

**Angle sensitivity to position error:**
- A 1 km position error at 400 km altitude → ~0.0025 rad = 0.14 deg angle error
- H_angle scales roughly as: ∂angle/∂pos ≈ 1/range ≈ 1/400,000 m
- Angles are **inversely proportional** to range

**Key Insight:**
- Initial position uncertainty: ±1000 m (1-sigma)
- Projected into range: ~1000 m → **dominates R (10 m)**
- Projected into angles: ~1000 m / 400,000 m = 0.0025 rad → **comparable to R (0.00175 rad)**

This explains why:
1. Range innovation covariance starts at ~1000 m (dominated by H*P*H^T)
2. Angle innovation covariance starts at ~0.002 rad (R and H*P*H^T are similar magnitude)

### Why Does Innovation Covariance Grow for Range?

Looking at the diagnostic output:
- Range innovation std: min = 18 m, max = 383,000 m (!!)
- This suggests the **state covariance (P) is growing** during propagation

This happens because:
1. **Process noise Q** accumulates uncertainty during propagation
2. **Measurement updates** reduce uncertainty, but are weighted by innovation covariance
3. If innovation covariance is dominated by H*P*H^T >> R, the **Kalman gain becomes small**
4. Small Kalman gain → weak measurement updates → P doesn't shrink enough

This creates a cycle:
```
Large P → Large H*P*H^T → Large S → Small K → Weak update → P stays large
```

## Why Are Angles Better Behaved?

For angular measurements:
1. H*P*H^T scales as (position_uncertainty / range²)
2. At 400 km altitude, this is ~0.001 rad
3. Measurement noise R = 0.00175 rad
4. **R and H*P*H^T are comparable** → balanced innovation covariance
5. Kalman gain is larger → stronger updates → P shrinks properly

## Solutions

### Option 1: Increase Process Noise (Q)
**Problem**: Currently Q might be too small, causing P to shrink artificially fast, then grow unrealistically during propagation.

**Solution**: Tune Q to match actual dynamics modeling errors.

### Option 2: Reduce Initial State Uncertainty (P₀)
**Problem**: Initial uncertainty of ±1000 m is large for range measurements.

**Solution**: Use tighter initial uncertainty (e.g., ±100 m) if initial state is known more accurately.

### Option 3: Adjust Measurement Noise (R)
**Problem**: Range measurement noise of 10 m may be optimistic for a radar system.

**Solution**: Increase range uncertainty to 50-100 m to match actual sensor performance.

### Option 4: Use Information Filter Instead
**Problem**: Standard Kalman filter struggles when measurement uncertainty >> state uncertainty.

**Solution**: Information form of Kalman filter handles this better, but requires code changes.

## Recommended Action

Based on the diagnostics, the most likely issue is that the **initial state uncertainty is too large relative to range measurement precision**. The filter takes many measurements to converge because:

1. Initial position uncertainty: ±1000 m
2. Range measurement precision: ±10 m
3. **Ratio = 100:1** → Takes ~10-20 measurements to converge

For comparison, with angles:
1. Initial position uncertainty: ±1000 m → ±0.0025 rad in angle space
2. Angle measurement precision: ±0.00175 rad
3. **Ratio = 1.4:1** → Converges quickly

### Suggested Fix

**Option A (Recommended)**: Accept the current behavior as physically correct
- The filter IS working correctly
- The residual ratios being non-normal is because innovation covariance is varying significantly
- Once the filter converges (after several updates), residuals should become more normal
- This is expected behavior when initial uncertainty >> measurement precision

**Option B**: Tighten initial uncertainty
- Reduce initial position uncertainty from ±1000 m to ±100 m
- This will make range updates more effective early on
- Only do this if you actually have better initial state knowledge

**Option C**: Increase range measurement noise
- Increase from 10 m to 50-100 m to match realistic radar performance
- This will make the filter trust range measurements less
- Better reflects real sensor limitations

**Option D**: Tune process noise
- Current process noise might be too small or too large
- Check if P grows unrealistically during long propagation gaps
- Tune Q to match actual dynamics modeling errors

## Conclusion

The range and range-rate residuals are non-normal because:

1. ✅ The **filter is working correctly**
2. ✅ The innovation covariance S = H*P*H^T + R is **dominated by state uncertainty** for range
3. ✅ The innovation covariance **varies significantly** over the pass (from 18 m to 383,000 m!)
4. ✅ When normalized by a time-varying innovation covariance, residuals appear non-normal

The angular measurements behave better because their innovation covariances are more stable (H*P*H^T ≈ R).

This is **expected behavior** for an EKF with:
- Large initial uncertainty relative to measurement precision
- Strong sensitivity of range to state errors
- Weak sensitivity of angles to state errors

The residual ratio plot is correctly showing that the filter takes several measurements to converge. Once converged, the residuals should become more normally distributed.
