"""Numba-accelerated kernels for the universal-variable Lambert solver.

Public functions (all take plain floats / numpy arrays, not QPointF):
    stumpff_all(z) -> (C, S, Cp, Sp)
    lambert_z_newton(r1, r2, A, dt, mu, z_init) -> (ok, z, C, S, y)
    lambert_solve_nb(r1_vec, r2_vec, dt, mu) -> (ok, v1, v2)
    lambert_with_jac_nb(r1_vec, r2_vec, dt, mu, z_init)
        -> (ok, v1, v2, Jv1_r1, Jv1_r2, Jv1_dt, Jv2_r1, Jv2_r2, Jv2_dt, z)

`ok == 0` means the Newton iteration failed and the caller should fall back
to the Python/brentq path.

Numerics mirror the pure-Python implementations in main.py. JIT is warmed
up at import time so the first UI interaction does not stall.
"""

import math

import numpy as np
from numba import njit


# Half-width of the antipodal-degenerate angular band (radians). Inside this
# band the natural sign(cross_z) tie-break is numerically unreliable; we use
# the per-segment `side` hint (±1) to pick a deterministic branch.
SIDE_EPS = 1.0e-3


@njit(cache=True, fastmath=False, inline="always")
def stumpff_all(z):
    """Return (C, S, Cp, Sp) sharing a single sqrt + trig/hyperbolic eval."""
    if abs(z) < 1e-6:
        z2 = z * z
        C = 0.5 - z / 24.0 + z2 / 720.0
        S = 1.0 / 6.0 - z / 120.0 + z2 / 5040.0
        Cp = -1.0 / 24.0 + z / 360.0 - z2 / 13440.0
        Sp = -1.0 / 120.0 + z / 2520.0 - z2 / 120960.0
        return C, S, Cp, Sp
    if z > 0.0:
        sqz = math.sqrt(z)
        c = math.cos(sqz)
        s = math.sin(sqz)
        C = (1.0 - c) / z
        S = (sqz - s) / (sqz * z)
    else:
        sqz = math.sqrt(-z)
        c = math.cosh(sqz)
        s = math.sinh(sqz)
        C = (c - 1.0) / (-z)
        S = (s - sqz) / (-sqz * z)
    inv_2z = 0.5 / z
    Cp = (1.0 - z * S - 2.0 * C) * inv_2z
    Sp = (C - 3.0 * S) * inv_2z
    return C, S, Cp, Sp


@njit(cache=True, fastmath=False)
def lambert_z_newton(r1, r2, A, dt, mu, z_init):
    """Newton iteration for the universal-variable Lambert equation.

    Returns (ok, z, C, S, y). ok == 0 indicates failure and signals the
    caller to fall back to a bracketed solver.
    """
    sqrtmu_dt = math.sqrt(mu) * dt
    Z_HIGH = 4.0 * math.pi * math.pi - 0.5
    Z_LOW = -200.0

    z = z_init
    if z < Z_LOW:
        z = Z_LOW
    elif z > Z_HIGH:
        z = Z_HIGH

    for _ in range(30):
        C, S, Cp, Sp = stumpff_all(z)
        if C <= 1e-30:
            break
        sqrtC = math.sqrt(C)
        y = r1 + r2 + A * (z * S - 1.0) / sqrtC
        if y < 0.0:
            z += 0.5
            continue
        sqrty = math.sqrt(y)
        yC = y / C
        yC15 = yC * math.sqrt(yC)
        Fval = yC15 * S + A * sqrty - sqrtmu_dt
        dchi_dz = (S + z * Sp) / sqrtC - (z * S - 1.0) * Cp / (2.0 * C * sqrtC)
        dy_dz = A * dchi_dz
        dF_dy = 1.5 * S * sqrty / (C * sqrtC) + A / (2.0 * sqrty)
        dF_dC = -1.5 * yC15 * S / C
        dF_dS = yC15
        dF_dz = dF_dy * dy_dz + dF_dC * Cp + dF_dS * Sp
        if abs(dF_dz) < 1e-20:
            break
        dz = -Fval / dF_dz
        z_new = z + dz
        if z_new < Z_LOW:
            z_new = 0.5 * (z + Z_LOW)
        elif z_new > Z_HIGH:
            z_new = 0.5 * (z + Z_HIGH)
        if abs(z_new - z) < 1e-12:
            z = z_new
            C, S, _, _ = stumpff_all(z)
            if C <= 1e-30:
                break
            sqrtC = math.sqrt(C)
            y = r1 + r2 + A * (z * S - 1.0) / sqrtC
            if y < 0.0:
                break
            return 1, z, C, S, y
        z = z_new
    return 0, 0.0, 0.0, 0.0, 0.0


@njit(cache=True, fastmath=False)
def lambert_solve_nb(r1_vec, r2_vec, dt, mu, side):
    """Lambert solver. Returns (ok, v1, v2). ok=0 -> caller should fall back.

    `side` selects the transfer branch when the geometry is within SIDE_EPS
    of antipodal: side > 0 forces a short-way (CCW, h_z>0) transfer; side < 0
    forces a long-way (CW, h_z<0) transfer. side == 0 means "auto": fall
    back to sign(cross_z) (or +1 if cross_z is exactly zero) inside the band.
    Outside the band, side is ignored."""
    v_fail = np.empty(2)
    v_fail[0] = 0.0
    v_fail[1] = 0.0

    r1 = math.sqrt(r1_vec[0] * r1_vec[0] + r1_vec[1] * r1_vec[1])
    r2 = math.sqrt(r2_vec[0] * r2_vec[0] + r2_vec[1] * r2_vec[1])
    if r1 < 1e-10 or r2 < 1e-10 or dt < 1e-10:
        return 0, v_fail, v_fail

    cos_dtheta = (r1_vec[0] * r2_vec[0] + r1_vec[1] * r2_vec[1]) / (r1 * r2)
    if cos_dtheta > 1.0:
        cos_dtheta = 1.0
    elif cos_dtheta < -1.0:
        cos_dtheta = -1.0
    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
    dtheta = math.acos(cos_dtheta)
    if cross_z < 0.0:
        dtheta = 2.0 * math.pi - dtheta
    sin_dtheta = math.sin(dtheta)
    if abs(1.0 - cos_dtheta) < 1e-14:
        # dtheta ~ 0: degenerate but not the antipodal case; bail.
        return 0, v_fail, v_fail
    if abs(sin_dtheta) < SIDE_EPS:
        side_eff = side
        if side_eff == 0.0:
            side_eff = 1.0 if cross_z >= 0.0 else -1.0
        # Rotate r2_vec by -copysign(SIDE_EPS, side_eff) so the geometry
        # (and hence A) is self-consistent with the chosen branch. Without
        # this, A=eps but r2_vec is still exactly antipodal to r1_vec, so
        # g_=A*sqrt(y/mu) is tiny while r2_vec - f*r1_vec is exactly zero,
        # giving 0/0 ~ tiny in the velocity reconstruction.
        phi = -math.copysign(SIDE_EPS, side_eff)
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        new_r2 = np.empty(2)
        new_r2[0] = cphi * r2_vec[0] - sphi * r2_vec[1]
        new_r2[1] = sphi * r2_vec[0] + cphi * r2_vec[1]
        r2_vec = new_r2
        cos_dtheta = (r1_vec[0] * r2_vec[0] + r1_vec[1] * r2_vec[1]) / (r1 * r2)
        cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
        dtheta = math.pi + phi  # = math.pi - copysign(SIDE_EPS, side_eff)
        sin_dtheta = math.sin(dtheta)

    A = sin_dtheta * math.sqrt(r1 * r2 / (1.0 - cos_dtheta))

    ok, z, C, S, y = lambert_z_newton(r1, r2, A, dt, mu, 0.0)
    if ok == 0:
        return 0, v_fail, v_fail

    f_ = 1.0 - y / r1
    g_ = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2

    v1 = np.empty(2)
    v2 = np.empty(2)
    v1[0] = (r2_vec[0] - f_ * r1_vec[0]) / g_
    v1[1] = (r2_vec[1] - f_ * r1_vec[1]) / g_
    v2[0] = (gdot * r2_vec[0] - r1_vec[0]) / g_
    v2[1] = (gdot * r2_vec[1] - r1_vec[1]) / g_
    return 1, v1, v2


@njit(cache=True, fastmath=False)
def lambert_with_jac_nb(r1_vec, r2_vec, dt, mu, z_init, side):
    """Lambert + analytic Jacobians.

    `side` selects the transfer branch in the antipodal-degenerate band
    (see lambert_solve_nb).

    Returns (ok, v1, v2, Jv1_r1, Jv1_r2, Jv1_dt, Jv2_r1, Jv2_r2, Jv2_dt, z).
    ok=0 signals the caller to fall back to the Python path (straight-line
    or brentq). All returned arrays are freshly allocated.
    """
    zero_v = np.zeros(2)
    zero_m = np.zeros((2, 2))

    r1 = math.sqrt(r1_vec[0] * r1_vec[0] + r1_vec[1] * r1_vec[1])
    r2 = math.sqrt(r2_vec[0] * r2_vec[0] + r2_vec[1] * r2_vec[1])
    if r1 < 1e-10 or r2 < 1e-10 or dt < 1e-10:
        return 0, zero_v, zero_v, zero_m, zero_m, zero_v, zero_m, zero_m, zero_v, 0.0

    cos_dtheta = (r1_vec[0] * r2_vec[0] + r1_vec[1] * r2_vec[1]) / (r1 * r2)
    if cos_dtheta > 1.0:
        cos_dtheta = 1.0
    elif cos_dtheta < -1.0:
        cos_dtheta = -1.0
    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
    dtheta = math.acos(cos_dtheta)
    if cross_z < 0.0:
        dtheta = 2.0 * math.pi - dtheta
    sin_dtheta = math.sin(dtheta)
    if abs(1.0 - cos_dtheta) < 1e-14:
        return 0, zero_v, zero_v, zero_m, zero_m, zero_v, zero_m, zero_m, zero_v, 0.0
    if abs(sin_dtheta) < SIDE_EPS:
        side_eff = side
        if side_eff == 0.0:
            side_eff = 1.0 if cross_z >= 0.0 else -1.0
        # See lambert_solve_nb for the rationale: rotate r2_vec so the
        # geometry is self-consistent with the chosen branch. Jacobians
        # below are wrt the perturbed r2_vec; for SIDE_EPS=1e-3 this is
        # an O(1e-3) approximation that the optimizer tolerates.
        phi = -math.copysign(SIDE_EPS, side_eff)
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        new_r2 = np.empty(2)
        new_r2[0] = cphi * r2_vec[0] - sphi * r2_vec[1]
        new_r2[1] = sphi * r2_vec[0] + cphi * r2_vec[1]
        r2_vec = new_r2
        cos_dtheta = (r1_vec[0] * r2_vec[0] + r1_vec[1] * r2_vec[1]) / (r1 * r2)
        cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
        dtheta = math.pi + phi
        sin_dtheta = math.sin(dtheta)

    A = sin_dtheta * math.sqrt(r1 * r2 / (1.0 - cos_dtheta))

    ok, z, C, S, y = lambert_z_newton(r1, r2, A, dt, mu, z_init)
    if ok == 0 or y <= 0.0 or C <= 0.0 or abs(A) < 1e-15:
        return 0, zero_v, zero_v, zero_m, zero_m, zero_v, zero_m, zero_m, zero_v, 0.0

    _, _, Cp, Sp = stumpff_all(z)
    sqrtC = math.sqrt(C)
    sqrty = math.sqrt(y)
    C15 = C * sqrtC
    C25 = C * C15
    y15 = y * sqrty
    chi = (z * S - 1.0) / sqrtC
    dchi_dz = (S + z * Sp) / sqrtC - (z * S - 1.0) * Cp / (2.0 * C15)

    # dA/dr1, dA/dr2 (2-vectors).
    inv_2A = 1.0 / (2.0 * A)
    dA_dr1_x = ((r2 / r1) * r1_vec[0] + r2_vec[0]) * inv_2A
    dA_dr1_y = ((r2 / r1) * r1_vec[1] + r2_vec[1]) * inv_2A
    dA_dr2_x = ((r1 / r2) * r2_vec[0] + r1_vec[0]) * inv_2A
    dA_dr2_y = ((r1 / r2) * r2_vec[1] + r1_vec[1]) * inv_2A

    # dy at fixed z.
    inv_r1 = 1.0 / r1
    inv_r2 = 1.0 / r2
    dyz_dr1_x = r1_vec[0] * inv_r1 + dA_dr1_x * chi
    dyz_dr1_y = r1_vec[1] * inv_r1 + dA_dr1_y * chi
    dyz_dr2_x = r2_vec[0] * inv_r2 + dA_dr2_x * chi
    dyz_dr2_y = r2_vec[1] * inv_r2 + dA_dr2_y * chi
    dy_dz = A * dchi_dz

    # F partials.
    dF_dy = 1.5 * S * sqrty / C15 + A / (2.0 * sqrty)
    dF_dC = -1.5 * y15 * S / C25
    dF_dS = y15 / C15
    dF_dA = sqrty
    dF_dz = dF_dy * dy_dz + dF_dC * Cp + dF_dS * Sp
    if abs(dF_dz) < 1e-20:
        return 0, zero_v, zero_v, zero_m, zero_m, zero_v, zero_m, zero_m, zero_v, 0.0

    dFz_dr1_x = dF_dy * dyz_dr1_x + dF_dA * dA_dr1_x
    dFz_dr1_y = dF_dy * dyz_dr1_y + dF_dA * dA_dr1_y
    dFz_dr2_x = dF_dy * dyz_dr2_x + dF_dA * dA_dr2_x
    dFz_dr2_y = dF_dy * dyz_dr2_y + dF_dA * dA_dr2_y
    dFz_dt = -math.sqrt(mu)

    inv_dFz = 1.0 / dF_dz
    dz_dr1_x = -dFz_dr1_x * inv_dFz
    dz_dr1_y = -dFz_dr1_y * inv_dFz
    dz_dr2_x = -dFz_dr2_x * inv_dFz
    dz_dr2_y = -dFz_dr2_y * inv_dFz
    dz_dt = -dFz_dt * inv_dFz

    dy_dr1_x = dyz_dr1_x + dy_dz * dz_dr1_x
    dy_dr1_y = dyz_dr1_y + dy_dz * dz_dr1_y
    dy_dr2_x = dyz_dr2_x + dy_dz * dz_dr2_x
    dy_dr2_y = dyz_dr2_y + dy_dz * dz_dr2_y
    dy_dt = dy_dz * dz_dt

    f_ = 1.0 - y / r1
    g_ = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2
    inv_g = 1.0 / g_
    v1x = (r2_vec[0] - f_ * r1_vec[0]) * inv_g
    v1y = (r2_vec[1] - f_ * r1_vec[1]) * inv_g
    v2x = (gdot * r2_vec[0] - r1_vec[0]) * inv_g
    v2y = (gdot * r2_vec[1] - r1_vec[1]) * inv_g

    # df, dgdot partials.
    y_over_r1sq = y / (r1 * r1)
    y_over_r2sq = y / (r2 * r2)
    df_dr1_x = -dy_dr1_x / r1 + y_over_r1sq * (r1_vec[0] * inv_r1)
    df_dr1_y = -dy_dr1_y / r1 + y_over_r1sq * (r1_vec[1] * inv_r1)
    df_dr2_x = -dy_dr2_x / r1
    df_dr2_y = -dy_dr2_y / r1
    df_dt = -dy_dt / r1

    dgdot_dr1_x = -dy_dr1_x / r2
    dgdot_dr1_y = -dy_dr1_y / r2
    dgdot_dr2_x = -dy_dr2_x / r2 + y_over_r2sq * (r2_vec[0] * inv_r2)
    dgdot_dr2_y = -dy_dr2_y / r2 + y_over_r2sq * (r2_vec[1] * inv_r2)
    dgdot_dt = -dy_dt / r2

    sq_yom = math.sqrt(y / mu)
    A_2sqyM = A / (2.0 * math.sqrt(y * mu))
    dg_dr1_x = dA_dr1_x * sq_yom + A_2sqyM * dy_dr1_x
    dg_dr1_y = dA_dr1_y * sq_yom + A_2sqyM * dy_dr1_y
    dg_dr2_x = dA_dr2_x * sq_yom + A_2sqyM * dy_dr2_x
    dg_dr2_y = dA_dr2_y * sq_yom + A_2sqyM * dy_dr2_y
    dg_dt = A_2sqyM * dy_dt

    v1 = np.empty(2)
    v1[0] = v1x
    v1[1] = v1y
    v2 = np.empty(2)
    v2[0] = v2x
    v2[1] = v2y

    # Jv1_r1 = (-outer(r1_vec, df_dr1) - f_ * I2) / g_ - outer(v1, dg_dr1) / g_
    Jv1_r1 = np.empty((2, 2))
    Jv1_r1[0, 0] = (-r1_vec[0] * df_dr1_x - f_) * inv_g - v1x * dg_dr1_x * inv_g
    Jv1_r1[0, 1] = (-r1_vec[0] * df_dr1_y) * inv_g - v1x * dg_dr1_y * inv_g
    Jv1_r1[1, 0] = (-r1_vec[1] * df_dr1_x) * inv_g - v1y * dg_dr1_x * inv_g
    Jv1_r1[1, 1] = (-r1_vec[1] * df_dr1_y - f_) * inv_g - v1y * dg_dr1_y * inv_g

    # Jv1_r2 = (I2 - outer(r1_vec, df_dr2)) / g_ - outer(v1, dg_dr2) / g_
    Jv1_r2 = np.empty((2, 2))
    Jv1_r2[0, 0] = (1.0 - r1_vec[0] * df_dr2_x) * inv_g - v1x * dg_dr2_x * inv_g
    Jv1_r2[0, 1] = (-r1_vec[0] * df_dr2_y) * inv_g - v1x * dg_dr2_y * inv_g
    Jv1_r2[1, 0] = (-r1_vec[1] * df_dr2_x) * inv_g - v1y * dg_dr2_x * inv_g
    Jv1_r2[1, 1] = (1.0 - r1_vec[1] * df_dr2_y) * inv_g - v1y * dg_dr2_y * inv_g

    # Jv1_dt = (-df_dt * r1_vec) / g_ - v1 * dg_dt / g_
    Jv1_dt = np.empty(2)
    Jv1_dt[0] = (-df_dt * r1_vec[0]) * inv_g - v1x * dg_dt * inv_g
    Jv1_dt[1] = (-df_dt * r1_vec[1]) * inv_g - v1y * dg_dt * inv_g

    # Jv2_r1 = (outer(r2_vec, dgdot_dr1) - I2) / g_ - outer(v2, dg_dr1) / g_
    Jv2_r1 = np.empty((2, 2))
    Jv2_r1[0, 0] = (r2_vec[0] * dgdot_dr1_x - 1.0) * inv_g - v2x * dg_dr1_x * inv_g
    Jv2_r1[0, 1] = (r2_vec[0] * dgdot_dr1_y) * inv_g - v2x * dg_dr1_y * inv_g
    Jv2_r1[1, 0] = (r2_vec[1] * dgdot_dr1_x) * inv_g - v2y * dg_dr1_x * inv_g
    Jv2_r1[1, 1] = (r2_vec[1] * dgdot_dr1_y - 1.0) * inv_g - v2y * dg_dr1_y * inv_g

    # Jv2_r2 = (outer(r2_vec, dgdot_dr2) + gdot * I2) / g_ - outer(v2, dg_dr2) / g_
    Jv2_r2 = np.empty((2, 2))
    Jv2_r2[0, 0] = (r2_vec[0] * dgdot_dr2_x + gdot) * inv_g - v2x * dg_dr2_x * inv_g
    Jv2_r2[0, 1] = (r2_vec[0] * dgdot_dr2_y) * inv_g - v2x * dg_dr2_y * inv_g
    Jv2_r2[1, 0] = (r2_vec[1] * dgdot_dr2_x) * inv_g - v2y * dg_dr2_x * inv_g
    Jv2_r2[1, 1] = (r2_vec[1] * dgdot_dr2_y + gdot) * inv_g - v2y * dg_dr2_y * inv_g

    # Jv2_dt = (dgdot_dt * r2_vec) / g_ - v2 * dg_dt / g_
    Jv2_dt = np.empty(2)
    Jv2_dt[0] = (dgdot_dt * r2_vec[0]) * inv_g - v2x * dg_dt * inv_g
    Jv2_dt[1] = (dgdot_dt * r2_vec[1]) * inv_g - v2y * dg_dt * inv_g

    return 1, v1, v2, Jv1_r1, Jv1_r2, Jv1_dt, Jv2_r1, Jv2_r2, Jv2_dt, z


# ---------------------------------------------------------------------------
# Batched fun_and_grad: runs the entire BFGS inner evaluation (all segments
# + dv-node accumulation) in a single njit call. Eliminates N Python->njit
# crossings per BFGS iteration and avoids per-segment numpy allocations.
#
# Segment-start/end kinds:
#   0 = movable (use x_vec[2*var_idx : 2*var_idx+2])
#   1 = shape triangle (use shape_pos_tri)
#   2 = shape square   (use shape_pos_sq)
#   3 = fixed (use seg_*_fixed[i])
#
# dv-node kinds:
#   0 = movable midpoint (has both in_seg and out_seg)
#   1 = shape outgoing   (shape is start of out_seg)
#   2 = shape incoming   (shape is end of in_seg)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=False, inline="always")
def _seg_pos(x_vec, kind, var_idx, shape_pos_tri, shape_pos_sq, fixed_xy):
    """Resolve a segment-endpoint position. fixed_xy is (2,) and only used when kind==3."""
    if kind == 0:
        out = np.empty(2)
        out[0] = x_vec[2 * var_idx]
        out[1] = x_vec[2 * var_idx + 1]
        return out
    if kind == 1:
        return shape_pos_tri
    if kind == 2:
        return shape_pos_sq
    return fixed_xy


@njit(cache=True, fastmath=False)
def _solve_segment(
    r0, rf, tof_seg, mult, z_init, side,
    center, mu, use_parabola, cg_mode, const_g_vec, g_mag,
    # output slots (all preallocated views into per-segment arrays):
    v0_out, vf_out,
    Jv0_r0_out, Jv0_rf_out, Jv0_dt_out,
    Jvf_r0_out, Jvf_rf_out, Jvf_dt_out,
):
    """Solve one segment (parabola or Lambert) and write results into
    preallocated out slots. dt-Jacobians already multiplied by mult.

    Returns (ok, z) where z is the converged universal-variable (for
    warm-starting; 0.0 for parabola).
    """
    if use_parabola:
        # Parabolic arc: r(t) = r0 + v0 t + 0.5 g t^2. g is constant in
        # constant-gravity env, else points from r0 -> center with magnitude g_mag.
        if cg_mode:
            gx = const_g_vec[0]
            gy = const_g_vec[1]
            dgvec_dr0_xx = 0.0
            dgvec_dr0_xy = 0.0
            dgvec_dr0_yx = 0.0
            dgvec_dr0_yy = 0.0
        else:
            dx = center[0] - r0[0]
            dy = center[1] - r0[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                gx = 0.0
                gy = 0.0
                dgvec_dr0_xx = 0.0
                dgvec_dr0_xy = 0.0
                dgvec_dr0_yx = 0.0
                dgvec_dr0_yy = 0.0
            else:
                inv_d = 1.0 / dist
                ghx = dx * inv_d
                ghy = dy * inv_d
                gx = g_mag * ghx
                gy = g_mag * ghy
                # dgvec_dr0 = (g_mag / dist) * (-I2 + outer(g_hat, g_hat))
                # note d(g_hat)/d(r0) = (g_hat g_hat^T - I) / dist, and g_vec = g_mag * (center - r0)/dist
                # so dg_vec/dr0 = -(g_mag / dist) (I - g_hat g_hat^T) = (g_mag / dist)(g_hat g_hat^T - I)
                fac = g_mag * inv_d
                dgvec_dr0_xx = fac * (ghx * ghx - 1.0)
                dgvec_dr0_xy = fac * (ghx * ghy)
                dgvec_dr0_yx = fac * (ghy * ghx)
                dgvec_dr0_yy = fac * (ghy * ghy - 1.0)
        inv_t = 1.0 / tof_seg
        disp_x = rf[0] - r0[0]
        disp_y = rf[1] - r0[1]
        meanv_x = disp_x * inv_t
        meanv_y = disp_y * inv_t
        half_g_t_x = 0.5 * gx * tof_seg
        half_g_t_y = 0.5 * gy * tof_seg
        v0_out[0] = meanv_x - half_g_t_x
        v0_out[1] = meanv_y - half_g_t_y
        vf_out[0] = meanv_x + half_g_t_x
        vf_out[1] = meanv_y + half_g_t_y
        # J_v0_r0 = -I/t - 0.5 t * dgvec_dr0
        Jv0_r0_out[0, 0] = -inv_t - 0.5 * tof_seg * dgvec_dr0_xx
        Jv0_r0_out[0, 1] = -0.5 * tof_seg * dgvec_dr0_xy
        Jv0_r0_out[1, 0] = -0.5 * tof_seg * dgvec_dr0_yx
        Jv0_r0_out[1, 1] = -inv_t - 0.5 * tof_seg * dgvec_dr0_yy
        # J_v0_rf = I/t
        Jv0_rf_out[0, 0] = inv_t
        Jv0_rf_out[0, 1] = 0.0
        Jv0_rf_out[1, 0] = 0.0
        Jv0_rf_out[1, 1] = inv_t
        # J_v0_tf_seg = -disp / t^2 - 0.5 * g_vec   ; then multiply by mult
        Jv0_dt_out[0] = (-disp_x * inv_t * inv_t - 0.5 * gx) * mult
        Jv0_dt_out[1] = (-disp_y * inv_t * inv_t - 0.5 * gy) * mult
        # J_vf_r0 = -I/t + 0.5 t * dgvec_dr0
        Jvf_r0_out[0, 0] = -inv_t + 0.5 * tof_seg * dgvec_dr0_xx
        Jvf_r0_out[0, 1] = 0.5 * tof_seg * dgvec_dr0_xy
        Jvf_r0_out[1, 0] = 0.5 * tof_seg * dgvec_dr0_yx
        Jvf_r0_out[1, 1] = -inv_t + 0.5 * tof_seg * dgvec_dr0_yy
        # J_vf_rf = I/t
        Jvf_rf_out[0, 0] = inv_t
        Jvf_rf_out[0, 1] = 0.0
        Jvf_rf_out[1, 0] = 0.0
        Jvf_rf_out[1, 1] = inv_t
        Jvf_dt_out[0] = (-disp_x * inv_t * inv_t + 0.5 * gx) * mult
        Jvf_dt_out[1] = (-disp_y * inv_t * inv_t + 0.5 * gy) * mult
        return 1, 0.0

    # Lambert branch. Use translated vectors (relative to gravity center).
    r1x = r0[0] - center[0]
    r1y = r0[1] - center[1]
    r2x = rf[0] - center[0]
    r2y = rf[1] - center[1]

    r1 = math.sqrt(r1x * r1x + r1y * r1y)
    r2 = math.sqrt(r2x * r2x + r2y * r2y)
    if r1 < 1e-10 or r2 < 1e-10 or tof_seg < 1e-10:
        # straight-line fallback
        return _fill_straight_line(
            r1x, r1y, r2x, r2y, tof_seg, mult,
            v0_out, vf_out,
            Jv0_r0_out, Jv0_rf_out, Jv0_dt_out,
            Jvf_r0_out, Jvf_rf_out, Jvf_dt_out,
        )

    cos_dtheta = (r1x * r2x + r1y * r2y) / (r1 * r2)
    if cos_dtheta > 1.0:
        cos_dtheta = 1.0
    elif cos_dtheta < -1.0:
        cos_dtheta = -1.0
    cross_z = r1x * r2y - r1y * r2x
    dtheta = math.acos(cos_dtheta)
    if cross_z < 0.0:
        dtheta = 2.0 * math.pi - dtheta
    sin_dtheta = math.sin(dtheta)
    if abs(1.0 - cos_dtheta) < 1e-14:
        return _fill_straight_line(
            r1x, r1y, r2x, r2y, tof_seg, mult,
            v0_out, vf_out,
            Jv0_r0_out, Jv0_rf_out, Jv0_dt_out,
            Jvf_r0_out, Jvf_rf_out, Jvf_dt_out,
        )
    if abs(sin_dtheta) < SIDE_EPS:
        side_eff = side
        if side_eff == 0.0:
            side_eff = 1.0 if cross_z >= 0.0 else -1.0
        dtheta = math.pi - math.copysign(SIDE_EPS, side_eff)
        sin_dtheta = math.sin(dtheta)
        cos_dtheta = math.cos(dtheta)

    A = sin_dtheta * math.sqrt(r1 * r2 / (1.0 - cos_dtheta))
    ok, z, C, S, y = lambert_z_newton(r1, r2, A, tof_seg, mu, z_init)
    if ok == 0 or y <= 0.0 or C <= 0.0 or abs(A) < 1e-15:
        # Caller will detect ok==0 and may retry via Python fallback.
        return 0, 0.0

    _, _, Cp, Sp = stumpff_all(z)
    sqrtC = math.sqrt(C)
    sqrty = math.sqrt(y)
    C15 = C * sqrtC
    C25 = C * C15
    y15 = y * sqrty
    chi = (z * S - 1.0) / sqrtC
    dchi_dz = (S + z * Sp) / sqrtC - (z * S - 1.0) * Cp / (2.0 * C15)

    inv_2A = 1.0 / (2.0 * A)
    dA_dr1_x = ((r2 / r1) * r1x + r2x) * inv_2A
    dA_dr1_y = ((r2 / r1) * r1y + r2y) * inv_2A
    dA_dr2_x = ((r1 / r2) * r2x + r1x) * inv_2A
    dA_dr2_y = ((r1 / r2) * r2y + r1y) * inv_2A

    inv_r1 = 1.0 / r1
    inv_r2 = 1.0 / r2
    dyz_dr1_x = r1x * inv_r1 + dA_dr1_x * chi
    dyz_dr1_y = r1y * inv_r1 + dA_dr1_y * chi
    dyz_dr2_x = r2x * inv_r2 + dA_dr2_x * chi
    dyz_dr2_y = r2y * inv_r2 + dA_dr2_y * chi
    dy_dz = A * dchi_dz

    dF_dy = 1.5 * S * sqrty / C15 + A / (2.0 * sqrty)
    dF_dC = -1.5 * y15 * S / C25
    dF_dS = y15 / C15
    dF_dA = sqrty
    dF_dz = dF_dy * dy_dz + dF_dC * Cp + dF_dS * Sp
    if abs(dF_dz) < 1e-20:
        return 0, 0.0

    dFz_dr1_x = dF_dy * dyz_dr1_x + dF_dA * dA_dr1_x
    dFz_dr1_y = dF_dy * dyz_dr1_y + dF_dA * dA_dr1_y
    dFz_dr2_x = dF_dy * dyz_dr2_x + dF_dA * dA_dr2_x
    dFz_dr2_y = dF_dy * dyz_dr2_y + dF_dA * dA_dr2_y
    dFz_dt = -math.sqrt(mu)

    inv_dFz = 1.0 / dF_dz
    dz_dr1_x = -dFz_dr1_x * inv_dFz
    dz_dr1_y = -dFz_dr1_y * inv_dFz
    dz_dr2_x = -dFz_dr2_x * inv_dFz
    dz_dr2_y = -dFz_dr2_y * inv_dFz
    dz_dt = -dFz_dt * inv_dFz

    dy_dr1_x = dyz_dr1_x + dy_dz * dz_dr1_x
    dy_dr1_y = dyz_dr1_y + dy_dz * dz_dr1_y
    dy_dr2_x = dyz_dr2_x + dy_dz * dz_dr2_x
    dy_dr2_y = dyz_dr2_y + dy_dz * dz_dr2_y
    dy_dt = dy_dz * dz_dt

    f_ = 1.0 - y / r1
    g_ = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2
    inv_g = 1.0 / g_
    v1x = (r2x - f_ * r1x) * inv_g
    v1y = (r2y - f_ * r1y) * inv_g
    v2x = (gdot * r2x - r1x) * inv_g
    v2y = (gdot * r2y - r1y) * inv_g

    y_over_r1sq = y / (r1 * r1)
    y_over_r2sq = y / (r2 * r2)
    df_dr1_x = -dy_dr1_x / r1 + y_over_r1sq * (r1x * inv_r1)
    df_dr1_y = -dy_dr1_y / r1 + y_over_r1sq * (r1y * inv_r1)
    df_dr2_x = -dy_dr2_x / r1
    df_dr2_y = -dy_dr2_y / r1
    df_dt = -dy_dt / r1

    dgdot_dr1_x = -dy_dr1_x / r2
    dgdot_dr1_y = -dy_dr1_y / r2
    dgdot_dr2_x = -dy_dr2_x / r2 + y_over_r2sq * (r2x * inv_r2)
    dgdot_dr2_y = -dy_dr2_y / r2 + y_over_r2sq * (r2y * inv_r2)
    dgdot_dt = -dy_dt / r2

    sq_yom = math.sqrt(y / mu)
    A_2sqyM = A / (2.0 * math.sqrt(y * mu))
    dg_dr1_x = dA_dr1_x * sq_yom + A_2sqyM * dy_dr1_x
    dg_dr1_y = dA_dr1_y * sq_yom + A_2sqyM * dy_dr1_y
    dg_dr2_x = dA_dr2_x * sq_yom + A_2sqyM * dy_dr2_x
    dg_dr2_y = dA_dr2_y * sq_yom + A_2sqyM * dy_dr2_y
    dg_dt = A_2sqyM * dy_dt

    v0_out[0] = v1x
    v0_out[1] = v1y
    vf_out[0] = v2x
    vf_out[1] = v2y

    Jv0_r0_out[0, 0] = (-r1x * df_dr1_x - f_) * inv_g - v1x * dg_dr1_x * inv_g
    Jv0_r0_out[0, 1] = (-r1x * df_dr1_y) * inv_g - v1x * dg_dr1_y * inv_g
    Jv0_r0_out[1, 0] = (-r1y * df_dr1_x) * inv_g - v1y * dg_dr1_x * inv_g
    Jv0_r0_out[1, 1] = (-r1y * df_dr1_y - f_) * inv_g - v1y * dg_dr1_y * inv_g

    Jv0_rf_out[0, 0] = (1.0 - r1x * df_dr2_x) * inv_g - v1x * dg_dr2_x * inv_g
    Jv0_rf_out[0, 1] = (-r1x * df_dr2_y) * inv_g - v1x * dg_dr2_y * inv_g
    Jv0_rf_out[1, 0] = (-r1y * df_dr2_x) * inv_g - v1y * dg_dr2_x * inv_g
    Jv0_rf_out[1, 1] = (1.0 - r1y * df_dr2_y) * inv_g - v1y * dg_dr2_y * inv_g

    Jv0_dt_out[0] = ((-df_dt * r1x) * inv_g - v1x * dg_dt * inv_g) * mult
    Jv0_dt_out[1] = ((-df_dt * r1y) * inv_g - v1y * dg_dt * inv_g) * mult

    Jvf_r0_out[0, 0] = (r2x * dgdot_dr1_x - 1.0) * inv_g - v2x * dg_dr1_x * inv_g
    Jvf_r0_out[0, 1] = (r2x * dgdot_dr1_y) * inv_g - v2x * dg_dr1_y * inv_g
    Jvf_r0_out[1, 0] = (r2y * dgdot_dr1_x) * inv_g - v2y * dg_dr1_x * inv_g
    Jvf_r0_out[1, 1] = (r2y * dgdot_dr1_y - 1.0) * inv_g - v2y * dg_dr1_y * inv_g

    Jvf_rf_out[0, 0] = (r2x * dgdot_dr2_x + gdot) * inv_g - v2x * dg_dr2_x * inv_g
    Jvf_rf_out[0, 1] = (r2x * dgdot_dr2_y) * inv_g - v2x * dg_dr2_y * inv_g
    Jvf_rf_out[1, 0] = (r2y * dgdot_dr2_x) * inv_g - v2y * dg_dr2_x * inv_g
    Jvf_rf_out[1, 1] = (r2y * dgdot_dr2_y + gdot) * inv_g - v2y * dg_dr2_y * inv_g

    Jvf_dt_out[0] = ((dgdot_dt * r2x) * inv_g - v2x * dg_dt * inv_g) * mult
    Jvf_dt_out[1] = ((dgdot_dt * r2y) * inv_g - v2y * dg_dt * inv_g) * mult

    return 1, z


@njit(cache=True, fastmath=False, inline="always")
def _fill_straight_line(
    r1x, r1y, r2x, r2y, tof_seg, mult,
    v0_out, vf_out,
    Jv0_r0_out, Jv0_rf_out, Jv0_dt_out,
    Jvf_r0_out, Jvf_rf_out, Jvf_dt_out,
):
    """Mirror of the Python _straight_line branch, writing into preallocated slots."""
    t = tof_seg if tof_seg > 1e-10 else 1e-10
    inv_t = 1.0 / t
    vx = (r2x - r1x) * inv_t
    vy = (r2y - r1y) * inv_t
    v0_out[0] = vx
    v0_out[1] = vy
    vf_out[0] = vx
    vf_out[1] = vy
    Jv0_r0_out[0, 0] = -inv_t
    Jv0_r0_out[0, 1] = 0.0
    Jv0_r0_out[1, 0] = 0.0
    Jv0_r0_out[1, 1] = -inv_t
    Jv0_rf_out[0, 0] = inv_t
    Jv0_rf_out[0, 1] = 0.0
    Jv0_rf_out[1, 0] = 0.0
    Jv0_rf_out[1, 1] = inv_t
    Jv0_dt_out[0] = -vx * inv_t * mult
    Jv0_dt_out[1] = -vy * inv_t * mult
    Jvf_r0_out[0, 0] = -inv_t
    Jvf_r0_out[0, 1] = 0.0
    Jvf_r0_out[1, 0] = 0.0
    Jvf_r0_out[1, 1] = -inv_t
    Jvf_rf_out[0, 0] = inv_t
    Jvf_rf_out[0, 1] = 0.0
    Jvf_rf_out[1, 0] = 0.0
    Jvf_rf_out[1, 1] = inv_t
    Jvf_dt_out[0] = -vx * inv_t * mult
    Jvf_dt_out[1] = -vy * inv_t * mult
    return 1, 0.0


@njit(cache=True, fastmath=False, inline="always")
def _apply_sundman_fixup(
    r0, rf, center, tau,
    r0_mag, rf_mag, sqrt_s, s_pow,
    Jv0_r0_out, Jv0_rf_out, Jv0_dt_out,
    Jvf_r0_out, Jvf_rf_out, Jvf_dt_out,
):
    """Apply chain-rule fixup converting per-tof_seg Jacobians (as written by
    `_solve_segment`) into per-tau Jacobians under the time scaling
        tof_seg = tau * s^1.5 * mult,   s = (|r0-c| + |rf-c|) / 2.

    Position-Jacobians gain an outer-product term from tof_seg's dependence
    on r0/rf via s. dt-Jacobians are rescaled from per-T_old to per-tau.

    NOTE: Position-extras must be applied BEFORE rescaling dt-Jacobians,
    since they use the original (per-T_old) values.

    Sign convention: stored Jv0_dt_out = (dv0/d_tof_seg) * mult = dv0/dT_old.
    Position chain-rule contribution:
        dv/dr0[a, b] += 0.75 * tau * sqrt_s * Jv0_dt_out[a] * r0_hat[b]
    where r0_hat = (r0 - center) / |r0 - center|.
    """
    fac = 0.75 * tau * sqrt_s
    if r0_mag > 1e-10:
        inv_r0 = 1.0 / r0_mag
        r0_hat_x = (r0[0] - center[0]) * inv_r0
        r0_hat_y = (r0[1] - center[1]) * inv_r0
        Jv0_r0_out[0, 0] += fac * Jv0_dt_out[0] * r0_hat_x
        Jv0_r0_out[0, 1] += fac * Jv0_dt_out[0] * r0_hat_y
        Jv0_r0_out[1, 0] += fac * Jv0_dt_out[1] * r0_hat_x
        Jv0_r0_out[1, 1] += fac * Jv0_dt_out[1] * r0_hat_y
        Jvf_r0_out[0, 0] += fac * Jvf_dt_out[0] * r0_hat_x
        Jvf_r0_out[0, 1] += fac * Jvf_dt_out[0] * r0_hat_y
        Jvf_r0_out[1, 0] += fac * Jvf_dt_out[1] * r0_hat_x
        Jvf_r0_out[1, 1] += fac * Jvf_dt_out[1] * r0_hat_y
    if rf_mag > 1e-10:
        inv_rf = 1.0 / rf_mag
        rf_hat_x = (rf[0] - center[0]) * inv_rf
        rf_hat_y = (rf[1] - center[1]) * inv_rf
        Jv0_rf_out[0, 0] += fac * Jv0_dt_out[0] * rf_hat_x
        Jv0_rf_out[0, 1] += fac * Jv0_dt_out[0] * rf_hat_y
        Jv0_rf_out[1, 0] += fac * Jv0_dt_out[1] * rf_hat_x
        Jv0_rf_out[1, 1] += fac * Jv0_dt_out[1] * rf_hat_y
        Jvf_rf_out[0, 0] += fac * Jvf_dt_out[0] * rf_hat_x
        Jvf_rf_out[0, 1] += fac * Jvf_dt_out[0] * rf_hat_y
        Jvf_rf_out[1, 0] += fac * Jvf_dt_out[1] * rf_hat_x
        Jvf_rf_out[1, 1] += fac * Jvf_dt_out[1] * rf_hat_y
    Jv0_dt_out[0] *= s_pow
    Jv0_dt_out[1] *= s_pow
    Jvf_dt_out[0] *= s_pow
    Jvf_dt_out[1] *= s_pow


@njit(cache=True, fastmath=False)
def fun_and_grad_batch(
    x_vec, n_pos_vars,
    # segment metadata (all length N)
    seg_start_kind, seg_end_kind,
    seg_start_var_idx, seg_end_var_idx,
    seg_start_fixed, seg_end_fixed,
    seg_mult, seg_side,
    # shape + env
    shape_pos_tri, shape_pos_sq,
    center_np, mu,
    use_parabola, cg_mode, const_g_vec, g_mag,
    # dv-node metadata (length M)
    dv_type, dv_in_seg, dv_out_seg,
    dv_center_var_idx, dv_shape_vel,
    # cost mode
    energy_mode, fuel_eps,
    # work buffers (length N; preallocated by caller)
    v0_buf, vf_buf,
    Jv0_r0_buf, Jv0_rf_buf, Jv0_dt_buf,
    Jvf_r0_buf, Jvf_rf_buf, Jvf_dt_buf,
    z_cache,
    # scratch grad buffer (length n_vars; preallocated)
    grad_out,
    ok_out,
):
    """Single-shot fun + grad. Returns cost (float). Writes grad into grad_out
    and per-segment ok flags into ok_out."""
    tof = x_vec[n_pos_vars]
    N = seg_mult.shape[0]

    # Zero the grad buffer.
    for k in range(grad_out.shape[0]):
        grad_out[k] = 0.0

    # --- Stage 1: per-segment velocities + Jacobians.
    for i in range(N):
        mult = seg_mult[i]

        # Resolve r0 and rf.
        ks = seg_start_kind[i]
        if ks == 0:
            idx = seg_start_var_idx[i]
            r0x = x_vec[2 * idx]
            r0y = x_vec[2 * idx + 1]
        elif ks == 1:
            r0x = shape_pos_tri[0]
            r0y = shape_pos_tri[1]
        elif ks == 2:
            r0x = shape_pos_sq[0]
            r0y = shape_pos_sq[1]
        else:
            r0x = seg_start_fixed[i, 0]
            r0y = seg_start_fixed[i, 1]

        ke = seg_end_kind[i]
        if ke == 0:
            idx = seg_end_var_idx[i]
            rfx = x_vec[2 * idx]
            rfy = x_vec[2 * idx + 1]
        elif ke == 1:
            rfx = shape_pos_tri[0]
            rfy = shape_pos_tri[1]
        elif ke == 2:
            rfx = shape_pos_sq[0]
            rfy = shape_pos_sq[1]
        else:
            rfx = seg_end_fixed[i, 0]
            rfy = seg_end_fixed[i, 1]

        r0 = np.empty(2)
        r0[0] = r0x
        r0[1] = r0y
        rf = np.empty(2)
        rf[0] = rfx
        rf[1] = rfy

        # Sundman-style time scaling: tof_seg = tau * s^1.5 * mult,
        # s = (|r0-c| + |rf-c|) / 2 (distance from earth_center).
        # Disabled in CG mode (uniform g) where there's no orbital scale.
        if cg_mode:
            r0_mag = 0.0
            rf_mag = 0.0
            sqrt_s = 1.0
            s_pow = 1.0
            tof_seg = tof * mult
        else:
            dx0 = r0x - center_np[0]
            dy0 = r0y - center_np[1]
            dxf = rfx - center_np[0]
            dyf = rfy - center_np[1]
            r0_mag = math.sqrt(dx0 * dx0 + dy0 * dy0)
            rf_mag = math.sqrt(dxf * dxf + dyf * dyf)
            s = 0.5 * (r0_mag + rf_mag)
            if s < 1e-12:
                s = 1e-12
            sqrt_s = math.sqrt(s)
            s_pow = s * sqrt_s
            tof_seg = tof * s_pow * mult

        ok, z_new = _solve_segment(
            r0, rf, tof_seg, mult, z_cache[i], seg_side[i],
            center_np, mu, use_parabola, cg_mode, const_g_vec, g_mag,
            v0_buf[i], vf_buf[i],
            Jv0_r0_buf[i], Jv0_rf_buf[i], Jv0_dt_buf[i],
            Jvf_r0_buf[i], Jvf_rf_buf[i], Jvf_dt_buf[i],
        )
        ok_out[i] = ok
        if ok == 1 and not use_parabola:
            z_cache[i] = z_new
        if ok == 1 and not cg_mode:
            _apply_sundman_fixup(
                r0, rf, center_np, tof,
                r0_mag, rf_mag, sqrt_s, s_pow,
                Jv0_r0_buf[i], Jv0_rf_buf[i], Jv0_dt_buf[i],
                Jvf_r0_buf[i], Jvf_rf_buf[i], Jvf_dt_buf[i],
            )

    # If any segment failed, caller will fall back. Return early with cost=0
    # to avoid propagating garbage; grad will be recomputed in Python.
    any_fail = 0
    for i in range(N):
        if ok_out[i] == 0:
            any_fail = 1
            break
    if any_fail == 1:
        return 0.0

    # --- Stage 2: dv-node accumulation.
    cost = 0.0
    M = dv_type.shape[0]
    for k in range(M):
        t = dv_type[k]
        if t == 0:
            # Movable midpoint: dv = v0[out] - vf[in]
            i_in = dv_in_seg[k]
            i_out = dv_out_seg[k]
            dvx = v0_buf[i_out, 0] - vf_buf[i_in, 0]
            dvy = v0_buf[i_out, 1] - vf_buf[i_in, 1]
        elif t == 1:
            # Shape outgoing: dv = v0[out] - shape_vel
            i_out = dv_out_seg[k]
            i_in = -1
            dvx = v0_buf[i_out, 0] - dv_shape_vel[k, 0]
            dvy = v0_buf[i_out, 1] - dv_shape_vel[k, 1]
        else:
            # Shape incoming: dv = shape_vel - vf[in]
            i_in = dv_in_seg[k]
            i_out = -1
            dvx = dv_shape_vel[k, 0] - vf_buf[i_in, 0]
            dvy = dv_shape_vel[k, 1] - vf_buf[i_in, 1]

        d2 = dvx * dvx + dvy * dvy
        if energy_mode:
            cost += d2
            factor = 2.0
        else:
            mag = math.sqrt(d2 + fuel_eps * fuel_eps)
            cost += mag
            if mag < 1e-30:
                factor = 0.0
            else:
                factor = 1.0 / mag

        if t == 0:
            # Contributions:
            #   upstream start (seg i_in's start): Jacobian = -Jvf_r0[i_in]
            #   center (midpoint): Jacobian = Jv0_r0[i_out] - Jvf_rf[i_in]
            #   downstream end (seg i_out's end): Jacobian = Jv0_rf[i_out]
            #   tof: Jv0_dt[i_out] - Jvf_dt[i_in]
            # Upstream start.
            us_kind = seg_start_kind[i_in]
            if us_kind == 0:
                us_idx = seg_start_var_idx[i_in]
                # grad += factor * dv @ (-Jvf_r0[i_in])
                J00 = Jvf_r0_buf[i_in, 0, 0]
                J01 = Jvf_r0_buf[i_in, 0, 1]
                J10 = Jvf_r0_buf[i_in, 1, 0]
                J11 = Jvf_r0_buf[i_in, 1, 1]
                grad_out[2 * us_idx]     += factor * (-(dvx * J00 + dvy * J10))
                grad_out[2 * us_idx + 1] += factor * (-(dvx * J01 + dvy * J11))
            # Center.
            c_idx = dv_center_var_idx[k]
            if c_idx >= 0:
                # J = Jv0_r0[i_out] - Jvf_rf[i_in]
                J00 = Jv0_r0_buf[i_out, 0, 0] - Jvf_rf_buf[i_in, 0, 0]
                J01 = Jv0_r0_buf[i_out, 0, 1] - Jvf_rf_buf[i_in, 0, 1]
                J10 = Jv0_r0_buf[i_out, 1, 0] - Jvf_rf_buf[i_in, 1, 0]
                J11 = Jv0_r0_buf[i_out, 1, 1] - Jvf_rf_buf[i_in, 1, 1]
                grad_out[2 * c_idx]     += factor * (dvx * J00 + dvy * J10)
                grad_out[2 * c_idx + 1] += factor * (dvx * J01 + dvy * J11)
            # Downstream end.
            ds_kind = seg_end_kind[i_out]
            if ds_kind == 0:
                ds_idx = seg_end_var_idx[i_out]
                J00 = Jv0_rf_buf[i_out, 0, 0]
                J01 = Jv0_rf_buf[i_out, 0, 1]
                J10 = Jv0_rf_buf[i_out, 1, 0]
                J11 = Jv0_rf_buf[i_out, 1, 1]
                grad_out[2 * ds_idx]     += factor * (dvx * J00 + dvy * J10)
                grad_out[2 * ds_idx + 1] += factor * (dvx * J01 + dvy * J11)
            # tof.
            grad_out[n_pos_vars] += factor * (
                dvx * (Jv0_dt_buf[i_out, 0] - Jvf_dt_buf[i_in, 0])
                + dvy * (Jv0_dt_buf[i_out, 1] - Jvf_dt_buf[i_in, 1])
            )
        elif t == 1:
            # Shape outgoing. Contrib: downstream end + tof.
            ds_kind = seg_end_kind[i_out]
            if ds_kind == 0:
                ds_idx = seg_end_var_idx[i_out]
                J00 = Jv0_rf_buf[i_out, 0, 0]
                J01 = Jv0_rf_buf[i_out, 0, 1]
                J10 = Jv0_rf_buf[i_out, 1, 0]
                J11 = Jv0_rf_buf[i_out, 1, 1]
                grad_out[2 * ds_idx]     += factor * (dvx * J00 + dvy * J10)
                grad_out[2 * ds_idx + 1] += factor * (dvx * J01 + dvy * J11)
            grad_out[n_pos_vars] += factor * (
                dvx * Jv0_dt_buf[i_out, 0] + dvy * Jv0_dt_buf[i_out, 1]
            )
        else:
            # Shape incoming. Contrib: upstream start + tof (both negated).
            us_kind = seg_start_kind[i_in]
            if us_kind == 0:
                us_idx = seg_start_var_idx[i_in]
                J00 = Jvf_r0_buf[i_in, 0, 0]
                J01 = Jvf_r0_buf[i_in, 0, 1]
                J10 = Jvf_r0_buf[i_in, 1, 0]
                J11 = Jvf_r0_buf[i_in, 1, 1]
                grad_out[2 * us_idx]     += factor * (-(dvx * J00 + dvy * J10))
                grad_out[2 * us_idx + 1] += factor * (-(dvx * J01 + dvy * J11))
            grad_out[n_pos_vars] += factor * (
                -(dvx * Jvf_dt_buf[i_in, 0] + dvy * Jvf_dt_buf[i_in, 1])
            )

    return cost


@njit(cache=True, fastmath=False)
def lm_eval_batch(
    x_vec, n_pos_vars,
    # segment metadata (length N)
    seg_start_kind, seg_end_kind,
    seg_start_var_idx, seg_end_var_idx,
    seg_start_fixed, seg_end_fixed,
    seg_mult, seg_side,
    # shape + env
    shape_pos_tri, shape_pos_sq,
    center_np, mu,
    use_parabola, cg_mode, const_g_vec, g_mag,
    # dv-node metadata (length M)
    dv_type, dv_in_seg, dv_out_seg,
    dv_center_var_idx, dv_shape_vel,
    # cost mode
    energy_mode, fuel_eps,
    # work buffers (length N; preallocated)
    v0_buf, vf_buf,
    Jv0_r0_buf, Jv0_rf_buf, Jv0_dt_buf,
    Jvf_r0_buf, Jvf_rf_buf, Jvf_dt_buf,
    z_cache,
    ok_out,
    # LM outputs (preallocated)
    r_out,   # (2M,) — residual dv vector stacked per-node
    J_out,   # (2M, n_vars) — Jacobian of r wrt x; zero-filled inside
):
    """Levenberg-Marquardt / Gauss-Newton evaluator. Returns cost (scalar),
    writes residual vector r_out and Jacobian J_out. Does NOT apply fuel
    weights — caller does that. Writes ok_out per segment; on any failure
    caller should reject the step."""
    tof = x_vec[n_pos_vars]
    N = seg_mult.shape[0]
    M = dv_type.shape[0]
    n_vars = n_pos_vars + 1

    # --- Stage 1: per-segment velocity + Jacobian solve (same as fun_and_grad_batch).
    for i in range(N):
        mult = seg_mult[i]

        ks = seg_start_kind[i]
        if ks == 0:
            idx = seg_start_var_idx[i]
            r0x = x_vec[2 * idx]
            r0y = x_vec[2 * idx + 1]
        elif ks == 1:
            r0x = shape_pos_tri[0]
            r0y = shape_pos_tri[1]
        elif ks == 2:
            r0x = shape_pos_sq[0]
            r0y = shape_pos_sq[1]
        else:
            r0x = seg_start_fixed[i, 0]
            r0y = seg_start_fixed[i, 1]

        ke = seg_end_kind[i]
        if ke == 0:
            idx = seg_end_var_idx[i]
            rfx = x_vec[2 * idx]
            rfy = x_vec[2 * idx + 1]
        elif ke == 1:
            rfx = shape_pos_tri[0]
            rfy = shape_pos_tri[1]
        elif ke == 2:
            rfx = shape_pos_sq[0]
            rfy = shape_pos_sq[1]
        else:
            rfx = seg_end_fixed[i, 0]
            rfy = seg_end_fixed[i, 1]

        r0 = np.empty(2)
        r0[0] = r0x
        r0[1] = r0y
        rf = np.empty(2)
        rf[0] = rfx
        rf[1] = rfy

        # Sundman-style time scaling. Disabled in CG mode.
        if cg_mode:
            r0_mag = 0.0
            rf_mag = 0.0
            sqrt_s = 1.0
            s_pow = 1.0
            tof_seg = tof * mult
        else:
            dx0 = r0x - center_np[0]
            dy0 = r0y - center_np[1]
            dxf = rfx - center_np[0]
            dyf = rfy - center_np[1]
            r0_mag = math.sqrt(dx0 * dx0 + dy0 * dy0)
            rf_mag = math.sqrt(dxf * dxf + dyf * dyf)
            s = 0.5 * (r0_mag + rf_mag)
            if s < 1e-12:
                s = 1e-12
            sqrt_s = math.sqrt(s)
            s_pow = s * sqrt_s
            tof_seg = tof * s_pow * mult

        ok, z_new = _solve_segment(
            r0, rf, tof_seg, mult, z_cache[i], seg_side[i],
            center_np, mu, use_parabola, cg_mode, const_g_vec, g_mag,
            v0_buf[i], vf_buf[i],
            Jv0_r0_buf[i], Jv0_rf_buf[i], Jv0_dt_buf[i],
            Jvf_r0_buf[i], Jvf_rf_buf[i], Jvf_dt_buf[i],
        )
        ok_out[i] = ok
        if ok == 1 and not use_parabola:
            z_cache[i] = z_new
        if ok == 1 and not cg_mode:
            _apply_sundman_fixup(
                r0, rf, center_np, tof,
                r0_mag, rf_mag, sqrt_s, s_pow,
                Jv0_r0_buf[i], Jv0_rf_buf[i], Jv0_dt_buf[i],
                Jvf_r0_buf[i], Jvf_rf_buf[i], Jvf_dt_buf[i],
            )

    # Zero-fill outputs.
    for i in range(2 * M):
        r_out[i] = 0.0
        for j in range(n_vars):
            J_out[i, j] = 0.0

    # If any segment failed, bail with cost=huge; caller rejects the step.
    any_fail = 0
    for i in range(N):
        if ok_out[i] == 0:
            any_fail = 1
            break
    if any_fail == 1:
        return 1e30

    # --- Stage 2: per-dv-node residual + Jacobian rows.
    cost = 0.0
    for k in range(M):
        t = dv_type[k]
        if t == 0:
            i_in = dv_in_seg[k]
            i_out = dv_out_seg[k]
            dvx = v0_buf[i_out, 0] - vf_buf[i_in, 0]
            dvy = v0_buf[i_out, 1] - vf_buf[i_in, 1]
        elif t == 1:
            i_out = dv_out_seg[k]
            i_in = -1
            dvx = v0_buf[i_out, 0] - dv_shape_vel[k, 0]
            dvy = v0_buf[i_out, 1] - dv_shape_vel[k, 1]
        else:
            i_in = dv_in_seg[k]
            i_out = -1
            dvx = dv_shape_vel[k, 0] - vf_buf[i_in, 0]
            dvy = dv_shape_vel[k, 1] - vf_buf[i_in, 1]

        rx = 2 * k
        ry = 2 * k + 1
        r_out[rx] = dvx
        r_out[ry] = dvy

        d2 = dvx * dvx + dvy * dvy
        if energy_mode:
            cost += d2
        else:
            cost += math.sqrt(d2 + fuel_eps * fuel_eps)

        # Write J rows. Use += so that if a node's upstream_start happens to
        # coincide with its downstream_end (tiny loops), contributions add.
        if t == 0:
            # dv = v0[i_out] - vf[i_in]
            # upstream_start: J = -Jvf_r0[i_in]
            us_kind = seg_start_kind[i_in]
            if us_kind == 0:
                us_idx = seg_start_var_idx[i_in]
                J_out[rx, 2 * us_idx]     += -Jvf_r0_buf[i_in, 0, 0]
                J_out[rx, 2 * us_idx + 1] += -Jvf_r0_buf[i_in, 0, 1]
                J_out[ry, 2 * us_idx]     += -Jvf_r0_buf[i_in, 1, 0]
                J_out[ry, 2 * us_idx + 1] += -Jvf_r0_buf[i_in, 1, 1]
            # center: J = Jv0_r0[i_out] - Jvf_rf[i_in]
            c_idx = dv_center_var_idx[k]
            if c_idx >= 0:
                J_out[rx, 2 * c_idx]     += Jv0_r0_buf[i_out, 0, 0] - Jvf_rf_buf[i_in, 0, 0]
                J_out[rx, 2 * c_idx + 1] += Jv0_r0_buf[i_out, 0, 1] - Jvf_rf_buf[i_in, 0, 1]
                J_out[ry, 2 * c_idx]     += Jv0_r0_buf[i_out, 1, 0] - Jvf_rf_buf[i_in, 1, 0]
                J_out[ry, 2 * c_idx + 1] += Jv0_r0_buf[i_out, 1, 1] - Jvf_rf_buf[i_in, 1, 1]
            # downstream_end: J = Jv0_rf[i_out]
            ds_kind = seg_end_kind[i_out]
            if ds_kind == 0:
                ds_idx = seg_end_var_idx[i_out]
                J_out[rx, 2 * ds_idx]     += Jv0_rf_buf[i_out, 0, 0]
                J_out[rx, 2 * ds_idx + 1] += Jv0_rf_buf[i_out, 0, 1]
                J_out[ry, 2 * ds_idx]     += Jv0_rf_buf[i_out, 1, 0]
                J_out[ry, 2 * ds_idx + 1] += Jv0_rf_buf[i_out, 1, 1]
            # tof
            J_out[rx, n_pos_vars] += Jv0_dt_buf[i_out, 0] - Jvf_dt_buf[i_in, 0]
            J_out[ry, n_pos_vars] += Jv0_dt_buf[i_out, 1] - Jvf_dt_buf[i_in, 1]
        elif t == 1:
            # dv = v0[i_out] - shape_vel; shape r0 is fixed, so only downstream_end + tof.
            ds_kind = seg_end_kind[i_out]
            if ds_kind == 0:
                ds_idx = seg_end_var_idx[i_out]
                J_out[rx, 2 * ds_idx]     += Jv0_rf_buf[i_out, 0, 0]
                J_out[rx, 2 * ds_idx + 1] += Jv0_rf_buf[i_out, 0, 1]
                J_out[ry, 2 * ds_idx]     += Jv0_rf_buf[i_out, 1, 0]
                J_out[ry, 2 * ds_idx + 1] += Jv0_rf_buf[i_out, 1, 1]
            J_out[rx, n_pos_vars] += Jv0_dt_buf[i_out, 0]
            J_out[ry, n_pos_vars] += Jv0_dt_buf[i_out, 1]
        else:
            # dv = shape_vel - vf[i_in]; shape rf is fixed, so only upstream_start + tof (negated).
            us_kind = seg_start_kind[i_in]
            if us_kind == 0:
                us_idx = seg_start_var_idx[i_in]
                J_out[rx, 2 * us_idx]     += -Jvf_r0_buf[i_in, 0, 0]
                J_out[rx, 2 * us_idx + 1] += -Jvf_r0_buf[i_in, 0, 1]
                J_out[ry, 2 * us_idx]     += -Jvf_r0_buf[i_in, 1, 0]
                J_out[ry, 2 * us_idx + 1] += -Jvf_r0_buf[i_in, 1, 1]
            J_out[rx, n_pos_vars] += -Jvf_dt_buf[i_in, 0]
            J_out[ry, n_pos_vars] += -Jvf_dt_buf[i_in, 1]

    return cost


def _warmup():
    """Force JIT compilation on a known-good geometry."""
    r1 = np.array([1.0, 0.0])
    r2 = np.array([0.3, 0.9])
    stumpff_all(0.0)
    lambert_z_newton(1.0, 1.0, 0.9, 1.5, 1.0, 0.0)
    lambert_solve_nb(r1, r2, 1.5, 1.0, 0.0)
    lambert_with_jac_nb(r1, r2, 1.5, 1.0, 0.0, 0.0)

    # Warm up batch path: trivial 1-segment two_body/conic case.
    x_vec = np.array([1.0, 0.0, 0.0, 1.0, 1.5])  # two movable dots + tof
    seg_start_kind = np.array([0], dtype=np.int64)
    seg_end_kind = np.array([0], dtype=np.int64)
    seg_start_var_idx = np.array([0], dtype=np.int64)
    seg_end_var_idx = np.array([1], dtype=np.int64)
    seg_start_fixed = np.zeros((1, 2))
    seg_end_fixed = np.zeros((1, 2))
    seg_mult = np.array([1.0])
    seg_side = np.array([0.0])
    shape_pos_tri = np.zeros(2)
    shape_pos_sq = np.zeros(2)
    center = np.zeros(2)
    const_g = np.array([0.0, -1.0])
    dv_type = np.zeros(0, dtype=np.int64)
    dv_in_seg = np.zeros(0, dtype=np.int64)
    dv_out_seg = np.zeros(0, dtype=np.int64)
    dv_center_var_idx = np.zeros(0, dtype=np.int64)
    dv_shape_vel = np.zeros((0, 2))
    v0_buf = np.zeros((1, 2))
    vf_buf = np.zeros((1, 2))
    Jv0_r0_buf = np.zeros((1, 2, 2))
    Jv0_rf_buf = np.zeros((1, 2, 2))
    Jv0_dt_buf = np.zeros((1, 2))
    Jvf_r0_buf = np.zeros((1, 2, 2))
    Jvf_rf_buf = np.zeros((1, 2, 2))
    Jvf_dt_buf = np.zeros((1, 2))
    z_cache = np.zeros(1)
    grad_out = np.zeros(5)
    ok_out = np.zeros(1, dtype=np.int64)
    r_out = np.zeros(0)
    J_out = np.zeros((0, 5))

    # Warm two_body + conic:
    fun_and_grad_batch(
        x_vec, 4,
        seg_start_kind, seg_end_kind,
        seg_start_var_idx, seg_end_var_idx,
        seg_start_fixed, seg_end_fixed,
        seg_mult, seg_side,
        shape_pos_tri, shape_pos_sq,
        center, 1.0,
        False, False, const_g, 1.0,
        dv_type, dv_in_seg, dv_out_seg,
        dv_center_var_idx, dv_shape_vel,
        True, 1e-4,
        v0_buf, vf_buf,
        Jv0_r0_buf, Jv0_rf_buf, Jv0_dt_buf,
        Jvf_r0_buf, Jvf_rf_buf, Jvf_dt_buf,
        z_cache, grad_out, ok_out,
    )
    # Warm parabola branch:
    fun_and_grad_batch(
        x_vec, 4,
        seg_start_kind, seg_end_kind,
        seg_start_var_idx, seg_end_var_idx,
        seg_start_fixed, seg_end_fixed,
        seg_mult, seg_side,
        shape_pos_tri, shape_pos_sq,
        center, 1.0,
        True, True, const_g, 1.0,
        dv_type, dv_in_seg, dv_out_seg,
        dv_center_var_idx, dv_shape_vel,
        False, 1e-4,
        v0_buf, vf_buf,
        Jv0_r0_buf, Jv0_rf_buf, Jv0_dt_buf,
        Jvf_r0_buf, Jvf_rf_buf, Jvf_dt_buf,
        z_cache, grad_out, ok_out,
    )
    # Warm LM evaluator. M=0 dv-nodes is fine; the per-segment solve still runs.
    lm_eval_batch(
        x_vec, 4,
        seg_start_kind, seg_end_kind,
        seg_start_var_idx, seg_end_var_idx,
        seg_start_fixed, seg_end_fixed,
        seg_mult, seg_side,
        shape_pos_tri, shape_pos_sq,
        center, 1.0,
        False, False, const_g, 1.0,
        dv_type, dv_in_seg, dv_out_seg,
        dv_center_var_idx, dv_shape_vel,
        True, 1e-4,
        v0_buf, vf_buf,
        Jv0_r0_buf, Jv0_rf_buf, Jv0_dt_buf,
        Jvf_r0_buf, Jvf_rf_buf, Jvf_dt_buf,
        z_cache, ok_out, r_out, J_out,
    )
    lm_eval_batch(
        x_vec, 4,
        seg_start_kind, seg_end_kind,
        seg_start_var_idx, seg_end_var_idx,
        seg_start_fixed, seg_end_fixed,
        seg_mult, seg_side,
        shape_pos_tri, shape_pos_sq,
        center, 1.0,
        True, True, const_g, 1.0,
        dv_type, dv_in_seg, dv_out_seg,
        dv_center_var_idx, dv_shape_vel,
        False, 1e-4,
        v0_buf, vf_buf,
        Jv0_r0_buf, Jv0_rf_buf, Jv0_dt_buf,
        Jvf_r0_buf, Jvf_rf_buf, Jvf_dt_buf,
        z_cache, ok_out, r_out, J_out,
    )


_warmup()