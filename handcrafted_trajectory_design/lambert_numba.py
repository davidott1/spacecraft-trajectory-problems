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
def lambert_solve_nb(r1_vec, r2_vec, dt, mu):
    """Lambert solver. Returns (ok, v1, v2). ok=0 -> caller should fall back."""
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
    if abs(sin_dtheta) < 1e-14 or abs(1.0 - cos_dtheta) < 1e-14:
        return 0, v_fail, v_fail

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
def lambert_with_jac_nb(r1_vec, r2_vec, dt, mu, z_init):
    """Lambert + analytic Jacobians.

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
    if abs(sin_dtheta) < 1e-14 or abs(1.0 - cos_dtheta) < 1e-14:
        return 0, zero_v, zero_v, zero_m, zero_m, zero_v, zero_m, zero_m, zero_v, 0.0

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


def _warmup():
    """Force JIT compilation on a known-good geometry."""
    r1 = np.array([1.0, 0.0])
    r2 = np.array([0.3, 0.9])
    stumpff_all(0.0)
    lambert_z_newton(1.0, 1.0, 0.9, 1.5, 1.0, 0.0)
    lambert_solve_nb(r1, r2, 1.5, 1.0)
    lambert_with_jac_nb(r1, r2, 1.5, 1.0, 0.0)


_warmup()
