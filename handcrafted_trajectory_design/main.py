import sys

import math
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import brentq, minimize as scipy_minimize

from PyQt6.QtCore import Qt, QPointF, QRectF, QEvent, QObject, pyqtSignal
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QLabel


class _OptSignals(QObject):
    """Helper QObject to marshal optimizer results from a worker thread back
    to the Qt main thread. Cross-thread signal emission uses a queued
    connection automatically, so the connected slot runs on the main thread."""

    done = pyqtSignal(object)
    progress = pyqtSignal(object)

# Canonical units: 1 DU = 6371 km, 1 TU = sqrt(DU^3 / mu_km) ~ 806.4 s, mu = 1 DU^3/TU^2.
GRAVITY_MAG = 1.0  # DU/TU^2 (~ 9.81e-3 km/s^2 in canonical units)
TIME_OF_FLIGHT = 1.5  # TU (~ 1200 s)
# Initial guess for the shared time decision variable, by env mode.
# In two-body env this is `tau` (Sundman-scaled); in constant-gravity env
# this is physical TOF per unit mult.
TAU_INIT_TWO_BODY = 30.0 * math.pi / 180.0
TOF_INIT_CONSTANT_G = 0.5
# Slider ranges for the time-variable slider, by env mode.
TAU_SLIDER_MAX = math.pi  # two-body: 0..180 deg
TOF_SLIDER_MAX = 2.0      # constant-gravity: 0..2 TU
ARC_NUM_POINTS = 50  # number of points to draw the parabolic arc
ORBIT_NUM_POINTS = 200  # number of points to draw a Kepler orbit
MU = 1.0  # canonical gravitational parameter
# Half-width (radians) of the antipodal-degenerate band where Lambert needs a
# `side` hint to disambiguate short-way vs long-way at dtheta ~ pi.
_SIDE_EPS = 1.0e-3
VEL_SCALE = 2.0  # display scale: velocity_end = center + vel * VEL_SCALE
# Display scales for dv arrows (DU on screen per DU/TU of dv). Picked larger
# than VEL_SCALE since |dv| << |v| in typical impulsive transfers, and
# tuned per optimizer mode since fuel-optimal solutions concentrate dv into
# fewer (larger) burns while energy-optimal spreads it more evenly.
DV_SCALE_DEFAULTS = {"energy": 30.0, "fuel": 10.0, None: 10.0}
DV_SCALE_MAX = 30.0 * VEL_SCALE  # slider upper bound (== 60.0)
# Hard-coded "small" delta-v threshold for the prune-small-dvs button (DU/TU).
SMALL_DV_THRESHOLD = 0.1
# Rotating frame: 1 revolution per 24 hours. With 1 TU ~ 806.4 s, 24 h ~ 107.14 TU.
ROT_PERIOD_TU = 24.0 * 3600.0 / 806.4
OMEGA_ROT = 2.0 * math.pi / ROT_PERIOD_TU  # rad / TU, +z (CCW in y-up world)


def propagate_kepler_period(pos_center, vel_vec, grav_center, mu, num_points):
    """Sample positions equally in time over one orbital period for an
    elliptic Kepler orbit. Returns (times, points) with num_points samples
    spanning t in [0, T]. Returns ([], []) if non-elliptic / degenerate."""
    r0 = np.array([pos_center.x() - grav_center.x(), pos_center.y() - grav_center.y()])
    v0 = np.asarray(vel_vec, dtype=float)
    r = np.linalg.norm(r0)
    v = np.linalg.norm(v0)
    if r < 1e-10 or v < 1e-10:
        return [], []
    energy = 0.5 * v * v - mu / r
    if energy >= 0:
        return [], []
    a = -mu / (2.0 * energy)
    h = r0[0] * v0[1] - r0[1] * v0[0]
    e_vec = (1.0 / mu) * ((v * v - mu / r) * r0 - np.dot(r0, v0) * v0)
    e = float(np.linalg.norm(e_vec))
    if e >= 1.0:
        return [], []
    p = a * (1.0 - e * e)
    n = math.sqrt(mu / a ** 3)
    period = 2.0 * math.pi / n
    if e > 1e-12:
        cos_th0 = float(np.dot(e_vec, r0) / (e * r))
        cos_th0 = max(-1.0, min(1.0, cos_th0))
        theta0 = math.acos(cos_th0)
        if np.dot(r0, v0) < 0:
            theta0 = -theta0
        e_hat = e_vec / e
    else:
        theta0 = 0.0
        e_hat = r0 / r
    p_hat = np.array([-e_hat[1], e_hat[0]])
    if h < 0:
        p_hat = -p_hat
    E0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(theta0 / 2.0),
                          math.sqrt(1.0 + e) * math.cos(theta0 / 2.0))
    M0 = E0 - e * math.sin(E0)

    cx, cy = grav_center.x(), grav_center.y()
    times, points = [], []
    for k in range(num_points):
        t = period * k / (num_points - 1)
        M = M0 + n * t
        E = M if e < 0.8 else math.pi
        for _ in range(50):
            f = E - e * math.sin(E) - M
            fp = 1.0 - e * math.cos(E)
            dE = -f / fp
            E += dE
            if abs(dE) < 1e-12:
                break
        theta = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                                 math.sqrt(1.0 - e) * math.cos(E / 2.0))
        rk = p / (1.0 + e * math.cos(theta))
        pos = rk * (math.cos(theta) * e_hat + math.sin(theta) * p_hat)
        times.append(t)
        points.append(QPointF(cx + pos[0], cy + pos[1]))
    return times, points


def compute_kepler_orbit(pos_center, vel_vec, grav_center, mu, num_points):
    """Compute a full 2D Kepler orbit by propagating true anomaly.
    pos_center: QPointF position of the node
    vel_vec: np.array([vx, vy]) velocity vector (pixels/s, but treated as arbitrary units)
    grav_center: QPointF center of gravity
    mu: gravitational parameter (pixels^3/s^2-ish)
    Returns list of QPointF or empty list if orbit is invalid."""
    r_vec = np.array([pos_center.x() - grav_center.x(), pos_center.y() - grav_center.y()])
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(vel_vec)
    if r < 1e-10 or v < 1e-10:
        return []

    # Specific angular momentum (scalar in 2D, z-component of cross product)
    h = r_vec[0] * vel_vec[1] - r_vec[1] * vel_vec[0]
    if abs(h) < 1e-10:
        return []  # degenerate (radial trajectory)

    # Specific energy
    energy = 0.5 * v * v - mu / r

    # Semi-latus rectum
    p = h * h / mu

    # Eccentricity vector
    e_vec = (1.0 / mu) * ((v * v - mu / r) * r_vec - np.dot(r_vec, vel_vec) * vel_vec)
    e = np.linalg.norm(e_vec)

    if e >= 1.0:
        # Hyperbolic or parabolic — just trace a partial arc
        theta_max = math.acos(max(-1.0, min(1.0, -1.0 / e))) - 0.01 if e > 1.0 else math.pi * 0.99
        thetas = np.linspace(-theta_max, theta_max, num_points)
    else:
        thetas = np.linspace(0, 2 * math.pi, num_points)

    # Perifocal frame: e_hat along eccentricity vector
    if e > 1e-10:
        e_hat = e_vec / e
    else:
        e_hat = r_vec / r
    # p_hat perpendicular (90 deg CCW)
    p_hat = np.array([-e_hat[1], e_hat[0]])
    # Make sure angular momentum sign is consistent
    if h < 0:
        p_hat = -p_hat

    cx, cy = grav_center.x(), grav_center.y()
    points = []
    for theta in thetas:
        denom = 1.0 + e * math.cos(theta)
        if abs(denom) < 1e-10:
            continue
        r_theta = p / denom
        x = cx + r_theta * (math.cos(theta) * e_hat[0] + math.sin(theta) * p_hat[0])
        y = cy + r_theta * (math.cos(theta) * e_hat[1] + math.sin(theta) * p_hat[1])
        points.append(QPointF(x, y))
    return points


def compute_parabolic_arc(p0, p1, center, time_of_flight, num_points, g_vec=None):
    """Compute parabolic arc points between p0 and p1 under constant gravity.
    If g_vec is None, gravity is GRAVITY_MAG directed from p0 toward center;
    otherwise the supplied (2,) vector is used directly."""
    r0 = np.array([p0.x(), p0.y()])
    rf = np.array([p1.x(), p1.y()])
    if g_vec is None:
        c = np.array([center.x(), center.y()])
        direction = c - r0
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            return [p0, p1]
        g_vec = GRAVITY_MAG * (direction / dist)
    else:
        g_vec = np.asarray(g_vec, dtype=float)
    tf = time_of_flight
    v0 = (rf - r0) / tf - 0.5 * g_vec * tf
    points = []
    for t in np.linspace(0, tf, num_points):
        pos = r0 + v0 * t + 0.5 * g_vec * t * t
        points.append(QPointF(pos[0], pos[1]))
    return points


def compute_parabolic_arc_velocities(p0, p1, center, time_of_flight, g_vec=None):
    """Return (v0, vf) numpy arrays for the parabolic arc from p0 to p1.
    If g_vec is None, gravity is GRAVITY_MAG directed from p0 toward center."""
    r0 = np.array([p0.x(), p0.y()])
    rf = np.array([p1.x(), p1.y()])
    if g_vec is None:
        c = np.array([center.x(), center.y()])
        direction = c - r0
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            disp = rf - r0
            v = disp / time_of_flight
            return v, v
        g_vec = GRAVITY_MAG * (direction / dist)
    else:
        g_vec = np.asarray(g_vec, dtype=float)
    tf = time_of_flight
    v0 = (rf - r0) / tf - 0.5 * g_vec * tf
    vf = v0 + g_vec * tf
    return v0, vf


def _stumpff_c(z):
    """Stumpff function C(z)."""
    if abs(z) < 1e-6:
        return 0.5 - z / 24.0 + z * z / 720.0
    elif z > 0:
        sqz = math.sqrt(z)
        return (1.0 - math.cos(sqz)) / z
    else:
        sqz = math.sqrt(-z)
        return (math.cosh(sqz) - 1.0) / (-z)


def _stumpff_s(z):
    """Stumpff function S(z)."""
    if abs(z) < 1e-6:
        return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0
    elif z > 0:
        sqz = math.sqrt(z)
        return (sqz - math.sin(sqz)) / (sqz ** 3)
    else:
        sqz = math.sqrt(-z)
        return (math.sinh(sqz) - sqz) / (sqz ** 3)


def _stumpff_cp(z):
    """dC/dz."""
    if abs(z) < 1e-6:
        return -1.0 / 24.0 + z / 360.0 - z * z / 13440.0
    return (1.0 - z * _stumpff_s(z) - 2.0 * _stumpff_c(z)) / (2.0 * z)


def _stumpff_sp(z):
    """dS/dz."""
    if abs(z) < 1e-6:
        return -1.0 / 120.0 + z / 2520.0 - z * z / 120960.0
    return (_stumpff_c(z) - 3.0 * _stumpff_s(z)) / (2.0 * z)


def _stumpff_all(z):
    """Return (C, S, Cp, Sp) computed together, sharing one sqrt + one
    trig/hyperbolic call. Equivalent to calling the four scalar helpers
    but ~3-4x faster on the hot Lambert path."""
    if abs(z) < 1e-6:
        z2 = z * z
        C = 0.5 - z / 24.0 + z2 / 720.0
        S = 1.0 / 6.0 - z / 120.0 + z2 / 5040.0
        Cp = -1.0 / 24.0 + z / 360.0 - z2 / 13440.0
        Sp = -1.0 / 120.0 + z / 2520.0 - z2 / 120960.0
        return C, S, Cp, Sp
    if z > 0:
        sqz = math.sqrt(z)
        c = math.cos(sqz)
        s = math.sin(sqz)
        C = (1.0 - c) / z
        S = (sqz - s) / (sqz * z)  # = (sqz - s) / sqz^3
    else:
        sqz = math.sqrt(-z)
        c = math.cosh(sqz)
        s = math.sinh(sqz)
        C = (c - 1.0) / (-z)
        S = (s - sqz) / (-sqz * z)  # = (sinh - sqz) / sqz^3
    inv_2z = 0.5 / z
    Cp = (1.0 - z * S - 2.0 * C) * inv_2z
    Sp = (C - 3.0 * S) * inv_2z
    return C, S, Cp, Sp


def _lambert_z_solve(r1, r2, A, dt, mu, z_init=0.0):
    """Solve the universal-variable Lambert equation F(z; r1, r2, A, dt) = 0
    via Newton iteration starting from z_init, falling back to brentq if Newton fails.
    Returns (z, C, S, y) on success or None on failure.

    F(z) = (y/C)^{3/2} S + A sqrt(y) - sqrt(mu) dt,
    y(z) = r1 + r2 + A (z S - 1) / sqrt(C).
    """
    sqrtmu_dt = math.sqrt(mu) * dt
    Z_HIGH = 4.0 * math.pi ** 2 - 0.5  # just below first singularity
    Z_LOW = -200.0

    def y_of(z, C, S):
        return r1 + r2 + A * (z * S - 1.0) / math.sqrt(max(C, 1e-30))

    # --- Newton from z_init ---
    z = max(Z_LOW, min(Z_HIGH, float(z_init)))
    for _ in range(30):
        C, S, Cp, Sp = _stumpff_all(z)
        if C <= 1e-30:
            break
        y = y_of(z, C, S)
        if y < 0:
            # y must be positive for sqrt(y); nudge toward positive region.
            z += 0.5
            continue
        sqrtC = math.sqrt(C)
        sqrty = math.sqrt(y)
        yC = y / C
        yC15 = yC * math.sqrt(yC)  # (y/C)^{3/2}
        Fval = yC15 * S + A * sqrty - sqrtmu_dt
        # dF/dz via chain rule on y(z), C(z), S(z).
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
        # Clamp to sensible range to avoid escaping into singular region.
        if z_new < Z_LOW:
            z_new = 0.5 * (z + Z_LOW)
        elif z_new > Z_HIGH:
            z_new = 0.5 * (z + Z_HIGH)
        if abs(z_new - z) < 1e-12:
            z = z_new
            C, S, _, _ = _stumpff_all(z)
            if C <= 1e-30:
                break
            y = y_of(z, C, S)
            if y < 0:
                break
            return z, C, S, y
        z = z_new

    # --- brentq fallback ---
    def F(z):
        C, S, _, _ = _stumpff_all(z)
        y = y_of(z, C, S)
        if y < 0 or C <= 0:
            return float('inf')
        return (y / C) ** 1.5 * S + A * math.sqrt(y) - sqrtmu_dt

    def y_only(z):
        C, S, _, _ = _stumpff_all(z)
        return y_of(z, C, S)

    z_low = Z_LOW
    while y_only(z_low) < 0 and z_low < Z_HIGH:
        z_low += 1.0
    try:
        F_low = F(z_low)
        F_high = F(Z_HIGH)
        if not math.isfinite(F_low) or not math.isfinite(F_high) or F_low * F_high > 0:
            return None
        z = brentq(F, z_low, Z_HIGH, xtol=1e-12, maxiter=300)
    except (ValueError, RuntimeError):
        return None
    C, S, _, _ = _stumpff_all(z)
    y = y_of(z, C, S)
    if y < 0 or C <= 0:
        return None
    return z, C, S, y


def lambert_solve(r1_vec, r2_vec, dt, mu, side=0.0):
    """Solve Lambert's problem in 2D using universal variable z-iteration.
    r1_vec, r2_vec: numpy arrays, position vectors relative to gravity center.
    dt: time of flight (seconds).
    mu: gravitational parameter.
    side: ±1 hint for the antipodal-degenerate band (see lambert_numba.SIDE_EPS);
          0 means "auto".
    Returns (v1, v2) numpy arrays (velocity at r1 and r2)."""
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    if r1 < 1e-10 or r2 < 1e-10 or dt < 1e-10:
        v = (r2_vec - r1_vec) / max(dt, 1e-10)
        return v, v

    cos_dtheta = np.dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)

    # 2D cross product (z-component) to determine transfer direction
    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
    dtheta = math.acos(cos_dtheta)
    if cross_z < 0:
        dtheta = 2 * math.pi - dtheta

    sin_dtheta = math.sin(dtheta)
    if abs(1 - cos_dtheta) < 1e-14:
        v = (r2_vec - r1_vec) / dt
        return v, v
    if abs(sin_dtheta) < _SIDE_EPS:
        side_eff = side if side != 0.0 else (1.0 if cross_z >= 0.0 else -1.0)
        phi = -math.copysign(_SIDE_EPS, side_eff)
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        r2_vec = np.array([cphi * r2_vec[0] - sphi * r2_vec[1],
                           sphi * r2_vec[0] + cphi * r2_vec[1]])
        cos_dtheta = float(np.dot(r1_vec, r2_vec) / (r1 * r2))
        cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
        dtheta = math.pi + phi
        sin_dtheta = math.sin(dtheta)

    A = sin_dtheta * math.sqrt(r1 * r2 / (1.0 - cos_dtheta))

    result = _lambert_z_solve(r1, r2, A, dt, mu)
    if result is None:
        # Fallback to straight-line approximation
        v = (r2_vec - r1_vec) / dt
        return v, v
    z, C, S, y = result

    f = 1.0 - y / r1
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (gdot * r2_vec - r1_vec) / g

    return v1, v2


def lambert_solve_with_jac(r1_vec, r2_vec, dt, mu, z_init=0.0):
    """Lambert solver that also returns analytic Jacobians of (v1, v2)
    w.r.t. (r1, r2, dt) via the implicit function theorem applied to the
    universal-variable equation F(z; r1, r2, dt) = 0.

    Returns:
        v1, v2 : (2,) numpy arrays
        Jv1_r1, Jv1_r2 : (2,2) numpy arrays
        Jv1_dt : (2,) numpy array
        Jv2_r1, Jv2_r2 : (2,2) numpy arrays
        Jv2_dt : (2,) numpy array
        z      : float, the converged universal-variable (for warm-starting
                 subsequent calls with similar geometry)

    Falls back to a straight-line approximation (with matching analytic
    Jacobian) for degenerate geometries where the universal-variable
    bracket cannot be established.
    """
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)
    I2 = np.eye(2)

    def _straight_line():
        tof = max(dt, 1e-10)
        v = (r2_vec - r1_vec) / tof
        Jv_r1 = -I2 / tof
        Jv_r2 = I2 / tof
        Jv_dt = -(r2_vec - r1_vec) / (tof * tof)
        return v, v, Jv_r1, Jv_r2, Jv_dt, Jv_r1, Jv_r2, Jv_dt, 0.0

    r1 = float(np.linalg.norm(r1_vec))
    r2 = float(np.linalg.norm(r2_vec))
    if r1 < 1e-10 or r2 < 1e-10 or dt < 1e-10:
        return _straight_line()

    cos_dtheta = float(np.dot(r1_vec, r2_vec) / (r1 * r2))
    cos_dtheta = max(-1.0, min(1.0, cos_dtheta))
    cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
    dtheta = math.acos(cos_dtheta)
    if cross_z < 0:
        dtheta = 2.0 * math.pi - dtheta

    sin_dtheta = math.sin(dtheta)
    if abs(1.0 - cos_dtheta) < 1e-14:
        return _straight_line()
    if abs(sin_dtheta) < _SIDE_EPS:
        side_eff = side if side != 0.0 else (1.0 if cross_z >= 0.0 else -1.0)
        phi = -math.copysign(_SIDE_EPS, side_eff)
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        r2_vec = np.array([cphi * r2_vec[0] - sphi * r2_vec[1],
                           sphi * r2_vec[0] + cphi * r2_vec[1]])
        cos_dtheta = float(np.dot(r1_vec, r2_vec) / (r1 * r2))
        cross_z = r1_vec[0] * r2_vec[1] - r1_vec[1] * r2_vec[0]
        dtheta = math.pi + phi
        sin_dtheta = math.sin(dtheta)

    A = sin_dtheta * math.sqrt(r1 * r2 / (1.0 - cos_dtheta))

    result = _lambert_z_solve(r1, r2, A, dt, mu, z_init=z_init)
    if result is None:
        return _straight_line()
    z, C, S, y = result
    if y <= 0 or C <= 0 or abs(A) < 1e-15:
        return _straight_line()
    _, _, Cp, Sp = _stumpff_all(z)

    sqrtC = math.sqrt(C)
    sqrty = math.sqrt(y)
    chi = (z * S - 1.0) / sqrtC
    dchi_dz = (S + z * Sp) / sqrtC - (z * S - 1.0) * Cp / (2.0 * C ** 1.5)

    # dA/dr1, dA/dr2; dA/dt = 0.
    # A^2 = r1*r2 * (1 + cos_dtheta) = r1*r2 + r1_vec . r2_vec.
    dAsq_dr1 = (r2 / r1) * r1_vec + r2_vec
    dAsq_dr2 = (r1 / r2) * r2_vec + r1_vec
    dA_dr1 = dAsq_dr1 / (2.0 * A)
    dA_dr2 = dAsq_dr2 / (2.0 * A)

    # y partials at fixed z, plus dy/dz.
    dyz_dr1 = (r1_vec / r1) + dA_dr1 * chi
    dyz_dr2 = (r2_vec / r2) + dA_dr2 * chi
    dy_dz = A * dchi_dz

    # F partials.
    dF_dy = (1.5 * S * sqrty) / (C ** 1.5) + A / (2.0 * sqrty)
    dF_dC = -1.5 * (y ** 1.5) * S / (C ** 2.5)
    dF_dS = (y / C) ** 1.5
    dF_dA = sqrty

    dF_dz = dF_dy * dy_dz + dF_dC * Cp + dF_dS * Sp
    if abs(dF_dz) < 1e-20:
        return _straight_line()

    dFz_dr1 = dF_dy * dyz_dr1 + dF_dA * dA_dr1
    dFz_dr2 = dF_dy * dyz_dr2 + dF_dA * dA_dr2
    dFz_dt = -math.sqrt(mu)

    dz_dr1 = -dFz_dr1 / dF_dz
    dz_dr2 = -dFz_dr2 / dF_dz
    dz_dt = -dFz_dt / dF_dz

    dy_dr1 = dyz_dr1 + dy_dz * dz_dr1
    dy_dr2 = dyz_dr2 + dy_dz * dz_dr2
    dy_dt = dy_dz * dz_dt

    f_ = 1.0 - y / r1
    g_ = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2
    v1 = (r2_vec - f_ * r1_vec) / g_
    v2 = (gdot * r2_vec - r1_vec) / g_

    df_dr1 = -dy_dr1 / r1 + (y / (r1 ** 2)) * (r1_vec / r1)
    df_dr2 = -dy_dr2 / r1
    df_dt = -dy_dt / r1

    dgdot_dr1 = -dy_dr1 / r2
    dgdot_dr2 = -dy_dr2 / r2 + (y / (r2 ** 2)) * (r2_vec / r2)
    dgdot_dt = -dy_dt / r2

    sq_yom = math.sqrt(y / mu)
    A_2sqyM = A / (2.0 * math.sqrt(y * mu))
    dg_dr1 = dA_dr1 * sq_yom + A_2sqyM * dy_dr1
    dg_dr2 = dA_dr2 * sq_yom + A_2sqyM * dy_dr2
    dg_dt = A_2sqyM * dy_dt

    Jv1_r1 = (-np.outer(r1_vec, df_dr1) - f_ * I2) / g_ - np.outer(v1, dg_dr1) / g_
    Jv1_r2 = (I2 - np.outer(r1_vec, df_dr2)) / g_ - np.outer(v1, dg_dr2) / g_
    Jv1_dt = (-df_dt * r1_vec) / g_ - v1 * dg_dt / g_

    Jv2_r1 = (np.outer(r2_vec, dgdot_dr1) - I2) / g_ - np.outer(v2, dg_dr1) / g_
    Jv2_r2 = (np.outer(r2_vec, dgdot_dr2) + gdot * I2) / g_ - np.outer(v2, dg_dr2) / g_
    Jv2_dt = (dgdot_dt * r2_vec) / g_ - v2 * dg_dt / g_

    return v1, v2, Jv1_r1, Jv1_r2, Jv1_dt, Jv2_r1, Jv2_r2, Jv2_dt, z


# --- Optional Numba acceleration -------------------------------------------
# Replace the two public Lambert entry points with thin wrappers around the
# njit kernels. On any failure flag (degenerate geometry, Newton non-
# convergence), fall back to the pure-Python implementations above, which
# include the brentq bracketed solver. Pure-Python versions are preserved
# under the `_py` names so tests / debugging can compare.
try:
    import lambert_numba as _lnb  # triggers JIT warmup on import
    _NUMBA_AVAILABLE = True
    _lambert_solve_py = lambert_solve
    _lambert_solve_with_jac_py = lambert_solve_with_jac

    def lambert_solve(r1_vec, r2_vec, dt, mu, side=0.0):
        r1_arr = np.ascontiguousarray(r1_vec, dtype=np.float64)
        r2_arr = np.ascontiguousarray(r2_vec, dtype=np.float64)
        ok, v1, v2 = _lnb.lambert_solve_nb(r1_arr, r2_arr, float(dt), float(mu), float(side))
        if ok:
            return v1, v2
        return _lambert_solve_py(r1_vec, r2_vec, dt, mu, side=side)

    def lambert_solve_with_jac(r1_vec, r2_vec, dt, mu, z_init=0.0, side=0.0):
        r1_arr = np.ascontiguousarray(r1_vec, dtype=np.float64)
        r2_arr = np.ascontiguousarray(r2_vec, dtype=np.float64)
        out = _lnb.lambert_with_jac_nb(
            r1_arr, r2_arr, float(dt), float(mu), float(z_init), float(side)
        )
        if out[0]:
            # (v1, v2, Jv1_r1, Jv1_r2, Jv1_dt, Jv2_r1, Jv2_r2, Jv2_dt, z)
            return out[1:]
        return _lambert_solve_with_jac_py(r1_vec, r2_vec, dt, mu, z_init=z_init, side=side)
except ImportError:
    _NUMBA_AVAILABLE = False
    pass


def _kepler_deriv(state, mu):
    """Derivative for 2D Kepler propagation. state = [rx, ry, vx, vy]."""
    r = math.sqrt(state[0] ** 2 + state[1] ** 2)
    if r < 1e-10:
        return np.array([state[2], state[3], 0.0, 0.0])
    f = -mu / (r ** 3)
    return np.array([state[2], state[3], f * state[0], f * state[1]])


def compute_dynamic_arc(p0, p1, center, time_of_flight, num_points, side=0.0):
    """Compute Keplerian arc points between p0 and p1 using Lambert solver + RK4."""
    r1_vec = np.array([p0.x() - center.x(), p0.y() - center.y()])
    r2_vec = np.array([p1.x() - center.x(), p1.y() - center.y()])

    if np.linalg.norm(r1_vec) < 1e-10 or np.linalg.norm(r2_vec) < 1e-10:
        return [p0, p1]

    v1, _ = lambert_solve(r1_vec, r2_vec, time_of_flight, MU, side=side)

    cx, cy = center.x(), center.y()
    state = np.array([r1_vec[0], r1_vec[1], v1[0], v1[1]])
    dt_step = time_of_flight / (num_points - 1)

    points = [p0]
    for _ in range(1, num_points):
        k1 = _kepler_deriv(state, MU)
        k2 = _kepler_deriv(state + 0.5 * dt_step * k1, MU)
        k3 = _kepler_deriv(state + 0.5 * dt_step * k2, MU)
        k4 = _kepler_deriv(state + dt_step * k3, MU)
        state = state + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        points.append(QPointF(cx + state[0], cy + state[1]))

    return points


def compute_arc_velocities(p0, p1, center, time_of_flight, side=0.0):
    """Return (v0, vf) numpy arrays for the Lambert arc from p0 to p1."""
    r1_vec = np.array([p0.x() - center.x(), p0.y() - center.y()])
    r2_vec = np.array([p1.x() - center.x(), p1.y() - center.y()])

    if np.linalg.norm(r1_vec) < 1e-10 or np.linalg.norm(r2_vec) < 1e-10:
        disp = np.array([p1.x() - p0.x(), p1.y() - p0.y()])
        v = disp / time_of_flight
        return v, v

    return lambert_solve(r1_vec, r2_vec, time_of_flight, MU, side=side)


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.trajectories = []  # list of {"dots": [QPointF], "segments": [(i, j, mult)]}
        self.dot_radius_px = 4.0  # screen pixels; world radius = dot_radius_px / zoom
        # Target screen-pixel arc length between dropped dots while drawing.
        # Using pixels (not radians or DU) keeps draw-density independent of
        # both the current zoom and the orbital radius being traced.
        self.trace_pixel_spacing = 20.0
        self.trace_spacing = 1.0  # DU; legacy world-space fallback (unused)
        self.trace_dtau = math.pi / 6.0  # legacy; kept for back-compat
        self.dragging = False
        self.dragging_shape = None  # "triangle" or "square" when dragging a shape
        self.drag_offset = QPointF(0, 0)
        self.dragging_dot = None  # QPointF reference when dragging a black node
        self.shift_click_first = None  # first node selected for shift-click linking
        self.dragging_vel_end = None  # "triangle" or "square" when dragging a velocity line
        self.dragging_vel_t = 1.0  # parameter along line where grab occurred (0=center, 1=tip)
        self.x_held = False  # X key held: click black node to delete + merge segments
        # Hover-highlight target while X is held so the user can see what an
        # X-click would delete. One of:
        #   ("triangle",) | ("square",)
        #   ("segment", traj, seg_idx) | ("node", traj, dot)
        # or None if nothing is hovered.
        self.x_hover = None
        self._last_mouse_world = None  # last cursor pos in world coords
        # Bounded undo stack of snapshots taken just before each X-deletion.
        # Each entry is the tuple returned by `_snapshot_for_undo()`.
        self.delete_history = []
        self.delete_history_max = 100
        self.s_held = False  # S key held: shape drag slides along its Kepler orbit
        # Shape positions (center points) — placed in orbit around earth (DU)
        self.earth_center = QPointF(0, 0)
        self.earth_radius = 1.0  # DU
        self.tri_center = QPointF(1.5, 0)
        # Shape sizes in screen pixels; world size = px / zoom (see properties below).
        self.tri_size_px = 16.0
        self.sq_center = QPointF(-4.0, 0)
        self.sq_size_px = 14.0

        # Initialize circular velocity for each shape (prograde = CCW with y-up)
        tri_r = math.hypot(self.tri_center.x(), self.tri_center.y())
        tri_v_circ = math.sqrt(MU / tri_r)
        tri_vx = -self.tri_center.y() / tri_r * tri_v_circ
        tri_vy = self.tri_center.x() / tri_r * tri_v_circ
        self.tri_velocity_end = QPointF(self.tri_center.x() + tri_vx * VEL_SCALE, self.tri_center.y() + tri_vy * VEL_SCALE)

        sq_r = math.hypot(self.sq_center.x(), self.sq_center.y())
        sq_v_circ = math.sqrt(MU / sq_r)
        sq_vx = -self.sq_center.y() / sq_r * sq_v_circ
        sq_vy = self.sq_center.x() / sq_r * sq_v_circ
        self.sq_velocity_end = QPointF(self.sq_center.x() + sq_vx * VEL_SCALE, self.sq_center.y() + sq_vy * VEL_SCALE)

        # Moon: massless body for now, just a visual node. Starts on the +x
        # axis at a distance scaled to Earth's radius and given a circular
        # prograde velocity for future propagation.
        self.moon_radius = 0.27  # DU (display only; ~1737 km / 6371 km)
        self.moon_center = QPointF(60.0 * self.earth_radius, 0.0)
        moon_r = math.hypot(self.moon_center.x(), self.moon_center.y())
        moon_v_circ = math.sqrt(MU / moon_r)
        moon_vx = -self.moon_center.y() / moon_r * moon_v_circ
        moon_vy = self.moon_center.x() / moon_r * moon_v_circ
        self.moon_velocity_end = QPointF(
            self.moon_center.x() + moon_vx * VEL_SCALE,
            self.moon_center.y() + moon_vy * VEL_SCALE,
        )
        self.moon_orbit_elements = self._orbit_elements_at(
            self.moon_center, np.array([moon_vx, moon_vy]),
        )

        self.tri_orbit = []  # list of QPointF for triangle Kepler orbit
        self.sq_orbit = []  # list of QPointF for square Kepler orbit
        self.tri_orbit_mode = True
        self.sq_orbit_mode = True
        # Orbit-rendezvous mode: when X+deleted, the green shape disappears
        # and the boundary endpoint is free to slide along the orbit.
        # Optimizer treats true anomaly nu as a decision variable and the
        # boundary velocity is the orbital velocity at nu.
        self.tri_deleted = False
        self.sq_deleted = False
        self.tri_nu = 0.0
        self.sq_nu = 0.0
        self.tri_orbit_elements = None  # set on delete: dict(h_z, e, omega, sgn)
        self.sq_orbit_elements = None
        self.env_mode = "two_body"  # "two_body" or "constant_gravity"
        self.frame_mode = "inertial"  # "inertial" or "rotating"
        # Per-segment time of flight used for rendering. Optimizers may overwrite.
        # In two-body env this is `tau` (Sundman); in CG env it's physical TOF.
        self.render_tof = TAU_INIT_TWO_BODY
        # Active continuous-optimization mode: None | "energy" | "fuel".
        # When set, every drag release re-runs the optimizer.
        self.optimize_mode = None
        # Per-mode dv display scales (DU on screen per DU/TU). Adjustable via
        # the maneuver-length slider; on_dv_scale_changed lets the UI track
        # mode toggles.
        self.dv_scale_by_mode = dict(DV_SCALE_DEFAULTS)
        self.on_dv_scale_changed = None
        # Slider sync hook for the shared time variable (tau / tof).
        self.on_render_tof_changed = None

        # --- Off-thread optimizer plumbing -----------------------------------
        # Solves triggered via _run_active_optimizer (drag-release) run on a
        # background thread so dragging stays responsive during long solves.
        # Button-click optimizers stay synchronous to give immediate feedback.
        self._opt_executor = ThreadPoolExecutor(max_workers=1)
        self._opt_signals = _OptSignals()
        self._opt_signals.done.connect(self._on_opt_done)
        self._opt_signals.progress.connect(self._on_opt_progress)
        self._opt_running = False
        self._opt_restart = False

        self.setStyleSheet("background-color: white;")
        self.zoom = 50.0  # pixels per DU
        self.pan_offset = QPointF(600, 450)

        # Compute initial orbits
        self._compute_orbit_for_shape("triangle")
        self._compute_orbit_for_shape("square")

    def get_env_button_text(self):
        if self.env_mode == "two_body":
            return "Env: Two-Body"
        return "Env: Constant-Gravity"

    def toggle_env_mode(self):
        if self.env_mode == "two_body":
            self.env_mode = "constant_gravity"
            self.render_tof = TOF_INIT_CONSTANT_G
        else:
            self.env_mode = "two_body"
            self.render_tof = TAU_INIT_TWO_BODY
        if self.on_render_tof_changed is not None:
            self.on_render_tof_changed(self.render_tof, self.env_mode)
        # Recompute orbits (will be ignored visually in CG mode)
        if self.tri_orbit_mode:
            self._compute_orbit_for_shape("triangle")
        if self.sq_orbit_mode:
            self._compute_orbit_for_shape("square")
        self._run_active_optimizer()
        self.update()

    def get_frame_button_text(self):
        if self.frame_mode == "inertial":
            return "Frame: Inertial"
        return "Frame: Rotating"

    def toggle_frame_mode(self):
        if self.frame_mode == "inertial":
            self.frame_mode = "rotating"
        else:
            self.frame_mode = "inertial"
        self.update()

    def _constant_g_vec(self):
        # Down in y-up world coordinates
        return np.array([0.0, -GRAVITY_MAG])

    def _compute_segment_arc(self, p0, p1, center, time_of_flight, num_points, side=0.0):
        if self.env_mode == "constant_gravity":
            return compute_parabolic_arc(
                p0, p1, center, time_of_flight, num_points,
                g_vec=self._constant_g_vec(),
            )
        return compute_dynamic_arc(p0, p1, center, time_of_flight, num_points, side=side)

    # ---------- Sundman-style time scaling ----------
    # In two-body env the shared optimizer decision variable is `tau` (stored
    # in `self.render_tof`), not physical time. Per-segment physical TOF is
    #     tof_seg = tau * s^alpha * mult
    # with alpha = 1.5 (Sundman/Kepler exponent) and
    #     s = (|r0 - earth_center| + |rf - earth_center|) / 2.
    # This matches Kepler's period scaling (T ~ a^1.5) and gives the optimizer
    # a more uniform parameterization across orbits at different scales.
    # In constant-gravity env, no scaling is applied: tof_seg = tau * mult.
    def _seg_pos_mag_star(self, p0, p1):
        cx = self.earth_center.x()
        cy = self.earth_center.y()
        return 0.5 * (math.hypot(p0.x() - cx, p0.y() - cy)
                      + math.hypot(p1.x() - cx, p1.y() - cy))

    def _seg_tof(self, p0, p1, mult):
        if self.env_mode == "constant_gravity":
            return self.render_tof * mult
        s = self._seg_pos_mag_star(p0, p1)
        return self.render_tof * s * math.sqrt(s) * mult  # tau * s^1.5 * mult

    def _compute_segment_velocities(self, p0, p1, center, time_of_flight, side=0.0):
        if self.env_mode == "constant_gravity":
            return compute_parabolic_arc_velocities(
                p0, p1, center, time_of_flight, g_vec=self._constant_g_vec(),
            )
        return compute_arc_velocities(p0, p1, center, time_of_flight, side=side)

    def _seed_side(self, p0, p1):
        """Initial side hint for a new segment: sign(cross_z) of (p0-c, p1-c)
        relative to earth_center. Returns 0.0 for exactly antipodal geometry
        (caller will rely on auto-pick at first solve)."""
        cx, cy = self.earth_center.x(), self.earth_center.y()
        rx0, ry0 = p0.x() - cx, p0.y() - cy
        rx1, ry1 = p1.x() - cx, p1.y() - cy
        cz = rx0 * ry1 - ry0 * rx1
        if cz > 0:
            return 1.0
        if cz < 0:
            return -1.0
        return 0.0

    # ---------- Rotating-frame rendering helpers ----------
    def _rot_angle(self, t):
        # Mapping inertial -> rotating uses R(-omega * t).
        return -OMEGA_ROT * t

    def _rotate_point_about_earth(self, p, t):
        """Rotate a world-coord QPointF about earth_center by R(-omega*t)."""
        if self.frame_mode != "rotating" or t == 0.0:
            return p
        ang = self._rot_angle(t)
        c, s = math.cos(ang), math.sin(ang)
        cx, cy = self.earth_center.x(), self.earth_center.y()
        dx = p.x() - cx
        dy = p.y() - cy
        return QPointF(cx + c * dx - s * dy, cy + s * dx + c * dy)

    def _rotate_vec(self, v, t):
        """Rotate a 2-vector by R(-omega*t)."""
        if self.frame_mode != "rotating" or t == 0.0:
            return np.asarray(v, dtype=float)
        ang = self._rot_angle(t)
        c, s = math.cos(ang), math.sin(ang)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    def _rotating_velocity(self, v_inertial, p_world, t):
        """Map inertial velocity at world point p (and time t) to the rotating frame.
        v_rot = R(-omega*t) * (v_inertial - omega x r).
        With omega = +omega * z_hat in 2D y-up: omega x r = omega * (-r_y, +r_x)."""
        if self.frame_mode != "rotating":
            return np.asarray(v_inertial, dtype=float)
        rx = p_world.x() - self.earth_center.x()
        ry = p_world.y() - self.earth_center.y()
        v_rel = np.array([v_inertial[0] - OMEGA_ROT * (-ry),
                          v_inertial[1] - OMEGA_ROT * (rx)])
        return self._rotate_vec(v_rel, t)

    def _traj_node_times(self, traj):
        """Return {dot_index: cumulative_tof_from_first_node} by walking segments
        in their list order, accumulating each segment's Sundman-scaled time."""
        node_t = {}
        dots = traj["dots"]
        n = len(dots)
        for i_start, i_end, mult, _side in traj["segments"]:
            # Defensive: skip segments referencing out-of-range indices. This
            # can happen transiently if a paint fires mid-mutation; we'd
            # rather render a partial trajectory than crash.
            if not (0 <= i_start < n and 0 <= i_end < n):
                continue
            if i_start not in node_t:
                node_t[i_start] = 0.0
            node_t[i_end] = node_t[i_start] + self._seg_tof(dots[i_start], dots[i_end], mult)
        return node_t

    def _square_time(self):
        """Square is the 'arrival' body (t = tf). Find the largest node time
        at which sq_center participates in any trajectory. Returns 0 if none."""
        t_max = 0.0
        for traj in self.trajectories:
            dots = traj["dots"]
            node_t = self._traj_node_times(traj)
            for k, dot in enumerate(dots):
                if dot is self.sq_center and k in node_t:
                    if node_t[k] > t_max:
                        t_max = node_t[k]
        return t_max

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_X:
            self.x_held = True
            if self._last_mouse_world is not None:
                self._update_x_hover(self._last_mouse_world)
        elif event.key() == Qt.Key.Key_S:
            self.s_held = True

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_X:
            self.x_held = False
            if self.x_hover is not None:
                self.x_hover = None
                self.update()
        elif event.key() == Qt.Key.Key_S:
            self.s_held = False

    def _screen_to_world(self, screen_pos):
        wx = (screen_pos.x() - self.pan_offset.x()) / self.zoom
        wy = (self.pan_offset.y() - screen_pos.y()) / self.zoom
        return QPointF(wx, wy)

    @property
    def tri_size(self):
        # World-space size; pixel-constant on screen (scales with zoom).
        return self.tri_size_px / self.zoom

    @property
    def sq_size(self):
        return self.sq_size_px / self.zoom

    def _triangle_polygon(self):
        x, y = self.tri_center.x(), self.tri_center.y()
        s = self.tri_size
        return QPolygonF([
            QPointF(x - s / 2, y - s / 2),
            QPointF(x + s / 2, y),
            QPointF(x - s / 2, y + s / 2),
        ])

    def _deleted_shape_pos_vel(self, shape):
        """Inertial (r_xy, v_xy) of a deleted shape at its current nu, or
        (None, None) if the shape isn't in the deleted-with-orbit state."""
        if shape == "triangle":
            if not self.tri_deleted or self.tri_orbit_elements is None:
                return None, None
            return self._orbit_pos_vel(self.tri_orbit_elements, self.tri_nu)
        if not self.sq_deleted or self.sq_orbit_elements is None:
            return None, None
        return self._orbit_pos_vel(self.sq_orbit_elements, self.sq_nu)

    def _hit_triangle(self, pos):
        # Circular grab pad: shape body OR within `pad` of center. The pad
        # must exceed the velocity-line grab tolerance so clicks near the
        # shape always translate it instead of grabbing the velocity arrow
        # that emerges from its center.
        pad = max(self.tri_size, 14.0 / self.zoom)
        if self.tri_deleted:
            r_xy, _ = self._deleted_shape_pos_vel("triangle")
            if r_xy is None:
                return False
            return math.hypot(pos.x() - r_xy[0], pos.y() - r_xy[1]) <= pad
        cx, cy = self.tri_center.x(), self.tri_center.y()
        return math.hypot(pos.x() - cx, pos.y() - cy) <= pad

    def _hit_square(self, pos):
        pad = max(self.sq_size, 14.0 / self.zoom)
        if self.sq_deleted:
            r_xy, _ = self._deleted_shape_pos_vel("square")
            if r_xy is None:
                return False
            return math.hypot(pos.x() - r_xy[0], pos.y() - r_xy[1]) <= pad
        cx, cy = self.sq_center.x(), self.sq_center.y()
        return math.hypot(pos.x() - cx, pos.y() - cy) <= pad

    def _slide_deleted_to(self, shape, world_pos):
        """Set the deleted shape's nu to the orbit point closest to world_pos."""
        elements = (self.tri_orbit_elements if shape == "triangle"
                    else self.sq_orbit_elements)
        if elements is None:
            return False
        nus = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
        pts = np.empty((len(nus), 2))
        for i, n in enumerate(nus):
            pts[i] = self._orbit_pos_vel(elements, float(n))[0]
        target = np.array([world_pos.x(), world_pos.y()])
        d2 = (pts[:, 0] - target[0]) ** 2 + (pts[:, 1] - target[1]) ** 2
        idx = int(np.argmin(d2))
        if shape == "triangle":
            self.tri_nu = float(nus[idx])
        else:
            self.sq_nu = float(nus[idx])
        return True

    def _rebase_deleted_to_rendered(self, shape):
        """Move the (center, vel_end) anchor of a deleted shape to its current
        rendered (orbit, nu) state, with nu reset to 0. Lets a deleted shape
        be dragged using the same code paths as a live shape."""
        r_xy, v_xy = self._deleted_shape_pos_vel(shape)
        if r_xy is None:
            return
        new_c = QPointF(float(r_xy[0]), float(r_xy[1]))
        new_ve = QPointF(float(r_xy[0] + v_xy[0] * VEL_SCALE),
                         float(r_xy[1] + v_xy[1] * VEL_SCALE))
        if shape == "triangle":
            self.tri_center.setX(new_c.x())
            self.tri_center.setY(new_c.y())
            self.tri_velocity_end = new_ve
            elements = self._orbit_elements_at(self.tri_center, np.array([v_xy[0], v_xy[1]]))
            if elements is not None:
                self.tri_orbit_elements = elements
                self.tri_nu = elements["nu0"]
        else:
            self.sq_center.setX(new_c.x())
            self.sq_center.setY(new_c.y())
            self.sq_velocity_end = new_ve
            elements = self._orbit_elements_at(self.sq_center, np.array([v_xy[0], v_xy[1]]))
            if elements is not None:
                self.sq_orbit_elements = elements
                self.sq_nu = elements["nu0"]

    def _refresh_deleted_orbit(self, shape):
        """Rebuild orbit_elements from the current (center, vel_end) anchor and
        reset nu to 0. Call after a free drag of a deleted shape."""
        if shape == "triangle":
            if self.tri_velocity_end is None:
                return
            v = np.array([
                self.tri_velocity_end.x() - self.tri_center.x(),
                self.tri_velocity_end.y() - self.tri_center.y(),
            ]) / VEL_SCALE
            elements = self._orbit_elements_at(self.tri_center, v)
            if elements is not None:
                self.tri_orbit_elements = elements
                self.tri_nu = elements["nu0"]
        else:
            if self.sq_velocity_end is None:
                return
            v = np.array([
                self.sq_velocity_end.x() - self.sq_center.x(),
                self.sq_velocity_end.y() - self.sq_center.y(),
            ]) / VEL_SCALE
            elements = self._orbit_elements_at(self.sq_center, v)
            if elements is not None:
                self.sq_orbit_elements = elements
                self.sq_nu = elements["nu0"]

    def _snap_target_for(self, pos, r_inner, r_outer):
        """Pick the closer of triangle/square if within r_outer of pos.
        Returns (shape_dot, distance, mode) where mode is 'inner' (d < r_inner),
        'outer' (r_inner <= d < r_outer), or None (no snap)."""
        d_tri = math.hypot(pos.x() - self.tri_center.x(), pos.y() - self.tri_center.y())
        d_sq = math.hypot(pos.x() - self.sq_center.x(), pos.y() - self.sq_center.y())
        if d_tri <= d_sq and d_tri < r_outer:
            shape, d = self.tri_center, d_tri
        elif d_sq < r_outer:
            shape, d = self.sq_center, d_sq
        else:
            return None, None, None
        return shape, d, ('inner' if d < r_inner else 'outer')

    def _find_nearest_dot(self, pos, max_dist=0.30):
        """Find the nearest dot (including triangle/square centers) across all
        trajectories within max_dist (DU).
        Returns (dot_point, trajectory_or_None) or (None, None)."""
        best_dot = None
        best_traj = None
        best_dist = max_dist
        for traj in self.trajectories:
            for dot in traj["dots"]:
                d = math.hypot(pos.x() - dot.x(), pos.y() - dot.y())
                if d < best_dist:
                    best_dist = d
                    best_dot = dot
                    best_traj = traj
        # Check triangle center
        d = math.hypot(pos.x() - self.tri_center.x(), pos.y() - self.tri_center.y())
        if d < best_dist:
            best_dist = d
            best_dot = self.tri_center
            best_traj = None
        # Check square center
        d = math.hypot(pos.x() - self.sq_center.x(), pos.y() - self.sq_center.y())
        if d < best_dist:
            best_dist = d
            best_dot = self.sq_center
            best_traj = None
        return best_dot, best_traj

    def _update_x_hover(self, pos):
        """Recompute self.x_hover given current cursor world-pos. Mirrors the
        same hit checks (in the same order) as the X+click delete path so the
        highlight matches what an X-click would actually delete."""
        new_hover = None
        if self._hit_triangle(pos):
            new_hover = ("triangle",)
        elif self._hit_square(pos):
            new_hover = ("square",)
        else:
            seg = self._find_segment_near(
                pos, arc_max_dist=8.0 / self.zoom,
                endpoint_min_dist=10.0 / self.zoom,
            )
            if seg is not None:
                new_hover = ("segment", seg[0], seg[1])
            else:
                dot, traj = self._find_nearest_dot(pos, max_dist=12.0 / self.zoom)
                if dot is not None and dot is not self.tri_center and dot is not self.sq_center:
                    new_hover = ("node", traj, dot)
        if new_hover != self.x_hover:
            self.x_hover = new_hover
            self.update()

    def _find_segment_near(self, pos, arc_max_dist, endpoint_min_dist):
        """Locate the segment whose arc passes nearest pos. Returns
        (traj, seg_idx) or None. Same geometry as _delete_segment_at but
        without mutating anything."""
        click = np.array([pos.x(), pos.y()])
        center = self.earth_center
        best = None
        dense = 200
        for traj in self.trajectories:
            dots = traj["dots"]
            for s_idx, (i_start, i_end, mult, _side) in enumerate(traj["segments"]):
                arc_pts = self._compute_segment_arc(
                    dots[i_start], dots[i_end], center,
                    self._seg_tof(dots[i_start], dots[i_end], mult), dense + 1,
                    side=_side,
                )
                d_min = float("inf")
                for j in range(len(arc_pts) - 1):
                    a = np.array([arc_pts[j].x(), arc_pts[j].y()])
                    b = np.array([arc_pts[j + 1].x(), arc_pts[j + 1].y()])
                    ab = b - a
                    L2 = float(ab @ ab)
                    if L2 < 1e-30:
                        d = float(np.linalg.norm(click - a))
                    else:
                        t = max(0.0, min(1.0, float((click - a) @ ab) / L2))
                        proj = a + t * ab
                        d = float(np.linalg.norm(click - proj))
                    if d < d_min:
                        d_min = d
                if best is None or d_min < best[0]:
                    best = (d_min, traj, s_idx)
        if best is None or best[0] > arc_max_dist:
            return None
        _, traj, s_idx = best
        i_start, i_end, _, _ = traj["segments"][s_idx]
        d_s = math.hypot(click[0] - traj["dots"][i_start].x(),
                         click[1] - traj["dots"][i_start].y())
        d_e = math.hypot(click[0] - traj["dots"][i_end].x(),
                         click[1] - traj["dots"][i_end].y())
        if d_s < endpoint_min_dist or d_e < endpoint_min_dist:
            return None
        return traj, s_idx

    def _segment_exists(self, p0, p1):
        """Check if a segment already exists between two points."""
        for traj in self.trajectories:
            dots = traj["dots"]
            for i_start, i_end, _, _ in traj["segments"]:
                if (dots[i_start] is p0 and dots[i_end] is p1) or (dots[i_start] is p1 and dots[i_end] is p0):
                    return True
        return False

    def _delete_segment_at(self, pos, arc_max_dist, endpoint_min_dist):
        """X+click delete-segment: locate the segment whose arc passes nearest
        pos. Succeeds only if the click is within arc_max_dist of the arc and
        farther than endpoint_min_dist from each endpoint. Removes the segment;
        any dots that become orphaned (no remaining incident segments) are
        removed too. Empty trajectories are pruned."""
        click = np.array([pos.x(), pos.y()])
        center = self.earth_center
        best = None  # (dist, traj, seg_idx)
        dense = 200
        for traj in self.trajectories:
            dots = traj["dots"]
            for s_idx, (i_start, i_end, mult, _side) in enumerate(traj["segments"]):
                arc_pts = self._compute_segment_arc(
                    dots[i_start], dots[i_end], center,
                    self._seg_tof(dots[i_start], dots[i_end], mult), dense + 1,
                    side=_side,
                )
                # Point-to-line-segment distance over consecutive samples.
                d_min = float("inf")
                for j in range(len(arc_pts) - 1):
                    a = np.array([arc_pts[j].x(), arc_pts[j].y()])
                    b = np.array([arc_pts[j + 1].x(), arc_pts[j + 1].y()])
                    ab = b - a
                    L2 = float(ab @ ab)
                    if L2 < 1e-30:
                        d = float(np.linalg.norm(click - a))
                    else:
                        t = max(0.0, min(1.0, float((click - a) @ ab) / L2))
                        proj = a + t * ab
                        d = float(np.linalg.norm(click - proj))
                    if d < d_min:
                        d_min = d
                if best is None or d_min < best[0]:
                    best = (d_min, traj, s_idx)
        if best is None or best[0] > arc_max_dist:
            return False
        _, traj, s_idx = best
        i_start, i_end, _, _ = traj["segments"][s_idx]
        d_s = math.hypot(click[0] - traj["dots"][i_start].x(),
                         click[1] - traj["dots"][i_start].y())
        d_e = math.hypot(click[0] - traj["dots"][i_end].x(),
                         click[1] - traj["dots"][i_end].y())
        if d_s < endpoint_min_dist or d_e < endpoint_min_dist:
            return False
        del traj["segments"][s_idx]
        return True

    def _insert_node_on_segment(self, pos, max_dist=None):
        """Cmd+click insert: find the segment whose arc passes nearest pos.
        Densely sample the arc to find the continuous time fraction
        t* in (0, N) closest to the click (N = segment mult). Round t* to
        nearest integer k. If k is 0 or N (existing endpoint), do nothing.
        Otherwise split into two segments with mults k and N-k.
        If max_dist is given, the click must lie within that world-distance
        of the chosen arc; otherwise no-op."""
        click_world = np.array([pos.x(), pos.y()])
        center = self.earth_center
        best = None  # (dist, traj, seg_idx, t_star, n_int)
        dense = 200  # samples per segment for continuous projection
        for traj in self.trajectories:
            dots = traj["dots"]
            for s_idx, (i_start, i_end, mult, _side) in enumerate(traj["segments"]):
                n_int = int(round(mult))
                if n_int < 2:
                    continue
                arc_pts = self._compute_segment_arc(
                    dots[i_start], dots[i_end], center,
                    self._seg_tof(dots[i_start], dots[i_end], mult), dense + 1,
                    side=_side,
                )
                # Find sample closest in space; t in [0, N].
                best_local = None
                for j, p in enumerate(arc_pts):
                    d = math.hypot(click_world[0] - p.x(), click_world[1] - p.y())
                    if best_local is None or d < best_local[0]:
                        best_local = (d, j)
                d_min, j_min = best_local
                t_star = (j_min / dense) * n_int
                if best is None or d_min < best[0]:
                    best = (d_min, traj, s_idx, t_star, n_int)
        if best is None:
            return False
        if max_dist is not None and best[0] > max_dist:
            return False
        _, traj, s_idx, t_star, n_int = best
        k = int(round(t_star))
        if k <= 0 or k >= n_int:
            return False
        i_start, i_end, mult, _side_existing = traj["segments"][s_idx]
        # Place the new dot at the arc point at time = k * tof along segment.
        arc_pts = self._compute_segment_arc(
            traj["dots"][i_start], traj["dots"][i_end], center,
            self._seg_tof(traj["dots"][i_start], traj["dots"][i_end], mult), n_int + 1,
            side=_side_existing,
        )
        new_pt = arc_pts[k]
        new_dot = QPointF(new_pt.x(), new_pt.y())
        traj["dots"].append(new_dot)
        new_idx = len(traj["dots"]) - 1
        traj["segments"][s_idx] = (i_start, new_idx, float(k), _side_existing)
        traj["segments"].insert(s_idx + 1, (new_idx, i_end, float(n_int - k), _side_existing))
        return True

    def _delete_black_node(self, dot):
        """X+click delete a black node. Cases:
        - 1 incoming + 1 outgoing segment in the same trajectory: remove
          the node and merge the two segments (mult = m_in + m_out).
        - 0 incident segments (orphan): just remove the node, prune empty
          trajectories.
        Other configurations are no-ops."""
        if dot is self.tri_center or dot is self.sq_center:
            return False
        for traj in self.trajectories:
            dots = traj["dots"]
            if dot not in dots:
                continue
            k = dots.index(dot)
            in_seg_idx = None
            out_seg_idx = None
            for s_idx, (i_start, i_end, _m, _s) in enumerate(traj["segments"]):
                if i_end == k:
                    if in_seg_idx is not None:
                        return False
                    in_seg_idx = s_idx
                elif i_start == k:
                    if out_seg_idx is not None:
                        return False
                    out_seg_idx = s_idx
            # Orphan node: no incident segments, just remove it.
            if in_seg_idx is None and out_seg_idx is None:
                del dots[k]
                renumbered = []
                for i_s, i_e, m, _s in traj["segments"]:
                    if i_s > k:
                        i_s -= 1
                    if i_e > k:
                        i_e -= 1
                    renumbered.append((i_s, i_e, m, _s))
                traj["segments"] = renumbered
                if not traj["dots"]:
                    self.trajectories.remove(traj)
                return True
            if in_seg_idx is None or out_seg_idx is None:
                return False
            i_prev, _, m_in, side_in = traj["segments"][in_seg_idx]
            _, i_next, m_out, _side_out = traj["segments"][out_seg_idx]
            # Build merged segment list, drop the two old segments. Insert
            # the merged segment at the earlier of the two removed slots so
            # traversal order is preserved (required by _traj_node_times
            # and the optimizer's first-occurrence convention).
            insert_at = min(in_seg_idx, out_seg_idx)
            kept = [s for s_idx, s in enumerate(traj["segments"])
                    if s_idx not in (in_seg_idx, out_seg_idx)]
            new_segments = (kept[:insert_at]
                            + [(i_prev, i_next, m_in + m_out, side_in)]
                            + kept[insert_at:])
            # Remove the dot and renumber indices in segments above k.
            del dots[k]
            renumbered = []
            for i_s, i_e, m, _s in new_segments:
                if i_s > k:
                    i_s -= 1
                if i_e > k:
                    i_e -= 1
                renumbered.append((i_s, i_e, m, _s))
            traj["segments"] = renumbered
            return True
        return False

    def _snapshot_for_undo(self):
        """Capture the state mutated by any X-deletion path. Trajectories
        are deep-copied with shape-center QPointF identity preserved (many
        code paths use `dot is self.tri_center`)."""
        trajs = []
        for traj in self.trajectories:
            new_dots = []
            for d in traj["dots"]:
                if d is self.tri_center or d is self.sq_center:
                    new_dots.append(d)
                else:
                    new_dots.append(QPointF(d.x(), d.y()))
            new_segs = [(i, j, m, sd) for (i, j, m, sd) in traj["segments"]]
            trajs.append({"dots": new_dots, "segments": new_segs})
        return {
            "trajectories": trajs,
            "tri_deleted": self.tri_deleted,
            "tri_nu": self.tri_nu,
            "tri_orbit_elements": (None if self.tri_orbit_elements is None
                                   else dict(self.tri_orbit_elements)),
            "sq_deleted": self.sq_deleted,
            "sq_nu": self.sq_nu,
            "sq_orbit_elements": (None if self.sq_orbit_elements is None
                                  else dict(self.sq_orbit_elements)),
        }

    def _push_delete_undo(self):
        self.delete_history.append(self._snapshot_for_undo())
        if len(self.delete_history) > self.delete_history_max:
            self.delete_history.pop(0)

    def _pop_delete_undo(self):
        """Discard the most recent snapshot without restoring (used when a
        speculative push didn't actually result in a deletion)."""
        if self.delete_history:
            self.delete_history.pop()

    def undo_delete(self):
        """Restore the state from the most recent X-deletion snapshot."""
        if not self.delete_history:
            return
        snap = self.delete_history.pop()
        self.trajectories = snap["trajectories"]
        self.tri_deleted = snap["tri_deleted"]
        self.tri_nu = snap["tri_nu"]
        self.tri_orbit_elements = snap["tri_orbit_elements"]
        self.sq_deleted = snap["sq_deleted"]
        self.sq_nu = snap["sq_nu"]
        self.sq_orbit_elements = snap["sq_orbit_elements"]
        # Refresh derived display state for shapes (orbit traces).
        if self.tri_orbit_mode:
            self._compute_orbit_for_shape("triangle")
        if self.sq_orbit_mode:
            self._compute_orbit_for_shape("square")
        self._run_active_optimizer()
        self.update()

    def _compute_orbit_for_shape(self, shape):
        """Compute Kepler orbit for triangle or square if it has a velocity vector."""
        if shape == "triangle" and self.tri_velocity_end is not None:
            vel = np.array([
                self.tri_velocity_end.x() - self.tri_center.x(),
                self.tri_velocity_end.y() - self.tri_center.y(),
            ]) / VEL_SCALE
            self.tri_orbit = compute_kepler_orbit(
                self.tri_center, vel, self.earth_center, MU, ORBIT_NUM_POINTS,
            )
        elif shape == "square" and self.sq_velocity_end is not None:
            vel = np.array([
                self.sq_velocity_end.x() - self.sq_center.x(),
                self.sq_velocity_end.y() - self.sq_center.y(),
            ]) / VEL_SCALE
            self.sq_orbit = compute_kepler_orbit(
                self.sq_center, vel, self.earth_center, MU, ORBIT_NUM_POINTS,
            )

    def _orbit_elements_at(self, center_qpt, vel_xy):
        """Return dict(h_z, e, omega, sgn, nu0) for the 2D Kepler orbit through\n        (center, vel), or None for degenerate cases."""
        ec = np.array([self.earth_center.x(), self.earth_center.y()])
        r = np.array([center_qpt.x() - ec[0], center_qpt.y() - ec[1]])
        v = np.asarray(vel_xy, dtype=float)
        r_mag = float(np.linalg.norm(r))
        if r_mag < 1e-12:
            return None
        h_z = float(r[0] * v[1] - r[1] * v[0])
        if abs(h_z) < 1e-12:
            return None
        e_vec = np.array([v[1] * h_z, -v[0] * h_z]) / MU - r / r_mag
        e = float(np.linalg.norm(e_vec))
        if e > 1e-9:
            omega = math.atan2(e_vec[1], e_vec[0])
        else:
            omega = math.atan2(r[1], r[0])
        sgn = 1.0 if h_z > 0 else -1.0
        theta = math.atan2(r[1], r[0])
        # theta = omega + sgn * nu  =>  nu = (theta - omega) * sgn
        nu = (theta - omega) * sgn
        nu = (nu + math.pi) % (2.0 * math.pi) - math.pi
        return {"h_z": h_z, "e": e, "omega": omega, "sgn": sgn, "nu0": nu}

    def _orbit_pos_vel(self, elements, nu):
        """Return (r_world_xy, v_xy) at true anomaly nu on the cached orbit."""
        h_z = elements["h_z"]
        e = elements["e"]
        omega = elements["omega"]
        sgn = elements["sgn"]
        p = h_z * h_z / MU
        cos_nu = math.cos(nu)
        sin_nu = math.sin(nu)
        denom = 1.0 + e * cos_nu
        if denom < 1e-12:
            denom = 1e-12
        r_mag = p / denom
        theta = omega + sgn * nu
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        r_xy = np.array([r_mag * cos_th, r_mag * sin_th])
        # Radial speed and angular rate (signed by h_z).
        rdot = e * sin_nu * abs(h_z) / p if p > 1e-12 else 0.0
        th_dot = h_z / (r_mag * r_mag)
        v_xy = np.array([rdot * cos_th - r_mag * sin_th * th_dot,
                         rdot * sin_th + r_mag * cos_th * th_dot])
        ec = np.array([self.earth_center.x(), self.earth_center.y()])
        return r_xy + ec, v_xy

    def _orbit_nu_after_dt(self, elements, dt):
        """Propagate elliptic Kepler elements: return ν at time elements['nu0']+dt.
        Returns nu0 unchanged for non-elliptic / degenerate orbits."""
        e = float(elements["e"])
        if e >= 1.0:
            return float(elements["nu0"])
        h_z = float(elements["h_z"])
        sgn = float(elements["sgn"])
        nu0 = float(elements["nu0"])
        p = h_z * h_z / MU
        denom = 1.0 - e * e
        if denom < 1e-12:
            return nu0
        a = p / denom
        if a <= 0.0:
            return nu0
        n = math.sqrt(MU / (a * a * a))
        sqrt1m = math.sqrt(max(1.0 - e, 0.0))
        sqrt1p = math.sqrt(max(1.0 + e, 0.0))
        E0 = 2.0 * math.atan2(sqrt1m * math.sin(nu0 / 2.0),
                              sqrt1p * math.cos(nu0 / 2.0))
        M0 = E0 - e * math.sin(E0)
        M = M0 + sgn * n * float(dt)
        E = M if e < 0.8 else math.pi
        for _ in range(50):
            f = E - e * math.sin(E) - M
            fp = 1.0 - e * math.cos(E)
            dE = -f / fp if fp != 0.0 else 0.0
            E += dE
            if abs(dE) < 1e-12:
                break
        nu = 2.0 * math.atan2(sqrt1p * math.sin(E / 2.0),
                              sqrt1m * math.cos(E / 2.0))
        return nu

    def _delete_shape(self, shape):
        """X+click on triangle/square: hide it but keep the orbit as a\n        rendezvous constraint. Optimizer becomes free to slide the boundary\n        along the orbit (true anomaly nu becomes a decision variable)."""
        if self.env_mode != "two_body":
            return False
        if shape == "triangle":
            if self.tri_deleted or self.tri_velocity_end is None:
                return False
            vel = np.array([
                self.tri_velocity_end.x() - self.tri_center.x(),
                self.tri_velocity_end.y() - self.tri_center.y(),
            ]) / VEL_SCALE
            elements = self._orbit_elements_at(self.tri_center, vel)
            if elements is None:
                return False
            self.tri_orbit_elements = elements
            self.tri_nu = elements["nu0"]
            self.tri_deleted = True
        else:
            if self.sq_deleted or self.sq_velocity_end is None:
                return False
            vel = np.array([
                self.sq_velocity_end.x() - self.sq_center.x(),
                self.sq_velocity_end.y() - self.sq_center.y(),
            ]) / VEL_SCALE
            elements = self._orbit_elements_at(self.sq_center, vel)
            if elements is None:
                return False
            self.sq_orbit_elements = elements
            self.sq_nu = elements["nu0"]
            self.sq_deleted = True
        return True

    def _restore_shape(self, shape):
        """Cmd+click on a deleted triangle/square's ν marker: bring the
        shape back at its current orbit position. The shape's center and
        velocity_end already reflect the latest optimizer ν, so we just
        clear the deleted flag."""
        if shape == "triangle":
            if not self.tri_deleted:
                return False
            self.tri_deleted = False
        else:
            if not self.sq_deleted:
                return False
            self.sq_deleted = False
        return True

    def _slide_shape_to(self, shape, world_pos):
        """Snap shape center to the nearest point on its Kepler orbit and
        update the velocity from conserved orbital elements (h, e, a) so the
        orbit shape is preserved. No-op (returns False) when there is no
        usable orbit (constant-gravity env, missing velocity, degenerate)."""
        if self.env_mode != "two_body":
            return False
        if shape == "triangle":
            center = self.tri_center
            vel_end = self.tri_velocity_end
            orbit = self.tri_orbit
        else:
            center = self.sq_center
            vel_end = self.sq_velocity_end
            orbit = self.sq_orbit
        if vel_end is None or not orbit:
            return False
        ec = self.earth_center
        r0 = np.array([center.x() - ec.x(), center.y() - ec.y()])
        v0 = np.array([vel_end.x() - center.x(), vel_end.y() - center.y()]) / VEL_SCALE
        h_z = r0[0] * v0[1] - r0[1] * v0[0]  # specific angular momentum (z)
        if abs(h_z) < 1e-12:
            return False
        r0_mag = float(np.linalg.norm(r0))
        if r0_mag < 1e-12:
            return False
        # Eccentricity vector: e = (v × h)/mu - r̂ ; in 2D with h = h_z·ẑ:
        # v × (h_z ẑ) = (v_y·h_z, -v_x·h_z)
        e_vec = np.array([v0[1] * h_z, -v0[0] * h_z]) / MU - r0 / r0_mag
        inv_a = 2.0 / r0_mag - float(v0 @ v0) / MU
        # Find the closest orbit-polyline sample to the cursor (world coords).
        pts = np.array([[p.x() - ec.x(), p.y() - ec.y()] for p in orbit])
        target = np.array([world_pos.x() - ec.x(), world_pos.y() - ec.y()])
        d2 = (pts[:, 0] - target[0]) ** 2 + (pts[:, 1] - target[1]) ** 2
        idx = int(np.argmin(d2))
        r_new = pts[idx]
        r_new_mag = float(np.linalg.norm(r_new))
        if r_new_mag < 1e-12:
            return False
        r_hat = r_new / r_new_mag
        # Tangential velocity from h_z (signed): v · t̂ = h_z / |r|, with
        # t̂ = ẑ × r̂ = (-r̂_y, r̂_x); so v_t_vec = (h_z/|r|) * t̂.
        t_hat = np.array([-r_hat[1], r_hat[0]])
        v_t = h_z / r_new_mag
        # Radial velocity: v_r = (mu/h_z) * e · sin(ν), with
        # e·sin(ν) = (e_x·r̂_y - e_y·r̂_x).
        v_r = (MU / h_z) * (e_vec[0] * r_hat[1] - e_vec[1] * r_hat[0])
        # Sanity: vis-viva for elliptic orbits.
        if inv_a > 0:
            v_mag2 = v_r * v_r + v_t * v_t
            target_v_mag2 = MU * (2.0 / r_new_mag - inv_a)
            if not (0.5 * target_v_mag2 < v_mag2 < 2.0 * target_v_mag2):
                return False
        v_new = v_r * r_hat + v_t * t_hat
        new_cx = r_new[0] + ec.x()
        new_cy = r_new[1] + ec.y()
        if shape == "triangle":
            self.tri_center.setX(new_cx)
            self.tri_center.setY(new_cy)
            self.tri_velocity_end = QPointF(
                new_cx + v_new[0] * VEL_SCALE,
                new_cy + v_new[1] * VEL_SCALE,
            )
        else:
            self.sq_center.setX(new_cx)
            self.sq_center.setY(new_cy)
            self.sq_velocity_end = QPointF(
                new_cx + v_new[0] * VEL_SCALE,
                new_cy + v_new[1] * VEL_SCALE,
            )
        return True

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._screen_to_world(QPointF(event.pos()))
            # X + click on a black node to delete it and merge its two
            # incident segments (mult = m_in + m_out).
            # X + click on a segment (within 1 px of arc, >5 px from each
            # endpoint) to delete that segment.
            if self.x_held:
                # Try shape delete first (X+click inside triangle/square
                # converts the boundary into an orbit-rendezvous endpoint).
                if self._hit_triangle(pos):
                    self._push_delete_undo()
                    if self._delete_shape("triangle"):
                        self._run_active_optimizer()
                        self.update()
                    else:
                        self._pop_delete_undo()
                    return
                if self._hit_square(pos):
                    self._push_delete_undo()
                    if self._delete_shape("square"):
                        self._run_active_optimizer()
                        self.update()
                    else:
                        self._pop_delete_undo()
                    return
                # Try segment delete first (requires click >5 px from each
                # endpoint), so a click on a line near a node doesn't take
                # the node with it.
                self._push_delete_undo()
                if self._delete_segment_at(pos, arc_max_dist=8.0 / self.zoom,
                                           endpoint_min_dist=10.0 / self.zoom):
                    self._run_active_optimizer()
                    self.update()
                    return
                dot, _traj = self._find_nearest_dot(pos, max_dist=12.0 / self.zoom)
                if dot is not None and self._delete_black_node(dot):
                    self._run_active_optimizer()
                    self.update()
                    return
                # Nothing was deleted — drop the speculative snapshot.
                self._pop_delete_undo()
                return
            # V + click on triangle/square to draw velocity line
            # O + click on triangle/square to compute Kepler orbit
            # Shift-click to link two existing nodes (including triangle/square)
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                dot, traj = self._find_nearest_dot(pos)
                if dot is not None:
                    if self.shift_click_first is None:
                        self.shift_click_first = (dot, traj)
                    else:
                        first_dot, first_traj = self.shift_click_first
                        if first_dot is not dot and not self._segment_exists(first_dot, dot):
                            # Find trajectory to add segment to, prefer first_traj, then traj
                            target_traj = first_traj or traj
                            if target_traj is None:
                                # Both are shape nodes, create a new trajectory
                                target_traj = {"dots": [first_dot, dot], "segments": [(0, 1, 1.0, self._seed_side(first_dot, dot))]}
                                self.trajectories.append(target_traj)
                            else:
                                # Add dots if not already present, then add segment
                                if first_dot not in target_traj["dots"]:
                                    target_traj["dots"].append(first_dot)
                                if dot not in target_traj["dots"]:
                                    target_traj["dots"].append(dot)
                                i0 = target_traj["dots"].index(first_dot)
                                i1 = target_traj["dots"].index(dot)
                                target_traj["segments"].append((i0, i1, 1.0, self._seed_side(first_dot, dot)))
                            self.update()
                        self.shift_click_first = None
                return
            # Check shapes for dragging, V-line, or O-orbit. Cmd is reserved
            # for trajectory drawing, so don't translate shapes or grab
            # velocity arrows while Cmd is held.
            cmd_held = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if not cmd_held and self._hit_triangle(pos):
                if self.tri_deleted:
                    self._rebase_deleted_to_rendered("triangle")
                self.dragging_shape = "triangle"
                self.drag_offset = self.tri_center - pos
                return
            if not cmd_held and self._hit_square(pos):
                if self.sq_deleted:
                    self._rebase_deleted_to_rendered("square")
                self.dragging_shape = "square"
                self.drag_offset = self.sq_center - pos
                return
            # Check for dragging anywhere on a velocity line
            if not event.modifiers():
                # Build (shape, center, vel_end) list including deleted shapes
                # (whose center/vel are computed from orbit at current nu).
                vel_targets = []
                if not self.tri_deleted:
                    vel_targets.append(("triangle", self.tri_center, self.tri_velocity_end))
                else:
                    r_xy, v_xy = self._deleted_shape_pos_vel("triangle")
                    if r_xy is not None:
                        c = QPointF(float(r_xy[0]), float(r_xy[1]))
                        ve = QPointF(c.x() + v_xy[0] * VEL_SCALE, c.y() + v_xy[1] * VEL_SCALE)
                        vel_targets.append(("triangle", c, ve))
                if not self.sq_deleted:
                    vel_targets.append(("square", self.sq_center, self.sq_velocity_end))
                else:
                    r_xy, v_xy = self._deleted_shape_pos_vel("square")
                    if r_xy is not None:
                        c = QPointF(float(r_xy[0]), float(r_xy[1]))
                        ve = QPointF(c.x() + v_xy[0] * VEL_SCALE, c.y() + v_xy[1] * VEL_SCALE)
                        vel_targets.append(("square", c, ve))
                for shape, center, vel_end in vel_targets:
                    if vel_end is not None:
                        # Don't grab the velocity arrow inside the shape body —
                        # those clicks should translate the shape instead.
                        if shape == "triangle" and self._hit_triangle(pos):
                            continue
                        if shape == "square" and self._hit_square(pos):
                            continue
                        # Closest point on line segment center->vel_end to pos
                        lx = vel_end.x() - center.x()
                        ly = vel_end.y() - center.y()
                        seg_len_sq = lx * lx + ly * ly
                        if seg_len_sq > 1e-6:
                            t = ((pos.x() - center.x()) * lx + (pos.y() - center.y()) * ly) / seg_len_sq
                            t = max(0.05, min(1.0, t))  # clamp, avoid division near zero
                            proj_x = center.x() + t * lx
                            proj_y = center.y() + t * ly
                            d = math.hypot(pos.x() - proj_x, pos.y() - proj_y)
                            if d < 12.0 / self.zoom:
                                if shape == "triangle" and self.tri_deleted:
                                    self._rebase_deleted_to_rendered("triangle")
                                elif shape == "square" and self.sq_deleted:
                                    self._rebase_deleted_to_rendered("square")
                                self.dragging_vel_end = shape
                                self.dragging_vel_t = t
                                return
            # Check for dragging a black node (no modifier)
            if not event.modifiers():
                dot, traj = self._find_nearest_dot(pos)
                if dot is not None and dot is not self.tri_center and dot is not self.sq_center:
                    self.dragging_dot = dot
                    return
            # Trajectory drawing requires Cmd
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                start_pos = self._screen_to_world(QPointF(event.pos()))
                # Cmd+click on a deleted shape's nu marker: restore it.
                r_restore = 30.0 / self.zoom
                for shape, deleted, ctr in [
                    ("triangle", self.tri_deleted, self.tri_center),
                    ("square", self.sq_deleted, self.sq_center),
                ]:
                    if not deleted:
                        continue
                    if math.hypot(start_pos.x() - ctr.x(), start_pos.y() - ctr.y()) <= r_restore:
                        if self._restore_shape(shape):
                            self._run_active_optimizer()
                            self.update()
                            return
                # If close to an existing arc, insert a node there instead of
                # starting a new trajectory.
                if self._insert_node_on_segment(start_pos, max_dist=25.0 / self.zoom):
                    self._run_active_optimizer()
                    self.update()
                    return
                self.dragging = True
                # Two-radius snap to triangle/square:
                #   inner (< 30 px): trajectory starts AT the shape itself
                #   outer (30..75 px): shape + free node at click + connector
                #   beyond 75 px: free node at click, no connection
                r_inner = 30.0 / self.zoom
                r_outer = 75.0 / self.zoom
                snap_node, _, snap_mode = self._snap_target_for(start_pos, r_inner, r_outer)
                if snap_mode == 'inner':
                    self.trajectories.append({"dots": [snap_node], "segments": []})
                elif snap_mode == 'outer':
                    self.trajectories.append({"dots": [snap_node, start_pos], "segments": [(0, 1, 1.0, self._seed_side(snap_node, start_pos))]})
                else:
                    self.trajectories.append({"dots": [start_pos], "segments": []})
                self.update()

    def mouseMoveEvent(self, event):
        pos = self._screen_to_world(QPointF(event.pos()))
        self._last_mouse_world = pos
        # While X is held with no active drag, highlight whatever an X-click
        # at the current cursor position would delete.
        if (self.x_held and self.dragging_vel_end is None and
                self.dragging_dot is None and self.dragging_shape is None and
                not self.dragging):
            self._update_x_hover(pos)
        # Handle velocity line dragging (grab anywhere, scale by 1/t)
        if self.dragging_vel_end is not None:
            if self.dragging_vel_end == "triangle":
                center = self.tri_center
            else:
                center = self.sq_center
            dx = pos.x() - center.x()
            dy = pos.y() - center.y()
            t = self.dragging_vel_t
            new_end = QPointF(center.x() + dx / t, center.y() + dy / t)
            if self.dragging_vel_end == "triangle":
                self.tri_velocity_end = new_end
                if self.tri_deleted:
                    self._refresh_deleted_orbit("triangle")
                if self.tri_orbit_mode:
                    self._compute_orbit_for_shape("triangle")
            else:
                self.sq_velocity_end = new_end
                if self.sq_deleted:
                    self._refresh_deleted_orbit("square")
                if self.sq_orbit_mode:
                    self._compute_orbit_for_shape("square")
            self._run_active_optimizer()
            self.update()
            return
        # Handle shape dragging
        if self.dragging_shape == "triangle":
            if self.tri_deleted:
                if self.s_held and self._slide_deleted_to("triangle", pos):
                    self._run_active_optimizer()
                    self.update()
                    return
            elif self.s_held and self._slide_shape_to("triangle", pos):
                self._run_active_optimizer()
                self.update()
                return
            new_center = pos + self.drag_offset
            delta = new_center - self.tri_center
            self.tri_center.setX(new_center.x())
            self.tri_center.setY(new_center.y())
            if self.tri_velocity_end is not None:
                self.tri_velocity_end = self.tri_velocity_end + delta
            if self.tri_deleted:
                self._refresh_deleted_orbit("triangle")
            if self.tri_orbit_mode:
                self._compute_orbit_for_shape("triangle")
            self._run_active_optimizer()
            self.update()
            return
        if self.dragging_shape == "square":
            if self.sq_deleted:
                if self.s_held and self._slide_deleted_to("square", pos):
                    self._run_active_optimizer()
                    self.update()
                    return
            elif self.s_held and self._slide_shape_to("square", pos):
                self._run_active_optimizer()
                self.update()
                return
            new_center = pos + self.drag_offset
            delta = new_center - self.sq_center
            self.sq_center.setX(new_center.x())
            self.sq_center.setY(new_center.y())
            if self.sq_velocity_end is not None:
                self.sq_velocity_end = self.sq_velocity_end + delta
            if self.sq_deleted:
                self._refresh_deleted_orbit("square")
            if self.sq_orbit_mode:
                self._compute_orbit_for_shape("square")
            self._run_active_optimizer()
            self.update()
            return
        # Handle black node dragging
        if self.dragging_dot is not None:
            self.dragging_dot.setX(pos.x())
            self.dragging_dot.setY(pos.y())
            self.update()
            return
        # Handle trajectory drawing
        if not self.dragging:
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.dragging = False
            return
        traj = self.trajectories[-1]
        last = traj["dots"][-1]
        pos = self._screen_to_world(QPointF(event.pos()))
        # Drop a new node every ~trace_pixel_spacing of on-screen arc length
        # along the interpolating arc about Earth. Choosing the angular
        # step from a screen-pixel target keeps draw-density independent of
        # zoom and orbital radius (otherwise a fixed Delta-theta gives
        # screen-arc = zoom * r * Delta-theta which blows up at large r and
        # high zoom, forcing the user to lift their finger).
        ex, ey = self.earth_center.x(), self.earth_center.y()
        rx0, ry0 = last.x() - ex, last.y() - ey
        rx1, ry1 = pos.x() - ex, pos.y() - ey
        r0 = math.hypot(rx0, ry0)
        r1 = math.hypot(rx1, ry1)
        if r0 > 1e-6 and r1 > 1e-6:
            a0 = math.atan2(ry0, rx0)
            a1 = math.atan2(ry1, rx1)
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            r_avg = 0.5 * (r0 + r1)
            # screen arc = zoom * r * dtheta  ->  dtheta = px / (zoom * r)
            step = self.trace_pixel_spacing / max(self.zoom * r_avg, 1e-6)
            if abs(da) >= step:
                num_dots = int(abs(da) // step)
                sgn = 1.0 if da >= 0 else -1.0
                for k in range(1, num_dots + 1):
                    a = a0 + sgn * step * k
                    r = r0 + (r1 - r0) * (k / max(num_dots, 1))
                    interp = QPointF(ex + r * math.cos(a), ey + r * math.sin(a))
                    # Append the dot FIRST so the segment never references
                    # an out-of-range index even transiently.
                    self.trajectories[-1]["dots"].append(interp)
                    n_dots = len(self.trajectories[-1]["dots"])
                    self.trajectories[-1]["segments"].append(
                        (n_dots - 2, n_dots - 1, 1.0, 0.0)
                    )
                self.update()
            return
        dx = pos.x() - last.x()
        dy = pos.y() - last.y()
        dist = math.hypot(dx, dy)
        # Linear-fallback spacing also keyed off screen pixels.
        spacing = self.trace_pixel_spacing / max(self.zoom, 1e-6)
        if dist >= spacing:
            num_dots = int(dist // spacing)
            ux = dx / dist
            uy = dy / dist
            for k in range(1, num_dots + 1):
                interp = QPointF(
                    last.x() + ux * spacing * k,
                    last.y() + uy * spacing * k,
                )
                self.trajectories[-1]["dots"].append(interp)
                n_dots = len(self.trajectories[-1]["dots"])
                self.trajectories[-1]["segments"].append(
                    (n_dots - 2, n_dots - 1, 1.0, 0.0)
                )
            self.update()

    def mouseDoubleClickEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            geometry_changed = False
            if self.dragging_vel_end:
                self.dragging_vel_end = None
                geometry_changed = True
            elif self.dragging_dot is not None:
                self.dragging_dot = None
                geometry_changed = True
            elif self.dragging_shape:
                self.dragging_shape = None
                geometry_changed = True
            elif self.dragging:
                self.dragging = False
                # Two-radius snap on release (mirror of press):
                #   inner (< 30 px): replace last free dot with the shape
                #   outer (30..75 px): append shape + connector segment
                #   beyond: leave as-is
                if self.trajectories:
                    traj = self.trajectories[-1]
                    if traj["segments"] and traj["dots"]:
                        last = traj["dots"][-1]
                        r_inner = 30.0 / self.zoom
                        r_outer = 75.0 / self.zoom
                        snap_node, _, snap_mode = self._snap_target_for(last, r_inner, r_outer)
                        if snap_node is not None and snap_node is not last:
                            i_last = len(traj["dots"]) - 1
                            if snap_mode == 'inner':
                                # Replace the last free dot with the shape;
                                # update any segments referencing i_last.
                                if snap_node in traj["dots"]:
                                    i_snap = traj["dots"].index(snap_node)
                                    new_segs = []
                                    for (i_s, i_e, m, _s) in traj["segments"]:
                                        if i_s == i_last:
                                            i_s = i_snap
                                        if i_e == i_last:
                                            i_e = i_snap
                                        if i_s != i_e:
                                            new_segs.append((i_s, i_e, m, _s))
                                    traj["segments"] = new_segs
                                    del traj["dots"][i_last]
                                else:
                                    traj["dots"][i_last] = snap_node
                                geometry_changed = True
                            else:  # 'outer'
                                if snap_node in traj["dots"]:
                                    i_snap = traj["dots"].index(snap_node)
                                else:
                                    traj["dots"].append(snap_node)
                                    i_snap = len(traj["dots"]) - 1
                                if not self._segment_exists(last, snap_node):
                                    traj["segments"].append((i_last, i_snap, 1.0, self._seed_side(last, snap_node)))
                                    geometry_changed = True
                            self.update()
            if geometry_changed:
                self._run_active_optimizer()

    def zoom_in(self):
        self.zoom *= 1.2
        self.update()

    def zoom_out(self):
        self.zoom /= 1.2
        self.update()

    def wheelEvent(self, event):
        delta = event.pixelDelta()
        if not delta.isNull():
            self.pan_offset = QPointF(
                self.pan_offset.x() - delta.x(),
                self.pan_offset.y() - delta.y(),
            )
            self.update()

    def _zoom_about(self, screen_pt, factor):
        """Multiply zoom by factor, keeping the world point under screen_pt fixed."""
        # World point under cursor before zoom: (sx - px)/z, (py - sy)/z.
        # After zoom z' = z*factor, we want same world point under same screen pt:
        #   px' = sx - (sx - px) * factor
        #   py' = sy + (py - sy) * factor   (y-flipped)
        sx, sy = screen_pt.x(), screen_pt.y()
        px, py = self.pan_offset.x(), self.pan_offset.y()
        self.pan_offset = QPointF(
            sx - (sx - px) * factor,
            sy + (py - sy) * factor,
        )
        self.zoom *= factor
        self.update()

    def event(self, ev):
        # macOS trackpad pinch is delivered as a NativeGesture (ZoomNativeGesture).
        if ev.type() == QEvent.Type.NativeGesture:
            if ev.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                # value() is incremental scale delta (e.g. +0.05 spread, -0.03 pinch).
                factor = 1.0 + ev.value()
                if factor > 0.01:
                    self._zoom_about(ev.position(), factor)
                return True
        return super().event(ev)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Apply pan and zoom (y-up convention: negative y-scale flips vertical)
        painter.translate(self.pan_offset.x(), self.pan_offset.y())
        painter.scale(self.zoom, -self.zoom)

        # Draw Kepler orbits as green ellipses only in inertial frame.
        # In rotating frame they're replaced by the time-parameterized
        # amber traces below.
        if self.env_mode == "two_body" and self.frame_mode == "inertial":
            pen = QPen(QColor("green"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for orbit in (self.tri_orbit, self.sq_orbit):
                if len(orbit) > 1:
                    for k in range(len(orbit) - 1):
                        painter.drawLine(orbit[k], orbit[k + 1])

        # Rotating-frame view of the triangle/square full orbits, time-parameterized.
        # Triangle: sample t_orb in [0, T_tri], absolute time = 0 + t_orb.
        # Square:   sample t_orb in [0, T_sq],  absolute time = tf + t_orb.
        if self.env_mode == "two_body" and self.frame_mode == "rotating":
            t_sq2 = self._square_time()
            specs = []
            # Triangle: use rendered (nu) state for deleted shapes so the
            # rotating-frame trace lines up with the drawn triangle and the
            # transfer arc's t=0 endpoint.
            if self.tri_deleted and self.tri_orbit_elements is not None:
                r_xy, v_xy = self._deleted_shape_pos_vel("triangle")
                if r_xy is not None:
                    specs.append((QPointF(float(r_xy[0]), float(r_xy[1])),
                                  np.array([v_xy[0], v_xy[1]]), 0.0))
            elif self.tri_velocity_end is not None:
                tri_vel = np.array([
                    self.tri_velocity_end.x() - self.tri_center.x(),
                    self.tri_velocity_end.y() - self.tri_center.y(),
                ]) / VEL_SCALE
                specs.append((self.tri_center, tri_vel, 0.0))
            if self.sq_deleted and self.sq_orbit_elements is not None:
                r_xy, v_xy = self._deleted_shape_pos_vel("square")
                if r_xy is not None:
                    specs.append((QPointF(float(r_xy[0]), float(r_xy[1])),
                                  np.array([v_xy[0], v_xy[1]]), t_sq2))
            elif self.sq_velocity_end is not None:
                sq_vel = np.array([
                    self.sq_velocity_end.x() - self.sq_center.x(),
                    self.sq_velocity_end.y() - self.sq_center.y(),
                ]) / VEL_SCALE
                specs.append((self.sq_center, sq_vel, t_sq2))
            for shape_center, vel, t_anchor in specs:
                times, pts = propagate_kepler_period(
                    shape_center, vel, self.earth_center, MU, 200,
                )
                if not pts:
                    continue
                pen = QPen(QColor("green"), 2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                rot_pts = [
                    self._rotate_point_about_earth(p, t_anchor + t)
                    for p, t in zip(pts, times)
                ]
                for k in range(len(rot_pts) - 1):
                    painter.drawLine(rot_pts[k], rot_pts[k + 1])

        # Draw velocity lines from triangle/square (triangle at t=0, square at t=tf).
        # Solid thick line for live shapes; hollow rectangle outline (matching
        # the hollow shape markers) for deleted shapes at their current nu.
        def _draw_shape_velocity(anchor_pt, v_disp, hollow):
            tip = QPointF(anchor_pt.x() + v_disp[0] * VEL_SCALE,
                          anchor_pt.y() + v_disp[1] * VEL_SCALE)
            if not hollow:
                pen_l = QPen(QColor("green"), 8)
                pen_l.setCosmetic(True)
                painter.setPen(pen_l)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(anchor_pt, tip)
                return
            dx = tip.x() - anchor_pt.x()
            dy = tip.y() - anchor_pt.y()
            L = math.hypot(dx, dy)
            if L < 1e-9:
                return
            # Perpendicular offset = half of the would-be 8 px stroke,
            # converted from screen pixels to world units.
            half_w = 4.0 / self.zoom
            px, py = -dy / L * half_w, dx / L * half_w
            poly = QPolygonF([
                QPointF(anchor_pt.x() + px, anchor_pt.y() + py),
                QPointF(tip.x() + px, tip.y() + py),
                QPointF(tip.x() - px, tip.y() - py),
                QPointF(anchor_pt.x() - px, anchor_pt.y() - py),
            ])
            pen_l = QPen(QColor("green"), 2)
            pen_l.setCosmetic(True)
            painter.setPen(pen_l)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(poly)

        if self.tri_velocity_end is not None and not self.tri_deleted:
            v_in = np.array([self.tri_velocity_end.x() - self.tri_center.x(),
                             self.tri_velocity_end.y() - self.tri_center.y()]) / VEL_SCALE
            v_disp = self._rotating_velocity(v_in, self.tri_center, 0.0)
            _draw_shape_velocity(self.tri_center, v_disp, hollow=False)
        elif self.tri_deleted and self.tri_orbit_elements is not None:
            r_xy, v_xy = self._orbit_pos_vel(self.tri_orbit_elements, self.tri_nu)
            anchor_pt = self._rotate_point_about_earth(QPointF(r_xy[0], r_xy[1]), 0.0)
            v_disp = self._rotating_velocity(v_xy, QPointF(r_xy[0], r_xy[1]), 0.0)
            _draw_shape_velocity(anchor_pt, v_disp, hollow=True)
        if self.sq_velocity_end is not None and not self.sq_deleted:
            v_in = np.array([self.sq_velocity_end.x() - self.sq_center.x(),
                             self.sq_velocity_end.y() - self.sq_center.y()]) / VEL_SCALE
            t_sq = self._square_time()
            v_disp = self._rotating_velocity(v_in, self.sq_center, t_sq)
            anchor = self._rotate_point_about_earth(self.sq_center, t_sq)
            _draw_shape_velocity(anchor, v_disp, hollow=False)
        elif self.sq_deleted and self.sq_orbit_elements is not None:
            t_sq = self._square_time()
            r_xy, v_xy = self._orbit_pos_vel(self.sq_orbit_elements, self.sq_nu)
            anchor_pt = self._rotate_point_about_earth(QPointF(r_xy[0], r_xy[1]), t_sq)
            v_disp = self._rotating_velocity(v_xy, QPointF(r_xy[0], r_xy[1]), t_sq)
            _draw_shape_velocity(anchor_pt, v_disp, hollow=True)

        # Draw blue circle (Earth) only in two-body env
        if self.env_mode == "two_body":
            painter.setBrush(QBrush(QColor("blue")))
            pen = QPen(QColor("blue"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawEllipse(self.earth_center, self.earth_radius, self.earth_radius)
            # Moon: massless visual node, propagated along its circular
            # orbit by tf = sum of every segment's flight time (same clock
            # the ghost square uses).
            tf_moon = 0.0
            for traj in self.trajectories:
                dots = traj["dots"]
                for i_start, i_end, mult, _side in traj["segments"]:
                    tf_moon += self._seg_tof(dots[i_start], dots[i_end], mult)
            if tf_moon > 0.0 and self.moon_orbit_elements is not None:
                nu_m = self._orbit_nu_after_dt(self.moon_orbit_elements, tf_moon)
                r_m, _ = self._orbit_pos_vel(self.moon_orbit_elements, nu_m)
                moon_pt = self._rotate_point_about_earth(
                    QPointF(float(r_m[0]), float(r_m[1])), tf_moon,
                )
            else:
                moon_pt = self.moon_center
            painter.setBrush(QBrush(QColor(160, 160, 160)))
            pen = QPen(QColor(160, 160, 160), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawEllipse(moon_pt, self.moon_radius, self.moon_radius)

        # Draw trajectories
        center = self.earth_center
        for traj in self.trajectories:
            dots = traj["dots"]
            segments = traj["segments"]
            node_t = self._traj_node_times(traj)

            # Draw segments
            for i_start, i_end, mult, _side in segments:
                pen = QPen(QColor("black"), 2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                seg_tof = self._seg_tof(dots[i_start], dots[i_end], mult)
                arc_points = self._compute_segment_arc(
                    dots[i_start], dots[i_end], center,
                    seg_tof, ARC_NUM_POINTS,
                    side=_side,
                )
                t_start = node_t.get(i_start, 0.0)
                if self.frame_mode == "rotating" and len(arc_points) > 1:
                    n = len(arc_points)
                    rot_pts = [
                        self._rotate_point_about_earth(
                            arc_points[j], t_start + (j / (n - 1)) * seg_tof
                        )
                        for j in range(n)
                    ]
                else:
                    rot_pts = arc_points
                for j in range(len(rot_pts) - 1):
                    painter.drawLine(rot_pts[j], rot_pts[j + 1])

        # Draw delta-velocity vectors at nodes with exactly 2 neighboring
        # segments, plus shape-attached dvs at the triangle/square. Collected
        # in two passes so we can scale the largest |dv| to the Earth radius
        # (dvs are typically much smaller than orbital velocities — using the
        # same VEL_SCALE as the green vectors makes them invisible).
        dv_arrows = []  # list of (anchor_QPointF, dv_disp_xy_array)
        for traj in self.trajectories:
            dots = traj["dots"]
            segments = traj["segments"]
            node_t = self._traj_node_times(traj)
            for k, dot in enumerate(dots):
                if dot is self.tri_center or dot is self.sq_center:
                    continue
                # Find segments where this node is an endpoint
                incoming_vel = None  # arrival velocity at node
                outgoing_vel = None  # departure velocity from node
                for i_start, i_end, mult, _side in segments:
                    if i_end == k:
                        _, vf = self._compute_segment_velocities(
                            dots[i_start], dots[i_end], center, self._seg_tof(dots[i_start], dots[i_end], mult))
                        incoming_vel = vf
                    elif i_start == k:
                        v0, _ = self._compute_segment_velocities(
                            dots[i_start], dots[i_end], center, self._seg_tof(dots[i_start], dots[i_end], mult))
                        outgoing_vel = v0
                if incoming_vel is not None and outgoing_vel is not None:
                    dv = outgoing_vel - incoming_vel
                    t_node = node_t.get(k, 0.0)
                    dv_disp = self._rotate_vec(dv, t_node)
                    anchor = self._rotate_point_about_earth(dot, t_node)
                    dv_arrows.append((anchor, dv_disp))

        # Shape-attached dvs (triangle/square endpoints).
        for shape_center, vel_end in [
            (self.tri_center, self.tri_velocity_end),
            (self.sq_center, self.sq_velocity_end),
        ]:
            if vel_end is None:
                continue
            shape_vel = np.array([
                vel_end.x() - shape_center.x(),
                vel_end.y() - shape_center.y(),
            ]) / VEL_SCALE
            for traj in self.trajectories:
                dots = traj["dots"]
                node_t = self._traj_node_times(traj)
                for i_start, i_end, mult, _side in traj["segments"]:
                    seg_tof = self._seg_tof(dots[i_start], dots[i_end], mult)
                    if dots[i_start] is shape_center:
                        v0, _ = self._compute_segment_velocities(
                            dots[i_start], dots[i_end], center, seg_tof)
                        dv = v0 - shape_vel
                        t_anchor = node_t.get(i_start, 0.0)
                    elif dots[i_end] is shape_center:
                        _, vf = self._compute_segment_velocities(
                            dots[i_start], dots[i_end], center, seg_tof)
                        dv = shape_vel - vf
                        t_anchor = node_t.get(i_end, 0.0)
                    else:
                        continue
                    anchor = self._rotate_point_about_earth(shape_center, t_anchor)
                    dv_disp = self._rotate_vec(dv, t_anchor)
                    dv_arrows.append((anchor, dv_disp))

        if dv_arrows:
            dv_scale = self.current_dv_scale()
            pen = QPen(QColor("black"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for anchor, dv_disp in dv_arrows:
                painter.drawLine(
                    anchor,
                    QPointF(anchor.x() + dv_disp[0] * dv_scale,
                            anchor.y() + dv_disp[1] * dv_scale),
                )

        # Draw green triangle and square (square at t=tf in rotating frame)
        painter.setBrush(QBrush(QColor("green")))
        painter.setPen(Qt.PenStyle.NoPen)
        if not self.tri_deleted:
            painter.drawPolygon(self._triangle_polygon())
        if not self.sq_deleted:
            sq_anchor = self._rotate_point_about_earth(self.sq_center, self._square_time())
            sq_x = sq_anchor.x() - self.sq_size / 2
            sq_y = sq_anchor.y() - self.sq_size / 2
            painter.drawRect(QRectF(sq_x, sq_y, self.sq_size, self.sq_size))
        # Hollow markers at the current nu for deleted shapes (orbit rendezvous).
        if self.tri_deleted and self.tri_orbit_elements is not None:
            r_xy, _ = self._orbit_pos_vel(self.tri_orbit_elements, self.tri_nu)
            anchor = self._rotate_point_about_earth(QPointF(r_xy[0], r_xy[1]), 0.0)
            pen = QPen(QColor("green"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            s = self.tri_size
            painter.drawPolygon(QPolygonF([
                QPointF(anchor.x() - s / 2, anchor.y() - s / 2),
                QPointF(anchor.x() + s / 2, anchor.y()),
                QPointF(anchor.x() - s / 2, anchor.y() + s / 2),
            ]))
        if self.sq_deleted and self.sq_orbit_elements is not None:
            r_xy, _ = self._orbit_pos_vel(self.sq_orbit_elements, self.sq_nu)
            anchor = self._rotate_point_about_earth(QPointF(r_xy[0], r_xy[1]), self._square_time())
            pen = QPen(QColor("green"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            half = self.sq_size * 0.5
            painter.drawRect(QRectF(anchor.x() - half, anchor.y() - half, 2 * half, 2 * half))

        # Ghost square: forward-propagate the square's current state along
        # its Kepler orbit by tf = sum of every segment's flight time across
        # all trajectories (not just those terminating at the square), so the
        # ghost advances every time a new segment is added. Drawn as a hollow
        # grey square 2x the live square's size.
        if self.env_mode == "two_body":
            tf = 0.0
            for traj in self.trajectories:
                dots = traj["dots"]
                for i_start, i_end, mult, _side in traj["segments"]:
                    tf += self._seg_tof(dots[i_start], dots[i_end], mult)
            elements = None
            if self.sq_deleted and self.sq_orbit_elements is not None:
                elements = self.sq_orbit_elements
            elif (not self.sq_deleted) and self.sq_velocity_end is not None:
                v_xy = np.array([
                    self.sq_velocity_end.x() - self.sq_center.x(),
                    self.sq_velocity_end.y() - self.sq_center.y(),
                ]) / VEL_SCALE
                elements = self._orbit_elements_at(self.sq_center, v_xy)
            if elements is not None and tf > 0.0:
                nu_g = self._orbit_nu_after_dt(elements, tf)
                r_g, _ = self._orbit_pos_vel(elements, nu_g)
                anchor_g = self._rotate_point_about_earth(
                    QPointF(float(r_g[0]), float(r_g[1])), tf,
                )
                pen = QPen(QColor(160, 160, 160), 2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                half = self.sq_size  # 2x the live square's half-extent
                painter.drawRect(QRectF(
                    anchor_g.x() - half, anchor_g.y() - half, 2 * half, 2 * half,
                ))

        # Draw all black dots on top of everything (including shapes)
        painter.setBrush(QBrush(QColor("black")))
        painter.setPen(Qt.PenStyle.NoPen)
        dot_r = self.dot_radius_px / self.zoom
        for traj in self.trajectories:
            node_t = self._traj_node_times(traj)
            for k, dot in enumerate(traj["dots"]):
                if dot is self.tri_center or dot is self.sq_center:
                    continue
                p = self._rotate_point_about_earth(dot, node_t.get(k, 0.0))
                painter.drawEllipse(p, dot_r, dot_r)

        # X-hover highlight: show what an X-click would delete.
        if self.x_held and self.x_hover is not None:
            hi_color = QColor(255, 220, 60)
            kind = self.x_hover[0]
            if kind == "triangle":
                pen = QPen(hi_color, 3)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolygon(self._triangle_polygon())
            elif kind == "square":
                pen = QPen(hi_color, 3)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                half = self.sq_size * 0.5
                painter.drawRect(QRectF(
                    self.sq_center.x() - half, self.sq_center.y() - half,
                    2 * half, 2 * half,
                ))
            elif kind == "segment":
                _, traj, s_idx = self.x_hover
                segs = traj["segments"]
                if 0 <= s_idx < len(segs):
                    i_start, i_end, mult, _side = segs[s_idx]
                    dots = traj["dots"]
                    seg_tof = self._seg_tof(dots[i_start], dots[i_end], mult)
                    arc_points = self._compute_segment_arc(
                        dots[i_start], dots[i_end], self.earth_center,
                        seg_tof, ARC_NUM_POINTS,
                        side=_side,
                    )
                    node_t = self._traj_node_times(traj)
                    t_start = node_t.get(i_start, 0.0)
                    if self.frame_mode == "rotating" and len(arc_points) > 1:
                        n = len(arc_points)
                        rot_pts = [
                            self._rotate_point_about_earth(
                                arc_points[j], t_start + (j / (n - 1)) * seg_tof
                            )
                            for j in range(n)
                        ]
                    else:
                        rot_pts = arc_points
                    pen = QPen(hi_color, 4)
                    pen.setCosmetic(True)
                    painter.setPen(pen)
                    for j in range(len(rot_pts) - 1):
                        painter.drawLine(rot_pts[j], rot_pts[j + 1])
            elif kind == "node":
                _, traj, dot = self.x_hover
                t_node = 0.0
                if traj is not None:
                    node_t = self._traj_node_times(traj)
                    for k, d in enumerate(traj["dots"]):
                        if d is dot:
                            t_node = node_t.get(k, 0.0)
                            break
                p = self._rotate_point_about_earth(dot, t_node)
                pen = QPen(hi_color, 3)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                r = (self.dot_radius_px * 2.5) / self.zoom
                painter.drawEllipse(p, r, r)

        # Bottom-left axes overlay: inertial (t=0) vs rotating-frame at t=tf.
        # Drawn in screen pixels so zoom/pan don't affect it. The frame the
        # observer is in stays fixed; the other frame rotates.
        #   inertial mode:  xhato fixed,         xhatf = R(+omega*tf) xhat
        #   rotating mode:  xhatf fixed at tf,   xhato = R(-omega*tf) xhat
        painter.resetTransform()
        ox, oy = 60, self.height() - 60
        L = 40  # axis length in pixels
        t_sq = self._square_time()
        # Small title above the axes labeling the *other* frame (the one that
        # rotates relative to the current view).
        title = "rotating frame" if self.frame_mode == "inertial" else "inertial frame"
        title_font = painter.font()
        prev_size = title_font.pointSize()
        title_font.setPointSize(11)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(180, 180, 180)))
        painter.drawText(QPointF(ox - L / 2, oy - L - 18), title)
        title_font.setPointSize(prev_size if prev_size > 0 else 10)
        painter.setFont(title_font)
        if self.frame_mode == "inertial":
            ang_o = 0.0
            ang_f = OMEGA_ROT * t_sq
        else:
            ang_o = -OMEGA_ROT * t_sq
            ang_f = 0.0
        # Inertial axes (xhato, yhato): white-ish
        co, so = math.cos(ang_o), math.sin(ang_o)
        pen = QPen(QColor(220, 220, 220), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        xo_tip = QPointF(ox + L * co, oy - L * so)
        yo_tip = QPointF(ox - L * so, oy - L * co)
        painter.drawLine(QPointF(ox, oy), xo_tip)
        painter.drawLine(QPointF(ox, oy), yo_tip)
        painter.drawText(QPointF(xo_tip.x() + 4, xo_tip.y() + 4), "xhato")
        painter.drawText(QPointF(yo_tip.x() + 4, yo_tip.y() - 2), "yhato")
        # Rotating-frame axes at t=tf (xhatf, yhatf): blue
        cf, sf = math.cos(ang_f), math.sin(ang_f)
        pen = QPen(QColor(120, 200, 255), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        xf_tip = QPointF(ox + L * cf, oy - L * sf)
        yf_tip = QPointF(ox - L * sf, oy - L * cf)
        painter.drawLine(QPointF(ox, oy), xf_tip)
        painter.drawLine(QPointF(ox, oy), yf_tip)
        painter.drawText(QPointF(xf_tip.x() + 4, xf_tip.y() + 4), "xhatf")
        painter.drawText(QPointF(yf_tip.x() + 4, yf_tip.y() - 2), "yhatf")

        painter.end()


    def clear(self):
        self.trajectories.clear()
        self.shift_click_first = None
        self.update()

    def _optimize_common(self, cost_mode, apply=True, progress_callback=None):
        """Shared optimizer logic. cost_mode='energy' uses sum(|dv|^2), 'fuel' uses sum(|dv|).

        If apply=False, the solve runs without mutating self; instead it
        returns (new_x, n_pos_vars, movable_dots) which the caller passes to
        `_optimize_common_apply` on the Qt main thread. This variant is safe
        to call from a worker thread.

        If progress_callback is provided, it is invoked as
        `progress_callback(xk, n_pos_vars, movable_dots)` on every BFGS
        iteration (scipy's own callback). Safe to call from a worker thread
        if the callback only emits signals.
        """
        # Collect all movable (black) dots that participate in at least one segment
        center_np = np.array([self.earth_center.x(), self.earth_center.y()])
        movable_dots = []  # list of QPointF references
        movable_set = set()  # ids to avoid duplicates

        for traj in self.trajectories:
            dots = traj["dots"]
            for i_start, i_end, _mult, _side in traj["segments"]:
                for idx in (i_start, i_end):
                    dot = dots[idx]
                    if dot is not self.tri_center and dot is not self.sq_center:
                        if id(dot) not in movable_set:
                            movable_set.add(id(dot))
                            movable_dots.append(dot)

        if not movable_dots:
            return

        # Map dot id -> index into decision variable array
        dot_id_to_var_idx = {id(d): i for i, d in enumerate(movable_dots)}
        n_pos_vars = len(movable_dots) * 2  # x, y per dot

        # Build segment list as (dot_start, dot_end, mult) with references
        all_segments = []
        for traj in self.trajectories:
            dots = traj["dots"]
            for i_start, i_end, mult, _side in traj["segments"]:
                all_segments.append((dots[i_start], dots[i_end], mult, _side))

        n_vars_orig = n_pos_vars + 1  # positions + single shared TOF (no nu yet)

        # Fixed shape data. Items are mutable lists so that orbit-rendezvous
        # mode can override shape_pos / shape_vel in place between solver
        # iterations (free shapes only).
        shape_data = []
        for shape_center, vel_end in [
            (self.tri_center, self.tri_velocity_end),
            (self.sq_center, self.sq_velocity_end),
        ]:
            if vel_end is not None:
                shape_vel = np.array([
                    vel_end.x() - shape_center.x(),
                    vel_end.y() - shape_center.y(),
                ]) / VEL_SCALE
                shape_pos = np.array([shape_center.x(), shape_center.y()])
                shape_data.append([shape_center, shape_pos, shape_vel])

        # Map dots to a fixed numpy address for fast lookup: movable dots come
        # from decision vars; shape dots come from shape_data; any other
        # (fixed) dot is rarely encountered but we still handle it.
        shape_pos_by_id = {id(s[0]): s[1] for s in shape_data}

        def get_pos(dot, x_vec):
            idx = dot_id_to_var_idx.get(id(dot))
            if idx is not None:
                return x_vec[2 * idx: 2 * idx + 2]
            shp = shape_pos_by_id.get(id(dot))
            if shp is not None:
                return shp
            return np.array([dot.x(), dot.y()])

        # Precompute first-occurrence maps (identical for objective and grad).
        # They depend only on segment ordering, not on x_vec.
        node_outgoing_seg = {}
        node_incoming_seg = {}
        for i, (dot_s, dot_e, _m, _side) in enumerate(all_segments):
            s_id = id(dot_s)
            e_id = id(dot_e)
            if s_id not in node_outgoing_seg:
                node_outgoing_seg[s_id] = i
            if e_id not in node_incoming_seg:
                node_incoming_seg[e_id] = i

        # Orbit-rendezvous "free shapes": deleted shapes that participate in
        # at least one segment. Their boundary position+velocity become a
        # function of a single decision variable (true anomaly nu) on the
        # cached orbit.
        free_shapes = []
        for shape_name, deleted, elements, nu_init, shape_center in [
            ("triangle", self.tri_deleted, self.tri_orbit_elements, self.tri_nu, self.tri_center),
            ("square", self.sq_deleted, self.sq_orbit_elements, self.sq_nu, self.sq_center),
        ]:
            if not deleted or elements is None:
                continue
            s_id = id(shape_center)
            if s_id not in node_outgoing_seg and s_id not in node_incoming_seg:
                continue
            sd_idx = None
            for k, sd in enumerate(shape_data):
                if sd[0] is shape_center:
                    sd_idx = k
                    break
            if sd_idx is None:
                continue
            free_shapes.append({
                "shape": shape_name,
                "shape_center": shape_center,
                "elements": elements,
                "nu_init": float(nu_init),
                "sd_idx": sd_idx,
                "dv_vel_rows": [],  # populated when numba arrays are built
                "nu_idx": None,     # filled in below
            })

        n_free = len(free_shapes)
        for i, fs in enumerate(free_shapes):
            fs["nu_idx"] = n_vars_orig + i
        n_vars = n_vars_orig + n_free  # positions + tof + one nu per free shape

        # Mass-leak smoothing so |dv| is differentiable at zero for gradient solvers.
        fuel_eps = 1e-4
        energy_mode = (cost_mode == "energy")

        # Mode selection
        cg_mode = self.env_mode == "constant_gravity"
        use_parabola = cg_mode
        const_g_vec = self._constant_g_vec() if cg_mode else None
        g_mag = GRAVITY_MAG
        I2 = np.eye(2)

        # Warm-start cache for Lambert z, keyed by segment index.
        z_cache = [0.0] * len(all_segments)

        def seg_parabola(r0, rf, tof_seg, mult):
            """Return (v0, vf, J_v0_r0, J_v0_rf, J_v0_tf*mult, J_vf_r0, J_vf_rf, J_vf_tf*mult)
            for the parabolic segment. J_*_tf already multiplied by mult."""
            if cg_mode:
                g_vec = const_g_vec
                inv_t = 1.0 / tof_seg
                disp = rf - r0
                mean_v = disp * inv_t
                half_g_t = 0.5 * g_vec * tof_seg
                v0 = mean_v - half_g_t
                vf = mean_v + half_g_t
                I_t = I2 * inv_t
                J_v0_r0 = -I_t
                J_v0_rf = I_t
                J_v0_tf_seg = -disp * (inv_t * inv_t) - 0.5 * g_vec
                J_vf_r0 = -I_t
                J_vf_rf = I_t
                J_vf_tf_seg = -disp * (inv_t * inv_t) + 0.5 * g_vec
            else:
                d = center_np - r0
                dist = math.sqrt(d[0] * d[0] + d[1] * d[1])
                inv_t = 1.0 / tof_seg
                disp = rf - r0
                mean_v = disp * inv_t
                if dist < 1e-12:
                    v0 = mean_v
                    vf = mean_v
                    I_t = I2 * inv_t
                    J_v0_r0 = -I_t
                    J_v0_rf = I_t
                    J_v0_tf_seg = -disp * (inv_t * inv_t)
                    J_vf_r0 = J_v0_r0
                    J_vf_rf = J_v0_rf
                    J_vf_tf_seg = J_v0_tf_seg
                else:
                    g_hat = d / dist
                    g_vec = g_mag * g_hat
                    half_g_t = 0.5 * g_vec * tof_seg
                    v0 = mean_v - half_g_t
                    vf = mean_v + half_g_t
                    dgvec_dr0 = (g_mag / dist) * (-I2 + np.outer(g_hat, g_hat))
                    I_t = I2 * inv_t
                    J_v0_r0 = -I_t - 0.5 * tof_seg * dgvec_dr0
                    J_v0_rf = I_t
                    J_v0_tf_seg = -disp * (inv_t * inv_t) - 0.5 * g_vec
                    J_vf_r0 = -I_t + 0.5 * tof_seg * dgvec_dr0
                    J_vf_rf = I_t
                    J_vf_tf_seg = -disp * (inv_t * inv_t) + 0.5 * g_vec
            return (v0, vf, J_v0_r0, J_v0_rf, J_v0_tf_seg * mult,
                    J_vf_r0, J_vf_rf, J_vf_tf_seg * mult)

        def seg_lambert(r0, rf, tof_seg, mult, z_init, side):
            """Direct numpy call, no QPointF. Returns same tuple as seg_parabola
            plus the converged z for warm-starting."""
            v0, vf, Jv0_r0, Jv0_rf, Jv0_dt, Jvf_r0, Jvf_rf, Jvf_dt, z_new = (
                lambert_solve_with_jac(r0 - center_np, rf - center_np, tof_seg,
                                       MU, z_init=z_init, side=side)
            )
            return (v0, vf, Jv0_r0, Jv0_rf, Jv0_dt * mult,
                    Jvf_r0, Jvf_rf, Jvf_dt * mult, z_new)

        def add_pos_grad(grad, dot, vec2):
            idx = dot_id_to_var_idx.get(id(dot))
            if idx is not None:
                grad[2 * idx] += vec2[0]
                grad[2 * idx + 1] += vec2[1]

        def fun_and_grad(x_vec):
            tof = x_vec[n_pos_vars]
            grad = np.zeros(n_vars)
            cost = 0.0

            # Per-segment velocities + Jacobians (one pass).
            seg_data_local = [None] * len(all_segments)
            for i, (dot_s, dot_e, mult, side) in enumerate(all_segments):
                r0 = get_pos(dot_s, x_vec)
                rf = get_pos(dot_e, x_vec)
                # Sundman-style time scaling (two-body only; off in CG).
                if cg_mode:
                    tof_seg = tof * mult
                else:
                    d0 = r0 - center_np
                    df = rf - center_np
                    r0_mag = math.sqrt(d0[0] * d0[0] + d0[1] * d0[1])
                    rf_mag = math.sqrt(df[0] * df[0] + df[1] * df[1])
                    s = 0.5 * (r0_mag + rf_mag)
                    if s < 1e-12:
                        s = 1e-12
                    sqrt_s = math.sqrt(s)
                    s_pow = s * sqrt_s
                    tof_seg = tof * s_pow * mult
                if use_parabola:
                    out = seg_parabola(r0, rf, tof_seg, mult)
                else:
                    out = seg_lambert(r0, rf, tof_seg, mult, z_cache[i], side)
                    z_cache[i] = out[-1]
                    out = out[:-1]
                # out = (v0, vf, Jv0_r0, Jv0_rf, Jv0_dt, Jvf_r0, Jvf_rf, Jvf_dt)
                v0_, vf_, Jv0_r0, Jv0_rf, Jv0_dt, Jvf_r0, Jvf_rf, Jvf_dt = out
                if not cg_mode:
                    # Apply Sundman chain-rule fixup (mirrors _apply_sundman_fixup in numba).
                    fac = 0.75 * tof * sqrt_s
                    if r0_mag > 1e-10:
                        r0_hat = d0 / r0_mag
                        Jv0_r0 = Jv0_r0 + fac * np.outer(Jv0_dt, r0_hat)
                        Jvf_r0 = Jvf_r0 + fac * np.outer(Jvf_dt, r0_hat)
                    if rf_mag > 1e-10:
                        rf_hat = df / rf_mag
                        Jv0_rf = Jv0_rf + fac * np.outer(Jv0_dt, rf_hat)
                        Jvf_rf = Jvf_rf + fac * np.outer(Jvf_dt, rf_hat)
                    Jv0_dt = Jv0_dt * s_pow
                    Jvf_dt = Jvf_dt * s_pow
                seg_data_local[i] = (
                    dot_s, dot_e, v0_, vf_,
                    Jv0_r0, Jv0_rf, Jv0_dt,
                    Jvf_r0, Jvf_rf, Jvf_dt,
                )

            # Accumulate cost and gradient contributions.
            def accumulate(dv, contribs):
                nonlocal cost
                d2 = dv[0] * dv[0] + dv[1] * dv[1]
                if energy_mode:
                    cost += d2
                    factor = 2.0
                else:
                    mag = math.sqrt(d2 + fuel_eps * fuel_eps)
                    cost += mag
                    factor = 1.0 / mag
                for kind, dot, J in contribs:
                    if kind == "pos":
                        add_pos_grad(grad, dot, factor * (dv @ J))
                    else:
                        grad[n_pos_vars] += factor * float(np.dot(dv, J))

            # Black movable midpoint nodes
            for dot in movable_dots:
                d_id = id(dot)
                i_in = node_incoming_seg.get(d_id)
                i_out = node_outgoing_seg.get(d_id)
                if i_in is None or i_out is None:
                    continue
                sin_s = seg_data_local[i_in]
                sout = seg_data_local[i_out]
                # tuple layout: (dot_s, dot_e, v0, vf, Jv0_r0, Jv0_rf, Jv0_tf, Jvf_r0, Jvf_rf, Jvf_tf)
                sin_vf = sin_s[3]
                sout_v0 = sout[2]
                dv = sout_v0 - sin_vf
                contribs = [
                    ("pos", sin_s[0], -sin_s[7]),  # -J_vf_r0 at upstream start
                    ("pos", dot, sout[4] - sin_s[8]),  # J_v0_r0 (this seg's start) - J_vf_rf (prev seg's end)
                    ("pos", sout[1], sout[5]),  # J_v0_rf at downstream end
                    ("tof", None, sout[6] - sin_s[9]),
                ]
                accumulate(dv, contribs)

            # Shape nodes
            for shape_center, shape_pos, shape_vel in shape_data:
                s_id = id(shape_center)
                i_out = node_outgoing_seg.get(s_id)
                if i_out is not None:
                    sout = seg_data_local[i_out]
                    dv = sout[2] - shape_vel
                    contribs = [
                        ("pos", sout[1], sout[5]),
                        ("tof", None, sout[6]),
                    ]
                    accumulate(dv, contribs)
                i_in = node_incoming_seg.get(s_id)
                if i_in is not None:
                    sin_s = seg_data_local[i_in]
                    dv = shape_vel - sin_s[3]
                    contribs = [
                        ("pos", sin_s[0], -sin_s[7]),
                        ("tof", None, -sin_s[9]),
                    ]
                    accumulate(dv, contribs)

            return cost, grad

        # --- Batched numba path -----------------------------------------------
        # When numba is available, build per-segment metadata arrays once,
        # preallocate all work buffers, and replace `fun_and_grad` with a
        # single njit call that does both velocity solve and dv-node
        # accumulation. Falls back to the Python `fun_and_grad` above if any
        # Lambert Newton iteration reports failure (rare; handled via
        # ok_out).
        fun_and_grad_active = fun_and_grad
        if _NUMBA_AVAILABLE:
            N = len(all_segments)

            # Resolve segment start/end kinds and variable indices.
            _KIND_MOV = 0
            _KIND_TRI = 1
            _KIND_SQ = 2
            _KIND_FIX = 3

            def _endpoint_meta(dot):
                idx = dot_id_to_var_idx.get(id(dot))
                if idx is not None:
                    return _KIND_MOV, idx, 0.0, 0.0
                if dot is self.tri_center:
                    return _KIND_TRI, -1, 0.0, 0.0
                if dot is self.sq_center:
                    return _KIND_SQ, -1, 0.0, 0.0
                return _KIND_FIX, -1, dot.x(), dot.y()

            seg_start_kind = np.empty(N, dtype=np.int64)
            seg_end_kind = np.empty(N, dtype=np.int64)
            seg_start_var_idx = np.empty(N, dtype=np.int64)
            seg_end_var_idx = np.empty(N, dtype=np.int64)
            seg_start_fixed = np.zeros((N, 2))
            seg_end_fixed = np.zeros((N, 2))
            seg_mult_arr = np.empty(N)
            seg_side_arr = np.empty(N)
            for i, (dot_s, dot_e, mult, side) in enumerate(all_segments):
                ks, vs, fsx, fsy = _endpoint_meta(dot_s)
                ke, ve, fex, fey = _endpoint_meta(dot_e)
                seg_start_kind[i] = ks
                seg_end_kind[i] = ke
                seg_start_var_idx[i] = vs
                seg_end_var_idx[i] = ve
                seg_start_fixed[i, 0] = fsx
                seg_start_fixed[i, 1] = fsy
                seg_end_fixed[i, 0] = fex
                seg_end_fixed[i, 1] = fey
                seg_mult_arr[i] = mult
                seg_side_arr[i] = side

            # Build dv-node metadata, mirroring the Python loops.
            dv_type_list = []
            dv_in_seg_list = []
            dv_out_seg_list = []
            dv_center_var_idx_list = []
            dv_shape_vel_list = []
            shape_dv_rows = {}  # id(shape_center) -> [row_idx, ...]

            for dot in movable_dots:
                d_id = id(dot)
                i_in = node_incoming_seg.get(d_id)
                i_out = node_outgoing_seg.get(d_id)
                if i_in is None or i_out is None:
                    continue
                dv_type_list.append(0)
                dv_in_seg_list.append(i_in)
                dv_out_seg_list.append(i_out)
                dv_center_var_idx_list.append(dot_id_to_var_idx[d_id])
                dv_shape_vel_list.append((0.0, 0.0))

            for shape_center, shape_pos, shape_vel in shape_data:
                s_id = id(shape_center)
                i_out = node_outgoing_seg.get(s_id)
                if i_out is not None:
                    shape_dv_rows.setdefault(s_id, []).append(len(dv_type_list))
                    dv_type_list.append(1)
                    dv_in_seg_list.append(-1)
                    dv_out_seg_list.append(i_out)
                    dv_center_var_idx_list.append(-1)
                    dv_shape_vel_list.append((float(shape_vel[0]), float(shape_vel[1])))
                i_in = node_incoming_seg.get(s_id)
                if i_in is not None:
                    shape_dv_rows.setdefault(s_id, []).append(len(dv_type_list))
                    dv_type_list.append(2)
                    dv_in_seg_list.append(i_in)
                    dv_out_seg_list.append(-1)
                    dv_center_var_idx_list.append(-1)
                    dv_shape_vel_list.append((float(shape_vel[0]), float(shape_vel[1])))

            dv_type_arr = np.array(dv_type_list, dtype=np.int64)
            dv_in_seg_arr = np.array(dv_in_seg_list, dtype=np.int64) if dv_in_seg_list else np.zeros(0, dtype=np.int64)
            dv_out_seg_arr = np.array(dv_out_seg_list, dtype=np.int64) if dv_out_seg_list else np.zeros(0, dtype=np.int64)
            dv_center_var_idx_arr = np.array(dv_center_var_idx_list, dtype=np.int64) if dv_center_var_idx_list else np.zeros(0, dtype=np.int64)
            dv_shape_vel_arr = np.array(dv_shape_vel_list) if dv_shape_vel_list else np.zeros((0, 2))

            shape_pos_tri = shape_pos_by_id.get(id(self.tri_center), np.array([self.tri_center.x(), self.tri_center.y()]))
            shape_pos_sq = shape_pos_by_id.get(id(self.sq_center), np.array([self.sq_center.x(), self.sq_center.y()]))

            const_g_vec_arr = const_g_vec if const_g_vec is not None else np.zeros(2)

            # Preallocate work buffers (shared across all BFGS iterations).
            v0_buf = np.empty((N, 2))
            vf_buf = np.empty((N, 2))
            Jv0_r0_buf = np.empty((N, 2, 2))
            Jv0_rf_buf = np.empty((N, 2, 2))
            Jv0_dt_buf = np.empty((N, 2))
            Jvf_r0_buf = np.empty((N, 2, 2))
            Jvf_rf_buf = np.empty((N, 2, 2))
            Jvf_dt_buf = np.empty((N, 2))
            z_cache_arr = np.zeros(N)
            grad_out = np.empty(n_vars)
            ok_out = np.empty(N, dtype=np.int64)

            def fun_and_grad_numba(x_vec):
                cost = _lnb.fun_and_grad_batch(
                    x_vec, n_pos_vars,
                    seg_start_kind, seg_end_kind,
                    seg_start_var_idx, seg_end_var_idx,
                    seg_start_fixed, seg_end_fixed,
                    seg_mult_arr, seg_side_arr,
                    shape_pos_tri, shape_pos_sq,
                    center_np, MU,
                    use_parabola, cg_mode, const_g_vec_arr, g_mag,
                    dv_type_arr, dv_in_seg_arr, dv_out_seg_arr,
                    dv_center_var_idx_arr, dv_shape_vel_arr,
                    energy_mode, fuel_eps,
                    v0_buf, vf_buf,
                    Jv0_r0_buf, Jv0_rf_buf, Jv0_dt_buf,
                    Jvf_r0_buf, Jvf_rf_buf, Jvf_dt_buf,
                    z_cache_arr, grad_out, ok_out,
                )
                if not np.all(ok_out == 1):
                    # Rare: Newton failed for at least one segment. Fall back to
                    # the pure-Python evaluator (which uses brentq) for this call.
                    return fun_and_grad(x_vec)
                # Copy grad so scipy's finite-difference checks don't see our
                # reused buffer mutate between iterations.
                return float(cost), grad_out.copy()

            fun_and_grad_active = fun_and_grad_numba

            # ---- Levenberg-Marquardt setup ---------------------------------
            # Count residual rows (2 per dv-node) for buffer sizing.
            M_dv = int(dv_type_arr.shape[0])
            r_buf = np.empty(2 * M_dv)
            J_buf = np.empty((2 * M_dv, n_vars))

            def lm_eval(x_vec):
                """Returns (cost, residual r, Jacobian J). cost is the actual
                BFGS cost (energy: sum |dv|^2; fuel: sum sqrt(|dv|^2 + eps^2)).
                If any segment fails, returns (inf, None, None)."""
                cost = _lnb.lm_eval_batch(
                    x_vec, n_pos_vars,
                    seg_start_kind, seg_end_kind,
                    seg_start_var_idx, seg_end_var_idx,
                    seg_start_fixed, seg_end_fixed,
                    seg_mult_arr, seg_side_arr,
                    shape_pos_tri, shape_pos_sq,
                    center_np, MU,
                    use_parabola, cg_mode, const_g_vec_arr, g_mag,
                    dv_type_arr, dv_in_seg_arr, dv_out_seg_arr,
                    dv_center_var_idx_arr, dv_shape_vel_arr,
                    energy_mode, fuel_eps,
                    v0_buf, vf_buf,
                    Jv0_r0_buf, Jv0_rf_buf, Jv0_dt_buf,
                    Jvf_r0_buf, Jvf_rf_buf, Jvf_dt_buf,
                    z_cache_arr, ok_out, r_buf, J_buf,
                )
                if not np.all(ok_out == 1):
                    return float("inf"), None, None
                return float(cost), r_buf.copy(), J_buf.copy()

        # Initial guess: positions + shared TOF (+ nu per free shape).
        x0 = np.zeros(n_vars)
        for i, dot in enumerate(movable_dots):
            x0[2 * i] = dot.x()
            x0[2 * i + 1] = dot.y()
        x0[n_pos_vars] = TAU_INIT_TWO_BODY if self.env_mode == "two_body" else TOF_INIT_CONSTANT_G
        for fs in free_shapes:
            x0[fs["nu_idx"]] = fs["nu_init"]

        # Orbit-rendezvous wrapper: at each cost evaluation, decode each
        # free shape's nu into (r, v) on its cached orbit and write into
        # shape_data + numba arrays in place. Compute the position+tof
        # gradient via the existing inner; compute nu gradient via forward
        # finite difference. Free shapes' position contributions are not
        # plumbed through the inner gradient (shape is fixed in inner's
        # view), so FD is the simplest correct path.
        if free_shapes:
            inner = fun_and_grad_active
            # dv_shape_vel_arr is built earlier inside the numba branch when
            # _NUMBA_AVAILABLE is True; otherwise the Python fun_and_grad
            # reads shape_vel directly from shape_data.
            _dv_vel_arr = dv_shape_vel_arr if _NUMBA_AVAILABLE else None

            def _set_free_shape_state(x_vec):
                for fs in free_shapes:
                    nu = float(x_vec[fs["nu_idx"]])
                    r_xy, v_xy = self._orbit_pos_vel(fs["elements"], nu)
                    sd = shape_data[fs["sd_idx"]]
                    sd[1][0] = r_xy[0]; sd[1][1] = r_xy[1]
                    sd[2][0] = v_xy[0]; sd[2][1] = v_xy[1]
                    if _dv_vel_arr is not None:
                        for row in fs["dv_vel_rows"]:
                            _dv_vel_arr[row, 0] = v_xy[0]
                            _dv_vel_arr[row, 1] = v_xy[1]

            # Attach numba dv-row metadata (built earlier in numba block).
            if _NUMBA_AVAILABLE:
                for fs in free_shapes:
                    fs["dv_vel_rows"] = list(shape_dv_rows.get(id(fs["shape_center"]), []))

            fd_h = 1e-6

            def fun_and_grad_with_nu(x_vec):
                _set_free_shape_state(x_vec)
                cost0, grad_inner = inner(x_vec)
                grad = grad_inner.copy()
                for fs in free_shapes:
                    x_pert = x_vec.copy()
                    x_pert[fs["nu_idx"]] += fd_h
                    _set_free_shape_state(x_pert)
                    cost_p, _ = inner(x_pert)
                    grad[fs["nu_idx"]] = (cost_p - cost0) / fd_h
                _set_free_shape_state(x_vec)
                return cost0, grad

            fun_and_grad_active = fun_and_grad_with_nu

        # ---- Solver: prefer LM/IRLS when numba is available --------------
        # LM/IRLS exploits the sum-of-squared-residuals structure: dv per
        # node is a small residual, the Jacobian is structured, and
        # H_GN = J^T W J converges super-linearly near the optimum. For
        # smoothed-L1 (fuel) we use IRLS weights w_i = 1/sqrt(|dv_i|^2+eps^2).
        # Falls back to BFGS on degeneracy. LM does not yet support free
        # shapes (nu decision variables); BFGS handles those via the
        # FD-wrapped fun_and_grad above.
        used_lm = False
        if _NUMBA_AVAILABLE and M_dv > 0 and not free_shapes:
            try:
                x_lm, lm_ok = self._lm_solve(
                    x0, lm_eval, n_pos_vars, energy_mode, fuel_eps,
                    progress_callback=progress_callback,
                    movable_dots=movable_dots,
                )
                if lm_ok:
                    used_lm = True
                    result = types.SimpleNamespace(x=x_lm)
            except Exception:
                used_lm = False

        if not used_lm:
            if progress_callback is not None:
                _free_meta_for_cb = [
                    {"shape": fs["shape"], "nu_idx": fs["nu_idx"], "elements": fs["elements"]}
                    for fs in free_shapes
                ]
                def scipy_cb(xk):
                    progress_callback(xk, n_pos_vars, movable_dots, _free_meta_for_cb)
                result = scipy_minimize(
                    fun_and_grad_active, x0, method="BFGS", jac=True,
                    options={"gtol": 1e-6}, callback=scipy_cb,
                )
            else:
                result = scipy_minimize(
                    fun_and_grad_active, x0, method="BFGS", jac=True,
                    options={"gtol": 1e-6},
                )

        if not apply:
            # Return everything needed to apply later on the main thread.
            # free_shape_meta: serializable list (shape, nu_idx, elements ref)
            free_shape_meta = [
                {"shape": fs["shape"], "nu_idx": fs["nu_idx"], "elements": fs["elements"]}
                for fs in free_shapes
            ]
            return result.x.copy(), n_pos_vars, movable_dots, free_shape_meta

        # Apply result: update positions and rendering tof
        for i, dot in enumerate(movable_dots):
            dot.setX(result.x[2 * i])
            dot.setY(result.x[2 * i + 1])
        self.render_tof = float(result.x[n_pos_vars])
        # Free shapes: write back nu and recompute shape pos+vel.
        for fs in free_shapes:
            nu = float(result.x[fs["nu_idx"]])
            r_xy, v_xy = self._orbit_pos_vel(fs["elements"], nu)
            shape = fs["shape"]
            if shape == "triangle":
                self.tri_nu = nu
                self.tri_center.setX(float(r_xy[0])); self.tri_center.setY(float(r_xy[1]))
                self.tri_velocity_end = QPointF(
                    float(r_xy[0] + v_xy[0] * VEL_SCALE),
                    float(r_xy[1] + v_xy[1] * VEL_SCALE),
                )
            else:
                self.sq_nu = nu
                self.sq_center.setX(float(r_xy[0])); self.sq_center.setY(float(r_xy[1]))
                self.sq_velocity_end = QPointF(
                    float(r_xy[0] + v_xy[0] * VEL_SCALE),
                    float(r_xy[1] + v_xy[1] * VEL_SCALE),
                )

        # Recompute orbits if active
        if self.tri_orbit_mode:
            self._compute_orbit_for_shape("triangle")
        if self.sq_orbit_mode:
            self._compute_orbit_for_shape("square")
        self.update()

    def _optimize_common_apply(self, new_x, n_pos_vars, movable_dots, free_shape_meta=None):
        """Apply an off-thread solve result on the main thread."""
        for i, dot in enumerate(movable_dots):
            dot.setX(float(new_x[2 * i]))
            dot.setY(float(new_x[2 * i + 1]))
        self.render_tof = float(new_x[n_pos_vars])
        # Free-shape (orbit-rendezvous) writeback: decode each nu into
        # (r, v) on the cached orbit and update shape center + velocity end.
        for fs in (free_shape_meta or []):
            nu = float(new_x[fs["nu_idx"]])
            r_xy, v_xy = self._orbit_pos_vel(fs["elements"], nu)
            shape = fs["shape"]
            if shape == "triangle":
                self.tri_nu = nu
                self.tri_center.setX(float(r_xy[0])); self.tri_center.setY(float(r_xy[1]))
                self.tri_velocity_end = QPointF(
                    float(r_xy[0] + v_xy[0] * VEL_SCALE),
                    float(r_xy[1] + v_xy[1] * VEL_SCALE),
                )
            else:
                self.sq_nu = nu
                self.sq_center.setX(float(r_xy[0])); self.sq_center.setY(float(r_xy[1]))
                self.sq_velocity_end = QPointF(
                    float(r_xy[0] + v_xy[0] * VEL_SCALE),
                    float(r_xy[1] + v_xy[1] * VEL_SCALE),
                )
        if self.tri_orbit_mode:
            self._compute_orbit_for_shape("triangle")
        if self.sq_orbit_mode:
            self._compute_orbit_for_shape("square")
        if self.on_render_tof_changed is not None:
            self.on_render_tof_changed(self.render_tof, self.env_mode)
        self.update()

    def _lm_solve(
        self, x0, lm_eval, n_pos_vars, energy_mode, fuel_eps,
        progress_callback=None, movable_dots=None,
        max_iter=10000, tol_grad=1e-7, tol_step=1e-9,
        lam_init=1e-3, lam_up=10.0, lam_down=0.4,
    ):
        """Levenberg-Marquardt / IRLS solver.

        Decision variables x: positions + shared TOF.
        Residual r = stacked dv vectors (per dv-node, 2 components each).
        Cost f(x) = sum_i phi(|dv_i|) where phi(u)=u^2 (energy) or
        phi(u)=sqrt(u^2 + eps^2) (fuel, smoothed-L1).

        Reduced normal equations:
          (J^T W J + lam D) p = -J^T W r
        where W = diag(w_k) with w_k = 1 (energy) or w_k = 1/sqrt(|dv_k|^2+eps^2)
        (IRLS); D = diag(diag(J^T W J)) (Marquardt scaling).

        Returns (x_new, ok). ok=False if no descent could be made; caller
        should fall back to BFGS.
        """
        x = x0.copy()
        cost, r, J = lm_eval(x)
        if r is None:
            return x, False

        n_vars = x.size
        M = r.size // 2

        def weights_from_r(r_vec):
            """IRLS weights per residual row. For energy: phi(u)=u^2 → derivative
            and Hessian both linear in r → effective weight=1. For fuel:
            phi(u)=sqrt(u^2+eps^2) → Gauss-Newton weight is 1/sqrt(...) per node."""
            if energy_mode:
                return np.ones(2 * M)
            w = np.empty(2 * M)
            for k in range(M):
                dvx = r_vec[2 * k]
                dvy = r_vec[2 * k + 1]
                wk = 1.0 / math.sqrt(dvx * dvx + dvy * dvy + fuel_eps * fuel_eps)
                w[2 * k] = wk
                w[2 * k + 1] = wk
            return w

        lam = lam_init
        for _it in range(max_iter):
            w = weights_from_r(r)
            # Build Gauss-Newton system: H = J^T W J, g = J^T W r
            JW = J * w[:, None]
            H = JW.T @ J
            g = JW.T @ r  # this is the gradient of f for energy (factor 2 absorbed only matters for scale; fine for LM)

            # Convergence check on gradient.
            if np.linalg.norm(g, ord=np.inf) < tol_grad:
                break

            diag_H = np.maximum(np.diag(H), 1e-12)

            # Try LM step; on failure, increase lam and retry.
            accepted = False
            for _inner in range(10):
                A = H + lam * np.diag(diag_H)
                try:
                    L = np.linalg.cholesky(A)
                    p = np.linalg.solve(L.T, np.linalg.solve(L, -g))
                except np.linalg.LinAlgError:
                    lam *= lam_up
                    continue

                x_new = x + p
                cost_new, r_new, J_new = lm_eval(x_new)
                if r_new is None or not np.isfinite(cost_new):
                    lam *= lam_up
                    continue

                if cost_new < cost:
                    # Accept.
                    step_norm = float(np.linalg.norm(p))
                    x = x_new
                    cost = cost_new
                    r = r_new
                    J = J_new
                    lam = max(lam * lam_down, 1e-12)
                    accepted = True

                    if progress_callback is not None and movable_dots is not None:
                        progress_callback(x, n_pos_vars, movable_dots)

                    if step_norm < tol_step * (1.0 + np.linalg.norm(x)):
                        return x, True
                    break
                else:
                    lam *= lam_up

            if not accepted:
                # No step found in 10 inner tries; bail to BFGS.
                return x, False

        return x, True

    def optimize_energy(self):
        self._optimize_common("energy")

    def optimize_fuel(self):
        self._optimize_common("fuel")

    def add_nodes(self):
        """Single pass: split every segment with mult >= 2 at its midpoint
        (k = mult // 2). Inverse of `prune_small_dvs` — each click adds one
        layer of refinement. Re-runs the active optimizer afterward."""
        center = self.earth_center
        any_split = False
        for traj in self.trajectories:
            new_segments = []
            for i_start, i_end, mult, _side in traj["segments"]:
                n_int = int(round(mult))
                if n_int < 2:
                    new_segments.append((i_start, i_end, mult, _side))
                    continue
                k = n_int // 2
                arc_pts = self._compute_segment_arc(
                    traj["dots"][i_start], traj["dots"][i_end], center,
                    self._seg_tof(traj["dots"][i_start], traj["dots"][i_end], mult),
                    n_int + 1,
                    side=_side,
                )
                new_pt = arc_pts[k]
                new_dot = QPointF(new_pt.x(), new_pt.y())
                traj["dots"].append(new_dot)
                new_idx = len(traj["dots"]) - 1
                new_segments.append((i_start, new_idx, float(k), _side))
                new_segments.append((new_idx, i_end, float(n_int - k), _side))
                any_split = True
            traj["segments"] = new_segments
        if any_split and self.optimize_mode is not None:
            self._optimize_common(self.optimize_mode)
        self.update()

    def prune_small_dvs(self, threshold=SMALL_DV_THRESHOLD):
        """Iteratively remove interior black nodes whose |dv| is below
        threshold. Removes one node at a time (smallest dv first) and
        re-runs the active optimizer between removals so the trajectory
        adjusts. Shape-attached dvs are not eligible."""
        center = self.earth_center
        skip = set()  # id(dot) for nodes _delete_black_node refused
        while True:
            candidates = []  # (mag, dot)
            for traj in self.trajectories:
                dots = traj["dots"]
                segments = traj["segments"]
                for k, dot in enumerate(dots):
                    if dot is self.tri_center or dot is self.sq_center:
                        continue
                    if id(dot) in skip:
                        continue
                    in_v = out_v = None
                    in_count = out_count = 0
                    for i_start, i_end, mult, _side in segments:
                        seg_tof = self._seg_tof(
                            dots[i_start], dots[i_end], mult)
                        if i_end == k:
                            in_count += 1
                            _, vf = self._compute_segment_velocities(
                                dots[i_start], dots[i_end], center, seg_tof)
                            in_v = vf
                        elif i_start == k:
                            out_count += 1
                            v0, _ = self._compute_segment_velocities(
                                dots[i_start], dots[i_end], center, seg_tof)
                            out_v = v0
                    if in_count != 1 or out_count != 1:
                        continue
                    mag = float(np.linalg.norm(out_v - in_v))
                    if mag < threshold:
                        candidates.append((mag, dot))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0])
            _, dot = candidates[0]
            if not self._delete_black_node(dot):
                skip.add(id(dot))
                continue
            if self.optimize_mode is not None:
                self._optimize_common(self.optimize_mode)
        self.update()

    def set_optimize_mode(self, mode):
        """mode in {None, 'energy', 'fuel'}. When set, runs once immediately
        and then on every subsequent drag release."""
        assert mode in (None, "energy", "fuel")
        self.optimize_mode = mode
        if self.on_dv_scale_changed is not None:
            self.on_dv_scale_changed(self.current_dv_scale())
        if mode is not None:
            self._optimize_common(mode)

    def current_dv_scale(self):
        return self.dv_scale_by_mode[self.optimize_mode]

    def set_dv_scale(self, value):
        self.dv_scale_by_mode[self.optimize_mode] = float(value)
        self.update()

    def set_render_tof(self, value, run_optimizer=True):
        """Manual override of the shared time variable from a UI slider.
        When run_optimizer is True, re-runs the active optimizer (which uses
        self.render_tof as its starting guess) so dot positions follow the
        new time. Pass False during a continuous drag and re-run on release."""
        self.render_tof = float(value)
        if self.tri_orbit_mode:
            self._compute_orbit_for_shape("triangle")
        if self.sq_orbit_mode:
            self._compute_orbit_for_shape("square")
        if run_optimizer:
            self._run_active_optimizer()
        self.update()

    def _run_active_optimizer(self):
        """Kick off the active optimizer asynchronously. If a solve is already
        in flight, mark that it should be restarted when it finishes so its
        (now-stale) result is discarded."""
        if self.optimize_mode is None:
            return
        if self._opt_running:
            self._opt_restart = True
            return
        self._opt_running = True
        self._opt_executor.submit(self._opt_worker, self.optimize_mode)

    def _opt_worker(self, cost_mode):
        """Worker-thread entry point. Runs the solve without mutating self
        and emits the result via a queued signal to the main thread. A
        throttled progress callback streams intermediate BFGS iterates so
        the UI can animate the trajectory morphing toward the optimum."""
        last_emit = [0.0]
        min_dt = 0.04  # ~25 fps cap for progress frames

        def cb(xk, n_pos_vars, movable_dots, free_shape_meta=None):
            if self._opt_restart:
                # User moved something else; skip stale progress frames.
                return
            now = time.perf_counter()
            if now - last_emit[0] < min_dt:
                return
            last_emit[0] = now
            self._opt_signals.progress.emit((xk.copy(), n_pos_vars, movable_dots, free_shape_meta or []))

        try:
            result = self._optimize_common(cost_mode, apply=False, progress_callback=cb)
        except Exception as e:  # noqa: BLE001 - surface any failure to main thread
            result = ("__err__", repr(e))
        self._opt_signals.done.emit(result)

    def _on_opt_progress(self, payload):
        """Main-thread slot: apply an intermediate BFGS iterate so the user
        sees a smooth morph into the optimum. Skip if a restart is queued."""
        if self._opt_restart:
            return
        new_x, n_pos_vars, movable_dots, free_shape_meta = payload
        self._optimize_common_apply(new_x, n_pos_vars, movable_dots, free_shape_meta)

    def _on_opt_done(self, result):
        """Main-thread handler for worker-thread results. Discards stale
        results if a restart was requested during the solve."""
        self._opt_running = False
        if self._opt_restart:
            self._opt_restart = False
            self._run_active_optimizer()
            return
        if result is None:
            return
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "__err__":
            print(f"optimize error: {result[1]}")
            return
        new_x, n_pos_vars, movable_dots, free_shape_meta = result
        self._optimize_common_apply(new_x, n_pos_vars, movable_dots, free_shape_meta)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand-Crafted Trajectory Design")
        self.setMinimumSize(1200, 900)

        canvas = Canvas()
        self.canvas = canvas

        clear_button = QPushButton("Clear")
        clear_button.setFixedSize(50, 25)
        clear_button.clicked.connect(canvas.clear)

        optimize_energy_btn = QPushButton("Minimize: Energy")
        optimize_energy_btn.setFixedSize(150, 25)
        optimize_energy_btn.setCheckable(True)

        optimize_fuel_btn = QPushButton("Minimize: Fuel")
        optimize_fuel_btn.setFixedSize(130, 25)
        optimize_fuel_btn.setCheckable(True)

        def on_energy_toggled(checked):
            if checked:
                # Mutually exclusive with fuel
                if optimize_fuel_btn.isChecked():
                    optimize_fuel_btn.blockSignals(True)
                    optimize_fuel_btn.setChecked(False)
                    optimize_fuel_btn.blockSignals(False)
                canvas.set_optimize_mode("energy")
            else:
                if not optimize_fuel_btn.isChecked():
                    canvas.set_optimize_mode(None)

        def on_fuel_toggled(checked):
            if checked:
                if optimize_energy_btn.isChecked():
                    optimize_energy_btn.blockSignals(True)
                    optimize_energy_btn.setChecked(False)
                    optimize_energy_btn.blockSignals(False)
                canvas.set_optimize_mode("fuel")
            else:
                if not optimize_energy_btn.isChecked():
                    canvas.set_optimize_mode(None)

        optimize_energy_btn.toggled.connect(on_energy_toggled)
        optimize_fuel_btn.toggled.connect(on_fuel_toggled)

        env_btn = QPushButton(canvas.get_env_button_text())
        env_btn.setFixedSize(180, 25)

        def toggle_env_button():
            canvas.toggle_env_mode()
            env_btn.setText(canvas.get_env_button_text())

        env_btn.clicked.connect(toggle_env_button)

        frame_btn = QPushButton(canvas.get_frame_button_text())
        frame_btn.setFixedSize(140, 25)

        def toggle_frame_button():
            canvas.toggle_frame_mode()
            frame_btn.setText(canvas.get_frame_button_text())

        frame_btn.clicked.connect(toggle_frame_button)

        top_layout = QHBoxLayout()
        top_layout.addSpacing(10)
        top_layout.addWidget(env_btn)
        top_layout.addSpacing(10)
        top_layout.addWidget(frame_btn)
        top_layout.addStretch()
        top_layout.addWidget(optimize_energy_btn)
        top_layout.addSpacing(10)
        top_layout.addWidget(optimize_fuel_btn)
        top_layout.addStretch()
        top_layout.addWidget(clear_button)
        top_layout.addSpacing(10)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(30, 30)
        zoom_in_btn.clicked.connect(canvas.zoom_in)

        zoom_out_btn = QPushButton("\u2212")
        zoom_out_btn.setFixedSize(30, 30)
        zoom_out_btn.clicked.connect(canvas.zoom_out)

        prune_btn = QPushButton("Remove nodes")
        prune_btn.setFixedSize(160, 25)
        prune_btn.clicked.connect(lambda: canvas.prune_small_dvs())

        add_btn = QPushButton("Add nodes")
        add_btn.setFixedSize(160, 25)
        add_btn.clicked.connect(lambda: canvas.add_nodes())

        undo_btn = QPushButton("Undo deletion")
        undo_btn.setFixedSize(160, 25)
        undo_btn.clicked.connect(lambda: canvas.undo_delete())

        zoom_layout = QHBoxLayout()
        zoom_layout.addSpacing(10)
        zoom_layout.addWidget(add_btn)
        zoom_layout.addWidget(prune_btn)
        zoom_layout.addWidget(undo_btn)
        zoom_layout.addStretch()
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addSpacing(10)

        # Vertical slider on the right of the canvas: dv display scale.
        # Range [0, 30 * VEL_SCALE]; integer slider uses 0.1-unit resolution.
        SLIDER_RES = 10
        dv_slider = QSlider(Qt.Orientation.Vertical)
        dv_slider.setRange(0, int(DV_SCALE_MAX * SLIDER_RES))
        dv_slider.setValue(int(canvas.current_dv_scale() * SLIDER_RES))
        dv_slider.setFixedWidth(20)
        dv_label = QLabel("Maneuver\nlength")
        dv_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        def on_slider_changed(v):
            canvas.set_dv_scale(v / SLIDER_RES)

        dv_slider.valueChanged.connect(on_slider_changed)

        # When optimize mode toggles, snap the slider to that mode's stored value.
        def on_dv_scale_changed(value):
            dv_slider.blockSignals(True)
            dv_slider.setValue(int(value * SLIDER_RES))
            dv_slider.blockSignals(False)

        canvas.on_dv_scale_changed = on_dv_scale_changed

        # Second slider: shared time variable (tau in two-body, tof in CG).
        TAU_SLIDER_RES = 1000
        tau_slider = QSlider(Qt.Orientation.Vertical)

        def _tau_slider_max():
            return TAU_SLIDER_MAX if canvas.env_mode == "two_body" else TOF_SLIDER_MAX

        tau_slider.setRange(0, int(_tau_slider_max() * TAU_SLIDER_RES))
        tau_slider.setValue(int(canvas.render_tof * TAU_SLIDER_RES))
        tau_slider.setFixedWidth(20)
        tau_label = QLabel("")
        tau_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        def _update_tau_label():
            if canvas.env_mode == "two_body":
                tau_label.setText("\u0394\u03c4")  # Δτ
            else:
                tau_label.setText("\u0394t")  # Δt

        _update_tau_label()

        def on_tau_slider_changed(v):
            # While the user is actively dragging, do not re-run the optimizer
            # on every tick — let tau vary freely. Re-optimize once on release.
            canvas.set_render_tof(v / TAU_SLIDER_RES,
                                  run_optimizer=not tau_slider.isSliderDown())

        tau_slider.valueChanged.connect(on_tau_slider_changed)

        def on_tau_slider_released():
            canvas.set_render_tof(tau_slider.value() / TAU_SLIDER_RES,
                                  run_optimizer=True)

        tau_slider.sliderReleased.connect(on_tau_slider_released)

        def on_render_tof_changed(value, env_mode):
            tau_slider.blockSignals(True)
            tau_slider.setRange(0, int(_tau_slider_max() * TAU_SLIDER_RES))
            tau_slider.setValue(int(max(0.0, min(_tau_slider_max(), value)) * TAU_SLIDER_RES))
            tau_slider.blockSignals(False)
            _update_tau_label()

        canvas.on_render_tof_changed = on_render_tof_changed

        slider_col = QVBoxLayout()
        slider_col.setContentsMargins(4, 4, 8, 4)
        slider_col.addStretch(1)
        slider_col.addWidget(dv_label)
        slider_col.addWidget(dv_slider, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        slider_col.addSpacing(20)
        slider_col.addWidget(tau_label)
        slider_col.addWidget(tau_slider, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
        slider_col.addStretch(1)

        canvas_row = QHBoxLayout()
        canvas_row.setContentsMargins(0, 0, 0, 0)
        canvas_row.setSpacing(0)
        canvas_row.addWidget(canvas, 1)
        canvas_row.addLayout(slider_col)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(top_layout)
        layout.addLayout(canvas_row)
        layout.addLayout(zoom_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        canvas.setFocus()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
