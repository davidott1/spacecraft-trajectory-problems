import sys

import math

import numpy as np
from scipy.optimize import brentq, minimize as scipy_minimize

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout

# Canonical units: 1 DU = 6371 km, 1 TU = sqrt(DU^3 / mu_km) ~ 806.4 s, mu = 1 DU^3/TU^2.
GRAVITY_MAG = 1.0  # DU/TU^2 (~ 9.81e-3 km/s^2 in canonical units)
TIME_OF_FLIGHT = 1.5  # TU (~ 1200 s)
ARC_NUM_POINTS = 50  # number of points to draw the parabolic arc
ORBIT_NUM_POINTS = 200  # number of points to draw a Kepler orbit
MU = 1.0  # canonical gravitational parameter
VEL_SCALE = 2.0  # display scale: velocity_end = center + vel * VEL_SCALE


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


def lambert_solve(r1_vec, r2_vec, dt, mu):
    """Solve Lambert's problem in 2D using universal variable z-iteration.
    r1_vec, r2_vec: numpy arrays, position vectors relative to gravity center.
    dt: time of flight (seconds).
    mu: gravitational parameter.
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
    if abs(sin_dtheta) < 1e-14 or abs(1 - cos_dtheta) < 1e-14:
        v = (r2_vec - r1_vec) / dt
        return v, v

    A = sin_dtheta * math.sqrt(r1 * r2 / (1.0 - cos_dtheta))

    def y_func(z):
        C = _stumpff_c(z)
        S = _stumpff_s(z)
        return r1 + r2 + A * (z * S - 1.0) / math.sqrt(max(C, 1e-30))

    def F(z):
        C = _stumpff_c(z)
        S = _stumpff_s(z)
        y = y_func(z)
        if y < 0:
            return float('inf')
        return (y / max(C, 1e-30)) ** 1.5 * S + A * math.sqrt(y) - math.sqrt(mu) * dt

    # Find bracket for z (z=0 is parabolic, z>0 elliptic, z<0 hyperbolic)
    z_low = -200.0
    z_high = 4.0 * math.pi ** 2 - 0.5  # just below first singularity

    # Ensure y > 0 at lower bound
    while y_func(z_low) < 0 and z_low < z_high:
        z_low += 1.0

    try:
        F_low = F(z_low)
        F_high = F(z_high)
        if not math.isfinite(F_low) or not math.isfinite(F_high) or F_low * F_high > 0:
            raise ValueError
        z = brentq(F, z_low, z_high, xtol=1e-10, maxiter=300)
    except (ValueError, RuntimeError):
        # Fallback to straight-line approximation
        v = (r2_vec - r1_vec) / dt
        return v, v

    C = _stumpff_c(z)
    y = y_func(z)

    f = 1.0 - y / r1
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (gdot * r2_vec - r1_vec) / g

    return v1, v2


def lambert_solve_with_jac(r1_vec, r2_vec, dt, mu):
    """Lambert solver that also returns analytic Jacobians of (v1, v2)
    w.r.t. (r1, r2, dt) via the implicit function theorem applied to the
    universal-variable equation F(z; r1, r2, dt) = 0.

    Returns:
        v1, v2 : (2,) numpy arrays
        Jv1_r1, Jv1_r2 : (2,2) numpy arrays
        Jv1_dt : (2,) numpy array
        Jv2_r1, Jv2_r2 : (2,2) numpy arrays
        Jv2_dt : (2,) numpy array

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
        return v, v, Jv_r1, Jv_r2, Jv_dt, Jv_r1, Jv_r2, Jv_dt

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
    if abs(sin_dtheta) < 1e-14 or abs(1.0 - cos_dtheta) < 1e-14:
        return _straight_line()

    A = sin_dtheta * math.sqrt(r1 * r2 / (1.0 - cos_dtheta))

    def y_func(z):
        C = _stumpff_c(z)
        S = _stumpff_s(z)
        return r1 + r2 + A * (z * S - 1.0) / math.sqrt(max(C, 1e-30))

    def F(z):
        C = _stumpff_c(z)
        S = _stumpff_s(z)
        y = y_func(z)
        if y < 0:
            return float('inf')
        return (y / max(C, 1e-30)) ** 1.5 * S + A * math.sqrt(y) - math.sqrt(mu) * dt

    z_low = -200.0
    z_high = 4.0 * math.pi ** 2 - 0.5
    while y_func(z_low) < 0 and z_low < z_high:
        z_low += 1.0

    try:
        F_low = F(z_low)
        F_high = F(z_high)
        if not math.isfinite(F_low) or not math.isfinite(F_high) or F_low * F_high > 0:
            raise ValueError
        z = brentq(F, z_low, z_high, xtol=1e-12, maxiter=300)
    except (ValueError, RuntimeError):
        return _straight_line()

    C = _stumpff_c(z)
    S = _stumpff_s(z)
    Cp = _stumpff_cp(z)
    Sp = _stumpff_sp(z)
    y = y_func(z)
    if y <= 0 or C <= 0 or abs(A) < 1e-15:
        return _straight_line()

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

    return v1, v2, Jv1_r1, Jv1_r2, Jv1_dt, Jv2_r1, Jv2_r2, Jv2_dt


def _kepler_deriv(state, mu):
    """Derivative for 2D Kepler propagation. state = [rx, ry, vx, vy]."""
    r = math.sqrt(state[0] ** 2 + state[1] ** 2)
    if r < 1e-10:
        return np.array([state[2], state[3], 0.0, 0.0])
    f = -mu / (r ** 3)
    return np.array([state[2], state[3], f * state[0], f * state[1]])


def compute_dynamic_arc(p0, p1, center, time_of_flight, num_points):
    """Compute Keplerian arc points between p0 and p1 using Lambert solver + RK4."""
    r1_vec = np.array([p0.x() - center.x(), p0.y() - center.y()])
    r2_vec = np.array([p1.x() - center.x(), p1.y() - center.y()])

    if np.linalg.norm(r1_vec) < 1e-10 or np.linalg.norm(r2_vec) < 1e-10:
        return [p0, p1]

    v1, _ = lambert_solve(r1_vec, r2_vec, time_of_flight, MU)

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


def compute_arc_velocities(p0, p1, center, time_of_flight):
    """Return (v0, vf) numpy arrays for the Lambert arc from p0 to p1."""
    r1_vec = np.array([p0.x() - center.x(), p0.y() - center.y()])
    r2_vec = np.array([p1.x() - center.x(), p1.y() - center.y()])

    if np.linalg.norm(r1_vec) < 1e-10 or np.linalg.norm(r2_vec) < 1e-10:
        disp = np.array([p1.x() - p0.x(), p1.y() - p0.y()])
        v = disp / time_of_flight
        return v, v

    return lambert_solve(r1_vec, r2_vec, time_of_flight, MU)


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.trajectories = []  # list of {"dots": [QPointF], "segments": [(i, j, mult)]}
        self.dot_radius = 0.10  # DU
        self.trace_spacing = 1.0  # DU
        self.dragging = False
        self.dragging_shape = None  # "triangle" or "square" when dragging a shape
        self.drag_offset = QPointF(0, 0)
        self.dragging_dot = None  # QPointF reference when dragging a black node
        self.shift_click_first = None  # first node selected for shift-click linking
        self.dragging_vel_end = None  # "triangle" or "square" when dragging a velocity line
        self.dragging_vel_t = 1.0  # parameter along line where grab occurred (0=center, 1=tip)
        self.x_held = False  # X key held: click black node to delete + merge segments
        # Shape positions (center points) — placed in orbit around earth (DU)
        self.earth_center = QPointF(0, 0)
        self.earth_radius = 1.0  # DU
        self.tri_center = QPointF(1.5, 0)
        self.tri_size = 0.4
        self.sq_center = QPointF(-1.5, 0)
        self.sq_size = 0.36

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

        self.tri_orbit = []  # list of QPointF for triangle Kepler orbit
        self.sq_orbit = []  # list of QPointF for square Kepler orbit
        self.tri_orbit_mode = True
        self.sq_orbit_mode = True
        self.arc_model_mode = "conic"  # "conic" (Lambert) or "parabola"
        self.env_mode = "two_body"  # "two_body" or "constant_gravity"
        self.frame_mode = "inertial"  # "inertial" or "rotating"
        # Per-segment time of flight used for rendering. Optimizers may overwrite.
        self.render_tof = TIME_OF_FLIGHT
        # Active continuous-optimization mode: None | "energy" | "fuel".
        # When set, every drag release re-runs the optimizer.
        self.optimize_mode = None

        self.setStyleSheet("background-color: white;")
        self.zoom = 50.0  # pixels per DU
        self.pan_offset = QPointF(600, 450)

        # Compute initial orbits
        self._compute_orbit_for_shape("triangle")
        self._compute_orbit_for_shape("square")

    def get_arc_model_button_text(self):
        if self.arc_model_mode == "conic":
            return "Arc Model: Conic"
        return "Arc Model: Parabola"

    def toggle_arc_model_mode(self):
        if self.arc_model_mode == "conic":
            self.arc_model_mode = "parabola"
        else:
            self.arc_model_mode = "conic"
        self._run_active_optimizer()
        self.update()

    def get_env_button_text(self):
        if self.env_mode == "two_body":
            return "Env: Two-Body"
        return "Env: Constant-Gravity"

    def toggle_env_mode(self):
        if self.env_mode == "two_body":
            self.env_mode = "constant_gravity"
        else:
            self.env_mode = "two_body"
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

    def _compute_segment_arc(self, p0, p1, center, time_of_flight, num_points):
        if self.env_mode == "constant_gravity":
            return compute_parabolic_arc(
                p0, p1, center, time_of_flight, num_points,
                g_vec=self._constant_g_vec(),
            )
        if self.arc_model_mode == "parabola":
            return compute_parabolic_arc(p0, p1, center, time_of_flight, num_points)
        return compute_dynamic_arc(p0, p1, center, time_of_flight, num_points)

    def _compute_segment_velocities(self, p0, p1, center, time_of_flight):
        if self.env_mode == "constant_gravity":
            return compute_parabolic_arc_velocities(
                p0, p1, center, time_of_flight, g_vec=self._constant_g_vec(),
            )
        if self.arc_model_mode == "parabola":
            return compute_parabolic_arc_velocities(p0, p1, center, time_of_flight)
        return compute_arc_velocities(p0, p1, center, time_of_flight)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_X:
            self.x_held = True

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_X:
            self.x_held = False

    def _screen_to_world(self, screen_pos):
        wx = (screen_pos.x() - self.pan_offset.x()) / self.zoom
        wy = (self.pan_offset.y() - screen_pos.y()) / self.zoom
        return QPointF(wx, wy)

    def _triangle_polygon(self):
        x, y = self.tri_center.x(), self.tri_center.y()
        s = self.tri_size
        return QPolygonF([
            QPointF(x - s / 2, y - s / 2),
            QPointF(x + s / 2, y),
            QPointF(x - s / 2, y + s / 2),
        ])

    def _hit_triangle(self, pos):
        return self._triangle_polygon().containsPoint(pos, Qt.FillRule.WindingFill)

    def _hit_square(self, pos):
        x, y = self.sq_center.x(), self.sq_center.y()
        s = self.sq_size
        return abs(pos.x() - x) <= s / 2 and abs(pos.y() - y) <= s / 2

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

    def _segment_exists(self, p0, p1):
        """Check if a segment already exists between two points."""
        for traj in self.trajectories:
            dots = traj["dots"]
            for i_start, i_end, _ in traj["segments"]:
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
            for s_idx, (i_start, i_end, mult) in enumerate(traj["segments"]):
                arc_pts = self._compute_segment_arc(
                    dots[i_start], dots[i_end], center,
                    self.render_tof * mult, dense + 1,
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
        i_start, i_end, _ = traj["segments"][s_idx]
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
            for s_idx, (i_start, i_end, mult) in enumerate(traj["segments"]):
                n_int = int(round(mult))
                if n_int < 2:
                    continue
                arc_pts = self._compute_segment_arc(
                    dots[i_start], dots[i_end], center,
                    self.render_tof * mult, dense + 1,
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
        i_start, i_end, mult = traj["segments"][s_idx]
        # Place the new dot at the arc point at time = k * tof along segment.
        arc_pts = self._compute_segment_arc(
            traj["dots"][i_start], traj["dots"][i_end], center,
            self.render_tof * mult, n_int + 1,
        )
        new_pt = arc_pts[k]
        new_dot = QPointF(new_pt.x(), new_pt.y())
        traj["dots"].append(new_dot)
        new_idx = len(traj["dots"]) - 1
        traj["segments"][s_idx] = (i_start, new_idx, float(k))
        traj["segments"].insert(s_idx + 1, (new_idx, i_end, float(n_int - k)))
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
            for s_idx, (i_start, i_end, _m) in enumerate(traj["segments"]):
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
                for i_s, i_e, m in traj["segments"]:
                    if i_s > k:
                        i_s -= 1
                    if i_e > k:
                        i_e -= 1
                    renumbered.append((i_s, i_e, m))
                traj["segments"] = renumbered
                if not traj["dots"]:
                    self.trajectories.remove(traj)
                return True
            if in_seg_idx is None or out_seg_idx is None:
                return False
            i_prev, _, m_in = traj["segments"][in_seg_idx]
            _, i_next, m_out = traj["segments"][out_seg_idx]
            # Build merged segment list, drop the two old segments.
            new_segments = [s for s_idx, s in enumerate(traj["segments"])
                            if s_idx not in (in_seg_idx, out_seg_idx)]
            new_segments.append((i_prev, i_next, m_in + m_out))
            # Remove the dot and renumber indices in segments above k.
            del dots[k]
            renumbered = []
            for i_s, i_e, m in new_segments:
                if i_s > k:
                    i_s -= 1
                if i_e > k:
                    i_e -= 1
                renumbered.append((i_s, i_e, m))
            traj["segments"] = renumbered
            return True
        return False

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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._screen_to_world(QPointF(event.pos()))
            # X + click on a black node to delete it and merge its two
            # incident segments (mult = m_in + m_out).
            # X + click on a segment (within 1 px of arc, >5 px from each
            # endpoint) to delete that segment.
            if self.x_held:
                # Try segment delete first (requires click >5 px from each
                # endpoint), so a click on a line near a node doesn't take
                # the node with it.
                if self._delete_segment_at(pos, arc_max_dist=2.0 / self.zoom,
                                           endpoint_min_dist=5.0 / self.zoom):
                    self._run_active_optimizer()
                    self.update()
                    return
                dot, _traj = self._find_nearest_dot(pos, max_dist=5.0 / self.zoom)
                if dot is not None and self._delete_black_node(dot):
                    self._run_active_optimizer()
                    self.update()
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
                                target_traj = {"dots": [first_dot, dot], "segments": [(0, 1, 1.0)]}
                                self.trajectories.append(target_traj)
                            else:
                                # Add dots if not already present, then add segment
                                if first_dot not in target_traj["dots"]:
                                    target_traj["dots"].append(first_dot)
                                if dot not in target_traj["dots"]:
                                    target_traj["dots"].append(dot)
                                i0 = target_traj["dots"].index(first_dot)
                                i1 = target_traj["dots"].index(dot)
                                target_traj["segments"].append((i0, i1, 1.0))
                            self.update()
                        self.shift_click_first = None
                return
            # Check shapes for dragging, V-line, or O-orbit. Cmd is reserved
            # for trajectory drawing, so don't translate shapes or grab
            # velocity arrows while Cmd is held.
            cmd_held = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if not cmd_held and self._hit_triangle(pos):
                self.dragging_shape = "triangle"
                self.drag_offset = self.tri_center - pos
                return
            if not cmd_held and self._hit_square(pos):
                self.dragging_shape = "square"
                self.drag_offset = self.sq_center - pos
                return
            # Check for dragging anywhere on a velocity line
            if not event.modifiers():
                for shape, center, vel_end in [
                    ("triangle", self.tri_center, self.tri_velocity_end),
                    ("square", self.sq_center, self.sq_velocity_end),
                ]:
                    if vel_end is not None:
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
                            if d < 0.30:
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
                    self.trajectories.append({"dots": [snap_node, start_pos], "segments": [(0, 1, 1.0)]})
                else:
                    self.trajectories.append({"dots": [start_pos], "segments": []})
                self.update()

    def mouseMoveEvent(self, event):
        pos = self._screen_to_world(QPointF(event.pos()))
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
                if self.tri_orbit_mode:
                    self._compute_orbit_for_shape("triangle")
            else:
                self.sq_velocity_end = new_end
                if self.sq_orbit_mode:
                    self._compute_orbit_for_shape("square")
            self._run_active_optimizer()
            self.update()
            return
        # Handle shape dragging
        if self.dragging_shape == "triangle":
            new_center = pos + self.drag_offset
            delta = new_center - self.tri_center
            self.tri_center.setX(new_center.x())
            self.tri_center.setY(new_center.y())
            if self.tri_velocity_end is not None:
                self.tri_velocity_end = self.tri_velocity_end + delta
            if self.tri_orbit_mode:
                self._compute_orbit_for_shape("triangle")
            self._run_active_optimizer()
            self.update()
            return
        if self.dragging_shape == "square":
            new_center = pos + self.drag_offset
            delta = new_center - self.sq_center
            self.sq_center.setX(new_center.x())
            self.sq_center.setY(new_center.y())
            if self.sq_velocity_end is not None:
                self.sq_velocity_end = self.sq_velocity_end + delta
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
        dx = pos.x() - last.x()
        dy = pos.y() - last.y()
        dist = math.hypot(dx, dy)
        if dist >= self.trace_spacing:
            # Interpolate dots at exact spacing intervals
            num_dots = int(dist // self.trace_spacing)
            ux = dx / dist
            uy = dy / dist
            for k in range(1, num_dots + 1):
                interp = QPointF(
                    last.x() + ux * self.trace_spacing * k,
                    last.y() + uy * self.trace_spacing * k,
                )
                self.trajectories[-1]["segments"].append(
                    (len(self.trajectories[-1]["dots"]) - 1,
                     len(self.trajectories[-1]["dots"]),
                     1.0)
                )
                self.trajectories[-1]["dots"].append(interp)
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
                                    for (i_s, i_e, m) in traj["segments"]:
                                        if i_s == i_last:
                                            i_s = i_snap
                                        if i_e == i_last:
                                            i_e = i_snap
                                        if i_s != i_e:
                                            new_segs.append((i_s, i_e, m))
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
                                    traj["segments"].append((i_last, i_snap, 1.0))
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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Apply pan and zoom (y-up convention: negative y-scale flips vertical)
        painter.translate(self.pan_offset.x(), self.pan_offset.y())
        painter.scale(self.zoom, -self.zoom)

        # Draw Kepler orbits (only in two-body env)
        if self.env_mode == "two_body":
            pen = QPen(QColor("green"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for orbit in (self.tri_orbit, self.sq_orbit):
                if len(orbit) > 1:
                    for k in range(len(orbit) - 1):
                        painter.drawLine(orbit[k], orbit[k + 1])

        # Draw velocity lines from triangle/square
        pen = QPen(QColor("green"), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        if self.tri_velocity_end is not None:
            painter.drawLine(self.tri_center, self.tri_velocity_end)
        if self.sq_velocity_end is not None:
            painter.drawLine(self.sq_center, self.sq_velocity_end)

        # Draw blue circle (Earth) only in two-body env
        if self.env_mode == "two_body":
            painter.setBrush(QBrush(QColor("blue")))
            pen = QPen(QColor("blue"), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawEllipse(self.earth_center, self.earth_radius, self.earth_radius)

        # Draw trajectories
        center = self.earth_center
        for traj in self.trajectories:
            dots = traj["dots"]
            segments = traj["segments"]

            # Draw segments
            for i_start, i_end, mult in segments:
                pen = QPen(QColor("black"), 2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                arc_points = self._compute_segment_arc(
                    dots[i_start], dots[i_end], center,
                    self.render_tof * mult, ARC_NUM_POINTS,
                )
                for j in range(len(arc_points) - 1):
                    painter.drawLine(arc_points[j], arc_points[j + 1])

        # Draw delta-velocity vectors at nodes with exactly 2 neighboring segments
        pen = QPen(QColor("black"), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for traj in self.trajectories:
            dots = traj["dots"]
            segments = traj["segments"]
            for k, dot in enumerate(dots):
                if dot is self.tri_center or dot is self.sq_center:
                    continue
                # Find segments where this node is an endpoint
                incoming_vel = None  # arrival velocity at node
                outgoing_vel = None  # departure velocity from node
                for i_start, i_end, mult in segments:
                    if i_end == k:
                        _, vf = self._compute_segment_velocities(
                            dots[i_start], dots[i_end], center, self.render_tof * mult)
                        incoming_vel = vf
                    elif i_start == k:
                        v0, _ = self._compute_segment_velocities(
                            dots[i_start], dots[i_end], center, self.render_tof * mult)
                        outgoing_vel = v0
                if incoming_vel is not None and outgoing_vel is not None:
                    dv = outgoing_vel - incoming_vel
                    painter.drawLine(
                        dot,
                        QPointF(dot.x() + dv[0] * VEL_SCALE, dot.y() + dv[1] * VEL_SCALE),
                    )

        # Draw delta-velocity vectors at triangle/square when they have
        # both a velocity vector and an attached segment
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
                for i_start, i_end, mult in traj["segments"]:
                    if dots[i_start] is shape_center:
                        v0, _ = self._compute_segment_velocities(
                            dots[i_start], dots[i_end], center, self.render_tof * mult)
                        dv = v0 - shape_vel
                        painter.drawLine(
                            shape_center,
                            QPointF(shape_center.x() + dv[0] * VEL_SCALE, shape_center.y() + dv[1] * VEL_SCALE),
                        )
                    elif dots[i_end] is shape_center:
                        _, vf = self._compute_segment_velocities(
                            dots[i_start], dots[i_end], center, self.render_tof * mult)
                        dv = shape_vel - vf
                        painter.drawLine(
                            shape_center,
                            QPointF(shape_center.x() + dv[0] * VEL_SCALE, shape_center.y() + dv[1] * VEL_SCALE),
                        )

        # Draw green triangle and square
        painter.setBrush(QBrush(QColor("green")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(self._triangle_polygon())
        sq_x = self.sq_center.x() - self.sq_size / 2
        sq_y = self.sq_center.y() - self.sq_size / 2
        painter.drawRect(QRectF(sq_x, sq_y, self.sq_size, self.sq_size))

        # Draw all black dots on top of everything (including shapes)
        painter.setBrush(QBrush(QColor("black")))
        painter.setPen(Qt.PenStyle.NoPen)
        for traj in self.trajectories:
            for dot in traj["dots"]:
                if dot is self.tri_center or dot is self.sq_center:
                    continue
                painter.drawEllipse(dot, self.dot_radius, self.dot_radius)

        painter.end()


    def clear(self):
        self.trajectories.clear()
        self.shift_click_first = None
        self.update()

    def _optimize_common(self, cost_mode):
        """Shared optimizer logic. cost_mode='energy' uses sum(|dv|^2), 'fuel' uses sum(|dv|)."""
        # Collect all movable (black) dots that participate in at least one segment
        center = np.array([self.earth_center.x(), self.earth_center.y()])
        movable_dots = []  # list of QPointF references
        movable_set = set()  # ids to avoid duplicates

        for traj in self.trajectories:
            dots = traj["dots"]
            for i_start, i_end, _mult in traj["segments"]:
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
            for i_start, i_end, mult in traj["segments"]:
                all_segments.append((dots[i_start], dots[i_end], mult))

        n_vars = n_pos_vars + 1  # positions + single shared TOF

        # Fixed shape data
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
                shape_data.append((shape_center, shape_pos, shape_vel))

        def get_pos(dot, x_vec):
            if id(dot) in dot_id_to_var_idx:
                idx = dot_id_to_var_idx[id(dot)]
                return x_vec[2 * idx: 2 * idx + 2]
            return np.array([dot.x(), dot.y()])

        def arc_vels(r0, rf, tof):
            p0 = QPointF(r0[0], r0[1])
            p1 = QPointF(rf[0], rf[1])
            return self._compute_segment_velocities(p0, p1, self.earth_center, tof)

        # Mass-leak smoothing so |dv| is differentiable at zero for gradient solvers.
        fuel_eps = 1e-4

        def dv_cost(dv):
            if cost_mode == "energy":
                return np.dot(dv, dv)
            else:
                return math.sqrt(dv[0] * dv[0] + dv[1] * dv[1] + fuel_eps * fuel_eps)

        def objective(x_vec):
            tof = x_vec[n_pos_vars]
            cost = 0.0
            # Build per-node incoming/outgoing velocities
            node_incoming = {}  # id(dot) -> vf
            node_outgoing = {}  # id(dot) -> v0
            for dot_s, dot_e, mult in all_segments:
                r0 = get_pos(dot_s, x_vec)
                rf = get_pos(dot_e, x_vec)
                v0, vf = arc_vels(r0, rf, tof * mult)
                # Outgoing at start node
                if id(dot_s) not in node_outgoing:
                    node_outgoing[id(dot_s)] = v0
                # Incoming at end node
                if id(dot_e) not in node_incoming:
                    node_incoming[id(dot_e)] = vf

            # Delta-v at black nodes with both incoming and outgoing
            for dot in movable_dots:
                d_id = id(dot)
                if d_id in node_incoming and d_id in node_outgoing:
                    dv = node_outgoing[d_id] - node_incoming[d_id]
                    cost += dv_cost(dv)

            # Delta-v at shape nodes
            for shape_center, shape_pos, shape_vel in shape_data:
                s_id = id(shape_center)
                if s_id in node_outgoing:
                    dv = node_outgoing[s_id] - shape_vel
                    cost += dv_cost(dv)
                if s_id in node_incoming:
                    dv = shape_vel - node_incoming[s_id]
                    cost += dv_cost(dv)

            return cost

        # Analytic gradient for the parabolic arc model.
        cg_mode = self.env_mode == "constant_gravity"
        const_g_vec = self._constant_g_vec() if cg_mode else None

        def parabola_gradient(x_vec):
            tof = x_vec[n_pos_vars]
            grad = np.zeros(n_vars)
            g = GRAVITY_MAG
            I2 = np.eye(2)

            # Per-segment velocities and Jacobians wrt (r0, rf, tf).
            seg_data_local = []
            for dot_s, dot_e, mult in all_segments:
                r0 = get_pos(dot_s, x_vec)
                rf = get_pos(dot_e, x_vec)
                tof_seg = tof * mult
                if cg_mode:
                    g_vec = const_g_vec
                    v0 = (rf - r0) / tof_seg - 0.5 * g_vec * tof_seg
                    vf = (rf - r0) / tof_seg + 0.5 * g_vec * tof_seg
                    # dg_vec / d r0 = 0 in constant-gravity mode
                    J_v0_r0 = -I2 / tof_seg
                    J_v0_rf = I2 / tof_seg
                    J_v0_tf = -(rf - r0) / (tof_seg * tof_seg) - 0.5 * g_vec
                    J_vf_r0 = -I2 / tof_seg
                    J_vf_rf = I2 / tof_seg
                    J_vf_tf = -(rf - r0) / (tof_seg * tof_seg) + 0.5 * g_vec
                else:
                    d = center - r0
                    dist = float(np.linalg.norm(d))
                    if dist < 1e-12:
                        v0 = (rf - r0) / tof_seg
                        vf = v0.copy()
                        J_v0_r0 = -I2 / tof_seg
                        J_v0_rf = I2 / tof_seg
                        J_v0_tf = -(rf - r0) / (tof_seg * tof_seg)
                        J_vf_r0 = J_v0_r0
                        J_vf_rf = J_v0_rf
                        J_vf_tf = J_v0_tf
                    else:
                        g_hat = d / dist
                        g_vec = g * g_hat
                        v0 = (rf - r0) / tof_seg - 0.5 * g_vec * tof_seg
                        vf = (rf - r0) / tof_seg + 0.5 * g_vec * tof_seg
                        # d g_hat / d r0 = (1/dist) * (-I + g_hat g_hat^T)
                        dgvec_dr0 = (g / dist) * (-I2 + np.outer(g_hat, g_hat))
                        J_v0_r0 = -I2 / tof_seg - 0.5 * tof_seg * dgvec_dr0
                        J_v0_rf = I2 / tof_seg
                        J_v0_tf = -(rf - r0) / (tof_seg * tof_seg) - 0.5 * g_vec
                        J_vf_r0 = -I2 / tof_seg + 0.5 * tof_seg * dgvec_dr0
                        J_vf_rf = I2 / tof_seg
                        J_vf_tf = -(rf - r0) / (tof_seg * tof_seg) + 0.5 * g_vec
                # Chain rule: d/d tof_var = (d/d tof_seg) * mult
                J_v0_tf = J_v0_tf * mult
                J_vf_tf = J_vf_tf * mult
                seg_data_local.append({
                    "dot_s": dot_s, "dot_e": dot_e,
                    "v0": v0, "vf": vf,
                    "J_v0_r0": J_v0_r0, "J_v0_rf": J_v0_rf, "J_v0_tf": J_v0_tf,
                    "J_vf_r0": J_vf_r0, "J_vf_rf": J_vf_rf, "J_vf_tf": J_vf_tf,
                })

            # Same first-occurrence convention as objective.
            node_outgoing_seg = {}
            node_incoming_seg = {}
            for i, sd in enumerate(seg_data_local):
                s_id = id(sd["dot_s"])
                e_id = id(sd["dot_e"])
                if s_id not in node_outgoing_seg:
                    node_outgoing_seg[s_id] = i
                if e_id not in node_incoming_seg:
                    node_incoming_seg[e_id] = i

            def add_pos_grad(dot, vec2):
                if id(dot) in dot_id_to_var_idx:
                    idx = dot_id_to_var_idx[id(dot)]
                    grad[2 * idx: 2 * idx + 2] += vec2

            def accumulate(dv, contribs):
                if cost_mode == "energy":
                    factor = 2.0
                else:
                    factor = 1.0 / math.sqrt(dv[0] * dv[0] + dv[1] * dv[1] + fuel_eps * fuel_eps)
                for kind, dot, J in contribs:
                    if kind == "pos":
                        add_pos_grad(dot, factor * (dv @ J))
                    else:
                        grad[n_pos_vars] += factor * float(np.dot(dv, J))

            # Black movable midpoint nodes
            for dot in movable_dots:
                d_id = id(dot)
                if d_id in node_incoming_seg and d_id in node_outgoing_seg:
                    sin = seg_data_local[node_incoming_seg[d_id]]
                    sout = seg_data_local[node_outgoing_seg[d_id]]
                    dv = sout["v0"] - sin["vf"]
                    contribs = [
                        ("pos", sin["dot_s"], -sin["J_vf_r0"]),
                        ("pos", dot, sout["J_v0_r0"] - sin["J_vf_rf"]),
                        ("pos", sout["dot_e"], sout["J_v0_rf"]),
                        ("tof", None, sout["J_v0_tf"] - sin["J_vf_tf"]),
                    ]
                    accumulate(dv, contribs)

            # Shape nodes (r_shape is fixed)
            for shape_center, shape_pos, shape_vel in shape_data:
                s_id = id(shape_center)
                if s_id in node_outgoing_seg:
                    sout = seg_data_local[node_outgoing_seg[s_id]]
                    dv = sout["v0"] - shape_vel
                    contribs = [
                        ("pos", sout["dot_e"], sout["J_v0_rf"]),
                        ("tof", None, sout["J_v0_tf"]),
                    ]
                    accumulate(dv, contribs)
                if s_id in node_incoming_seg:
                    sin = seg_data_local[node_incoming_seg[s_id]]
                    dv = shape_vel - sin["vf"]
                    contribs = [
                        ("pos", sin["dot_s"], -sin["J_vf_r0"]),
                        ("tof", None, -sin["J_vf_tf"]),
                    ]
                    accumulate(dv, contribs)

            return grad

        def lambert_gradient(x_vec):
            """Analytic gradient for the conic (Lambert) arc model in two-body env.
            Mirrors parabola_gradient but uses lambert_solve_with_jac per segment.
            """
            tof = x_vec[n_pos_vars]
            grad = np.zeros(n_vars)

            seg_data_local = []
            for dot_s, dot_e, mult in all_segments:
                r0 = get_pos(dot_s, x_vec)
                rf = get_pos(dot_e, x_vec)
                v0, vf, J_v0_r0, J_v0_rf, J_v0_tf, J_vf_r0, J_vf_rf, J_vf_tf = (
                    lambert_solve_with_jac(r0, rf, tof * mult, MU)
                )
                # Chain rule: d/d tof_var = (d/d tof_seg) * mult
                J_v0_tf = J_v0_tf * mult
                J_vf_tf = J_vf_tf * mult
                seg_data_local.append({
                    "dot_s": dot_s, "dot_e": dot_e,
                    "v0": v0, "vf": vf,
                    "J_v0_r0": J_v0_r0, "J_v0_rf": J_v0_rf, "J_v0_tf": J_v0_tf,
                    "J_vf_r0": J_vf_r0, "J_vf_rf": J_vf_rf, "J_vf_tf": J_vf_tf,
                })

            node_outgoing_seg = {}
            node_incoming_seg = {}
            for i, sd in enumerate(seg_data_local):
                s_id = id(sd["dot_s"])
                e_id = id(sd["dot_e"])
                if s_id not in node_outgoing_seg:
                    node_outgoing_seg[s_id] = i
                if e_id not in node_incoming_seg:
                    node_incoming_seg[e_id] = i

            def add_pos_grad(dot, vec2):
                if id(dot) in dot_id_to_var_idx:
                    idx = dot_id_to_var_idx[id(dot)]
                    grad[2 * idx: 2 * idx + 2] += vec2

            def accumulate(dv, contribs):
                if cost_mode == "energy":
                    factor = 2.0
                else:
                    factor = 1.0 / math.sqrt(dv[0] * dv[0] + dv[1] * dv[1] + fuel_eps * fuel_eps)
                for kind, dot, J in contribs:
                    if kind == "pos":
                        add_pos_grad(dot, factor * (dv @ J))
                    else:
                        grad[n_pos_vars] += factor * float(np.dot(dv, J))

            for dot in movable_dots:
                d_id = id(dot)
                if d_id in node_incoming_seg and d_id in node_outgoing_seg:
                    sin = seg_data_local[node_incoming_seg[d_id]]
                    sout = seg_data_local[node_outgoing_seg[d_id]]
                    dv = sout["v0"] - sin["vf"]
                    contribs = [
                        ("pos", sin["dot_s"], -sin["J_vf_r0"]),
                        ("pos", dot, sout["J_v0_r0"] - sin["J_vf_rf"]),
                        ("pos", sout["dot_e"], sout["J_v0_rf"]),
                        ("tof", None, sout["J_v0_tf"] - sin["J_vf_tf"]),
                    ]
                    accumulate(dv, contribs)

            for shape_center, shape_pos, shape_vel in shape_data:
                s_id = id(shape_center)
                if s_id in node_outgoing_seg:
                    sout = seg_data_local[node_outgoing_seg[s_id]]
                    dv = sout["v0"] - shape_vel
                    contribs = [
                        ("pos", sout["dot_e"], sout["J_v0_rf"]),
                        ("tof", None, sout["J_v0_tf"]),
                    ]
                    accumulate(dv, contribs)
                if s_id in node_incoming_seg:
                    sin = seg_data_local[node_incoming_seg[s_id]]
                    dv = shape_vel - sin["vf"]
                    contribs = [
                        ("pos", sin["dot_s"], -sin["J_vf_r0"]),
                        ("tof", None, -sin["J_vf_tf"]),
                    ]
                    accumulate(dv, contribs)

            return grad

        # Initial guess: positions + shared TOF
        x0 = np.zeros(n_vars)
        for i, dot in enumerate(movable_dots):
            x0[2 * i] = dot.x()
            x0[2 * i + 1] = dot.y()
        x0[n_pos_vars] = TIME_OF_FLIGHT

        # BFGS is unconstrained; pick the analytic gradient that matches the model.
        use_parabola = self.env_mode == "constant_gravity" or self.arc_model_mode == "parabola"
        if use_parabola:
            result = scipy_minimize(objective, x0, method="BFGS", jac=parabola_gradient)
        else:
            result = scipy_minimize(objective, x0, method="BFGS", jac=lambert_gradient)

        # Apply result: update positions and rendering tof
        for i, dot in enumerate(movable_dots):
            dot.setX(result.x[2 * i])
            dot.setY(result.x[2 * i + 1])
        self.render_tof = float(result.x[n_pos_vars])

        # Recompute orbits if active
        if self.tri_orbit_mode:
            self._compute_orbit_for_shape("triangle")
        if self.sq_orbit_mode:
            self._compute_orbit_for_shape("square")
        self.update()

    def optimize_energy(self):
        self._optimize_common("energy")

    def optimize_fuel(self):
        self._optimize_common("fuel")

    def set_optimize_mode(self, mode):
        """mode in {None, 'energy', 'fuel'}. When set, runs once immediately
        and then on every subsequent drag release."""
        assert mode in (None, "energy", "fuel")
        self.optimize_mode = mode
        if mode is not None:
            self._optimize_common(mode)

    def _run_active_optimizer(self):
        if self.optimize_mode is not None:
            self._optimize_common(self.optimize_mode)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trajectory Builder")
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

        arc_model_btn = QPushButton(canvas.get_arc_model_button_text())
        arc_model_btn.setFixedSize(155, 25)

        def toggle_arc_model_button():
            canvas.toggle_arc_model_mode()
            arc_model_btn.setText(canvas.get_arc_model_button_text())

        arc_model_btn.clicked.connect(toggle_arc_model_button)

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
        top_layout.addWidget(arc_model_btn)
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

        zoom_layout = QHBoxLayout()
        zoom_layout.addStretch()
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addSpacing(10)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(top_layout)
        layout.addWidget(canvas)
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
