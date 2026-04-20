import sys

import math

import numpy as np
from scipy.optimize import brentq, minimize as scipy_minimize

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout

GRAVITY_MAG = 9.81  # m/s^2, constant magnitude
TIME_OF_FLIGHT = 3600.0  # seconds (1 hour)
ARC_NUM_POINTS = 50  # number of points to draw the parabolic arc
ORBIT_NUM_POINTS = 200  # number of points to draw a Kepler orbit
MU = 3.986e5  # gravitational parameter for Lambert arcs (consistent with Kepler orbits)
VEL_SCALE = 1000.0  # display scale: velocity_end = center + vel * VEL_SCALE


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


# --- Old constant-gravity arc functions (commented out) ---
# def compute_dynamic_arc(p0, p1, center, time_of_flight, num_points):
#     """Compute parabolic arc points between p0 and p1 under constant gravity
#     directed from p0 toward center."""
#     r0 = np.array([p0.x(), p0.y()])
#     rf = np.array([p1.x(), p1.y()])
#     c = np.array([center.x(), center.y()])
#     direction = c - r0
#     dist = np.linalg.norm(direction)
#     if dist < 1e-10:
#         return [p0, p1]
#     g_hat = direction / dist
#     g_vec = GRAVITY_MAG * g_hat
#     tf = time_of_flight
#     v0 = (rf - r0) / tf - 0.5 * g_vec * tf
#     points = []
#     for t in np.linspace(0, tf, num_points):
#         pos = r0 + v0 * t + 0.5 * g_vec * t * t
#         points.append(QPointF(pos[0], pos[1]))
#     return points
#
# def compute_arc_velocities(p0, p1, center, time_of_flight):
#     """Return (v0, vf) numpy arrays for the dynamic arc from p0 to p1."""
#     r0 = np.array([p0.x(), p0.y()])
#     rf = np.array([p1.x(), p1.y()])
#     c = np.array([center.x(), center.y()])
#     direction = c - r0
#     dist = np.linalg.norm(direction)
#     if dist < 1e-10:
#         disp = rf - r0
#         v = disp / time_of_flight
#         return v, v
#     g_hat = direction / dist
#     g_vec = GRAVITY_MAG * g_hat
#     tf = time_of_flight
#     v0 = (rf - r0) / tf - 0.5 * g_vec * tf
#     vf = v0 + g_vec * tf
#     return v0, vf
# --- End old functions ---


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
        self.trajectories = []  # list of {"dots": [QPointF], "segments": [(i, j)]}
        self.dot_radius = 637
        self.trace_spacing = 6371
        self.dragging = False
        self.dragging_shape = None  # "triangle" or "square" when dragging a shape
        self.drag_offset = QPointF(0, 0)
        self.dragging_dot = None  # QPointF reference when dragging a black node
        self.shift_click_first = None  # first node selected for shift-click linking
        self.dragging_vel_end = None  # "triangle" or "square" when dragging a velocity line
        self.dragging_vel_t = 1.0  # parameter along line where grab occurred (0=center, 1=tip)
        # Shape positions (center points) — placed in orbit around earth
        self.earth_center = QPointF(0, 0)
        self.earth_radius = 6371
        self.tri_center = QPointF(9550, 0)
        self.tri_size = 2548
        self.sq_center = QPointF(-9550, 0)
        self.sq_size = 2293

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

        self.setStyleSheet("background-color: white;")
        self.zoom = 50.0 / 6371.0
        self.pan_offset = QPointF(600, 450)

        # Compute initial orbits
        self._compute_orbit_for_shape("triangle")
        self._compute_orbit_for_shape("square")

    def keyPressEvent(self, event):
        pass

    def keyReleaseEvent(self, event):
        pass

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

    def _find_nearest_dot(self, pos, max_dist=1911):
        """Find the nearest dot (including triangle/square centers) across all
        trajectories within max_dist pixels.
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
            for i_start, i_end in traj["segments"]:
                if (dots[i_start] is p0 and dots[i_end] is p1) or (dots[i_start] is p1 and dots[i_end] is p0):
                    return True
        return False

    def _compute_orbit_for_shape(self, shape):
        """Compute Kepler orbit for triangle or square if it has a velocity vector."""
        if shape == "triangle" and self.tri_velocity_end is not None:
            vel = np.array([
                self.tri_velocity_end.x() - self.tri_center.x(),
                self.tri_velocity_end.y() - self.tri_center.y(),
            ]) / VEL_SCALE
            mu = 3.986e5  # km^3/s^2, Earth gravitational parameter
            self.tri_orbit = compute_kepler_orbit(
                self.tri_center, vel, self.earth_center, mu, ORBIT_NUM_POINTS,
            )
        elif shape == "square" and self.sq_velocity_end is not None:
            vel = np.array([
                self.sq_velocity_end.x() - self.sq_center.x(),
                self.sq_velocity_end.y() - self.sq_center.y(),
            ]) / VEL_SCALE
            mu = 3.986e5
            self.sq_orbit = compute_kepler_orbit(
                self.sq_center, vel, self.earth_center, mu, ORBIT_NUM_POINTS,
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._screen_to_world(QPointF(event.pos()))
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
                                target_traj = {"dots": [first_dot, dot], "segments": [(0, 1)]}
                                self.trajectories.append(target_traj)
                            else:
                                # Add dots if not already present, then add segment
                                if first_dot not in target_traj["dots"]:
                                    target_traj["dots"].append(first_dot)
                                if dot not in target_traj["dots"]:
                                    target_traj["dots"].append(dot)
                                i0 = target_traj["dots"].index(first_dot)
                                i1 = target_traj["dots"].index(dot)
                                target_traj["segments"].append((i0, i1))
                            self.update()
                        self.shift_click_first = None
                return
            # Check shapes for dragging, V-line, or O-orbit
            if self._hit_triangle(pos):
                self.dragging_shape = "triangle"
                self.drag_offset = self.tri_center - pos
                return
            if self._hit_square(pos):
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
                            if d < 1911:
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
                self.dragging = True
                self.trajectories.append({"dots": [], "segments": []})
                self.trajectories[-1]["dots"].append(self._screen_to_world(QPointF(event.pos())))
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
                     len(self.trajectories[-1]["dots"]))
                )
                self.trajectories[-1]["dots"].append(interp)
            self.update()

    def mouseDoubleClickEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging_vel_end:
                self.dragging_vel_end = None
                return
            if self.dragging_dot is not None:
                self.dragging_dot = None
                return
            if self.dragging_shape:
                self.dragging_shape = None
                return
            if self.dragging:
                self.dragging = False

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
                self.pan_offset.x() + delta.x(),
                self.pan_offset.y() + delta.y(),
            )
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Apply pan and zoom (y-up convention: negative y-scale flips vertical)
        painter.translate(self.pan_offset.x(), self.pan_offset.y())
        painter.scale(self.zoom, -self.zoom)

        # Draw Kepler orbits
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

        # Draw blue circle (Earth)
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
            for i_start, i_end in segments:
                pen = QPen(QColor("black"), 2)
                pen.setCosmetic(True)
                painter.setPen(pen)
                arc_points = compute_dynamic_arc(
                    dots[i_start], dots[i_end], center,
                    TIME_OF_FLIGHT, ARC_NUM_POINTS,
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
                for i_start, i_end in segments:
                    if i_end == k:
                        _, vf = compute_arc_velocities(
                            dots[i_start], dots[i_end], center, TIME_OF_FLIGHT)
                        incoming_vel = vf
                    elif i_start == k:
                        v0, _ = compute_arc_velocities(
                            dots[i_start], dots[i_end], center, TIME_OF_FLIGHT)
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
                for i_start, i_end in traj["segments"]:
                    if dots[i_start] is shape_center:
                        v0, _ = compute_arc_velocities(
                            dots[i_start], dots[i_end], center, TIME_OF_FLIGHT)
                        dv = v0 - shape_vel
                        painter.drawLine(
                            shape_center,
                            QPointF(shape_center.x() + dv[0] * VEL_SCALE, shape_center.y() + dv[1] * VEL_SCALE),
                        )
                    elif dots[i_end] is shape_center:
                        _, vf = compute_arc_velocities(
                            dots[i_start], dots[i_end], center, TIME_OF_FLIGHT)
                        dv = shape_vel - vf
                        painter.drawLine(
                            shape_center,
                            QPointF(shape_center.x() + dv[0] * VEL_SCALE, shape_center.y() + dv[1] * VEL_SCALE),
                        )

        # Draw all dots on top of everything
        painter.setBrush(QBrush(QColor("black")))
        painter.setPen(Qt.PenStyle.NoPen)
        for traj in self.trajectories:
            for dot in traj["dots"]:
                if dot is self.tri_center or dot is self.sq_center:
                    continue
                painter.drawEllipse(dot, self.dot_radius, self.dot_radius)

        # Draw green triangle and square on top of everything
        painter.setBrush(QBrush(QColor("green")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(self._triangle_polygon())
        sq_x = self.sq_center.x() - self.sq_size / 2
        sq_y = self.sq_center.y() - self.sq_size / 2
        painter.drawRect(int(sq_x), int(sq_y), self.sq_size, self.sq_size)

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
            for i_start, i_end in traj["segments"]:
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

        # Build segment list as (dot_start, dot_end) with references
        all_segments = []
        for traj in self.trajectories:
            dots = traj["dots"]
            for i_start, i_end in traj["segments"]:
                all_segments.append((dots[i_start], dots[i_end]))

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
            r1_rel = r0 - center
            r2_rel = rf - center
            if np.linalg.norm(r1_rel) < 1e-10 or np.linalg.norm(r2_rel) < 1e-10:
                v = (rf - r0) / tof
                return v, v
            return lambert_solve(r1_rel, r2_rel, tof, MU)

        def dv_cost(dv):
            if cost_mode == "energy":
                return np.dot(dv, dv)
            else:
                return np.linalg.norm(dv)

        def objective(x_vec):
            tof = x_vec[n_pos_vars]
            cost = 0.0
            # Build per-node incoming/outgoing velocities
            node_incoming = {}  # id(dot) -> vf
            node_outgoing = {}  # id(dot) -> v0
            for dot_s, dot_e in all_segments:
                r0 = get_pos(dot_s, x_vec)
                rf = get_pos(dot_e, x_vec)
                v0, vf = arc_vels(r0, rf, tof)
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

        # Initial guess: positions + shared TOF
        x0 = np.zeros(n_vars)
        for i, dot in enumerate(movable_dots):
            x0[2 * i] = dot.x()
            x0[2 * i + 1] = dot.y()
        x0[n_pos_vars] = TIME_OF_FLIGHT

        # Bounds: no bounds on positions, TOF bounded to [60, 36000] seconds
        bounds = [(None, None)] * n_pos_vars + [(60.0, 36000.0)]

        result = scipy_minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

        # Apply result: update positions
        for i, dot in enumerate(movable_dots):
            dot.setX(result.x[2 * i])
            dot.setY(result.x[2 * i + 1])

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

        optimize_energy_btn = QPushButton("Optimize: Min Energy")
        optimize_energy_btn.setFixedSize(150, 25)
        optimize_energy_btn.clicked.connect(canvas.optimize_energy)

        optimize_fuel_btn = QPushButton("Optimize: Min Fuel")
        optimize_fuel_btn.setFixedSize(130, 25)
        optimize_fuel_btn.clicked.connect(canvas.optimize_fuel)

        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(optimize_energy_btn)
        top_layout.addSpacing(10)
        top_layout.addWidget(optimize_fuel_btn)
        top_layout.addSpacing(10)
        top_layout.addWidget(clear_button)
        top_layout.addSpacing(20)

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
        zoom_layout.addSpacing(20)

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
