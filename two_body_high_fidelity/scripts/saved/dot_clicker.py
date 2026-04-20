import sys

import math

import numpy as np

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen, QPolygonF
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout

GRAVITY_MAG = 9.81  # m/s^2, constant magnitude
TIME_OF_FLIGHT = 5.0  # seconds
ARC_NUM_POINTS = 50  # number of points to draw the parabolic arc
ORBIT_NUM_POINTS = 200  # number of points to draw a Kepler orbit


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


def compute_dynamic_arc(p0, p1, center, time_of_flight, num_points):
    """Compute parabolic arc points between p0 and p1 under constant gravity
    directed from p0 toward center."""
    r0 = np.array([p0.x(), p0.y()])
    rf = np.array([p1.x(), p1.y()])
    c = np.array([center.x(), center.y()])

    # Gravity direction fixed at initial node
    direction = c - r0
    dist = np.linalg.norm(direction)
    if dist < 1e-10:
        # Node is at center; fall back to straight line
        return [p0, p1]

    g_hat = direction / dist
    g_vec = GRAVITY_MAG * g_hat

    # v0 = (rf - r0) / tf - 0.5 * g * tf
    tf = time_of_flight
    v0 = (rf - r0) / tf - 0.5 * g_vec * tf

    # Sample the trajectory
    points = []
    for t in np.linspace(0, tf, num_points):
        pos = r0 + v0 * t + 0.5 * g_vec * t * t
        points.append(QPointF(pos[0], pos[1]))
    return points


def compute_arc_velocities(p0, p1, center, time_of_flight):
    """Return (v0, vf) numpy arrays for the dynamic arc from p0 to p1."""
    r0 = np.array([p0.x(), p0.y()])
    rf = np.array([p1.x(), p1.y()])
    c = np.array([center.x(), center.y()])
    direction = c - r0
    dist = np.linalg.norm(direction)
    if dist < 1e-10:
        disp = rf - r0
        v = disp / time_of_flight
        return v, v
    g_hat = direction / dist
    g_vec = GRAVITY_MAG * g_hat
    tf = time_of_flight
    v0 = (rf - r0) / tf - 0.5 * g_vec * tf
    vf = v0 + g_vec * tf
    return v0, vf


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.trajectories = []  # list of {"dots": [QPointF], "segments": [(i, j)]}
        self.dot_radius = 5
        self.trace_spacing = 50
        self.dragging = False
        self.dragging_shape = None  # "triangle" or "square" when dragging a shape
        self.drag_offset = QPointF(0, 0)
        self.dragging_dot = None  # QPointF reference when dragging a black node
        self.shift_click_first = None  # first node selected for shift-click linking
        self.v_dragging = None  # "triangle" or "square" when drawing a velocity line
        self.dragging_vel_end = None  # "triangle" or "square" when dragging a velocity line
        self.dragging_vel_t = 1.0  # parameter along line where grab occurred (0=center, 1=tip)
        self.tri_velocity_end = None  # QPointF end of triangle velocity line
        self.sq_velocity_end = None  # QPointF end of square velocity line
        self.tri_orbit = []  # list of QPointF for triangle Kepler orbit
        self.sq_orbit = []  # list of QPointF for square Kepler orbit
        self.tri_orbit_mode = False
        self.sq_orbit_mode = False
        self._o_held = False

        # Shape positions (center points)
        self.tri_center = QPointF(25, 20)
        self.tri_size = 20
        self.sq_center = QPointF(55, 20)
        self.sq_size = 18

        self.setStyleSheet("background-color: white;")
        self._v_held = False
        self.zoom = 1.0
        self.pan_offset = QPointF(0, 0)
        self.earth_center = QPointF(600, 450)
        self.earth_radius = 50

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_V:
            self._v_held = True
        elif event.key() == Qt.Key.Key_O:
            self._o_held = True

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_V:
            self._v_held = False
        elif event.key() == Qt.Key.Key_O:
            self._o_held = False

    def _screen_to_world(self, screen_pos):
        wx = (screen_pos.x() - self.pan_offset.x()) / self.zoom
        wy = (screen_pos.y() - self.pan_offset.y()) / self.zoom
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

    def _find_nearest_dot(self, pos, max_dist=15):
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
            ])
            mu = 3.986e5  # km^3/s^2, Earth gravitational parameter
            self.tri_orbit = compute_kepler_orbit(
                self.tri_center, vel, self.earth_center, mu, ORBIT_NUM_POINTS,
            )
        elif shape == "square" and self.sq_velocity_end is not None:
            vel = np.array([
                self.sq_velocity_end.x() - self.sq_center.x(),
                self.sq_velocity_end.y() - self.sq_center.y(),
            ])
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
                if self._v_held:
                    self.v_dragging = "triangle"
                    self.tri_velocity_end = QPointF(pos)
                    return
                if self._o_held:
                    self.tri_orbit_mode = not self.tri_orbit_mode
                    if self.tri_orbit_mode:
                        self._compute_orbit_for_shape("triangle")
                    else:
                        self.tri_orbit = []
                    self.update()
                    return
                self.dragging_shape = "triangle"
                self.drag_offset = self.tri_center - pos
                return
            if self._hit_square(pos):
                if self._v_held:
                    self.v_dragging = "square"
                    self.sq_velocity_end = QPointF(pos)
                    return
                if self._o_held:
                    self.sq_orbit_mode = not self.sq_orbit_mode
                    if self.sq_orbit_mode:
                        self._compute_orbit_for_shape("square")
                    else:
                        self.sq_orbit = []
                    self.update()
                    return
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
                            if d < 10:
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
        # Handle velocity line dragging
        if self.v_dragging == "triangle":
            self.tri_velocity_end = pos
            if self.tri_orbit_mode:
                self._compute_orbit_for_shape("triangle")
            self.update()
            return
        if self.v_dragging == "square":
            self.sq_velocity_end = pos
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
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._screen_to_world(QPointF(event.pos()))
            if self._hit_triangle(pos):
                self.tri_orbit_mode = not self.tri_orbit_mode
                if self.tri_orbit_mode:
                    self._compute_orbit_for_shape("triangle")
                else:
                    self.tri_orbit = []
                self.update()
                return
            if self._hit_square(pos):
                self.sq_orbit_mode = not self.sq_orbit_mode
                if self.sq_orbit_mode:
                    self._compute_orbit_for_shape("square")
                else:
                    self.sq_orbit = []
                self.update()
                return

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging_vel_end:
                self.dragging_vel_end = None
                return
            if self.v_dragging:
                self.v_dragging = None
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

        # Apply pan and zoom
        painter.translate(self.pan_offset.x(), self.pan_offset.y())
        painter.scale(self.zoom, self.zoom)

        # Draw Kepler orbits
        painter.setPen(QPen(QColor("green"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for orbit in (self.tri_orbit, self.sq_orbit):
            if len(orbit) > 1:
                for k in range(len(orbit) - 1):
                    painter.drawLine(orbit[k], orbit[k + 1])

        # Draw velocity lines from triangle/square
        painter.setPen(QPen(QColor("green"), 2))
        if self.tri_velocity_end is not None:
            painter.drawLine(self.tri_center, self.tri_velocity_end)
        if self.sq_velocity_end is not None:
            painter.drawLine(self.sq_center, self.sq_velocity_end)

        # Draw blue circle (Earth)
        painter.setBrush(QBrush(QColor("blue")))
        painter.setPen(QPen(QColor("blue"), 2))
        painter.drawEllipse(self.earth_center, self.earth_radius, self.earth_radius)

        # Draw trajectories
        center = self.earth_center
        for traj in self.trajectories:
            dots = traj["dots"]
            segments = traj["segments"]

            # Draw segments
            for i_start, i_end in segments:
                painter.setPen(QPen(QColor("black"), 2))
                arc_points = compute_dynamic_arc(
                    dots[i_start], dots[i_end], center,
                    TIME_OF_FLIGHT, ARC_NUM_POINTS,
                )
                for j in range(len(arc_points) - 1):
                    painter.drawLine(arc_points[j], arc_points[j + 1])

        # Draw delta-velocity vectors at nodes with exactly 2 neighboring segments
        painter.setPen(QPen(QColor("black"), 2))
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
                        QPointF(dot.x() + dv[0], dot.y() + dv[1]),
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
            ])
            for traj in self.trajectories:
                dots = traj["dots"]
                for i_start, i_end in traj["segments"]:
                    if dots[i_start] is shape_center:
                        v0, _ = compute_arc_velocities(
                            dots[i_start], dots[i_end], center, TIME_OF_FLIGHT)
                        dv = v0 - shape_vel
                        painter.drawLine(
                            shape_center,
                            QPointF(shape_center.x() + dv[0], shape_center.y() + dv[1]),
                        )
                    elif dots[i_end] is shape_center:
                        _, vf = compute_arc_velocities(
                            dots[i_start], dots[i_end], center, TIME_OF_FLIGHT)
                        dv = shape_vel - vf
                        painter.drawLine(
                            shape_center,
                            QPointF(shape_center.x() + dv[0], shape_center.y() + dv[1]),
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

        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(clear_button)

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
