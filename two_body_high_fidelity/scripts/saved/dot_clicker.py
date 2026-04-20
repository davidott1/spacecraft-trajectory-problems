import sys

import math

import numpy as np

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout

GRAVITY_MAG = 9.81  # m/s^2, constant magnitude
TIME_OF_FLIGHT = 5.0  # seconds
ARC_NUM_POINTS = 50  # number of points to draw the parabolic arc


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


class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.trajectories = []  # list of {"dots": [], "segment_types": []}
        self.segment_mode = "straight"  # current mode: "straight" or "dynamic"
        self.dot_radius = 5
        self.trace_spacing = 50
        self.dragging = False
        self.setStyleSheet("background-color: white;")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S:
            self.segment_mode = "straight"
            self.window().setWindowTitle("Dot Clicker — Mode: Straight")
        elif event.key() == Qt.Key.Key_D:
            self.segment_mode = "dynamic"
            self.window().setWindowTitle("Dot Clicker — Mode: Dynamic")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.dragging = True
            # Start a new trajectory
            self.trajectories.append({"dots": [], "segment_types": []})
            self.trajectories[-1]["dots"].append(QPointF(event.pos()))
            self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging:
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.dragging = False
            return
        traj = self.trajectories[-1]
        last = traj["dots"][-1]
        pos = event.pos()
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
                self.trajectories[-1]["segment_types"].append(self.segment_mode)
                self.trajectories[-1]["dots"].append(interp)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            self.dragging = False
            traj = self.trajectories[-1]
            if len(traj["dots"]) > 1:
                traj["segment_types"].append(self.segment_mode)
                traj["dots"].append(QPointF(traj["dots"][0]))
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw blue circle in center
        center_x = self.width() / 2
        center_y = self.height() / 2
        circle_radius = 50
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("blue"), 2))
        painter.drawEllipse(QPointF(center_x, center_y), circle_radius, circle_radius)

        # Draw trajectories
        center = QPointF(center_x, center_y)
        for traj in self.trajectories:
            dots = traj["dots"]
            seg_types = traj["segment_types"]

            # Draw segments
            if len(dots) > 1:
                for i in range(len(dots) - 1):
                    seg_type = seg_types[i] if i < len(seg_types) else "straight"
                    if seg_type == "straight":
                        painter.setPen(QPen(QColor("black"), 2))
                        painter.drawLine(dots[i], dots[i + 1])
                    else:  # dynamic
                        painter.setPen(QPen(QColor("red"), 2))
                        arc_points = compute_dynamic_arc(
                            dots[i], dots[i + 1], center,
                            TIME_OF_FLIGHT, ARC_NUM_POINTS,
                        )
                        for j in range(len(arc_points) - 1):
                            painter.drawLine(arc_points[j], arc_points[j + 1])

            # Draw dots on top
            painter.setBrush(QBrush(QColor("black")))
            painter.setPen(Qt.PenStyle.NoPen)
            for dot in dots:
                painter.drawEllipse(dot, self.dot_radius, self.dot_radius)
        painter.end()


    def clear(self):
        self.trajectories.clear()
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dot Clicker — Mode: Straight")
        self.setMinimumSize(800, 600)

        canvas = Canvas()

        clear_button = QPushButton("Clear")
        clear_button.setFixedHeight(30)
        clear_button.clicked.connect(canvas.clear)

        layout = QVBoxLayout()
        layout.addWidget(clear_button)
        layout.addWidget(canvas)

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
