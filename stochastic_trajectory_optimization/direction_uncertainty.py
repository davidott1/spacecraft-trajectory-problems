"""
Direction Uncertainty on a Unit Sphere

Investigating how direction errors propagate and are represented.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Nominal direction: x-axis
nominal_dir = np.array([1.0, 0.0, 0.0])

# Perpendicular axes for yaw and pitch
e_yaw = np.array([0.0, 1.0, 0.0])    # Y-axis: yaw (side to side)
e_pitch = np.array([0.0, 0.0, 1.0])  # Z-axis: pitch (up and down)

# Discretize angles from -90 to 90 degrees in 5° increments
angles = np.arange(-90, 95, 5)  # -90, -85, -80, ..., 80, 85, 90

# Rotation matrices
def Rz(theta):
    """Rotation about Z-axis (yaw)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Ry(theta):
    """Rotation about Y-axis (pitch / new-longitude)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

# Generate yaw-only directions (rotation about Z, great circles through ±Y)
yaw_directions = []
for yaw_deg in angles:
    yaw_rad = np.radians(yaw_deg)
    d = Rz(yaw_rad) @ nominal_dir
    yaw_directions.append(d)
yaw_directions = np.array(yaw_directions)

# Generate pitch directions (rotation about Y, great circles through ±Z)
pitch_directions = []
for pitch_deg in angles:
    pitch_rad = np.radians(pitch_deg)
    d = Ry(pitch_rad) @ nominal_dir
    pitch_directions.append(d)
pitch_directions = np.array(pitch_directions)

# Generate full grid using tangent plane projection (gnomonic)
# This maintains uniform spacing - both directions are "longitude-like"
# d = normalize(nominal + tan(yaw)*e_y + tan(pitch)*e_z)
# where e_y and e_z are tangent directions at nominal=[1,0,0]
grid_points = []
for yaw_deg in angles:
    for pitch_deg in angles:
        yaw_rad = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)
        # Tangent plane offset, then project to sphere
        d = nominal_dir + np.tan(yaw_rad) * e_yaw + np.tan(pitch_rad) * e_pitch
        d = d / np.linalg.norm(d)  # Project onto unit sphere
        grid_points.append(d)
grid_points = np.array(grid_points)

# Generate lat/lon grid (longitude = yaw about Z, latitude = elevation from XY plane)
# d = [cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)]
latlon_points = []
for lon_deg in angles:  # longitude (like yaw)
    for lat_deg in angles:  # latitude (elevation)
        lon_rad = np.radians(lon_deg)
        lat_rad = np.radians(lat_deg)
        d = np.array([np.cos(lat_rad) * np.cos(lon_rad),
                      np.cos(lat_rad) * np.sin(lon_rad),
                      np.sin(lat_rad)])
        latlon_points.append(d)
latlon_points = np.array(latlon_points)

# Generate axis-angle grid (Rodrigues rotation for uniform angular spacing)
axis_angle_points = []
for yaw_deg in angles:
    for pitch_deg in angles:
        yaw_rad = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)
        theta = np.sqrt(yaw_rad**2 + pitch_rad**2)
        if theta < 1e-10:
            d = nominal_dir.copy()
        else:
            axis = (yaw_rad * e_pitch - pitch_rad * e_yaw) / theta
            d = (nominal_dir * np.cos(theta) + 
                 np.cross(axis, nominal_dir) * np.sin(theta) + 
                 axis * np.dot(axis, nominal_dir) * (1 - np.cos(theta)))
        axis_angle_points.append(d)
axis_angle_points = np.array(axis_angle_points)

# --- Function to compute grid cell areas (flat approximation) ---
def compute_cell_areas_flat(points, n_angles):
    """Compute approximate FLAT area of each grid cell using cross product."""
    areas = []
    for i in range(n_angles):
        for j in range(n_angles):
            idx = i * n_angles + j
            i_prev = max(0, i - 1)
            i_next = min(n_angles - 1, i + 1)
            j_prev = max(0, j - 1)
            j_next = min(n_angles - 1, j + 1)
            
            p_left = points[i_prev * n_angles + j]
            p_right = points[i_next * n_angles + j]
            p_down = points[i * n_angles + j_prev]
            p_up = points[i * n_angles + j_next]
            
            v1 = p_right - p_left
            v2 = p_up - p_down
            area = np.linalg.norm(np.cross(v1, v2)) / 4.0
            areas.append(area)
    return np.array(areas)

# --- Function to compute spherical triangle area ---
def spherical_triangle_area(p1, p2, p3):
    """Compute solid angle of spherical triangle using spherical excess formula.
    
    For unit sphere, area = spherical excess = sum of angles - pi
    Uses the formula: tan(E/2) = |p1 · (p2 × p3)| / (1 + p1·p2 + p2·p3 + p3·p1)
    """
    # Triple product gives signed volume of parallelepiped
    triple = np.dot(p1, np.cross(p2, p3))
    
    # Dot products
    d12 = np.dot(p1, p2)
    d23 = np.dot(p2, p3)
    d31 = np.dot(p3, p1)
    
    # Spherical excess formula (L'Huilier's theorem variant)
    denom = 1.0 + d12 + d23 + d31
    if abs(denom) < 1e-10:
        return 0.0
    
    tan_half_E = abs(triple) / denom
    E = 2.0 * np.arctan(tan_half_E)
    return E  # For unit sphere, area = E (solid angle in steradians)

# --- Function to compute grid cell areas (true spherical) ---
def compute_cell_areas_spherical(points, n_angles):
    """Compute true SPHERICAL area (solid angle) of each grid cell."""
    areas = []
    for i in range(n_angles):
        for j in range(n_angles):
            idx = i * n_angles + j
            i_prev = max(0, i - 1)
            i_next = min(n_angles - 1, i + 1)
            j_prev = max(0, j - 1)
            j_next = min(n_angles - 1, j + 1)
            
            # Get the 4 corner points of the cell
            p_center = points[idx]
            p_left = points[i_prev * n_angles + j]
            p_right = points[i_next * n_angles + j]
            p_down = points[i * n_angles + j_prev]
            p_up = points[i * n_angles + j_next]
            
            # Midpoints to define cell corners
            c1 = (p_center + p_left + p_down) / 3.0
            c1 = c1 / np.linalg.norm(c1)
            c2 = (p_center + p_right + p_down) / 3.0
            c2 = c2 / np.linalg.norm(c2)
            c3 = (p_center + p_right + p_up) / 3.0
            c3 = c3 / np.linalg.norm(c3)
            c4 = (p_center + p_left + p_up) / 3.0
            c4 = c4 / np.linalg.norm(c4)
            
            # Split quadrilateral into 2 triangles and sum areas
            area = spherical_triangle_area(c1, c2, c3) + spherical_triangle_area(c1, c3, c4)
            areas.append(area)
    return np.array(areas)

n_angles = len(angles)
# Flat areas
latlon_areas_flat = compute_cell_areas_flat(latlon_points, n_angles)
gnomonic_areas_flat = compute_cell_areas_flat(grid_points, n_angles)
axisangle_areas_flat = compute_cell_areas_flat(axis_angle_points, n_angles)
# Spherical areas
latlon_areas_sph = compute_cell_areas_spherical(latlon_points, n_angles)
gnomonic_areas_sph = compute_cell_areas_spherical(grid_points, n_angles)
axisangle_areas_sph = compute_cell_areas_spherical(axis_angle_points, n_angles)

# Create figure with 3x3 subplots
fig = plt.figure(figsize=(18, 14))

# Draw unit sphere (wireframe) - shared data
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 25)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
axis_len = 1.3

# --- Subplot 1: Lat/Lon grid ---
ax1 = fig.add_subplot(331, projection='3d')
ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.3, linewidth=0.5)
ax1.quiver(0, 0, 0, nominal_dir[0], nominal_dir[1], nominal_dir[2], 
          color='red', arrow_length_ratio=0.1, linewidth=3, label='Nominal')
ax1.scatter(*nominal_dir, color='red', s=100, zorder=5)
ax1.scatter(yaw_directions[:, 0], yaw_directions[:, 1], yaw_directions[:, 2],
            color='green', s=30, alpha=0.8, label='Longitude (Rz)')
ax1.scatter(pitch_directions[:, 0], pitch_directions[:, 1], pitch_directions[:, 2],
            color='orange', s=30, alpha=0.8, label='Latitude (Ry)')
ax1.scatter(latlon_points[:, 0], latlon_points[:, 1], latlon_points[:, 2],
            color='blue', s=15, alpha=0.5, label=f'Lat/Lon grid ({len(angles)}×{len(angles)})')
ax1.quiver(0, 0, 0, axis_len, 0, 0, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax1.quiver(0, 0, 0, 0, axis_len, 0, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax1.quiver(0, 0, 0, 0, 0, axis_len, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax1.text(axis_len + 0.1, 0, 0, 'X', fontsize=12)
ax1.text(0, axis_len + 0.1, 0, 'Y', fontsize=12)
ax1.text(0, 0, axis_len + 0.1, 'Z', fontsize=12)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Lat/Lon Grid')
ax1.set_xlim([-1.2, 1.2])
ax1.set_ylim([-1.2, 1.2])
ax1.set_zlim([-1.2, 1.2])
ax1.set_box_aspect([1, 1, 1])
ax1.legend(loc='upper left')
ax1.view_init(elev=20, azim=30)

# --- Subplot 2: Gnomonic grid ---
ax2 = fig.add_subplot(332, projection='3d')
ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.3, linewidth=0.5)
ax2.quiver(0, 0, 0, nominal_dir[0], nominal_dir[1], nominal_dir[2], 
          color='red', arrow_length_ratio=0.1, linewidth=3, label='Nominal')
ax2.scatter(*nominal_dir, color='red', s=100, zorder=5)
ax2.scatter(yaw_directions[:, 0], yaw_directions[:, 1], yaw_directions[:, 2],
            color='green', s=30, alpha=0.8, label='Rz (yaw) → GC thru ±Y')
ax2.scatter(pitch_directions[:, 0], pitch_directions[:, 1], pitch_directions[:, 2],
            color='orange', s=30, alpha=0.8, label='Ry (pitch) → GC thru ±Z')
ax2.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
            color='blue', s=15, alpha=0.5, label=f'Gnomonic grid ({len(angles)}×{len(angles)})')
ax2.quiver(0, 0, 0, axis_len, 0, 0, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax2.quiver(0, 0, 0, 0, axis_len, 0, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax2.quiver(0, 0, 0, 0, 0, axis_len, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax2.text(axis_len + 0.1, 0, 0, 'X', fontsize=12)
ax2.text(0, axis_len + 0.1, 0, 'Y', fontsize=12)
ax2.text(0, 0, axis_len + 0.1, 'Z', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Gnomonic Grid')
ax2.set_xlim([-1.2, 1.2])
ax2.set_ylim([-1.2, 1.2])
ax2.set_zlim([-1.2, 1.2])
ax2.set_box_aspect([1, 1, 1])
ax2.legend(loc='upper left')
ax2.view_init(elev=20, azim=30)

# --- Subplot 3: Axis-angle grid ---
ax3 = fig.add_subplot(333, projection='3d')
ax3.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.3, linewidth=0.5)
ax3.quiver(0, 0, 0, nominal_dir[0], nominal_dir[1], nominal_dir[2], 
          color='red', arrow_length_ratio=0.1, linewidth=3, label='Nominal')
ax3.scatter(*nominal_dir, color='red', s=100, zorder=5)
ax3.scatter(yaw_directions[:, 0], yaw_directions[:, 1], yaw_directions[:, 2],
            color='green', s=30, alpha=0.8, label='Rz (yaw)')
ax3.scatter(pitch_directions[:, 0], pitch_directions[:, 1], pitch_directions[:, 2],
            color='orange', s=30, alpha=0.8, label='Ry (pitch)')
ax3.scatter(axis_angle_points[:, 0], axis_angle_points[:, 1], axis_angle_points[:, 2],
            color='purple', s=15, alpha=0.5, label=f'Axis-angle grid ({len(angles)}×{len(angles)})')
ax3.quiver(0, 0, 0, axis_len, 0, 0, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax3.quiver(0, 0, 0, 0, axis_len, 0, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax3.quiver(0, 0, 0, 0, 0, axis_len, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax3.text(axis_len + 0.1, 0, 0, 'X', fontsize=12)
ax3.text(0, axis_len + 0.1, 0, 'Y', fontsize=12)
ax3.text(0, 0, axis_len + 0.1, 'Z', fontsize=12)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Axis-Angle Grid')
ax3.set_xlim([-1.2, 1.2])
ax3.set_ylim([-1.2, 1.2])
ax3.set_zlim([-1.2, 1.2])
ax3.set_box_aspect([1, 1, 1])
ax3.legend(loc='upper left')
ax3.view_init(elev=20, azim=30)

# --- Row 2: Flat Area Histograms ---

# --- Subplot 4: Histogram for Lat/Lon (Flat) ---
ax4 = fig.add_subplot(334)
ax4.hist(latlon_areas_flat, bins=15, color='blue', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Cell Area')
ax4.set_ylabel('Count')
ax4.set_title(f'Lat/Lon FLAT Area\nstd/mean = {np.std(latlon_areas_flat)/np.mean(latlon_areas_flat):.2f}')
ax4.axvline(np.mean(latlon_areas_flat), color='red', linestyle='--', label=f'Mean: {np.mean(latlon_areas_flat):.4f}')
ax4.legend()

# --- Subplot 5: Histogram for Gnomonic (Flat) ---
ax5 = fig.add_subplot(335)
ax5.hist(gnomonic_areas_flat, bins=15, color='blue', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Cell Area')
ax5.set_ylabel('Count')
ax5.set_title(f'Gnomonic FLAT Area\nstd/mean = {np.std(gnomonic_areas_flat)/np.mean(gnomonic_areas_flat):.2f}')
ax5.axvline(np.mean(gnomonic_areas_flat), color='red', linestyle='--', label=f'Mean: {np.mean(gnomonic_areas_flat):.4f}')
ax5.legend()

# --- Subplot 6: Histogram for Axis-Angle (Flat) ---
ax6 = fig.add_subplot(336)
ax6.hist(axisangle_areas_flat, bins=15, color='purple', alpha=0.7, edgecolor='black')
ax6.set_xlabel('Cell Area')
ax6.set_ylabel('Count')
ax6.set_title(f'Axis-Angle FLAT Area\nstd/mean = {np.std(axisangle_areas_flat)/np.mean(axisangle_areas_flat):.2f}')
ax6.axvline(np.mean(axisangle_areas_flat), color='red', linestyle='--', label=f'Mean: {np.mean(axisangle_areas_flat):.4f}')
ax6.legend()

# --- Row 3: Spherical Area Histograms ---

# --- Subplot 7: Histogram for Lat/Lon (Spherical) ---
ax7 = fig.add_subplot(337)
ax7.hist(latlon_areas_sph, bins=15, color='blue', alpha=0.7, edgecolor='black')
ax7.set_xlabel('Solid Angle (sr)')
ax7.set_ylabel('Count')
ax7.set_title(f'Lat/Lon SPHERICAL Area\nstd/mean = {np.std(latlon_areas_sph)/np.mean(latlon_areas_sph):.2f}')
ax7.axvline(np.mean(latlon_areas_sph), color='red', linestyle='--', label=f'Mean: {np.mean(latlon_areas_sph):.4f}')
ax7.legend()

# --- Subplot 8: Histogram for Gnomonic (Spherical) ---
ax8 = fig.add_subplot(338)
ax8.hist(gnomonic_areas_sph, bins=15, color='blue', alpha=0.7, edgecolor='black')
ax8.set_xlabel('Solid Angle (sr)')
ax8.set_ylabel('Count')
ax8.set_title(f'Gnomonic SPHERICAL Area\nstd/mean = {np.std(gnomonic_areas_sph)/np.mean(gnomonic_areas_sph):.2f}')
ax8.axvline(np.mean(gnomonic_areas_sph), color='red', linestyle='--', label=f'Mean: {np.mean(gnomonic_areas_sph):.4f}')
ax8.legend()

# --- Subplot 9: Histogram for Axis-Angle (Spherical) ---
ax9 = fig.add_subplot(339)
ax9.hist(axisangle_areas_sph, bins=15, color='purple', alpha=0.7, edgecolor='black')
ax9.set_xlabel('Solid Angle (sr)')
ax9.set_ylabel('Count')
ax9.set_title(f'Axis-Angle SPHERICAL Area\nstd/mean = {np.std(axisangle_areas_sph)/np.mean(axisangle_areas_sph):.2f}')
ax9.axvline(np.mean(axisangle_areas_sph), color='red', linestyle='--', label=f'Mean: {np.mean(axisangle_areas_sph):.4f}')
ax9.legend()

plt.tight_layout()
plt.savefig('direction_sphere_nominal.png', dpi=150)

# =============================================================================
# FIGURE 2: Cube corners on sphere
# =============================================================================

# Cube corners: 8 vertices at (±1, ±1, ±1) normalized to unit sphere
# Each corner is at distance sqrt(3) from origin, so normalize by sqrt(3)
cube_corners = []
for x in [-1, 1]:
    for y in [-1, 1]:
        for z in [-1, 1]:
            d = np.array([x, y, z]) / np.sqrt(3)
            cube_corners.append(d)
cube_corners = np.array(cube_corners)

# For area calculation with only 8 points, we can't use the same grid method
# Instead, compute the 6 face areas (each face is a spherical quadrilateral)
# Each face connects 4 adjacent corners

# Face corners (indices into cube_corners array):
face_indices = [
    [0, 2, 6, 4],  # x=-1 face
    [1, 3, 7, 5],  # x=+1 face
    [0, 1, 5, 4],  # y=-1 face
    [2, 3, 7, 6],  # y=+1 face
    [0, 1, 3, 2],  # z=-1 face
    [4, 5, 7, 6],  # z=+1 face
]

# Compute FLAT face areas using cross products
cube_areas_flat = []
for face in face_indices:
    p0, p1, p2, p3 = [cube_corners[i] for i in face]
    diag1 = p2 - p0
    diag2 = p3 - p1
    area = 0.5 * np.linalg.norm(np.cross(diag1, diag2))
    cube_areas_flat.append(area)
cube_areas_flat = np.array(cube_areas_flat)

# Compute SPHERICAL face areas using spherical excess
cube_areas_sph = []
for face in face_indices:
    p0, p1, p2, p3 = [cube_corners[i] for i in face]
    # Split quadrilateral into 2 triangles
    area = spherical_triangle_area(p0, p1, p2) + spherical_triangle_area(p0, p2, p3)
    cube_areas_sph.append(area)
cube_areas_sph = np.array(cube_areas_sph)

fig2 = plt.figure(figsize=(8, 14))

# --- Subplot 1: 3D view of cube corners ---
ax2_1 = fig2.add_subplot(311, projection='3d')
ax2_1.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.3, linewidth=0.5)

# Plot cube corners
ax2_1.scatter(cube_corners[:, 0], cube_corners[:, 1], cube_corners[:, 2],
              color='red', s=100, alpha=0.9, label='Cube corners (8 points)')

# Draw edges of the cube (on the sphere surface as great circle arcs)
# For visualization, just draw straight lines between corners
edges = [
    (0, 1), (0, 2), (0, 4),  # from corner 0
    (1, 3), (1, 5),          # from corner 1
    (2, 3), (2, 6),          # from corner 2
    (3, 7),                   # from corner 3
    (4, 5), (4, 6),          # from corner 4
    (5, 7),                   # from corner 5
    (6, 7),                   # from corner 6
]
for i, j in edges:
    ax2_1.plot([cube_corners[i, 0], cube_corners[j, 0]],
               [cube_corners[i, 1], cube_corners[j, 1]],
               [cube_corners[i, 2], cube_corners[j, 2]],
               'b-', alpha=0.5, linewidth=1)

# Draw coordinate axes
ax2_1.quiver(0, 0, 0, axis_len, 0, 0, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax2_1.quiver(0, 0, 0, 0, axis_len, 0, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax2_1.quiver(0, 0, 0, 0, 0, axis_len, color='black', alpha=0.3, arrow_length_ratio=0.05, linewidth=1)
ax2_1.text(axis_len + 0.1, 0, 0, 'X', fontsize=12)
ax2_1.text(0, axis_len + 0.1, 0, 'Y', fontsize=12)
ax2_1.text(0, 0, axis_len + 0.1, 'Z', fontsize=12)

ax2_1.set_xlabel('X')
ax2_1.set_ylabel('Y')
ax2_1.set_zlabel('Z')
ax2_1.set_title('Cube Corners on Unit Sphere\n8 points at (±1, ±1, ±1)/√3')
ax2_1.set_xlim([-1.2, 1.2])
ax2_1.set_ylim([-1.2, 1.2])
ax2_1.set_zlim([-1.2, 1.2])
ax2_1.set_box_aspect([1, 1, 1])
ax2_1.legend(loc='upper left')
ax2_1.view_init(elev=20, azim=30)

# --- Subplot 2: Flat Face Areas ---
ax2_2 = fig2.add_subplot(312)
ax2_2.bar(range(6), cube_areas_flat, color='red', alpha=0.7, edgecolor='black')
ax2_2.set_xlabel('Face Index')
ax2_2.set_ylabel('Flat Area')
ax2_2.set_title(f'Cube FLAT Face Areas\nstd/mean = {np.std(cube_areas_flat)/np.mean(cube_areas_flat):.4f}')
ax2_2.axhline(np.mean(cube_areas_flat), color='blue', linestyle='--', label=f'Mean: {np.mean(cube_areas_flat):.4f}')
ax2_2.legend()
ax2_2.set_xticks(range(6))
ax2_2.set_xticklabels(['x=-1', 'x=+1', 'y=-1', 'y=+1', 'z=-1', 'z=+1'])

# --- Subplot 3: Spherical Face Areas ---
ax2_3 = fig2.add_subplot(313)
ax2_3.bar(range(6), cube_areas_sph, color='red', alpha=0.7, edgecolor='black')
ax2_3.set_xlabel('Face Index')
ax2_3.set_ylabel('Solid Angle (sr)')
ax2_3.set_title(f'Cube SPHERICAL Face Areas\nstd/mean = {np.std(cube_areas_sph)/np.mean(cube_areas_sph):.4f}')
ax2_3.axhline(np.mean(cube_areas_sph), color='blue', linestyle='--', label=f'Mean: {np.mean(cube_areas_sph):.4f} sr')
ax2_3.axhline(4*np.pi/6, color='green', linestyle=':', label=f'Ideal: {4*np.pi/6:.4f} sr (4π/6)')
ax2_3.legend()
ax2_3.set_xticks(range(6))
ax2_3.set_xticklabels(['x=-1', 'x=+1', 'y=-1', 'y=+1', 'z=-1', 'z=+1'])

plt.tight_layout()
plt.savefig('direction_sphere_cube.png', dpi=150)
plt.show()