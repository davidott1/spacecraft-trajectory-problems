import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_icosahedron():
    """Generates the 12 vertices of a Platonic Icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of an icosahedron
    verts = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ]
    # Normalize to unit sphere
    verts = np.array(verts)
    return verts / np.linalg.norm(verts[0])

def get_fibonacci_sphere(n=500):
    """Uses the Golden Spiral (Fibonacci) to place N points."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2 
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def get_geodesic_subdivision(subdivisions=2):
    """Subdivides an Icosahedron by recursively adding edge midpoints.
    
    subdivisions=1: 42 points
    subdivisions=2: 162 points  
    subdivisions=3: 642 points
    subdivisions=4: 2562 points
    """
    # Start with icosahedron vertices
    verts = list(get_icosahedron())
    
    # Define the 20 faces of the icosahedron (vertex indices)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}
        
        for face in faces:
            v0, v1, v2 = face
            
            # Get or create midpoints for each edge
            mids = []
            for edge in [(v0, v1), (v1, v2), (v2, v0)]:
                edge_key = tuple(sorted(edge))
                if edge_key not in edge_midpoints:
                    mid = (np.array(verts[edge[0]]) + np.array(verts[edge[1]])) / 2
                    mid = mid / np.linalg.norm(mid)  # Project to sphere
                    edge_midpoints[edge_key] = len(verts)
                    verts.append(mid)
                mids.append(edge_midpoints[edge_key])
            
            m01, m12, m20 = mids
            
            # Create 4 new faces from this face
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])
        
        faces = new_faces
    
    return np.array(verts)

def get_force_directed(n=100, iterations=60):
    """Simulates points as repelling electrons to find equilibrium."""
    points = np.random.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    for _ in range(iterations):
        forces = np.zeros_like(points)
        for i in range(n):
            diff = points[i] - points
            dist_sq = np.sum(diff**2, axis=1)
            dist_sq[i] = np.inf # Avoid self-repulsion
            f = diff / (dist_sq[:, None]**1.5 + 1e-8)
            forces[i] = np.sum(f, axis=0)
        points += forces * 0.02 # Step size
        points /= np.linalg.norm(points, axis=1)[:, None]
    return points

def compute_voronoi_areas(points):
    """Compute approximate Voronoi cell areas using Delaunay triangulation."""
    from scipy.spatial import ConvexHull, SphericalVoronoi
    
    # Normalize points to exactly unit sphere
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Use SphericalVoronoi to get Voronoi regions on the sphere
    try:
        sv = SphericalVoronoi(points, radius=1, center=np.array([0., 0., 0.]))
        sv.sort_vertices_of_regions()
        
        areas = []
        for region in sv.regions:
            if len(region) >= 3:
                # Compute spherical polygon area using cross products (flat approx)
                verts = sv.vertices[region]
                center = np.mean(verts, axis=0)
                center /= np.linalg.norm(center)
                
                area = 0.0
                n = len(verts)
                for i in range(n):
                    v1 = verts[i] - center
                    v2 = verts[(i + 1) % n] - center
                    area += 0.5 * np.linalg.norm(np.cross(v1, v2))
                areas.append(area)
        return np.array(areas)
    except Exception as e:
        print(f"SphericalVoronoi failed: {e}")
        return np.array([])

# --- Plotting ---
fig = plt.figure(figsize=(12, 10))

# 1. Platonic
ax1 = fig.add_subplot(221, projection='3d')
p1 = get_icosahedron()
ax1.scatter(p1[:,0], p1[:,1], p1[:,2], color='red', s=50)
ax1.set_title("Platonic (Icosahedron: 12 pts)")

# 2. Fibonacci
ax2 = fig.add_subplot(222, projection='3d')
p2 = get_fibonacci_sphere(500)
ax2.scatter(p2[:,0], p2[:,1], p2[:,2], s=10, color='blue', alpha=0.5)
ax2.set_title("Fibonacci Lattice (500 pts)")

# 3. Geodesic
ax3 = fig.add_subplot(223, projection='3d')
p3 = get_geodesic_subdivision(subdivisions=2)
ax3.scatter(p3[:,0], p3[:,1], p3[:,2], color='green', s=15)
ax3.set_title(f"Geodesic Subdivision ({len(p3)} pts)")

# 4. Force-Directed
ax4 = fig.add_subplot(224, projection='3d')
p4 = get_force_directed(n=13, iterations=10000)
ax4.scatter(p4[:,0], p4[:,1], p4[:,2], color='purple', s=30)
ax4.set_title(f"Force-Directed Repulsion ({len(p4)} pts)")

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

plt.tight_layout()

# =============================================================================
# FIGURE 2: Histograms of Voronoi cell areas
# =============================================================================

# Compute Voronoi areas for each distribution
areas1 = compute_voronoi_areas(p1)
areas2 = compute_voronoi_areas(p2)
areas3 = compute_voronoi_areas(p3)
areas4 = compute_voronoi_areas(p4)

fig2 = plt.figure(figsize=(12, 10))

# 1. Platonic
ax2_1 = fig2.add_subplot(221)
if len(areas1) > 0:
    ax2_1.hist(areas1, bins=15, color='red', alpha=0.7, edgecolor='black', weights=np.ones(len(areas1))/len(areas1))
    ax2_1.axvline(np.mean(areas1), color='blue', linestyle='--', label=f'Mean: {np.mean(areas1):.4f}')
    ax2_1.set_title(f'Platonic (12 pts)\nstd/mean = {np.std(areas1)/np.mean(areas1):.4f}')
    ax2_1.legend()
ax2_1.set_xlabel('Cell Area')
ax2_1.set_ylabel('Frequency')

# 2. Fibonacci
ax2_2 = fig2.add_subplot(222)
if len(areas2) > 0:
    ax2_2.hist(areas2, bins=30, color='blue', alpha=0.7, edgecolor='black', weights=np.ones(len(areas2))/len(areas2))
    ax2_2.axvline(np.mean(areas2), color='red', linestyle='--', label=f'Mean: {np.mean(areas2):.4f}')
    ax2_2.set_title(f'Fibonacci (500 pts)\nstd/mean = {np.std(areas2)/np.mean(areas2):.4f}')
    ax2_2.legend()
ax2_2.set_xlabel('Cell Area')
ax2_2.set_ylabel('Frequency')

# 3. Geodesic
ax2_3 = fig2.add_subplot(223)
if len(areas3) > 0:
    ax2_3.hist(areas3, bins=15, color='green', alpha=0.7, edgecolor='black', weights=np.ones(len(areas3))/len(areas3))
    ax2_3.axvline(np.mean(areas3), color='red', linestyle='--', label=f'Mean: {np.mean(areas3):.4f}')
    ax2_3.set_title(f'Geodesic (42 pts)\nstd/mean = {np.std(areas3)/np.mean(areas3):.4f}')
    ax2_3.legend()
ax2_3.set_xlabel('Cell Area')
ax2_3.set_ylabel('Frequency')

# 4. Force-Directed
ax2_4 = fig2.add_subplot(224)
if len(areas4) > 0:
    ax2_4.hist(areas4, bins=15, color='purple', alpha=0.7, edgecolor='black', weights=np.ones(len(areas4))/len(areas4))
    ax2_4.axvline(np.mean(areas4), color='red', linestyle='--', label=f'Mean: {np.mean(areas4):.4f}')
    ax2_4.set_title(f'Force-Directed (30 pts)\nstd/mean = {np.std(areas4)/np.mean(areas4):.4f}')
    ax2_4.legend()
ax2_4.set_xlabel('Cell Area')
ax2_4.set_ylabel('Frequency')

plt.tight_layout()

# =============================================================================
# FIGURE 3: Sorted Area Plot (all methods on one plot)
# =============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 6))

# Normalize by mean area so 1.0 = average (perfect uniformity = flat line at 1.0)
if len(areas1) > 0:
    sorted1 = np.sort(areas1) / np.mean(areas1)
    x1 = np.linspace(0, 1, len(sorted1))
    ax3.plot(x1, sorted1, 'r-o', markersize=8, label=f'Platonic (12 pts), CV={np.std(areas1)/np.mean(areas1):.4f}')

if len(areas2) > 0:
    sorted2 = np.sort(areas2) / np.mean(areas2)
    x2 = np.linspace(0, 1, len(sorted2))
    ax3.plot(x2, sorted2, 'b-', linewidth=1, alpha=0.7, label=f'Fibonacci (500 pts), CV={np.std(areas2)/np.mean(areas2):.4f}')

if len(areas3) > 0:
    sorted3 = np.sort(areas3) / np.mean(areas3)
    x3 = np.linspace(0, 1, len(sorted3))
    ax3.plot(x3, sorted3, 'g-s', markersize=5, label=f'Geodesic (42 pts), CV={np.std(areas3)/np.mean(areas3):.4f}')

if len(areas4) > 0:
    sorted4 = np.sort(areas4) / np.mean(areas4)
    x4 = np.linspace(0, 1, len(sorted4))
    ax3.plot(x4, sorted4, 'm-^', markersize=6, label=f'Force-Directed (30 pts), CV={np.std(areas4)/np.mean(areas4):.4f}')

ax3.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Mean (1.0)')
ax3.set_xlabel('Percentile')
ax3.set_ylabel('Cell Area / Mean Area')
ax3.set_title('Sorted Voronoi Cell Areas (Normalized by Mean)\n(flatter = more uniform)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()