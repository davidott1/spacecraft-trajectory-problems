"""
Example: Patched Conic Lunar Transfer
======================================

Demonstrates solving for a minimum-ΔV Earth-to-Moon transfer
using the patched conic approximation.

Usage:
  python -m scripts.run_lunar_transfer

  or from the repo root:
  python scripts/run_lunar_transfer.py
"""
import sys
import os
import numpy as np
import spiceypy as spice

# Ensure project root is on the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from datetime import datetime

from src.model.constants      import SOLARSYSTEMCONSTANTS, NAIFIDS
from src.model.time_converter import utc_to_et
from src.schemas.optimization  import LunarTransferConfig
from src.optimization.lunar_transfer import LunarTransferOptimizer
from src.model.orbital_mechanics     import compute_circular_velocity


# --------------------------------------------------------------------------
# 1. Load SPICE Kernels
# --------------------------------------------------------------------------
KERNEL_DIR = os.path.join(ROOT, 'data', 'spice_kernels')

kernels = [
  os.path.join(KERNEL_DIR, 'naif0012.tls'),     # Leap seconds
  os.path.join(KERNEL_DIR, 'de440.bsp'),         # Planetary ephemeris
  os.path.join(KERNEL_DIR, 'pck00010.tpc'),      # Body orientation / radii
]

for k in kernels:
  if not os.path.isfile(k):
    print(f"[ERROR] Missing kernel: {k}")
    sys.exit(1)
  spice.furnsh(k)

print("SPICE kernels loaded.")

# --------------------------------------------------------------------------
# 2. Build Initial LEO State (circular, equatorial, 200 km altitude)
# --------------------------------------------------------------------------
EARTH_RADIUS = SOLARSYSTEMCONSTANTS.EARTH.RADIUS.EQUATOR   # 6 378 137 m
EARTH_GP     = SOLARSYSTEMCONSTANTS.EARTH.GP                # 3.986e14 m³/s²

leo_altitude = 200_000.0                                    # 200 km
r_leo        = EARTH_RADIUS + leo_altitude
v_circ_leo   = compute_circular_velocity(r_leo, EARTH_GP)

# Equatorial circular orbit: pos along +X, vel along +Y
initial_state = np.array([
  r_leo,      0.0,  0.0,          # position [m]
  0.0,  v_circ_leo,  0.0,         # velocity [m/s]
])

print(f"LEO radius   : {r_leo/1000:.1f} km")
print(f"LEO V_circ   : {v_circ_leo:.2f} m/s")

# --------------------------------------------------------------------------
# 3. Configure the Transfer
# --------------------------------------------------------------------------
config = LunarTransferConfig(
  leo_altitude_m            = leo_altitude,
  llo_altitude_m            = 100_000.0,       # 100 km LLO
  departure_epoch           = datetime(2025, 6, 1),
  max_transfer_time_s       = 7.0 * 86400.0,   # 7-day max transfer
  dv1_search_bounds_m_s     = (2800.0, 3400.0),
  departure_search_window_s = 30.0 * 86400.0,  # 30-day search window
  n_departure_candidates    = 60,               # coarse grid (fast)
  llo_coast_orbits          = 2,
  atol                      = 1e-10,
  rtol                      = 1e-10,
)

# --------------------------------------------------------------------------
# 4. Run the Optimizer
# --------------------------------------------------------------------------
optimizer = LunarTransferOptimizer(config, initial_state)
result    = optimizer.solve()

# --------------------------------------------------------------------------
# 5. Check Results
# --------------------------------------------------------------------------
if not result.success:
  print(f"\n[FAILED] {result.message}")
  sys.exit(1)

print("\n" + "=" * 50)
print("Transfer solved successfully!")
print("=" * 50)
print(f"  ΔV₁  = {result.delta_vel_mag_1:.4f} m/s")
print(f"  ΔV₂  = {result.delta_vel_mag_2:.4f} m/s")
print(f"  Total = {result.delta_vel_total:.4f} m/s")
print(f"  Transfer time = {result.transfer_time_s/86400:.2f} days")
print(f"  Periapsis alt = {result.periapsis_altitude_m/1000:.1f} km")

if result.combined_trajectory is not None:
  n_pts = result.combined_trajectory.state.shape[1]
  print(f"  Trajectory points = {n_pts}")

# --------------------------------------------------------------------------
# 6. Quick Plot (optional — needs matplotlib)
# --------------------------------------------------------------------------
try:
  import matplotlib.pyplot as plt
  from src.model.frame_and_vector_converter import BodyVectorConverter

  states = result.combined_trajectory.state  # (6, N) Earth-centered J2000

  MOON_RADIUS = SOLARSYSTEMCONSTANTS.MOON.RADIUS.EQUATOR

  # Moon is ~1,737 km radius at ~384,000 km distance — invisible at true scale.
  # Scale up the drawn Moon radius so it's actually visible on the plot.
  MOON_DRAW_SCALE = 10  # draw Moon 10× actual size
  MOON_DRAW_R     = MOON_RADIUS / 1e3 * MOON_DRAW_SCALE

  # Moon position at SOI crossing
  t_soi       = result.soi_crossing_et
  moon_soi    = BodyVectorConverter.get_body_state(NAIFIDS.MOON, t_soi, NAIFIDS.EARTH) / 1e3  # km
  mx_soi, my_soi, mz_soi = moon_soi[0], moon_soi[1], moon_soi[2]

  # Moon position at trajectory end (LLO coast end)
  t_end       = result.combined_trajectory.time.grid.relative_initial[-1] + \
                utc_to_et(result.combined_trajectory.time.initial.utc)
  moon_end    = BodyVectorConverter.get_body_state(NAIFIDS.MOON, t_end, NAIFIDS.EARTH) / 1e3  # km
  mx_end, my_end, mz_end = moon_end[0], moon_end[1], moon_end[2]

  # SOI radius and patch point (SOI crossing position in Earth-centered J2000)
  SOI_R_KM    = optimizer.soi_moon / 1e3  # ~66,194 km
  patch_pt    = result.soi_state_earth_j2000[0:3] / 1e3  # km
  px, py, pz  = patch_pt[0], patch_pt[1], patch_pt[2]

  # Sphere mesh for 3D bodies
  u_sphere = np.linspace(0, 2 * np.pi, 40)
  v_sphere = np.linspace(0, np.pi, 20)
  sx = np.outer(np.cos(u_sphere), np.sin(v_sphere))
  sy = np.outer(np.sin(u_sphere), np.sin(v_sphere))
  sz = np.outer(np.ones_like(u_sphere), np.cos(v_sphere))

  # Circle for 2D bodies
  theta_circ = np.linspace(0, 2 * np.pi, 200)

  fig = plt.figure(figsize=(22, 7))

  # ----- XY plane -----
  ax = fig.add_subplot(131)
  ax.set_aspect('equal')
  ax.set_title('Earth-Moon Transfer (J2000 XY)')
  ax.set_xlabel('X [km]')
  ax.set_ylabel('Y [km]')

  # Earth (filled circle)
  ax.fill(EARTH_RADIUS/1e3 * np.cos(theta_circ),
          EARTH_RADIUS/1e3 * np.sin(theta_circ),
          color='dodgerblue', alpha=0.7, label='Earth')
  ax.plot(EARTH_RADIUS/1e3 * np.cos(theta_circ),
          EARTH_RADIUS/1e3 * np.sin(theta_circ),
          color='blue', lw=0.8)

  # Moon at SOI crossing (filled circle, scaled up + X marker)
  ax.fill(mx_soi + MOON_DRAW_R * np.cos(theta_circ),
          my_soi + MOON_DRAW_R * np.sin(theta_circ),
          color='silver', alpha=0.8, label=f'Moon (SOI, {MOON_DRAW_SCALE}× scale)')
  ax.plot(mx_soi + MOON_DRAW_R * np.cos(theta_circ),
          my_soi + MOON_DRAW_R * np.sin(theta_circ),
          color='gray', lw=0.8)
  ax.plot(mx_soi, my_soi, 'kx', ms=14, mew=3)

  # Moon at trajectory end (filled circle, scaled up + X marker)
  ax.fill(mx_end + MOON_DRAW_R * np.cos(theta_circ),
          my_end + MOON_DRAW_R * np.sin(theta_circ),
          color='lightyellow', alpha=0.8, label=f'Moon (end, {MOON_DRAW_SCALE}× scale)')
  ax.plot(mx_end + MOON_DRAW_R * np.cos(theta_circ),
          my_end + MOON_DRAW_R * np.sin(theta_circ),
          color='goldenrod', lw=0.8)
  ax.plot(mx_end, my_end, 'kx', ms=14, mew=3)

  # Moon SOI sphere (circle in 2D) centered on Moon at SOI-crossing time
  ax.plot(mx_soi + SOI_R_KM * np.cos(theta_circ),
          my_soi + SOI_R_KM * np.sin(theta_circ),
          color='green', lw=1.0, ls='--', alpha=0.6, label='Moon SOI')

  # Trajectory
  ax.plot(states[0]/1e3, states[1]/1e3, 'r-', lw=0.6, label='Trajectory')

  # Patch point (SOI crossing)
  ax.plot(px, py, 'g^', ms=10, mew=1.5, mec='darkgreen', zorder=6, label='Patch point')

  # ΔV₁ arrow at LEO departure (Earth-centered J2000)
  dv1_pos = result.earth_departure_leg.j2000_state_vec[0:3, 0] / 1e3  # km
  dv1_dir = result.delta_vel_vec_1 / np.linalg.norm(result.delta_vel_vec_1)
  # Compute a good arrow length relative to the plot extent
  plot_extent = max(abs(states[0]).max(), abs(states[1]).max()) / 1e3
  arrow_len   = plot_extent * 0.06
  ax.annotate('', xy=(dv1_pos[0] + dv1_dir[0]*arrow_len,
                      dv1_pos[1] + dv1_dir[1]*arrow_len),
              xytext=(dv1_pos[0], dv1_pos[1]),
              arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
  ax.plot(dv1_pos[0], dv1_pos[1], 'ro', ms=6, zorder=5)
  ax.text(dv1_pos[0] + dv1_dir[0]*arrow_len*1.15,
          dv1_pos[1] + dv1_dir[1]*arrow_len*1.15,
          f'ΔV₁ = {result.delta_vel_mag_1:.0f} m/s', fontsize=8, color='red',
          ha='left', va='bottom')

  # ΔV₂ arrow at LLO insertion (periapsis, in Earth-centered J2000)
  # Periapsis is last point of lunar_arrival_leg (Moon-centered) — transform to Earth
  dv2_state_earth = BodyVectorConverter.j2000_xyz__rel_moon_to_rel_earth(
    result.lunar_arrival_leg.j2000_state_vec[:, -1],
    result.lunar_arrival_leg.time.grid.et[-1],
  )
  dv2_pos_earth = dv2_state_earth[0:3] / 1e3
  # ΔV₂ is in Moon-centered frame; transform direction to Earth J2000
  # (just use the velocity direction at periapsis — ΔV₂ is retrograde)
  dv2_dir = result.delta_vel_vec_2 / np.linalg.norm(result.delta_vel_vec_2)
  ax.annotate('', xy=(dv2_pos_earth[0] + dv2_dir[0]*arrow_len,
                      dv2_pos_earth[1] + dv2_dir[1]*arrow_len),
              xytext=(dv2_pos_earth[0], dv2_pos_earth[1]),
              arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
  ax.plot(dv2_pos_earth[0], dv2_pos_earth[1], 'ro', ms=6, zorder=5)
  ax.text(dv2_pos_earth[0] + dv2_dir[0]*arrow_len*1.15,
          dv2_pos_earth[1] + dv2_dir[1]*arrow_len*1.15,
          f'ΔV₂ = {result.delta_vel_mag_2:.0f} m/s', fontsize=8, color='red',
          ha='left', va='top')

  ax.legend(fontsize=8, loc='upper left')
  ax.grid(True, alpha=0.3)

  # ----- Key event + LLO coast snapshots: Moon + S/C paired markers -----
  # Key events: patch point, ΔV₂ maneuver, LLO coast steps, final time
  llo_times_et  = result.llo_coast_leg.time.grid.et
  # Transform LLO coast states (Moon-centered) to Earth-centered for panels 1 & 2
  llo_mc        = result.llo_coast_leg.j2000_state_vec  # (6, N) Moon-centered
  llo_sc_earth  = np.zeros_like(llo_mc)
  for _i in range(llo_mc.shape[1]):
    llo_sc_earth[:, _i] = BodyVectorConverter.j2000_xyz__rel_moon_to_rel_earth(llo_mc[:, _i], llo_times_et[_i])

  # -- Precompute all snapshot positions --
  snap_data = []  # list of (label, clr, sc_pos_km, mn_pos_km, marker_sc, marker_mn, ms_sc, ms_mn)

  # 1) Patch point (SOI crossing)
  t_patch    = result.soi_crossing_et
  sc_patch   = result.soi_state_earth_j2000[0:3] / 1e3
  mn_patch   = BodyVectorConverter.get_body_state(NAIFIDS.MOON, t_patch, NAIFIDS.EARTH)[0:3] / 1e3
  snap_data.append(('Patch', 'limegreen', sc_patch, mn_patch, '^', 'o', 10, 9))

  # 2) ΔV₂ maneuver (periapsis / LLO insertion)
  t_dv2      = result.lunar_arrival_leg.time.grid.et[-1]
  sc_dv2_e   = BodyVectorConverter.j2000_xyz__rel_moon_to_rel_earth(
    result.lunar_arrival_leg.j2000_state_vec[:, -1], t_dv2,
  )
  sc_dv2     = sc_dv2_e[0:3] / 1e3
  mn_dv2     = BodyVectorConverter.get_body_state(NAIFIDS.MOON, t_dv2, NAIFIDS.EARTH)[0:3] / 1e3
  snap_data.append(('ΔV₂', 'orangered', sc_dv2, mn_dv2, 'v', 'o', 10, 9))

  # 3) LLO coast intermediate steps (equal spacing)
  n_snapshots   = 8
  snap_indices  = np.linspace(0, len(llo_times_et) - 1, n_snapshots + 2, dtype=int)[1:-1]
  cmap_snap     = plt.cm.cool
  for j, idx in enumerate(snap_indices):
    t_snap = llo_times_et[idx]
    frac   = j / (len(snap_indices) - 1) if len(snap_indices) > 1 else 0
    clr    = cmap_snap(frac)
    sc_pos = llo_sc_earth[0:3, idx] / 1e3
    mn_pos = BodyVectorConverter.get_body_state(NAIFIDS.MOON, t_snap, NAIFIDS.EARTH)[0:3] / 1e3
    snap_data.append((None, clr, sc_pos, mn_pos, 's', 'o', 5, 7))

  # 4) Final time (end of LLO coast)
  t_final    = llo_times_et[-1]
  sc_final   = llo_sc_earth[0:3, -1] / 1e3
  mn_final   = BodyVectorConverter.get_body_state(NAIFIDS.MOON, t_final, NAIFIDS.EARTH)[0:3] / 1e3
  snap_data.append(('Final', 'magenta', sc_final, mn_final, 'D', 'o', 8, 9))

  # 2D snapshot markers
  first_step = True
  for lbl, clr, sc_pos, mn_pos, mk_sc, mk_mn, ms_sc, ms_mn in snap_data:
    # Moon dot
    ax.plot(mn_pos[0], mn_pos[1], mk_mn, color=clr, ms=ms_mn,
            mec='k', mew=0.4, zorder=7)
    # S/C dot
    ax.plot(sc_pos[0], sc_pos[1], mk_sc, color=clr, ms=ms_sc,
            mec='k', mew=0.4, zorder=7)
    # Connecting line
    ax.plot([mn_pos[0], sc_pos[0]], [mn_pos[1], sc_pos[1]],
            '-', color=clr, lw=0.6, alpha=0.5)
    # Label key events
    if lbl is not None:
      ax.text(sc_pos[0], sc_pos[1] + arrow_len * 0.3, lbl,
              fontsize=7, color=clr, ha='center', va='bottom', fontweight='bold')

  # Refresh legend
  ax.legend(fontsize=7, loc='upper left')

  # ----- 3D view -----
  ax3 = fig.add_subplot(132, projection='3d')
  ax3.set_title('3D View')

  # Earth sphere
  ax3.plot_surface(EARTH_RADIUS/1e3 * sx,
                   EARTH_RADIUS/1e3 * sy,
                   EARTH_RADIUS/1e3 * sz,
                   color='dodgerblue', alpha=0.6, linewidth=0)

  # Moon sphere at SOI crossing (scaled up + X marker)
  ax3.plot_surface(mx_soi + MOON_DRAW_R * sx,
                   my_soi + MOON_DRAW_R * sy,
                   mz_soi + MOON_DRAW_R * sz,
                   color='silver', alpha=0.7, linewidth=0)
  ax3.plot([mx_soi], [my_soi], [mz_soi], 'kx', ms=14, mew=3, zorder=10)
  ax3.text(mx_soi, my_soi, mz_soi + 12000, 'Moon (SOI)', fontsize=8,
           ha='center', color='black')

  # Moon sphere at trajectory end (scaled up + X marker)
  ax3.plot_surface(mx_end + MOON_DRAW_R * sx,
                   my_end + MOON_DRAW_R * sy,
                   mz_end + MOON_DRAW_R * sz,
                   color='lightyellow', alpha=0.7, linewidth=0)
  ax3.plot([mx_end], [my_end], [mz_end], 'kx', ms=14, mew=3, zorder=10)
  ax3.text(mx_end, my_end, mz_end + 12000, 'Moon (end)', fontsize=8,
           ha='center', color='black')

  # Moon SOI wireframe sphere (centered on Moon at SOI-crossing time)
  ax3.plot_wireframe(mx_soi + SOI_R_KM * sx,
                     my_soi + SOI_R_KM * sy,
                     mz_soi + SOI_R_KM * sz,
                     color='green', alpha=0.15, linewidth=0.4,
                     rstride=2, cstride=2)

  # Trajectory
  ax3.plot(states[0]/1e3, states[1]/1e3, states[2]/1e3, 'r-', lw=0.5)

  # ΔV₁ arrow at LEO departure (3D)
  dv1_p3 = result.earth_departure_leg.j2000_state_vec[0:3, 0] / 1e3
  ax3.quiver(dv1_p3[0], dv1_p3[1], dv1_p3[2],
             dv1_dir[0], dv1_dir[1], dv1_dir[2],
             length=arrow_len, color='red', arrow_length_ratio=0.3, linewidth=2.5)
  ax3.plot([dv1_p3[0]], [dv1_p3[1]], [dv1_p3[2]], 'ro', ms=6, zorder=5)
  ax3.text(dv1_p3[0], dv1_p3[1], dv1_p3[2] + 8000,
           f'ΔV₁={result.delta_vel_mag_1:.0f} m/s', fontsize=7, color='red')

  # ΔV₂ arrow at LLO insertion (3D)
  dv2_p3 = dv2_state_earth[0:3] / 1e3  # already computed above
  ax3.quiver(dv2_p3[0], dv2_p3[1], dv2_p3[2],
             dv2_dir[0], dv2_dir[1], dv2_dir[2],
             length=arrow_len, color='red', arrow_length_ratio=0.3, linewidth=2.5)
  ax3.plot([dv2_p3[0]], [dv2_p3[1]], [dv2_p3[2]], 'ro', ms=6, zorder=5)
  ax3.text(dv2_p3[0], dv2_p3[1], dv2_p3[2] + 8000,
           f'ΔV₂={result.delta_vel_mag_2:.0f} m/s', fontsize=7, color='red')

  # 3D snapshot markers (key events + LLO coast steps)
  for lbl, clr, sc_pos, mn_pos, mk_sc, mk_mn, ms_sc, ms_mn in snap_data:
    ax3.plot([mn_pos[0]], [mn_pos[1]], [mn_pos[2]], mk_mn, color=clr,
            ms=ms_mn, mec='k', mew=0.4, zorder=7)
    ax3.plot([sc_pos[0]], [sc_pos[1]], [sc_pos[2]], mk_sc, color=clr,
            ms=ms_sc, mec='k', mew=0.4, zorder=7)
    ax3.plot([mn_pos[0], sc_pos[0]], [mn_pos[1], sc_pos[1]], [mn_pos[2], sc_pos[2]],
            '-', color=clr, lw=0.6, alpha=0.5)
    if lbl is not None:
      ax3.text(sc_pos[0], sc_pos[1], sc_pos[2] + 8000, lbl,
               fontsize=7, color=clr, ha='center', fontweight='bold')

  ax3.set_xlabel('X [km]')
  ax3.set_ylabel('Y [km]')
  ax3.set_zlabel('Z [km]')

  # Equal axes for 3D (true 1:1:1 aspect)
  traj_km   = states[0:3] / 1e3
  all_x     = np.concatenate([traj_km[0], [mx_soi, mx_end]])
  all_y     = np.concatenate([traj_km[1], [my_soi, my_end]])
  all_z     = np.concatenate([traj_km[2], [mz_soi, mz_end]])
  mins      = np.array([all_x.min(), all_y.min(), all_z.min()])
  maxs      = np.array([all_x.max(), all_y.max(), all_z.max()])
  mid       = (maxs + mins) / 2
  span      = (maxs - mins).max() / 2 * 1.1
  ax3.set_xlim(mid[0] - span, mid[0] + span)
  ax3.set_ylim(mid[1] - span, mid[1] + span)
  ax3.set_zlim(mid[2] - span, mid[2] + span)
  ax3.set_box_aspect([1, 1, 1])

  # ----- Moon-centered hyperbolic arrival (3D) -----
  ax_m = fig.add_subplot(133, projection='3d')
  ax_m.set_title('Lunar Arrival (Moon-Centered)')
  ax_m.set_xlabel('X [km]')
  ax_m.set_ylabel('Y [km]')
  ax_m.set_zlabel('Z [km]')

  # Moon surface sphere
  MOON_R_KM = MOON_RADIUS / 1e3
  ax_m.plot_surface(MOON_R_KM * sx, MOON_R_KM * sy, MOON_R_KM * sz,
                    color='silver', alpha=0.7, linewidth=0)

  # SOI wireframe sphere
  ax_m.plot_wireframe(SOI_R_KM * sx, SOI_R_KM * sy, SOI_R_KM * sz,
                      color='green', alpha=0.12, linewidth=0.3,
                      rstride=2, cstride=2)

  # Hyperbolic trajectory (Moon-centered, from SOI to periapsis)
  moon_states_bc = result.lunar_arrival_leg.j2000_state_vec  # (6, N) Moon-centered
  ax_m.plot(moon_states_bc[0] / 1e3, moon_states_bc[1] / 1e3, moon_states_bc[2] / 1e3,
            'r-', lw=1.5, label='Hyperbolic trajectory')

  # LLO coast (Moon-centered)
  llo_states_bc = result.llo_coast_leg.j2000_state_vec  # (6, N) Moon-centered
  ax_m.plot(llo_states_bc[0] / 1e3, llo_states_bc[1] / 1e3, llo_states_bc[2] / 1e3,
            'b-', lw=0.8, alpha=0.7, label='LLO coast')

  # Patch point (SOI entry)
  soi_mc = result.soi_state_moon
  ax_m.plot([soi_mc[0]/1e3], [soi_mc[1]/1e3], [soi_mc[2]/1e3],
            'g^', ms=10, mec='darkgreen', mew=1.5, zorder=6, label='Patch (SOI entry)')
  ax_m.text(soi_mc[0]/1e3, soi_mc[1]/1e3, soi_mc[2]/1e3 + 3000,
            'Patch', fontsize=7, color='darkgreen', ha='center', fontweight='bold')

  # Periapsis (ΔV₂ location)
  peri_mc = result.lunar_arrival_leg.j2000_state_vec[0:3, -1]
  peri_r_km = np.linalg.norm(peri_mc) / 1e3
  ax_m.plot([peri_mc[0]/1e3], [peri_mc[1]/1e3], [peri_mc[2]/1e3],
            'rv', ms=10, mec='darkred', mew=1.5, zorder=6,
            label=f'Periapsis ({peri_r_km:.0f} km)')

  # ΔV₂ quiver at periapsis (Moon-centered)
  moon_arrow_len = SOI_R_KM * 0.08
  ax_m.quiver(peri_mc[0]/1e3, peri_mc[1]/1e3, peri_mc[2]/1e3,
              dv2_dir[0], dv2_dir[1], dv2_dir[2],
              length=moon_arrow_len, color='red', arrow_length_ratio=0.3, linewidth=2.5)
  ax_m.text(peri_mc[0]/1e3, peri_mc[1]/1e3, peri_mc[2]/1e3 - 4000,
            f'ΔV₂={result.delta_vel_mag_2:.0f} m/s', fontsize=7, color='red',
            ha='center', fontweight='bold')

  # V∞ velocity arrow at SOI entry
  v_soi = soi_mc[3:6]
  v_hat = v_soi / np.linalg.norm(v_soi)
  ax_m.quiver(soi_mc[0]/1e3, soi_mc[1]/1e3, soi_mc[2]/1e3,
              v_hat[0], v_hat[1], v_hat[2],
              length=moon_arrow_len, color='orange', arrow_length_ratio=0.3, linewidth=2)
  ax_m.text(soi_mc[0]/1e3 + v_hat[0]*moon_arrow_len*1.1,
            soi_mc[1]/1e3 + v_hat[1]*moon_arrow_len*1.1,
            soi_mc[2]/1e3 + v_hat[2]*moon_arrow_len*1.1,
            f'V∞={result.v_infinity_mag:.0f} m/s', fontsize=7, color='orange',
            ha='left', fontweight='bold')

  # LLO coast S/C snapshots (Moon-centered, same colors as panels 1 & 2)
  # ΔV₂ point (orangered, same marker)
  dv2_mc = result.lunar_arrival_leg.j2000_state_vec[0:3, -1] / 1e3
  ax_m.plot([dv2_mc[0]], [dv2_mc[1]], [dv2_mc[2]],
            'v', color='orangered', ms=10, mec='k', mew=0.4, zorder=7)

  # Intermediate LLO coast steps (same indices and colormap as other panels)
  for j, idx in enumerate(snap_indices):
    frac = j / (len(snap_indices) - 1) if len(snap_indices) > 1 else 0
    clr  = cmap_snap(frac)
    sc_mc = llo_states_bc[0:3, idx] / 1e3
    ax_m.plot([sc_mc[0]], [sc_mc[1]], [sc_mc[2]],
              's', color=clr, ms=5, mec='k', mew=0.4, zorder=7)

  # Final point (magenta diamond, same as other panels)
  sc_final_mc = llo_states_bc[0:3, -1] / 1e3
  ax_m.plot([sc_final_mc[0]], [sc_final_mc[1]], [sc_final_mc[2]],
            'D', color='magenta', ms=8, mec='k', mew=0.4, zorder=7)
  ax_m.text(sc_final_mc[0], sc_final_mc[1], sc_final_mc[2] + 2000,
            'Final', fontsize=7, color='magenta', ha='center', fontweight='bold')

  ax_m.legend(fontsize=7, loc='upper left')

  # Equal axes for Moon-centered 3D
  mc_all = np.hstack([moon_states_bc[0:3]/1e3, llo_states_bc[0:3]/1e3])
  mc_mins = mc_all.min(axis=1)
  mc_maxs = mc_all.max(axis=1)
  mc_mid  = (mc_maxs + mc_mins) / 2
  mc_span = (mc_maxs - mc_mins).max() / 2 * 1.1
  ax_m.set_xlim(mc_mid[0] - mc_span, mc_mid[0] + mc_span)
  ax_m.set_ylim(mc_mid[1] - mc_span, mc_mid[1] + mc_span)
  ax_m.set_zlim(mc_mid[2] - mc_span, mc_mid[2] + mc_span)
  ax_m.set_box_aspect([1, 1, 1])

  plt.tight_layout()
  out_path = os.path.join(ROOT, 'output', 'lunar_transfer_plot.png')
  plt.savefig(out_path, dpi=150)
  print(f"\nPlot saved to {out_path}")
  plt.show()

except ImportError:
  print("\n(matplotlib not installed — skipping plot)")

# --------------------------------------------------------------------------
# 7. Cleanup SPICE
# --------------------------------------------------------------------------
spice.kclear()
