#!/usr/bin/env python3
"""
2D Patched Conic Transfer: Circular LEO -> Circular LLO
========================================================
 
Analytical single-solution approach. No grid search.

Setup:
  - Earth at origin (0, 0)
  - Moon at theta_moon_0 at t = 0, circular orbit counterclockwise
  - Two-body gravity only (patched conics at Moon SOI)
  - No SPICE -- analytical Moon ephemeris

Analytical targeting (baseline):
  1. Hohmann SMA and transfer time delta_t are known from vis-viva
  2. Moon position at arrival: theta_moon = theta_moon_0 + omega_moon * delta_t
  3. Apse line must point at Moon: departure (periapsis) at theta_moon - pi
  4. Tangential prograde burn at that point on LEO circle

Sweep:
  Vary delta_anomaly ±30 deg around the analytical departure angle.
  For each offset, propagate transfer -> SOI -> Moon periapsis.
  Plot trajectories color-coded by offset, plus periapsis altitude vs offset.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize  import brentq

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# Constants (SI: meters, seconds)
# ============================================================
GP_EARTH        = 3.986004418e14    # m^3/s^2
GP_MOON         = 4.902800066e12    # m^3/s^2
RADIUS_EARTH    = 6.371e6           # m
RADIUS_MOON     = 1.7374e6          # m
DISTANCE_MOON   = 3.844e8           # m (Earth-Moon distance)

MOON_PERIOD__S  = 2.0 * np.pi * np.sqrt(DISTANCE_MOON**3 / GP_EARTH)
MOON_OMEGA      = 2.0 * np.pi / MOON_PERIOD__S     # rad/s
SOI_RADIUS_MOON = DISTANCE_MOON * (GP_MOON / GP_EARTH)**(2.0 / 5.0)

LEO_ALTITUDE__M = 200_000.0
LLO_ALTITUDE__M = 100_000.0


# ============================================================
# Analytical Moon ephemeris (circular orbit, +x at t=0)
# ============================================================

def moon_position(t):
  """Moon position [m] at time t [s]. Circular orbit, counterclockwise."""
  theta = MOON_OMEGA * t
  return DISTANCE_MOON * np.array([np.cos(theta), np.sin(theta)])


def moon_velocity(t):
  """Moon velocity [m/s] at time t [s]."""
  theta = MOON_OMEGA * t
  speed = MOON_OMEGA * DISTANCE_MOON
  return speed * np.array([-np.sin(theta), np.cos(theta)])


# ============================================================
# 2D two-body propagation
# ============================================================

def propagate_2d(state_0, t_span, gp, n_points=5000):
  """
  Propagate 2D two-body dynamics.

  State = [x, y, vx, vy].
  Returns scipy OdeResult with dense_output.
  """
  def eom(t, y):
    r_vec = y[0:2]
    r_mag = np.linalg.norm(r_vec)
    accel = -gp * r_vec / r_mag**3
    return [y[2], y[3], accel[0], accel[1]]

  t_eval = np.linspace(t_span[0], t_span[1], n_points)
  return solve_ivp(
    eom, t_span, state_0,
    method='DOP853', rtol=1e-12, atol=1e-12,
    t_eval=t_eval, dense_output=True,
  )


# ============================================================
# Event detection
# ============================================================

def find_soi_crossing(sol):
  """Scan for Moon SOI entry. Returns (t_cross, state_cross) or (None, None)."""
  for i in range(len(sol.t) - 1):
    t_a, t_b = sol.t[i], sol.t[i + 1]
    d_a = np.linalg.norm(sol.y[0:2, i]     - moon_position(t_a)) - SOI_RADIUS_MOON
    d_b = np.linalg.norm(sol.y[0:2, i + 1] - moon_position(t_b)) - SOI_RADIUS_MOON

    if d_a > 0 and d_b <= 0:
      def residual(t):
        state = sol.sol(t)
        return np.linalg.norm(state[0:2] - moon_position(t)) - SOI_RADIUS_MOON

      t_cross = brentq(residual, t_a, t_b, xtol=1e-3)
      return t_cross, sol.sol(t_cross)

  return None, None


def find_periapsis(sol):
  """Scan for periapsis (r dot v transitions negative -> positive)."""
  for i in range(len(sol.t) - 1):
    rdv_a = np.dot(sol.y[0:2, i],     sol.y[2:4, i])
    rdv_b = np.dot(sol.y[0:2, i + 1], sol.y[2:4, i + 1])

    if rdv_a < 0 and rdv_b >= 0:
      def rdotv(t):
        state = sol.sol(t)
        return np.dot(state[0:2], state[2:4])

      t_peri = brentq(rdotv, sol.t[i], sol.t[i + 1], xtol=1e-3)
      return t_peri, sol.sol(t_peri)

  return None, None


# ============================================================
# Single transfer solver
# ============================================================

def solve_transfer(theta_depart, vel_transfer_depart, delta_vel_mag_1, transfer_time__s):
  """Propagate one patched conic transfer from a given departure angle.

  Returns dict with trajectory data, or None if SOI/periapsis not reached.
  """
  radius_leo = RADIUS_EARTH + LEO_ALTITUDE__M

  pos_depart       = radius_leo * np.array([np.cos(theta_depart), np.sin(theta_depart)])
  vel_hat_prograde = np.array([-np.sin(theta_depart), np.cos(theta_depart)])
  vel_depart       = vel_transfer_depart * vel_hat_prograde

  state_depart = np.array([pos_depart[0], pos_depart[1],
                            vel_depart[0], vel_depart[1]])

  # Phase 1: Earth-centered transfer
  max_prop_time__s = 1.5 * transfer_time__s
  sol_transfer = propagate_2d(state_depart, (0, max_prop_time__s), GP_EARTH, n_points=20000)

  t_soi, state_soi = find_soi_crossing(sol_transfer)
  if t_soi is None:
    return None

  # Transform to Moon-centered frame at SOI
  moon_pos_at_soi = moon_position(t_soi)
  moon_vel_at_soi = moon_velocity(t_soi)

  state_moon_frame = np.array([
    state_soi[0] - moon_pos_at_soi[0],
    state_soi[1] - moon_pos_at_soi[1],
    state_soi[2] - moon_vel_at_soi[0],
    state_soi[3] - moon_vel_at_soi[1],
  ])

  # Phase 2: Moon-centered approach -> periapsis
  sol_moon = propagate_2d(state_moon_frame, (0, 1.0 * 86400.0), GP_MOON, n_points=10000)

  t_peri, state_peri = find_periapsis(sol_moon)
  if t_peri is None:
    return None

  radius_peri   = np.linalg.norm(state_peri[0:2])
  vel_mag_peri  = np.linalg.norm(state_peri[2:4])
  altitude_peri = radius_peri - RADIUS_MOON

  vel_circ_at_peri = np.sqrt(GP_MOON / radius_peri)
  delta_vel_mag_2  = abs(vel_mag_peri - vel_circ_at_peri)
  delta_vel_total  = delta_vel_mag_1 + delta_vel_mag_2

  return {
    'theta_depart'    : theta_depart,
    'delta_vel_mag_1' : delta_vel_mag_1,
    'delta_vel_mag_2' : delta_vel_mag_2,
    'delta_vel_total' : delta_vel_total,
    't_soi'           : t_soi,
    'state_soi'       : state_soi,
    'sol_transfer'    : sol_transfer,
    'sol_moon'        : sol_moon,
    't_peri'          : t_peri,
    'state_peri'      : state_peri,
    'state_moon_frame': state_moon_frame,
    'radius_peri'     : radius_peri,
    'altitude_peri'   : altitude_peri,
    'pos_depart'      : pos_depart,
  }


# ============================================================
# Main
# ============================================================

def main():
  # --- Orbital parameters ---
  radius_leo   = RADIUS_EARTH + LEO_ALTITUDE__M
  radius_llo   = RADIUS_MOON  + LLO_ALTITUDE__M
  vel_circ_leo = np.sqrt(GP_EARTH / radius_leo)

  # --- Hohmann transfer estimates ---
  sma_transfer        = (radius_leo + DISTANCE_MOON) / 2.0
  vel_transfer_depart = np.sqrt(GP_EARTH * (2.0 / radius_leo - 1.0 / sma_transfer))
  delta_vel_mag_1     = vel_transfer_depart - vel_circ_leo
  transfer_time__s    = np.pi * np.sqrt(sma_transfer**3 / GP_EARTH)

  # --- Analytical baseline ---
  theta_moon_0       = 0.0
  theta_moon_arrival = theta_moon_0 + MOON_OMEGA * transfer_time__s
  theta_nominal      = theta_moon_arrival - np.pi

  # --- Sweep delta_anomaly ±30 deg ---
  n_sweep             = 61
  delta_anomaly__deg  = np.linspace(-30.0, 30.0, n_sweep)
  delta_anomaly__rad  = np.radians(delta_anomaly__deg)
  departure_angles    = theta_nominal + delta_anomaly__rad

  print("=" * 70)
  print("2D Patched Conic: Anomaly Sweep ±30 deg Around Analytical Solution")
  print("=" * 70)
  print(f"  LEO radius            : {radius_leo / 1e3:.1f} km")
  print(f"  Hohmann transfer time : {transfer_time__s / 86400:.2f} days")
  print(f"  delta_vel_mag_1       : {delta_vel_mag_1:.2f} m/s")
  print(f"  theta_nominal         : {np.degrees(theta_nominal):.1f} deg")
  print(f"  Sweep                 : {n_sweep} points, [{delta_anomaly__deg[0]:.0f}, {delta_anomaly__deg[-1]:.0f}] deg")
  print()

  # Solve each departure
  results = []
  for i, theta in enumerate(departure_angles):
    da_deg = delta_anomaly__deg[i]
    result = solve_transfer(theta, vel_transfer_depart, delta_vel_mag_1, transfer_time__s)

    if result is None:
      print(f"  [{i+1:3d}/{n_sweep}]  delta_anomaly = {da_deg:+6.1f} deg  -- NO SOI / NO PERIAPSIS")
      results.append(None)
    else:
      status = "IMPACT" if result['radius_peri'] < RADIUS_MOON else "OK"
      print(f"  [{i+1:3d}/{n_sweep}]  delta_anomaly = {da_deg:+6.1f} deg  "
            f"delta_vel_total = {result['delta_vel_total']:8.1f} m/s  "
            f"peri_alt = {result['altitude_peri']/1e3:8.1f} km  [{status}]")
      results.append(result)

  valid = [r for r in results if r is not None]
  print()
  print(f"  {len(valid)} / {n_sweep} reached periapsis")

  if not valid:
    print("  No valid solutions found.")
    return

  # --------------------------------------------------
  # Plot: 4 panels (skip impacts where altitude < 0)
  # --------------------------------------------------
  print()
  print("  Plotting...")

  fig = plt.figure(figsize=(24, 8))
  ax_earth = fig.add_subplot(1, 4, 1)
  ax_moon  = fig.add_subplot(1, 4, 2)
  ax_sweep = fig.add_subplot(1, 4, 3)
  ax_dv2   = fig.add_subplot(1, 4, 4)
  theta_circle = np.linspace(0, 2 * np.pi, 300)

  # Color map by delta_anomaly
  cmap = plt.cm.coolwarm
  norm = plt.Normalize(vmin=-30.0, vmax=30.0)

  # ==============================
  # Left: Earth-centered
  # ==============================
  ax_earth.set_aspect('equal')
  ax_earth.set_title('Earth-Centered Inertial', fontsize=12)
  ax_earth.set_xlabel('X [km]')
  ax_earth.set_ylabel('Y [km]')

  ax_earth.fill(
    RADIUS_EARTH / 1e3 * np.cos(theta_circle),
    RADIUS_EARTH / 1e3 * np.sin(theta_circle),
    color='steelblue', alpha=0.7,
  )
  ax_earth.plot(
    DISTANCE_MOON / 1e3 * np.cos(theta_circle),
    DISTANCE_MOON / 1e3 * np.sin(theta_circle),
    'gray', linestyle='--', alpha=0.3, linewidth=0.8,
  )

  for i, r in enumerate(results):
    if r is None or r['altitude_peri'] < 0:
      continue
    da_deg = delta_anomaly__deg[i]
    color  = cmap(norm(da_deg))
    lw     = 2.0 if abs(da_deg) < 0.5 else 0.8
    alpha  = 1.0 if abs(da_deg) < 0.5 else 0.5

    # Earth-centered leg
    mask = r['sol_transfer'].t <= r['t_soi']
    ax_earth.plot(
      r['sol_transfer'].y[0, mask] / 1e3,
      r['sol_transfer'].y[1, mask] / 1e3,
      color=color, linewidth=lw, alpha=alpha,
    )

    # Moon-centered leg, transformed to Earth frame
    mask_moon      = r['sol_moon'].t <= r['t_peri']
    approach_times = r['sol_moon'].t[mask_moon]
    moon_pos_arr   = np.array([moon_position(r['t_soi'] + t) for t in approach_times])
    ax_earth.plot(
      (r['sol_moon'].y[0, mask_moon] + moon_pos_arr[:, 0]) / 1e3,
      (r['sol_moon'].y[1, mask_moon] + moon_pos_arr[:, 1]) / 1e3,
      color=color, linewidth=lw, alpha=alpha,
    )

    # Departure marker
    ax_earth.plot(r['pos_depart'][0] / 1e3, r['pos_depart'][1] / 1e3,
                  '^', color=color, markersize=4, zorder=5)

  # Moon at t=0
  moon_at_t0 = moon_position(0)
  ax_earth.plot(moon_at_t0[0] / 1e3, moon_at_t0[1] / 1e3, 'o', color='gray', markersize=10)
  ax_earth.annotate('Moon (t=0)', xy=(moon_at_t0[0] / 1e3, moon_at_t0[1] / 1e3),
                     xytext=(15, 10), textcoords='offset points', fontsize=8)

  ax_earth.grid(True, alpha=0.3)

  # ==============================
  # Center: Moon-centered
  # ==============================
  ax_moon.set_aspect('equal')
  ax_moon.set_title('Moon-Centered', fontsize=12)
  ax_moon.set_xlabel('X [km]')
  ax_moon.set_ylabel('Y [km]')

  ax_moon.fill(
    RADIUS_MOON / 1e3 * np.cos(theta_circle),
    RADIUS_MOON / 1e3 * np.sin(theta_circle),
    color='gray', alpha=0.5,
  )
  ax_moon.plot(
    SOI_RADIUS_MOON / 1e3 * np.cos(theta_circle),
    SOI_RADIUS_MOON / 1e3 * np.sin(theta_circle),
    'k:', alpha=0.4, linewidth=0.8,
  )

  for i, r in enumerate(results):
    if r is None or r['altitude_peri'] < 0:
      continue
    da_deg = delta_anomaly__deg[i]
    color  = cmap(norm(da_deg))
    lw     = 2.0 if abs(da_deg) < 0.5 else 0.8
    alpha  = 1.0 if abs(da_deg) < 0.5 else 0.5

    mask_moon = r['sol_moon'].t <= r['t_peri']
    ax_moon.plot(
      r['sol_moon'].y[0, mask_moon] / 1e3,
      r['sol_moon'].y[1, mask_moon] / 1e3,
      color=color, linewidth=lw, alpha=alpha,
    )

    # Periapsis
    ax_moon.plot(
      r['state_peri'][0] / 1e3, r['state_peri'][1] / 1e3,
      '*', color=color, markersize=8, zorder=5,
    )

  ax_moon.grid(True, alpha=0.3)

  # ==============================
  # Right: periapsis altitude vs delta_anomaly
  # ==============================
  ax_sweep.set_title('Periapsis Altitude vs Anomaly Offset', fontsize=12)
  ax_sweep.set_xlabel('delta_anomaly [deg]')
  ax_sweep.set_ylabel('Periapsis altitude [km]')

  sweep_da  = []
  sweep_alt = []
  sweep_dv  = []
  sweep_dv2 = []
  for i, r in enumerate(results):
    if r is None or r['altitude_peri'] < 0:
      continue
    sweep_da.append(delta_anomaly__deg[i])
    sweep_alt.append(r['altitude_peri'] / 1e3)
    sweep_dv.append(r['delta_vel_total'])
    sweep_dv2.append(r['delta_vel_mag_2'])

  scatter = ax_sweep.scatter(sweep_da, sweep_alt, c=sweep_dv, cmap='viridis_r',
                              s=30, zorder=5, edgecolors='k', linewidths=0.5)
  ax_sweep.plot(sweep_da, sweep_alt, 'k-', linewidth=0.8, alpha=0.4)

  # Mark the analytical solution (delta_anomaly = 0)
  ax_sweep.axvline(0, color='red', linestyle='--', alpha=0.5, label='Analytical (0 deg)')

  # Mark lunar surface
  ax_sweep.axhline(0, color='gray', linestyle='-', alpha=0.5, label='Lunar surface')

  # Mark target LLO altitude
  ax_sweep.axhline(LLO_ALTITUDE__M / 1e3, color='green', linestyle=':', alpha=0.5, label=f'LLO ({LLO_ALTITUDE__M/1e3:.0f} km)')

  cbar = plt.colorbar(scatter, ax=ax_sweep, shrink=0.8, pad=0.02)
  cbar.set_label('delta_vel_total [m/s]', fontsize=9)
  ax_sweep.legend(fontsize=8)
  ax_sweep.grid(True, alpha=0.3)

  # ==============================
  # Fourth: delta_vel_mag_2 vs periapsis altitude
  # ==============================
  ax_dv2.set_title('Circularization Cost vs Periapsis Altitude', fontsize=12)
  ax_dv2.set_xlabel('Periapsis altitude [km]')
  ax_dv2.set_ylabel('delta_vel_mag_2 [m/s]')

  # Compute v_infinity from the nominal analytical solution for the theory curve
  # Use the median v_infinity from the sweep results as representative
  v_inf_values = []
  for r in results:
    if r is not None and r['altitude_peri'] >= 0:
      v_inf = np.linalg.norm(r['state_moon_frame'][2:4])
      v_inf_values.append(v_inf)
  v_inf_median = np.median(v_inf_values) if v_inf_values else 800.0

  # Theoretical curve: delta_vel_mag_2(r_p) = sqrt(v_inf^2 + 2*mu/r_p) - sqrt(mu/r_p)
  alt_theory__km  = np.linspace(1, max(sweep_alt) * 1.1 if sweep_alt else 1000, 500)
  r_p_theory      = (alt_theory__km * 1e3) + RADIUS_MOON
  delta_vel_mag_2_theory = (
    np.sqrt(v_inf_median**2 + 2.0 * GP_MOON / r_p_theory)
    - np.sqrt(GP_MOON / r_p_theory)
  )

  # Theoretical minimum at r_p = 2*mu / v_inf^2
  r_p_optimal     = 2.0 * GP_MOON / v_inf_median**2
  alt_optimal__km = (r_p_optimal - RADIUS_MOON) / 1e3
  delta_vel_mag_2_optimal = (
    np.sqrt(v_inf_median**2 + 2.0 * GP_MOON / r_p_optimal)
    - np.sqrt(GP_MOON / r_p_optimal)
  )

  ax_dv2.plot(alt_theory__km, delta_vel_mag_2_theory, 'k-', linewidth=1.5,
              alpha=0.6, label=f'Theory ($v_\\infty$ = {v_inf_median:.0f} m/s)')

  # Scatter: simulated data
  scatter_colors = [delta_anomaly__deg[i] for i, r in enumerate(results)
                    if r is not None and r['altitude_peri'] >= 0]
  ax_dv2.scatter(sweep_alt, sweep_dv2, c=scatter_colors, cmap='coolwarm',
                  norm=norm, s=30, zorder=5, edgecolors='k', linewidths=0.5,
                  label='Sweep results')

  # Mark optimal altitude
  if alt_optimal__km > 0:
    ax_dv2.axvline(alt_optimal__km, color='red', linestyle='--', alpha=0.6,
                    label=f'Optimal ({alt_optimal__km:.0f} km)')
    ax_dv2.plot(alt_optimal__km, delta_vel_mag_2_optimal,
                'r*', markersize=14, zorder=6)

  # Mark target LLO
  ax_dv2.axvline(LLO_ALTITUDE__M / 1e3, color='green', linestyle=':', alpha=0.5,
                  label=f'LLO ({LLO_ALTITUDE__M/1e3:.0f} km)')

  ax_dv2.legend(fontsize=8)
  ax_dv2.grid(True, alpha=0.3)

  # Shared colorbar for trajectory panels
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])
  cbar_traj = fig.colorbar(sm, ax=[ax_earth, ax_moon], shrink=0.7, pad=0.02,
                            location='bottom', aspect=40)
  cbar_traj.set_label('delta_anomaly [deg]', fontsize=10)

  plt.tight_layout()
  save_path = os.path.join(PROJECT_ROOT, 'output', 'patched_conic_2d_sweep.png')
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path, dpi=150, bbox_inches='tight')
  plt.show()

  print(f"  Saved: {save_path}")
  print("  Done!")


if __name__ == '__main__':
  main()
