"""
POC porkchop plot for circular-to-circular coplanar rendezvous around Earth.

Assumptions:
- Two-body Keplerian dynamics, Earth as point mass.
- Spacecraft starts on a circular parking orbit (prograde, equatorial).
- Target is a massless point on a larger circular orbit (prograde, coplanar).
- Two impulsive burns: depart parking orbit, arrive at target (match its velocity).
- Single-revolution Lambert (Type I and II).
"""

import multiprocessing as mp
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import root, minimize, least_squares

# Analytic-Jacobian Lambert kernel (∂v1,∂v2 w.r.t. r1,r2,dt), reused from the
# sibling handcrafted_trajectory_design project. Unit-agnostic (mu passed in).
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "handcrafted_trajectory_design"))
from lambert_numba import lambert_with_jac_nb  # noqa: E402

# -------- Constants / scenario --------
MU = 398600.4418            # km^3/s^2 — Earth
MU_MOON_FULL = 4902.800066  # km^3/s^2 — Moon
MU_MOON_SCALE = 1.0         # homotopy knob: 0.0 = conic (no moon gravity), 1.0 = full
MU_MOON = MU_MOON_FULL * MU_MOON_SCALE  # effective value used in dynamics

# Target offset from moon center to avoid the 1/r^2 singularity.
# Arrival point sits this far Earthward of moon center (near-side surface).
MOON_ARRIVAL_OFFSET = 1737.0  # km (≈ lunar radius)

# Abort propagation if the spacecraft passes inside this distance of moon center.
# Stops adaptive-step collapse when the trajectory dives into the Moon's gravity well.
MOON_CLOSE_APPROACH_ABORT = 500.0  # km

# Skip Lambert seeds whose specific angular momentum falls below this.
# Hohmann h ≈ 70,580 km²/s; circular-park h ≈ 51,580 km²/s; rectilinear h → 0.
H_MIN_SEED = 5000.0  # km²/s

R_EARTH = 6378.0            # km

R_PARK = R_EARTH + 300.0    # parking orbit radius (300 km LEO)
R_TGT = 96_100.0            # target circular orbit radius (25% of lunar distance)

THETA_PARK_0 = 0.0          # spacecraft true anomaly at t=0 [rad]
THETA_TGT_0 = np.deg2rad(90.0)  # target mean anomaly at t=0 [rad]

# Lunar (target) orbit eccentricity. 0.0 => circle of radius R_TGT (a = R_TGT,
# periapsis along +x). THETA_TGT_0 + N_TGT*t is the Moon's MEAN anomaly, so at
# ecc=0 this reduces exactly to the previous circular model. Default value; the
# grid threads the active value through worker args (globals don't cross spawn).
MOON_ECC = 0.0

N_PARK = np.sqrt(MU / R_PARK**3)
N_TGT = np.sqrt(MU / R_TGT**3)

# Approximate two-impulse Hohmann ΔV between the coplanar circular orbits.
_A_HOHMANN = 0.5 * (R_PARK + R_TGT)
DV_HOHMANN = (abs(np.sqrt(MU * (2 / R_PARK - 1 / _A_HOHMANN)) - np.sqrt(MU / R_PARK))
              + abs(np.sqrt(MU / R_TGT) - np.sqrt(MU * (2 / R_TGT - 1 / _A_HOHMANN))))

# Skip Newton shooting when the conic Lambert seed's total ΔV is already above this.
# Anything beyond is uninteresting (Hohmann ΔV ≈ 4.14 km/s for reference).
DV_MAX_SEED = 5.5  # km/s

# Grid (hours)
T_DEP = np.linspace(0.0, 2.0, 41)       # departure window
TOF = np.linspace(2.0, 200.0, 100)      # time of flight


# -------- Lambert (universal variable, JIT-compiled) --------
@njit(cache=True, fastmath=True)
def _stumpff_C(z):
    if z > 1e-6:
        s = np.sqrt(z)
        return (1.0 - np.cos(s)) / z
    if z < -1e-6:
        s = np.sqrt(-z)
        return (1.0 - np.cosh(s)) / z
    return 0.5


@njit(cache=True, fastmath=True)
def _stumpff_S(z):
    if z > 1e-6:
        s = np.sqrt(z)
        return (s - np.sin(s)) / s**3
    if z < -1e-6:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / s**3
    return 1.0 / 6.0


@njit(cache=True, fastmath=True)
def _t_and_y(psi, A, r12, mu):
    """Returns (t, y, valid)."""
    C = _stumpff_C(psi)
    S = _stumpff_S(psi)
    if C <= 0.0:
        return 0.0, 0.0, False
    y = r12 + A * (psi * S - 1.0) / np.sqrt(C)
    if y < 0.0:
        return 0.0, 0.0, False
    chi = np.sqrt(y / C)
    t = (chi**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
    return t, y, True


@njit(cache=True, fastmath=True)
def _lambert_njit(r1x, r1y, r1z, r2x, r2y, r2z,
                  tof, mu, prograde, n_rev, branch_short, tol, max_iter):
    """Numba-compiled Lambert kernel.
    prograde: 1 = prograde, 0 = retrograde
    branch_short: 1 = short branch (only for n_rev > 0), 0 = long branch
    Returns (ok, v1x, v1y, v1z, v2x, v2y, v2z).
    """
    r1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    r2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
    cos_dnu = (r1x * r2x + r1y * r2y + r1z * r2z) / (r1 * r2)
    if cos_dnu > 1.0:
        cos_dnu = 1.0
    elif cos_dnu < -1.0:
        cos_dnu = -1.0
    # z-component of r1 x r2 (works in 2D where r1z = r2z = 0)
    cross_z = r1x * r2y - r1y * r2x

    dnu = np.arccos(cos_dnu)
    if prograde == 1:
        if cross_z < 0.0:
            dnu = 2.0 * np.pi - dnu
    else:
        if cross_z >= 0.0:
            dnu = 2.0 * np.pi - dnu

    sin_dnu = np.sin(dnu)
    if abs(sin_dnu) < 1e-10:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    A = sin_dnu * np.sqrt(r1 * r2 / (1.0 - cos_dnu))
    if abs(A) < 1e-10:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    r12 = r1 + r2
    y = 0.0

    if n_rev == 0:
        psi_low = -4.0 * np.pi
        psi_up = (2.0 * np.pi)**2 - 1e-6
        psi = 0.0
        converged = False
        for _ in range(max_iter):
            t, y_calc, valid = _t_and_y(psi, A, r12, mu)
            bump = 0
            while (not valid) and bump < 200:
                psi_low += 0.1
                psi = 0.5 * (psi_low + psi_up)
                t, y_calc, valid = _t_and_y(psi, A, r12, mu)
                bump += 1
            if not valid:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            y = y_calc
            if abs(t - tof) < tol:
                converged = True
                break
            if t < tof:
                psi_low = psi
            else:
                psi_up = psi
            psi = 0.5 * (psi_low + psi_up)
        if not converged:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        psi_low_band = (2.0 * np.pi * n_rev)**2 + 1e-6
        psi_up_band = (2.0 * np.pi * (n_rev + 1))**2 - 1e-6

        # Golden-section search for psi at min t in this band
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a_g = psi_low_band
        b_g = psi_up_band
        c_g = b_g - gr * (b_g - a_g)
        d_g = a_g + gr * (b_g - a_g)
        for _ in range(80):
            tc, _yc, vc = _t_and_y(c_g, A, r12, mu)
            td_val, _yd, vd = _t_and_y(d_g, A, r12, mu)
            if not vc:
                tc = 1e30
            if not vd:
                td_val = 1e30
            if tc < td_val:
                b_g = d_g
            else:
                a_g = c_g
            c_g = b_g - gr * (b_g - a_g)
            d_g = a_g + gr * (b_g - a_g)
        psi_min = 0.5 * (a_g + b_g)
        t_min, _ym, v_min_ok = _t_and_y(psi_min, A, r12, mu)
        if (not v_min_ok) or tof < t_min:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if branch_short == 1:
            psi_low = psi_low_band
            psi_up = psi_min
        else:
            psi_low = psi_min
            psi_up = psi_up_band

        psi = 0.5 * (psi_low + psi_up)
        converged = False
        for _ in range(max_iter):
            t, y_calc, valid = _t_and_y(psi, A, r12, mu)
            if not valid:
                psi_low += 1e-3
                psi = 0.5 * (psi_low + psi_up)
                continue
            y = y_calc
            if abs(t - tof) < tol:
                converged = True
                break
            if branch_short == 1:
                if t < tof:
                    psi_up = psi
                else:
                    psi_low = psi
            else:
                if t < tof:
                    psi_low = psi
                else:
                    psi_up = psi
            psi = 0.5 * (psi_low + psi_up)
        if not converged:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    f = 1.0 - y / r1
    g = A * np.sqrt(y / mu)
    gdot = 1.0 - y / r2
    if abs(g) < 1e-12:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    v1x = (r2x - f * r1x) / g
    v1y = (r2y - f * r1y) / g
    v1z = (r2z - f * r1z) / g
    v2x = (gdot * r2x - r1x) / g
    v2y = (gdot * r2y - r1y) / g
    v2z = (gdot * r2z - r1z) / g
    return True, v1x, v1y, v1z, v2x, v2y, v2z


def lambert_uv(r1_vec, r2_vec, tof, mu, prograde=True, tol=1e-8, max_iter=400):
    """Single-rev Lambert wrapper. Returns (v1, v2) or None."""
    pg = 1 if prograde else 0
    ok, a, b, c, d, e, f = _lambert_njit(
        r1_vec[0], r1_vec[1], r1_vec[2],
        r2_vec[0], r2_vec[1], r2_vec[2],
        tof, mu, pg, 0, 0, tol, max_iter,
    )
    if not ok:
        return None
    return np.array([a, b, c]), np.array([d, e, f])


def lambert_uv_multirev(r1_vec, r2_vec, tof, mu, prograde=True,
                        n_rev=0, branch="long", tol=1e-8, max_iter=400):
    """Multi-rev Lambert wrapper. Returns (v1, v2) or None."""
    pg = 1 if prograde else 0
    bs = 1 if branch == "short" else 0
    ok, a, b, c, d, e, f = _lambert_njit(
        r1_vec[0], r1_vec[1], r1_vec[2],
        r2_vec[0], r2_vec[1], r2_vec[2],
        tof, mu, pg, n_rev, bs, tol, max_iter,
    )
    if not ok:
        return None
    return np.array([a, b, c]), np.array([d, e, f])


# -------- State helpers --------
def circ_state(R, theta0, n, t):
    th = theta0 + n * t
    r = R * np.array([np.cos(th), np.sin(th), 0.0])
    v = R * n * np.array([-np.sin(th), np.cos(th), 0.0])
    return r, v


# -------- Grid search (parallel + JIT) --------
def _solve_cell_n(args):
    """Worker: solve one (i, j) cell for one N (both branches if N>0).
    Returns (i, j, best_dv) — best_dv is np.inf if no solution exists.
    """
    i, j, t_d, tof_s, n_rev = args
    # Inline circ_state to avoid extra function call overhead
    th_park = THETA_PARK_0 + N_PARK * t_d
    r1x = R_PARK * np.cos(th_park)
    r1y = R_PARK * np.sin(th_park)
    vpx = -R_PARK * N_PARK * np.sin(th_park)
    vpy = R_PARK * N_PARK * np.cos(th_park)

    th_tgt = THETA_TGT_0 + N_TGT * (t_d + tof_s)
    r2x = R_TGT * np.cos(th_tgt)
    r2y = R_TGT * np.sin(th_tgt)
    vtx = -R_TGT * N_TGT * np.sin(th_tgt)
    vty = R_TGT * N_TGT * np.cos(th_tgt)

    if n_rev == 0:
        branches = (0,)  # branch_short ignored; pick one
    else:
        branches = (0, 1)  # long, short

    best = np.inf
    for bs in branches:
        ok, v1x, v1y, _v1z, v2x, v2y, _v2z = _lambert_njit(
            r1x, r1y, 0.0, r2x, r2y, 0.0,
            tof_s, MU, 1, n_rev, bs, 1e-8, 400,
        )
        if not ok:
            continue
        dv1 = np.sqrt((v1x - vpx) ** 2 + (v1y - vpy) ** 2)
        dv2 = np.sqrt((vtx - v2x) ** 2 + (vty - v2y) ** 2)
        dv = dv1 + dv2
        if dv < best:
            best = dv
    return i, j, best


def run_grid_multirev(max_n_rev=None, hard_cap=30, verbose=True, n_workers=None):
    """Parallel grid sweep over N=0,1,2,... until no cell has any solution.
    Returns (dv_best, n_best, dv_n0):
      dv_best[i,j] = min total ΔV across all N (NaN if unsolvable)
      n_best[i,j]  = N that achieves dv_best (or -1)
      dv_n0[i,j]   = ΔV for N=0 specifically (NaN if no N=0 solution)
    """
    n_t, n_d = len(TOF), len(T_DEP)
    dv_best = np.full((n_t, n_d), np.inf)
    n_best = np.full((n_t, n_d), -1, dtype=np.int32)
    dv_n0 = np.full((n_t, n_d), np.inf)

    if n_workers is None:
        n_workers = mp.cpu_count()

    n_rev = 0
    with mp.Pool(n_workers) as pool:
        while True:
            cell_args = [
                (i, j, float(T_DEP[j] * 3600.0), float(TOF[i] * 3600.0), n_rev)
                for i in range(n_t) for j in range(n_d)
            ]
            chunksize = max(1, len(cell_args) // (n_workers * 4))
            results = pool.map(_solve_cell_n, cell_args, chunksize=chunksize)

            any_solution = False
            for i, j, dv in results:
                if np.isfinite(dv):
                    any_solution = True
                    if n_rev == 0:
                        dv_n0[i, j] = dv
                    if dv < dv_best[i, j]:
                        dv_best[i, j] = dv
                        n_best[i, j] = n_rev

            if verbose:
                n_cells = int(np.sum(n_best == n_rev))
                if any_solution:
                    print(f"  N={n_rev}: best in {n_cells} cells")
                else:
                    print(f"  N={n_rev}: no solutions, stopping")

            if not any_solution:
                break
            if max_n_rev is not None and n_rev >= max_n_rev:
                break
            n_rev += 1
            if n_rev > hard_cap:
                if verbose:
                    print(f"  reached hard cap N={hard_cap}, stopping")
                break

    dv_best[~np.isfinite(dv_best)] = np.nan
    dv_n0[~np.isfinite(dv_n0)] = np.nan
    return dv_best, n_best, dv_n0


# -------- 3-body dynamics + shooting --------
def _moon_rv(M, a, n, ecc):
    """Moon position & velocity (2-D) on a Keplerian ellipse: semi-major axis
    a, mean motion n, eccentricity ecc, periapsis along +x, mean anomaly M.
    Returns (r[2], v[2]) with v = dr/dt (true Keplerian velocity), so the
    optimizer's ∂r/∂t = v chain-rule term stays exact. At ecc=0 this is the
    circular orbit a·[cosM, sinM], velocity a·n·[-sinM, cosM]."""
    if ecc <= 0.0:
        c, s = np.cos(M), np.sin(M)
        return (np.array([a * c, a * s]),
                np.array([-a * n * s, a * n * c]))
    twopi = 2.0 * np.pi
    Mr = M - twopi * np.floor(M / twopi)          # wrap to [0, 2π) for Newton
    E = Mr if ecc < 0.8 else np.pi
    for _ in range(60):
        dE = (E - ecc * np.sin(E) - Mr) / (1.0 - ecc * np.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break
    cE, sE = np.cos(E), np.sin(E)
    b = a * np.sqrt(1.0 - ecc * ecc)
    Edot = n / (1.0 - ecc * cE)                    # dE/dt
    return (np.array([a * (cE - ecc), b * sE]),
            np.array([-a * sE * Edot, b * cE * Edot]))


def _moon_state(t, ecc=None):
    if ecc is None:
        ecc = MOON_ECC
    rm, vm = _moon_rv(THETA_TGT_0 + N_TGT * t, R_TGT, N_TGT, ecc)
    return (np.array([rm[0], rm[1], 0.0]),
            np.array([vm[0], vm[1], 0.0]))


@njit(cache=True, fastmath=True)
def _dynamics_em(t, y, mu_moon):
    """Earth + Moon point-mass dynamics. mu_moon passed in to support homotopy scaling."""
    th = THETA_TGT_0 + N_TGT * t
    rmx = R_TGT * np.cos(th)
    rmy = R_TGT * np.sin(th)

    rx = y[0]; ry = y[1]; rz = y[2]
    vx = y[3]; vy_ = y[4]; vz = y[5]

    dx = rmx - rx
    dy_ = rmy - ry
    dz = -rz

    r2e = rx * rx + ry * ry + rz * rz
    r3e = r2e * np.sqrt(r2e)
    r2m = dx * dx + dy_ * dy_ + dz * dz
    r3m = r2m * np.sqrt(r2m)

    out = np.empty(6)
    out[0] = vx
    out[1] = vy_
    out[2] = vz
    out[3] = -MU * rx / r3e + mu_moon * dx / r3m
    out[4] = -MU * ry / r3e + mu_moon * dy_ / r3m
    out[5] = -MU * rz / r3e + mu_moon * dz / r3m
    return out


def _close_moon_event(t, y, mu_moon):
    """solve_ivp terminal event: distance-to-moon-center minus the abort radius.
    Crossing zero (from above) ends the integration before the singular pull-in."""
    th = THETA_TGT_0 + N_TGT * t
    rmx = R_TGT * np.cos(th)
    rmy = R_TGT * np.sin(th)
    dx = rmx - y[0]
    dy_ = rmy - y[1]
    dz = -y[2]
    return np.sqrt(dx * dx + dy_ * dy_ + dz * dz) - MOON_CLOSE_APPROACH_ABORT


_close_moon_event.terminal = True
_close_moon_event.direction = -1  # only abort when approaching, not receding


def _propagate_em(r0, v0, t0, t1, rtol=1e-6, atol=1e-1, mu_moon=None):
    if mu_moon is None:
        mu_moon = MU_MOON
    sol = solve_ivp(_dynamics_em, (t0, t1), np.concatenate([r0, v0]),
                    method="DOP853", rtol=rtol, atol=atol, args=(mu_moon,),
                    events=_close_moon_event)
    return sol.y[:3, -1], sol.y[3:, -1]


@njit(cache=True, fastmath=True)
def _dynamics_em_stm(t, y, mu_moon):
    """Augmented dynamics: 6-state + 36-element row-major STM (Phi). mu_moon passed in."""
    th = THETA_TGT_0 + N_TGT * t
    rmx = R_TGT * np.cos(th)
    rmy = R_TGT * np.sin(th)

    rx = y[0]; ry = y[1]; rz = y[2]
    vx = y[3]; vy_ = y[4]; vz = y[5]

    dx = rmx - rx
    dy_ = rmy - ry
    dz = -rz

    r2e = rx * rx + ry * ry + rz * rz
    r3e = r2e * np.sqrt(r2e)
    r5e = r3e * r2e
    r2m = dx * dx + dy_ * dy_ + dz * dz
    r3m = r2m * np.sqrt(r2m)
    r5m = r3m * r2m

    out = np.empty(42)
    out[0] = vx
    out[1] = vy_
    out[2] = vz
    out[3] = -MU * rx / r3e + mu_moon * dx / r3m
    out[4] = -MU * ry / r3e + mu_moon * dy_ / r3m
    out[5] = -MU * rz / r3e + mu_moon * dz / r3m

    # Gravity gradient tensor G = ∂a/∂r  (symmetric)
    inv_r3e = 1.0 / r3e
    inv_r5e = 1.0 / r5e
    inv_r3m = 1.0 / r3m
    inv_r5m = 1.0 / r5m

    G00 = -MU * inv_r3e + 3.0 * MU * rx * rx * inv_r5e \
          - mu_moon * inv_r3m + 3.0 * mu_moon * dx * dx * inv_r5m
    G11 = -MU * inv_r3e + 3.0 * MU * ry * ry * inv_r5e \
          - mu_moon * inv_r3m + 3.0 * mu_moon * dy_ * dy_ * inv_r5m
    G22 = -MU * inv_r3e + 3.0 * MU * rz * rz * inv_r5e \
          - mu_moon * inv_r3m + 3.0 * mu_moon * dz * dz * inv_r5m
    G01 = 3.0 * MU * rx * ry * inv_r5e + 3.0 * mu_moon * dx * dy_ * inv_r5m
    G02 = 3.0 * MU * rx * rz * inv_r5e + 3.0 * mu_moon * dx * dz * inv_r5m
    G12 = 3.0 * MU * ry * rz * inv_r5e + 3.0 * mu_moon * dy_ * dz * inv_r5m

    # dPhi/dt = A @ Phi, with A = [[0, I], [G, 0]]
    # Phi stored row-major: Phi[i, k] = y[6 + i*6 + k]
    for k in range(6):
        p0k = y[6 + 0 * 6 + k]
        p1k = y[6 + 1 * 6 + k]
        p2k = y[6 + 2 * 6 + k]
        out[6 + 0 * 6 + k] = y[6 + 3 * 6 + k]
        out[6 + 1 * 6 + k] = y[6 + 4 * 6 + k]
        out[6 + 2 * 6 + k] = y[6 + 5 * 6 + k]
        out[6 + 3 * 6 + k] = G00 * p0k + G01 * p1k + G02 * p2k
        out[6 + 4 * 6 + k] = G01 * p0k + G11 * p1k + G12 * p2k
        out[6 + 5 * 6 + k] = G02 * p0k + G12 * p1k + G22 * p2k
    return out


def _propagate_em_stm(r0, v0, t0, t1, rtol=1e-9, atol=1e-3, mu_moon=None):
    """Propagate state + STM. Returns (r_f, v_f, Phi_rv) where
    Phi_rv[i, j] = ∂r_f[i] / ∂v0[j]."""
    if mu_moon is None:
        mu_moon = MU_MOON
    y0 = np.zeros(42)
    y0[0] = r0[0]; y0[1] = r0[1]; y0[2] = r0[2]
    y0[3] = v0[0]; y0[4] = v0[1]; y0[5] = v0[2]
    for i in range(6):
        y0[6 + i * 6 + i] = 1.0
    sol = solve_ivp(_dynamics_em_stm, (t0, t1), y0,
                    method="DOP853", rtol=rtol, atol=atol, args=(mu_moon,),
                    events=_close_moon_event)
    yend = sol.y[:, -1]
    r_f = yend[:3]
    v_f = yend[3:6]
    phi_rv = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            phi_rv[i, j] = yend[6 + i * 6 + (j + 3)]
    return r_f, v_f, phi_rv


# Status codes returned by the shooter / cell worker
STATUS_OK = 0
STATUS_LAMBERT_FAIL = 1
STATUS_MAX_ITER = 2
STATUS_SINGULAR_JAC = 3
STATUS_SEED_DEGENERATE = 4  # |h| below threshold (rectilinear-ish)
STATUS_SEED_OVER_CAP = 5    # conic seed ΔV above DV_MAX_SEED — Newton skipped


def _shoot_to_moon(r1, v_guess, t_dep, t_arr, r_target,
                   pos_tol=50.0, max_iter=8, mu_moon=None):
    """Newton shooter using analytical STM Jacobian.
    Returns (v_t1, v_final, status, final_residual_km).
    status: STATUS_OK / STATUS_MAX_ITER / STATUS_SINGULAR_JAC
    """
    if mu_moon is None:
        mu_moon = MU_MOON
    v = v_guess.astype(np.float64).copy()
    v_f = v
    last_residual = np.inf
    # --- Newton iteration commented out: propagate the conic seed under moon
    #     gravity once, no STM correction step ---
    # for _ in range(max_iter):
    #     r_f, v_f, phi_rv = _propagate_em_stm(r1, v, t_dep, t_arr, mu_moon=mu_moon)
    #     residual = r_f - r_target
    #     last_residual = float(np.linalg.norm(residual))
    #     if last_residual < pos_tol:
    #         return v, v_f, STATUS_OK, last_residual
    #     try:
    #         dv = np.linalg.solve(phi_rv, -residual)
    #     except np.linalg.LinAlgError:
    #         return v, v_f, STATUS_SINGULAR_JAC, last_residual
    #     v = v + dv
    r_f, v_f, _ = _propagate_em_stm(r1, v, t_dep, t_arr, mu_moon=mu_moon)
    last_residual = float(np.linalg.norm(r_f - r_target))
    if last_residual < pos_tol:
        return v, v_f, STATUS_OK, last_residual
    return v, v_f, STATUS_MAX_ITER, last_residual


def _newton_shoot_counted(r1, v0, t_dep, t_arr, r_target,
                          mu_moon=None, pos_tol=50.0, max_iter=30):
    """Same STM-Jacobian Newton shoot as _shoot_to_moon, but returns the
    iteration count: (v, v_f, status, n_iter, residual_km). n_iter is the
    number of Newton correction steps taken to reach pos_tol (0 if the
    initial guess already satisfies it)."""
    if mu_moon is None:
        mu_moon = MU_MOON
    v = np.asarray(v0, dtype=np.float64).copy()
    v_f = v
    last_residual = np.inf
    for it in range(max_iter):
        r_f, v_f, phi_rv = _propagate_em_stm(r1, v, t_dep, t_arr,
                                             mu_moon=mu_moon)
        last_residual = float(np.linalg.norm(r_f - r_target))
        if last_residual < pos_tol:
            return v, v_f, STATUS_OK, it, last_residual
        try:
            dv = np.linalg.solve(phi_rv, -(r_f - r_target))
        except np.linalg.LinAlgError:
            return v, v_f, STATUS_SINGULAR_JAC, it, last_residual
        v = v + dv
    return v, v_f, STATUS_MAX_ITER, max_iter, last_residual


def _solve_cell_with_moon(args):
    """Worker: cell with Moon gravity, sweeping N=0 + N=1 (short, long) Lambert seeds.
    Returns the best (lowest-ΔV) successful shoot for the cell."""
    i, j, t_d, tof_s = args
    th_park = THETA_PARK_0 + N_PARK * t_d
    r1 = np.array([R_PARK * np.cos(th_park), R_PARK * np.sin(th_park), 0.0])
    v_park = np.array([-R_PARK * N_PARK * np.sin(th_park),
                        R_PARK * N_PARK * np.cos(th_park), 0.0])

    r_moon, v_moon = _moon_state(t_d + tof_s)
    r_moon_mag = float(np.linalg.norm(r_moon))
    r_target = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / r_moon_mag)

    best_dv = np.inf
    best_n = -1
    best_residual = np.nan
    last_status = STATUS_LAMBERT_FAIL  # most recent non-OK status (for reporting)

    # (n_rev, branch_short) pairs to try. N=0 branch is irrelevant; N=1 tries both.
    # attempts = [(0, 0), (1, 1), (1, 0)]  # N=1 revs commented out
    attempts = [(0, 0)]  # N=0 only
    for n_rev, branch_short in attempts:
        ok, v1x, v1y, _v1z, v2x, v2y, _v2z = _lambert_njit(
            r1[0], r1[1], 0.0, r_target[0], r_target[1], 0.0,
            tof_s, MU, 1, n_rev, branch_short, 1e-8, 400,
        )
        if not ok:
            last_status = STATUS_LAMBERT_FAIL
            continue

        h_z = r1[0] * v1y - r1[1] * v1x
        if abs(h_z) < H_MIN_SEED:
            last_status = STATUS_SEED_DEGENERATE
            continue

        # Cheap conic-seed ΔV screen: if Lambert already says it's expensive,
        # don't bother integrating Earth+Moon dynamics through Newton.
        dv1_seed = np.sqrt((v1x - v_park[0]) ** 2 + (v1y - v_park[1]) ** 2)
        dv2_seed = np.sqrt((v_moon[0] - v2x) ** 2 + (v_moon[1] - v2y) ** 2)
        if dv1_seed + dv2_seed > DV_MAX_SEED:
            last_status = STATUS_SEED_OVER_CAP
            continue

        v_guess = np.array([v1x, v1y, 0.0])
        v_t1, v_f, st, res = _shoot_to_moon(r1, v_guess, t_d, t_d + tof_s, r_target)
        if st != STATUS_OK:
            last_status = st
            continue

        dv1 = float(np.linalg.norm(v_t1 - v_park))
        dv2 = float(np.linalg.norm(v_moon - v_f))
        total = dv1 + dv2
        if total < best_dv:
            best_dv = total
            best_n = n_rev
            best_residual = res

    if best_n >= 0:
        return i, j, best_dv, best_n, STATUS_OK, best_residual
    return i, j, np.nan, -1, last_status, np.nan


def run_grid_with_moon(verbose=True, n_workers=None):
    """Porkchop grid with Moon gravity, sweeping N=0 + N=1 branches per cell.
    Returns (dv, n_best, status, residual) — each shape (n_TOF, n_DEP).
    n_best is the rev count of the winning seed (-1 where no solution exists)."""
    import time
    n_t, n_d = len(TOF), len(T_DEP)
    dv = np.full((n_t, n_d), np.nan)
    n_best = np.full((n_t, n_d), -1, dtype=np.int8)
    status = np.full((n_t, n_d), -1, dtype=np.int8)
    residual = np.full((n_t, n_d), np.nan)

    if n_workers is None:
        n_workers = mp.cpu_count()

    cell_args = [
        (i, j, float(T_DEP[j] * 3600.0), float(TOF[i] * 3600.0))
        for i in range(n_t) for j in range(n_d)
    ]
    chunksize = max(1, len(cell_args) // (n_workers * 4))

    total = len(cell_args)
    t_start = time.time()
    with mp.Pool(n_workers) as pool:
        for n_done, (i, j, val, nb, st, res) in enumerate(
            pool.imap_unordered(_solve_cell_with_moon, cell_args, chunksize=chunksize), 1
        ):
            dv[i, j] = val
            n_best[i, j] = nb
            status[i, j] = st
            residual[i, j] = res
            if verbose:
                pct = 100.0 * n_done / total
                elapsed = time.time() - t_start
                eta = elapsed * (total - n_done) / n_done if n_done > 0 else 0.0
                print(f"\r  moon-grid: {n_done}/{total}  ({pct:5.1f}%)  "
                      f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s",
                      end="", flush=True)
        if verbose:
            print()
            n_ok = int(np.sum(status == STATUS_OK))
            n_lambert = int(np.sum(status == STATUS_LAMBERT_FAIL))
            n_maxiter = int(np.sum(status == STATUS_MAX_ITER))
            n_sing = int(np.sum(status == STATUS_SINGULAR_JAC))
            n_degen = int(np.sum(status == STATUS_SEED_DEGENERATE))
            n_cap = int(np.sum(status == STATUS_SEED_OVER_CAP))
            n_n0 = int(np.sum(n_best == 0))
            n_n1 = int(np.sum(n_best == 1))
            print(f"  status: ok={n_ok}, lambert_fail={n_lambert}, "
                  f"max_iter={n_maxiter}, singular_jac={n_sing}, "
                  f"seed_degen={n_degen}, seed_over_cap={n_cap}")
            print(f"  winning N: N=0 in {n_n0} cells, N=1 in {n_n1} cells")

    return dv, n_best, status, residual


# -------- Plot --------
def print_summary(dv_total):
    a_h = 0.5 * (R_PARK + R_TGT)
    v_park = np.sqrt(MU / R_PARK)
    v_tgt = np.sqrt(MU / R_TGT)
    v_h1 = np.sqrt(MU * (2 / R_PARK - 1 / a_h))
    v_h2 = np.sqrt(MU * (2 / R_TGT - 1 / a_h))
    dv_hohmann = abs(v_h1 - v_park) + abs(v_tgt - v_h2)
    print(f"Hohmann reference ΔV = {dv_hohmann:.4f} km/s")
    print(f"Grid (N=0) min ΔV    = {np.nanmin(dv_total):.4f} km/s")


def transfer_arc(r1_vec, v1_vec, dnu, mu, n_pts=200):
    """Sample the Lambert transfer arc analytically from (r1, v1) over angle dnu."""
    r1 = np.linalg.norm(r1_vec)
    h_vec = np.cross(r1_vec, v1_vec)
    e_vec = np.cross(v1_vec, h_vec) / mu - r1_vec / r1
    e = np.linalg.norm(e_vec)
    p = np.dot(h_vec, h_vec) / mu  # h^2/mu

    if e > 1e-10:
        cos_nu1 = np.clip(np.dot(e_vec, r1_vec) / (e * r1), -1.0, 1.0)
        nu1 = np.arccos(cos_nu1)
        if np.dot(r1_vec, v1_vec) < 0:
            nu1 = -nu1
        omega = np.arctan2(e_vec[1], e_vec[0])
    else:
        nu1 = 0.0
        omega = np.arctan2(r1_vec[1], r1_vec[0])

    nus = np.linspace(nu1, nu1 + dnu, n_pts)
    rs = p / (1.0 + e * np.cos(nus))
    xp, yp = rs * np.cos(nus), rs * np.sin(nus)
    c, s = np.cos(omega), np.sin(omega)
    return xp * c - yp * s, xp * s + yp * c


def plot_trajectories_fixed_dep(t_dep_hr, n_arrivals=40, out_name=None):
    """Fix departure time and sweep arrival time evenly; color by total ΔV."""
    t_dep_s = t_dep_hr * 3600.0
    # Evenly-spaced arrival times over the TOF range used by the grid
    tofs_hr = np.linspace(TOF[0], TOF[-1], n_arrivals)
    arrivals_hr = t_dep_hr + tofs_hr

    r1, v_park = circ_state(R_PARK, THETA_PARK_0, N_PARK, t_dep_s)

    arcs = []  # (xs, ys, dv_total, r2)
    for tof_hr in tofs_hr:
        tof_s = tof_hr * 3600.0
        r2, v_tgt = circ_state(R_TGT, THETA_TGT_0, N_TGT, t_dep_s + tof_s)
        sol = lambert_uv(r1, r2, tof_s, MU, prograde=True)
        if sol is None:
            arcs.append(None)
            continue
        v_t1, v_t2 = sol
        dv_total = np.linalg.norm(v_t1 - v_park) + np.linalg.norm(v_tgt - v_t2)
        cos_dnu = np.clip(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)), -1.0, 1.0)
        dnu = np.arccos(cos_dnu)
        if np.cross(r1, r2)[2] < 0:
            dnu = 2 * np.pi - dnu
        xs, ys = transfer_arc(r1, v_t1, dnu, MU)
        arcs.append((xs, ys, dv_total, r2))

    dvs = np.array([a[2] if a is not None else np.nan for a in arcs])
    vmin = float(np.nanmin(dvs))
    vmax = vmin + 4.0
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 10))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_PARK * np.cos(th), R_PARK * np.sin(th), "k--", lw=0.7, alpha=0.5, label="Parking orbit")
    ax.plot(R_TGT * np.cos(th), R_TGT * np.sin(th), "k--", lw=0.7, alpha=0.5, label="Target orbit")
    ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#6fa8dc", alpha=0.6))

    # Draw high-ΔV arcs first so the optimum is on top
    order = np.argsort(-dvs)
    best_idx = int(np.nanargmin(dvs))
    for k in order:
        a = arcs[k]
        if a is None:
            continue
        xs, ys, dv, _ = a
        ax.plot(xs, ys, color=cmap(norm(dv)), lw=1.0, alpha=0.85)

    best = arcs[best_idx]
    xs, ys, dv_best, r2_best = best
    ax.plot(xs, ys, color="red", lw=2.0,
            label=f"Min ΔV = {dv_best:.3f} km/s  (arr={arrivals_hr[best_idx]:.1f} hr)")
    ax.plot([r1[0]], [r1[1]], "o", color="red", ms=7, label="Departure")
    ax.plot([r2_best[0]], [r2_best[1]], "*", color="red", ms=13, label="Arrival (min ΔV)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Total ΔV [km/s]")

    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(
        f"Trajectories: t_dep = {t_dep_hr:.2f} hr, "
        f"{n_arrivals} arrivals from {arrivals_hr[0]:.1f}–{arrivals_hr[-1]:.1f} hr"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    if out_name is None:
        out_name = f"trajectories_tdep_{t_dep_hr:.2f}hr.png"
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  (best ΔV = {dv_best:.4f} km/s)")


def plot_trajectories_multirev(t_dep_hr, n_arrivals=40, out_name=None,
                               max_n_rev=None, hard_cap=30):
    """Fix departure time and sweep arrival time evenly; include all reachable N-rev solutions.
    If max_n_rev is None, keeps going up in N until no arrival has any solution at that N.
    """
    t_dep_s = t_dep_hr * 3600.0
    tofs_hr = np.linspace(TOF[0], TOF[-1], n_arrivals)
    arrivals_hr = t_dep_hr + tofs_hr

    r1, v_park = circ_state(R_PARK, THETA_PARK_0, N_PARK, t_dep_s)

    # Precompute per-arrival r2, v_tgt, and the geometric transfer angle
    per_tof = []
    for tof_hr in tofs_hr:
        tof_s = tof_hr * 3600.0
        r2, v_tgt = circ_state(R_TGT, THETA_TGT_0, N_TGT, t_dep_s + tof_s)
        cos_dnu = np.clip(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)), -1.0, 1.0)
        dnu_geom = np.arccos(cos_dnu)
        if np.cross(r1, r2)[2] < 0:
            dnu_geom = 2 * np.pi - dnu_geom
        per_tof.append((tof_s, r2, v_tgt, dnu_geom))

    # Sweep N=0,1,2,... until a whole N pass produces no solutions
    arcs = []
    n_rev = 0
    while True:
        any_at_this_n = False
        branches = ["none"] if n_rev == 0 else ["short", "long"]
        for (tof_s, r2, v_tgt, dnu_geom) in per_tof:
            for branch in branches:
                sol = lambert_uv_multirev(r1, r2, tof_s, MU,
                                          prograde=True, n_rev=n_rev, branch=branch)
                if sol is None:
                    continue
                any_at_this_n = True
                v_t1, v_t2 = sol
                dv = np.linalg.norm(v_t1 - v_park) + np.linalg.norm(v_tgt - v_t2)
                total_angle = 2 * np.pi * n_rev + dnu_geom
                xs, ys = transfer_arc(r1, v_t1, total_angle, MU, n_pts=600)
                arcs.append(dict(xs=xs, ys=ys, dv=dv, r2=r2,
                                 n_rev=n_rev, branch=branch))
        if not any_at_this_n:
            break
        if max_n_rev is not None and n_rev >= max_n_rev:
            break
        n_rev += 1
        if n_rev > hard_cap:
            break
    n_rev_max = max((a["n_rev"] for a in arcs), default=0)

    dvs = np.array([a["dv"] for a in arcs])
    if len(dvs) == 0:
        print(f"No solutions for t_dep={t_dep_hr} hr")
        return
    vmin = float(np.min(dvs))
    vmax = vmin + 4.0
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 10))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_PARK * np.cos(th), R_PARK * np.sin(th), "k--", lw=0.7, alpha=0.5, label="Parking orbit")
    ax.plot(R_TGT * np.cos(th), R_TGT * np.sin(th), "k--", lw=0.7, alpha=0.5, label="Target orbit")
    ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#6fa8dc", alpha=0.6))

    # draw high-ΔV under low-ΔV
    order = np.argsort(-dvs)
    best_idx = int(np.argmin(dvs))
    # distinguish revs by line style — cycle a small set
    style_cycle = ["-", "--", ":", "-."]
    def style_for(n):
        return style_cycle[n % len(style_cycle)]
    for k in order:
        a = arcs[k]
        ax.plot(a["xs"], a["ys"],
                color=cmap(norm(a["dv"])),
                lw=1.0, alpha=0.85,
                linestyle=style_for(a["n_rev"]))

    best = arcs[best_idx]
    label_min = f"Min ΔV = {best['dv']:.3f} km/s  (N={best['n_rev']}"
    if best["n_rev"] > 0:
        label_min += f"-{best['branch']}"
    label_min += ")"
    ax.plot(best["xs"], best["ys"], color="red", lw=2.0, label=label_min)
    ax.plot([r1[0]], [r1[1]], "o", color="red", ms=7, label="Departure")
    ax.plot([best["r2"][0]], [best["r2"][1]], "*", color="red", ms=13)

    # legend entries for line styles (one per N actually present)
    from matplotlib.lines import Line2D
    ns_present = sorted({a["n_rev"] for a in arcs})
    style_handles = [
        Line2D([0], [0], color="gray", linestyle=style_for(n), label=f"N={n}")
        for n in ns_present
    ]

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Total ΔV [km/s]")

    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(
        f"Trajectories (N=0..{n_rev_max}): t_dep = {t_dep_hr:.2f} hr, "
        f"{n_arrivals} arrivals over {arrivals_hr[0]:.1f}–{arrivals_hr[-1]:.1f} hr\n"
        f"{len(arcs)} total Lambert solutions"
    )
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles + style_handles, [h.get_label() for h in handles + style_handles],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    if out_name is None:
        out_name = f"trajectories_multirev_tdep_{t_dep_hr:.2f}hr.png"
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  (best ΔV = {best['dv']:.4f} km/s, "
          f"N={best['n_rev']}{'-' + best['branch'] if best['n_rev'] > 0 else ''})")


def plot_porkchop_arrival(dv_total, out_name="porkchop_arrival.png",
                          title="Porkchop (arrival axes)",
                          dv_span=2.0):
    """Total ΔV contour in (departure time, arrival time) axes (N=0 only).
    Color range is clipped to [vmin, vmin + dv_span] km/s for porkchop sharpness."""
    t_arr_grid = np.linspace(T_DEP[0] + TOF[0], T_DEP[-1] + TOF[-1], 300)
    DEP, ARR = np.meshgrid(T_DEP, t_arr_grid)
    TOF_query = ARR - DEP

    Z = np.full_like(DEP, np.nan)
    for j in range(len(T_DEP)):
        col_tof = TOF_query[:, j]
        valid = (col_tof >= TOF[0]) & (col_tof <= TOF[-1])
        Z[valid, j] = np.interp(col_tof[valid], TOF, dv_total[:, j])

    vmin = float(np.nanmin(Z))
    vmax = vmin + dv_span
    # Mask cells above the cap so they render as the grey background instead of
    # saturating the colormap and drowning out the porkchop bowl.
    Z_clip = np.where(Z > vmax, np.nan, Z)
    levels = np.linspace(vmin, vmax, 40)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("lightgrey")
    cs = ax.contourf(DEP, ARR, Z_clip, levels=levels, cmap="viridis", extend="neither")
    ax.contour(DEP, ARR, Z_clip, levels=12, colors="k", linewidths=0.4, alpha=0.5)
    plt.colorbar(cs, ax=ax, label="Total ΔV [km/s]")
    ax.set_xlabel("Departure time [hr]")
    ax.set_ylabel("Arrival time [hr]")
    ax.set_title(
        f"{title}: circ {R_PARK:.0f} km → circ {R_TGT:.0f} km\n"
        f"min ΔV = {vmin:.3f} km/s"
    )
    fig.tight_layout()
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  (min ΔV = {vmin:.4f} km/s)")


def plot_porkchop_grid_raw(dv, out_name="porkchop_discretized_raw.png",
                           title="Discretized grid (raw cells, no interp)",
                           dv_span=2.0, vmin=None, vmax=None):
    """Raw grid: one colored square per (departure, time-of-flight) cell, NO
    interpolation. NaN cells (screened / non-converged) render grey. This is
    the faithful picture of exactly what the optimizer produced per cell.

    vmin/vmax pin the color scale (e.g. to a shared min/max across several
    grids). If left None they default to [nanmin(dv), nanmin(dv)+dv_span]."""
    dv = np.asarray(dv, dtype=np.float64)              # (n_TOF, n_DEP)
    data_min = float(np.nanmin(dv))
    vmin = data_min if vmin is None else float(vmin)
    vmax = (vmin + dv_span) if vmax is None else float(vmax)

    # Cell-edge arrays so each cell is a square (pcolormesh, flat shading).
    ddep = T_DEP[1] - T_DEP[0]
    dtof = TOF[1] - TOF[0]
    dep_edges = np.concatenate([T_DEP - 0.5 * ddep, [T_DEP[-1] + 0.5 * ddep]])
    tof_edges = np.concatenate([TOF - 0.5 * dtof, [TOF[-1] + 0.5 * dtof]])

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgrey")
    Zm = np.ma.masked_invalid(dv)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("lightgrey")
    pc = ax.pcolormesh(dep_edges, tof_edges, Zm, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="flat")
    plt.colorbar(pc, ax=ax, label="Total ΔV [km/s]", extend="max")
    n_ok = int(np.sum(np.isfinite(dv)))
    ax.set_xlabel("Departure time [hr]")
    ax.set_ylabel("Time of flight [hr]")
    ax.set_title(
        f"{title}: circ {R_PARK:.0f} km → circ {R_TGT:.0f} km\n"
        f"min ΔV = {data_min:.3f} km/s   ({n_ok} converged cells)"
    )
    fig.tight_layout()
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  (min ΔV = {data_min:.4f} km/s, {n_ok} cells)")


def _replay_failed_cell(args):
    """Propagate the Lambert-seed trajectory for a single failed cell.
    Module-level so multiprocessing.Pool can pickle it."""
    i, j, mu_moon = args
    t_d = float(T_DEP[j] * 3600.0)
    tof_s = float(TOF[i] * 3600.0)

    th_park = THETA_PARK_0 + N_PARK * t_d
    r1 = np.array([R_PARK * np.cos(th_park), R_PARK * np.sin(th_park), 0.0])
    r_moon, _ = _moon_state(t_d + tof_s)
    r_target = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / float(np.linalg.norm(r_moon)))

    ok, v1x, v1y, *_ = _lambert_njit(
        r1[0], r1[1], 0.0, r_target[0], r_target[1], 0.0,
        tof_s, MU, 1, 0, 0, 1e-8, 400,
    )
    if not ok:
        return None

    v = np.array([v1x, v1y, 0.0])
    t_eval = np.linspace(t_d, t_d + tof_s, 300)
    # Use a much looser tolerance and cap step size aggressively to avoid
    # the integrator grinding through close-approach singularities.
    sol = solve_ivp(_dynamics_em, (t_d, t_d + tof_s),
                    np.concatenate([r1, v]),
                    method="RK45", rtol=1e-3, atol=10.0,
                    t_eval=t_eval, args=(mu_moon,),
                    max_step=tof_s / 200.0,
                    first_step=tof_s / 200.0)
    if not sol.success or sol.y.shape[1] < 2:
        return None
    return {
        "x": sol.y[0],
        "y": sol.y[1],
        "r1": r1,
        "r_target": r_target,
        "r_moon": r_moon,
        "final_residual": float(np.linalg.norm(sol.y[:3, -1] - r_target)),
        "t_d_hr": float(T_DEP[j]),
        "tof_hr": float(TOF[i]),
    }


def plot_max_iter_failure_trajectories(status,
                                       out_name="failure_trajectories.png",
                                       n_workers=None,
                                       max_to_plot=1):
    """Plot up to `max_to_plot` max-iter failure trajectories on one 2D figure.
    Defaults to 1 for diagnostics — bump up once we know things don't hang."""
    fail_ij = np.argwhere(status == STATUS_MAX_ITER)
    if len(fail_ij) == 0:
        print("No max-iter failures to plot.")
        return

    fail_ij = fail_ij[:max_to_plot]
    n_fail = len(fail_ij)
    import time
    print(f"  replaying {n_fail} max-iter failures (cap={max_to_plot})...")
    for i, j in fail_ij:
        print(f"    cell (i={int(i)}, j={int(j)}): "
              f"tof={TOF[int(i)]:.2f} hr, t_dep={T_DEP[int(j)]:.3f} hr")

    # Run sequentially with progress prints — easier to see where things hang
    results = []
    for i, j in fail_ij:
        t0 = time.time()
        info = _replay_failed_cell((int(i), int(j), float(MU_MOON)))
        dt = time.time() - t0
        if info is None:
            print(f"    cell (i={int(i)}, j={int(j)}) → integrator failed in {dt:.2f}s")
        else:
            print(f"    cell (i={int(i)}, j={int(j)}) replayed in {dt:.2f}s, "
                  f"|residual|={info['final_residual']:.0f} km")
        results.append(info)

    fig, ax = plt.subplots(figsize=(10, 10))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_PARK * np.cos(th), R_PARK * np.sin(th), "k--", lw=0.7, alpha=0.5,
            label="Parking orbit")
    ax.plot(R_TGT * np.cos(th), R_TGT * np.sin(th), "k--", lw=0.7, alpha=0.5,
            label="Target orbit")
    ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#6fa8dc", alpha=0.6))

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=max(n_fail - 1, 1))

    moon_xs, moon_ys, tgt_xs, tgt_ys, final_residuals = [], [], [], [], []
    n_plotted = 0
    for k, info in enumerate(results):
        if info is None:
            continue
        ax.plot(info["x"], info["y"], color=cmap(norm(k)), lw=0.9, alpha=0.7)
        moon_xs.append(info["r_moon"][0]); moon_ys.append(info["r_moon"][1])
        tgt_xs.append(info["r_target"][0]); tgt_ys.append(info["r_target"][1])
        final_residuals.append(info["final_residual"])
        n_plotted += 1

    if moon_xs:
        ax.scatter(moon_xs, moon_ys, marker="o", s=18,
                   facecolor="white", edgecolor="black",
                   label="Moon at arrival", zorder=5)
        ax.scatter(tgt_xs, tgt_ys, marker="x", s=24, color="red",
                   label="Target (moon − offset)", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    median_res = float(np.median(final_residuals)) if final_residuals else 0.0
    ax.set_title(
        f"Max-iter failure trajectories  (n={n_plotted}/{n_fail}, "
        f"median final residual = {median_res:.0f} km)"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  (plotted {n_plotted}/{n_fail} failed trajectories)")


def _kepler_uv(r0_vec, v0_vec, dt, mu, tol=1e-10, max_iter=100):
    """Universal-variable Kepler propagation of a state by time dt.
    Returns (r_vec, v_vec). Valid for elliptic / parabolic / hyperbolic conics.
    Reuses the JIT Stumpff functions used by the Lambert solver."""
    if dt == 0.0:
        return r0_vec.copy(), v0_vec.copy()
    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    vr0 = np.dot(r0_vec, v0_vec) / r0
    alpha = 2.0 / r0 - v0 * v0 / mu          # = 1/a  (sign => conic type)
    sqrt_mu = np.sqrt(mu)

    if alpha > 1e-12:                         # ellipse
        chi = sqrt_mu * alpha * dt
    elif alpha < -1e-12:                      # hyperbola
        a = 1.0 / alpha
        chi = (np.sign(dt) * np.sqrt(-a) *
               np.log((-2.0 * mu * alpha * dt) /
                      (np.dot(r0_vec, v0_vec) + np.sign(dt) *
                       np.sqrt(-mu * a) * (1.0 - r0 * alpha))))
    else:                                     # near-parabolic
        chi = sqrt_mu * dt / r0

    for _ in range(max_iter):
        z = alpha * chi * chi
        C = _stumpff_C(z)
        S = _stumpff_S(z)
        F = (r0 * vr0 / sqrt_mu * chi * chi * C +
             (1.0 - alpha * r0) * chi ** 3 * S +
             r0 * chi - sqrt_mu * dt)
        dF = (r0 * vr0 / sqrt_mu * chi * (1.0 - alpha * chi * chi * S) +
              (1.0 - alpha * r0) * chi * chi * C + r0)
        dchi = F / dF
        chi -= dchi
        if abs(dchi) < tol:
            break

    z = alpha * chi * chi
    C = _stumpff_C(z)
    S = _stumpff_S(z)
    f = 1.0 - chi * chi / r0 * C
    g = dt - chi ** 3 / sqrt_mu * S
    r_vec = f * r0_vec + g * v0_vec
    r = np.linalg.norm(r_vec)
    gdot = 1.0 - chi * chi / r * C
    fdot = sqrt_mu / (r * r0) * (alpha * chi ** 3 * S - chi)
    v_vec = fdot * r0_vec + gdot * v0_vec
    return r_vec, v_vec


def _march_conic_sundman(r1, v1, mu, delta_tau, tof):
    """ONE Sundman march from (r1, v1) to tof (NO period bisection):
        Δt_k = (|r_init| + |r_fin|)^(3/2) / √μ · Δτ   (implicit -> inner FP)
    Nodes are Kepler-propagated from (r1, v1) by cumulative time; the final
    segment is truncated to land exactly at tof. ~n_seg·(few) Kepler calls,
    so ~15 ms — cheap (the old cost was the bisection's ~70 marches).
    Returns (node_r, node_v, seg_dt)."""
    sqrt_mu = np.sqrt(mu)
    node_r = [r1.copy()]
    node_v = [v1.copy()]
    seg_dt = []
    t_acc = 0.0
    r_init_mag = np.linalg.norm(r1)
    eps = 1e-9 * tof
    while t_acc < tof - eps and len(seg_dt) < 100_000:
        rstar = 2.0 * r_init_mag                       # predictor
        dt = rstar ** 1.5 / sqrt_mu * delta_tau
        rk, vk = _kepler_uv(r1, v1, t_acc + dt, mu)
        for _ in range(30):                            # implicit rstar
            rstar = r_init_mag + np.linalg.norm(rk)
            dt_new = rstar ** 1.5 / sqrt_mu * delta_tau
            if abs(dt_new - dt) <= 1e-12 * dt:
                dt = dt_new
                break
            dt = dt_new
            rk, vk = _kepler_uv(r1, v1, t_acc + dt, mu)
        if t_acc + dt >= tof:                          # final partial segment
            dt = tof - t_acc
            rk, vk = _kepler_uv(r1, v1, tof, mu)
        t_acc += dt
        node_r.append(rk)
        node_v.append(vk)
        seg_dt.append(dt)
        r_init_mag = np.linalg.norm(rk)
    # Merge a degenerate near-zero final segment (truncation sliver) into the
    # previous one. Such a Δt≈0 / Δθ≈0 leg is singular for Lambert and
    # corrupts the optimizer warm start (huge chord/Δt velocity) -> spurious
    # high-ΔV basin. Drop the second-to-last node so the final leg is whole.
    if len(seg_dt) >= 2 and seg_dt[-1] < 0.1 * seg_dt[-2]:
        seg_dt[-2] += seg_dt[-1]
        seg_dt.pop()
        node_r.pop(-2)
        node_v.pop(-2)
    return node_r, node_v, seg_dt


def _sundman_nodes(r1, v1, mu, tof, n_seg, alpha=1.5):
    """Place n_seg+1 nodes by integrating the two-body problem in the Sundman
    independent variable τ.  7-state y = [r(3), v(3), t]:

        dr/dτ = v · r^α,   dv/dτ = (−μ r/|r|³) · r^α,   dt/dτ = r^α

    Shoot on total τ: integrate until the time state reaches tof (terminal
    event), giving τ_f. Then Δτ = τ_f/n_seg and the nodes are the integrated
    states sampled at k·Δτ (uniform in τ ⇒ Sundman spacing, dense near
    periapsis). Returns (Δτ, node_r, node_v, seg_dt)."""
    def rhs(tau, y):
        px, py, pz, vx, vy, vz, _t = y
        r = np.sqrt(px * px + py * py + pz * pz)
        s = r ** alpha
        ar = -mu / r ** 3 * s
        return [vx * s, vy * s, vz * s, ar * px, ar * py, ar * pz, s]

    def hit_tof(tau, y):
        return y[6] - tof
    hit_tof.terminal = True
    hit_tof.direction = 1.0

    y0 = [r1[0], r1[1], r1[2], v1[0], v1[1], v1[2], 0.0]
    tau_max = 10.0 * tof / (np.linalg.norm(r1) ** alpha)   # safe upper bound
    sol = solve_ivp(rhs, (0.0, tau_max), y0, method="DOP853",
                    rtol=1e-10, atol=1e-3, events=hit_tof, dense_output=True)
    tau_f = sol.t_events[0][0]
    dtau = tau_f / n_seg
    node_r, node_v = [], []
    for k in range(n_seg + 1):
        yk = sol.sol(k * dtau)
        node_r.append(np.array([yk[0], yk[1], yk[2]]))
        node_v.append(np.array([yk[3], yk[4], yk[5]]))
    seg_dt = [float(sol.sol((k + 1) * dtau)[6] - sol.sol(k * dtau)[6])
              for k in range(n_seg)]
    return dtau, node_r, node_v, seg_dt


def discretize_conic_segments(r1, v1, tof, mu, segs_per_rev=24):
    """Break the conic into N Sundman segments with NO truncated last segment:
    find Δτ so N segments (Δt=rstar^1.5/√μ·Δτ) span exactly tof, landing the
    final node on conic(tof)=r_target. The nodes are then Sundman-consistent
    with the optimizer's own Δt rule, so the warm-start net ΔV ≈ Σ|Δv_moon|."""
    r1n = np.linalg.norm(r1)
    v1n = np.linalg.norm(v1)
    inv_a = 2.0 / r1n - v1n * v1n / mu        # = 1/a (vis-viva)
    elliptic = inv_a > 1e-12
    T = 2.0 * np.pi * np.sqrt((1.0 / inv_a) ** 3 / mu) if elliptic else np.nan

    n_seg = int(round(segs_per_rev * tof / T)) if elliptic else segs_per_rev
    n_seg = max(4, n_seg)
    delta_tau, node_r, node_v_true, seg_dt = _sundman_nodes(
        r1, v1, mu, tof, n_seg)

    seg_v_out, seg_v_in = [], []
    for k in range(n_seg):
        sol = lambert_uv(node_r[k], node_r[k + 1], seg_dt[k], mu, prograde=True)
        seg_v_out.append(sol[0] if sol else None)
        seg_v_in.append(sol[1] if sol else None)

    node_dv = []
    for k in range(1, n_seg):
        if seg_v_in[k - 1] is None or seg_v_out[k] is None:
            node_dv.append(np.nan)
        else:
            node_dv.append(float(np.linalg.norm(seg_v_out[k] - seg_v_in[k - 1])))
    node_dv = np.array(node_dv)

    return {
        "n_seg": n_seg, "T": T, "seg_dt": seg_dt,
        "delta_tau": delta_tau, "dtau_resid": 0.0,
        "node_r": node_r, "node_v_true": node_v_true,
        "seg_v_out": seg_v_out, "seg_v_in": seg_v_in,
        "node_dv": node_dv,
    }


def moon_gravity_node_delta_v(node_r, seg_dt, t_dep_s, mu_moon):
    """Lumped Moon-gravity Δv at every node, using a ~= Δv/Δt:

        Δv_k = a_moon(r_k, t_k) · Δt*_k

    where Δt*_k is half the previous segment's flight time plus half the next
    segment's (one-sided at the two endpoints). a_moon is the *direct* lunar
    term  mu_moon·(r_moon - r_sc)/|r_moon - r_sc|³, matching the Moon term in
    _dynamics_em (_moon_state gives the Moon position at absolute time t).
    In this simplified scenario the Moon sits at R_TGT — the same circle the
    conic arc arrives on — so the arrival node coincides with the Moon and the
    point-mass term diverges. Nodes within MOON_ARRIVAL_OFFSET (≈lunar radius,
    the model's "surface") are returned as NaN: the 1/r² model is meaningless
    there.

    Returns (dv [n_node,3] km/s, t_nodes [n_node] s absolute, min_sep km)."""
    n_node = len(node_r)
    n_seg = len(seg_dt)
    t_nodes = t_dep_s + np.concatenate([[0.0], np.cumsum(seg_dt)])
    dv = np.full((n_node, 3), np.nan)
    min_sep = np.inf
    for k in range(n_node):
        prev_dt = seg_dt[k - 1] if k - 1 >= 0 else 0.0
        next_dt = seg_dt[k] if k < n_seg else 0.0
        dt_star = 0.5 * prev_dt + 0.5 * next_dt
        r_moon, _ = _moon_state(t_nodes[k])
        d = r_moon - node_r[k]
        dn = np.linalg.norm(d)
        min_sep = min(min_sep, dn)
        if dn < MOON_ARRIVAL_OFFSET:          # inside the Moon: model invalid
            continue
        a_moon = mu_moon * d / dn ** 3
        dv[k] = a_moon * dt_star
    return dv, t_nodes, min_sep


def plot_evolved_solutions(cells, segs_per_rev=24,
                           out_name="evolved_solutions.png"):
    """For each (t_dep_hr, t_arr_hr) grid cell: run the exact grid path
    (conic seed -> discretize -> net-Δv min, target = Moon offset) and plot
    the conic warm start vs the evolved (final) trajectory, with the same
    ΔV the grid records plus the diagnostics that expose bad cells (min
    node-to-Moon separation, dep/arr burns, Σ|Δv_net|, n_fail, ok)."""
    n = len(cells)
    nc = min(2, n)
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(7.5 * nc, 7.0 * nr),
                             squeeze=False)
    th = np.linspace(0, 2 * np.pi, 360)
    for idx, (t_dep_hr, t_arr_hr) in enumerate(cells):
        ax = axes[idx // nc][idx % nc]
        t_d = t_dep_hr * 3600.0
        tof_s = (t_arr_hr - t_dep_hr) * 3600.0
        r1, v_park = circ_state(R_PARK, THETA_PARK_0, N_PARK, t_d)
        r_moon, v_moon = _moon_state(t_d + tof_s)
        r_tgt = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / np.linalg.norm(r_moon))
        ok, v1x, v1y, _z, _x, _y, _z2 = _lambert_njit(
            r1[0], r1[1], 0.0, r_tgt[0], r_tgt[1], 0.0,
            tof_s, MU, 1, 0, 0, 1e-8, 400)
        if not ok:
            ax.set_title(f"t_dep={t_dep_hr:.2f},t_arr={t_arr_hr:.2f}: "
                         "conic Lambert FAIL")
            continue
        v1 = np.array([v1x, v1y, 0.0])
        d = discretize_conic_segments(r1, v1, tof_s, MU, segs_per_rev)
        opt = optimize_net_delta_v(r1, r_tgt, d["node_r"], d["seg_dt"],
                                   t_d, MU, MU_MOON)
        nrf = opt["node_r"]
        sdf = opt["seg_dt"]
        # ΔV exactly as the grid records it
        s0 = lambert_uv(nrf[0], nrf[1], sdf[0], MU, prograde=True)
        sN = lambert_uv(nrf[-2], nrf[-1], sdf[-1], MU, prograde=True)
        dv_dep = np.linalg.norm(s0[0] - v_park) if s0 else np.nan
        dv_arr = np.linalg.norm(v_moon - sN[1]) if sN else np.nan
        dv_tot = dv_dep + dv_arr + opt["sum_final"]
        # min node-to-Moon separation along the evolved path
        tn = t_d + np.concatenate([[0.0], np.cumsum(sdf)])
        seps = [np.linalg.norm(_moon_state(tn[k])[0] - np.asarray(nrf[k]))
                for k in range(len(nrf))]
        min_sep = min(seps)

        # reference circles / bodies
        ax.plot(R_PARK * np.cos(th), R_PARK * np.sin(th), "k--",
                lw=0.5, alpha=0.4)
        ax.plot(R_TGT * np.cos(th), R_TGT * np.sin(th), "k--",
                lw=0.5, alpha=0.4)
        ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#6fa8dc", alpha=0.5))
        # Moon keep-out ring at the arrival node
        ax.add_patch(plt.Circle((r_tgt[0], r_tgt[1]), MOON_ARRIVAL_OFFSET,
                                color="grey", alpha=0.25))
        # conic warm start (blue) vs evolved (red)
        cs, _, _ = _piecewise_lambert_xy(d["node_r"], d["seg_dt"], MU)
        for leg in cs:
            if leg is not None:
                ax.plot(leg[:, 0], leg[:, 1], color="#1f77b4",
                        lw=1.0, alpha=0.5)
        es, _, _ = _piecewise_lambert_xy(nrf, sdf, MU)
        for leg in es:
            if leg is not None:
                ax.plot(leg[:, 0], leg[:, 1], color="#d62728", lw=1.6)
        Pf = np.array(nrf)
        ax.plot(Pf[:, 0], Pf[:, 1], "o", color="#2ca02c", ms=3.5)
        ax.plot([r1[0]], [r1[1]], "ko", ms=6)
        ax.plot([r_tgt[0]], [r_tgt[1]], "k*", ms=13)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        flag = "" if opt["ok"] else "  [NOT-CONV]"
        ax.set_title(
            f"t_dep={t_dep_hr:.2f}, t_arr={t_arr_hr:.2f} hr  "
            f"ΔV={dv_tot:.3f} km/s{flag}\n"
            f"dep={dv_dep:.3f}  arr={dv_arr:.3f}  "
            f"Σ|Δv_net|={opt['sum_final']*1000:.2f} m/s  "
            f"nf={opt['n_fail']}  min Moon sep={min_sep:.0f} km",
            fontsize=9)
        print(f"  ({t_dep_hr:.2f},{t_arr_hr:.2f}) dv={dv_tot:.4f} "
              f"dep={dv_dep:.4f} arr={dv_arr:.4f} "
              f"sumfinal={opt['sum_final']*1000:.3f}m/s nf={opt['n_fail']} "
              f"ok={opt['ok']} min_sep={min_sep:.0f}km n_seg={d['n_seg']}")
    for k in range(n, nr * nc):
        axes[k // nc][k % nc].axis("off")
    fig.suptitle("Evolved discretized solutions at selected grid cells "
                 "(blue = conic warm start, red = evolved; grey ring = "
                 "Moon keep-out)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}")


def plot_solution_evolution(place=(0.5, 130.0), segs_per_rev=24,
                            out_name="solution_evolution.png"):
    """ONE figure: the discretized trajectory for ONE solution at every BFGS
    iteration (one subplot per iteration). Shows how the net-Δv-minimizing
    optimizer bends the conic warm start into the Moon-perturbed arc."""
    t_dep_hr, t_arr_hr = place
    t_d = t_dep_hr * 3600.0
    tof_s = (t_arr_hr - t_dep_hr) * 3600.0
    r1, v_park = circ_state(R_PARK, THETA_PARK_0, N_PARK, t_d)
    r_moon, v_moon = _moon_state(t_d + tof_s)
    r_tgt = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / np.linalg.norm(r_moon))

    ok, v1x, v1y, _z1, _vx, _vy, _z2 = _lambert_njit(
        r1[0], r1[1], 0.0, r_tgt[0], r_tgt[1], 0.0,
        tof_s, MU, 1, 0, 0, 1e-8, 400)
    v1 = np.array([v1x, v1y, 0.0])
    d = discretize_conic_segments(r1, v1, tof_s, MU, segs_per_rev)
    opt = optimize_net_delta_v(r1, r_tgt, d["node_r"], d["seg_dt"],
                               t_d, MU, MU_MOON, record_iters=True)
    its = opt["iters"]
    n = len(its)
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.2 * nr),
                             squeeze=False)
    th = np.linspace(0, 2 * np.pi, 240)
    # common extent from the initial guess so all panels share a frame
    all0 = np.array(its[0]["node_r"])
    span = 1.15 * np.max(np.abs(all0[:, :2]))
    for idx in range(nr * nc):
        ax = axes[idx // nc][idx % nc]
        if idx >= n:
            ax.axis("off")
            continue
        it = its[idx]
        ax.plot(R_PARK * np.cos(th), R_PARK * np.sin(th), "k--",
                lw=0.4, alpha=0.4)
        ax.plot(R_TGT * np.cos(th), R_TGT * np.sin(th), "k--",
                lw=0.4, alpha=0.4)
        ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#6fa8dc", alpha=0.5))
        segs, _, _ = _piecewise_lambert_xy(it["node_r"], it["seg_dt"], MU)
        for leg in segs:
            if leg is not None:
                ax.plot(leg[:, 0], leg[:, 1], color="#d62728", lw=1.0)
        P = np.array(it["node_r"])
        ax.plot(P[:, 0], P[:, 1], "o", color="#2ca02c", ms=2.5)
        ax.plot([r1[0]], [r1[1]], "ko", ms=4)
        ax.plot([r_tgt[0]], [r_tgt[1]], "k*", ms=9)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        tag = "init" if idx == 0 else f"it {idx}"
        ax.set_title(f"{tag}: Σ|Δv_net|={it['sum'] * 1000:.1f} m/s"
                     + (f"  nf={it['n_fail']}" if it["n_fail"] else ""),
                     fontsize=8)
    fig.suptitle(
        f"Net-Δv minimization, one solution  t_dep={t_dep_hr:.2f} hr, "
        f"t_arr={t_arr_hr:.2f} hr  ({d['n_seg']} segs, {n - 1} BFGS iters, "
        f"Σ|Δv_net|: {opt['sum_init'] * 1000:.0f}→"
        f"{opt['sum_final'] * 1000:.3f} m/s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  ({n} iterations, "
          f"Σ|Δv_net| {opt['sum_init'] * 1000:.1f} -> "
          f"{opt['sum_final'] * 1000:.4f} m/s, ok={opt['ok']})")


def _piecewise_lambert_xy(node_r, seg_dt, mu, n_pts=30):
    """Sample the piecewise trajectory: per segment solve Lambert between the
    given nodes, then propagate that leg analytically. Returns a list of
    (xs, ys) arrays (one per segment) plus the per-segment v_out / v_in."""
    segs, vouts, vins = [], [], []
    for k in range(len(seg_dt)):
        sol = lambert_uv(node_r[k], node_r[k + 1], seg_dt[k], mu,
                         prograde=True)
        if sol is None:
            segs.append(None)
            vouts.append(None)
            vins.append(None)
            continue
        vo, vi = sol
        ss = np.linspace(0.0, seg_dt[k], n_pts)
        leg = np.array([_kepler_uv(node_r[k], vo, s, mu)[0] for s in ss])
        segs.append(leg)
        vouts.append(vo)
        vins.append(vi)
    return segs, vouts, vins


def optimize_net_delta_v(r1, r2, node_r0, seg_dt, t_dep_s, mu, mu_moon,
                         record_iters=False, v_dep_ref=None, v_arr_ref=None,
                         burn_weight=0.0, moon_ecc=0.0):
    """Joint BFGS solve over the decision vector  x = [Δτ, interior nodes].

    Segment time is explicit from the decision vector (no Δτ bisection / no
    Kepler marching):

        Δt_k = (|r_k| + |r_{k+1}|)^(3/2) / √μ · Δτ

    so rstar is never implicit. Δτ is otherwise free; the flight-time closure
    is added as a penalty. Objective minimized by BFGS:

        J = Σ_k ‖Δv_net_k‖²  +  W·((Σ Δt_k − tof)/tof)²
        Δv_net_k = (v_out_k − v_in_k) − a_moon(r_k, t_k)·Δt*_k

    Endpoints r1, r2 and the flight time tof = Σ seg_dt are fixed; interior
    positions are nondimensionalized by L so the vector is well scaled for
    BFGS (Δτ ≈ 2π/segs with the 1/√μ factor). Returns the optimized
    nodes / seg_dt / Δτ and init+final Δv_net diagnostics (same dict keys as
    before, plus 'seg_dt', 'delta_tau', 'time_resid')."""
    n_seg = len(seg_dt)
    n_int = n_seg - 1
    W = 1.0e4                                   # flight-time-closure weight
    FAIL_PEN = 1.0                              # canonical Δv per bad Lambert

    # ---- Canonical units: LU = R_TGT, TU = sqrt(LU^3/mu_SI)  =>  mu = 1.
    # All optimizer-internal math is O(1)-scaled (the SI km/s problem is
    # pathologically conditioned for BFGS: positions ~1e5, Δτ ~0.2). Inputs
    # are converted in here; outputs are converted back to SI on return.
    mu_SI = float(mu)
    NT_SI = globals()["N_TGT"]                  # module globals (N_TGT/R_TGT
    LU = float(globals()["R_TGT"])              # are shadowed as locals below)
    TU = np.sqrt(LU ** 3 / mu_SI)
    VU = LU / TU
    seg_dt_c = np.asarray(seg_dt, dtype=np.float64) / TU
    node0 = np.asarray(node_r0, dtype=np.float64)[:, :2] / LU
    r1xy = np.asarray(r1[:2], dtype=np.float64) / LU
    r2xy = np.asarray(r2[:2], dtype=np.float64) / LU

    # Canonical shadows of the names the closures use (mu = 1, sqrt_mu = 1).
    mu = 1.0
    sqrt_mu = 1.0
    mu_moon = mu_moon / mu_SI
    t_dep_s = t_dep_s / TU
    R_TGT = LU / LU                                        # = 1.0
    N_TGT = NT_SI * TU
    OFF = MOON_ARRIVAL_OFFSET / LU
    tof = float(seg_dt_c.sum())
    L = 1.0                                                # positions already O(1)

    # Step rule:  Δt_k = rstar_k^1.5 · Δτ  (mu=1 canonical, no mult).
    # "Find Δτ that satisfies the flight time" is closed-form for fixed nodes:
    #   tof = Σ Δt_k = Δτ · Σ rstar_k^1.5   =>   Δτ0 = tof / Σ rstar_k^1.5.
    # Δτ then stays a free decision variable so it keeps satisfying the time
    # term as the interior nodes move during the joint minimization.
    rmag0 = np.linalg.norm(node0, axis=1)
    rstar0_seg = rmag0[:-1] + rmag0[1:]                     # (n_seg,)
    dtau0 = tof / float(np.sum(rstar0_seg ** 1.5))

    x0 = np.empty(1 + 2 * n_int)
    x0[0] = dtau0
    x0[1::2] = node0[1:n_seg, 0] / L
    x0[2::2] = node0[1:n_seg, 1] / L

    z_cache = [0.0] * n_seg

    def _pack(x):
        """Per-segment Lambert + analytic Jacobians at decision vector x."""
        dtau = x[0]
        P = np.zeros((n_seg + 1, 2))
        P[0] = r1xy
        P[-1] = r2xy
        P[1:n_seg, 0] = x[1::2] * L
        P[1:n_seg, 1] = x[2::2] * L
        rmag = np.linalg.norm(P, axis=1)
        rstar = rmag[:-1] + rmag[1:]                       # (n_seg,)
        coef = rstar ** 1.5 / sqrt_mu                     # Δt = coef·Δτ
        Dt = coef * dtau
        f = 1.5 * Dt / rstar
        gA = f[:, None] * (P[:-1] / rmag[:-1, None])       # ∂Δt_k/∂P_k
        gB = f[:, None] * (P[1:] / rmag[1:, None])         # ∂Δt_k/∂P_{k+1}
        t_node = t_dep_s + np.concatenate([[0.0], np.cumsum(Dt)])
        v1 = np.zeros((n_seg, 2))
        v2 = np.zeros((n_seg, 2))
        J1r1 = np.zeros((n_seg, 2, 2))
        J1r2 = np.zeros((n_seg, 2, 2))
        J1dt = np.zeros((n_seg, 2))
        J2r1 = np.zeros((n_seg, 2, 2))
        J2r2 = np.zeros((n_seg, 2, 2))
        J2dt = np.zeros((n_seg, 2))
        okk = np.ones(n_seg, dtype=bool)
        I2 = np.eye(2)
        for k in range(n_seg):
            dtk = Dt[k]
            o = (lambert_with_jac_nb(P[k], P[k + 1], dtk, mu, z_cache[k])
                 if dtk > 0.0 else (0,))
            if dtk > 0.0 and o[0] != 0:
                v1[k], v2[k] = o[1], o[2]
                J1r1[k], J1r2[k], J1dt[k] = o[3], o[4], o[5]
                J2r1[k], J2r2[k], J2dt[k] = o[6], o[7], o[8]
                z_cache[k] = o[9]
            else:
                # Straight-line constant-velocity fallback for a degenerate
                # sub-arc, with a CONSISTENT Jacobian, so the gradient is
                # always defined and BFGS can step away from the degeneracy
                # (instead of dying at nit=0 on a FAIL_PEN flat spot — this
                # matches handcrafted_trajectory_design). okk[k]=False marks
                # that this segment is not a true Lambert arc.
                okk[k] = False
                dts = dtk if dtk > 1e-9 else 1e-9
                chord = P[k + 1] - P[k]
                v1[k] = v2[k] = chord / dts
                J1r1[k] = J2r1[k] = -I2 / dts
                J1r2[k] = J2r2[k] = I2 / dts
                J1dt[k] = J2dt[k] = -chord / (dts * dts)
        return (dtau, P, rmag, rstar, coef, Dt, gA, gB, t_node,
                v1, v2, J1r1, J1r2, J1dt, J2r1, J2r2, J2dt, okk)

    def _node_terms(pk):
        """Per interior node k (i = k-1): (dvnet, S=2·dvnet, a, M, vmoon,
        Dtstar, w=∂dvnet/∂t_node[k]) or None if a touching leg failed."""
        (dtau, P, rmag, rstar, coef, Dt, gA, gB, t_node,
         v1, v2, J1r1, J1r2, J1dt, J2r1, J2r2, J2dt, okk) = pk
        out = []
        for k in range(1, n_seg):
            # v1/v2/J always populated now (real Lambert or straight-line
            # fallback), so every node yields a differentiable term.
            Dtstar = 0.5 * (Dt[k - 1] + Dt[k])
            rm, vm = _moon_rv(THETA_TGT_0 + N_TGT * t_node[k],
                              R_TGT, N_TGT, moon_ecc)
            dvec = rm - P[k]
            raw = np.linalg.norm(dvec)
            dist = max(raw, OFF)
            a = mu_moon * dvec / dist ** 3
            dvnet = (v1[k] - v2[k - 1]) - a * Dtstar
            if raw >= OFF:
                M = mu_moon * (np.eye(2) / dist ** 3 -
                               3.0 * np.outer(dvec, dvec) / dist ** 5)
            else:                                         # clamp: a frozen
                M = mu_moon * np.eye(2) / dist ** 3
            w = -(M @ vm) * Dtstar          # ∂dvnet/∂t_node[k]
            out.append((dvnet, 2.0 * dvnet, a, M, vm, Dtstar, w))
        return out

    # ---- Levenberg–Marquardt / least-squares form (as in the reference):
    # residual r = [ Δv_net_k components (canonical) , √W·tr ] so that
    # Σ r² = Σ‖Δv_net‖² + W·tr² = J. Gauss-Newton/LM exploits this
    # sum-of-squares structure and is far more robust than BFGS on the
    # degenerate-sub-arc non-convexity.
    n_var = 1 + 2 * n_int
    sqrtW = np.sqrt(W)
    # Boundary-burn residuals: minimize departure |v_dep - v_park| and arrival
    # |v_arr - v_moon| too (total-Δv "energy" objective, as in the reference).
    # This selects the min-Δv member among the otherwise-arbitrary net-zero
    # solutions, so the porkchop varies smoothly cell-to-cell.
    has_burn = (v_dep_ref is not None and v_arr_ref is not None
                and burn_weight > 0.0)
    wb = float(burn_weight)
    if has_burn:
        vdr = np.asarray(v_dep_ref[:2], dtype=np.float64) / VU
        var = np.asarray(v_arr_ref[:2], dtype=np.float64) / VU
    n_res = 2 * n_int + 1 + (4 if has_burn else 0)
    iT = 2 * n_int                                # time-closure residual index

    def _resid(x):
        pk = _pack(x)
        nt = _node_terms(pk)
        Dt = pk[5]
        r = np.empty(n_res)
        for i, t in enumerate(nt):
            r[2 * i] = t[0][0]
            r[2 * i + 1] = t[0][1]
        r[iT] = sqrtW * (float(np.sum(Dt)) - tof) / tof
        if has_burn:
            v1, v2 = pk[9], pk[10]
            r[iT + 1:iT + 3] = wb * (v1[0] - vdr)
            r[iT + 3:iT + 5] = wb * (v2[-1] - var)
        return r

    def _jac(x):
        pk = _pack(x)
        (dtau, P, rmag, rstar, coef, Dt, gA, gB, t_node,
         v1, v2, J1r1, J1r2, J1dt, J2r1, J2r2, J2dt, okk) = pk
        nt = _node_terms(pk)
        Jm = np.zeros((n_res, n_var))
        cT = sqrtW / tof
        ns1 = n_seg - 1
        for iv in range(n_var):
            dP = np.zeros((n_seg + 1, 2))
            dDt = np.zeros(n_seg)
            if iv == 0:                                    # Δτ
                dDt[:] = coef
            else:
                node = (iv - 1) // 2 + 1                    # interior node
                ax = (iv - 1) % 2
                dP[node, ax] = L
                dDt[node - 1] = gB[node - 1, ax] * L
                dDt[node] = gA[node, ax] * L
            dtnode = np.concatenate([[0.0], np.cumsum(dDt)])
            Jm[iT, iv] = cT * dDt.sum()
            for i, k in enumerate(range(1, n_seg)):
                dvnet, S, a, M, vm, Dtstar, w = nt[i]
                dv1 = J1r1[k] @ dP[k] + J1r2[k] @ dP[k + 1] + J1dt[k] * dDt[k]
                dv2 = (J2r1[k - 1] @ dP[k - 1] + J2r2[k - 1] @ dP[k] +
                       J2dt[k - 1] * dDt[k - 1])
                dconic = dv1 - dv2
                dDtstar = 0.5 * (dDt[k - 1] + dDt[k])
                ddvec = vm * dtnode[k] - dP[k]
                ddmoon = (M @ ddvec) * Dtstar + a * dDtstar
                dnet = dconic - ddmoon
                Jm[2 * i, iv] = dnet[0]
                Jm[2 * i + 1, iv] = dnet[1]
            if has_burn:
                # ∂v_dep/∂x = ∂v1[0]; node0 fixed so only via seg-0 endpoint1+Δt
                dvdep = J1r2[0] @ dP[1] + J1dt[0] * dDt[0]
                # ∂v_arr/∂x = ∂v2[-1]; node n fixed so via seg-(n-1) start+Δt
                dvarr = (J2r1[ns1] @ dP[ns1] + J2dt[ns1] * dDt[ns1])
                Jm[iT + 1, iv] = wb * dvdep[0]
                Jm[iT + 2, iv] = wb * dvdep[1]
                Jm[iT + 3, iv] = wb * dvarr[0]
                Jm[iT + 4, iv] = wb * dvarr[1]
        return Jm

    def _vectors(x):
        pk = _pack(x)
        P = pk[1]
        Dt = pk[5]
        okk = pk[-1]
        nt = _node_terms(pk)
        rn = np.full((n_int, 3), np.nan)
        dvm = np.full((n_int, 3), np.nan)
        dvn = np.full((n_int, 3), np.nan)
        nf = 0
        for i, k in enumerate(range(1, n_seg)):
            rn[i] = (P[k, 0], P[k, 1], 0.0)               # full row finite
            dvnet, S, a, M, vm, Dtstar, w = nt[i]
            dvn[i] = (dvnet[0], dvnet[1], 0.0)
            avm = a * Dtstar
            dvm[i] = (avm[0], avm[1], 0.0)
            if not (okk[k] and okk[k - 1]):               # straight-line seg
                nf += 1                                   # -> not a true arc
        node_r = [np.array([P[k, 0], P[k, 1], 0.0])
                  for k in range(n_seg + 1)]
        tr = (float(np.sum(Dt)) - tof) / tof
        return node_r, list(Dt), float(pk[0]), rn, dvm, dvn, nf, tr

    nr0c, _, _, rn0c, dvm0c, dvn0c, _, _ = _vectors(x0)

    iters = []                                  # per-iteration snapshots (SI)

    def _snap(xk):
        nrc, dtc, _, _, _, dvc, nfk, trk = _vectors(np.asarray(xk))
        iters.append({
            "node_r": [np.array([p[0] * LU, p[1] * LU, 0.0]) for p in nrc],
            "seg_dt": [dd * TU for dd in dtc],
            "sum": float(np.nansum(np.linalg.norm(dvc * VU, axis=1))),
            "n_fail": int(nfk), "tres": float(trk),
        })

    if record_iters:
        _snap(x0)                               # frame 0 = initial guess
    # Levenberg–Marquardt (analytic Jacobian), the reference's solver for
    # this sum-of-squared-residuals transcription.
    sol = least_squares(_resid, x0, jac=_jac, method="lm",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
    if record_iters:
        _snap(sol.x)                            # frame 1 = converged
    nrfc, dtfc, dtauf, rnfc, dvmfc, dvnfc, nff, trf = _vectors(sol.x)

    # ---- Convert canonical -> SI for all returned quantities.
    pconv = np.array([LU, LU, 1.0])
    node_r = [np.array([p[0] * LU, p[1] * LU, 0.0]) for p in nrfc]
    seg_dt_out = [d * TU for d in dtfc]
    rn0, dvm0, dvn0 = rn0c * pconv, dvm0c * VU, dvn0c * VU
    rnf, dvmf, dvnf = rnfc * pconv, dvmfc * VU, dvnfc * VU

    # Departure/arrival velocities straight from the optimizer's own solution
    # (first-leg v_out, last-leg v_in) so callers don't re-solve the legs with
    # a different Lambert solver — that disagreement was dropping converged
    # cells as spurious lambert_fail holes.
    pkf = _pack(sol.x)
    v_dep = np.array([pkf[9][0, 0], pkf[9][0, 1], 0.0]) * VU      # v1[0]
    v_arr = np.array([pkf[10][-1, 0], pkf[10][-1, 1], 0.0]) * VU  # v2[-1]

    mag_init = np.linalg.norm(dvn0, axis=1)
    mag_fin = np.linalg.norm(dvnf, axis=1)
    sum_final = float(np.nansum(mag_fin))
    # "Converged" = a ballistic arc was found: net Δv driven to ~0 and the
    # flight time closed. We do NOT require n_fail==0 — a single straight-line
    # fallback on a sub-arc that crosses the 180° Lambert singularity still
    # yields a valid net-zero trajectory (garbage cells fail sum_final<5e-3
    # anyway, with leftover net Δv of 0.1–30 km/s).
    ok = bool(sum_final < 5.0e-3 and abs(trf) < 1.0e-4)
    return {
        "node_r": node_r,
        "seg_dt": seg_dt_out, "delta_tau": float(dtauf),
        "time_resid": float(trf),
        "v_dep": v_dep, "v_arr": v_arr,
        "int_r0": rn0, "dv_moon0": dvm0, "dv_net0": dvn0,
        "int_r": rnf, "dv_moon": dvmf, "dv_net": dvnf,
        "mag_init": mag_init, "mag_final": mag_fin,
        "sum_init": float(np.nansum(mag_init)),
        "sum_final": sum_final,
        "ok": ok, "n_fail": int(nff), "iters": iters,
    }


def _interp_nodes(node_prev, n_new, r1, r2):
    """Resample a previous solution's node positions onto n_new+1 nodes by
    normalized index (linear), pinning the endpoints to this cell's r1, r2.
    Used to warm-start a cell from its converged neighbor across n_seg changes."""
    P = np.array([[p[0], p[1]] for p in node_prev])
    sp = np.linspace(0.0, 1.0, len(P))
    sn = np.linspace(0.0, 1.0, n_new + 1)
    x = np.interp(sn, sp, P[:, 0])
    y = np.interp(sn, sp, P[:, 1])
    nodes = [np.array([x[k], y[k], 0.0]) for k in range(n_new + 1)]
    nodes[0] = np.array([r1[0], r1[1], 0.0])
    nodes[-1] = np.array([r2[0], r2[1], 0.0])
    return nodes


def _discretized_cell(i, j, t_d, tof_s, segs_per_rev, mu_moon, warm_node=None,
                      moon_ecc=0.0):
    """Solve one cell. warm_node=None -> cold start (Sundman-ODE discretize);
    else continue from the neighbor's converged nodes (interpolated to this
    cell's n_seg). Returns (status, dv_total, opt_dict_or_None)."""
    th_park = THETA_PARK_0 + N_PARK * t_d
    r1 = np.array([R_PARK * np.cos(th_park), R_PARK * np.sin(th_park), 0.0])
    v_park = np.array([-R_PARK * N_PARK * np.sin(th_park),
                        R_PARK * N_PARK * np.cos(th_park), 0.0])
    r_moon, v_moon = _moon_state(t_d + tof_s, moon_ecc)
    r_target = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / np.linalg.norm(r_moon))

    ok, v1x, v1y, _z1, v2x, v2y, _z2 = _lambert_njit(
        r1[0], r1[1], 0.0, r_target[0], r_target[1], 0.0,
        tof_s, MU, 1, 0, 0, 1e-8, 400)
    if not ok:
        return STATUS_LAMBERT_FAIL, np.nan, None
    if abs(r1[0] * v1y - r1[1] * v1x) < H_MIN_SEED:
        return STATUS_SEED_DEGENERATE, np.nan, None
    if (np.hypot(v1x - v_park[0], v1y - v_park[1]) +
            np.hypot(v_moon[0] - v2x, v_moon[1] - v2y)) > DV_MAX_SEED:
        return STATUS_SEED_OVER_CAP, np.nan, None

    v1 = np.array([v1x, v1y, 0.0])
    try:
        if warm_node is None:
            d = discretize_conic_segments(r1, v1, tof_s, MU, segs_per_rev)
            node_r0, seg_dt0 = d["node_r"], d["seg_dt"]
        else:                                          # continuation warm start
            rn = np.linalg.norm(r1)
            inv_a = 2.0 / rn - (v1x * v1x + v1y * v1y) / MU
            T = (2.0 * np.pi * np.sqrt((1.0 / inv_a) ** 3 / MU)
                 if inv_a > 1e-12 else tof_s)
            n_seg = max(4, int(round(segs_per_rev * tof_s / T)))
            node_r0 = _interp_nodes(warm_node, n_seg, r1, r_target)
            seg_dt0 = [tof_s / n_seg] * n_seg
        opt = optimize_net_delta_v(r1, r_target, node_r0, seg_dt0,
                                   t_d, MU, mu_moon, moon_ecc=moon_ecc)
    except Exception:
        return STATUS_MAX_ITER, np.nan, None

    if not opt["ok"]:
        return STATUS_MAX_ITER, np.nan, opt
    dv_total = (float(np.linalg.norm(opt["v_dep"] - v_park)) +
                float(np.linalg.norm(v_moon - opt["v_arr"])) +
                opt["sum_final"])
    return STATUS_OK, dv_total, opt


def _solve_cell_discretized(args):
    """Single-cell cold solve (used by the Newton-seed comparison, etc.)."""
    i, j, t_d, tof_s, segs_per_rev, mu_moon = args
    st, dv, _ = _discretized_cell(i, j, t_d, tof_s, segs_per_rev, mu_moon)
    return i, j, dv, st


def _solve_cell_cold(args):
    """Cold-start worker (Phase 1): every cell independent -> each finds its
    own best solution (smooth where it converges). Returns
    (i, j, dv, status, node_r) — node_r kept for converged cells so a
    non-converged neighbor can warm-start from it in the rescue pass."""
    i, j, t_d, tof_s, segs_per_rev, mu_moon, moon_ecc = args
    st, dv, opt = _discretized_cell(i, j, t_d, tof_s, segs_per_rev,
                                    mu_moon, warm_node=None, moon_ecc=moon_ecc)
    node = opt["node_r"] if (st == STATUS_OK and opt is not None) else None
    return i, j, dv, st, node


def run_grid_discretized(segs_per_rev=24, verbose=True, n_workers=None,
                         mu_moon=None, moon_ecc=0.0):
    """Porkchop grid (N=0) solved by the discretized net-Δv-minimization
    method. mu_moon=None uses MU_MOON (Moon-perturbed); pass 0.0 for the pure
    two-body (Kepler) grid on the identical cells/target/screen. moon_ecc sets
    the lunar (target) orbit eccentricity (threaded through worker args).
    Returns (dv, status) — each shape (n_TOF, n_DEP)."""
    import time
    if mu_moon is None:
        mu_moon = MU_MOON
    n_t, n_d = len(TOF), len(T_DEP)
    dv = np.full((n_t, n_d), np.nan)
    status = np.full((n_t, n_d), -1, dtype=np.int8)

    if n_workers is None:
        n_workers = mp.cpu_count()
    t_d_of = [float(T_DEP[j] * 3600.0) for j in range(n_d)]
    tof_of = [float(TOF[i] * 3600.0) for i in range(n_t)]

    # --- Phase 1: cold-start every cell in parallel (independent -> smooth).
    node_of = {}                                   # (i,j) -> converged node_r
    cell_args = [(i, j, t_d_of[j], tof_of[i], segs_per_rev, mu_moon, moon_ecc)
                 for i in range(n_t) for j in range(n_d)]
    chunksize = max(1, len(cell_args) // (n_workers * 4))
    total = len(cell_args)
    t_start = time.time()
    with mp.Pool(n_workers) as pool:
        for n_done, (i, j, val, st, node) in enumerate(
            pool.imap_unordered(_solve_cell_cold, cell_args,
                                chunksize=chunksize), 1
        ):
            dv[i, j] = val
            status[i, j] = st
            if node is not None:
                node_of[(i, j)] = node
            if verbose and n_done % 200 == 0:
                print(f"\r  disc-grid cold: {n_done}/{total}  "
                      f"elapsed {time.time() - t_start:6.1f}s",
                      end="", flush=True)

    # --- Phase 2: rescue non-converged cells from their best converged
    # neighbor (any of the 4 directions), warm-started. Repeat until a pass
    # rescues nothing. Only touches the handful of stragglers.
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for _pass in range(6):
        rescued = 0
        targets = list(zip(*np.where(status == STATUS_MAX_ITER)))
        for i, j in targets:
            cand = [(dv[i + a, j + b], (i + a, j + b)) for a, b in nbrs
                    if (i + a, j + b) in node_of]
            if not cand:
                continue
            _, best = min(cand)                    # cheapest converged neighbor
            st, val, opt = _discretized_cell(
                i, j, t_d_of[j], tof_of[i], segs_per_rev, mu_moon,
                warm_node=node_of[best], moon_ecc=moon_ecc)
            if st == STATUS_OK and opt is not None:
                dv[i, j] = val
                status[i, j] = STATUS_OK
                node_of[(i, j)] = opt["node_r"]
                rescued += 1
        if verbose:
            print(f"\r  disc-grid rescue pass {_pass + 1}: "
                  f"+{rescued} cells          ")
        if rescued == 0:
            break

    if verbose:
        print()
        n_ok = int(np.sum(status == STATUS_OK))
        n_lam = int(np.sum(status == STATUS_LAMBERT_FAIL))
        n_mi = int(np.sum(status == STATUS_MAX_ITER))
        n_dg = int(np.sum(status == STATUS_SEED_DEGENERATE))
        n_cap = int(np.sum(status == STATUS_SEED_OVER_CAP))
        print(f"  status: ok={n_ok}, lambert_fail={n_lam}, "
              f"not_converged={n_mi}, seed_degen={n_dg}, "
              f"seed_over_cap={n_cap}")
        if n_ok:
            print(f"  discretized grid min ΔV = {np.nanmin(dv):.4f} km/s")
    return dv, status


def plot_homotopy_trajectories(t_dep_hr, t_arr_hr, factors=None,
                               segs_per_rev=24, out_name=None):
    """One figure for a single grid cell: overlay the conic (μ_moon=0)
    trajectory and the converged trajectory at every homotopy factor
    (warm-started up the ladder). Up to 1+len(factors) curves, colored by
    factor — shows how the arc bends as lunar gravity ramps and where (if
    anywhere) the continuation breaks."""
    if factors is None:
        factors = [round(0.1 * k, 1) for k in range(1, 11)]
    t_d = t_dep_hr * 3600.0
    tof_s = (t_arr_hr - t_dep_hr) * 3600.0
    r1, v_park = circ_state(R_PARK, THETA_PARK_0, N_PARK, t_d)
    r_moon, v_moon = _moon_state(t_d + tof_s)
    r_tgt = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / np.linalg.norm(r_moon))
    ok, v1x, v1y, *_ = _lambert_njit(r1[0], r1[1], 0.0, r_tgt[0], r_tgt[1], 0.0,
                                     tof_s, MU, 1, 0, 0, 1e-8, 400)
    v1 = np.array([v1x, v1y, 0.0])
    d = discretize_conic_segments(r1, v1, tof_s, MU, segs_per_rev)
    node_r, seg_dt = d["node_r"], d["seg_dt"]

    # (factor, node_r, seg_dt) — start with the conic (μ_moon = 0).
    chain = [(0.0, node_r, seg_dt)]
    fail_at = None
    for f in factors:
        opt = optimize_net_delta_v(r1, r_tgt, node_r, seg_dt, t_d, MU,
                                   f * MU_MOON_FULL)
        if not opt["ok"]:
            fail_at = f
            break
        node_r, seg_dt = opt["node_r"], opt["seg_dt"]
        chain.append((f, node_r, seg_dt))

    fig, ax = plt.subplots(figsize=(9, 9))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_PARK * np.cos(th), R_PARK * np.sin(th), "k--", lw=0.5, alpha=0.4)
    ax.plot(R_TGT * np.cos(th), R_TGT * np.sin(th), "k--", lw=0.5, alpha=0.4)
    ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#6fa8dc", alpha=0.5))
    cmap = plt.cm.viridis
    for (f, nr, sd) in chain:
        col = cmap(f)
        segs, _, _ = _piecewise_lambert_xy(nr, sd, MU)
        first = True
        for leg in segs:
            if leg is None:
                continue
            ax.plot(leg[:, 0], leg[:, 1], color=col, lw=1.3, alpha=0.9,
                    label=(f"μ_moon ×{f:.1f}" if first else None))
            first = False
    ax.plot([r1[0]], [r1[1]], "ko", ms=7)
    ax.plot([r_tgt[0]], [r_tgt[1]], "k*", ms=14)
    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    fail_txt = (f"  — FAILED at μ_moon ×{fail_at:.1f}" if fail_at is not None
                else "  — reached ×1.0")
    ax.set_title(f"Homotopy trajectories  t_dep={t_dep_hr:.2f}, "
                 f"t_arr={t_arr_hr:.2f} hr{fail_txt}", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="μ_moon factor")
    fig.tight_layout()
    if out_name is None:
        out_name = f"homotopy_traj_d{t_dep_hr:.2f}_a{t_arr_hr:.2f}.png"
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  ({len(chain)} trajectories, "
          f"{'failed at x%.1f' % fail_at if fail_at else 'reached x1.0'})")
    return fail_at


def _solve_cell_homotopy(args):
    """Moon-gravity homotopy for one cell: start from the conic discretization
    and ramp μ_moon through `factors` (×MU_MOON_FULL), warm-starting each step
    from the previous factor's converged solution. Small μ steps keep the cell
    on the conic-continued family (no basin jumps). Returns (i, j, dv_list,
    status) where dv_list[k] is the total ΔV at factors[k] (NaN if it dropped)."""
    i, j, t_d, tof_s, segs_per_rev, factors = args
    nf = len(factors)
    th_park = THETA_PARK_0 + N_PARK * t_d
    r1 = np.array([R_PARK * np.cos(th_park), R_PARK * np.sin(th_park), 0.0])
    v_park = np.array([-R_PARK * N_PARK * np.sin(th_park),
                        R_PARK * N_PARK * np.cos(th_park), 0.0])
    r_moon, v_moon = _moon_state(t_d + tof_s)
    r_target = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / np.linalg.norm(r_moon))

    ok, v1x, v1y, _z1, v2x, v2y, _z2 = _lambert_njit(
        r1[0], r1[1], 0.0, r_target[0], r_target[1], 0.0,
        tof_s, MU, 1, 0, 0, 1e-8, 400)
    if not ok:
        return i, j, [np.nan] * nf, STATUS_LAMBERT_FAIL
    if abs(r1[0] * v1y - r1[1] * v1x) < H_MIN_SEED:
        return i, j, [np.nan] * nf, STATUS_SEED_DEGENERATE
    if (np.hypot(v1x - v_park[0], v1y - v_park[1]) +
            np.hypot(v_moon[0] - v2x, v_moon[1] - v2y)) > DV_MAX_SEED:
        return i, j, [np.nan] * nf, STATUS_SEED_OVER_CAP

    v1 = np.array([v1x, v1y, 0.0])
    try:
        d = discretize_conic_segments(r1, v1, tof_s, MU, segs_per_rev)
        node_r, seg_dt = d["node_r"], d["seg_dt"]      # conic initial guess
    except Exception:
        return i, j, [np.nan] * nf, STATUS_MAX_ITER

    dvs = []
    for f in factors:
        mu_moon = f * MU_MOON_FULL
        try:
            opt = optimize_net_delta_v(r1, r_target, node_r, seg_dt,
                                       t_d, MU, mu_moon)
        except Exception:
            opt = None
        if opt is None or not opt["ok"]:
            dvs += [np.nan] * (nf - len(dvs))          # dropped at this factor
            break
        dvs.append(float(np.linalg.norm(opt["v_dep"] - v_park)) +
                   float(np.linalg.norm(v_moon - opt["v_arr"])) +
                   opt["sum_final"])
        node_r, seg_dt = opt["node_r"], opt["seg_dt"]  # warm start next factor
    st = (STATUS_OK if len(dvs) == nf and not np.isnan(dvs[-1])
          else STATUS_MAX_ITER)
    return i, j, dvs, st


def run_grid_homotopy(segs_per_rev=24, factors=None, verbose=True,
                      n_workers=None):
    """Moon-gravity homotopy porkchop: each cell starts from its conic
    solution and ramps μ_moon through `factors`, warm-starting each step from
    the previous. Returns (dv_by_factor, status, factors) — dv_by_factor[k] is
    the (n_TOF, n_DEP) ΔV grid at factors[k]."""
    import time
    if factors is None:
        factors = [round(0.1 * k, 1) for k in range(1, 11)]   # 0.1 .. 1.0
    n_t, n_d = len(TOF), len(T_DEP)
    dv_by_factor = [np.full((n_t, n_d), np.nan) for _ in factors]
    status = np.full((n_t, n_d), -1, dtype=np.int8)
    if n_workers is None:
        n_workers = mp.cpu_count()
    cell_args = [(i, j, float(T_DEP[j] * 3600.0), float(TOF[i] * 3600.0),
                  segs_per_rev, tuple(factors))
                 for i in range(n_t) for j in range(n_d)]
    chunksize = max(1, len(cell_args) // (n_workers * 4))
    total = len(cell_args)
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for n, (i, j, dvs, st) in enumerate(
            pool.imap_unordered(_solve_cell_homotopy, cell_args,
                                chunksize=chunksize), 1):
            status[i, j] = st
            for k, val in enumerate(dvs):
                dv_by_factor[k][i, j] = val
            if verbose and n % 200 == 0:
                print(f"\r  homotopy: {n}/{total}  "
                      f"elapsed {time.time() - t0:6.1f}s", end="", flush=True)
    if verbose:
        print()
        for k, f in enumerate(factors):
            nok = int(np.sum(np.isfinite(dv_by_factor[k])))
            mn = (np.nanmin(dv_by_factor[k]) if nok else np.nan)
            print(f"  μ_moon ×{f:.1f}: {nok} cells, min ΔV = {mn:.4f} km/s")
    return dv_by_factor, status, factors


def plot_discretized_compare(dv_total, places, segs_per_rev=24,
                             out_name="compare_discretized.png"):
    """One figure: the conic porkchop plus three solution panels. Each panel
    shows the undiscretized conic, the initial-guess discretization, and the
    net-Δv-minimized (Moon-perturbed) final solution."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    ax_pork = axes[0, 0]
    sol_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]

    # ---- Porkchop panel (same construction as plot_porkchop_arrival) ----
    t_arr_grid = np.linspace(T_DEP[0] + TOF[0], T_DEP[-1] + TOF[-1], 300)
    DEP, ARR = np.meshgrid(T_DEP, t_arr_grid)
    TOF_query = ARR - DEP
    Z = np.full_like(DEP, np.nan)
    for j in range(len(T_DEP)):
        col_tof = TOF_query[:, j]
        valid = (col_tof >= TOF[0]) & (col_tof <= TOF[-1])
        Z[valid, j] = np.interp(col_tof[valid], TOF, dv_total[:, j])
    vmin = float(np.nanmin(Z))
    vmax = vmin + 2.0
    Z_clip = np.where(Z > vmax, np.nan, Z)
    levels = np.linspace(vmin, vmax, 40)
    ax_pork.set_facecolor("lightgrey")
    cs = ax_pork.contourf(DEP, ARR, Z_clip, levels=levels, cmap="viridis",
                          extend="neither")
    ax_pork.contour(DEP, ARR, Z_clip, levels=12, colors="k",
                    linewidths=0.4, alpha=0.5)
    plt.colorbar(cs, ax=ax_pork, label="Total ΔV [km/s]")
    ax_pork.set_xlabel("Departure time [hr]")
    ax_pork.set_ylabel("Arrival time [hr]")
    ax_pork.set_title(f"Porkchop (conic, N=0)   min ΔV = {vmin:.3f} km/s")

    # ---- Three solution panels ----
    for idx, (ax, (t_dep_hr, t_arr_hr)) in enumerate(zip(sol_axes, places), 1):
        t_d = t_dep_hr * 3600.0
        tof_s = (t_arr_hr - t_dep_hr) * 3600.0
        r1, v_park = circ_state(R_PARK, THETA_PARK_0, N_PARK, t_d)
        r2, v_tgt = circ_state(R_TGT, THETA_TGT_0, N_TGT, t_d + tof_s)

        # Mark this place on the porkchop.
        ax_pork.plot([t_dep_hr], [t_arr_hr], "*", color="red", ms=15)
        ax_pork.annotate(f"({idx})", (t_dep_hr, t_arr_hr),
                         textcoords="offset points", xytext=(6, 4),
                         color="red", fontsize=11, fontweight="bold")

        sol = lambert_uv(r1, r2, tof_s, MU, prograde=True)
        if sol is None:
            ax.set_title(f"({idx}) t_dep={t_dep_hr:.2f}, "
                         f"t_arr={t_arr_hr:.2f} hr — no Lambert solution")
            continue
        v_t1, v_t2 = sol
        dv_tot = np.linalg.norm(v_t1 - v_park) + np.linalg.norm(v_tgt - v_t2)

        # Undiscretized conic, time-sampled with the same propagator the
        # discretization uses (so any visible gap is a real mismatch).
        ts = np.linspace(0.0, tof_s, 600)
        und = np.array([_kepler_uv(r1, v_t1, t, MU)[0] for t in ts])
        ax.plot(und[:, 0], und[:, 1], color="#1f77b4", lw=2.6, alpha=0.9,
                label=f"Undiscretized  (ΔV={dv_tot:.3f} km/s)")

        # Initial guess: conic discretization (per-segment Lambert).
        d = discretize_conic_segments(r1, v_t1, tof_s, MU, segs_per_rev)
        for k in range(d["n_seg"]):
            vo = d["seg_v_out"][k]
            if vo is None:
                continue
            ss = np.linspace(0.0, d["seg_dt"][k], 30)
            seg = np.array([_kepler_uv(d["node_r"][k], vo, s, MU)[0]
                            for s in ss])
            ax.plot(seg[:, 0], seg[:, 1], color="#d62728", lw=1.0,
                    linestyle="--", alpha=0.8,
                    label="Initial guess (conic disc.)" if k == 0 else None)
        nodes0 = np.array(d["node_r"])
        ax.plot(nodes0[:, 0], nodes0[:, 1], "o", color="#d62728",
                ms=3.0, alpha=0.7)

        # Minimize Σ|Δv_net|, Δv_net = Δv_conic − Δv_moon, over interior nodes.
        opt = optimize_net_delta_v(r1, r2, d["node_r"], d["seg_dt"],
                                   t_d, MU, MU_MOON)
        segs_f, _, _ = _piecewise_lambert_xy(opt["node_r"], opt["seg_dt"], MU)
        for k, leg in enumerate(segs_f):
            if leg is None:
                continue
            ax.plot(leg[:, 0], leg[:, 1], color="#2ca02c", lw=1.6,
                    alpha=0.9,
                    label="Final (min Σ|Δv_net|)" if k == 0 else None)
        nodes_f = np.array(opt["node_r"])
        ax.plot(nodes_f[:, 0], nodes_f[:, 1], "o", color="#2ca02c", ms=3.5)

        # Δv vectors at the interior nodes, for BOTH the initial guess and the
        # final solution. Δv_moon shares one scale (init vs final comparable);
        # each Δv_net field gets its own scale so it stays visible (init
        # Δv_net≈−Δv_moon is large; final Δv_net≈0 for converged places).
        span = max(np.ptp(und[:, 0]), np.ptp(und[:, 1]))
        arrow_len = 10.0                               # 10x longer vectors

        def _Smag(*fields):
            m = max((float(np.nanmax(np.hypot(f[:, 0], f[:, 1])))
                     for f in fields
                     if np.isfinite(f).any() and np.nanmax(
                         np.hypot(f[:, 0], f[:, 1])) > 0), default=0.0)
            return (m / (arrow_len * 0.18 * span)) if m > 0 else 1.0

        def _sum_mps(f):
            return np.nansum(np.hypot(f[:, 0], f[:, 1])) * 1000.0

        ir0, dvm0, dvn0 = opt["int_r0"], opt["dv_moon0"], opt["dv_net0"]
        ir, dvm, dvn = opt["int_r"], opt["dv_moon"], opt["dv_net"]
        f0 = ~np.isnan(dvm0[:, 0])
        ff = ~np.isnan(dvm[:, 0])

        aS_m = _Smag(dvm0, dvm)                         # shared Moon scale
        aS_n0 = _Smag(dvn0)                             # init-net own scale
        aS_nf = _Smag(dvn)                              # final-net own scale
        qopts = dict(angles="xy", scale_units="xy", width=0.004)
        # Convention: light shade = initial guess, saturated = final;
        # green hue = Δv_moon, pink/magenta hue = Δv_net.
        if f0.any():
            ax.quiver(ir0[f0, 0], ir0[f0, 1], dvm0[f0, 0], dvm0[f0, 1],
                      scale=aS_m, color="#98df8a", alpha=0.85, **qopts,
                      label=f"Δv_moon init  (Σ={_sum_mps(dvm0):.1f} m/s)")
            ax.quiver(ir0[f0, 0], ir0[f0, 1], dvn0[f0, 0], dvn0[f0, 1],
                      scale=aS_n0, color="#f7b6d2", alpha=0.9, **qopts,
                      label=f"Δv_net init  (Σ={_sum_mps(dvn0):.1f} m/s)")
        if ff.any():
            ax.quiver(ir[ff, 0], ir[ff, 1], dvm[ff, 0], dvm[ff, 1],
                      scale=aS_m, color="#2ca02c", alpha=0.85, **qopts,
                      label=f"Δv_moon final (Σ={_sum_mps(dvm):.1f} m/s)")
            ax.quiver(ir[ff, 0], ir[ff, 1], dvn[ff, 0], dvn[ff, 1],
                      scale=aS_nf, color="#e377c2", alpha=0.95, **qopts,
                      label=(f"Δv_net final (Σ={opt['sum_final']*1000:.2f}"
                             f" m/s, indep.)"))

        mxf = np.nanmax(opt["mag_final"]) if opt["mag_final"].size else 0.0
        mxf = 0.0 if np.isnan(mxf) else mxf
        print(f"  place {idx} BFGS Σ|Δv_net|: "
              f"init={opt['sum_init'] * 1000:.2f} -> "
              f"final={opt['sum_final'] * 1000:.4f} m/s  "
              f"(max node {mxf * 1000:.4f} m/s, Δτ={opt['delta_tau']:.4g}, "
              f"time_resid={opt['time_resid']:.1e}, "
              f"{'ok' if opt['ok'] else 'NO-CONV'}, "
              f"{d['n_seg']} segs, {opt['n_fail']} bad seg, "
              f"μ_moon×{MU_MOON_SCALE})")

        fail_note = (f", {opt['n_fail']} bad seg" if opt["n_fail"] else "")
        seg_info = (f"{d['n_seg']} segs  "
                    f"Σ|Δv_net|: {opt['sum_init'] * 1000:.1f}→"
                    f"{opt['sum_final'] * 1000:.3f} m/s  "
                    f"(Δτ={opt['delta_tau']:.3g}, "
                    f"tres={opt['time_resid']:.0e}){fail_note}")

        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(R_PARK * np.cos(th), R_PARK * np.sin(th), "k--",
                lw=0.5, alpha=0.4)
        ax.plot(R_TGT * np.cos(th), R_TGT * np.sin(th), "k--",
                lw=0.5, alpha=0.4)
        ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#6fa8dc", alpha=0.5))
        ax.plot([r1[0]], [r1[1]], "ko", ms=7)
        ax.plot([r2[0]], [r2[1]], "k*", ms=14)
        ax.set_aspect("equal")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"({idx}) t_dep={t_dep_hr:.2f}, t_arr={t_arr_hr:.2f} hr\n"
                     f"{seg_info}")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Net-Δv minimization: initial conic discretization vs "
                 f"Moon-perturbed final ({segs_per_rev} segs/period)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}")


def compare_newton_guesses(places, segs_per_rev=48, mu_moon=None):
    """For each place, run the Moon-gravity Newton shooter from two initial
    guesses and tabulate the iteration counts:

      A: conic Lambert solution velocity (the legacy seed)
      B: departure velocity of the net-Δv-minimized discretized solution

    Both target the Moon offset point r_moon·(1−R_moon/|r_moon|) and refine
    under full Earth+Moon dynamics. A lower count for B means the discretized
    solution is a better Newton seed."""
    if mu_moon is None:
        mu_moon = MU_MOON
    name = {STATUS_OK: "ok", STATUS_MAX_ITER: "max_iter",
            STATUS_SINGULAR_JAC: "singular"}
    rows = []
    for (t_dep_hr, t_arr_hr) in places:
        t_d = t_dep_hr * 3600.0
        tof_s = (t_arr_hr - t_dep_hr) * 3600.0
        t_a = t_d + tof_s
        r1, _ = circ_state(R_PARK, THETA_PARK_0, N_PARK, t_d)
        r_moon, _ = _moon_state(t_a)
        r_tgt = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / np.linalg.norm(r_moon))

        solA = lambert_uv(r1, r_tgt, tof_s, MU, prograde=True)
        gA = None if solA is None else solA[0]

        gB = None
        if gA is not None:
            d = discretize_conic_segments(r1, gA, tof_s, MU, segs_per_rev)
            opt = optimize_net_delta_v(r1, r_tgt, d["node_r"], d["seg_dt"],
                                       t_d, MU, mu_moon)
            sB = lambert_uv(r1, opt["node_r"][1], opt["seg_dt"][0], MU,
                            prograde=True)
            gB = None if sB is None else sB[0]

        recA = (_newton_shoot_counted(r1, gA, t_d, t_a, r_tgt,
                                      mu_moon=mu_moon)
                if gA is not None else None)
        recB = (_newton_shoot_counted(r1, gB, t_d, t_a, r_tgt,
                                      mu_moon=mu_moon)
                if gB is not None else None)
        rows.append((t_dep_hr, t_arr_hr, recA, recB))

    def _fmt(rec):
        if rec is None:
            return f"{'—':>5} {'(no seed)':>10} {'—':>10}"
        _, _, st, nit, res = rec
        return f"{nit:>5d} {name.get(st, st):>10} {res:>10.2f}"

    print()
    print(f"  Newton initial-guess comparison  (μ_moon×{MU_MOON_SCALE}, "
          f"{segs_per_rev} segs/period, pos_tol=50 km)")
    print("  " + "-" * 78)
    print(f"  {'place (t_dep,t_arr) hr':<24}"
          f"|{'conic seed':^28}|{'discretized seed':^28}")
    print(f"  {'':<24}|{'iters':>5} {'status':>10} {'res km':>10} "
          f"|{'iters':>5} {'status':>10} {'res km':>10} ")
    print("  " + "-" * 78)
    for (td, ta, recA, recB) in rows:
        print(f"  ({td:>4.2f}, {ta:>6.2f}){'':<10}|{_fmt(recA)} |{_fmt(recB)} ")
    print("  " + "-" * 78)
    return rows


def plot_compare_solutions(places, out_name="compare_solutions.png",
                           mu_moon=None):
    """For each (t_dep_hr, t_arr_hr) in places, draw four trajectories:
    conic N=0, conic N=1, Newton N=0, Newton N=1.
    Three places × four solutions = 12 trajectories."""
    if mu_moon is None:
        mu_moon = MU_MOON
    n_panels = len(places)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    for ax, (t_dep_hr, t_arr_hr) in zip(axes, places):
        t_d = t_dep_hr * 3600.0
        tof_s = (t_arr_hr - t_dep_hr) * 3600.0
        t_a = t_d + tof_s

        th_park = THETA_PARK_0 + N_PARK * t_d
        r1 = np.array([R_PARK * np.cos(th_park),
                       R_PARK * np.sin(th_park), 0.0])
        v_park = np.array([-R_PARK * N_PARK * np.sin(th_park),
                            R_PARK * N_PARK * np.cos(th_park), 0.0])

        r_moon, v_moon = _moon_state(t_a)
        r_tgt_newton = r_moon * (1.0 - MOON_ARRIVAL_OFFSET / np.linalg.norm(r_moon))

        cos_dnu = np.clip(np.dot(r1, r_moon) /
                          (np.linalg.norm(r1) * np.linalg.norm(r_moon)), -1.0, 1.0)
        dnu_geom = np.arccos(cos_dnu)
        if np.cross(r1, r_moon)[2] < 0.0:
            dnu_geom = 2.0 * np.pi - dnu_geom

        legend_entries = []

        # ---- Conic Lambert trajectories (target = moon center) ----
        for n_rev, ls, label in [(0, "-", "Conic N=0"), (1, "--", "Conic N=1")]:
            best = None
            branches = (0,) if n_rev == 0 else (0, 1)
            for bs in branches:
                ok, v1x, v1y, _v1z, v2x, v2y, _v2z = _lambert_njit(
                    r1[0], r1[1], 0.0, r_moon[0], r_moon[1], 0.0,
                    tof_s, MU, 1, n_rev, bs, 1e-8, 400,
                )
                if not ok:
                    continue
                v_t1 = np.array([v1x, v1y, 0.0])
                v_t2 = np.array([v2x, v2y, 0.0])
                dv_tot = np.linalg.norm(v_t1 - v_park) + np.linalg.norm(v_moon - v_t2)
                if best is None or dv_tot < best[0]:
                    best = (dv_tot, v_t1)
            if best is None:
                continue
            total_angle = 2 * np.pi * n_rev + dnu_geom
            xs, ys = transfer_arc(r1, best[1], total_angle, MU, n_pts=600)
            ax.plot(xs, ys, color="#1f77b4", linestyle=ls, lw=1.8, alpha=0.85,
                    label=f"{label}  ({best[0]:.2f} km/s)")

        # ---- Newton trajectories (target = moon offset) ----
        for n_rev, ls, label in [(0, "-", "Newton N=0"), (1, "--", "Newton N=1")]:
            best = None
            branches = (0,) if n_rev == 0 else (0, 1)
            for bs in branches:
                ok, v1x, v1y, _v1z, _v2x, _v2y, _v2z = _lambert_njit(
                    r1[0], r1[1], 0.0, r_tgt_newton[0], r_tgt_newton[1], 0.0,
                    tof_s, MU, 1, n_rev, bs, 1e-8, 400,
                )
                if not ok:
                    continue
                if abs(r1[0] * v1y - r1[1] * v1x) < H_MIN_SEED:
                    continue
                v_guess = np.array([v1x, v1y, 0.0])
                v_t1, v_f, status, _res = _shoot_to_moon(
                    r1, v_guess, t_d, t_a, r_tgt_newton, mu_moon=mu_moon)
                if status != STATUS_OK:
                    continue
                dv_tot = np.linalg.norm(v_t1 - v_park) + np.linalg.norm(v_moon - v_f)
                if best is None or dv_tot < best[0]:
                    best = (dv_tot, v_t1)
            if best is None:
                continue
            t_eval = np.linspace(t_d, t_a, 400)
            sol = solve_ivp(_dynamics_em, (t_d, t_a),
                            np.concatenate([r1, best[1]]),
                            method="DOP853", rtol=1e-6, atol=1e-1,
                            t_eval=t_eval, args=(mu_moon,))
            if not sol.success:
                continue
            ax.plot(sol.y[0], sol.y[1], color="#d62728", linestyle=ls,
                    lw=1.8, alpha=0.85,
                    label=f"{label}  ({best[0]:.2f} km/s)")

        # References
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(R_PARK * np.cos(th), R_PARK * np.sin(th), "k--", lw=0.5, alpha=0.4)
        ax.plot(R_TGT * np.cos(th), R_TGT * np.sin(th), "k--", lw=0.5, alpha=0.4)
        ax.add_patch(plt.Circle((0, 0), R_EARTH, color="#6fa8dc", alpha=0.5))
        ax.plot([r1[0]], [r1[1]], "ko", ms=7)
        ax.plot([r_moon[0]], [r_moon[1]], "k*", ms=14)

        ax.set_aspect("equal")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"t_dep = {t_dep_hr:.2f} hr,  t_arr = {t_arr_hr:.2f} hr")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Conic vs Newton solutions  (μ_moon × {MU_MOON_SCALE}, "
        f"R_TGT = {R_TGT:.0f} km)"
    )
    fig.tight_layout()
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}")


def plot_porkchop_failure_map(status, residual,
                              out_name="porkchop_failure_map.png"):
    """Categorical map of shooter failure modes on (departure, arrival) axes.
    Side panel shows |r_f - r_target| for the cells that didn't converge."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    # Pack status into the (dep, arr) grid via nearest-TOF lookup
    t_arr_grid = np.linspace(T_DEP[0] + TOF[0], T_DEP[-1] + TOF[-1], 300)
    DEP, ARR = np.meshgrid(T_DEP, t_arr_grid)
    TOF_query = ARR - DEP

    S = np.full_like(DEP, -1)
    R = np.full_like(DEP, np.nan)
    for j in range(len(T_DEP)):
        col_tof = TOF_query[:, j]
        valid = (col_tof >= TOF[0]) & (col_tof <= TOF[-1])
        idx = np.clip(np.searchsorted(TOF, col_tof[valid]), 0, len(TOF) - 1)
        S[valid, j] = status[idx, j]
        R[valid, j] = residual[idx, j]

    # Categorical colormap: -1 out-of-grid, 0 OK, 1 Lambert fail, 2 max-iter,
    # 3 Jacobian singular, 4 seed degenerate (|h|<H_MIN), 5 seed over cap
    colors = ["#f0f0f0", "#7fbf7f", "#ffcc66", "#ff7f7f",
              "#9966cc", "#404040", "#cccccc"]
    labels = ["out of grid", "OK", "Lambert failed", "Newton max-iter",
              "Jacobian singular",
              f"seed |h| < {H_MIN_SEED:.0f}",
              f"seed ΔV > {DV_MAX_SEED:.1f}"]
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))

    # Left: failure category map
    im = axL.pcolormesh(DEP, ARR, S, cmap=cmap, norm=norm, shading="auto")
    cbar = plt.colorbar(im, ax=axL, ticks=[-1, 0, 1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(labels)
    axL.set_xlabel("Departure time [hr]")
    axL.set_ylabel("Arrival time [hr]")
    axL.set_title("Shooter status by cell")

    # Right: residual norm for non-converged cells (log scale, masked where OK)
    R_fail = np.where(S > 0, R, np.nan)
    if np.isfinite(R_fail).any():
        vmin = max(50.0, float(np.nanmin(R_fail)))
        vmax = float(np.nanmax(R_fail))
        from matplotlib.colors import LogNorm
        im2 = axR.pcolormesh(DEP, ARR, R_fail, cmap="magma",
                             norm=LogNorm(vmin=vmin, vmax=vmax), shading="auto")
        plt.colorbar(im2, ax=axR, label="|r_f - r_target| [km]")
    axR.set_xlabel("Departure time [hr]")
    axR.set_ylabel("Arrival time [hr]")
    axR.set_title("Final position residual (failed cells)")

    # Tally
    counts = {lbl: int(np.sum(status == code))
              for code, lbl in zip([0, 1, 2, 3, 4, 5], labels[1:])}
    summary = "  ".join(f"{k}: {v}" for k, v in counts.items())
    fig.suptitle(f"Newton shooter failure analysis  ({summary})")

    fig.tight_layout()
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  ({summary})")


def plot_porkchop_arrival_delta(dv_a, dv_b,
                                out_name="porkchop_arrival_delta.png",
                                title="Δ(ΔV): conic - moon"):
    """Plot (dv_a - dv_b) on the (departure, arrival) grid.
    Cells where the moon (dv_b) has no Newton solution are rendered grey.
    """
    t_arr_grid = np.linspace(T_DEP[0] + TOF[0], T_DEP[-1] + TOF[-1], 300)
    DEP, ARR = np.meshgrid(T_DEP, t_arr_grid)
    TOF_query = ARR - DEP

    # Nearest-neighbor lookup on the raw (TOF, T_DEP) grid so NaN cells in dv_b
    # remain NaN in the rendered plot (no interpolation across missing data).
    Z = np.full_like(DEP, np.nan)
    for j in range(len(T_DEP)):
        col_tof = TOF_query[:, j]
        valid = (col_tof >= TOF[0]) & (col_tof <= TOF[-1])
        idx_nearest = np.clip(np.searchsorted(TOF, col_tof[valid]), 0, len(TOF) - 1)
        a_col = dv_a[idx_nearest, j]
        b_col = dv_b[idx_nearest, j]
        Z[valid, j] = a_col - b_col   # NaN propagates where either side is NaN

    absmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
    if absmax == 0:
        absmax = 1.0

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("lightgrey")  # NaN cells inherit this colour
    cs = ax.contourf(DEP, ARR, Z, levels=40, cmap="RdBu_r",
                     vmin=-absmax, vmax=absmax)
    ax.contour(DEP, ARR, Z, levels=10, colors="k", linewidths=0.3, alpha=0.4)
    plt.colorbar(cs, ax=ax, label="Δ(ΔV) = conic - moon [km/s]")
    ax.set_xlabel("Departure time [hr]")
    ax.set_ylabel("Arrival time [hr]")
    ax.set_title(
        f"{title}  (grey = no Newton solution)\n"
        f"max |Δ| = {absmax:.3f} km/s"
    )
    fig.tight_layout()
    fig.savefig(out_name, dpi=120)
    print(f"Saved {out_name}  (max |Δ| = {absmax:.4f} km/s)")


def plot_porkchop_arrival_multirev(dv_best, n_best):
    """Best-of-N-rev total ΔV contour in (departure, arrival) axes, with N regions overlaid."""
    t_arr_grid = np.linspace(T_DEP[0] + TOF[0], T_DEP[-1] + TOF[-1], 300)
    DEP, ARR = np.meshgrid(T_DEP, t_arr_grid)
    TOF_query = ARR - DEP

    Z = np.full_like(DEP, np.nan)
    N = np.full_like(DEP, np.nan)
    for j in range(len(T_DEP)):
        col_tof = TOF_query[:, j]
        valid = (col_tof >= TOF[0]) & (col_tof <= TOF[-1])
        Z[valid, j] = np.interp(col_tof[valid], TOF, dv_best[:, j])
        idx = np.clip(np.searchsorted(TOF, col_tof[valid]), 1, len(TOF) - 1)
        N[valid, j] = n_best[idx, j]

    vmin = float(np.nanmin(Z))
    vmax = vmin + 2.0
    Z_clip = np.where(Z > vmax, np.nan, Z)
    levels = np.linspace(vmin, vmax, 40)
    n_max = int(np.nanmax(n_best))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("lightgrey")
    cs = ax.contourf(DEP, ARR, Z_clip, levels=levels, cmap="viridis", extend="neither")
    ax.contour(DEP, ARR, Z_clip, levels=12, colors="k", linewidths=0.4, alpha=0.5)
    plt.colorbar(cs, ax=ax, label="Best total ΔV [km/s]")

    # Overlay boundaries between regions where different N wins
    if n_max >= 1:
        boundary_levels = np.arange(0.5, n_max + 0.5, 1.0)
        try:
            ax.contour(DEP, ARR, N, levels=boundary_levels,
                       colors="white", linewidths=1.0, linestyles="--")
        except Exception:
            pass

        # Annotate each N region with its label at the centroid
        for n_val in range(n_max + 1):
            mask = (N == n_val)
            if mask.sum() < 30:
                continue
            xc = float(np.nanmean(DEP[mask]))
            yc = float(np.nanmean(ARR[mask]))
            ax.text(xc, yc, f"N={n_val}",
                    color="white", fontsize=10, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=2))

    ax.set_xlabel("Departure time [hr]")
    ax.set_ylabel("Arrival time [hr]")
    ax.set_title(
        f"Porkchop (best of N=0..{n_max}): circ {R_PARK:.0f} km → circ {R_TGT:.0f} km\n"
        f"(white dashed = inter-N boundaries)"
    )
    # Mark the global minimum on the (dep, arr) plot with a star
    i_min, j_min = np.unravel_index(np.nanargmin(dv_best), dv_best.shape)
    n_at_min = int(n_best[i_min, j_min])
    t_dep_min = T_DEP[j_min]
    t_arr_min = T_DEP[j_min] + TOF[i_min]
    ax.plot(t_dep_min, t_arr_min, marker="*", color="red", ms=18,
            markeredgecolor="white", markeredgewidth=1.0,
            label=f"min ΔV = {vmin:.3f} km/s (N={n_at_min})")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out = "porkchop_arrival_multirev.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}  (min best ΔV = {vmin:.4f} km/s at N={n_at_min}, "
          f"t_dep={t_dep_min:.3f} hr, t_arr={t_arr_min:.2f} hr; N up to {n_max})")


if __name__ == "__main__":
    import time
    # Raw-cell porkchops at several lunar (target) orbit eccentricities. For
    # each ecc: full Moon gravity + pure Kepler (no Moon). The eccentric lunar
    # orbit moves the target, so the Kepler grid also varies with ecc.
    eccs = [0.0, 0.1, 0.5]
    grids = {}                                 # (kind, ecc) -> dv array
    for e in eccs:
        t2 = time.time()
        dv_disc, _ = run_grid_discretized(segs_per_rev=24, moon_ecc=e)
        dv_kep, _ = run_grid_discretized(segs_per_rev=24, mu_moon=0.0,
                                         moon_ecc=e)
        print(f"  ecc={e:.1f} grids done in {time.time() - t2:.1f}s")
        grids[("moon", e)] = dv_disc
        grids[("kep", e)] = dv_kep

    # One shared color scale across all six: min over every grid, single top
    # at the DV_MAX_SEED cap so all plots are directly comparable.
    shared_min = float(min(np.nanmin(g) for g in grids.values()))
    shared_max = DV_MAX_SEED  # km/s
    for e in eccs:
        plot_porkchop_grid_raw(
            grids[("moon", e)],
            out_name=f"porkchop_moon_gravity_raw_ecc{e:.1f}.png",
            title=f"Discretized net-Δv min, N=0, lunar ecc={e:.1f} (raw cells)",
            vmin=shared_min, vmax=shared_max)
        plot_porkchop_grid_raw(
            grids[("kep", e)],
            out_name=f"porkchop_kepler_raw_ecc{e:.1f}.png",
            title=f"Kepler (no Moon gravity), N=0, lunar ecc={e:.1f} (raw cells)",
            vmin=shared_min, vmax=shared_max)

    # --- Newton section commented out: conic set only for now ---
    # # Conic grid restricted to N=0 for an apples-to-apples Newton comparison
    # # (N=0,1 sweep commented out -> N=0 only)
    # dv_conic_n01, _, _ = run_grid_multirev(max_n_rev=0, verbose=False)
    #
    # # Newton grid (moon-gravity shooting, N=0,1). Slower.
    # t1 = time.time()
    # dv_newton, newton_n_best, newton_status, newton_residual = run_grid_with_moon()
    # print(f"  newton grid done in {time.time() - t1:.1f}s")
    # plot_porkchop_arrival(dv_newton,
    #                       out_name="porkchop_arrival_newton.png",
    #                       title="Porkchop (Newton, best of N=0,1)")
    # plot_porkchop_arrival_delta(dv_conic_n01, dv_newton,
    #                             out_name="porkchop_arrival_delta.png",
    #                             title="Δ(ΔV): conic best(N=0,1) − newton best(N=0,1)")
    # plot_porkchop_failure_map(newton_status, newton_residual)
    # plot_compare_solutions(places=[(0.5, 18.0), (0.5, 50.0), (0.5, 130.0)])
