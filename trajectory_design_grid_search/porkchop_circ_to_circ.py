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

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import root

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

# Skip Newton shooting when the conic Lambert seed's total ΔV is already above this.
# Anything well beyond the porkchop loins is uninteresting; saves runtime.
DV_MAX_SEED = 7.0  # km/s
R_EARTH = 6378.0            # km

R_PARK = R_EARTH + 300.0    # parking orbit radius (300 km LEO)
R_TGT = 96_100.0            # target circular orbit radius (25% of lunar distance)

THETA_PARK_0 = 0.0          # spacecraft true anomaly at t=0 [rad]
THETA_TGT_0 = np.deg2rad(90.0)  # target true anomaly at t=0 [rad]

N_PARK = np.sqrt(MU / R_PARK**3)
N_TGT = np.sqrt(MU / R_TGT**3)

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
def _moon_state(t):
    th = THETA_TGT_0 + N_TGT * t
    r = R_TGT * np.array([np.cos(th), np.sin(th), 0.0])
    v = R_TGT * N_TGT * np.array([-np.sin(th), np.cos(th), 0.0])
    return r, v


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
    t0 = time.time()
    dv_best, n_best, dv_n0 = run_grid_multirev(max_n_rev=0)  # N>0 revs commented out
    print(f"  conic grid done in {time.time() - t0:.1f}s")
    print_summary(dv_n0)
    plot_porkchop_arrival(dv_n0,
                          out_name="porkchop_arrival.png",
                          title="Porkchop (conic, N=0)")
    plot_porkchop_arrival_multirev(dv_best, n_best)
    plot_trajectories_fixed_dep(t_dep_hr=0.0)
    plot_trajectories_fixed_dep(t_dep_hr=1.4)
    # multi-rev trajectory plots commented out (conic set is N=0 only)
    # plot_trajectories_multirev(t_dep_hr=0.0, max_n_rev=2)
    # plot_trajectories_multirev(t_dep_hr=1.4, max_n_rev=2)

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
