"""
Gravity-assist analysis — system setup.

Inertial, planar frame with the central body fixed at the origin. A massive
secondary ("assist") body orbits the central body on a circular, planar orbit.
A spacecraft starts on a smaller circular orbit about the central body and can
swing past the assist body to gain/lose energy (the gravity assist).

Canonical units: the assist orbit radius is the length unit (DU) and the
central body's gravitational parameter is 1. The assist body's mean motion and
orbital period are then 1 and 2*pi (treating the central body as fixed — i.e.
the assist body's pull on the central body is neglected, standard for analysing
the restricted problem). Multiply by a real system's (R_assist, sqrt(mu/R^3))
to dimensionalize.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---- Central body (fixed at the origin) ----
MU_C = 1.0                       # central-body gravitational parameter [canonical]

# ---- Assist body: massive, on a circular planar orbit ----
R_A = 1.0                        # assist orbit radius -> defines the length unit
MU_A = 1.0e-2                    # assist-body gravitational parameter (its mass)
R_BODY = 0.03                    # assist-body physical radius (min flyby periapsis) [DU]
N_A = np.sqrt(MU_C / R_A ** 3)   # assist mean motion (central fixed)
THETA_A0 = 0.0                   # assist phase angle at t = 0 [rad]

# ---- Spacecraft initial orbit: a smaller circle about the central body ----
R_INIT = 0.5                     # initial circular orbit radius (< R_A)
THETA_SC0 = np.deg2rad(180.0)    # spacecraft start angle at t = 0 [rad]


def assist_state(t):
    """Position & velocity (2-D) of the assist body at time t (circular)."""
    th = THETA_A0 + N_A * t
    r = R_A * np.array([np.cos(th), np.sin(th)])
    v = R_A * N_A * np.array([-np.sin(th), np.cos(th)])
    return r, v


def circular_state(radius, theta, mu=MU_C):
    """A prograde circular-orbit state at the given radius and polar angle."""
    n = np.sqrt(mu / radius ** 3)
    r = radius * np.array([np.cos(theta), np.sin(theta)])
    v = radius * n * np.array([-np.sin(theta), np.cos(theta)])
    return r, v


def accel(t, rx, ry):
    """Planar acceleration on the spacecraft from the central body plus the
    moving assist body (both point masses)."""
    r = np.array([rx, ry])
    a = -MU_C * r / np.linalg.norm(r) ** 3
    ra, _ = assist_state(t)
    d = ra - r
    a = a + MU_A * d / np.linalg.norm(d) ** 3
    return a


def dynamics(t, y):
    """State derivative for y = [x, y, vx, vy]."""
    ax, ay = accel(t, y[0], y[1])
    return [y[2], y[3], ax, ay]


# ----------------------------------------------------------------------------
# Patched-conic gravity assist
#
# The spacecraft Hohmann-transfers from its inner circular orbit (R_INIT) out
# to the assist body's orbit (R_A), arriving tangentially at the transfer
# apoapsis with speed v_apo < v_circ(R_A) — i.e. moving slower than the assist
# body. Its velocity RELATIVE to the body is the hyperbolic excess velocity
# v_inf; the flyby rotates v_inf by the turn angle delta (its magnitude is
# conserved), and the new heliocentric velocity is v_a + v_inf_rotated. Smaller
# flyby periapsis -> larger turn -> bigger change. Work in the local
# (tangential, radial) frame at the encounter.
# ----------------------------------------------------------------------------
def soi_radius(mu_a=MU_A, mu_c=MU_C, r_a=R_A):
    """Assist body's sphere-of-influence radius (Laplace)."""
    return r_a * (mu_a / mu_c) ** 0.4


def hohmann_to_assist():
    """Hohmann transfer R_INIT -> R_A. Returns (dv_depart, v_apo, v_circ_A,
    v_inf): the departure burn off the inner circular orbit, the spacecraft's
    tangential arrival speed at R_A, the assist body's circular speed, and the
    excess-speed magnitude |v_sc - v_a| at arrival."""
    a_t = 0.5 * (R_INIT + R_A)
    v_circ_init = np.sqrt(MU_C / R_INIT)
    v_peri = np.sqrt(MU_C * (2.0 / R_INIT - 1.0 / a_t))   # transfer periapsis
    v_apo = np.sqrt(MU_C * (2.0 / R_A - 1.0 / a_t))        # transfer apoapsis
    v_circ_A = np.sqrt(MU_C / R_A)
    return v_peri - v_circ_init, v_apo, v_circ_A, abs(v_circ_A - v_apo)


def turn_angle(v_inf, r_p, mu_a=MU_A):
    """Flyby turn angle [rad] for excess speed v_inf and periapsis radius r_p."""
    return 2.0 * np.arcsin(1.0 / (1.0 + r_p * v_inf ** 2 / mu_a))


def flyby_outcome(r_p, sign=+1):
    """Patched-conic flyby at periapsis r_p. `sign` (+1/-1) picks the flyby side
    (turns v_inf one way or the other -> same energy here, mirrored radial
    component). Returns the post-flyby heliocentric orbit and the assist's
    heliocentric ΔV."""
    _, v_apo, v_cA, v_inf = hohmann_to_assist()
    v_a = np.array([v_cA, 0.0])                 # assist velocity (tangential)
    vinf_in = np.array([v_apo - v_cA, 0.0])     # = -v_inf * t_hat (S/C slower)
    d = sign * turn_angle(v_inf, r_p)
    c, s = np.cos(d), np.sin(d)
    vinf_out = np.array([c * vinf_in[0] - s * vinf_in[1],
                         s * vinf_in[0] + c * vinf_in[1]])
    v_out = v_a + vinf_out
    speed = float(np.linalg.norm(v_out))
    eps = 0.5 * speed ** 2 - MU_C / R_A          # specific orbital energy
    h = R_A * v_out[0]                            # h = r * v_tangential
    a_new = (-MU_C / (2.0 * eps)) if eps < 0 else np.inf
    e = np.sqrt(max(0.0, 1.0 + 2.0 * eps * h ** 2 / MU_C ** 2))
    ra = a_new * (1.0 + e) if (np.isfinite(a_new) and e < 1.0) else np.inf
    rp_orb = a_new * (1.0 - e) if np.isfinite(a_new) else np.nan
    return dict(delta=abs(d), v_out=v_out, vinf_in=vinf_in, vinf_out=vinf_out,
                v_a=v_a, speed=speed, eps=eps, a=a_new, e=e, ra=ra, rp=rp_orb,
                dv_helio=float(np.linalg.norm(vinf_out - vinf_in)))


def plot_concept(out_name="gravity_assist_concept.png"):
    """Concept curves for a single patched-conic flyby from the Hohmann arrival."""
    dv_dep, v_apo, v_cA, v_inf = hohmann_to_assist()
    r_soi = soi_radius()
    r_p = np.linspace(R_BODY, r_soi, 300)
    delta = np.degrees(turn_angle(v_inf, r_p))
    out = [flyby_outcome(rp, +1) for rp in r_p]
    speed = np.array([o["speed"] for o in out])
    ra = np.array([o["ra"] for o in out])
    dvh = np.array([o["dv_helio"] for o in out])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # (1) v∞ circle: reachable heliocentric velocities lie on a circle of
    # radius v∞ about the assist-velocity tip. Show v_a, v_sc_in, and the two
    # turned v_sc_out (lead/trail) at the closest flyby.
    ax = axs[0, 0]
    o, o2 = flyby_outcome(R_BODY, +1), flyby_outcome(R_BODY, -1)
    va, vin, vout = o["v_a"], o["vinf_in"], o["vinf_out"]
    ph = np.linspace(0, 2 * np.pi, 200)
    ax.plot(va[0] + v_inf * np.cos(ph), va[1] + v_inf * np.sin(ph),
            "0.7", lw=1, ls="--")                    # reachable-velocity circle
    ax.annotate("", xy=va, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=2))
    ax.annotate("", xy=va + vin, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
    ax.annotate("", xy=va + vout, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2))
    ax.annotate("", xy=va + o2["vinf_out"], xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#9467bd", lw=2))
    for vv, txt, col, dy in [(va, "v_a", "0.3", 0.03),
                             (va + vin, "v_sc in", "#d62728", -0.05),
                             (va + vout, "out (lead)", "#2ca02c", 0.03),
                             (va + o2["vinf_out"], "out (trail)", "#9467bd", 0.03)]:
        ax.text(vv[0], vv[1] + dy, txt, color=col, fontsize=8, ha="center")
    ax.plot([0], [0], "ko", ms=4)
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, va[0] + v_inf + 0.15)
    ax.set_ylim(-(v_inf + 0.12), v_inf + 0.12)
    ax.set_xlabel("tangential speed [DU/TU]")
    ax.set_ylabel("radial speed [DU/TU]")
    ax.set_title(f"v∞ circle (r_p=R_body, δ={o['delta']*180/np.pi:.0f}°)\n"
                 "all post-flyby v_sc lie on the dashed circle")
    ax.grid(alpha=0.3)

    # (2) turn angle vs flyby periapsis
    ax = axs[0, 1]
    ax.plot(r_p, delta, "k-")
    ax.axvline(R_BODY, color="r", ls=":", label="body surface")
    ax.set_xlabel("flyby periapsis r_p [DU]")
    ax.set_ylabel("turn angle δ [deg]")
    ax.set_title("Turn angle vs flyby periapsis")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (3) heliocentric speed + assist ΔV vs flyby periapsis
    ax = axs[1, 0]
    ax.plot(r_p, speed, "b-", label="post-flyby speed")
    ax.axhline(v_apo, color="0.5", ls="--", label="arrival speed v_apo")
    ax.axhline(v_cA, color="0.7", ls=":", label="assist speed v_a")
    ax.plot(r_p, dvh, "g-", label="assist ΔV = |Δv∞|")
    ax.set_xlabel("flyby periapsis r_p [DU]")
    ax.set_ylabel("speed / ΔV [DU/TU]")
    ax.set_title("Heliocentric speed & free ΔV from the assist")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (4) resulting orbit apoapsis ("reach") vs flyby periapsis
    ax = axs[1, 1]
    ax.plot(r_p, ra, "m-")
    ax.axhline(R_A, color="0.5", ls="--", label="assist orbit R_a")
    ax.set_xlabel("flyby periapsis r_p [DU]")
    ax.set_ylabel("post-flyby apoapsis [DU]")
    ax.set_title("New orbit apoapsis after one assist")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Single gravity assist (patched conic): Hohmann "
        f"{R_INIT:g}→{R_A:g}, μ_a={MU_A:g}\n"
        f"v∞={v_inf:.4f}, SOI={r_soi:.3f} DU, "
        f"δ_max={delta[0]:.0f}°, max apoapsis={np.nanmax(ra):.2f} DU",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")
    print(f"  departure burn (R_INIT circ -> transfer) = {dv_dep:.4f} DU/TU")
    print(f"  arrival v_apo={v_apo:.4f}, assist v_a={v_cA:.4f}, v∞={v_inf:.4f}")
    print(f"  SOI radius = {r_soi:.4f} DU  (flyby r_p swept {R_BODY:g}..{r_soi:.3f})")
    print(f"  closest flyby: δ={delta[0]:.1f}°, speed {v_apo:.3f}->{speed[0]:.3f}, "
          f"apoapsis {R_A:g}->{ra[0]:.3f} DU, free ΔV={dvh[0]:.4f}")


def _prop2(r0, v0, T, n=700, mu=MU_C):
    """Sample a pure two-body (central-body-only) arc from (r0, v0) over time T."""
    def f(t, y):
        rr = (y[0] ** 2 + y[1] ** 2) ** 1.5
        return [y[2], y[3], -mu * y[0] / rr, -mu * y[1] / rr]
    sol = solve_ivp(f, (0.0, T), [r0[0], r0[1], v0[0], v0[1]],
                    t_eval=np.linspace(0.0, T, n), rtol=1e-9, atol=1e-12)
    return sol.y[0], sol.y[1]


def plot_before_after(r_p=R_BODY, out_name="gravity_assist_before_after.png"):
    """Inertial-frame paths: the Hohmann transfer out to the assist body
    (before) and the post-flyby heliocentric orbit (after), for both flyby
    sides. The encounter is placed at the assist body's t=0 position (R_A, 0);
    the flyby is the instantaneous velocity kink in patched conics."""
    th_enc = 0.0
    rhat = np.array([np.cos(th_enc), np.sin(th_enc)])
    that = np.array([-np.sin(th_enc), np.cos(th_enc)])     # prograde tangent
    a_t = 0.5 * (R_INIT + R_A)

    # Before: depart the inner orbit at the transfer periapsis (opposite side),
    # coast half the transfer ellipse to apoapsis at the encounter.
    th_dep = th_enc + np.pi
    rhat_d = np.array([np.cos(th_dep), np.sin(th_dep)])
    that_d = np.array([-np.sin(th_dep), np.cos(th_dep)])
    v_peri = np.sqrt(MU_C * (2.0 / R_INIT - 1.0 / a_t))
    r_dep = R_INIT * rhat_d
    v_dep = v_peri * that_d
    T_trans = np.pi * np.sqrt(a_t ** 3 / MU_C)
    bx, by = _prop2(r_dep, v_dep, T_trans)
    r_enc = R_A * rhat

    # After: apply the flyby, propagate the new heliocentric orbit one period.
    afters = []
    for sign, lab, col in [(+1, "after (lead flyby)", "#2ca02c"),
                           (-1, "after (trail flyby)", "#9467bd")]:
        o = flyby_outcome(r_p, sign)
        v_in = o["v_out"][0] * that + o["v_out"][1] * rhat   # -> inertial
        T_new = (2.0 * np.pi * np.sqrt(o["a"] ** 3 / MU_C)
                 if np.isfinite(o["a"]) else 4.0 * T_trans)
        ax_, ay_ = _prop2(r_enc, v_in, T_new)
        afters.append((ax_, ay_, lab, col, o))

    th = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(R_A * np.cos(th), R_A * np.sin(th), "k--", lw=0.7, alpha=0.5,
            label="Assist orbit")
    ax.plot(R_INIT * np.cos(th), R_INIT * np.sin(th), "b--", lw=0.6, alpha=0.4,
            label="Initial orbit")
    ax.plot(bx, by, "-", color="#1f77b4", lw=2.2, label="Before: Hohmann transfer")
    for ax_, ay_, lab, col, o in afters:
        ax.plot(ax_, ay_, "-", color=col, lw=1.8,
                label=f"{lab}: r_a={o['ra']:.2f}")
    ax.plot([0], [0], "o", color="gold", ms=14, mec="k", label="Central body")
    ax.plot([r_dep[0]], [r_dep[1]], "b^", ms=10, label="Departure")
    ax.plot([r_enc[0]], [r_enc[1]], "o", color="0.4", ms=11)
    ax.plot([r_enc[0]], [r_enc[1]], "r*", ms=15, label="Flyby (assist body)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("x [DU]")
    ax.set_ylabel("y [DU]")
    ax.set_title("Gravity assist — before vs after  "
                 f"(flyby r_p={r_p:g} DU)\n"
                 f"apoapsis {R_A:g} → {afters[0][4]['ra']:.2f} DU")
    fig.tight_layout()
    fig.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")


def plot_config(out_name="gravity_assist_config.png"):
    """Sketch the system geometry at t = 0."""
    th = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(R_A * np.cos(th), R_A * np.sin(th), "k--", lw=0.8, alpha=0.6,
            label="Assist orbit")
    ax.plot(R_INIT * np.cos(th), R_INIT * np.sin(th), "b--", lw=0.8, alpha=0.6,
            label="Initial S/C orbit")
    ax.plot([0], [0], "o", color="gold", ms=15, mec="k", label="Central body")
    ra, _ = assist_state(0.0)
    ax.plot([ra[0]], [ra[1]], "o", color="0.4", ms=10, label="Assist body (t=0)")
    rs, _ = circular_state(R_INIT, THETA_SC0)
    ax.plot([rs[0]], [rs[1]], "b^", ms=9, label="Spacecraft (t=0)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x [DU]")
    ax.set_ylabel("y [DU]")
    ax.set_title("Gravity-assist setup (canonical units)\n"
                 f"μ_c={MU_C:g}, μ_a={MU_A:g}, R_a={R_A:g}, R_init={R_INIT:g}")
    fig.tight_layout()
    fig.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")


if __name__ == "__main__":
    plot_config()
    plot_concept()
    plot_before_after()
