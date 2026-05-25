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

# ---- Central body: Earth (fixed at the origin) ----
MU_C = 1.0                       # central-body grav. parameter [canonical] = Earth

# ---- Assist body: the Moon, on a circular planar orbit ----
# Mass set to the Moon's fraction of Earth: mu_moon / mu_earth ≈ 0.0123.
_MU_EARTH_SI = 398600.4418       # km^3/s^2
_MU_MOON_SI = 4902.800066        # km^3/s^2
R_A = 1.0                        # assist orbit radius -> defines the length unit
MU_A = MU_C * (_MU_MOON_SI / _MU_EARTH_SI)   # ≈ 0.0123 (Moon's fraction of Earth)
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


def _hyperbola_xy(v_inf, r_p, din, dout, mu_a=MU_A, r_max=None, n=500):
    """Flyby hyperbola in the assist-body frame, oriented so its incoming
    asymptote points along `din` and its outgoing asymptote along `dout`
    (so it matches the analytic v∞ turn). Body at the origin. Clipped to r_max."""
    e = 1.0 + r_p * v_inf ** 2 / mu_a
    p = r_p * (1.0 + e)
    nu_inf = np.arccos(-1.0 / e)
    nu = np.linspace(-nu_inf + 0.03, nu_inf - 0.03, n)
    r = p / (1.0 + e * np.cos(nu))
    if r_max is not None:
        m = r <= r_max
        nu, r = nu[m], r[m]
    best = None
    for side in (1.0, -1.0):                       # pick the side matching dout
        P = np.vstack([r * np.cos(nu), side * r * np.sin(nu)])
        di = P[:, 1] - P[:, 0]
        di = di / np.linalg.norm(di)
        ang = np.arctan2(din[1], din[0]) - np.arctan2(di[1], di[0])
        c, s = np.cos(ang), np.sin(ang)
        Pr = np.array([[c, -s], [s, c]]) @ P
        do = Pr[:, -1] - Pr[:, -2]
        do = do / np.linalg.norm(do)
        score = float(np.dot(do, dout))
        if best is None or score > best[0]:
            best = (score, Pr)
    return best[1][0], best[1][1]


def plot_flyby_explainer(r_p=R_BODY, out_name="gravity_assist_explainer.png"):
    """Three linked frames showing how the body-frame hyperbola maps to the
    inertial orbit change, with the two flyby sides color-coded throughout."""
    _, v_apo, v_cA, v_inf = hohmann_to_assist()
    r_soi = soi_radius()
    th = np.linspace(0, 2 * np.pi, 400)

    # Encounter at (R_A, 0): tangential = +y, radial = +x.
    rhat = np.array([1.0, 0.0])
    that = np.array([0.0, 1.0])
    v_a = v_cA * that
    din = (-v_inf * that)                          # incoming v∞ (inertial dir)
    din = din / np.linalg.norm(din)

    cases = []                                     # (sign, label, color, outcome)
    for sign, col in [(+1, "#2ca02c"), (-1, "#9467bd")]:
        o = flyby_outcome(r_p, sign)
        vout = o["vinf_out"][0] * that + o["vinf_out"][1] * rhat
        dout = vout / np.linalg.norm(vout)
        hx, hy = _hyperbola_xy(v_inf, r_p, din, dout, r_max=r_soi)
        # which side of the body the path passes: +x = outer (away from the
        # central body), -x = inner (toward it). Here v∞ is purely retrograde,
        # so the two sides differ in radial direction, not front/back.
        k = int(np.argmin(np.hypot(hx, hy)))
        edge = "outer-side" if hx[k] > 0 else "inner-side"
        cases.append(dict(sign=sign, col=col, o=o, vout=vout, dout=dout,
                          hx=hx, hy=hy, edge=edge))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2))

    # ---- Panel A: assist-body frame (the hyperbola) ----
    # Same v∞ arrows as panel B (red in, green/purple out) — here they are the
    # asymptotes of the actual flyby path, i.e. the *cause* of the v∞ rotation.
    ax = axs[0]
    L = 0.33 * r_soi                              # arrow length
    ax.plot(r_soi * np.cos(th), r_soi * np.sin(th), "0.8", ls="--", lw=1,
            label="sphere of influence")
    ax.add_patch(plt.Circle((0, 0), R_BODY, color="0.4"))
    ax.annotate("", xy=0.11 * that, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="0.6", lw=1.3))
    ax.text(0.01, 0.11, "v_a (body motion)", color="0.6", fontsize=8)
    for c in cases:                              # the two flyby paths
        ax.plot(c["hx"], c["hy"], "-", color=c["col"], lw=2, label=c["edge"])
    # incoming v∞ (shared) — tangent to the incoming asymptote, toward the body
    p_in = 0.85 * r_soi * (-din)                  # up where the S/C comes from
    ax.annotate("", xy=p_in + L * din, xytext=p_in,
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.5))
    ax.text(p_in[0] + 0.012, p_in[1], "v∞ in", color="#d62728", fontsize=10)
    # outgoing v∞ for each side — tangent to that asymptote, away from the body
    for c in cases:
        tail = np.array([c["hx"][-1], c["hy"][-1]])
        ax.annotate("", xy=tail + L * c["dout"], xytext=tail,
                    arrowprops=dict(arrowstyle="-|>", color=c["col"], lw=2.5))
        ax.text(tail[0] + L * c["dout"][0], tail[1] + L * c["dout"][1] - 0.012,
                "v∞ out", color=c["col"], fontsize=9, ha="center")
    ax.plot([0], [0], "o", color="0.4", ms=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35 * r_soi, 1.35 * r_soi)
    ax.set_ylim(-1.35 * r_soi, 1.35 * r_soi)
    ax.set_xlabel("x [DU]")
    ax.set_ylabel("y [DU]")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("1. Assist-body frame (co-moving): path = hyperbola\n"
                 f"S/C speed = v∞={v_inf:.3f}; gravity rotates it by "
                 f"δ={cases[0]['o']['delta']*180/np.pi:.0f}°")

    # ---- Panel B: velocity addition v_sc = v_a + v∞ (the bridge) ----
    ax = axs[1]
    ax.plot(v_a[0] + v_inf * np.cos(th), v_a[1] + v_inf * np.sin(th),
            "0.7", ls="--", lw=1, label="v∞ circle")
    ax.annotate("", xy=v_a, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="0.5", lw=2))
    ax.text(v_a[0] + 0.02, v_a[1], "v_a", color="0.4", fontsize=9)
    ax.annotate("", xy=v_a + (-v_inf * that), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
    ax.text(0.02, v_apo, "v_sc in", color="#d62728", fontsize=9)
    for c in cases:
        tip = v_a + c["vout"]
        ax.annotate("", xy=tip, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c["col"], lw=2))
        ax.annotate("", xy=tip, xytext=v_a,
                    arrowprops=dict(arrowstyle="->", color=c["col"], lw=1.2,
                                    alpha=0.5))
        ax.plot([tip[0]], [tip[1]], "o", color=c["col"], ms=4)
    ax.plot([0], [0], "ko", ms=4)
    ax.set_aspect("equal")
    ax.set_xlabel("v_x [DU/TU]")
    ax.set_ylabel("v_y [DU/TU]")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("2. Velocity bridge: v_sc = v_a + v∞\n"
                 "turning v∞ moves v_sc on the circle")

    # ---- Panel C: inertial frame (before / after) ----
    ax = axs[2]
    a_t = 0.5 * (R_INIT + R_A)
    th_dep = np.pi
    that_d = np.array([-np.sin(th_dep), np.cos(th_dep)])
    v_peri = np.sqrt(MU_C * (2.0 / R_INIT - 1.0 / a_t))
    r_dep = R_INIT * np.array([np.cos(th_dep), np.sin(th_dep)])
    bx, by = _prop2(r_dep, v_peri * that_d, np.pi * np.sqrt(a_t ** 3 / MU_C))
    r_enc = R_A * rhat
    ax.plot(R_A * np.cos(th), R_A * np.sin(th), "k--", lw=0.6, alpha=0.4)
    ax.plot(R_INIT * np.cos(th), R_INIT * np.sin(th), "b--", lw=0.5, alpha=0.3)
    ax.plot(bx, by, "-", color="#1f77b4", lw=2, label="before")
    for c in cases:
        v_in = v_a + c["vout"]          # full heliocentric velocity = v_a + v∞_out
        T_new = 2 * np.pi * np.sqrt(c["o"]["a"] ** 3 / MU_C)
        ax_, ay_ = _prop2(r_enc, v_in, T_new)
        ax.plot(ax_, ay_, "-", color=c["col"], lw=1.7,
                label=f"after {c['edge'].split()[0]} (r_a={c['o']['ra']:.2f})")
        ex = r_enc + r_soi * v_in / np.linalg.norm(v_in)   # zero-SOI exit
        ax.plot([ex[0]], [ex[1]], "o", mfc="none", mec=c["col"], ms=9, mew=1.6)
    # SOI of the assist body (centered on it). The S/C enters along v∞ (the
    # upstream side of the relative velocity = top here), not where the
    # heliocentric line happens to cross.
    ax.add_patch(plt.Circle((r_enc[0], r_enc[1]), r_soi, fill=False,
                            ec="0.5", ls=":", lw=1.3, label="SOI"))
    e0 = r_enc - r_soi * din                              # on SOI, upstream side
    ax.plot([e0[0]], [e0[1]], "r*", ms=11, label="enters along v∞")
    ax.annotate("", xy=e0 + 0.6 * r_soi * din, xytext=e0,
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2))
    ax.plot([0], [0], "o", color="gold", ms=12, mec="k")
    ax.plot([r_dep[0]], [r_dep[1]], "b^", ms=8)
    ax.plot([r_enc[0]], [r_enc[1]], "r*", ms=14)
    ax.set_aspect("equal")
    ax.set_xlabel("x [DU]")
    ax.set_ylabel("y [DU]")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("3. Inertial frame: before & after  (SOI dotted)\n"
                 f"apoapsis {R_A:g} → {cases[0]['o']['ra']:.2f} DU")

    fig.suptitle("Gravity assist across frames — green/purple = the two flyby "
                 "sides (same v∞, opposite turn)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")


def plot_soi_entry(out_name="gravity_assist_soi_entry.png"):
    """Inertial-frame zoom on the encounter showing where the S/C enters the
    SOI: along its velocity *relative* to the body, v∞ = v_sc - v_a, which here
    points retrograde (down). The heliocentric orbit approaches from below, but
    relative to the moving body the S/C slips in from the upstream (top) side —
    the offset between the two is exactly v_a."""
    _, v_apo, v_cA, v_inf = hohmann_to_assist()
    r_soi = soi_radius()
    rhat = np.array([1.0, 0.0])
    that = np.array([0.0, 1.0])
    v_a = v_cA * that
    din = -v_inf * that
    din = din / np.linalg.norm(din)                 # incoming v∞ direction (down)
    r_enc = R_A * rhat

    a_t = 0.5 * (R_INIT + R_A)
    r_dep = R_INIT * np.array([-1.0, 0.0])
    bx, by = _prop2(r_dep, np.sqrt(MU_C * (2.0 / R_INIT - 1.0 / a_t)) *
                    np.array([0.0, -1.0]), np.pi * np.sqrt(a_t ** 3 / MU_C))

    th = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(R_A * np.cos(th), R_A * np.sin(th), "k--", lw=0.6, alpha=0.4)
    ax.plot(bx, by, "-", color="#1f77b4", lw=2, label="before (heliocentric)")
    for sign, col, lab in [(+1, "#2ca02c", "outer"), (-1, "#9467bd", "inner")]:
        o = flyby_outcome(R_BODY, sign)
        vsc = v_a + o["vinf_out"][0] * that + o["vinf_out"][1] * rhat
        ax_, ay_ = _prop2(r_enc, vsc,
                          2 * np.pi * np.sqrt(o["a"] ** 3 / MU_C))
        ax.plot(ax_, ay_, "-", color=col, lw=1.7,
                label=f"after {lab} (r_a={o['ra']:.2f})")
        # zero-SOI exit: where the straight outgoing asymptote (v_sc_out) meets
        # the reference circle — symmetric with the entry, no SOI curving.
        ex = r_enc + r_soi * vsc / np.linalg.norm(vsc)
        ax.plot([ex[0]], [ex[1]], "o", mfc="none", mec=col, ms=11, mew=1.8)
        ax.plot([r_enc[0], ex[0]], [r_enc[1], ex[1]], "-", color=col, lw=0.8,
                alpha=0.5)
    ax.add_patch(plt.Circle((1, 0), r_soi, fill=False, ec="0.5", ls=":",
                            lw=1.5, label="SOI"))
    ax.add_patch(plt.Circle((1, 0), R_BODY, color="0.4"))
    ax.annotate("", xy=r_enc + 0.45 * r_soi * that, xytext=r_enc,
                arrowprops=dict(arrowstyle="-|>", color="0.55", lw=2))
    ax.text(r_enc[0] + 0.015, r_enc[1] + 0.42 * r_soi, "v_a", color="0.5",
            fontsize=9)
    ent = r_enc - r_soi * din                         # upstream side -> top
    ax.plot([ent[0]], [ent[1]], "r*", ms=16, label="enters along v∞")
    ax.annotate("", xy=ent + 0.55 * r_soi * din, xytext=ent,
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.5))
    ax.text(ent[0] + 0.02, ent[1], "v∞ in", color="#d62728", fontsize=10)
    ax.plot([0], [0], "o", color="gold", ms=12, mec="k")
    ax.set_aspect("equal")
    ax.set_xlim(0.6, 1.4)
    ax.set_ylim(-0.45, 0.45)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("x [DU]")
    ax.set_ylabel("y [DU]")
    ax.set_title("Where the S/C enters the SOI (inertial frame)\n"
                 "enters along v∞ = v_sc − v_a (top); orbit approaches from below")
    fig.tight_layout()
    fig.savefig(out_name, dpi=150)
    print(f"Saved {out_name}  (SOI radius {r_soi:.3f} DU)")


def flyby_inclination(r_p):
    """Maximum heliocentric inclination change [rad] from a flyby at periapsis
    r_p: turn v∞ fully OUT of the orbital plane (toward ±z). The turn angle δ is
    fixed by r_p, so the component of v∞_out along v_a (hence the energy) is the
    same as the in-plane case — only the perpendicular kick moves from radial to
    out-of-plane. tan(i) = v∞·sinδ / (v_a − v∞·cosδ)."""
    _, _, v_cA, v_inf = hohmann_to_assist()
    d = turn_angle(v_inf, r_p)
    return np.arctan2(v_inf * np.sin(d), v_cA - v_inf * np.cos(d))


def _prop3(r0, v0, T, n=700, mu=MU_C):
    """Sample a pure two-body (central-body) arc in 3-D from (r0, v0) over T."""
    def f(t, y):
        rr = (y[0] ** 2 + y[1] ** 2 + y[2] ** 2) ** 1.5
        return [y[3], y[4], y[5],
                -mu * y[0] / rr, -mu * y[1] / rr, -mu * y[2] / rr]
    sol = solve_ivp(f, (0.0, T), [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]],
                    t_eval=np.linspace(0.0, T, n), rtol=1e-9, atol=1e-12)
    return sol.y[0], sol.y[1], sol.y[2]


def plot_inclination(r_p=R_BODY, out_name="gravity_assist_inclination.png"):
    """Out-of-plane (inclination) flybys. THREE reference flybys are shown in
    EVERY panel with consistent colors, so the curve's markers match the orbits:
      closest (blue, δ=135°, least inclined, biggest orbit),
      δ=90°   (orange, max out-of-plane KICK),
      peak    (green,  δ≈79°, max inclination ANGLE, smallest orbit).
    Only the +z side is drawn (−z is the mirror). The trade: more inclination
    comes with a smaller post-flyby orbit."""
    _, v_apo, v_cA, v_inf = hohmann_to_assist()
    r_soi = soi_radius()

    BLUE, ORANGE, GREEN = "#1f77b4", "#ff7f0e", "#2ca02c"
    # (name, color, marker, turn angle δ)
    specials = [
        ("closest", BLUE, "o", turn_angle(v_inf, r_p)),
        ("δ=90° (max kick)", ORANGE, "D", np.pi / 2.0),
        ("peak Δi", GREEN, "s", np.arccos(v_inf / v_cA)),
    ]

    def incl(delta):                              # heliocentric inclination
        return np.arctan2(v_inf * np.sin(delta), v_cA - v_inf * np.cos(delta))

    def rp_of(delta):                             # periapsis giving this turn
        return (1.0 / np.sin(delta / 2.0) - 1.0) * MU_A / v_inf ** 2

    rhat = np.array([1.0, 0.0, 0.0])
    that = np.array([0.0, 1.0, 0.0])
    zhat = np.array([0.0, 0.0, 1.0])
    r_enc = R_A * rhat
    v_a = v_cA * that

    def orbit_xyz(delta):                         # +z tilted orbit for this δ
        vy = v_cA - v_inf * np.cos(delta)
        vz = v_inf * np.sin(delta)
        v_sc = np.array([0.0, vy, vz])
        a = -MU_C / (2.0 * (0.5 * (vy * vy + vz * vz) - MU_C / R_A))
        return _prop3(r_enc, v_sc, 2.0 * np.pi * np.sqrt(a ** 3 / MU_C))

    fig = plt.figure(figsize=(13, 11))
    axL = fig.add_subplot(2, 2, 1)
    axM = fig.add_subplot(2, 2, 2, projection="3d")
    axR = fig.add_subplot(2, 2, 3)
    axV = fig.add_subplot(2, 2, 4)

    # ---- Panel 1: Δi vs r_p (full curve) + the three reference flybys ----
    rp = np.linspace(R_BODY, 1.0, 600)
    icurve = np.degrees(np.array([incl(turn_angle(v_inf, x)) for x in rp]))
    valid = rp <= r_soi
    axL.plot(rp[valid], icurve[valid], "k-", lw=2, label="valid (r_p ≤ SOI)")
    axL.plot(rp[~valid], icurve[~valid], "k--", alpha=0.4, label="beyond SOI")
    axL.axvline(r_soi, color="0.6", ls=":")
    for name, col, mk, dd in specials:
        axL.plot([rp_of(dd)], [np.degrees(incl(dd))], mk, color=col, ms=10,
                 label=f"{name}: {np.degrees(incl(dd)):.1f}° (δ={np.degrees(dd):.0f}°)")
    axL.set_xlabel("flyby periapsis r_p [DU]")
    axL.set_ylabel("inclination change Δi [deg]")
    axL.set_title("Δi vs flyby periapsis (out-of-plane turn)")
    axL.legend(fontsize=7)
    axL.grid(alpha=0.3)

    # ---- Panel 2: 3-D orbits (in-plane + the three +z tilted) ----
    o = flyby_outcome(r_p, +1)
    xp, yp, zp = _prop3(r_enc, o["v_out"][0] * that + o["v_out"][1] * rhat,
                        2 * np.pi * np.sqrt(o["a"] ** 3 / MU_C))
    axM.plot(xp, yp, zp, "-", color="0.6", lw=1.4, label="in-plane (i=0)")
    for name, col, mk, dd in specials:
        x, y, z = orbit_xyz(dd)
        axM.plot(x, y, z, "-", color=col, lw=1.5,
                 label=f"{name} (i={np.degrees(incl(dd)):.1f}°)")
    axM.scatter([0], [0], [0], color="gold", s=70, ec="k")
    axM.scatter([r_enc[0]], [0], [0], color="red", marker="*", s=130)
    axM.set_box_aspect((1, 1, 0.45))
    axM.view_init(elev=12, azim=-35)
    axM.set_xlabel("x [DU]"); axM.set_ylabel("y [DU]")
    axM.set_zlabel("z [DU]  (exagg.)")
    axM.legend(fontsize=7, loc="upper left")
    axM.set_title("3-D orbits (+z) — green (peak) is most tilted, smallest")

    # ---- Panel 3: SOI entry/exit (turn plane y–z); exits for the three δ ----
    axR.add_patch(plt.Circle((0, 0), r_soi, fill=False, ec="0.5", ls=":",
                             lw=1.5, label="SOI"))
    axR.add_patch(plt.Circle((0, 0), R_BODY, color="0.4"))
    Larr = 0.4 * r_soi
    din = np.array([-1.0, 0.0])                  # v∞_in in (y, z): retrograde = -y
    ent = -r_soi * din
    axR.plot([ent[0]], [ent[1]], "r*", ms=15, label="enter (v∞ in)")
    axR.annotate("", xy=ent + Larr * din, xytext=ent,
                 arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.5))
    for name, col, mk, dd in specials:
        dout = np.array([-np.cos(dd), np.sin(dd)])     # v∞_out (+z) in (y, z)
        dout = dout / np.linalg.norm(dout)
        ex = r_soi * dout
        axR.plot([ex[0]], [ex[1]], mk, color=col, ms=10, label=f"{name} exit")
        axR.annotate("", xy=ex + Larr * dout, xytext=ex,
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    axR.annotate("", xy=(0.5 * r_soi, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=1.5))
    axR.text(0.5 * r_soi, 0.05 * r_soi, "v_a", color="0.5", fontsize=8)
    axR.plot([0], [0], "o", color="0.4", ms=4)
    axR.set_aspect("equal")
    axR.set_xlim(-1.3 * r_soi, 1.3 * r_soi)
    axR.set_ylim(-1.3 * r_soi, 1.3 * r_soi)
    axR.set_xlabel("y [DU]  (in-plane tangential)")
    axR.set_ylabel("z [DU]  (out of plane)")
    axR.legend(fontsize=7, loc="lower center")
    axR.set_title("SOI entry / +z exits (turn plane y–z)")

    # ---- Panel 4: velocity bridge, v_sc = v_a + v∞ (y–z velocity plane) ----
    ph = np.linspace(0, 2 * np.pi, 200)
    axV.plot(v_cA + v_inf * np.cos(ph), v_inf * np.sin(ph), "0.7", ls="--",
             lw=1, label="v∞ circle")
    axV.annotate("", xy=(v_cA, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=2))
    axV.text(0.5 * v_cA, -0.045, "v_a", color="0.4", fontsize=9, ha="center")
    axV.annotate("", xy=(v_apo, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.5))
    axV.text(0.5 * v_apo, 0.03, "v_sc in", color="#d62728", fontsize=8,
             ha="center")
    for name, col, mk, dd in specials:
        vy = v_cA - v_inf * np.cos(dd)
        vz = v_inf * np.sin(dd)
        axV.annotate("", xy=(vy, vz), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        axV.plot([vy], [vz], mk, color=col, ms=6,
                 label=f"{name} out (i={np.degrees(incl(dd)):.1f}°)")
    axV.plot([0], [0], "ko", ms=4)
    axV.set_aspect("equal")
    axV.set_xlabel("v_y  (in-plane) [DU/TU]")
    axV.set_ylabel("v_z  (out of plane) [DU/TU]")
    axV.legend(fontsize=7, loc="lower left")
    axV.set_title("Velocity bridge — green tilts most (max inclination)")

    fig.suptitle("Inclination flybys — closest (blue) vs δ=90° (orange) vs "
                 "peak (green), consistent everywhere", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")
    for name, col, mk, dd in specials:
        print(f"  {name}: δ={np.degrees(dd):.0f}°, i={np.degrees(incl(dd)):.2f}°, "
              f"r_p={rp_of(dd):.3f}")


def plot_inclination_views(r_p=R_BODY,
                           out_name="gravity_assist_inclination_views.png"):
    """The inclination-flyby orbits as a 3-D view + its three orthographic
    projections. Top-down (xy) hides the tilt but shows the size/energy
    difference; the edge-on views (xz, yz) reveal the inclination."""
    _, v_apo, v_cA, v_inf = hohmann_to_assist()
    rhat = np.array([1.0, 0.0, 0.0])
    that = np.array([0.0, 1.0, 0.0])
    zhat = np.array([0.0, 0.0, 1.0])
    r_enc = R_A * rhat
    v_a = v_cA * that

    def incl(delta):
        return np.degrees(np.arctan2(v_inf * np.sin(delta),
                                     v_cA - v_inf * np.cos(delta)))

    def orbit(delta):                              # +z tilted orbit for this δ
        vy = v_cA - v_inf * np.cos(delta)
        vz = v_inf * np.sin(delta)
        a = -MU_C / (2.0 * (0.5 * (vy * vy + vz * vz) - MU_C / R_A))
        return _prop3(r_enc, np.array([0.0, vy, vz]),
                      2.0 * np.pi * np.sqrt(a ** 3 / MU_C))

    # Same three reference flybys / colors as plot_inclination (+z only).
    o = flyby_outcome(r_p, +1)
    orbits = [("in-plane (i=0)", "0.6", "-")
              + _prop3(r_enc, o["v_out"][0] * that + o["v_out"][1] * rhat,
                       2 * np.pi * np.sqrt(o["a"] ** 3 / MU_C))]
    for name, col, dd in [("closest", "#1f77b4", turn_angle(v_inf, r_p)),
                          ("δ=90°", "#ff7f0e", np.pi / 2.0),
                          ("peak", "#2ca02c", np.arccos(v_inf / v_cA))]:
        orbits.append((f"{name} (i={incl(dd):.1f}°)", col, "-") + orbit(dd))

    fig = plt.figure(figsize=(12, 11))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    axxy = fig.add_subplot(2, 2, 2)
    axxz = fig.add_subplot(2, 2, 3)
    axyz = fig.add_subplot(2, 2, 4)
    for lab, col, ls, x, y, z in orbits:
        ax3.plot(x, y, z, ls, color=col, lw=1.3, label=lab)
        axxy.plot(x, y, ls, color=col, lw=1.3, label=lab)
        axxz.plot(x, z, ls, color=col, lw=1.3)
        axyz.plot(y, z, ls, color=col, lw=1.3)

    ax3.scatter([0], [0], [0], color="gold", s=60, ec="k")
    ax3.scatter([r_enc[0]], [0], [0], color="red", marker="*", s=110)
    ax3.set_box_aspect((1, 1, 0.45))
    ax3.view_init(elev=12, azim=-35)
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z (exagg.)")
    ax3.set_title("3-D view")
    ax3.legend(fontsize=6, loc="upper left")

    for ax2, fx, fy, xl, yl, ttl, eq in [
            (axxy, r_enc[0], 0.0, "x [DU]", "y [DU]",
             "xy (top-down) — tilt hidden, size differs", True),
            (axxz, r_enc[0], 0.0, "x [DU]", "z [DU]",
             "xz (edge-on) — z not to scale", False),
            (axyz, 0.0, 0.0, "y [DU]", "z [DU]",
             "yz (edge-on) — z not to scale", False)]:
        ax2.plot([0], [0], "o", color="gold", ms=11, mec="k")
        ax2.plot([fx], [fy], "r*", ms=13)
        if eq:
            ax2.set_aspect("equal")
        ax2.set_xlabel(xl); ax2.set_ylabel(yl)
        ax2.set_title(ttl); ax2.grid(alpha=0.3)
    axxy.legend(fontsize=6, loc="upper left")

    fig.suptitle("Inclination flybys — 3-D view and orthographic projections",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")


# ----------------------------------------------------------------------------
# Resonant multi-flyby analysis (2:1 resonance POC)
# After each GA, the s/c has a new orbit. For 2:1 resonance, it should complete
# 2 orbits while the assist body completes 1, meeting it again for the next GA.
# Target: a_sc = R_A / 2^(2/3) ≈ 0.63 for 2:1 resonance.
# ----------------------------------------------------------------------------

def find_resonance_rp(target_sma=None, v_inf=None, verbose=True):
    """Search for r_p that produces the target SMA after a single flyby.
    For 2:1 resonance: a_sc = R_A / 2^(2/3) ≈ 0.6299.
    Returns (r_p, achieved_sma, outcome_dict)."""
    if target_sma is None:
        target_sma = R_A / (2.0 ** (2.0 / 3.0))  # 2:1 resonance SMA
    if v_inf is None:
        _, _, _, v_inf = hohmann_to_assist()

    r_soi = soi_radius()
    r_p_vals = np.linspace(R_BODY, r_soi * 0.99, 200)
    sma_vals = []
    outcomes = []

    for r_p in r_p_vals:
        o = flyby_outcome(r_p, +1)
        outcomes.append(o)
        if np.isfinite(o["a"]):
            sma_vals.append(o["a"])
        else:
            sma_vals.append(np.inf)

    sma_vals = np.array(sma_vals)
    # Find closest to target SMA
    idx = np.argmin(np.abs(sma_vals - target_sma))

    if verbose:
        finite_smas = sma_vals[np.isfinite(sma_vals)]
        print(f"  Target SMA: {target_sma:.4f} (2:1 resonance)")
        if len(finite_smas) > 0:
            print(f"  Achievable SMA range: {np.min(finite_smas):.4f} to {np.max(finite_smas):.4f}")
        print(f"  Best match: r_p={r_p_vals[idx]:.4f}, a={sma_vals[idx]:.4f}")

    return r_p_vals[idx], sma_vals[idx], outcomes[idx]


def encounter_time_2to1(a_sc, period_ratio=0.5):
    """Time for next encounter in 2:1 resonance: s/c does 2 orbits, assist does 1.
    Returns the time when they next meet."""
    # Period of assist body at R_A
    p_assist = 2.0 * np.pi * np.sqrt(R_A ** 3 / MU_C)
    # S/C period at a_sc
    p_sc = 2.0 * np.pi * np.sqrt(a_sc ** 3 / MU_C)
    # For 2:1: p_sc = p_assist / 2, so they meet after one assist period
    # (at which point s/c has done 2 orbits)
    return p_assist


def propagate_multi_flyby(n_flybys=3, r_p=None):
    """Propagate 3 gravity assists with the same r_p (POC).
    Returns list of dicts with positions and inclinations at each encounter.
    Note: this is a simplified POC that assumes encounters at the same inertial
    location with the same r_p; a full solution would find actual resonant times."""
    if r_p is None:
        r_p, achieved_sma, _ = find_resonance_rp(verbose=False)

    # Initial Hohmann trajectory parameters
    dv_dep, v_apo, v_cA, v_inf = hohmann_to_assist()
    rhat = np.array([1.0, 0.0])
    that = np.array([0.0, 1.0])

    encounters = []

    for flyby_num in range(n_flybys):
        # Each "encounter" is at the Moon's orbital position (simplified POC)
        r_enc = R_A * rhat
        v_a = v_cA * that

        # Flyby with same r_p
        o = flyby_outcome(r_p, +1)
        delta_deg = np.degrees(o["delta"])
        v_out_inertial = v_a + o["vinf_out"]
        i_after = np.degrees(flyby_inclination(r_p))

        encounters.append({
            "flyby": flyby_num + 1,
            "r_enc": r_enc.copy(),
            "v_out": v_out_inertial.copy(),
            "a": o["a"],
            "delta_deg": delta_deg,
            "i_deg": i_after,
            "outcome": o
        })

    return encounters


def plot_multi_flyby(n_flybys=6, r_p=None, out_name="gravity_assist_multi_flyby.png"):
    """Plot multi-view: 3D view and top-down xy view showing all orbits and GA legs."""
    if r_p is None:
        r_p, _, _ = find_resonance_rp(verbose=False)

    encounters = propagate_multi_flyby(n_flybys, r_p)

    dv_dep, v_apo, v_cA, v_inf = hohmann_to_assist()
    rhat = np.array([1.0, 0.0])
    that = np.array([0.0, 1.0])
    r_enc = R_A * rhat

    fig = plt.figure(figsize=(16, 7))

    # ---- 3D view (left) ----
    ax3d = fig.add_subplot(121, projection="3d")

    # Background orbits
    th = np.linspace(0, 2*np.pi, 300)
    x_init = R_INIT * np.cos(th)
    y_init = R_INIT * np.sin(th)
    z_init = np.zeros_like(th)
    ax3d.plot(x_init, y_init, z_init, "--", color="#d62728", lw=2, label="initial orbit", alpha=0.7)

    x_moon = R_A * np.cos(th)
    y_moon = R_A * np.sin(th)
    z_moon = np.zeros_like(th)
    ax3d.plot(x_moon, y_moon, z_moon, "-", color="#9467bd", lw=2.5, label="Moon orbit", alpha=0.8)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    linestyles = ["-", "-", "-", "-", "-", "-"]  # all solid
    total_inc = 0.0
    sma_check = []

    # Plot GA legs in 3D — cumulative inclination at every half-revolution
    # All use same post-GA velocity, but rotated around x-axis for tilt
    # Alternate between ascending node (x=1) and descending node (x=-1)
    delta = encounters[0]["outcome"]["delta"]
    v_y_base = v_cA - v_inf * np.cos(delta)  # post-GA tangential speed
    v_z_base = v_inf * np.sin(delta)         # post-GA out-of-plane speed
    a_sc = encounters[0]["a"]

    for i in range(n_flybys):
        # Alternate between ascending node (+x) and descending node (-x)
        x_sign = 1.0 if (i % 2 == 0) else -1.0

        # Rotate the base velocity around x-axis by cumulative inclination angle
        i_rad = np.radians(total_inc)
        cos_i = np.cos(i_rad)
        sin_i = np.sin(i_rad)
        v_y = cos_i * v_y_base - sin_i * v_z_base
        v_z = sin_i * v_y_base + cos_i * v_z_base
        v_out_3d = np.array([0.0, v_y, v_z])
        r_enc_3d = np.array([x_sign * R_A, 0.0, 0.0])

        # Compute SMA from energy at encounter point
        v_mag_sq = np.sum(v_out_3d ** 2)
        E = 0.5 * v_mag_sq - MU_C / np.linalg.norm(r_enc_3d)
        a_computed = -MU_C / (2.0 * E) if E < 0 else np.inf
        sma_check.append(a_computed)

        if np.isfinite(a_sc):
            t_prop = 1.5 * 2.0 * np.pi * np.sqrt(a_sc ** 3 / MU_C)
        else:
            t_prop = 10.0

        x, y, z = _prop3(r_enc_3d, v_out_3d, t_prop, n=500)
        col_idx = i % len(colors)
        ax3d.plot(x, y, z, linestyles[col_idx], color=colors[col_idx], lw=2.5,
                 label=f"Leg {i+1}: i={total_inc+encounters[0]['i_deg']:.1f}°")
        ax3d.scatter([r_enc_3d[0]], [r_enc_3d[1]], [0], s=120, color=colors[col_idx],
                    marker="*", edgecolors="k", linewidths=1)
        total_inc += encounters[0]["i_deg"]

    ax3d.scatter([0], [0], [0], s=100, color="gold", edgecolors="k", linewidths=2)
    ax3d.set_xlabel("x [DU]", fontsize=10)
    ax3d.set_ylabel("y [DU]", fontsize=10)
    ax3d.set_zlabel("z [DU]", fontsize=10)
    ax3d.set_xlim([-1.2, 1.2])
    ax3d.set_ylim([-1.2, 1.2])
    ax3d.set_zlim([-1.2, 1.2])
    ax3d.set_box_aspect([1, 1, 1])  # equal aspect ratio
    ax3d.view_init(elev=25, azim=-45)
    ax3d.set_title("3D view (inclined orbits)", fontsize=11, fontweight="bold")
    ax3d.legend(fontsize=8, loc="upper left")
    ax3d.grid(alpha=0.2)

    # ---- 2D top-down view (right) ----
    ax2d = fig.add_subplot(122)

    # Initial and Moon orbits
    ax2d.plot(x_init, y_init, "--", color="#d62728", lw=2, label="initial orbit", alpha=0.7)
    ax2d.plot(x_moon, y_moon, "-", color="#9467bd", lw=2.5, label="Moon orbit", alpha=0.8)

    # GA legs — cumulative inclination at every half-revolution
    total_inc_2d = 0.0
    delta_inc = encounters[0]["i_deg"]
    for i in range(n_flybys):
        # Alternate between ascending node (+x) and descending node (-x)
        x_sign = 1.0 if (i % 2 == 0) else -1.0

        # Rotate the base velocity around x-axis by cumulative inclination angle
        i_rad = np.radians(total_inc_2d)
        cos_i = np.cos(i_rad)
        sin_i = np.sin(i_rad)
        v_y = cos_i * v_y_base - sin_i * v_z_base
        v_z = sin_i * v_y_base + cos_i * v_z_base
        v_out_3d = np.array([0.0, v_y, v_z])
        r_enc_3d = np.array([x_sign * R_A, 0.0, 0.0])

        if np.isfinite(a_sc):
            t_prop = 1.5 * 2.0 * np.pi * np.sqrt(a_sc ** 3 / MU_C)
        else:
            t_prop = 10.0

        x, y, z = _prop3(r_enc_3d, v_out_3d, t_prop, n=500)
        col_idx = i % len(colors)
        ax2d.plot(x, y, linestyles[col_idx], color=colors[col_idx], lw=2.5,
                 label=f"Leg {i+1}: i={total_inc_2d+delta_inc:.1f}°")
        ax2d.plot(r_enc_3d[0], r_enc_3d[1], "*", color=colors[col_idx], ms=12,
                 markeredgecolor="k", markeredgewidth=0.5)
        total_inc_2d += delta_inc

    ax2d.plot(0, 0, "o", color="gold", ms=10, markeredgecolor="k", markeredgewidth=1)
    ax2d.set_xlabel("x [DU]", fontsize=10)
    ax2d.set_ylabel("y [DU]", fontsize=10)
    ax2d.set_xlim([-1.2, 1.2])
    ax2d.set_ylim([-1.2, 1.2])
    ax2d.set_aspect("equal")
    ax2d.set_title("Top-down view (xy-plane)", fontsize=11, fontweight="bold")
    ax2d.legend(fontsize=8, loc="upper left")
    ax2d.grid(alpha=0.2)

    fig.suptitle(f"Multi-Flyby Trajectory: Initial Orbit → 3 GA Legs → Cumulative Δi={total_inc:.1f}°\n"
                 f"r_p={r_p:.4f} DU",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")

    # Print summary
    print(f"\nMulti-Flyby Trajectory Summary (6 legs at half-rev intervals):")
    print(f"  Initial orbit:  R_INIT = {R_INIT} DU")
    print(f"  Moon orbit:     R_A = {R_A} DU")
    print(f"  Flyby r_p:      {r_p:.4f} DU")
    print(f"\n  Leg | Location | δ (°) | Δi (°) | i_total | a (DU) | a_check (DU)")
    print(f"  ----+----------+-------+--------+---------+--------+----------")
    for i in range(n_flybys):
        location = "asc.node" if (i % 2 == 0) else "desc.node"
        i_total = (i + 1) * encounters[0]["i_deg"]
        print(f"   {i+1}  | {location:8} | {encounters[0]['delta_deg']:5.1f} | {encounters[0]['i_deg']:6.1f} | {i_total:7.1f} | {encounters[0]['a']:.4f} | {sma_check[i]:.4f}")
    print(f"  ----+----------+-------+--------+---------+--------+----------")
    total_inc_final = n_flybys * encounters[0]["i_deg"]
    print(f"  Total accumulated inclination: {total_inc_final:.1f}°")
    print(f"  SMA consistency check: all legs should have same a ✓" if np.allclose(sma_check, sma_check[0], rtol=1e-4) else f"  SMA MISMATCH! ✗")


if __name__ == "__main__":
    plot_concept()
    plot_flyby_explainer()
    plot_soi_entry()
    plot_inclination()
    plot_inclination_views()
    plot_multi_flyby()
