# trajectory_design_gravity_assist

Patched-conic gravity-assist (GA) analysis in a canonical, planar Earth–Moon
system. One file, [gravity_assist.py](gravity_assist.py), builds a sequence of
explainer figures that walk from a **single flyby** to a **multi-flyby
inclination-cranking campaign**, and finally to the **v∞-sphere** and
**inclination-floor** views.

The numerics are intentionally simple — two-body propagation per leg, with the
flyby itself collapsed to an instantaneous v∞ rotation (`turn_angle`) — so the
geometry of v∞, turn δ, and resonance is the star, not the integrator.

## Canonical setup

| symbol | value | meaning |
|---|---|---|
| `MU_C` | 1 | central-body μ (Earth, fixed at origin) |
| `R_A` | 1 | assist-body orbit radius (Moon, circular planar) — sets the length unit (DU) |
| `MU_A` | `MU_C · μ_moon/μ_earth ≈ 0.0123` | assist-body μ |
| `R_BODY` | 0.03 | assist-body physical radius (min flyby periapsis) |
| `R_INIT` | 0.5 | spacecraft initial circular orbit radius |
| `N_A`, time unit | `sqrt(MU_C / R_A³) = 1`, `TU = 1` | Moon mean motion / time unit |

Multiply by `(R_moon_km, sqrt(μ_earth/R_moon³))` to dimensionalize.

## What the file computes

Three concept layers, built up in order:

1. **Single patched-conic GA** from a Hohmann arrival. `hohmann_to_assist`
   gives the arrival v∞; `turn_angle(v∞, r_p)` is the GA's only knob;
   `flyby_outcome(r_p, sign)` returns the post-flyby heliocentric orbit
   (a, e, r_a, free Δv contributed by the assist).
2. **Inclination flybys.** `flyby_inclination(r_p)` rotates v∞ out of plane
   instead of in plane. The resulting heliocentric inclination is bounded by
   the v∞ ceiling `arcsin(v∞/v_body)`.
3. **Resonant inclination-cranking campaign.** `build_inclination_sequence`
   designs a sequence of `p:q` resonant orbits that walk inclination from 0 →
   target in equal steps. Each leg's GA cranks v∞ a step around the
   Moon-velocity axis; |v∞| and the period stay fixed (resonance pins SMA),
   so the spacecraft returns to the same encounter point next time.

## Plotter → PNG map

Each plot function writes a PNG into this folder. Run order in
`if __name__ == "__main__"`:

| function | PNG | what it shows |
|---|---|---|
| `plot_concept` | `gravity_assist_concept.png` | 4-panel: v∞ circle, δ(r_p), post-flyby speed + free Δv, post-flyby apoapsis |
| `plot_flyby_explainer` | `gravity_assist_explainer.png` | 3 frames side-by-side: hyperbola in body frame, v_sc = v_a + v∞ bridge, before/after heliocentric orbit (two flyby sides) |
| `plot_soi_entry` | `gravity_assist_soi_entry.png` | inertial zoom on the SOI — S/C enters along v∞, not along the heliocentric line |
| `plot_inclination` | `gravity_assist_inclination.png` | Δi vs r_p + the three reference flybys (closest blue, δ=90° orange, peak green) tied across panels |
| `plot_inclination_views` | `gravity_assist_inclination_views.png` | same three orbits as 3-D + xy/xz/yz projections |
| `plot_multi_flyby` | `gravity_assist_multi_flyby.png` | 3-D and top-down view of the cranking campaign — initial orbit, Hohmann, all resonant legs |
| `plot_floor_trajectories` | `flyby_inclination_floor_trajectories.png` | β=35° inclination-**floor** family across three frames: real flyby hyperbolae (GA-body frame) + inclination vs b-plane roll + heliocentric orbit each one produces |
| `plot_vinf_sphere` | `gravity_assist_vinf_sphere.png` | v∞ sphere with one arrow per leg — cranking lifts v∞ out of plane while |v∞| stays fixed (ends in `plt.show()` for rotation) |

The trailing `plt.show()` in `plot_vinf_sphere` keeps that window open;
`plt.close("all")` between the campaign plots prevents the static figures from
stacking up.

## Vocabulary used in figure titles / commentary

- **v∞** — spacecraft velocity relative to the assist body at SOI. Magnitude
  is conserved by the flyby; only the direction rotates.
- **Turn angle δ** — `2·arcsin(1 / (1 + r_p · v∞² / μ_a))`. Smaller r_p ⇒
  bigger δ. The "free Δv" the assist gives the spacecraft is `|Δv∞|`.
- **Pump** — flyby that changes the orbital energy (a, ε) of the heliocentric
  orbit. Happens when v∞ has an along-track component to flip.
- **Crank** — flyby that changes the inclination of the heliocentric orbit at
  constant energy. Rotates v∞ around the assist-velocity axis (`e_v`).
- **VILM** — *Velocity-Infinity Leveraging Maneuver* (small Δv applied far
  from the body, typically at apoapsis). Doesn't escape on its own; it
  retunes the orbit so the **next** flyby has the right v∞/geometry to pump
  or crank usefully.
- **Resonance p:q** — spacecraft completes p orbits in the time the assist
  body completes q. Pins SMA (`a = R_A · (q/p)^(2/3)`) so the encounter
  repeats at the same point. Used by the cranking campaign.
- **Inclination floor / ceiling** — for a fixed v∞ with declination β out of
  the reference plane, b-plane roll alone reaches `i ∈ [β, 180°−β]`. The
  closer v∞ lies to the equator, the smaller the reachable cone — hence the
  "floor" family illustrated by `plot_floor_trajectories`.

## Usage

```bash
python gravity_assist.py
```

Writes all 8 PNGs into this folder and ends with the v∞-sphere shown
interactively. Console output for each plot summarizes the key numbers
(v∞, δ, free Δv, post-flyby r_a, feasibility per leg, etc.).

No CLI flags — knobs are the module constants near the top of the file
(`R_INIT`, `R_BODY`, `MU_A`) and the `target_inc_deg` / `n_legs` /
`resonance` args of `build_inclination_sequence`.

## Outputs

All PNGs above are written into this folder when the script runs; they are
**not version-controlled**.

There are also two PNGs in this folder that the **script does not produce**:
`flyby_pump_zero_inc_change.png` and `flyby_vilm_pump_crank.png`. They were
generated from interactive sessions exploring the **pump-vs-crank** and
**VILM-unlocks-escape** stories on top of the same machinery (`flyby_outcome`
+ `turn_angle`). They aren't reproducible by re-running `gravity_assist.py`.

## Dependencies

`numpy`, `matplotlib`, `scipy.integrate.solve_ivp` (used for the two-body
propagation in `_prop2` / `_prop3` / `_prop3_state`). No external project
imports.
