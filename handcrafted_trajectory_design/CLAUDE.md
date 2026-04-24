# handcrafted_trajectory_design

Interactive PyQt6 tool for hand-building 2D impulsive-burn trajectories around a
single central body, with optional continuous re-optimization. Main app:
[main.py](main.py). Numba-accelerated Lambert kernels:
[lambert_numba.py](lambert_numba.py).

## Run

```
python main.py
```

Requires: `PyQt6`, `numpy`, `scipy`, `numba` (optional — falls back to pure
Python if missing).

## Units & constants (canonical)

- 1 DU = 6371 km, 1 TU ≈ 806.4 s, `MU = 1` DU³/TU².
- `GRAVITY_MAG = 1.0` DU/TU² (used only in constant-gravity env).
- `TIME_OF_FLIGHT = 1.5` TU is the default per-segment render TOF.
- `VEL_SCALE = 2.0` is a display-only multiplier: a velocity vector `v` is
  drawn from a point `p` to `p + v * VEL_SCALE`. When reading vectors back
  from the UI (velocity endpoints), divide by `VEL_SCALE`.
- Rotating frame: `ROT_PERIOD_TU = 24 h / 806.4 s`, `OMEGA_ROT = 2π / ROT_PERIOD_TU`,
  +z spin (CCW in y-up world).

Coordinates are y-up world-space. The `QPainter` transform applies
`scale(zoom, -zoom)` to flip y for drawing.

## Scene model

A scene is a list of `trajectories`. Each trajectory is:

```python
{"dots": [QPointF, ...], "segments": [(i_start, i_end, mult), ...]}
```

- `dots` may include shared references to `canvas.tri_center` and
  `canvas.sq_center` (the triangle/square shape centers). Identity (`is`) is
  used to detect these, not value equality — do not copy them.
- `segments` reference indices into `dots`. `mult` is a positive float; segment
  TOF = `render_tof * mult`. An inserted midpoint splits `mult` into `k` and
  `N-k` with integer `k` so cumulative time along a trajectory stays consistent.
- Node time along a trajectory is computed by `_traj_node_times` walking
  segments in list order and accumulating `render_tof * mult` from `i_start`
  to `i_end`. This assumes segments appear in traversal order.
- The triangle anchors `t = 0`. The square anchors `t = tf` where `tf` is the
  largest node time at which `sq_center` participates (`_square_time`).

## Modes

Toggled via top-bar buttons; setters re-run the active optimizer and repaint.

- `env_mode`: `"two_body"` (central gravity, earth drawn) vs
  `"constant_gravity"` (uniform `g` pointing -y; earth hidden; Kepler orbits
  suppressed).
- `arc_model_mode`: `"conic"` (Lambert) vs `"parabola"`. In constant-gravity
  env, segments are always parabolic regardless of this toggle.
- `frame_mode`: `"inertial"` vs `"rotating"`. Rotating view rotates world
  points about `earth_center` by `R(-ω t)` at each point's anchor time, and
  maps velocities via `v_rot = R(-ω t)(v_inertial - ω × r)`.
- `optimize_mode`: `None | "energy" | "fuel"`. When set, every geometry-
  changing release re-runs the optimizer via `_run_active_optimizer`.

## Arc computation

- `compute_dynamic_arc` / `compute_arc_velocities`: Lambert (universal-variable
  z-iteration in `lambert_solve`) plus RK4 for rendering.
- `compute_parabolic_arc` / `compute_parabolic_arc_velocities`: closed-form
  parabola under constant `g_vec`. In two-body + parabola mode, `g_vec` points
  from `r0` toward `earth_center` with magnitude `GRAVITY_MAG`.
- `lambert_solve_with_jac`: Lambert velocities with analytic Jacobians w.r.t.
  `(r1, r2, dt)` via the implicit function theorem on the universal-variable
  equation. Used by the BFGS optimizer for the conic model.
- Straight-line fallback is used for degenerate geometries; both the values
  and the returned Jacobians are consistent for that fallback.

## Optimizer (`_optimize_common`)

- Decision vars: `(x_i, y_i)` for every movable (non-shape) black dot that
  participates in at least one segment, plus a single shared TOF scalar.
- Objective: sum over dv-application nodes of either `|dv|^2` (energy) or
  `sqrt(|dv|^2 + ε^2)` (fuel, smoothed with `fuel_eps = 1e-4` so it is
  differentiable at zero for BFGS).
- dv locations: black nodes with both incoming and outgoing segments, plus
  triangle/square whenever they have a velocity vector and an attached
  segment (dv = departure − shape_vel on outgoing, shape_vel − arrival on
  incoming).
- Per-node first-occurrence convention: if a node is the start/end of
  multiple segments, only the first in `all_segments` order is used. Gradient
  code mirrors this.
- Single fused evaluator: `fun_and_grad(x_vec)` computes per-segment
  `(v0, vf, Jacobians)` once and uses them for both the cost and gradient,
  passed to BFGS as `jac=True`. Parabolic and Lambert branches share the
  same accumulation code (`seg_parabola`, `seg_lambert`).
- Chain rule applies `* mult` when converting per-segment `tof_seg`
  derivatives to the shared `tof` variable.
- Solver (preferred, when numba is available): **Levenberg-Marquardt /
  IRLS** in `Canvas._lm_solve`. Exploits the sum-of-squared-residuals
  structure: residual `r` = stacked dv vectors (2 per dv-node), Jacobian
  `J` = ∂r/∂x assembled inside the `lm_eval_batch` njit kernel.
  - Normal equations: `(J^T W J + λ diag(J^T W J)) p = -J^T W r`, solved
    via Cholesky.
  - Energy mode: W = I (Gauss-Newton).
  - Fuel mode: IRLS with weights `w_k = 1/sqrt(|dv_k|^2 + ε^2)` per
    dv-node, automatically rederived from the smoothed-L1 cost.
  - Adaptive damping: `λ ← λ · 0.4` on accept, `λ ← λ · 10` on reject.
    Inner loop tries up to 10 damping growths per iteration.
  - 5-15 outer iterations typical; bails to BFGS if no descent found
    or if any Lambert segment fails (Newton + brentq fallback then
    runs in the BFGS path).
  - Yields 3-15× speedup over batched-numba BFGS, particularly at
    larger N and in fuel mode. Often finds better local minima at
    N≥100 because LM damping handles non-convexity better than
    BFGS's quasi-Newton update from a poor initial Hessian.
- Solver (fallback): `scipy.optimize.minimize(method="BFGS", jac=True,
  options={"gtol": 1e-6})`. Unconstrained. Used when numba is missing,
  M_dv=0, or LM bails (degenerate geometry).
- Warm-start cache `z_cache[i]` persists the converged Lambert universal
  variable `z` per segment across BFGS iterations — subsequent Newton
  solves typically converge in 1–2 iterations.
- After solving, writes positions back into the QPointF dot objects and
  updates `render_tof`.

### Performance notes

Optimizations applied (benchmark: 8 midpoints, two_body + conic):

- Newton iteration on the Lambert universal-variable equation `F(z)=0`,
  with `brentq` as a fallback. Replaces bisection-only search.
- `_stumpff_all(z)` returns `(C, S, C', S')` in one call, sharing a single
  `sqrt` + trig/hyperbolic evaluation.
- Merged objective + gradient into one `fun_and_grad` call (halves Lambert
  work vs. separate `fun` and `jac`).
- Direct numpy in the optimizer hot path — no QPointF construction or
  `.x()/.y()` attribute access per segment evaluation.
- Warm-starting `z` between BFGS iterations.
- `node_incoming_seg` / `node_outgoing_seg` maps computed once outside the
  solver loop (depend only on segment ordering).
- Numba `@njit` kernels for `stumpff_all`, `lambert_z_newton`,
  `lambert_solve_nb`, and `lambert_with_jac_nb` in
  [lambert_numba.py](lambert_numba.py). `main.py` wraps the public
  `lambert_solve` / `lambert_solve_with_jac` names so the Numba path is
  used when available and the pure-Python implementations (including the
  `brentq` fallback) run on failure or when Numba is missing. JIT is
  warmed up on import, so the first UI interaction does not stall.
- Batched evaluator `fun_and_grad_batch` in
  [lambert_numba.py](lambert_numba.py): a single `@njit` call does every
  segment's velocity solve plus the entire dv-node cost+gradient
  accumulation in one pass, eliminating N Python→njit crossings and
  per-segment numpy allocations per BFGS iteration. `_optimize_common`
  builds segment/dv-node metadata arrays once and reuses preallocated
  work buffers across BFGS iterations. Gradients match the non-batched
  path to machine precision (cost bitwise equal, grad ≤1.5e-14 at N=100).
  On Newton failure for any segment, the batched path falls back to the
  Python `fun_and_grad` closure (which uses `brentq`) for that BFGS call.

Measured speedups vs. original (`scipy.optimize.minimize` wall time):

| env / model / cost | N | original | pre-numba | numba-scalar | numba-batch |
|---|---:|---:|---:|---:|---:|
| two_body / conic / energy | 8 | 515 ms | 117 ms | 33 ms | 12 ms |
| two_body / conic / fuel | 8 | 1086 ms | 251 ms | 77 ms | 23 ms |
| two_body / conic / energy | 100 | — | — | ~500 ms | ~460 ms |
| two_body / conic / fuel | 100 | — | — | ~3.2 s | ~940 ms |
| two_body / parabola / fuel | 100 | — | — | ~7.3 s | ~950 ms |

Batch-vs-scalar speedup scales with N:

| N | conic/energy | conic/fuel | parabola/fuel |
|---:|---:|---:|---:|
| 8 | 2.7× | 2.7× | 3.8× |
| 32 | 5.1× | 2.7× | 9.7× |
| 100 | 1.1×¹ | 3.4×¹ | 7.7× |

¹ At N=100 two_body/conic the two paths converge to different local
minima (non-convex problem + different rounding order changes BFGS's
trajectory). Gradients are bitwise identical — this is a BFGS-path
effect, not a correctness bug.

Standalone Lambert microbenchmark:

| | original | pre-numba | with numba |
|---|---:|---:|---:|
| `lambert_solve` | ~215 µs | 76 µs | ~4 µs |
| `lambert_solve_with_jac` | ~407 µs | 172 µs | ~7 µs |

RK4 (`compute_dynamic_arc`) is only used for rendering, never inside the
optimizer loop.

## Interaction (input)

World-space distances are converted from screen pixels via `self.zoom`.

- Trajectory drawing: hold Cmd (Ctrl on non-mac), left-drag on empty space.
  Dots are laid down at `trace_spacing` (1.0 DU) intervals along the drag.
  On press/release near a shape, two-radius snap:
  - inner (`<30 px`): trajectory endpoint becomes the shape center.
  - outer (`30..75 px`): shape added + connector segment.
  - beyond: free dot, no connection.
- Cmd + click near an existing arc (within `25 px`): `_insert_node_on_segment`
  splits the segment at the nearest integer sub-step (only if `mult >= 2`).
- Shift + click two nodes: create a segment linking them (dedup via
  `_segment_exists`). May create a new trajectory if both endpoints are
  shapes.
- X + click: `_delete_segment_at` if click is within `2 px` of an arc and
  `>5 px` from each endpoint; else `_delete_black_node` which also merges
  the two incident segments (`mult = m_in + m_out`) when the node has
  exactly one incoming and one outgoing segment in the same trajectory.
- Plain drag on a black node moves it. Plain drag on a velocity line tip
  (or anywhere along it) rescales the endpoint; grab parameter `t` is
  clamped to `[0.05, 1.0]` to avoid divide-by-zero.
- Plain drag on triangle/square translates the shape and drags its velocity
  endpoint rigidly with it.
- Wheel / trackpad pan via `wheelEvent`; trackpad pinch via `NativeGesture`
  (`_zoom_about` keeps the point under the cursor fixed).

## Rendering notes

- Earth disk drawn only in two-body env.
- In inertial + two-body, full Kepler orbits for triangle and square are
  drawn as static green ellipses.
- In rotating + two-body, those orbits are re-sampled in time
  (`propagate_kepler_period`) and each sample is rotated by its absolute
  time — triangle anchors at `t=0`, square at `t=_square_time()`.
- dv arrows at interior nodes are drawn using `_rotate_vec(dv, t_node)` at
  the node's rotated world anchor.
- Bottom-left overlay shows inertial vs rotating-frame axes at `t = tf`;
  the label above the overlay names the *other* frame.

## Editing rules of thumb

- Preserve identity (`is`) of `tri_center` / `sq_center` — many code paths
  check `dot is self.tri_center` to treat shape nodes specially.
- If you add or remove ways trajectories can be mutated, call
  `_run_active_optimizer()` after the geometry change and then `self.update()`.
- Segment order within `traj["segments"]` matters for `_traj_node_times` and
  for the optimizer's first-occurrence velocity assignment. Insertions must
  keep traversal order (see `_insert_node_on_segment`).
- When adding new arc models, also provide a matching
  `_compute_segment_velocities` path and, if it needs to participate in
  optimization, an analytic gradient branch consistent with the
  first-occurrence convention.
- Velocity vectors are stored as endpoint QPointFs offset from a center;
  always convert with `(end - center) / VEL_SCALE` before using as a
  physical velocity and multiply back for display.
- `render_tof` is both a display parameter and the optimizer's shared TOF
  decision variable — treat it as authoritative after any optimize call.
