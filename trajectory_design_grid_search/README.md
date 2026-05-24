# trajectory_design_grid_search

Porkchop-plot generator for an Earth-departure transfer to the Moon, solved by a
**grid search over multi-revolution Lambert arcs** (`porkchop.py`).

For every (departure time, time-of-flight) cell it sweeps the number of complete
revolutions `N = 0, 1, 2, …` (both Lambert branches for `N ≥ 1`) and keeps the
cheapest total ΔV. The result is a porkchop where each cell is colored by its
best ΔV and annotated with the `N` that achieves it.

## Problem setup

- **Two-body (Kepler) only** — Earth point mass, `μ = 398600.4418 km³/s²`. The
  Moon is a massless target (its gravity is *not* modeled; that machinery exists
  in the file but is commented out).
- **Departure:** circular parking orbit, `R_PARK = 6678 km` (300 km LEO).
- **Target:** the Moon on a planar Keplerian orbit, default circular at
  `R_TGT = 96100 km` (~25% of the lunar distance). `THETA_TGT_0 = 90°` is its
  mean anomaly at epoch. Eccentricity and semi-major axis are tunable (see CLI).
- **ΔV accounting:** departure burn `|v_depart − v_park|` + arrival burn
  `|v_moon − v_arrive|` (rendezvous: match the Moon's position *and* velocity).
  Aim is the Moon's exact position (no standoff — there's no singularity without
  lunar gravity).
- **Reference:** the two-impulse Hohmann ΔV for this geometry is
  `DV_HOHMANN ≈ 4.142 km/s`. The `N=0` minimum reproduces it as a sanity check.
- **Screening:** cells whose cheapest solution exceeds `DV_MAX_SEED = 5.5 km/s`
  are dropped (rendered grey). This cap also sets the colorbar top.

**Grid:** `T_DEP = linspace(0, 2, 41)` hr × `TOF = linspace(2, 200, 100)` hr
(41 × 100 = 4100 cells), run in parallel with `multiprocessing`.

## Usage

```bash
# Batch — writes three porkchop PNGs (lunar ecc = 0.0 / 0.1 / 0.5)
python porkchop.py

# Interactive window
python porkchop.py --interactive
python porkchop.py --interactive --ecc 0.3          # eccentric lunar orbit
python porkchop.py --interactive --sma 60000        # tighter target orbit
python porkchop.py --interactive --ecc 0.2 --sma 110000

python porkchop.py --help
```

| flag | meaning | default |
|------|---------|---------|
| `--interactive`, `-i` | open the interactive window | off (batch) |
| `--ecc` | lunar orbit eccentricity | `0.0` |
| `--sma` | lunar orbit semi-major axis [km] | `96100` (`R_TGT`) |

Batch mode sweeps `ecc ∈ {0.0, 0.1, 0.5}` at the given `--sma`.

## Interactive view

A single window, two panels:

- **Left — porkchop.** Each cell is colored by best ΔV and shows its best `N`
  inside it. The numbers auto-scale as you zoom in (and clip to the axes), so a
  tight zoom is readable.
- **Right — trajectories.** **⌘-click** (or **Ctrl-click**) a cell to draw *all*
  its N-rev Lambert solutions: the conic arcs colored by `N` (turbo), the
  cheapest drawn bold and starred, with a legend listing each
  `N=k short/long: ΔV`. Plus the parking orbit, the lunar orbit, Earth, and the
  departure/arrival points.

## Outputs

- `porkchop_kepler_raw_ecc{0.0,0.1,0.5}.png` — raw-cell porkchops (one square
  per cell, no interpolation; grey = screened/no solution) at 300 DPI, on a
  shared color scale (`min` over all three → `DV_MAX_SEED`), each cell labeled
  with its best `N`.

(Generated images are not version-controlled.)

## "short" vs "long" branches

For `N ≥ 1` a given (departure, arrival, TOF) has **two** transfer orbits:
- **short** — smaller semi-major axis → shorter period → lower apoapsis.
- **long** — larger semi-major axis → longer period → higher apoapsis.

They are the two roots on either side of the minimum-time orbit within the
revolution's universal-variable band; as `N` grows they converge toward that
minimum-time orbit.

## Tuning (constants near the top of `porkchop.py`)

| constant | meaning |
|----------|---------|
| `R_PARK`, `R_TGT` | parking / nominal target radii [km] |
| `THETA_TGT_0` | Moon mean anomaly at epoch [rad] |
| `DV_MAX_SEED` | ΔV screen cap **and** colorbar top [km/s] |
| `T_DEP`, `TOF` | grid axes (departure window / time-of-flight) [hr] |

## Dependencies

`numpy`, `matplotlib`, `numba`, `scipy`, and the sibling project
`../handcrafted_trajectory_design/lambert_numba.py` (imported at load time).
The interactive window needs a GUI backend (tries `macosx`, then `TkAgg`/`Qt`).
