# High-Fidelity Orbit Propagator

A Python-based high-fidelity orbital mechanics propagation and orbit determination tool for Earth-orbiting spacecraft. This project provides numerical integration of spacecraft trajectories with configurable force models, ground station tracking simulation, comparison against reference ephemerides, and comprehensive visualization.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Examples](#examples)
  - [Custom State Vectors](#custom-state-vectors)
- [Force Models](#force-models)
- [Orbit Determination](#orbit-determination)
  - [Ground Station Tracking](#ground-station-tracking)
  - [Tracker Configuration](#tracker-configuration)
  - [Skyplot Visualization](#skyplot-visualization)
- [Data Sources](#data-sources)
- [Data Download Tools](#data-download-tools)
- [Output](#output)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [References](#references)

## Overview

This propagator integrates spacecraft equations of motion using high-fidelity force models including:

- **Central body gravity** with zonal harmonics (J2, J3, J4) or full spherical harmonic gravity field models (EGM2008)
- **Third-body perturbations** from Sun, Moon, and planets (via SPICE ephemerides)
- **Atmospheric drag** with exponential density model
- **Solar radiation pressure** with cylindrical Earth shadow model
- **General relativity** effects (optional)
- **Solid and ocean tides** (optional)

The tool supports validation against JPL Horizons ephemerides and SGP4/TLE propagation for comparison studies.

## Features

- **Configurable Force Models**: Enable/disable individual perturbations
- **Multiple Initial State Sources**: JPL Horizons, TLE/SGP4, or custom state vectors
- **Impulsive Maneuvers**: Delta-V burns with multi-frame support (J2000, RIC, RTN)
- **Reference Comparisons**: Validate against JPL Horizons or SGP4 propagation
- **Comprehensive Visualization**: 3D trajectories, time series, error plots, skyplots, and covariance plots
- **Frame Conversions**: J2000/GCRS, TEME, IAU_EARTH, and RIC/RTN frames
- **Orbital Element Computation**: Classical and Modified Equinoctial Elements
- **SPICE Integration**: High-accuracy planetary ephemerides via DE440
- **Orbit Determination**: Ground station tracking with topocentric coordinates and Extended Kalman Filter (EKF)
- **Extended Kalman Filter**: Sequential state estimation from ground-based tracking measurements
- **Skyplot Visualization**: Polar plots showing satellite visibility from ground stations
- **Geographic Coordinate Conversion**: Geodetic and geocentric coordinate systems with WGS84 ellipsoid
- **Configuration File Support**: YAML-based configuration files for complex run scenarios

## Project Structure

```
two_body_high_fidelity/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore patterns
│
├── data/                            # Downloaded data (gitignored)
│   ├── ephems/                      # JPL Horizons ephemeris files
│   ├── tles/                        # Two-Line Element files
│   ├── spice_kernels/               # SPICE kernel files (DE440, leap seconds, PCK, BPC)
│   ├── gravity_models/              # Gravity field coefficient files (EGM2008)
│   └── supported_objects.yaml       # Satellite catalog with physical properties
│
├── input/                           # User configuration files
│   ├── configs/                     # Run configuration YAML files
│   ├── state_vectors/               # Custom initial state vector YAML files
│   ├── trackers/                    # Ground station tracker configurations
│   └── maneuvers/                   # Impulsive maneuver specifications
│
├── output/                          # Generated output (gitignored)
│   └── <timestamp>/                 # Timestamped run folders
│       ├── figures/                 # Generated plots
│       └── files/                   # Log files and data exports
│
└── src/                             # Source code
    ├── __init__.py
    ├── main.py                      # Main entry point
    │
    ├── download/                    # Data download utilities
    │   ├── kernels/                 # SPICE kernel downloader
    │   ├── ephems/                  # Ephemeris downloader
    │   ├── tles/                    # TLE downloader
    │   ├── ephems_and_tles/         # Combined downloader
    │   └── gravity_models.py        # Gravity model downloader
    │
    ├── input/                       # Input handling
    │   ├── cli.py                   # Command-line argument parsing
    │   ├── configuration.py         # Configuration management
    │   └── loader.py                # Data loading (SPICE, Horizons, TLEs)
    │
    ├── model/                       # Core physics models
    │   ├── constants.py             # Physical constants and unit conversions
    │   ├── dynamics.py              # Acceleration models (gravity, drag, SRP, relativity, tides)
    │   ├── frame_converter.py       # Reference frame transformations
    │   ├── orbit_converter.py       # Orbital element conversions
    │   ├── time_converter.py        # Time system conversions
    │   └── gravity_field.py         # Spherical harmonic gravity field models
    │
    ├── propagation/                 # Numerical integration
    │   ├── propagator.py            # ODE integration and SGP4 wrapper
    │   └── state_initializer.py     # Initial state determination
    │
    ├── orbit_determination/         # Orbit determination
    │   ├── topocentric.py           # Topocentric coordinate computation
    │   ├── measurement_simulator.py # Measurement simulation
    │   ├── extended_kalman_filter.py # EKF implementation
    │   └── ekf_processor.py         # EKF processing and workflow
    │
    ├── plot/                        # Visualization
    │   ├── plot_3d.py               # 3D trajectory plots
    │   ├── plot_timeseries.py       # Time series plots
    │   ├── plot_groundtrack.py      # Ground track plots
    │   ├── plot_skyplot.py          # Skyplot visualization
    │   ├── plot_covariance.py       # Covariance plots for EKF
    │   ├── plot_generator.py        # Plot generation orchestration
    │   └── utility.py               # Plot helper functions
    │
    ├── schemas/                     # Data schemas
    │   ├── config.py                # Configuration dataclasses
    │   ├── gravity.py               # Gravity model configuration
    │   ├── propagation.py           # Propagation results
    │   ├── spacecraft.py            # Spacecraft properties
    │   ├── state.py                 # State and coordinate representations
    │   └── measurement.py           # Measurement models
    │
    ├── utility/                     # Utilities
    │   ├── logger.py                # Logging configuration
    │   ├── printer.py               # Console output formatting
    │   ├── time_helper.py           # Time parsing utilities
    │   └── tle_helper.py            # TLE parsing utilities
    │
    └── validation/                  # Test suite
        ├── __init__.py
        ├── conftest.py              # Pytest configuration and fixtures
        ├── fixtures/                # Static test data
        ├── test_dynamics.py         # Dynamics model tests
        ├── test_frame_converter.py
        ├── test_orbit_converter.py
        ├── test_propagator.py
        └── test_regression.py       # End-to-end regression tests
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/spacecraft-trajectory-problems.git
   cd spacecraft-trajectory-problems/two_body_high_fidelity
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SPICE kernels**:

   Use the automated download script:
   ```bash
   python -m src.download.kernels
   ```

   Or manually download and place in `data/spice_kernels/`:
   - **DE440 ephemeris**: `de440.bsp` from [NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/)
   - **Leap seconds kernel**: `naif0012.tls` from [NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/)
   - **Planetary constants**: `pck00010.tpc`
   - **Earth orientation**: `earth_latest_high_prec.bpc`

## Quick Start

Propagate the ISS orbit for one day with basic perturbations:

```bash
python -m src.main \
  --initial-state-source jpl_horizons \
  --initial-state-norad-id 25544 \
  --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
  --gravity-harmonics J2 \
  --compare-jpl-horizons
```

## Usage

### Command Line Interface

```
python -m src.main [OPTIONS]

Required Arguments:
  --initial-state-norad-id ID    NORAD catalog ID (e.g., 25544 for ISS)
  --timespan START END           Start and end times in ISO format

Optional Arguments:
  --config FILE                  Load configuration from YAML file (CLI args override config)
  --initial-state-source SOURCE  Initial state source: jpl_horizons, tle, or sv
                                 (default: jpl_horizons)
  --initial-state-filename FILE  YAML file for custom state vector (required if source=sv)
  --gravity-harmonics J2 J3 J4   Enable gravity harmonics (specify which ones)
  --gravity-harmonics-degree-order N M  Use spherical harmonics up to degree N, order M
  --gravity-model-filename FILE  Gravity field coefficient file (e.g., EGM2008)
  --third-bodies sun moon        Enable third-body gravity (specify bodies)
  --drag                         Enable atmospheric drag
  --srp                          Enable solar radiation pressure
  --relativity                   Enable general relativity effects
  --solid-tides                  Enable solid Earth tides
  --ocean-tides                  Enable ocean tides
  --compare-jpl-horizons         Compare results with JPL Horizons ephemeris
  --compare-tle                  Compare results with SGP4/TLE propagation
  --include-orbit-determination  Run Extended Kalman Filter for orbit determination
  --tracker-filename FILE        Ground station tracker configuration file
  --maneuver-filename FILE       Impulsive maneuver YAML file (assumes input/maneuvers/ folder)
```

### Examples

**1. Basic two-body propagation:**
```bash
python -m src.main \
  --initial-state-norad-id 25544 \
  --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00
```

**2. Full force model with all perturbations:**
```bash
python -m src.main \
  --initial-state-source jpl_horizons \
  --initial-state-norad-id 25544 \
  --timespan 2025-10-01T00:00:00 2025-10-08T00:00:00 \
  --gravity-harmonics J2 J3 J4 \
  --third-bodies sun moon \
  --drag \
  --srp \
  --compare-jpl-horizons \
  --compare-tle
```

**3. Using TLE as initial state:**
```bash
python -m src.main \
  --initial-state-source tle \
  --initial-state-norad-id 25544 \
  --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
  --gravity-harmonics J2
```

**4. Custom state vector:**
```bash
python -m src.main \
  --initial-state-source sv \
  --initial-state-filename equatorial.yaml \
  --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
  --gravity-harmonics J2 J3 J4
```

**5. Using configuration file:**
```bash
python -m src.main --config my_scenario.yaml
```

**6. High-fidelity gravity field:**
```bash
python -m src.main \
  --initial-state-norad-id 25544 \
  --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
  --gravity-harmonics-degree-order 50 50 \
  --gravity-model-filename EGM2008.gfc
```

**7. Orbit determination with EKF:**
```bash
python -m src.main \
  --initial-state-norad-id 25544 \
  --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
  --gravity-harmonics J2 \
  --include-orbit-determination \
  --tracker-filename trackers_set1.yaml
```

**8. Propagation with impulsive maneuvers:**
```bash
python -m src.main \
  --initial-state-norad-id 25544 \
  --timespan 2025-10-01T00:00:00 2025-10-02T00:00:00 \
  --gravity-harmonics J2 \
  --maneuver-filename example_hohmann_transfer.yaml
```

### Configuration Files

For complex scenarios, use YAML configuration files instead of long command lines. Configuration files support all command-line options and can be placed in `input/configs/`.

```yaml
# input/configs/iss_high_fidelity.yaml
initial_state_source: jpl_horizons
initial_state_norad_id: 25544
timespan:
  - "2025-10-01T00:00:00"
  - "2025-10-08T00:00:00"

# Force models
gravity_harmonics_degree_order: [50, 50]
gravity_harmonics_filename: EGM2008.gfc
third_bodies: [sun, moon]
include_drag: true
include_srp: true
include_relativity: true

# Comparisons
compare_jpl_horizons: true
compare_tle: false

# Orbit determination
include_orbit_determination: true
tracker_filename: trackers_set1.yaml
include_tracker_skyplots: true

# Numerical integration
atol: 1.0e-12
rtol: 1.0e-10
```

Run with:
```bash
python -m src.main --config iss_high_fidelity.yaml
```

Command-line arguments override configuration file values.

### Custom State Vectors

Create a YAML file in `input/state_vectors/`:

```yaml
# input/state_vectors/equatorial.yaml
frame             : J2000
epoch             : 2025-10-01T00:00:00Z
coordinate_system : cartesian
pos_vec__m        : 7000.0e3, 0.0e0, 0.0e0
vel_vec__m_per_s  :    0.0e0, 7.5e3, 0.0e0
mass__kg          : 1000.0
drag              :
  coeff           : 2.2
  area__m2        : 10.0
srp               :
  coeff           : 1.3
  area__m2        : 10.0
```

## Force Models

### Central Body Gravity

| Model | Description |
|-------|-------------|
| Point Mass | Keplerian two-body gravity |
| J2 | Earth oblateness (dominant perturbation for LEO) |
| J3 | North-south asymmetry |
| J4 | Higher-order oblateness |
| C22, S22 | Ellipticity of the equator (tesseral harmonics) |
| EGM2008 | Full spherical harmonic model (degree/order up to 2190) |

### Third-Body Gravity

Perturbations from celestial bodies using SPICE DE440 ephemerides:
- Sun
- Moon
- Planets (Mars, Jupiter, Saturn, etc.)

### Atmospheric Drag

Exponential density model with rotating atmosphere:
- Reference density: 1.225 kg/m³ (sea level)
- Scale height: 8500 m
- Accounts for Earth rotation

### Solar Radiation Pressure

Cannonball model with:
- Configurable reflectivity coefficient (Cr)
- Cylindrical Earth shadow model
- Inverse-square intensity scaling

### General Relativity

Schwarzschild and Lense-Thirring relativistic corrections:
- Schwarzschild effect (gravitational time dilation)
- Lense-Thirring effect (frame-dragging)

### Solid Earth Tides

Solid body tides due to third-body gravitational effects:
- Deformation of Earth's shape
- Time-varying gravitational potential

### Ocean Tides

Ocean loading effects on Earth's gravitational field:
- Time-varying mass redistribution
- Additional perturbations for LEO satellites

## Impulsive Maneuvers

The propagator supports impulsive Delta-V maneuvers at specified times. Maneuvers are modeled as instantaneous velocity changes and can be specified in multiple reference frames.

### Maneuver Specification

Maneuvers are defined in YAML files placed in `input/maneuvers/`:

```yaml
maneuvers:
  - time_iso_utc: "2025-10-01T06:00:00"
    delta_vel__m_per_s: [0.0, 50.0, 0.0]  # In-track burn
    frame: "RIC"

  - time_iso_utc: "2025-10-01T12:00:00"
    delta_vel__m_per_s: [0.0, 30.0, 0.0]  # Circularization burn
    frame: "RIC"
```

### Reference Frames

- **J2000**: Inertial frame (X, Y, Z components)
- **RIC**: Radial-In-track-Cross-track (orbit-aligned)
  - R: Radial direction (away from Earth center)
  - I: In-track direction (along velocity vector)
  - C: Cross-track direction (perpendicular to orbital plane)
- **RTN**: Radial-Tangential-Normal (orbit-aligned, similar to RIC)

### Implementation

- Maneuvers are applied as instantaneous velocity changes
- Propagation is automatically segmented around maneuver times
- Frame conversions (RIC/RTN → J2000) are handled automatically
- Multiple maneuvers can be specified in sequence

### Example Maneuver Files

The propagator includes example maneuver files:

- `example_hohmann_transfer.yaml`: Two-burn Hohmann transfer
- `example_plane_change.yaml`: Single out-of-plane maneuver


## Orbit Determination

The propagator includes comprehensive orbit determination capabilities for simulating ground station tracking and analyzing satellite visibility.

### Ground Station Tracking

The tool computes topocentric coordinates (azimuth, elevation, range) from ground stations to satellites:

- **Azimuth**: Angle from North (0°) clockwise to East (90°), South (180°), West (270°)
- **Elevation**: Angle above the horizon (0°) to zenith (90°)
- **Range**: Slant distance from ground station to satellite (m)

Additional computed quantities:
- **Azimuth rate** (rad/s): Angular velocity in azimuth
- **Elevation rate** (rad/s): Angular velocity in elevation
- **Range rate** (m/s): Radial velocity (range-rate)

These measurements are fundamental for:
- Orbit determination from tracking data
- Ground station scheduling and operations
- Communication link analysis
- Satellite visibility prediction

### Tracker Configuration

Ground station configurations are defined in YAML files located in `input/trackers/`. Multiple tracker sets are available:

**trackers_set1.yaml** - Standard network:
```yaml
- name: "Canberra DSN"
  position:
    latitude__deg: -35.4
    longitude__deg: 148.98
    altitude__m: 550.0
  performance:
    azimuth_min_max__deg: -180.0, 180.0
    elevation_min_max__deg: 10.0, 90.0
    range_min_max__m: 0.0, 2.5e6
```

**trackers_set2.yaml** - Global coverage:
- Equatorial Tracker
- Northern Tracker (45°N)
- Southern Tracker (45°S)
- Equatorial Eastern Tracker

**trackers_set3.yaml** - Real-world stations:
- Canberra DSN (Australia)
- Svalbard Arctic (Norway)
- Santiago Chile

Each tracker specifies:
- **Position**: Geodetic latitude, longitude, altitude (WGS84)
- **Performance**: Azimuth/elevation limits, maximum range

### Skyplot Visualization

Skyplots provide polar visualizations of satellite visibility from ground stations:

**Features**:
- Polar plot with elevation (radial axis) and azimuth (angular axis)
- Color-coded ground track showing elevation history
- Visibility statistics (percentage, max elevation, min range)
- Field-of-view (FOV) hemisphere visualization
- Time progression indicators
- Automatic visibility filtering based on tracker performance limits

**Example skyplot output**:
```
output/<timestamp>/figures/skyplot_<tracker>_<satellite>.png
```

The skyplot shows:
- **Center**: Zenith (90° elevation)
- **Outer ring**: Horizon (0° elevation)
- **Angular direction**: Azimuth (North=top, East=right, South=bottom, West=left)
- **Color gradient**: Elevation angle (low to high)
- **FOV cone**: Ground station's field-of-view projection

### Extended Kalman Filter (EKF)

The propagator includes a full Extended Kalman Filter implementation for sequential orbit determination from ground-based tracking measurements.

**State Vector**:
- Position and velocity (6 states): `[r_x, r_y, r_z, v_x, v_y, v_z]`
- Optional augmentation: drag coefficient (C_d), SRP coefficient (C_r)

**Measurements**:
- Range, azimuth, elevation
- Range rate, azimuth rate, elevation rate
- Configurable measurement subsets based on tracker capabilities

**Features**:
- Linearized dynamics via state transition matrix (STM)
- Measurement Jacobian for topocentric observations
- Process noise and measurement noise covariance
- Covariance propagation and filtering
- Multiple ground station support

**Usage**:
Enable EKF orbit determination with the `--include-orbit-determination` flag. Results include:
- State estimate time series
- Covariance evolution plots
- Estimation error analysis
- Residual statistics

## Data Sources

### JPL Horizons

Download ephemeris data from [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/):
- Format: CSV with position/velocity in J2000 frame
- Place files in `data/ephems/`
- Naming convention: `horizons_ephem_<norad_id>_<name>_<start>_<end>_<step>.csv`

### Two-Line Elements (TLE)

Download TLEs from [CelesTrak](https://celestrak.org/):
- Format: Standard 2-line or 3-line TLE
- Place files in `data/tles/`

### SPICE Kernels

Required kernels in `data/spice_kernels/`:
- `de440.bsp` - Planetary ephemerides
- `naif0012.tls` - Leap seconds

### Gravity Field Models

For high-fidelity gravity field propagation, download coefficient files:
- `EGM2008.gfc` - Earth Gravitational Model 2008 (spherical harmonics up to degree/order 2190)

Place files in `data/gravity_models/`

## Data Download Tools

The project includes automated download scripts for ephemeris, TLE, SPICE kernels, and gravity field data. These tools can fetch data directly from JPL Horizons, CelesTrak, NAIF, and ICGEM without manual browsing.

### Download SPICE Kernels

```bash
python -m src.download.kernels
```

Downloads all required SPICE kernels:
- DE440 planetary ephemeris (~120 MB)
- Leap seconds kernel (naif0012.tls)
- Planetary constants (pck00010.tpc)
- Earth orientation data (earth_latest_high_prec.bpc)

Files are saved to `data/spice_kernels/`.

### Download Both Ephemeris and TLE

```bash
python -m src.download.ephems_and_tles <norad_id> <start_time> <end_time> [step]
```

Example:
```bash
python -m src.download.ephems_and_tles 25544 "2025-10-01T00:00:00Z" "2025-10-02T00:00:00Z" 1m
```

### Download Ephemeris Only

```bash
python -m src.download.ephems <norad_id> <start_time> <end_time> [step]
```

Example:
```bash
python -m src.download.ephems 25544 "2025-10-01T00:00:00Z" "2025-10-02T00:00:00Z" 5m
```

### Download TLE Only

```bash
python -m src.download.tles <norad_id> <start_time> <end_time>
```

Example:
```bash
python -m src.download.tles 25544 "2025-10-01T00:00:00Z" "2025-10-02T00:00:00Z"
```

### Download Gravity Models

```bash
python -m src.download.gravity_models
```

This downloads the EGM2008 gravity field coefficient file (~100 MB) from ICGEM and saves it to `data/gravity_models/EGM2008.gfc`.

### Download Parameters

| Parameter | Description |
|-----------|-------------|
| `norad_id` | NORAD catalog ID (e.g., 25544 for ISS) |
| `start_time` | Start time in ISO format (YYYY-MM-DDTHH:MM:SSZ) |
| `end_time` | End time in ISO format |
| `step` | Time step for ephemeris (e.g., 1m, 5m, 1h) - ephemeris only |

### Automatic Download During Execution

The main program will automatically prompt to download missing data when needed:
- If you specify `--compare-jpl-horizons` but ephemeris data is missing, it will offer to download
- If you specify `--initial-state-source tle` but TLE data is missing, it will offer to download

### Supported Objects

The propagator currently supports the following satellites, with pre-configured mass and drag/SRP parameters:

| Satellite | NORAD ID | Orbit Type | Description |
|-----------|----------|------------|-------------|
| ISS | 25544 | LEO | International Space Station |
| Terra | 25994 | LEO | Earth observation satellite |
| Aqua | 27424 | LEO | Earth observation satellite |
| GPS BIIRM-5 | 26407 | MEO | NAVSTAR-48 (PRN 31) |
| GPS IIF-2 | 38833 | MEO | NAVSTAR-67 |
| GPS IIF-3 | 39166 | MEO | NAVSTAR-68 |
| GOES-16 | 41866 | GEO | Geostationary weather satellite (GOES-R) |
| GOES-17 | 43226 | GEO | Geostationary weather satellite (GOES-S) |
| GOES-18 | 51850 | GEO | Geostationary weather satellite (GOES-T) |

These objects are defined in `data/supported_objects.yaml` with their physical properties. To add support for additional satellites, update this configuration file with the appropriate NORAD ID, mass, and drag/SRP parameters.

### Downloaded File Locations

- **Ephemeris files**: Saved to `data/ephems/` with naming convention:
  - `horizons_ephem_<norad_id>_<name>_<start>_<end>_<step>.csv`
- **TLE files**: Saved to `data/tles/` with naming convention:
  - `celestrak_tle_<norad_id>_<name>_<start>_<end>.txt`

## Output

Each run creates a timestamped folder in `output/`:

```
output/20251216_143052/
├── figures/
│   ├── 3d_j2000_earth_centered_high_fidelity_iss.png
│   ├── 3d_j2000_earth_centered_jpl_horizons_iss.png
│   ├── 3d_j2000_earth_centered_sgp4_iss.png
│   ├── 3d_iau_earth_high_fidelity_iss.png
│   ├── 3d_j2000_sun_centered_high_fidelity_iss.png
│   ├── timeseries_cart_coe_mee_high_fidelity_iss.png
│   ├── timeseries_cart_coe_mee_jpl_horizons_iss.png
│   ├── timeseries_cart_coe_mee_sgp4_iss.png
│   ├── error_timeseries_high_fidelity_rel_jpl_horizons_iss.png
│   ├── error_timeseries_high_fidelity_rel_sgp4_iss.png
│   ├── groundtrack_high_fidelity_iss.png
│   ├── skyplot_canberra_dsn_high_fidelity_iss.png
│   ├── timeseries_meas_canberra_dsn_high_fidelity_iss.png
│   └── skyplot_svalbard_arctic_high_fidelity_iss.png
└── files/
    └── output.log
```

### Plot Types

| Plot | Description |
|------|-------------|
| 3D Trajectory (J2000) | Position and velocity in 3D inertial frame with Earth wireframe |
| 3D Trajectory (IAU_EARTH) | Body-fixed 3D trajectory with rotating Earth |
| 3D Trajectory (Sun-centered) | Heliocentric trajectory visualization |
| Time Series (Cart/COE/MEE) | Cartesian state, Classical Orbital Elements, and Modified Equinoctial Elements vs time |
| Error Time Series | RIC-frame position/velocity errors and orbital element differences |
| Ground Track | Geographic ground track with latitude/longitude |
| Skyplots | Polar plots showing satellite visibility from ground stations with azimuth/elevation tracks |
| Measurement Time Series | Range, azimuth, and elevation measurements from ground stations vs time |
| Covariance Plots | EKF state covariance evolution (position and velocity uncertainties) |
| Residual Plots | EKF measurement residuals and innovation statistics |

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest src/verification/ -v

# Run specific test module
python -m pytest src/verification/test_dynamics.py -v

# Run with coverage
pip install pytest-cov
python -m pytest src/verification/ --cov=src --cov-report=html
```

### Test Coverage

| Module | Description |
|--------|-------------|
| `test_dynamics.py` | Gravity, drag, SRP acceleration models |
| `test_frame_converter.py` | J2000↔TEME, XYZ↔RIC transformations |
| `test_orbit_converter.py` | Cartesian↔Keplerian conversions |
| `test_propagator.py` | Integration and SGP4 propagation |
| `test_regression.py` | End-to-end regression tests |

## Dependencies

Core dependencies (see `requirements.txt` for versions):

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `scipy` | ODE integration (solve_ivp) |
| `matplotlib` | Plotting and visualization |
| `spiceypy` | SPICE ephemeris interface |
| `sgp4` | SGP4/SDP4 propagation |
| `astropy` | Frame transformations, time handling |
| `pandas` | Data handling for ephemeris files |
| `pyyaml` | Configuration file parsing |
| `pytest` | Testing framework |

## References

### Textbooks

1. Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications* (4th ed.). Microcosm Press.
2. Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods and Applications*. Springer.
3. Curtis, H. D. (2020). *Orbital Mechanics for Engineering Students* (4th ed.). Butterworth-Heinemann.

### Technical References

- [SPICE Toolkit Documentation](https://naif.jpl.nasa.gov/naif/toolkit.html)
- [JPL Horizons System](https://ssd.jpl.nasa.gov/horizons/)
- [CelesTrak](https://celestrak.org/)
- [SGP4 Theory and History](https://celestrak.org/publications/AIAA/2006-6753/)

### Ephemeris Data

- Park, R. S., et al. (2021). "The JPL Planetary and Lunar Ephemerides DE440 and DE441." *The Astronomical Journal*, 161(3), 105.

