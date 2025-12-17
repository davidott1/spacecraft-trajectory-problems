# High-Fidelity Orbit Propagator

A Python-based high-fidelity orbital mechanics propagation tool for Earth-orbiting spacecraft. This project provides numerical integration of spacecraft trajectories with configurable force models, comparison against reference ephemerides, and comprehensive visualization.

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
- [Data Sources](#data-sources)
- [Output](#output)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [References](#references)

## Overview

This propagator integrates spacecraft equations of motion using high-fidelity force models including:

- **Central body gravity** with zonal harmonics (J2, J3, J4)
- **Third-body perturbations** from Sun, Moon, and planets (via SPICE ephemerides)
- **Atmospheric drag** with exponential density model
- **Solar radiation pressure** with cylindrical Earth shadow model

The tool supports validation against JPL Horizons ephemerides and SGP4/TLE propagation for comparison studies.

## Features

- **Configurable Force Models**: Enable/disable individual perturbations
- **Multiple Initial State Sources**: JPL Horizons, TLE/SGP4, or custom state vectors
- **Reference Comparisons**: Validate against JPL Horizons or SGP4 propagation
- **Comprehensive Visualization**: 3D trajectories, time series, and error plots
- **Frame Conversions**: J2000/GCRS, TEME, and RIC/RTN frames
- **Orbital Element Computation**: Classical elements from Cartesian states
- **SPICE Integration**: High-accuracy planetary ephemerides via DE440

## Project Structure

```
two_body_high_fidelity/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
│
├── data/                        # Input data (mostly gitignored)
│   ├── ephems/                  # JPL Horizons ephemeris files
│   ├── tles/                    # Two-Line Element files
│   ├── spice_kernels/           # SPICE kernel files (DE440, leap seconds)
│   └── state_vectors/           # Custom initial state vector YAML files
│
├── output/                      # Generated output (gitignored)
│   └── <timestamp>/             # Timestamped run folders
│       ├── figures/             # Generated plots
│       └── files/               # Log files and data exports
│
└── src/                         # Source code
    ├── __init__.py
    ├── main.py                  # Main entry point
    │
    ├── input/                   # Input handling
    │   ├── cli.py               # Command-line argument parsing
    │   ├── configuration.py     # Configuration management
    │   └── loader.py            # Data loading (SPICE, Horizons, TLEs)
    │
    ├── model/                   # Core physics models
    │   ├── constants.py         # Physical constants and unit conversions
    │   ├── dynamics.py          # Acceleration models (gravity, drag, SRP)
    │   ├── frame_converter.py   # Reference frame transformations
    │   ├── orbit_converter.py   # Orbital element conversions
    │   └── time_converter.py    # Time system conversions
    │
    ├── propagation/             # Numerical integration
    │   ├── propagator.py        # ODE integration and SGP4 wrapper
    │   └── state_initializer.py # Initial state determination
    │
    ├── plot/                    # Visualization
    │   ├── trajectory.py        # Trajectory and error plotting
    │   └── utility.py           # Plot helper functions
    │
    ├── utility/                 # Utilities
    │   ├── logger.py            # Logging configuration
    │   ├── printer.py           # Console output formatting
    │   ├── time_helper.py       # Time parsing utilities
    │   └── tle_helper.py        # TLE parsing utilities
    │
    └── validation/              # Test suite
        ├── __init__.py
        ├── conftest.py          # Pytest configuration and fixtures
        ├── fixtures/            # Static test data
        ├── test_dynamics.py     # Dynamics model tests
        ├── test_frame_converter.py
        ├── test_orbit_converter.py
        ├── test_propagator.py
        └── test_regression.py   # End-to-end regression tests
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
   
   Download and place in `data/spice_kernels/`:
   - **DE440 ephemeris**: `de440.bsp` from [NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/)
   - **Leap seconds kernel**: `naif0012.tls` from [NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/)

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
  --initial-state-source SOURCE  Initial state source: jpl_horizons, tle, or sv
                                 (default: jpl_horizons)
  --initial-state-filename FILE  YAML file for custom state vector (required if source=sv)
  --gravity-harmonics J2 J3 J4   Enable gravity harmonics (specify which ones)
  --third-bodies sun moon        Enable third-body gravity (specify bodies)
  --drag                         Enable atmospheric drag
  --srp                          Enable solar radiation pressure
  --compare-jpl-horizons         Compare results with JPL Horizons ephemeris
  --compare-tle                  Compare results with SGP4/TLE propagation
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

### Custom State Vectors

Create a YAML file in `data/state_vectors/`:

```yaml
# data/state_vectors/equatorial.yaml
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

## Output

Each run creates a timestamped folder in `output/`:

```
output/20251216_143052/
├── figures/
│   ├── 3d_iss_high_fidelity.png
│   ├── 3d_iss_jpl_horizons.png
│   ├── 3d_iss_sgp4.png
│   ├── timeseries_iss_high_fidelity.png
│   ├── timeseries_iss_jpl_horizons.png
│   ├── timeseries_iss_sgp4.png
│   ├── error_timeseries_iss_high_fidelity_relative_to_jpl_horizons.png
│   └── error_timeseries_iss_high_fidelity_relative_to_sgp4.png
└── files/
    └── output.log
```

### Plot Types

| Plot | Description |
|------|-------------|
| 3D Trajectory | Position and velocity in 3D with Earth wireframe |
| Time Series | Position, velocity, and orbital elements vs time |
| Error Plots | RIC-frame position/velocity errors and orbital element differences |

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest src/validation/ -v

# Run specific test module
python -m pytest src/validation/test_dynamics.py -v

# Run with coverage
pip install pytest-cov
python -m pytest src/validation/ --cov=src --cov-report=html
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

