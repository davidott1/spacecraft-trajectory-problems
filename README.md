# Spacecraft Trajectory Problems

A collection of Python-based tools and solvers for spacecraft trajectory analysis, optimization, and estimation. This repository contains multiple projects covering orbital mechanics, optimal control, and orbit determination.

## Projects

### [two_body_high_fidelity](two_body_high_fidelity/)
High-fidelity orbital propagator with comprehensive force models including:
- Spherical harmonic gravity (EGM2008)
- Third-body perturbations (Sun, Moon, planets)
- Atmospheric drag and solar radiation pressure
- Extended Kalman Filter for orbit determination
- Ground station tracking and visibility analysis

### [one_body_trajectory_optimization](one_body_trajectory_optimization/)
Optimal trajectory solver for free-body spacecraft using indirect methods:
- Fuel, energy, and combined energy-fuel minimization
- Hamiltonian-based optimal control with Pontryagin's Minimum Principle
- Fixed and free boundary conditions
- Thrust and thrust-acceleration constraints

### [two_body_thrust_estimation](two_body_thrust_estimation/)
Thrust estimation tools for two-body orbital mechanics.

### [one_body_thrust_estimation](one_body_thrust_estimation/)
Thrust estimation tools for one-body trajectory problems.

### [mimic_sgp4_prop](mimic_sgp4_prop/)
SGP4 propagator validation and comparison tools.

## Installation

Each project has its own dependencies. Navigate to the specific project directory and follow its README for installation instructions.

General workflow:
```bash
cd <project_directory>
pip install -r requirements.txt
```

## Getting Started

1. Choose a project based on your needs (see project descriptions above)
2. Navigate to that project's directory
3. Follow the project-specific README for setup and usage
4. Run examples to verify installation

## Requirements

- Python 3.9 or higher
- Project-specific dependencies (see individual `requirements.txt` files)

## References

- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications* (4th ed.)
- Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models, Methods and Applications*
- Curtis, H. D. (2020). *Orbital Mechanics for Engineering Students* (4th ed.)
