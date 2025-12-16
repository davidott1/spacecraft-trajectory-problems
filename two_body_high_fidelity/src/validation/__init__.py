"""
Validation Package
==================

Test suite for the high-fidelity orbit propagator.

Modules:
--------
- test_dynamics        : Unit tests for acceleration models
- test_orbit_converter : Tests for orbital element conversions
- test_frame_converter : Tests for reference frame transformations
- test_propagator      : Integration tests for the propagator

Usage:
------
Run all tests:
  python -m pytest src/validation/ -v

Run a specific test module:
  python -m pytest src/validation/test_dynamics -v

Run a specific test class:
  python -m pytest src/validation/test_dynamics::TestTwoBodyGravity -v

Run a specific test:
  python -m pytest src/validation/test_dynamics::TestTwoBodyGravity::test_sanity_check_point_mass_direction -v

Run with coverage:
  python -m pytest src/validation/ --cov=src --cov-report=html
"""
