"""
State Initializer Module
========================

Determines the initial state vector from various sources:
  - JPL Horizons ephemeris
  - Celestrak TLE
  - Custom state vector (YAML)
  - Maneuver plan (DecisionState)
"""

from src.state_initializer.initializer import get_initial_state, get_initial_state_from_maneuver_plan
