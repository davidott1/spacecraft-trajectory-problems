"""
Integration Tests for CLI Entry Point
======================================

Tests that invoke the CLI via subprocess to verify argument parsing
and end-to-end execution through the command-line interface.

Tests:
------
TestCLI
  - test_basic_propagation : CLI invocation completes for a basic propagation

Usage:
------
  python -m pytest src/verification/integration/test_cli.py -v
"""
import subprocess
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class TestCLI:
  """
  Integration tests for the CLI entry point (python -m src.main).
  Verifies that argument parsing and the full pipeline work when
  invoked as a subprocess.
  """

  def test_basic_propagation(self):
    """
    Basic two-body propagation via CLI completes with exit code 0.

    Equivalent CLI:
      python -m src.main \
        --initial-state-norad-id 25544 \
        --timespan 2025-10-01T00:00:00 2025-10-01T00:30:00
    """
    result = subprocess.run(
      [
        sys.executable, '-m', 'src.main',
        '--initial-state-norad-id', '25544',
        '--timespan', '2025-10-01T00:00:00', '2025-10-01T00:30:00',
      ],
      capture_output = True,
      text           = True,
      cwd            = str(PROJECT_ROOT),
    )

    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    assert "High-Fidelity Model" in result.stdout
    assert "Generate and Save Plots" in result.stdout
