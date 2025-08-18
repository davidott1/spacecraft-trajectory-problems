import pytest
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path to allow imports from src
# This assumes tests are run from the project root directory 'one_body_trajectory_optimization'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import main

@pytest.mark.parametrize("example_num", ["01", "02"])
def test_run_example(monkeypatch, example_num):
    """
    Tests running the main script with an example input.
    Checks if the expected output PNG file is created and a success
    message is printed.
    """
     # Temporarily disable plt.show() to prevent tests from hanging
    monkeypatch.setattr(plt, "show", lambda: None)

    # Define file and folder paths
    project_folderpath   = Path(__file__).parent.parent
    test_data_folderpath = project_folderpath / "tests" / "data"
    input_filepath       = test_data_folderpath / f"example/{example_num}.json"
    output_folderpath    = project_folderpath / "output"
    expected_output_file = output_folderpath / f"example_{example_num}_optimal_trajectory.png"

    # Ensure the output file doesn't exist before running the test
    if expected_output_file.exists():
        expected_output_file.unlink()

    # Use monkeypatch to set command-line arguments for main.py
    # This simulates running: python main.py <input_file> <output_folder>
    monkeypatch.setattr(sys, 'argv', ['main.py', str(input_filepath), str(output_folderpath)])

    # Run the main program and assert that it returns True
    assert main() is True

    # Assert that the output file was created
    assert expected_output_file.exists(), f"Output file {expected_output_file} was not created."

    # Optional: Clean up the created file after the test
    if expected_output_file.exists():
        expected_output_file.unlink()