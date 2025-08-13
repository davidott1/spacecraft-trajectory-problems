# Rocket Trajectory Simulation

This project simulates the trajectory of a rocket using basic physics principles. It computes the dynamics of the rocket based on its state, including position, velocity, and mass, and integrates these dynamics over time to produce a trajectory.

## Project Structure

```
freebody_trajectory_optimization/
├── input/
│   └── examples/
│       ├── example_01.json
│       ├── example_02.json
│       ├── example_03.json
│       ├── example_04.json
│       ├── example_05.json
│       ├── example_06.json
│       ├── example_07.json
│       ├── example_08.json
│       ├── example_09.json
│       └── example_10.json
├── output/
│   └── examples/
├── src/
│   ├── data/
│   ├── initial_guess/
│   ├── load/
│   ├── model/
│   ├── optimize/
│   ├── plot/
│   └── utility/
├── tests/
├── untracked/
├── main.py                   # main driver
├── README.md                 # documentation
└── requirements.txt          # required external packages
xxx
```

## Installation

To install the required external packages, run the following command:
```
pip install -r requirements.txt
```

## Usage

To run the program, execute the following general command
```
python main.py <input_filepath> [<output_folderpath>]
```
An example execution is
```
python main.py input/examples/example_01.json output/examples
```

## Dependencies

This project requires the following Python packages:
```
argparse==1.40
astropy==6.0.1
matplotlib==3.9.4
numpy==1.26.4
scipy== 1.13.1
```