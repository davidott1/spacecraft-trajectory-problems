# Rocket Trajectory Simulation

This project simulates the trajectory of a rocket using basic physics principles. It computes the dynamics of the rocket based on its state, including position, velocity, and mass, and integrates these dynamics over time to produce a trajectory.

## Project Structure

```
rocket-trajectory-simulation
├── src
│   ├── rocket_dynamics.py   # Contains the rocket_dynamics function for computing dynamics
│   ├── simulation.py         # Responsible for simulating the rocket trajectory
│   └── types
│       └── index.py         # Defines custom types and data structures
├── requirements.txt          # Lists project dependencies
└── README.md                 # Project documentation
xxx
```

## Installation

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

To run the simulation, execute the `simulation.py` file. This will initialize the rocket's state and compute its trajectory based on the defined dynamics.

```
python3c main.py <input_filepath> [<output_folderpath>]
```

Example
```
python3c main.py input/examples/example_01.json output/examples
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

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.