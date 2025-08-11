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
```

## Installation

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage

To run the simulation, execute the `simulation.py` file. This will initialize the rocket's state and compute its trajectory based on the defined dynamics.

```bash
python src/simulation.py
```

## Dependencies

This project requires the following Python packages:

- `numpy`: For numerical computations.
- `astropy`: For handling physical quantities and units.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.