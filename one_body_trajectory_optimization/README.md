# Free-body Trajectory Optimization

This project simulates the optimal trajectory of a free-body. The spacecraft dynamics is free or uses no external acceleration. The optimal control law uses an indirect method, leveraging the Hamiltonian. The flight time is fixed and boundary conditions are fixed. The spacecraft travels from a fixed position and velocity to a fixed position and velocity. 

## Derivation

xxx

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
│   │   └── guesser.py
│   ├── load/
│   │   ├── configurer.py
│   │   ├── parser.py
│   │   ├── processor.py
│   │   └── reader.py
│   ├── model/
│   │   └── dynamics.py
│   ├── optimize/
│   │   └── optimizer.py
│   ├── plot/
│   │   └── final_results.py
│   └── utility/
│       └── bounding_functions.py
├── tests/                        : collection of tests
├── untracked/                    : untracked files for convenience
├── .gitignore                    : contains files and folders to ignore by git
├── main.py                       : main driver
├── README.md                     : documentation
└── requirements.txt              : required external packages
```

## Installation

To install the required external packages, run the following command:
```
pip install -r requirements.txt
```

## Usage

To run the program, change directory to `freebody_trajectory_optimization`:
```
cd ~/github/spacecraft-trajectory-problems/freebody_trajectory_optimization/
```
Execute the following general command
```
python main.py <input_filepath> [<output_folderpath>]
```
or more specifically
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

## Examples

### Example 01: `input/examples/example_01.json`
```
python main.py input/examples/example_01.json output/examples
```

### Example 02: `input/examples/example_02.json`
```
python main.py input/examples/example_02.json output/examples
```

### Example 03: `input/examples/example_03.json`
```
python main.py input/examples/example_03.json output/examples
```

### Example 04: `input/examples/example_04.json`
```
python main.py input/examples/example_04.json output/examples
```

### Example 05: `input/examples/example_05.json`
```
python main.py input/examples/example_05.json output/examples
```

### Example 06: `input/examples/example_06.json`
```
python main.py input/examples/example_06.json output/examples
```

### Example 07: `input/examples/example_07.json`
```
python main.py input/examples/example_07.json output/examples
```

### Example 08: `input/examples/example_08.json`
```
python main.py input/examples/example_08.json output/examples
```

### Example 09: `input/examples/example_09.json`
```
python main.py input/examples/example_09.json output/examples
```

### Example 10: `input/examples/example_10.json`
```
python main.py input/examples/example_10.json output/examples
```