# One-body Trajectory Optimization

This project simulates the optimal trajectory of a free-body. The spacecraft dynamics is free or uses no external acceleration. The optimal control law uses an indirect method, leveraging the Hamiltonian. The flight time is fixed and boundary conditions are fixed. The spacecraft travels from a fixed position and velocity to a fixed position and velocity. 

## Derivation

xxx

## Project Structure

```
one_body_trajectory_optimization/
├── input/
│   └── example/
│       ├── 01.json
│       ├── 02.json
│       ├── 03.json
│       ├── 04.json
│       ├── 05.json
│       ├── 06.json
│       ├── 07.json
│       ├── 08.json
│       ├── 09.json
│       └── 10.json
├── output/
│   └── example/
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

To run the program, change directory to `one_body_trajectory_optimization`:
```
cd ~/github/spacecraft-trajectory-problems/one_body_trajectory_optimization/
```
Execute the following general command
```
python main.py <input_filepath> [<output_folderpath>]
```
or more specifically
```
python main.py input/example/01.json output/example
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

### Example 01: `input/example/01.json`
```
python main.py input/example/01.json output/example
```

### Example 02: `input/example/02.json`
```
python main.py input/example/02.json output/example
```

### Example 03: `input/example/03.json`
```
python main.py input/example/03.json output/example
```

### Example 04: `input/example/04.json`
```
python main.py input/example/04.json output/example
```

### Example 05: `input/example/05.json`
```
python main.py input/example/05.json output/example
```

### Example 06: `input/example/06.json`
```
python main.py input/example/06.json output/example
```

### Example 07: `input/example/07.json`
```
python main.py input/example/07.json output/example
```

### Example 08: `input/example/08.json`
```
python main.py input/example/08.json output/example
```

### Example 09: `input/example/09.json`
```
python main.py input/example/09.json output/example
```

### Example 10: `input/example/10.json`
```
python main.py input/example/10.json output/example
```