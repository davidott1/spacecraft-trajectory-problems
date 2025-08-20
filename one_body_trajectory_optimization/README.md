# One-Body Trajectory Optimization

This project simulates the optimal trajectory of a free-body. The spacecraft dynamics is free or uses no external acceleration. The optimal control law uses an indirect method, leveraging the Hamiltonian. The flight time is fixed and boundary conditions are fixed. The spacecraft travels from a fixed position and velocity to a fixed position and velocity. 

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
argparse==1.4.0
astropy==6.0.1
matplotlib==3.9.4
numpy==1.26.4
pytest==8.3.2
scipy==1.13.1
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

## Derivation

Of course. Here is the derivation of the optimal control problem using the indirect method.

The optimal control law is found by minimizing the Hamiltonian. For this problem, the optimal control inputs, `Gamma_x*(t)` and `Gamma_y*(t)`, are equal to the negative of their corresponding velocity co-states:
> `Gamma_x*(t) = -lambda_vx(t)`
> `Gamma_y*(t) = -lambda_vy(t)`

The full solution requires solving a system of eight ordinary differential equations (the state and co-state equations) subject to boundary conditions, which constitutes a two-point boundary value problem (TPBVP).

---
## Hamiltonian Formulation
The first step in the indirect method is to define the **Hamiltonian**, `H`, which combines the cost function and the system dynamics.

* **State Vector** `x`: `[r_x, r_y, v_x, v_y]^T`
* **Control Vector** `u`: `[Gamma_x, Gamma_y]^T`
* **Cost Integrand** `L`: `(1/2) * (Gamma_x^2 + Gamma_y^2)`
* **Co-state Vector** `lambda`: `[lambda_rx, lambda_ry, lambda_vx, lambda_vy]^T`

The Hamiltonian is defined as `H = L + lambda^T * f(x, u)`, where `f` represents the system dynamics.

> `H = (1/2)*(Gamma_x^2 + Gamma_y^2) + lambda_rx*r_x_dot + lambda_ry*r_y_dot + lambda_vx*v_x_dot + lambda_vy*v_y_dot`

Substituting the system dynamics (`r_x_dot = v_x`, `r_y_dot = v_y`, `v_x_dot = Gamma_x`, `v_y_dot = Gamma_y`):

> `H = (1/2)*(Gamma_x^2 + Gamma_y^2) + lambda_rx*v_x + lambda_ry*v_y + lambda_vx*Gamma_x + lambda_vy*Gamma_y`

---
## Necessary Conditions for Optimality
From the Hamiltonian, we derive the necessary conditions for optimality using **Pontryagin's Minimum Principle**. This involves finding the co-state (adjoint) equations and the optimality condition for the control.

### Co-state (Adjoint) Equations
The co-state dynamics are given by `lambda_dot = -dH/dx`.

* `lambda_rx_dot = -dH/dr_x = 0`
* `lambda_ry_dot = -dH/dr_y = 0`
* `lambda_vx_dot = -dH/dv_x = -lambda_rx`
* `lambda_vy_dot = -dH/dv_y = -lambda_ry`

### Optimality Condition
The optimal control `u*` must minimize the Hamiltonian. This condition is found by setting the partial derivative of the Hamiltonian with respect to the control to zero, `dH/du = 0`.

* `dH/dGamma_x = Gamma_x + lambda_vx = 0`  => **`Gamma_x* = -lambda_vx`**
* `dH/dGamma_y = Gamma_y + lambda_vy = 0`  => **`Gamma_y* = -lambda_vy`**

This result explicitly defines the optimal control inputs in terms of the co-states associated with the velocity components.

---
## The Two-Point Boundary Value Problem (TPBVP)
By substituting the optimal control law back into the state and co-state equations, we get a complete system of first-order ordinary differential equations (ODEs).

### State Equations
1.  `r_x_dot = v_x`
2.  `r_y_dot = v_y`
3.  `v_x_dot = Gamma_x* = -lambda_vx`
4.  `v_y_dot = Gamma_y* = -lambda_vy`

### Co-state Equations
5.  `lambda_rx_dot = 0`
6.  `lambda_ry_dot = 0`
7.  `lambda_vx_dot = -lambda_rx`
8.  `lambda_vy_dot = -lambda_ry`

Solving this system of 8 ODEs requires 8 boundary conditions (e.g., initial and final positions and velocities), forming a TPBVP. The solution yields the optimal trajectories for the states, co-states, and the control history `Gamma(t)`.