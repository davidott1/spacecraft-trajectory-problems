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

---
## Optimal Control Problem Derivation

The optimal control problem is solved using an indirect method. The objective `J` minimizes fuel and energy, representative as the integral of magnitude `Gamma` or square `Gamma^2` of thrust acceleration, respectively. The one-body dynamics `x_vec_dot` are free from natural acceleration with control is thrust acceleration `Gamma_vec`. The equality conditions or boundary conditions are variable: flight time is fixed or free, as well as final position and velocity. Flight timee is `delta_t = t_f - t_o = t_f`. The initial time is assumed to be `t_o = 0`. Free initial position and velocity is not implemented, but the problem structure is reversible in time. The inequality conditions are variables as well. For minimum energy problems, thrust or thrust acceleration is either unconstrained or less than a maximum. For minimum fuel problems, thrust or thrust acceleration is necessarily less than a maximum. The coordinate system is Cartesian in two dimensions and with respect to an inertial frame.

The problem is summarized:
// ...existing code...
The problem is summarized:

// ...existing code...
The problem is summarized:

| a | b | c |
| :--- | :--- | :--- |
| **Objective** | $J$ | min fuel: $J = \int_{t_o}^{t_f} \Gamma \ dt$ <br> min energy: $J = \int_{t_o}^{t_f} \tfrac{1}{2} \Gamma^2 \ dt$ |
| **Timespan** | $t$ | $t \in [t_o, t_f]$ |
| **State** | $\underline{x}(t)$ | $\underline{x}=[r_x \ \ \ r_y \ \ \ v_x \ \ \ v_y]^T$ |
| **Control** | $\underline{u}(t)$ | $\underline{u}=[\Gamma_x \ \ \ \Gamma_y]^T$ |
| **Dynamics** | $\underline{f}(t,\underline{x},\underline{u})$ | $\underline{f}=[v_x \ \ \ v_y \ \ \ \Gamma_x \ \ \ \Gamma_y]^T$ |
| **Constraints** <br> **Equality** <br> <br> **Inequality** <br> <br> | $\Theta(t)$ <br> $\Psi(t)$ | <br> Initial : $t_o=t_{os} \ \ \ \underline{r}(t_0)={\underline{r}} _{o,s} \ \ \ v(t_0)=v_{os}$ <br> Final : $t_f=t_{fs} \ \ \ r(t_f)=r_{fs} \ \ \ v(t_f)=v_{fs}$ <br> min fuel : $\Gamma(t) \le \Gamma_{\max}$ or $T\le T_{\max}$ <br> min energy : $\underline{\Gamma}(t)\le \Gamma_{\max}$ or $T\le T_{\max}$ or unconstrained |



<!-- Initial : $t_o=t_{o,s},\ r(t_0)=r_{o,s},\ v(t_0)=v_{0,s}$ <br>  |
| **Equality**      | Initial : $t_o=t_{o,s},\ r(t_0)=r_{o,s},\ v(t_0)=v_{0,s}$ <br> 
|                   | Final   : $t_f=t_{f,s},\ r(t_f)=r_{f,s},\ v(t_f)=v_{f,s}$ <br> 
**Inequality** <br> 
min fuel   : $|\Gamma(t)|\le \Gamma_{\max}$ or $T\le T_{\max}$ <br> 
min energy : $|\underline{\Gamma}(t)|\le \Gamma_{\max}$ or $T\le T_{\max}$ or unconstrained | -->


### Constraints

**Equality Conditions**

$$
\begin{aligned}
\text{Initial:} && t_0 &= t_{0,s} \\
&& \mathbf{r}(t_0) &= \mathbf{r}_{0,s} \\
&& \mathbf{v}(t_0) &= \mathbf{v}_{0,s}
\end{aligned}
\qquad \qquad
\begin{aligned}
\text{Final:} && t_f &= t_{f,s} \\
&& \mathbf{r}(t_f) &= \mathbf{r}_{f,s} \\
&& \mathbf{v}(t_f) &= \mathbf{v}_{f,s}
\end{aligned}
$$

**Inequality Conditions**

* **min fuel**: $\lvert\Gamma(t)\rvert \le \Gamma_{\max} \text{ or } T \le T_{\max}$
* **min energy**: $\lvert\Gamma(t)\rvert \le \Gamma_{\max} \text{ or } T \le T_{\max} \text{ or unconstrained}$

---

The optimal control law is found by minimizing the Hamiltonian, taking two forms for min fuel and energy. 

---
### Hamiltonian Formulation
The indirect method means to derive the optimal control law a Hamiltonian `H` must be formed to minimize. The derivatives of the Hamiltonian will provide the necessary, but not sufficient, conditions for a minimum solution. The Hamiltonian is a function of the integrand `L` of the objective `J`, state `x_vec`, co-state `lambda_vec`, dynamics `x_vec_dot`, and control `Gamma_vec`. In particular, the co-state in component form is `lambda_vec = [ lambda_r_x, lambda_r_y, lambda_v_x, lambda_v_y ]^T`.

The Hamiltonian `H` is in general
```
H = L + lambda_vec^T f(x, u)
```
and more specifically in component form, 
```
H = 1/2 (Gamma_x^2 + Gamma_y^2) + lambda_r_x r_x_dot + lambda_r_y r_y_dot + lambda_v_x v_x_dot + lambda_v_y v_y_dot
```
The time-derivative of the state `x_vec_dot` must conform to the dynamics, so `x_vec_dot = f(x_vec,u_vec)`. Substituting the dynamics into the Hamilitonian yields
```
H = 1/2 (Gamma_x^2 + Gamma_y^2) + lambda_r_x v_x + lambda_r_y v_y + lambda_v_x Gamma_x + lambda_v_y Gamma_y
```

---
### Necessary Conditions for Optimality
From the Hamiltonian, we derive the necessary conditions for optimality using Pontryagin's Minimum Principle, deriving the co-state dynamical equations and the optimal control.

#### Co-state Equations
The co-state dynamics are given by `lambda_vec_dot = -dH/dx_vec`.
```
lambda_r_x_dot = -dH/dr_x  ==>  lambda_r_x_dot = 0
lambda_r_y_dot = -dH/dr_y  ==>  lambda_r_y_dot = 0
lambda_v_x_dot = -dH/dv_x  ==>  lambda_v_x_dot = -lambda_r_x
lambda_v_y_dot = -dH/dv_y  ==>  lambda_v_y_dot = -lambda_r_y
```
#### Optimal Control
The optimal control `u_*` must minimize the Hamiltonian. This condition is found by setting the partial derivative of the Hamiltonian with respect to the control to zero, `dH/du = 0`.
```
dH/dGamma_x = 0  ==>  Gamma_x + lambda_v_x = 0  ==>  Gamma_x_* = -lambda_v_x
dH/dGamma_y = 0  ==>  Gamma_y + lambda_v_y = 0  ==>  Gamma_y_* = -lambda_v_y
```
This result explicitly defines the optimal control inputs in terms of the co-states associated with the velocity components.

---
### Two-Point Boundary Value Problem (TPBVP)
By substituting the optimal control law back into the state and co-state equations, we get a complete system of first-order ordinary differential equations (ODEs).

#### State Equations (4)
```
r_x_dot = v_x
r_y_dot = v_y
v_x_dot = -lambda_v_x
v_y_dot = -lambda_v_y
```

#### Co-state Equations (4)
```
lambda_r_x_dot = 0
lambda_r_y_dot = 0
lambda_v_x_dot = -lambda_r_x
lambda_v_y_dot = -lambda_r_y
```

Solving this system of eight ODEs requires eight boundary conditions (e.g., initial and final positions and velocities), forming a TPBVP. The solution yields the optimal trajectories for the states, co-states, and the control.

---
