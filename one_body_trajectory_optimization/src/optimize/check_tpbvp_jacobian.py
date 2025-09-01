import os
import sys
from pathlib import Path
import numpy as np

# approx_derivative import with fallback for environments missing the public symbol
try:
    from scipy.optimize import approx_derivative  # type: ignore[attr-defined]
except Exception:
    try:
        from scipy.optimize._numdiff import approx_derivative  # type: ignore
    except Exception:
        breakpoint()
        def approx_derivative(fun, x0, method="3-point", rel_step=1e-6):
            x0 = np.asarray(x0, dtype=float)
            f0 = np.asarray(fun(x0), dtype=float)
            n = x0.size
            m = f0.size
            J = np.empty((m, n), dtype=float)
            for j in range(n):
                h = rel_step * max(1.0, abs(x0[j]))
                xp = x0.copy()
                xm = x0.copy()
                xp[j] += h
                xm[j] -= h
                fp = np.asarray(fun(xp), dtype=float)
                fm = np.asarray(fun(xm), dtype=float)
                J[:, j] = (fp - fm) / (2.0 * h)
            return J

# Make 'one_body_trajectory_optimization' importable
repo_root = Path(__file__).resolve().parents[2]
pkg_root  = repo_root / "one_body_trajectory_optimization"
sys.path.insert(0, str(pkg_root))

from src.optimize.two_point_boundary_value_problem import (
    tpbvp_objective_and_jacobian,
    check_jacobian,
)

def pack_decision_state(eq):
    """Pack decision_state in the order expected by tpbvp_objective_and_jacobian."""
    def a1(x):
        x = np.asarray(x, dtype=float)
        return x.reshape(-1)
    return np.hstack([
        a1(eq['time'    ]['o']['pls']),
        a1(eq['pos_vec' ]['o']['pls']),
        a1(eq['vel_vec' ]['o']['pls']),
        a1(eq['copos_vec']['o']['pls']),
        a1(eq['covel_vec']['o']['pls']),
        a1(eq['ham'     ]['o']['pls']),
        a1(eq['time'    ]['f']['mns']),
        a1(eq['pos_vec' ]['f']['mns']),
        a1(eq['vel_vec' ]['f']['mns']),
        a1(eq['copos_vec']['f']['mns']),
        a1(eq['covel_vec']['f']['mns']),
        a1(eq['ham'     ]['f']['mns']),
    ])

def make_default_params():
    # Modes: 'fixed' or 'free'
    def bnd(pls, mns, mode="fixed"):
        return {"pls": np.array(pls, dtype=float), "mns": np.array(mns, dtype=float), "mode": mode}

    equality_parameters = {
        "time":      {"o": bnd([0.0], [0.0]), "f": bnd([10.0], [10.0])},
        "pos_vec":   {"o": bnd([1.0, 0.0], [1.0, 0.0]), "f": bnd([0.0, 1.0], [0.0, 1.0])},
        "vel_vec":   {"o": bnd([0.0, 1.0], [0.0, 1.0]), "f": bnd([0.0, 0.0], [0.0, 0.0])},
        "copos_vec": {"o": bnd([0.1, 0.2], [0.1, 0.2]), "f": bnd([0.0, 0.0], [0.0, 0.0])},
        "covel_vec": {"o": bnd([0.3, 0.4], [0.3, 0.4]), "f": bnd([0.0, 0.0], [0.0, 0.0])},
        "ham":       {"o": bnd([0.0], [0.0]), "f": bnd([0.0], [0.0])},
    }

    optimization_parameters = {
        "min_type": "energy",          # or 'fuel'
        "include_jacobian": True,
    }

    integration_state_parameters = {
        "mass_o": 1.0,
        "include_scstm": True,
        "post_process": False,
    }

    inequality_parameters = {
        "use_thrust_acc_limits": False,
        "use_thrust_acc_smoothing": False,
        "thrust_acc_min": 0.0,
        "thrust_acc_max": 0.0,
        "use_thrust_limits": False,
        "use_thrust_smoothing": False,
        "thrust_min": 0.0,
        "thrust_max": 0.0,
        "k_steepness": 100.0,
    }

    return optimization_parameters, integration_state_parameters, equality_parameters, inequality_parameters

def main():
    optimization_parameters, integration_state_parameters, equality_parameters, inequality_parameters = make_default_params()

    decision_state = pack_decision_state(equality_parameters)

    # Analytic
    opt_an = dict(optimization_parameters)
    opt_an["include_jacobian"] = True
    err_an, jac_an = tpbvp_objective_and_jacobian(
        decision_state,
        opt_an,
        integration_state_parameters,
        equality_parameters,
        inequality_parameters,
    )

    # Numeric FD
    opt_fd = dict(optimization_parameters)
    opt_fd["include_jacobian"] = False

    def f_only(x):
        return tpbvp_objective_and_jacobian(
            x,
            opt_fd,
            integration_state_parameters,
            equality_parameters,
            inequality_parameters,
        )

    jac_fd = approx_derivative(f_only, decision_state, method="3-point", rel_step=1e-6)

    # Compare
    diff = jac_an - jac_fd
    max_abs = np.max(np.abs(diff))
    i_max, j_max = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(diff) / np.maximum(1.0, np.abs(jac_fd))
    max_rel = np.nanmax(rel)

    print("Analytic vs FD Jacobian:")
    print(f"  max |J_an - J_fd| = {max_abs:.3e} at ({i_max},{j_max})")
    print(f"  max rel error     = {max_rel:.3e}")
    print(f"  J_an[{i_max},{j_max}] = {jac_an[i_max,j_max]: .6e}")
    print(f"  J_fd[{i_max},{j_max}] = {jac_fd[i_max,j_max]: .6e}")

    # Full 20x20 diff matrix (analytic - FD), each row on one line
    print("\nFull diff matrix (J_an - J_fd):")
    for r in range(diff.shape[0]):
        print(f"{r}")
        print(" ".join(f"{jac_an[r, c]: .1e}" for c in range(diff.shape[1])))
        print(" ".join(f"{jac_fd[r, c]: .1e}" for c in range(diff.shape[1])))
        print(" ".join(f"{diff[r, c]: .1e}" for c in range(diff.shape[1])))
        print()

    # Optional: directional derivative check
    rng = np.random.default_rng(0)
    v = rng.standard_normal(decision_state.size)
    v /= np.linalg.norm(v) + 1e-16
    eps = 1e-6
    f_plus  = f_only(decision_state + eps*v)
    f_minus = f_only(decision_state - eps*v)
    dd_fd   = (f_plus - f_minus) / (2*eps)
    dd_an   = jac_an @ v
    dd_err  = np.linalg.norm(dd_an - dd_fd, ord=np.inf)
    print(f"  dir-deriv |Jv - FD| = {dd_err:.3e}")

    # Or use the built-in checker
    print("\nRunning built-in check_jacobian()")
    report = check_jacobian(
        decision_state,
        optimization_parameters,
        integration_state_parameters,
        equality_parameters,
        inequality_parameters,
        rel_step=1e-6,
    )
    print(report["worst_index"], report["max_abs_error"], report["max_rel_error"])

if __name__ == "__main__":
    main()