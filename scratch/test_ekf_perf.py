"""Test EKF performance to diagnose hanging."""
import numpy as np
import time as time_module
from datetime import datetime
from scipy.integrate import solve_ivp
from pathlib import Path

from src.input.loader import load_spice_files
from src.model.gravity_field import load_gravity_field
from src.model.dynamics import AccelerationSTMDot, GeneralStateEquationsOfMotion
from src.schemas.gravity import GravityModelConfig
from src.schemas.spacecraft import SpacecraftProperties

def main():
    # Load SPICE
    data_path = Path('data')
    load_spice_files(data_path / 'spice_kernels', data_path / 'spice_kernels/naif0012.tls')

    # Load gravity model
    print("Loading gravity model...")
    grav_model = load_gravity_field(data_path / 'gravity_models/EGM2008.gfc', 2, 0)

    # Create gravity config
    from src.model.constants import SOLARSYSTEMCONSTANTS
    from src.schemas.gravity import SphericalHarmonicsConfig
    
    sh_config = SphericalHarmonicsConfig(
        degree=2,
        order=0,
        model=grav_model
    )
    gravity_config = GravityModelConfig(
        gp=SOLARSYSTEMCONSTANTS.EARTH.GP,
        spherical_harmonics=sh_config,
        use_analytic_jacobian=False,
        use_approx_jacobian=True,
        jacobian_approx_eps=1e-6  # Default value
    )

    # Create spacecraft
    spacecraft = SpacecraftProperties()

    # Create dynamics
    print("Creating dynamics...")
    accel = AccelerationSTMDot(
        gravity_config=gravity_config,
        spacecraft=spacecraft,
    )
    dynamics = GeneralStateEquationsOfMotion(accel)

    # Initial state
    time_et = 758548919.184310
    pos_vec = np.array([1.995678851849e+06, 1.050059496737e+07, -6.135147692071e+06])
    vel_vec = np.array([-3.490162331222e+03, 2.736160078363e+03, 3.499447006340e+03])

    # State + STM initial conditions
    stm_initial = np.eye(6)
    y0 = np.zeros(42)
    y0[0:6] = np.concatenate([pos_vec, vel_vec])
    y0[6:42] = stm_initial.flatten()

    # Test a single derivative call
    print("Testing state_stm_time_derivative...")
    start = time_module.time()
    result = dynamics.state_stm_time_derivative(time_et, y0)
    elapsed = time_module.time() - start
    print(f"  One derivative call: {elapsed*1000:.2f} ms")

    # Test a short propagation (60 seconds)
    dt = 60.0
    print(f"Testing short integration (60s)...")
    start = time_module.time()
    sol = solve_ivp(
        fun=dynamics.state_stm_time_derivative,
        t_span=[time_et, time_et + dt],
        y0=y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-12,
    )
    elapsed = time_module.time() - start
    print(f"  60s integration: {elapsed:.2f}s, {sol.nfev} function evaluations")
    print(f"  Time per evaluation: {elapsed/sol.nfev*1000:.2f} ms")

    # Test 10 minute propagation
    dt = 600.0
    print(f"Testing longer integration (10 min)...")
    start = time_module.time()
    sol = solve_ivp(
        fun=dynamics.state_stm_time_derivative,
        t_span=[time_et, time_et + dt],
        y0=y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-12,
    )
    elapsed = time_module.time() - start
    print(f"  10min integration: {elapsed:.2f}s, {sol.nfev} function evaluations")

    # Estimate full EKF run time
    # 247 measurements over 6 hours, each requires some propagation
    n_measurements = 247
    total_time = 6 * 3600  # 6 hours in seconds
    avg_gap = total_time / n_measurements  # ~87 seconds per measurement
    
    # Estimate based on 60s run
    est_time = elapsed * (total_time / dt) * n_measurements / n_measurements
    print(f"  Estimated full EKF run: {est_time/60:.1f} minutes")
    
    print("SUCCESS")

if __name__ == "__main__":
    main()
