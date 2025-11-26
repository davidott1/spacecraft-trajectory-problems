import numpy as np

from datetime import timedelta
from typing   import Optional

from src.model.time_converter import utc_to_et


def print_results_summary(
  result_horizons      : Optional[dict],
  result_high_fidelity : dict,
) -> None:
  """
  Print a summary of the propagation results.
  
  Input:
  ------
    result_horizons : dict | None
      Horizons ephemeris result.
    result_high_fidelity : dict
      High-fidelity propagation result.
  """
  print("\nResults Summary")
  
  # Print final Cartesian state and classical orbital elements (high-fidelity)
  if result_high_fidelity.get('success'):
    # Calculate final epoch
    epoch_str = "n/a"
    if result_horizons and result_horizons.get('success'):
      final_dt = result_horizons['time_o'] + timedelta(seconds=result_high_fidelity['plot_time_s'][-1])
      try:
        final_et  = utc_to_et(final_dt)
        epoch_str = f"{final_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC ({final_et:.6f} ET)"
      except:
        epoch_str = f"{final_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"

    # Extract final state
    pos_vec_f = result_high_fidelity['state'][0:3, -1]
    vel_vec_f = result_high_fidelity['state'][3:6, -1]
    
    # Extract final COEs
    sma  = result_high_fidelity['coe']['sma'][-1]
    ecc  = result_high_fidelity['coe']['ecc'][-1]
    inc  = np.rad2deg(result_high_fidelity['coe']['inc'][-1])
    raan = np.rad2deg(result_high_fidelity['coe']['raan'][-1])
    argp = np.rad2deg(result_high_fidelity['coe']['argp'][-1])
    ta   = np.rad2deg(result_high_fidelity['coe']['ta'][-1])

    print(f"  Final State (High-Fidelity)")
    print(f"    Epoch : {epoch_str}")
    print(f"    Frame : J2000")
    print(f"    Cartesian State")
    print(f"      Position : {pos_vec_f[0]:>19.12e}  {pos_vec_f[1]:>19.12e}  {pos_vec_f[2]:>19.12e} m")
    print(f"      Velocity : {vel_vec_f[0]:>19.12e}  {vel_vec_f[1]:>19.12e}  {vel_vec_f[2]:>19.12e} m/s")
    print(f"    Classical Orbital Elements")
    print(f"      SMA  : { sma:>19.12e} m")
    print(f"      ECC  : { ecc:>19.12e}")
    print(f"      INC  : { inc:>19.12e} deg")
    print(f"      RAAN : {raan:>19.12e} deg")
    print(f"      ARGP : {argp:>19.12e} deg")
    print(f"      TA   : {  ta:>19.12e} deg")
