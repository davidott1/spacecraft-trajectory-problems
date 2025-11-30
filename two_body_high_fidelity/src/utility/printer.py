import numpy as np

from typing   import Optional

from src.model.time_converter import et_to_utc
from src.model.constants      import CONVERTER


def print_results_summary(
  result_high_fidelity : dict,
) -> None:
  """
  Print a summary of the propagation results.
  
  Input:
  ------
    result_high_fidelity : dict
      High-fidelity propagation result.
  """
  print("\nResults Summary")
  
  # Print final Cartesian state and classical orbital elements (high-fidelity)
  if result_high_fidelity.get('success'):
    # Final time of high-fidelity propagation
    time_et_f      = result_high_fidelity['time'][-1]
    time_utc_f_str = f"{et_to_utc(time_et_f)} UTC ({time_et_f:.6f} ET)"

    # Final position and velocity vectors
    pos_vec_f = result_high_fidelity['state'][0:3, -1]
    vel_vec_f = result_high_fidelity['state'][3:6, -1]
    
    # Extract final COEs
    sma  = result_high_fidelity['coe']['sma' ][-1]
    ecc  = result_high_fidelity['coe']['ecc' ][-1]
    inc  = result_high_fidelity['coe']['inc' ][-1] * CONVERTER.DEG_PER_RAD
    raan = result_high_fidelity['coe']['raan'][-1] * CONVERTER.DEG_PER_RAD
    argp = result_high_fidelity['coe']['argp'][-1] * CONVERTER.DEG_PER_RAD
    ta   = result_high_fidelity['coe']['ta'  ][-1] * CONVERTER.DEG_PER_RAD

    print(f"  Final State (High-Fidelity)")
    print(f"    Epoch : {time_utc_f_str}")
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

