from src.model.time_converter import et_to_utc
from src.model.constants      import CONVERTER

from src.schemas.propagation import PropagationResult


def print_results_summary(
  result_high_fidelity : PropagationResult,
) -> None:
  """
  Print a summary of the propagation results.
  
  Input:
  ------
    result_high_fidelity : PropagationResult
      High-fidelity propagation result object.
      
  Output:
  -------
    None
  """
  title = "Results Summary"
  print("\n" + "-" * len(title))
  print(title)
  print("-" * len(title))
  
  # Print final Cartesian state and classical orbital elements (high-fidelity)
  if result_high_fidelity.success:
    # Final time of high-fidelity propagation
    time_et_f      = result_high_fidelity.time[-1]
    time_utc_f_str = f"{et_to_utc(time_et_f)} UTC / {time_et_f:.6f} ET"

    # Final position and velocity vectors
    pos_vec_f = result_high_fidelity.state[0:3, -1]
    vel_vec_f = result_high_fidelity.state[3:6, -1]
    
    # Extract final COEs
    coe = result_high_fidelity.coe
    sma  = coe.sma[-1]
    ecc  = coe.ecc[-1]
    inc  = coe.inc[-1] * CONVERTER.DEG_PER_RAD
    raan = coe.raan[-1] * CONVERTER.DEG_PER_RAD
    aop  = coe.aop[-1] * CONVERTER.DEG_PER_RAD
    ta   = coe.ta[-1] * CONVERTER.DEG_PER_RAD
    ea   = coe.ea[-1] * CONVERTER.DEG_PER_RAD
    ma   = coe.ma[-1] * CONVERTER.DEG_PER_RAD

    print()
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
    print(f"      AOP  : { aop:>19.12e} deg")
    print(f"      TA   : {  ta:>19.12e} deg")
    print(f"      EA   : {  ea:>19.12e} deg")
    print(f"      MA   : {  ma:>19.12e} deg")

