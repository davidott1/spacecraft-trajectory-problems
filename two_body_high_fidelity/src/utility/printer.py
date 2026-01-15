from src.schemas.propagation import PropagationResult


def print_results_summary(
  result_high_fidelity : PropagationResult,  # noqa: ARG001
) -> None:
  """
  Print a summary of the propagation results.

  NOTE: This function is deprecated. Results are now printed directly
  in the High-Fidelity Model Summary section.

  Input:
  ------
    result_high_fidelity : PropagationResult
      High-fidelity propagation result object.

  Output:
  -------
    None
  """
  # Results now printed in the High-Fidelity Model Summary
  pass

def final_print(final_message: str = "") -> None:
  """
  Print a final message indicating the end of the program.
  
  Output:
  -------
    None
  """
  print(final_message)