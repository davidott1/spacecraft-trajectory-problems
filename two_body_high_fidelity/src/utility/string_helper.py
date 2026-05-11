"""
String utility functions for filename sanitization and formatting.
"""

def sanitize_filename(
  name : str,
) -> str:
  """
  Sanitize a string for use in filenames.
  Converts to lowercase, replaces spaces and special characters with underscores.
  
  Example: "GOES-17 - GOES-S" -> "goes_17_goes_s"
           "HST - Hubble Space Telescope" -> "hst_hubble_space_telescope"
  
  Input:
  ------
  name : str
    Original name string.
  
  Output:
  -------
  str:
    Sanitized filename-safe string.
  """
  # Convert to lowercase
  result = name.lower()
  # Replace " - " pattern first (before individual character replacements)
  result = result.replace(' - ', '_')
  # Replace common special characters with underscores
  for char in [' ', '-', '(', ')', '[', ']', '/', '\\', '.', ',', "'", '"']:
    result = result.replace(char, '_')
  # Remove consecutive underscores
  while '__' in result:
    result = result.replace('__', '_')
  # Remove leading/trailing underscores
  result = result.strip('_')
  return result
