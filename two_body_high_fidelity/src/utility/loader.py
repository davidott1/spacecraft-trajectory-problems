import yaml
import spiceypy as spice
from pathlib import Path

def load_supported_objects() -> dict:
  """
  Load supported objects from YAML configuration file.
  
  Output:
  -------
    dict
      Dictionary of supported objects loaded from the YAML file.
  """
  # Adjust path traversal: utility -> src -> project_root
  project_root = Path(__file__).parent.parent.parent
  config_path  = project_root / 'data' / 'supported_objects.yaml'
  
  if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
  with open(config_path, 'r') as f:
    return yaml.safe_load(f)

def load_spice_files(
  use_spice                : bool,
  spice_kernels_folderpath : Path,
  lsk_filepath             : Path,
) -> None:
  """
  Load required data files, e.g., SPICE kernels.
  
  Input:
  ------
    use_spice : bool
      Flag to enable/disable SPICE usage.
    spice_kernels_folderpath : Path
      Path to the SPICE kernels folder.
    lsk_filepath : Path
      Path to the leap seconds kernel file.
  """
  if use_spice:
    if not spice_kernels_folderpath.exists():
      raise FileNotFoundError(f"SPICE kernels folder not found: {spice_kernels_folderpath}")
    if not lsk_filepath.exists():
      raise FileNotFoundError(f"SPICE leap seconds kernel not found: {lsk_filepath}")

    try:
      rel_path = spice_kernels_folderpath.relative_to(Path.cwd())
      display_path = f"<project_folderpath>/{rel_path}"
    except ValueError:
      display_path = spice_kernels_folderpath

    print(f"  Spice Kernels")
    print(f"    Folderpath : {display_path}")
    
    # Load leap seconds kernel first (minimal kernel set for time conversion)
    spice.furnsh(str(lsk_filepath))


def unload_spice_files(
  use_spice : bool,
) -> None:
  """
  Unload all SPICE kernels if they were loaded.
  
  Input:
  ------
    use_spice : bool
      Flag to enable/disable SPICE usage.
  """
  if use_spice:
    spice.kclear()
