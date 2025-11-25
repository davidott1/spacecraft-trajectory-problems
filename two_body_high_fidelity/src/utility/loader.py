import yaml
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
