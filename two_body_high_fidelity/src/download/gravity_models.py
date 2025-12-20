"""
Download gravity field coefficient files from ICGEM.

Usage:
  python -m src.download.gravity_models
"""

import urllib.request
import os
from pathlib import Path


# Direct download URL for EGM2008 from ICGEM
EGM2008_URL = 'https://icgem.gfz.de/getmodel/gfc/c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340/EGM2008.gfc'


def download_gravity_model(
  model_name : str = 'EGM2008',
  output_dir : Path = None,
) -> Path:
  """
  Download a gravity field coefficient file.
  
  Input:
  ------
    model_name : str
      Name of the model (default: 'EGM2008').
    output_dir : Path
      Directory to save the file.
  
  Output:
  -------
    filepath : Path
      Path to downloaded file.
  """
  if output_dir is None:
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'data' / 'gravity_models'
  
  output_dir.mkdir(parents=True, exist_ok=True)
  
  filepath = output_dir / f'{model_name}.gfc'
  
  if filepath.exists():
    print(f"  {model_name}.gfc already exists at {filepath}")
    return filepath
  
  print(f"  Downloading {model_name} gravity model...")
  print(f"  This may take a few minutes (file is ~100 MB for full EGM2008)...")
  
  # Use direct URL for EGM2008
  if model_name.upper() == 'EGM2008':
    url = EGM2008_URL
  else:
    url = f'https://icgem.gfz.de/getmodel/gfc/{model_name}.gfc'
  
  try:
    urllib.request.urlretrieve(url, filepath)
    print(f"  Downloaded to: {filepath}")
  except Exception as e:
    print(f"  Download failed: {e}")
    print(f"\n  Please download manually:")
    print(f"    1. Go to: https://icgem.gfz.de/tom_longtime")
    print(f"    2. Select '{model_name}' and download in 'icgem' format")
    print(f"    3. Save to: {filepath}")
    raise
  
  return filepath


if __name__ == '__main__':
  print("Gravity Model Downloader")
  print("========================\n")
  
  print("Attempting to download EGM2008...")
  print("Note: Full file is ~100 MB. This may take a few minutes.\n")
  
  try:
    filepath = download_gravity_model('EGM2008')
    print(f"\nSuccess! File saved to: {filepath}")
  except Exception as e:
    print(f"\nAutomatic download failed.")
    print("\nManual download instructions:")
    print("  1. Go to: https://icgem.gfz.de/tom_longtime")
    print("  2. Select 'EGM2008'")
    print("  3. Set max degree to 120 (or your desired max)")
    print("  4. Download in 'icgem' format")
    print("  5. Save to: data/gravity_models/EGM2008.gfc")