"""
Download SPICE kernels for third-body perturbations.

Usage:
  python -m src.download.kernels
"""

import urllib.request
import ssl
from pathlib import Path

# Base URL
BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

# Kernel files to download
KERNELS = {
    'naif0012.tls' : 'lsk/naif0012.tls',
    'de440.bsp'    : 'spk/planets/de440.bsp',
    'pck00010.tpc' : 'pck/pck00010.tpc',
}

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'spice_kernels'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create a default SSL context that doesn't verify certificates (for NASA site).
# This is a workaround for older Python versions that don't support the 'context'
# argument in urlretrieve.
ssl._create_default_https_context = ssl._create_unverified_context

def download_with_progress(url, output_file):
    """Download file with progress bar"""
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  Progress: {percent:5.1f}% ({mb_downloaded:6.1f} / {mb_total:6.1f} MB)", 
              end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, output_file, reporthook=reporthook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False

print("Downloading SPICE kernels...")
print(f"Output directory: {OUTPUT_DIR}\n")

for filename, url_path in KERNELS.items():
    output_file = OUTPUT_DIR / filename
    
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ {filename} already exists ({size_mb:.1f} MB), skipping")
        continue
    
    url = f"{BASE_URL}/{url_path}"
    print(f"Downloading {filename}...")
    
    success = download_with_progress(url, output_file)
    
    if success:
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ {filename} complete ({size_mb:.1f} MB)")
    else:
        print(f"✗ {filename} failed")
        if output_file.exists():
            output_file.unlink()  # Remove partial download

print("\n" + "="*60)
print("Download complete!")
print(f"Kernels location: {OUTPUT_DIR}")
print("="*60)
