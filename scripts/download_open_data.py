"""Scaffold for downloading open environmental data.

For the core exercise, students may use synthetic flow fields. For the extension,
use HYCOM, Copernicus Marine, or NASA Earthdata tooling to retrieve currents or
surface conditions and convert them into a 2-D vector field saved under data/processed/.
"""
from pathlib import Path


def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    print("See README assignment instructions for HYCOM/Copernicus/NOAA data links.")

if __name__ == "__main__":
    main()
