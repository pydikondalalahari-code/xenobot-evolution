# src/voxcraft_runner.py


import subprocess
import tempfile
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from src.representation import EMPTY, PASSIVE, ACTIVE_P, ACTIVE_N, genome_to_vxa


VOXCRAFT_BIN = Path(os.environ.get("VOXCRAFT_BIN", "voxcraft-sim"))

MATERIAL_MAP = {
    PASSIVE:  {"r": 0.01, "E": 1e6,  "rho": 1e3, "nu": 0.35},
    ACTIVE_P: {"r": 0.01, "E": 5e5,  "rho": 1e3, "nu": 0.35,
               "cilia": True, "phase": 0.0},
    ACTIVE_N: {"r": 0.01, "E": 5e5,  "rho": 1e3, "nu": 0.35,
               "cilia": True, "phase": 3.14159},
}


def run_voxcraft(vxa_path: str, timeout: int = 120) -> tuple:

    if not VOXCRAFT_BIN.exists() and not _binary_on_path():
        print(f"[voxcraft] Binary not found at '{VOXCRAFT_BIN}'. "
              "Set VOXCRAFT_BIN environment variable or build first.")
        return 0.0, 0.0, 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = subprocess.run(
                [str(VOXCRAFT_BIN), "-i", str(vxa_path), "-o", tmpdir],
                capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            print("[voxcraft] Simulation timed out.")
            return 0.0, 0.0, 0.0
        except FileNotFoundError:
            print("[voxcraft] Binary not found.")
            return 0.0, 0.0, 0.0

        if result.returncode != 0:
            print(f"[voxcraft] Non-zero return code: {result.returncode}")
            print(result.stderr[:500])
            return 0.0, 0.0, 0.0

        # output XML 
        output_files = list(Path(tmpdir).glob("*.xml"))
        if not output_files:
            print("[voxcraft] No output XML found.")
            return 0.0, 0.0, 0.0

        return _parse_displacement(output_files[0])


def _parse_displacement(xml_path: Path) -> tuple:
    """Parse (dx, dy, dz) from VoxCraft output XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # VoxCraft output format: <detail><robot><displacement x="..." y="..." z="..."/>
        for elem in root.iter("displacement"):
            dx = float(elem.get("x", 0))
            dy = float(elem.get("y", 0))
            dz = float(elem.get("z", 0))
            return dx, dy, dz

        # Alternative: look for a <fitness> or <distance> tag
        for elem in root.iter("fitness"):
            return float(elem.text or 0), 0.0, 0.0

    except Exception as e:
        print(f"[voxcraft] XML parse error: {e}")

    return 0.0, 0.0, 0.0


def evaluate_genome_voxcraft(genome: np.ndarray, sim_time: float = 1.0) -> tuple:

    from src.representation import largest_connected_component
    genome = largest_connected_component(genome)

    if (genome != EMPTY).sum() == 0:
        return (0.0,)

    with tempfile.TemporaryDirectory() as tmpdir:
        vxa_path = Path(tmpdir) / "robot.vxa"
        genome_to_vxa(genome, str(vxa_path), sim_time=sim_time)
        dx, dy, dz = run_voxcraft(str(vxa_path))

    # Horizontal displacement only (ignore vertical)
    dist = float(np.sqrt(dx**2 + dy**2))
    return (max(0.0, dist),)


def _binary_on_path() -> bool:
    """Check if voxcraft-sim is on the system PATH."""
    try:
        subprocess.run(["voxcraft-sim", "--help"],
                       capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
