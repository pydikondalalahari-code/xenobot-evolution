#!/bin/bash
# ============================================================
# Xenobot Exercise – Dependency Installer for WSL/Ubuntu
# Run this from inside your xenobot-alife-cleanup-starter/ dir
# with your venv ALREADY activated.
#
#   source .venv/bin/activate
#   bash install_dependencies.sh
# ============================================================

set -e  # exit on first error

echo "=== Xenobot: Installing additional dependencies ==="
echo ""

# Upgrade pip first
pip install --upgrade pip

# Core scientific stack
pip install numpy>=1.24 scipy>=1.11 matplotlib>=3.7

# Evolutionary algorithms
pip install deap>=1.4

# 3-D visualisation (PyVista) – headless safe version
pip install pyvista>=0.42

# Jupyter for notebooks
pip install jupyter notebook ipykernel

# Additional utilities
pip install tqdm pytest

echo ""
echo "=== Checking imports ==="
python -c "
pkgs = ['numpy','scipy','matplotlib','deap','pyvista','jupyter']
for p in pkgs:
    try:
        __import__(p)
        print(f'  OK: {p}')
    except ImportError:
        print(f'  MISSING: {p}  <-- install failed, check errors above')
"

echo ""
echo "=== Setup results directory ==="
mkdir -p results

echo ""
echo "Done! Activate your venv and run:"
echo "  jupyter notebook notebooks/milestone1_notebook.ipynb"
