# Xenobot-Alife Cleanup: Evolving Soft-Bodied Collectors in Real Ocean-Inspired Flow Fields

Graduate Embodied AI / Artificial Life starter repository.

## Big question
Can a soft, living-machine-inspired body plan and a tiny distributed controller be co-designed to collect drifting debris in a simulated water channel under realistic current perturbations?

Students will implement a simulated xenobot-like soft robot made of voxels, evolve its morphology and controller, and evaluate whether it can collect particles while remaining robust to flow, energy limits, and body-damage perturbations.

## Core stack
- Python 3.10 recommended
- NumPy, SciPy, pandas, matplotlib
- Gymnasium-style API
- Optional: EvoGym for voxel soft-body simulation
- Optional: PyBullet for rigid-body comparison baselines
- Optional: Jupyter notebooks for analysis

## Free data/resources
- NOAA MDMAP: public shoreline debris monitoring and assessment resources
- HYCOM: public ocean-current model outputs
- Copernicus Marine: free/open ocean state data, including currents

This starter repo includes a lightweight 2-D differentiable-ish particle/current world so the assignment is runnable even before students install EvoGym. The extension path asks students to replace the minimal simulator with EvoGym or Voxelyze/Evosoro.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .[dev]
python scripts/run_baseline.py --config configs/baseline.yaml
pytest -q
```

## Repository map

```text
configs/baseline.yaml          Baseline environment and evolution config
data/raw/README.md             Where to place downloaded public datasets
data/processed/README.md       Processed current/debris-field files
notebooks/analysis.ipynb       Empty analysis notebook shell
scripts/download_open_data.py  Data-access scaffold
scripts/run_baseline.py        Runs the starter evolutionary loop
src/xenobot_alife/             Main package
  body.py                      Voxel-body genome and validation
  controller.py                Distributed oscillator controller skeleton
  env.py                       Current field + debris collection environment
  evolution.py                 Evolutionary algorithm skeleton
  fitness.py                   Multi-objective fitness functions
  render.py                    Visualization helpers
tests/                         Minimal tests
```

## Student TODOs
1. Implement genotype-to-phenotype decoding in `body.py`.
2. Implement coupled oscillator control in `controller.py`.
3. Implement sensing and action effects in `env.py`.
4. Implement multi-objective selection in `evolution.py`.
5. Compare emergent behaviors under laminar, vortex, and real-data-derived currents.
6. Write a short scientific report with ablations.

## Suggested report sections
- Motivation and alife framing
- Methods: body encoding, controller, simulation, open data
- Results: fitness curves, trajectories, morphology gallery
- Ablations: body-only, controller-only, flow-only, damage robustness
- Discussion: emergence, autopoiesis analogy, limits of simulation, ethics

