# Multiresolution HRL Optimization

This repository collects conceptual designs for multiresolution optimization methods powered by hierarchical reinforcement learning (HRL).

## Steel–aluminum body-in-white welding optimization

See `docs/hybrid-body-optimization.md` for a detailed mathematical formulation of the multi-objective mixed nonlinear dynamic programming problem that jointly optimizes workstation sequencing, weld-point allocation, and process parameters (temperature, power, path) for a steel–aluminum body-in-white welding line. The document also outlines a hierarchical RL strategy coupling high-level sequencing with low-level process control.

## Getting started with the Python prototype
A minimal HRL prototype is provided under `welding_hrl/` with hierarchical policies and a training harness in `train.py`.

1. Install dependencies: `pip install -r requirements.txt`.
2. Run a demo training loop: `python train.py`.

The environment wraps the mathematical model defined in `welding_hrl/math_model.py` and exposes hierarchical actions (station selection + process parameters) for experimentation with different policy updates.
