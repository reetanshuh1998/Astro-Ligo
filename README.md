# ns_infer: Neutron Star Parameter Estimation & EOS Machine Learning Inference

A research-grade Python package designed to perform fast, physics-informed neutron star parameter estimation and Equation of State (EOS) constraints using modern machine learning.

## 🌌 Features
1. **Relativistic ODE Core (`physics/tov_solver.py`)**: Solves background Tolman-Oppenheimer-Volkoff (TOV) and Regge-Wheeler $l=2$ metric perturbation equations to compute exact stellar profiles and dimensionless tidal deformability $\Lambda$.
2. **Analytical Sound Speed**: Employs exact derivatives for piecewise polytropic EOS core segments ($c_s^2 = \frac{\Gamma P}{\epsilon + P}$) for thermodynamic consistency and noise-free integration.
3. **Physical EOS Generator (`data/dataset_builder.py`)**: Generates balanced synthetic datasets of ~5,000 neutron stars filtered by causality, micro-stability, and astronomical mass limits ($M_{\text{max}} \ge 1.97 M_\odot$).
4. **Empirical Posteriors (`data/observational_data.py`)**: Features a dual-mode manager to load official LIGO/Virgo and NICER posterior samples, with a high-fidelity KDE fallback mode for lightweight/local execution.
5. **Uncertainty-Aware ML Suite (`models/`)**: Trains XGBoost, PyTorch MLPs, and a Gaussian Deep Ensemble + MC Dropout model to output exact predictions and robust physical confidence intervals.
6. **Diagnostics & Calibration (`evaluation/`)**: Validates interval coverage and plots Probability Integral Transform (PIT) Histograms.

## 🛠️ Setup & Installation

To run the pipeline locally, create a virtual environment and install the package in editable mode:

```bash
# Create local virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

*Note: For CPU-only installations (recommended for fast inference benchmarks), use `pip install torch --index-url https://download.pytorch.org/whl/cpu`.*

## 🚀 Execution

To execute the entire pipeline (data generation, training, evaluation, and generating figures) in mock mode:
```bash
python3 src/ns_infer/run_pipeline.py --mode mock --num-eos 100 --epochs 30
```
