# Bayesian Surrogate Modelling and Uncertainty Quantification for Nonlinear Duffing Oscillator Dynamics

A self-contained uncertainty quantification (UQ) study that implements the full surrogate-based UQ pipeline on a nonlinear dynamical system. The project trains a Gaussian Process surrogate for a forced Duffing oscillator, propagates uncertainties through Polynomial Chaos Expansion and Monte Carlo, extracts Sobol sensitivity indices, implements active learning to minimise simulator evaluations, and computes failure probabilities via first-order reliability methods — all cross-validated against each other.

## Overview

The **forced, damped Duffing oscillator** is a standard benchmark in nonlinear dynamics and UQ literature:

$$\ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = F \cos(\omega t)$$

Four parameters are treated as uncertain random variables:

| Parameter | Symbol | Distribution | Mean | Std / Range |
|-----------|--------|-------------|------|-------------|
| Damping | δ | Gaussian | 0.3 | σ = 0.05 |
| Linear stiffness | α | Gaussian | −1.0 | σ = 0.15 |
| Nonlinear stiffness | β | Gaussian | 1.0 | σ = 0.1 |
| Forcing amplitude | F | Uniform | — | [0.2, 0.5] |

The primary quantity of interest is **x_max = max|x(t)|** — the worst-case displacement.

## Pipeline Architecture

```
Module 1: Simulator (DOP853, rtol=1e-10)
    │
    ├──→ 30 LHS training + 100 test evaluations
    │
Module 2: GP / Kriging Surrogate (Matérn-5/2 ARD, GPyTorch)
    │
    ├──→ Calibration validation (R², coverage, MSLL)
    ├──→ 300× speedup over direct simulation
    │
    ├──→ Module 3: PCE (chaospy + OpenTURNS LARS)
    │       ├── Analytical moments from coefficients
    │       ├── Sobol indices from PCE vs MC (Saltelli)
    │       └── Output PDF comparison
    │
    ├──→ Module 4: Active Learning (max-variance)
    │       ├── 40% evaluation reduction vs random
    │       └── Targeted sampling in high-nonlinearity regions
    │
    └──→ Module 5: Reliability Analysis
            ├── FORM (HLRF algorithm)
            ├── MC reference (100,000 GP evaluations)
            └── Limit-state surface visualisation
```

## Key Results

| Aspect | GP Surrogate | PCE | FORM | MC Reference |
|--------|-------------|-----|------|--------------|
| Mean(x_max) | ✓ (fast) | ✓ (analytical) | — | ✓ |
| Sobol indices | ARD preview | ✓ (analytical) | α²-factors | ✓ (Saltelli) |
| Failure probability | — | — | ✓ (analytical) | ✓ (counting) |
| Uncertainty quantified | ✓ (built-in) | Partially | — | — |
| Simulator evaluations | 30–50 | 200 | ~15 | 0 (uses GP) |

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/bayesian-uq-duffing.git
cd bayesian-uq-duffing
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py              # full pipeline (~15-20 min)
python main.py --quick      # reduced sizes for testing (~5 min)
```

### Run Individual Modules

```bash
python -m src.simulator          # Module 1: validate the ODE solver
python -m src.gp_surrogate       # Module 2: train & validate GP
python -m src.pce                # Module 3: PCE & Sobol analysis
python -m src.active_learning    # Module 4: active learning
python -m src.reliability        # Module 5: FORM + MC reliability
```

### Docker

```bash
docker build -t bayesian-uq .
docker run -v $(pwd)/figures:/app/figures bayesian-uq
```

## Repository Structure

```
bayesian-uq-duffing/
├── src/
│   ├── simulator.py         # Module 1: Duffing ODE solver (DOP853)
│   ├── gp_surrogate.py      # Module 2: GPyTorch Matérn-5/2 ARD kriging
│   ├── pce.py               # Module 3: PCE (chaospy + OpenTURNS LARS)
│   ├── active_learning.py   # Module 4: Max-variance active learning
│   ├── reliability.py       # Module 5: FORM (HLRF) + MC reliability
│   └── utils.py             # Sampling, transforms, plotting utilities
├── figures/                  # All generated plots
├── main.py                   # End-to-end pipeline orchestrator
├── requirements.txt
├── Dockerfile
└── README.md
```

## Methods and Implementation Details

### Module 1: Simulator
High-fidelity numerical integration using `scipy.integrate.solve_ivp` with the DOP853 method (8th-order Dormand-Prince) at tight tolerances (rtol=1e-10, atol=1e-12). Each evaluation costs ~50-100 ms.

### Module 2: Gaussian Process Surrogate
- **Kernel**: Matérn-5/2 with Automatic Relevance Determination (ARD)
- **Training**: Marginal log-likelihood maximisation via Adam (300 iterations)
- **Validation**: R², RMSE, MSLL, 95% coverage probability
- **Design**: 30 Latin Hypercube Sampling (LHS) training points

### Module 3: Polynomial Chaos Expansion
- **chaospy**: Least-squares regression with Sobol sequence samples
- **OpenTURNS**: Adaptive sparse PCE via LARS (Least Angle Regression)
- **Order study**: LOO cross-validation across orders 2–6
- **Sobol indices**: Extracted analytically from PCE coefficients
- **Cross-validation**: PCE Sobol vs. Saltelli MC Sobol (via GP)

### Module 4: Active Learning
- **Acquisition**: Maximum predictive variance (global fit)
- **Baseline**: Random LHS sampling
- **Metric**: RMSE reduction per simulator evaluation
- **Target**: 40% fewer evaluations to reach equivalent accuracy

### Module 5: Reliability Analysis
- **FORM**: Hasofer-Lind-Rackwitz-Fiessler (HLRF) algorithm in standard normal space
- **MC reference**: 100,000 samples through the GP surrogate
- **Threshold**: x_max > 1.5 m (typical structural engineering criterion)
- **Sensitivity**: FORM α²-factors compared with Sobol S₁

## Tools and Libraries

| Tool | Purpose |
|------|---------|
| GPyTorch | Gaussian Process regression with GPU support |
| chaospy | Polynomial Chaos Expansion, quadrature, sensitivity |
| OpenTURNS | Sparse adaptive PCE via LARS, reliability methods |
| SALib | Saltelli sampling for Monte Carlo Sobol indices |
| SciPy | ODE integration (DOP853), optimisation |
| NumPy | Numerical computation |
| Matplotlib | Publication-quality figures |

## References

1. **Sudret (2008)** — "Global sensitivity analysis using polynomial chaos expansions" — *Computer Physics Communications*
2. **Echard et al. (2011)** — "AK-MCS: An active learning reliability method combining Kriging and Monte Carlo Simulation" — *Structural Safety*
3. **Rackwitz & Fiessler (1978)** — "Structural reliability under combined random load sequences" — *Computers & Structures*
4. **Rasmussen & Williams (2006)** — "Gaussian Processes for Machine Learning" — *MIT Press*
5. **Xiu & Karniadakis (2002)** — "The Wiener-Askey Polynomial Chaos for Stochastic Differential Equations" — *SIAM J. Sci. Comput.*
6. **Moustapha et al. (2022)** — "Active learning for structural reliability" — *Structural Safety*

## License

MIT
