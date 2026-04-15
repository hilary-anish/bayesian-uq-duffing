"""
Utility Functions
=================

Shared helpers used across all modules:
    - Latin Hypercube Sampling (LHS)
    - Parameter space definitions and transformations
    - Plotting utilities with consistent style
    - Data normalisation / standardisation
"""

import numpy as np
from scipy.stats import norm, uniform
from pyDOE3 import lhs
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameter distributions
# ---------------------------------------------------------------------------
# Each entry: (scipy frozen distribution, label, unit)
DISTRIBUTIONS = [
    (norm(loc=0.3, scale=0.05),   'δ (damping)',   '1/s'),
    (norm(loc=-1.0, scale=0.15),  'α (lin. stiff.)', '1/s²'),
    (norm(loc=1.0, scale=0.1),    'β (nonlin. stiff.)', '1/(m²·s²)'),
    (uniform(loc=0.2, scale=0.3), 'F (forcing)',    'm/s²'),
]

PARAM_NAMES = ['δ', 'α', 'β', 'F']
N_DIM = 4

# Physical bounds for sampling (±4σ for Gaussians, exact for Uniform)
PARAM_BOUNDS = np.array([
    [0.3 - 4*0.05,  0.3 + 4*0.05],    # delta
    [-1.0 - 4*0.15, -1.0 + 4*0.15],   # alpha
    [1.0 - 4*0.1,   1.0 + 4*0.1],     # beta
    [0.2, 0.5],                         # F
])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def lhs_sample(n_samples, bounds=None, seed=42):
    """
    Generate Latin Hypercube samples in the physical parameter space.

    Parameters
    ----------
    n_samples : int
    bounds : ndarray (4, 2), optional – defaults to PARAM_BOUNDS
    seed : int

    Returns
    -------
    ndarray (n_samples, 4)
    """
    if bounds is None:
        bounds = PARAM_BOUNDS
    rng = np.random.RandomState(seed)
    unit_samples = lhs(N_DIM, samples=n_samples, criterion='maximin', random_state=rng)
    # Scale from [0,1] to physical bounds
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return unit_samples * (hi - lo) + lo


def sample_from_distributions(n_samples, seed=42):
    """
    Sample from the actual probability distributions (for MC / PCE).

    Returns
    -------
    ndarray (n_samples, 4)
    """
    rng = np.random.RandomState(seed)
    samples = np.column_stack([
        rng.normal(0.3, 0.05, n_samples),
        rng.normal(-1.0, 0.15, n_samples),
        rng.normal(1.0, 0.1, n_samples),
        rng.uniform(0.2, 0.5, n_samples),
    ])
    return samples


# ---------------------------------------------------------------------------
# Data normalisation
# ---------------------------------------------------------------------------
class Normaliser:
    """Min-max normalisation to [0, 1] for GP inputs."""
    def __init__(self):
        self.lo = None
        self.hi = None

    def fit(self, X):
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)
        # Avoid division by zero
        self.hi = np.where(self.hi == self.lo, self.lo + 1.0, self.hi)
        return self

    def transform(self, X):
        return (X - self.lo) / (self.hi - self.lo)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Standardiser:
    """Zero-mean, unit-variance standardisation for GP outputs."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        self.mean = y.mean()
        self.std = y.std()
        if self.std < 1e-12:
            self.std = 1.0
        return self

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean

    def fit_transform(self, y):
        return self.fit(y).transform(y)


# ---------------------------------------------------------------------------
# Nataf transformation (for FORM)
# ---------------------------------------------------------------------------
def nataf_transform(params, distributions=None):
    """
    Transform from physical space to standard normal space.

    For independent marginals this is simply:
        u_i = Φ⁻¹( F_i(x_i) )
    where F_i is the CDF of the i-th marginal.
    """
    if distributions is None:
        distributions = [d for d, _, _ in DISTRIBUTIONS]
    u = np.zeros_like(params, dtype=float)
    for i, (val, dist) in enumerate(zip(params, distributions)):
        p = dist.cdf(val)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        u[i] = norm.ppf(p)
    return u


def inverse_nataf(u, distributions=None):
    """Transform from standard normal space back to physical space."""
    if distributions is None:
        distributions = [d for d, _, _ in DISTRIBUTIONS]
    params = np.zeros_like(u, dtype=float)
    for i, (ui, dist) in enumerate(zip(u, distributions)):
        p = norm.cdf(ui)
        params[i] = dist.ppf(p)
    return params


# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
def set_plot_style():
    """Apply a clean, publication-quality matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 120,
    })

set_plot_style()  # apply on import
