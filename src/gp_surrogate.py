"""
Module 2: Gaussian Process / Kriging Surrogate
===============================================

Builds a GP surrogate using GPyTorch that replaces the expensive Duffing
simulator.  The GP provides:

    - A mean prediction μ(x) at any parameter point
    - A predictive uncertainty σ(x) that grows in under-sampled regions

Kernel: Matérn-5/2 with Automatic Relevance Determination (ARD).
ARD learns a separate length-scale per input dimension; short length-scale ⟹
the QoI is sensitive to that parameter.

Training: Marginal log-likelihood maximisation via Adam.

References:
    - Rasmussen & Williams (2006), "GP for Machine Learning", MIT Press.
    - GPyTorch documentation: https://gpytorch.ai
"""

import numpy as np
import torch
import gpytorch
import time
from src.simulator import simulate_duffing, evaluate_batch
from src.utils import (
    lhs_sample, Normaliser, Standardiser, PARAM_NAMES, set_plot_style,
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# GP Model definition
# ---------------------------------------------------------------------------
class ExactGPModel(gpytorch.models.ExactGP):
    """
    Standard exact GP with Matérn-5/2 ARD kernel.

    Architecture:
        mean:   constant mean function  m(x) = c
        kernel: σ² · Matérn₅⸝₂(x, x'; ℓ₁,…,ℓ_d)
                where ℓ_i is the ARD length-scale for dimension i.
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[1],
            )
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_gp(train_x, train_y, n_iter=300, lr=0.05, verbose=True):
    """
    Train a GP model by maximising the marginal log-likelihood.

    Parameters
    ----------
    train_x : Tensor (N, d) – normalised inputs
    train_y : Tensor (N,)   – standardised outputs
    n_iter  : int   – number of Adam iterations
    lr      : float – learning rate
    verbose : bool

    Returns
    -------
    model, likelihood – trained GP objects
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(n_iter):
        optimiser.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if verbose and (i + 1) % 100 == 0:
            print(f"  Iter {i+1:4d}/{n_iter}  loss = {loss.item():.4f}")

    return model, likelihood


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_gp(model, likelihood, test_x):
    """
    Make predictions with a trained GP.

    Returns
    -------
    mean  : ndarray (N,)
    std   : ndarray (N,)
    lower : ndarray (N,) – lower 95 % CI
    upper : ndarray (N,) – upper 95 % CI
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        mean = pred.mean.numpy()
        std = pred.stddev.numpy()
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
    return mean, std, lower, upper


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_std):
    """
    Compute standard GP validation metrics.

    Returns dict with:
        rmse     – root mean squared error
        r2       – coefficient of determination
        coverage – fraction of true values inside ±1.96 σ
        msll     – mean standardised log-likelihood
    """
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1.0 - ss_res / ss_tot

    inside = np.abs(residuals) <= 1.96 * y_std
    coverage = np.mean(inside)

    # Mean standardised log-likelihood
    msll = np.mean(
        0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * (residuals / y_std)**2
    )

    return {'rmse': rmse, 'r2': r2, 'coverage': coverage, 'msll': msll}


# ---------------------------------------------------------------------------
# Full pipeline: train, validate, report
# ---------------------------------------------------------------------------
def build_gp_surrogate(
    n_train=30,
    n_test=100,
    qoi_key='x_max',
    seed_train=42,
    seed_test=123,
    n_iter=300,
):
    """
    End-to-end GP surrogate pipeline.

    Steps:
        1. Generate LHS training & test sets
        2. Evaluate the simulator at all points
        3. Normalise inputs, standardise outputs
        4. Train the GP
        5. Validate on the test set
        6. Report metrics, speedup, and ARD length-scales

    Returns
    -------
    dict with trained model, data, normalisers, metrics, etc.
    """
    print("=" * 60)
    print("Module 2 – GP Surrogate: Build & Validate")
    print("=" * 60)

    # --- 1. Sample ---
    X_train_raw = lhs_sample(n_train, seed=seed_train)
    X_test_raw = lhs_sample(n_test, seed=seed_test)

    # --- 2. Evaluate simulator ---
    print(f"\nEvaluating simulator at {n_train} training points …")
    t0 = time.perf_counter()
    Y_train_raw = evaluate_batch(X_train_raw, qoi_key=qoi_key, verbose=True)
    sim_time_train = time.perf_counter() - t0

    print(f"Evaluating simulator at {n_test} test points …")
    t0 = time.perf_counter()
    Y_test_raw = evaluate_batch(X_test_raw, qoi_key=qoi_key, verbose=True)
    sim_time_test = time.perf_counter() - t0

    # --- 3. Normalise / standardise ---
    x_norm = Normaliser().fit(X_train_raw)
    y_std_obj = Standardiser().fit(Y_train_raw)

    X_train = torch.FloatTensor(x_norm.transform(X_train_raw))
    X_test = torch.FloatTensor(x_norm.transform(X_test_raw))
    Y_train = torch.FloatTensor(y_std_obj.transform(Y_train_raw))

    # --- 4. Train ---
    print(f"\nTraining GP (Matérn-5/2 ARD, {n_iter} iterations) …")
    model, likelihood = train_gp(X_train, Y_train, n_iter=n_iter)

    # --- 5. Predict on test set ---
    t0 = time.perf_counter()
    mean_std, std_std, lower_std, upper_std = predict_gp(model, likelihood, X_test)
    gp_time = time.perf_counter() - t0

    # Back to physical scale
    mean_phys = y_std_obj.inverse_transform(mean_std)
    std_phys = std_std * y_std_obj.std
    lower_phys = y_std_obj.inverse_transform(lower_std)
    upper_phys = y_std_obj.inverse_transform(upper_std)

    # --- 6. Metrics ---
    metrics = compute_metrics(Y_test_raw, mean_phys, std_phys)

    print(f"\n{'Metric':<15} {'Value':>10}")
    print("-" * 27)
    for k, v in metrics.items():
        print(f"  {k:<13} {v:>10.4f}")

    speedup = (sim_time_test) / max(gp_time, 1e-9)
    print(f"\n  Speedup:  {speedup:.0f}×  ({sim_time_test:.2f}s sim vs {gp_time*1000:.1f}ms GP)")

    # --- ARD length-scales ---
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
    print(f"\n  ARD length-scales (shorter = more sensitive):")
    for name, l in zip(PARAM_NAMES, ls):
        print(f"    {name}: {l:.4f}")

    # --- Package results ---
    result = {
        'model': model,
        'likelihood': likelihood,
        'x_normaliser': x_norm,
        'y_standardiser': y_std_obj,
        'X_train_raw': X_train_raw,
        'Y_train_raw': Y_train_raw,
        'X_test_raw': X_test_raw,
        'Y_test_raw': Y_test_raw,
        'Y_pred_mean': mean_phys,
        'Y_pred_std': std_phys,
        'Y_pred_lower': lower_phys,
        'Y_pred_upper': upper_phys,
        'metrics': metrics,
        'speedup': speedup,
        'ard_lengthscales': ls,
    }
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_calibration(result, save_path='figures/02_gp_calibration.png'):
    """
    Calibration plot: predicted mean ± 2σ vs. true values, sorted by
    predicted mean.  ≈95 % of true values should lie inside the band.
    """
    y_true = result['Y_test_raw']
    y_pred = result['Y_pred_mean']
    y_lower = result['Y_pred_lower']
    y_upper = result['Y_pred_upper']
    metrics = result['metrics']

    order = np.argsort(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: sorted calibration ---
    ax = axes[0]
    idx = np.arange(len(order))
    ax.fill_between(idx, y_lower[order], y_upper[order],
                     alpha=0.25, color='#3b82f6', label='95 % CI')
    ax.plot(idx, y_pred[order], '-', color='#1d4ed8', lw=1.2, label='GP mean')
    ax.scatter(idx, y_true[order], s=12, c='#ef4444', zorder=5, label='True')
    ax.set_xlabel('Test sample (sorted by GP mean)')
    ax.set_ylabel('x_max')
    ax.set_title(f'GP Calibration  (R²={metrics["r2"]:.3f}, '
                 f'coverage={metrics["coverage"]:.0%})')
    ax.legend()

    # --- Right: predicted vs true scatter ---
    ax = axes[1]
    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='ideal')
    ax.errorbar(y_true, y_pred, yerr=1.96 * result['Y_pred_std'],
                fmt='o', ms=4, color='#2563eb', ecolor='#93c5fd',
                elinewidth=0.6, capsize=0, label='GP ± 1.96σ')
    ax.set_xlabel('True x_max')
    ax.set_ylabel('Predicted x_max')
    ax.set_title('Predicted vs. True')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_ard(result, save_path='figures/02_ard_lengthscales.png'):
    """Bar chart of ARD length-scales."""
    ls = result['ard_lengthscales']
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2563eb', '#dc2626', '#059669', '#d97706']
    ax.barh(PARAM_NAMES, 1.0 / ls, color=colors)
    ax.set_xlabel('Inverse length-scale (∝ sensitivity)')
    ax.set_title('ARD Sensitivity (higher = more sensitive)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = build_gp_surrogate(n_train=30, n_test=100)
    plot_calibration(result)
    plot_ard(result)
