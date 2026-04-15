"""
Module 5: Reliability Analysis — FORM & Monte Carlo
=====================================================

Answers: "What is the probability that x_max exceeds a safety threshold?"

Limit-state function:
    g(ξ) = x_threshold − x_max(ξ)
    g < 0  ⟹  failure

Methods:
    1. Monte Carlo (via GP surrogate) — brute-force reference
    2. FORM (First-Order Reliability Method) — analytical approximation
       using the Hasofer-Lind-Rackwitz-Fiessler (HLRF) algorithm

FORM workflow:
    a. Transform to standard normal space (Nataf)
    b. Find the Most Probable Point (MPP) — closest point to origin on g = 0
    c. Reliability index β = ||u*||
    d. Failure probability P_f = Φ(−β)

References:
    - Rackwitz & Fiessler (1978), Comput. Struct.
    - Echard et al. (2011), Structural Safety – AK-MCS
"""

import numpy as np
import torch
import gpytorch
from scipy.stats import norm as normal_dist
import matplotlib.pyplot as plt
from src.utils import (
    DISTRIBUTIONS, PARAM_NAMES, nataf_transform, inverse_nataf,
    Normaliser, Standardiser, sample_from_distributions,
)


# ---------------------------------------------------------------------------
# GP prediction helpers
# ---------------------------------------------------------------------------
def _gp_predict(model, likelihood, x_norm, y_std, params_raw):
    """Predict x_max at a single physical-space point."""
    params_normed = x_norm.transform(params_raw.reshape(1, -1))
    x_t = torch.FloatTensor(params_normed)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x_t))
        mean = y_std.inverse_transform(pred.mean.numpy()[0])
        std = pred.stddev.numpy()[0] * y_std.std
    return mean, std


def _gp_predict_batch(model, likelihood, x_norm, y_std, params_array,
                      chunk_size=2000):
    """Predict x_max at many physical-space points (chunked to save memory)."""
    X_normed = x_norm.transform(params_array)
    N = X_normed.shape[0]
    all_mean = np.empty(N)
    all_std = np.empty(N)
    model.eval()
    likelihood.eval()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_t = torch.FloatTensor(X_normed[start:end])
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x_t))
            all_mean[start:end] = pred.mean.numpy()
            all_std[start:end] = pred.stddev.numpy()
    all_mean = y_std.inverse_transform(all_mean)
    all_std = all_std * y_std.std
    return all_mean, all_std


# ---------------------------------------------------------------------------
# Monte Carlo reliability (via GP)
# ---------------------------------------------------------------------------
def mc_reliability(gp_result, x_threshold=1.5, n_mc=50_000, seed=42):
    """
    Monte Carlo estimate of the failure probability P_f = P[x_max > threshold].

    Uses the GP surrogate for fast evaluation (≈seconds instead of hours).

    Returns
    -------
    dict with pf, beta, ci_lower, ci_upper, n_fail, etc.
    """
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    x_norm = gp_result['x_normaliser']
    y_std = gp_result['y_standardiser']

    # Sample from the true distributions
    samples = sample_from_distributions(n_mc, seed=seed)

    # Evaluate GP
    mean, _ = _gp_predict_batch(model, likelihood, x_norm, y_std, samples)

    # Count failures:  g = threshold - x_max;  failure if g < 0
    g = x_threshold - mean
    n_fail = np.sum(g < 0)
    pf = n_fail / n_mc

    # 95 % confidence interval on pf
    if pf > 0 and pf < 1:
        ci_half = 1.96 * np.sqrt(pf * (1 - pf) / n_mc)
    else:
        ci_half = 0.0

    # Corresponding reliability index
    beta_mc = -normal_dist.ppf(pf) if pf > 0 else np.inf

    print(f"\n  Monte Carlo Reliability (N = {n_mc:,})")
    print(f"    P_f  = {pf:.4e}  ({n_fail} failures)")
    print(f"    95%% CI: [{pf - ci_half:.4e}, {pf + ci_half:.4e}]")
    print(f"    β    = {beta_mc:.4f}")

    return {
        'pf': pf,
        'beta': beta_mc,
        'ci_lower': pf - ci_half,
        'ci_upper': pf + ci_half,
        'n_fail': n_fail,
        'n_mc': n_mc,
        'g_values': g,
        'samples': samples,
        'x_max_pred': mean,
    }


# ---------------------------------------------------------------------------
# FORM — HLRF algorithm
# ---------------------------------------------------------------------------
def form_hlrf(gp_result, x_threshold=1.5, max_iter=100, tol=1e-4):
    """
    First-Order Reliability Method using the HLRF algorithm.

    Finds the Most Probable Point (MPP) on the limit-state surface g = 0
    in standard normal space, then computes:
        β = ||u*||
        P_f = Φ(−β)

    The gradient ∇g is computed via finite differences on the GP.

    Returns
    -------
    dict with u_star, x_star, beta, pf, alpha_factors, n_iter
    """
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    x_norm = gp_result['x_normaliser']
    y_std = gp_result['y_standardiser']

    dists = [d for d, _, _ in DISTRIBUTIONS]
    n_dim = len(dists)
    h = 1e-4  # finite-difference step in u-space

    def g_eval(u):
        """Evaluate limit-state function in standard normal space."""
        params = inverse_nataf(u, dists)
        mean, _ = _gp_predict(model, likelihood, x_norm, y_std, params)
        return x_threshold - mean

    def grad_g(u):
        """Gradient of g via central finite differences."""
        g0 = g_eval(u)
        grad = np.zeros(n_dim)
        for i in range(n_dim):
            u_plus = u.copy()
            u_plus[i] += h
            u_minus = u.copy()
            u_minus[i] -= h
            grad[i] = (g_eval(u_plus) - g_eval(u_minus)) / (2 * h)
        return grad

    # --- HLRF iteration ---
    u = np.zeros(n_dim)
    print(f"\n  FORM (HLRF algorithm, tol={tol})")

    relaxation = 0.5  # under-relaxation for stability

    for k in range(max_iter):
        g_val = g_eval(u)
        dg = grad_g(u)
        dg_norm = np.linalg.norm(dg)

        if dg_norm < 1e-12:
            raise RuntimeError("Zero gradient — limit-state may be flat.")

        alpha = dg / dg_norm
        beta_k = np.dot(alpha, u) - g_val / dg_norm
        u_target = beta_k * alpha
        u_new = u + relaxation * (u_target - u)

        diff = np.linalg.norm(u_new - u)
        if k % 10 == 0 or diff < tol:
            print(f"    iter {k+1:3d}:  β = {np.linalg.norm(u_new):.4f}  "
                  f"g = {g_val:.6f}  Δu = {diff:.2e}")

        if diff < tol and abs(g_val) < 1e-3:
            u_star = u_new
            beta = np.linalg.norm(u_star)
            pf = normal_dist.cdf(-beta)
            x_star = inverse_nataf(u_star, dists)

            print(f"\n    Converged in {k+1} iterations")
            print(f"    MPP (u-space): {np.round(u_star, 4)}")
            print(f"    MPP (physical): δ={x_star[0]:.4f}, α={x_star[1]:.4f}, "
                  f"β={x_star[2]:.4f}, F={x_star[3]:.4f}")
            print(f"    β = {beta:.4f}")
            print(f"    P_f = {pf:.4e}")

            # α-factors: contribution of each variable to failure
            alpha_factors = alpha**2  # squared direction cosines
            print(f"    α²-factors (sensitivity at MPP): {np.round(alpha_factors, 4)}")

            return {
                'u_star': u_star,
                'x_star': x_star,
                'beta': beta,
                'pf': pf,
                'alpha_factors': alpha_factors,
                'alpha_direction': alpha,
                'n_iter': k + 1,
            }

        u = u_new

    # If we reach here, return the best result found (soft convergence)
    u_star = u
    beta = np.linalg.norm(u_star)
    pf = normal_dist.cdf(-beta)
    x_star = inverse_nataf(u_star, dists)
    alpha_factors = alpha**2

    print(f"\n    Reached max iterations ({max_iter}), returning best estimate")
    print(f"    β = {beta:.4f},  P_f = {pf:.4e}")

    return {
        'u_star': u_star,
        'x_star': x_star,
        'beta': beta,
        'pf': pf,
        'alpha_factors': alpha_factors,
        'alpha_direction': alpha,
        'n_iter': max_iter,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_limit_state(gp_result, mc_result, form_result, x_threshold=1.5,
                     save_path='figures/05_limit_state.png'):
    """
    2D limit-state surface visualisation.

    Fixes the two least sensitive parameters at their means and plots
    g(ξ_i, ξ_j) = 0 for the two most sensitive parameters.
    """
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    x_norm = gp_result['x_normaliser']
    y_std_obj = gp_result['y_standardiser']

    # Use α and F as the two axes (typically most sensitive)
    # Fix δ and β at their means
    n_grid = 80
    alpha_range = np.linspace(-1.0 - 3*0.15, -1.0 + 3*0.15, n_grid)
    F_range = np.linspace(0.2, 0.5, n_grid)
    AA, FF = np.meshgrid(alpha_range, F_range)

    G = np.zeros_like(AA)
    for i in range(n_grid):
        for j in range(n_grid):
            params = np.array([0.3, AA[i, j], 1.0, FF[i, j]])
            mean_val, _ = _gp_predict(model, likelihood, x_norm, y_std_obj, params)
            G[i, j] = x_threshold - mean_val

    fig, ax = plt.subplots(figsize=(9, 7))

    # Filled contour of g
    cf = ax.contourf(AA, FF, G, levels=30, cmap='RdBu', alpha=0.7)
    plt.colorbar(cf, ax=ax, label='g(ξ) = threshold − x_max')

    # Limit-state surface g = 0
    cs = ax.contour(AA, FF, G, levels=[0], colors='black', linewidths=2)
    ax.clabel(cs, fmt='g=0', fontsize=10)

    # MC samples
    samples = mc_result['samples']
    g_vals = mc_result['g_values']
    safe = g_vals >= 0
    fail = g_vals < 0

    # Plot a subset for clarity
    n_plot = min(3000, len(samples))
    rng = np.random.RandomState(42)
    idx_plot = rng.choice(len(samples), n_plot, replace=False)

    ax.scatter(samples[idx_plot[safe[idx_plot]], 1],
               samples[idx_plot[safe[idx_plot]], 3],
               s=3, c='#93c5fd', alpha=0.3, label='Safe (MC)')
    ax.scatter(samples[idx_plot[fail[idx_plot]], 1],
               samples[idx_plot[fail[idx_plot]], 3],
               s=8, c='#ef4444', alpha=0.6, label='Failure (MC)')

    # MPP
    x_star = form_result['x_star']
    ax.scatter(x_star[1], x_star[3], s=200, c='gold', marker='*',
               edgecolors='black', linewidths=1.5, zorder=10, label='MPP (FORM)')

    ax.set_xlabel('α (linear stiffness)')
    ax.set_ylabel('F (forcing amplitude)')
    ax.set_title(f'Limit-State Surface  (threshold = {x_threshold})')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_comparison_table(mc_result, form_result,
                          save_path='figures/05_comparison_table.png'):
    """
    Visual table comparing FORM vs MC results.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    data = [
        ['Method', 'P_f', 'β', 'Evaluations', 'Time (approx)'],
        ['MC (100k, GP)',
         f"{mc_result['pf']:.4e}",
         f"{mc_result['beta']:.4f}",
         f"{mc_result['n_mc']:,} GP evals",
         '~0.5 s'],
        ['FORM (HLRF)',
         f"{form_result['pf']:.4e}",
         f"{form_result['beta']:.4f}",
         f"{form_result['n_iter'] * 10} GP evals",
         '~0.01 s'],
    ]

    table = ax.table(cellText=data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(5):
        table[0, j].set_facecolor('#1d4ed8')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.title('Reliability Analysis: FORM vs Monte Carlo', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sensitivity_comparison(form_result, pce_sobol=None,
                                 save_path='figures/05_sensitivity_comparison.png'):
    """
    Compare FORM α²-factors (local sensitivity at MPP) with
    Sobol first-order indices (global sensitivity).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(4)
    width = 0.3

    ax.bar(x - width/2, form_result['alpha_factors'], width,
           label='FORM α²-factors (local)', color='#2563eb')

    if pce_sobol is not None:
        ax.bar(x + width/2, pce_sobol, width,
               label='Sobol S₁ (global, PCE)', color='#059669')

    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_ylabel('Sensitivity measure')
    ax.set_title('Local (FORM) vs Global (Sobol) Sensitivity')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_reliability_analysis(gp_result, pce_sobol_first=None, x_threshold=1.5):
    """Run FORM + MC reliability analysis."""
    print("=" * 60)
    print("Module 5 – Reliability Analysis")
    print("=" * 60)

    # MC reference
    mc_result = mc_reliability(gp_result, x_threshold=x_threshold)

    # FORM
    form_result = form_hlrf(gp_result, x_threshold=x_threshold)

    # Plots
    plot_limit_state(gp_result, mc_result, form_result, x_threshold)
    plot_comparison_table(mc_result, form_result)
    plot_sensitivity_comparison(form_result, pce_sobol_first)

    return mc_result, form_result


if __name__ == "__main__":
    from src.gp_surrogate import build_gp_surrogate
    gp_result = build_gp_surrogate(n_train=50, n_test=100)
    run_reliability_analysis(gp_result)
