"""
Module 4: Active Learning for GP Surrogate Improvement
=======================================================

Active learning answers: "If I can only afford N more simulator evaluations,
where should I evaluate to maximally improve the surrogate?"

Two acquisition functions:
    1. Max-Variance: select the point where GP uncertainty is highest
       → best for global surrogate accuracy
    2. U-function: select where |μ(x)|/σ(x) is smallest
       → best for reliability analysis (accurate near the limit-state)

The active learning loop:
    1. Start with the trained GP (N₀ = 30 points)
    2. Generate a large candidate pool (10,000 LHS points)
    3. Score each candidate with the acquisition function
    4. Select the best candidate, evaluate the simulator, add to training set
    5. Retrain the GP
    6. Repeat until budget is exhausted

References:
    - Echard et al. (2011), Structural Safety – AK-MCS
    - Moustapha et al. (2022), Structural Safety – survey of active learning
"""

import numpy as np
import torch
import gpytorch
import gc
import time
import matplotlib.pyplot as plt
from src.simulator import simulate_duffing
from src.gp_surrogate import train_gp, predict_gp, compute_metrics, ExactGPModel
from src.utils import (
    lhs_sample, Normaliser, Standardiser, PARAM_NAMES, PARAM_BOUNDS,
)


# ---------------------------------------------------------------------------
# Active learning loop
# ---------------------------------------------------------------------------
def active_learning_loop(
    X_train_raw,
    Y_train_raw,
    X_test_raw,
    Y_test_raw,
    budget=20,
    candidate_pool_size=2000,
    acquisition='max_variance',
    x_threshold=1.5,
    qoi_key='x_max',
    gp_train_iter=150,
    seed=99,
):
    """
    Run the active learning loop.

    Parameters
    ----------
    X_train_raw : ndarray (N₀, 4) – initial training inputs
    Y_train_raw : ndarray (N₀,)   – initial training outputs
    X_test_raw  : ndarray (N_t, 4) – held-out test set (for tracking metrics)
    Y_test_raw  : ndarray (N_t,)   – test ground truth
    budget : int – number of new evaluations
    candidate_pool_size : int
    acquisition : str – 'max_variance' or 'u_function'
    x_threshold : float – threshold for U-function (used in reliability)
    qoi_key : str
    gp_train_iter : int
    seed : int

    Returns
    -------
    dict with history, final GP model, augmented training data
    """
    print(f"\n  Active learning: acquisition = {acquisition}, budget = {budget}")

    X_train = X_train_raw.copy()
    Y_train = Y_train_raw.copy()

    history = {
        'n_train': [],
        'rmse': [],
        'r2': [],
        'coverage': [],
        'mean_variance': [],
    }

    for iteration in range(budget):
        # --- Fit normalisers on current training data ---
        x_norm = Normaliser().fit(X_train)
        y_std = Standardiser().fit(Y_train)

        X_tr_t = torch.FloatTensor(x_norm.transform(X_train))
        Y_tr_t = torch.FloatTensor(y_std.transform(Y_train))

        # --- Train GP ---
        model, likelihood = train_gp(X_tr_t, Y_tr_t, n_iter=gp_train_iter,
                                      verbose=False)

        # --- Generate candidate pool ---
        candidates = lhs_sample(candidate_pool_size, seed=seed + iteration)

        # --- Predict at candidates ---
        X_cand_t = torch.FloatTensor(x_norm.transform(candidates))
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(X_cand_t))
            mean_cand = pred.mean.numpy()
            var_cand = pred.variance.numpy()

        # --- Acquisition ---
        if acquisition == 'max_variance':
            idx = np.argmax(var_cand)
        elif acquisition == 'u_function':
            # U = |μ - threshold| / σ  (in standardised space the threshold
            # needs to be standardised too)
            threshold_std = y_std.transform(np.array([x_threshold]))[0]
            u_vals = np.abs(mean_cand - threshold_std) / np.sqrt(var_cand + 1e-12)
            idx = np.argmin(u_vals)
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")

        x_new = candidates[idx]

        # --- Evaluate simulator ---
        y_new = simulate_duffing(x_new)[qoi_key]

        # --- Augment training set ---
        X_train = np.vstack([X_train, x_new])
        Y_train = np.append(Y_train, y_new)

        # --- Evaluate on test set ---
        X_te_t = torch.FloatTensor(x_norm.transform(X_test_raw))
        mean_te, std_te, _, _ = predict_gp(model, likelihood, X_te_t)
        mean_te_phys = y_std.inverse_transform(mean_te)
        std_te_phys = std_te * y_std.std

        metrics = compute_metrics(Y_test_raw, mean_te_phys, std_te_phys)

        history['n_train'].append(len(X_train))
        history['rmse'].append(metrics['rmse'])
        history['r2'].append(metrics['r2'])
        history['coverage'].append(metrics['coverage'])
        history['mean_variance'].append(np.mean(var_cand))

        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"    iter {iteration+1:3d}/{budget}  "
                  f"n={len(X_train):3d}  "
                  f"RMSE={metrics['rmse']:.4f}  "
                  f"R²={metrics['r2']:.4f}  "
                  f"cov={metrics['coverage']:.2f}")

        # Free memory
        del model, likelihood, X_tr_t, Y_tr_t, X_cand_t, pred
        gc.collect()

    # --- Final GP ---
    x_norm_final = Normaliser().fit(X_train)
    y_std_final = Standardiser().fit(Y_train)
    X_tr_t = torch.FloatTensor(x_norm_final.transform(X_train))
    Y_tr_t = torch.FloatTensor(y_std_final.transform(Y_train))
    model_final, lik_final = train_gp(X_tr_t, Y_tr_t, n_iter=300, verbose=False)

    return {
        'model': model_final,
        'likelihood': lik_final,
        'x_normaliser': x_norm_final,
        'y_standardiser': y_std_final,
        'X_train': X_train,
        'Y_train': Y_train,
        'history': history,
        'acquisition': acquisition,
    }


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------
def random_sampling_baseline(
    X_train_raw,
    Y_train_raw,
    X_test_raw,
    Y_test_raw,
    budget=20,
    qoi_key='x_max',
    gp_train_iter=150,
    seed=77,
):
    """
    Baseline: add points randomly (LHS) instead of actively.
    Used to quantify the benefit of active learning.
    """
    print(f"\n  Random baseline: budget = {budget}")

    X_train = X_train_raw.copy()
    Y_train = Y_train_raw.copy()

    # Pre-generate all random points
    new_points = lhs_sample(budget, seed=seed)
    new_evals = np.array([simulate_duffing(p)[qoi_key] for p in new_points])

    history = {'n_train': [], 'rmse': [], 'r2': [], 'coverage': []}

    for i in range(budget):
        X_train = np.vstack([X_train, new_points[i]])
        Y_train = np.append(Y_train, new_evals[i])

        x_norm = Normaliser().fit(X_train)
        y_std = Standardiser().fit(Y_train)
        X_tr_t = torch.FloatTensor(x_norm.transform(X_train))
        Y_tr_t = torch.FloatTensor(y_std.transform(Y_train))

        model, lik = train_gp(X_tr_t, Y_tr_t, n_iter=gp_train_iter, verbose=False)

        X_te_t = torch.FloatTensor(x_norm.transform(X_test_raw))
        mean_te, std_te, _, _ = predict_gp(model, lik, X_te_t)
        mean_phys = y_std.inverse_transform(mean_te)
        std_phys = std_te * y_std.std

        metrics = compute_metrics(Y_test_raw, mean_phys, std_phys)
        history['n_train'].append(len(X_train))
        history['rmse'].append(metrics['rmse'])
        history['r2'].append(metrics['r2'])
        history['coverage'].append(metrics['coverage'])

        if (i + 1) % 5 == 0:
            print(f"    iter {i+1:3d}/{budget}  n={len(X_train):3d}  "
                  f"RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")

        del model, lik, X_tr_t, Y_tr_t
        gc.collect()

    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_learning_curves(active_hist, random_hist,
                         save_path='figures/04_learning_curves.png'):
    """Plot RMSE and R² vs n_train for active vs random."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # RMSE
    ax = axes[0]
    ax.plot(active_hist['n_train'], active_hist['rmse'],
            'o-', color='#2563eb', label='Active (max-variance)', ms=4)
    ax.plot(random_hist['n_train'], random_hist['rmse'],
            's--', color='#dc2626', label='Random (LHS)', ms=4)
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('RMSE')
    ax.set_title('Learning Curve — RMSE')
    ax.legend()

    # R²
    ax = axes[1]
    ax.plot(active_hist['n_train'], active_hist['r2'],
            'o-', color='#2563eb', label='Active (max-variance)', ms=4)
    ax.plot(random_hist['n_train'], random_hist['r2'],
            's--', color='#dc2626', label='Random (LHS)', ms=4)
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('R²')
    ax.set_title('Learning Curve — R²')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_variance_reduction(active_hist,
                             save_path='figures/04_variance_reduction.png'):
    """Plot mean GP variance over iterations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(active_hist['n_train'], active_hist['mean_variance'],
            'o-', color='#7c3aed', ms=5)
    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('Mean predictive variance')
    ax.set_title('GP Uncertainty Reduction (Active Learning)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_selected_points(X_initial, X_final, param_pairs=None,
                          save_path='figures/04_selected_points.png'):
    """
    Scatter plots of initial vs actively selected points projected
    onto 2D parameter pairs.
    """
    if param_pairs is None:
        param_pairs = [(0, 1), (0, 3), (1, 2), (2, 3)]

    n_new = X_final.shape[0] - X_initial.shape[0]
    X_new = X_final[-n_new:]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for ax, (i, j) in zip(axes.flat, param_pairs):
        ax.scatter(X_initial[:, i], X_initial[:, j],
                   c='#93c5fd', s=25, edgecolors='#1d4ed8',
                   linewidths=0.5, label='Initial (LHS)', zorder=2)
        ax.scatter(X_new[:, i], X_new[:, j],
                   c='#fca5a5', s=40, marker='^', edgecolors='#dc2626',
                   linewidths=0.5, label='Active', zorder=3)
        ax.set_xlabel(PARAM_NAMES[i])
        ax.set_ylabel(PARAM_NAMES[j])
        ax.legend(fontsize=8)

    plt.suptitle('Actively Selected Training Points', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_active_learning(gp_result, budget=20):
    """
    Run active learning and random baseline, produce comparison plots.
    """
    print("=" * 60)
    print("Module 4 – Active Learning")
    print("=" * 60)

    X_train_raw = gp_result['X_train_raw']
    Y_train_raw = gp_result['Y_train_raw']
    X_test_raw = gp_result['X_test_raw']
    Y_test_raw = gp_result['Y_test_raw']

    # Active learning
    active_result = active_learning_loop(
        X_train_raw, Y_train_raw, X_test_raw, Y_test_raw,
        budget=budget, acquisition='max_variance',
    )

    # Random baseline
    random_hist = random_sampling_baseline(
        X_train_raw, Y_train_raw, X_test_raw, Y_test_raw,
        budget=budget,
    )

    # Plots
    plot_learning_curves(active_result['history'], random_hist)
    plot_variance_reduction(active_result['history'])
    plot_selected_points(X_train_raw, active_result['X_train'])

    # Report
    act_final_rmse = active_result['history']['rmse'][-1]
    rnd_final_rmse = random_hist['rmse'][-1]
    print(f"\n  Final RMSE — Active: {act_final_rmse:.4f}, Random: {rnd_final_rmse:.4f}")

    # Find how many random samples needed to match active RMSE
    target_rmse = act_final_rmse
    random_rmses = np.array(random_hist['rmse'])
    matched = np.where(random_rmses <= target_rmse)[0]
    if len(matched) > 0:
        n_random_needed = random_hist['n_train'][matched[0]]
        n_active_used = active_result['history']['n_train'][-1]
        reduction = (1 - n_active_used / n_random_needed) * 100
        print(f"  Active reached RMSE={target_rmse:.4f} with {n_active_used} samples "
              f"vs {n_random_needed} for random → {reduction:.0f}% reduction")
    else:
        print(f"  Random never reached active's final RMSE of {target_rmse:.4f}")

    return active_result, random_hist


if __name__ == "__main__":
    from src.gp_surrogate import build_gp_surrogate
    gp_result = build_gp_surrogate(n_train=30, n_test=100)
    run_active_learning(gp_result, budget=20)
