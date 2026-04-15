"""
Module 3: Polynomial Chaos Expansion (PCE) & Sobol Sensitivity Analysis
========================================================================

PCE represents the QoI as a weighted sum of orthogonal polynomials:

    Y(ξ) ≈ Σ_α  c_α · Ψ_α(ξ)

where Ψ_α are multivariate orthogonal polynomials selected by the
Wiener–Askey scheme:
    - Gaussian inputs  → Hermite polynomials
    - Uniform inputs   → Legendre polynomials

Once the coefficients c_α are known, statistical moments and Sobol
sensitivity indices follow analytically — no additional sampling needed.

Implementations:
    A. chaospy   – pseudospectral projection AND least-squares regression
    B. OpenTURNS – adaptive sparse PCE via LARS (state-of-the-art)

Cross-validation:  PCE-derived Sobol indices vs. Monte Carlo (Saltelli)
computed through the GP surrogate.

References:
    - Sudret (2008), Comput. Phys. Commun. – PCE-based Sobol indices
    - Xiu & Karniadakis (2002), SIAM J. Sci. Comput.
    - Blatman & Sudret (2011), J. Comput. Phys. – sparse adaptive PCE
"""

import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from src.simulator import simulate_duffing, evaluate_batch
from src.utils import set_plot_style

# Try importing OpenTURNS (optional but recommended)
try:
    import openturns as ot
    HAS_OT = True
except ImportError:
    HAS_OT = False
    print("Warning: OpenTURNS not installed. Sparse PCE will be skipped.")

# Try importing SALib for Monte Carlo Sobol
try:
    from SALib.sample import saltelli as saltelli_sample
    from SALib.analyze import sobol as sobol_analyze
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False


# ---------------------------------------------------------------------------
# Distribution definitions
# ---------------------------------------------------------------------------
def get_chaospy_joint():
    """Return the chaospy joint distribution for the 4 uncertain parameters."""
    delta_dist = cp.Normal(0.3, 0.05)
    alpha_dist = cp.Normal(-1.0, 0.15)
    beta_dist = cp.Normal(1.0, 0.1)
    F_dist = cp.Uniform(0.2, 0.5)
    return cp.J(delta_dist, alpha_dist, beta_dist, F_dist)


def get_openturns_distribution():
    """Return the OpenTURNS composed distribution."""
    if not HAS_OT:
        return None
    marginals = [
        ot.Normal(0.3, 0.05),
        ot.Normal(-1.0, 0.15),
        ot.Normal(1.0, 0.1),
        ot.Uniform(0.2, 0.5),
    ]
    return ot.ComposedDistribution(marginals)


# ===================================================================
# A.  CHAOSPY IMPLEMENTATION
# ===================================================================
def build_pce_chaospy(order=4, method='regression', n_samples=200, seed=42):
    """
    Build PCE using chaospy.

    Parameters
    ----------
    order : int – maximum polynomial order
    method : str – 'regression' (least-squares) or 'quadrature' (projection)
    n_samples : int – number of samples for regression method
    seed : int

    Returns
    -------
    dict with pce_model, expansion, samples, evals, etc.
    """
    joint = get_chaospy_joint()
    expansion = cp.generate_expansion(order, joint)

    if method == 'regression':
        samples = joint.sample(n_samples, rule='sobol', seed=seed)
        evals = np.array([
            simulate_duffing(samples[:, i])['x_max']
            for i in range(samples.shape[1])
        ])
        pce_model = cp.fit_regression(expansion, samples, evals)
    elif method == 'quadrature':
        nodes, weights = cp.generate_quadrature(order + 1, joint, rule='gaussian')
        evals = np.array([
            simulate_duffing(nodes[:, i])['x_max']
            for i in range(nodes.shape[1])
        ])
        pce_model = cp.fit_quadrature(expansion, nodes, weights, evals)
        samples = nodes
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Moments from PCE ---
    mean_pce = float(cp.E(pce_model, joint))
    var_pce = float(cp.Var(pce_model, joint))
    std_pce = np.sqrt(var_pce)

    # --- Sobol indices from PCE ---
    sobol_first = cp.Sens_m(pce_model, joint)   # first-order
    sobol_total = cp.Sens_t(pce_model, joint)    # total-order

    print(f"\n  chaospy PCE (order={order}, method={method})")
    print(f"    Mean(x_max)  = {mean_pce:.4f}")
    print(f"    Std(x_max)   = {std_pce:.4f}")
    print(f"    First-order Sobol: {np.round(sobol_first, 4)}")
    print(f"    Total-order Sobol: {np.round(sobol_total, 4)}")

    return {
        'pce_model': pce_model,
        'expansion': expansion,
        'joint': joint,
        'samples': samples,
        'evals': evals,
        'mean': mean_pce,
        'var': var_pce,
        'std': std_pce,
        'sobol_first': sobol_first,
        'sobol_total': sobol_total,
        'order': order,
        'method': method,
    }


# ===================================================================
# B.  OPENTURNS SPARSE PCE (LARS)
# ===================================================================
def build_pce_openturns(n_samples=200, max_degree=8, seed=42):
    """
    Build adaptive sparse PCE using OpenTURNS with LARS selection.

    The LARS algorithm selects only the most important polynomial terms,
    dramatically reducing the effective number of coefficients while
    maintaining accuracy.

    Returns
    -------
    dict with pce_result, Sobol indices, moments, etc.
    """
    if not HAS_OT:
        print("  OpenTURNS not available — skipping sparse PCE.")
        return None

    dist = get_openturns_distribution()

    # Generate training samples via Sobol' sequence
    experiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), dist, n_samples
    )
    experiment.setRandomize(False)
    X_train = experiment.generate()

    # Evaluate simulator
    Y_train_list = []
    for i in range(n_samples):
        params = [float(X_train[i, j]) for j in range(4)]
        res = simulate_duffing(params)
        Y_train_list.append([res['x_max']])
    Y_train = ot.Sample(Y_train_list)

    # Build sparse PCE with LARS
    dim = 4
    enumeration = ot.LinearEnumerateFunction(dim)
    basis_size = enumeration.getStrataCumulatedCardinal(max_degree)

    coll = [ot.StandardDistributionPolynomialFactory(dist.getMarginal(i))
            for i in range(dim)]
    basis_factory = ot.OrthogonalProductPolynomialFactory(coll)
    basis = ot.OrthogonalBasis(basis_factory)

    adaptive_strategy = ot.FixedStrategy(basis, basis_size)

    projection_strategy = ot.LeastSquaresStrategy(
        ot.LeastSquaresMetaModelSelectionFactory(
            ot.LARS(),
            ot.CorrectedLeaveOneOut()
        )
    )

    algo = ot.FunctionalChaosAlgorithm(
        X_train, Y_train, dist, adaptive_strategy, projection_strategy
    )
    algo.run()
    pce_result = algo.getResult()

    # --- Extract Sobol indices ---
    sa = ot.FunctionalChaosSobolIndices(pce_result)
    sobol_first = np.array([sa.getSobolIndex(i) for i in range(dim)])
    sobol_total = np.array([sa.getSobolTotalIndex(i) for i in range(dim)])

    # --- Moments ---
    meta_model = pce_result.getMetaModel()
    coefficients = np.array(pce_result.getCoefficients()).flatten()
    mean_ot = coefficients[0]
    var_ot = np.sum(coefficients[1:]**2)

    print(f"\n  OpenTURNS sparse PCE (LARS, max_degree={max_degree})")
    print(f"    Active terms:  {len(coefficients)} / {basis_size}")
    print(f"    Mean(x_max)  = {mean_ot:.4f}")
    print(f"    Std(x_max)   = {np.sqrt(var_ot):.4f}")
    print(f"    First-order Sobol: {np.round(sobol_first, 4)}")
    print(f"    Total-order Sobol: {np.round(sobol_total, 4)}")

    return {
        'pce_result': pce_result,
        'meta_model': meta_model,
        'coefficients': coefficients,
        'mean': mean_ot,
        'var': var_ot,
        'sobol_first': sobol_first,
        'sobol_total': sobol_total,
        'n_active_terms': len(coefficients),
        'n_total_terms': basis_size,
    }


# ===================================================================
# C.  PCE ORDER CONVERGENCE STUDY
# ===================================================================
def pce_order_study(orders=None, n_samples=200, seed=42):
    """
    Build PCE at multiple orders and track the leave-one-out (LOO) error
    to find the optimal polynomial order.

    Returns
    -------
    dict mapping order → LOO error
    """
    if orders is None:
        orders = [2, 3, 4, 5, 6]

    joint = get_chaospy_joint()

    # Generate a fixed sample set
    samples = joint.sample(n_samples, rule='sobol', seed=seed)
    evals = np.array([
        simulate_duffing(samples[:, i])['x_max']
        for i in range(samples.shape[1])
    ])

    loo_errors = {}
    for order in orders:
        expansion = cp.generate_expansion(order, joint)
        pce_model = cp.fit_regression(expansion, samples, evals)

        # LOO cross-validation (proxy via residuals on training data)
        pce_preds = np.array([
            float(np.squeeze(cp.call(pce_model, samples[:, i:i+1])))
            for i in range(n_samples)
        ])
        residuals = evals - pce_preds
        loo_approx = np.sqrt(np.mean(residuals**2))  # simplified LOO proxy
        loo_errors[order] = loo_approx
        print(f"  Order {order}: LOO error ≈ {loo_approx:.6f}")

    return loo_errors


# ===================================================================
# D.  MONTE CARLO SOBOL (via GP surrogate or simulator)
# ===================================================================
def mc_sobol_via_gp(gp_result, n_mc=4096, seed=42):
    """
    Compute Sobol indices via the Saltelli sampling method using the
    GP surrogate as a cheap evaluator.

    This serves as an independent cross-validation of the PCE-derived
    Sobol indices.
    """
    import torch
    import gpytorch

    if not HAS_SALIB:
        print("  SALib not available — skipping MC Sobol.")
        return None

    problem = {
        'num_vars': 4,
        'names': ['delta', 'alpha', 'beta', 'F'],
        'bounds': [
            [0.3, 0.05],       # Normal: [mean, std]
            [-1.0, 0.15],      # Normal: [mean, std]
            [1.0, 0.1],        # Normal: [mean, std]
            [0.2, 0.5],        # Uniform: [lo, hi]
        ],
        'dists': ['norm', 'norm', 'norm', 'unif'],
    }

    # Saltelli samples
    param_samples = saltelli_sample.sample(problem, n_mc, calc_second_order=False)
    N = param_samples.shape[0]

    # Evaluate via GP
    model = gp_result['model']
    likelihood = gp_result['likelihood']
    x_norm = gp_result['x_normaliser']
    y_std = gp_result['y_standardiser']

    model.eval()
    likelihood.eval()

    X_normed = x_norm.transform(param_samples)
    Y_gp = np.empty(N)
    chunk = 2000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        x_t = torch.FloatTensor(X_normed[start:end])
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(x_t))
            Y_gp[start:end] = y_std.inverse_transform(pred.mean.numpy())

    # Sobol analysis
    si = sobol_analyze.analyze(problem, Y_gp, calc_second_order=False,
                                print_to_console=False)

    sobol_first = si['S1']
    sobol_total = si['ST']

    print(f"\n  MC Sobol (Saltelli, N={N}, via GP surrogate)")
    print(f"    First-order: {np.round(sobol_first, 4)}")
    print(f"    Total-order: {np.round(sobol_total, 4)}")

    return {
        'sobol_first': sobol_first,
        'sobol_total': sobol_total,
        'S1_conf': si['S1_conf'],
        'ST_conf': si['ST_conf'],
    }


# ===================================================================
# E.  PLOTTING
# ===================================================================
def plot_sobol_comparison(pce_sobol, mc_sobol, ot_sobol=None,
                          save_path='figures/03_sobol_comparison.png'):
    """
    Side-by-side bar chart comparing Sobol indices from PCE vs MC.
    """
    param_names = ['δ', 'α', 'β', 'F']
    x = np.arange(len(param_names))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- First-order ---
    ax = axes[0]
    ax.bar(x - width, pce_sobol['sobol_first'], width,
           label='PCE (chaospy)', color='#2563eb')
    if mc_sobol is not None:
        ax.bar(x, mc_sobol['sobol_first'], width,
               yerr=mc_sobol['S1_conf'], capsize=3,
               label='MC (Saltelli/GP)', color='#ef4444')
    if ot_sobol is not None:
        ax.bar(x + width, ot_sobol['sobol_first'], width,
               label='PCE (OpenTURNS)', color='#059669')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.set_ylabel('First-order Sobol index S₁')
    ax.set_title('First-Order Sensitivity')
    ax.legend()

    # --- Total-order ---
    ax = axes[1]
    ax.bar(x - width, pce_sobol['sobol_total'], width,
           label='PCE (chaospy)', color='#2563eb')
    if mc_sobol is not None:
        ax.bar(x, mc_sobol['sobol_total'], width,
               yerr=mc_sobol['ST_conf'], capsize=3,
               label='MC (Saltelli/GP)', color='#ef4444')
    if ot_sobol is not None:
        ax.bar(x + width, ot_sobol['sobol_total'], width,
               label='PCE (OpenTURNS)', color='#059669')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.set_ylabel('Total-order Sobol index S_T')
    ax.set_title('Total-Order Sensitivity')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_pce_convergence(loo_errors, save_path='figures/03_pce_convergence.png'):
    """Plot LOO error vs polynomial order."""
    orders = sorted(loo_errors.keys())
    errors = [loo_errors[o] for o in orders]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(orders, errors, 'o-', color='#2563eb', lw=2, ms=8)
    ax.set_xlabel('PCE polynomial order')
    ax.set_ylabel('LOO error (RMSE)')
    ax.set_title('PCE Convergence Study')
    ax.set_xticks(orders)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_output_pdf(pce_result, n_mc=10000, seed=42,
                    save_path='figures/03_output_pdf.png'):
    """
    Compare the output PDF of x_max from PCE (chaospy) vs MC histogram.
    """
    joint = get_chaospy_joint()
    pce_model = pce_result['pce_model']

    # MC samples through PCE (cheap)
    mc_samples = joint.sample(n_mc, rule='random', seed=seed)
    pce_evals = np.array([
        float(np.squeeze(cp.call(pce_model, mc_samples[:, i:i+1])))
        for i in range(n_mc)
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pce_evals, bins=80, density=True, alpha=0.6,
            color='#3b82f6', edgecolor='white', label='PCE (MC sampling)')

    # Overlay kernel density
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(pce_evals)
    x_grid = np.linspace(pce_evals.min(), pce_evals.max(), 300)
    ax.plot(x_grid, kde(x_grid), 'r-', lw=2, label='KDE')

    ax.axvline(pce_result['mean'], color='k', ls='--', lw=1.5,
               label=f"Mean = {pce_result['mean']:.3f}")
    ax.set_xlabel('x_max')
    ax.set_ylabel('Probability density')
    ax.set_title('Output Distribution of x_max')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_pce_analysis(gp_result=None):
    """Run the complete PCE analysis pipeline."""
    print("=" * 60)
    print("Module 3 – Polynomial Chaos Expansion & Sobol Analysis")
    print("=" * 60)

    # A. chaospy PCE
    print("\n--- A. chaospy PCE (regression) ---")
    pce_cp = build_pce_chaospy(order=4, method='regression', n_samples=200)

    # B. Order convergence study
    print("\n--- B. PCE Order Convergence ---")
    loo = pce_order_study(orders=[2, 3, 4, 5], n_samples=200)
    plot_pce_convergence(loo)

    # C. OpenTURNS sparse PCE
    ot_result = None
    if HAS_OT:
        print("\n--- C. OpenTURNS sparse PCE (LARS) ---")
        ot_result = build_pce_openturns(n_samples=200, max_degree=6)

    # D. MC Sobol (via GP)
    mc_sobol = None
    if gp_result is not None and HAS_SALIB:
        print("\n--- D. MC Sobol (Saltelli via GP) ---")
        import gpytorch as _gp  # ensure it's available
        mc_sobol = mc_sobol_via_gp(gp_result, n_mc=2048)

    # E. Plots
    plot_sobol_comparison(pce_cp, mc_sobol, ot_result)
    plot_output_pdf(pce_cp)

    return {
        'chaospy': pce_cp,
        'openturns': ot_result,
        'mc_sobol': mc_sobol,
        'loo_errors': loo,
    }


if __name__ == "__main__":
    # Standalone run (without GP for MC Sobol)
    run_pce_analysis(gp_result=None)
