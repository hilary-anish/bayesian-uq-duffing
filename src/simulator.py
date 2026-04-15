"""
Module 1: Forced Duffing Oscillator Simulator
=============================================

Implements the forced, damped Duffing oscillator ODE:

    ẍ + δ·ẋ + α·x + β·x³ = F·cos(ω·t)

where:
    - x(t): displacement [m]
    - δ: damping coefficient [1/s]
    - α: linear stiffness [1/s²]  (α < 0 gives double-well potential)
    - β: cubic nonlinear stiffness [1/(m²·s²)]
    - F: forcing amplitude [m/s²]
    - ω: forcing frequency [rad/s]

The simulator uses scipy's DOP853 (8th-order Dormand-Prince) integrator
at tight tolerances to serve as a high-fidelity "expensive" reference.

Quantities of Interest (QoI):
    1. x_max: maximum absolute displacement over [0, T]
    2. steady_amplitude: amplitude of periodic response after transient decay
    3. energy: time-averaged total energy (kinetic + potential)

References:
    - Kovacic & Brennan (2011), "The Duffing Equation: Nonlinear Oscillators
      and their Behaviour", Wiley.
    - Xiu & Karniadakis (2002), SIAM J. Sci. Comput.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
import time


# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
OMEGA = 1.2        # forcing frequency [rad/s]
T_END = 50.0       # simulation duration [s]
N_POINTS = 5000    # number of output time-steps

# Parameter distributions (for reference / documentation)
PARAM_INFO = {
    'delta': {'dist': 'Normal', 'mean': 0.3,  'std': 0.05,  'desc': 'damping coefficient'},
    'alpha': {'dist': 'Normal', 'mean': -1.0, 'std': 0.15,  'desc': 'linear stiffness'},
    'beta':  {'dist': 'Normal', 'mean': 1.0,  'std': 0.1,   'desc': 'cubic stiffness'},
    'F':     {'dist': 'Uniform', 'lo': 0.2,   'hi': 0.5,    'desc': 'forcing amplitude'},
}

# Nominal (mean) parameter values
NOMINAL_PARAMS = np.array([0.3, -1.0, 1.0, 0.35])


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------
def duffing_rhs(t, y, delta, alpha, beta, F, omega):
    """
    Right-hand side of the Duffing ODE rewritten as a first-order system.

    State vector:  y = [x, ẋ]
    Equations:
        ẋ   = y[1]
        ẍ   = F·cos(ω·t) − δ·y[1] − α·y[0] − β·y[0]³

    Parameters
    ----------
    t : float – current time
    y : array-like, shape (2,) – state [displacement, velocity]
    delta, alpha, beta, F, omega : float – physical parameters

    Returns
    -------
    list of float – [ẋ, ẍ]
    """
    x, xdot = y
    xddot = F * np.cos(omega * t) - delta * xdot - alpha * x - beta * x**3
    return [xdot, xddot]


# ---------------------------------------------------------------------------
# Main simulator function
# ---------------------------------------------------------------------------
def simulate_duffing(params, t_end=T_END, n_points=N_POINTS, omega=OMEGA):
    """
    High-fidelity simulation of the Duffing oscillator.

    Parameters
    ----------
    params : array-like, shape (4,)
        [delta, alpha, beta, F] – uncertain physical parameters.
    t_end : float
        Simulation end time [s].
    n_points : int
        Number of uniformly spaced output points.
    omega : float
        Forcing frequency [rad/s].

    Returns
    -------
    dict with keys:
        'x_max'             : float – max|x(t)| over [0, t_end]
        'steady_amplitude'  : float – amplitude of steady-state response
        'energy'            : float – time-averaged total energy
        'trajectory'        : tuple (t, x, xdot) – full solution arrays
    """
    delta, alpha, beta, F = params
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(
        duffing_rhs,
        [0, t_end],
        [0.0, 0.0],                         # IC: x(0) = 0, ẋ(0) = 0
        args=(delta, alpha, beta, F, omega),
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    t = sol.t
    x = sol.y[0]
    xdot = sol.y[1]

    # --- QoI 1: maximum absolute displacement ---
    x_max = np.max(np.abs(x))

    # --- QoI 2: steady-state amplitude ---
    # Use the last 20 % of the signal (transients should have decayed)
    idx_steady = int(0.8 * len(x))
    x_steady = x[idx_steady:]
    steady_amplitude = (np.max(x_steady) - np.min(x_steady)) / 2.0

    # --- QoI 3: time-averaged total energy ---
    # E(t) = ½ ẋ² + ½ α x² + ¼ β x⁴
    KE = 0.5 * xdot**2
    PE = 0.5 * alpha * x**2 + 0.25 * beta * x**4
    energy = np.trapezoid(KE + PE, t) / t_end

    return {
        'x_max': x_max,
        'steady_amplitude': steady_amplitude,
        'energy': energy,
        'trajectory': (t, x, xdot),
    }


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------
def evaluate_batch(param_array, qoi_key='x_max', verbose=False):
    """
    Evaluate the simulator at many parameter combinations.

    Parameters
    ----------
    param_array : ndarray, shape (N, 4)
        Each row is [delta, alpha, beta, F].
    qoi_key : str
        Which QoI to return ('x_max', 'steady_amplitude', or 'energy').
    verbose : bool
        Print progress every 50 evaluations.

    Returns
    -------
    ndarray, shape (N,)
    """
    N = param_array.shape[0]
    results = np.empty(N)
    for i in range(N):
        res = simulate_duffing(param_array[i])
        results[i] = res[qoi_key]
        if verbose and (i + 1) % 50 == 0:
            print(f"  evaluated {i+1}/{N}")
    return results


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------
def benchmark_simulator(n_runs=10):
    """Time the simulator at nominal parameters."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        simulate_duffing(NOMINAL_PARAMS)
        times.append(time.perf_counter() - t0)
    mean_t = np.mean(times) * 1000  # ms
    std_t = np.std(times) * 1000
    print(f"Simulator cost: {mean_t:.1f} ± {std_t:.1f} ms per evaluation")
    return mean_t


# ---------------------------------------------------------------------------
# Quick validation (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Module 1 – Duffing Oscillator Simulator Validation")
    print("=" * 60)

    # Benchmark
    benchmark_simulator()

    # Nominal simulation
    res = simulate_duffing(NOMINAL_PARAMS)
    t, x, xdot = res['trajectory']

    print(f"\nNominal QoIs:")
    print(f"  x_max            = {res['x_max']:.4f}")
    print(f"  steady_amplitude = {res['steady_amplitude']:.4f}")
    print(f"  energy           = {res['energy']:.4f}")

    # --- Figure 1: Time history ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(t, x, lw=0.6, color='#2563eb')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Displacement x(t) [m]')
    axes[0].set_title('Time History')
    axes[0].grid(alpha=0.3)

    # --- Figure 2: Phase portrait ---
    axes[1].plot(x, xdot, lw=0.3, color='#dc2626')
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('ẋ [m/s]')
    axes[1].set_title('Phase Portrait')
    axes[1].grid(alpha=0.3)

    # --- Figure 3: Poincaré section ---
    # Sample the state at integer multiples of the forcing period
    T_forcing = 2 * np.pi / OMEGA
    poincare_times = np.arange(0, T_END, T_forcing)
    idx_poincare = np.searchsorted(t, poincare_times)
    idx_poincare = idx_poincare[idx_poincare < len(x)]
    axes[2].scatter(x[idx_poincare], xdot[idx_poincare], s=8, c='#059669')
    axes[2].set_xlabel('x [m]')
    axes[2].set_ylabel('ẋ [m/s]')
    axes[2].set_title('Poincaré Section')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/01_simulator_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: figures/01_simulator_validation.png")
