"""
EBIT spectra multi-baseline correction - Integrated version
Combines peak removal (Chauvenet) + Bayesian optimization + MSBC joint correction

Workflow:
  Step 1) Peak removal: Clean abnormal peaks in the specified region using the iterative Chauvenet method
  Step 2) Bayesian optimization: Optimize baseline correction parameters (lambda, p, mu0) in the background region
  Step 3) MSBC fitting: Perform multi-spectra joint baseline correction using the optimized parameters
  Step 4) Outputs: Baselines, corrected spectra, normalized spectra, and diagnostic plots

Dependencies: numpy, pandas, matplotlib, scipy, bayes_opt, openpyxl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm, normaltest, skew
from bayes_opt import BayesianOptimization
import time

# ==================== Configuration ====================
# I/O
INPUT_XLSX = 'spectra.xlsx'  # Input file (must include x0, y0, y4–y10)
OUTPUT_DIR = 'ebit_msbc_BO'  # Output directory

# Peak removal (Step 1)
PEAK_REGION = (20.0, 40.0)  # Peak region (nm)
CHAUVENET_TAU = 2.5  # Chauvenet threshold (sigma multiplier)
MAX_ITER_CHAUVENET = 20  # Maximum iterations
NORMALITY_PVAL = 0.05  # p-value threshold for normality test

# Bayesian optimization (Step 2)
BG_REGION = (10.0, 20.0)  # Background region used for optimization (nm)
BO_INIT_POINTS = 8  # Initial random samples for Bayesian optimization
BO_N_ITER = 25  # Iterations for Bayesian optimization

# Parameter search ranges — more conservative to force against overfitting
LOGLAM_BOUNDS = (3, 10)  # lambda: 1e3–1e10 (log10 bounds; narrowed operationally by penalties)
P_BOUNDS = (0.005, 0.5)  # p: 0.005–0.5 (minimum enforced)
LOGMU_BOUNDS = (7.5, 9.5)  # mu0: 3e7–3e9 (smoother baselines)

# MSBC algorithm
MSBC_MAX_ITER = 10  # MSBC iterations
ALS_TOL = 1e-6  # Convergence tolerance

# Data columns
WAVELENGTH_COL = 'x0'
REFERENCE_COL = 'y0'
SIGNAL_COLS = [f'y{i}' for i in range(4, 11)]

MAKE_PLOTS = True


# ==================== Utilities ====================
def finite_mask(arr):
    """Return a boolean mask of finite values."""
    return np.isfinite(arr)


def chauvenet_iterative(y, tau=2.5, max_iter=20, p_thresh=0.05, verbose=False):
    """
    Iterative outlier removal using the Chauvenet criterion.

    Args:
        y: 1D array
        tau: threshold (sigma multiplier)
        max_iter: maximum number of iterations
        p_thresh: p-value threshold for normality test
        verbose: whether to print progress

    Returns:
        keep: boolean mask (True = retained)
        stats: dict of statistics (mean, std, p_value, iterations, counts)
    """
    y = np.asarray(y, dtype=float)
    keep = finite_mask(y).copy()
    p_val = np.nan

    for it in range(1, max_iter + 1):
        if np.sum(keep) < 3:
            break

        mu = np.mean(y[keep])
        sigma = np.std(y[keep], ddof=1)

        if not np.isfinite(sigma) or sigma == 0:
            break

        deviation = np.abs(y - mu) / sigma
        new_keep = keep & (deviation < tau)

        # Normality test on the retained subset
        if np.sum(new_keep) >= 8:
            _, p_val = normaltest(y[new_keep])

        if verbose:
            n_removed = np.sum(keep & ~new_keep)
            print(f'  Iter {it}: removed {n_removed}, kept {np.sum(new_keep)}, p={p_val:.3g}')

        # Stop if normality achieved or mask stabilized
        if (p_val == p_val and p_val > p_thresh) or np.array_equal(new_keep, keep):
            keep = new_keep
            break

        keep = new_keep

    stats = {
        'mu': np.mean(y[keep]) if np.any(keep) else np.nan,
        'sigma': np.std(y[keep], ddof=1) if np.sum(keep) > 1 else np.nan,
        'p_value': p_val,
        'iterations': it,
        'n_kept': np.sum(keep),
        'n_removed': len(y) - np.sum(keep)
    }

    return keep, stats


# ==================== MSBC Core Algorithm ====================
def msbc(data, lambda_param, mu0, p=0.001, max_iter=10, tol=1e-6):
    """
    Multi-Spectra Baseline Correction (MSBC) — robust version.

    Args:
        data: array (m, n), m spectra, n data points each
        lambda_param: fitting error weight
        mu0: smoothness parameter (scalar or length-m array)
        p: asymmetric weight
        max_iter: maximum iterations
        tol: convergence tolerance (relative change of baseline)

    Returns:
        z: (m, n) estimated baselines
        a: relaxation factor history
        converged: boolean flag indicating convergence
    """
    m, n = data.shape

    # Preprocess: handle NaNs with linear interpolation
    data_clean = np.copy(data)
    for k in range(m):
        mask = np.isfinite(data[k, :])
        if np.sum(mask) < n * 0.5:
            raise ValueError(f"Spectrum {k} has fewer than 50% valid data points.")
        if np.any(~mask):
            x_valid = np.where(mask)[0]
            y_valid = data[k, mask]
            x_invalid = np.where(~mask)[0]
            data_clean[k, x_invalid] = np.interp(x_invalid, x_valid, y_valid)

    # Handle mu parameter (scalar or per-spectrum array)
    if np.isscalar(mu0):
        mu = np.ones(m) * mu0
    else:
        mu = np.array(mu0)

    # Second-order difference matrix for smoothness
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    DTD = D.T @ D

    # Initialization
    w0 = np.ones((m, n))
    # Initial baseline via low percentile to avoid peak bias
    z0 = np.percentile(data_clean, 10, axis=1).reshape(-1, 1) * np.ones((1, n))
    a0 = np.ones(m)

    z = np.zeros((m, n))
    z_prev = np.copy(z0)
    a0_prev = np.ones(m)

    converged = False

    for iteration in range(max_iter):
        for k in range(m):
            try:
                # Ensure valid weights
                w_k = np.clip(w0[k, :], 1e-8, 1.0)
                Q_k = sparse.diags(w_k, 0, shape=(n, n))
                g_k = np.clip(a0[k] * (2 - a0[k]), 0.1, 2.0)

                # Tiny regularization to avoid singularity
                regularization = 1e-8 * sparse.eye(n)

                # Linear system
                A = lambda_param * Q_k + mu[k] * DTD + (m - g_k) * sparse.eye(n) + regularization
                b = (m * data_clean[k, :] - g_k * z0[k, :] -
                     g_k * (data_clean - z0).sum(axis=0) +
                     lambda_param * data_clean[k, :] * w_k)

                # Ensure b is finite
                b = np.nan_to_num(b, nan=0.0, posinf=1e10, neginf=-1e10)

                # Solve
                z[k, :] = spsolve(A.tocsr(), b)

                # Ensure baseline is finite
                z[k, :] = np.nan_to_num(z[k, :], nan=np.nanmin(data_clean[k, :]))

            except Exception as e:
                print(f"  Warning: spectrum {k} solve failed, using a simple baseline: {e}")
                # Fallback: rolling minimum
                window = max(3, n // 50)
                z[k, :] = pd.Series(data_clean[k, :]).rolling(window, center=True, min_periods=1).min().values

            # Update asymmetric weights
            w0[k, :] = (1 - p) * (data_clean[k, :] <= z[k, :]) + p * (data_clean[k, :] > z[k, :])
            w0[k, :] = np.clip(w0[k, :], 1e-8, 1.0)
            z0[k, :] = z[k, :]

        # Update relaxation factors a0
        theta = (data_clean - z).mean(axis=0)
        theta_norm = np.dot(theta, theta)
        if theta_norm > 1e-12:
            a0 = ((data_clean - z) @ theta) / theta_norm
            a0 = np.clip(a0, 0.5, 1.5)
        else:
            a0 = np.ones(m)

        # Convergence check: both z and a0 must stabilize
        z_change = np.linalg.norm(z - z_prev) / max(np.linalg.norm(z_prev), 1e-12)
        a0_change = np.linalg.norm(a0 - a0_prev) / max(np.linalg.norm(a0_prev), 1e-12)
        if z_change < tol and a0_change < tol:
            converged = True
            break
        z_prev = np.copy(z)
        a0_prev = a0.copy()

    return z, converged


# ==================== Step 1: Peak removal ====================
def step1_peak_removal(df, peak_region, tau, max_iter, p_thresh):
    """Step 1: Apply Chauvenet-based outlier removal within the peak region."""

    print("\n" + "=" * 70)
    print("Step 1: Peak removal (Chauvenet)")
    print("=" * 70)

    step1_dir = os.path.join(OUTPUT_DIR, 'step1_peak_removal')
    os.makedirs(step1_dir, exist_ok=True)

    x = df[WAVELENGTH_COL].values
    peak_mask = (x >= peak_region[0]) & (x <= peak_region[1])

    signal_cols = [c for c in SIGNAL_COLS if c in df.columns]
    cleaned_data = {WAVELENGTH_COL: x}
    stats_list = []

    for col in signal_cols:
        print(f"\nProcessing {col}:")
        y = df[col].values.astype(float)
        keep = np.ones_like(y, dtype=bool)

        if np.any(peak_mask):
            keep_peak, stats = chauvenet_iterative(
                y[peak_mask], tau=tau, max_iter=max_iter,
                p_thresh=p_thresh, verbose=True
            )
            keep[peak_mask] = keep_peak
            stats_list.append({'column': col, **stats})

        # Save cleaned data (removed points set to NaN)
        y_clean = y.copy()
        y_clean[~keep] = np.nan
        cleaned_data[col] = y_clean
        cleaned_data[f'{col}_mask'] = keep.astype(int)

        # Visualization
        if MAKE_PLOTS:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Subplot 1: before vs after removal
            axes[0].plot(x, y, 'gray', alpha=0.4, label='Original')
            axes[0].plot(x[keep], y[keep], 'b.', markersize=3, label='Retained')
            if np.any(~keep):
                axes[0].plot(x[~keep], y[~keep], 'rx', markersize=5, label='Removed')
            axes[0].axvspan(peak_region[0], peak_region[1], alpha=0.1, color='red', label='Peak region')
            axes[0].set_xlabel('Wavelength (nm)')
            axes[0].set_ylabel('Intensity')
            axes[0].set_title(f'{col}: Peak Removal (τ={tau}, iters={stats["iterations"]})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Subplot 2: histogram of retained values
            kept_vals = y[keep & peak_mask]
            if kept_vals.size > 0:
                axes[1].hist(kept_vals, bins=30, density=True, alpha=0.7, label='Retained')
                mu, sigma = stats['mu'], stats['sigma']
                if np.isfinite(sigma) and sigma > 0:
                    xpdf = np.linspace(kept_vals.min(), kept_vals.max(), 200)
                    axes[1].plot(xpdf, norm.pdf(xpdf, mu, sigma), 'r-', lw=2,
                                 label=f'N({mu:.2g}, {sigma:.2g}²)')
                axes[1].set_xlabel('Intensity')
                axes[1].set_ylabel('Density')
                axes[1].set_title(f'Retained Distribution (p={stats["p_value"]:.3g})')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(step1_dir, f'{col}_removal.png'), dpi=200)
            plt.close()

    # Save cleaned data and stats
    cleaned_path = os.path.join(step1_dir, 'cleaned_data.xlsx')
    pd.DataFrame(cleaned_data).to_excel(cleaned_path, index=False)

    stats_path = os.path.join(step1_dir, 'removal_stats.xlsx')
    pd.DataFrame(stats_list).to_excel(stats_path, index=False)

    print(f"\n✓ Cleaned data saved: {cleaned_path}")
    print(f"✓ Statistics saved: {stats_path}")

    return pd.DataFrame(cleaned_data), signal_cols


# ==================== Step 2: Bayesian optimization ====================
def step2_bayesian_optimization(df_clean, df_orig, signal_cols, bg_region):
    """Step 2: Use Bayesian optimization in the background region to find optimal parameters."""

    print("\n" + "=" * 70)
    print("Step 2: Bayesian optimization of parameters")
    print("=" * 70)

    step2_dir = os.path.join(OUTPUT_DIR, 'step2_optimization')
    os.makedirs(step2_dir, exist_ok=True)

    x = df_clean[WAVELENGTH_COL].values
    bg_mask = (x >= bg_region[0]) & (x <= bg_region[1])

    if np.sum(bg_mask) < 5:
        raise ValueError(f"Too few data points in background region {bg_region}")

    # Prepare spectra matrix
    spectra_data = []
    for col in signal_cols:
        y = df_clean[col].values
        valid = finite_mask(y)
        if np.sum(valid & bg_mask) < 3:
            print(f"Warning: insufficient valid points in background for {col}, skipped")
            continue
        spectra_data.append(y)

    spectra_data = np.array(spectra_data)
    m, n = spectra_data.shape

    print(f"\nObjective: minimize residual MSE in the background region")
    print(f"Background region: {bg_region[0]:.1f}–{bg_region[1]:.1f} nm ({np.sum(bg_mask)} points)")
    print(f"Number of spectra: {m}")

    # Define objective with MSE-relative penalties for consistent scaling
    def objective(log_lam, p, log_mu):
        lam = 10 ** log_lam
        mu = 10 ** log_mu

        if p < 0.008 or p > 0.08:
            return -1e20

        try:
            z, converged = msbc(spectra_data, lam, mu, p, max_iter=MSBC_MAX_ITER, tol=ALS_TOL)

            residuals = (spectra_data - z)[:, bg_mask]
            residuals = residuals[finite_mask(residuals)]

            if residuals.size < 3:
                return -1e30

            mse = np.mean(residuals ** 2)
            mse_ref = max(mse, 1e-12)

            penalties = 0.0

            if not converged:
                penalties += 10.0 * mse_ref

            if residuals.size >= 8:
                _, p_norm = normaltest(residuals)
                if p_norm < 0.05:
                    penalties += mse_ref * (1.0 - p_norm / 0.05)

            residual_skew = abs(skew(residuals))
            if residual_skew > 1.0:
                penalties += mse_ref * (residual_skew - 1.0)

            for k in range(z.shape[0]):
                data_median = np.median(spectra_data[k, :])
                baseline_mean = np.mean(z[k, :])
                if abs(data_median) > 1e-12:
                    ratio = baseline_mean / data_median
                    if ratio > 0.8:
                        penalties += mse_ref * (ratio - 0.8) * 5.0
                    if ratio < 0.0:
                        penalties += mse_ref * abs(ratio) * 5.0

            total_loss = mse + penalties
            return -float(total_loss)

        except Exception as e:
            print(f"    Evaluation failed (lambda={lam:.1e}, p={p:.4f}, mu={mu:.1e}): {e}")
            return -1e30

    # Run Bayesian optimization
    print(f"\nStarting Bayesian optimization...")
    print(f"  Initial samples: {BO_INIT_POINTS}")
    print(f"  Iterations: {BO_N_ITER}")

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            'log_lam': LOGLAM_BOUNDS,
            'p': P_BOUNDS,
            'log_mu': LOGMU_BOUNDS
        },
        random_state=42,
        verbose=2
    )

    start_time = time.time()
    optimizer.maximize(init_points=BO_INIT_POINTS, n_iter=BO_N_ITER)
    elapsed = time.time() - start_time

    # Best parameters
    best_params = optimizer.max['params']
    best_lambda = 10 ** best_params['log_lam']
    best_p = best_params['p']
    best_mu = 10 ** best_params['log_mu']
    best_score = optimizer.max['target']

    print(f"\n✓ Optimization finished! Elapsed {elapsed:.2f} s")
    print(f"\nBest parameters:")
    print(f"  lambda = {best_lambda:.3e}")
    print(f"  p = {best_p:.6f}")
    print(f"  mu0 = {best_mu:.3e}")
    print(f"  objective value = {best_score:.3e}")

    # Sanity checks
    print(f"\nParameter sanity checks:")
    if best_p < 0.005:
        print(f"  ⚠️  p too small ({best_p:.6f}) may cause overfitting.")
        print(f"     Suggestion: set p = 0.01 and rerun.")
    elif best_p > 0.05:
        print(f"  ⚠️  p too large ({best_p:.6f}) may lift the baseline into peak regions.")
        print(f"     Suggestion: verify that the background region is appropriate.")
    else:
        print(f"  ✓ p is reasonable.")

    if best_lambda < 1e6:
        print(f"  ⚠️  lambda too small; baseline may be insufficiently smooth.")
    elif best_lambda > 5e7:
        print(f"  ⚠️  lambda too large; potential over-smoothing.")
    else:
        print(f"  ✓ lambda is reasonable.")

    if best_mu < 1e7:
        print(f"  ⚠️  mu0 too small; smoothness may be insufficient.")
    elif best_mu > 5e9:
        print(f"  ⚠️  mu0 too large; baseline may be over-smoothed.")
    else:
        print(f"  ✓ mu0 is reasonable.")

    # Save optimization history
    history = []
    for i, res in enumerate(optimizer.res):
        history.append({
            'iteration': i + 1,
            'lambda': 10 ** res['params']['log_lam'],
            'p': res['params']['p'],
            'mu0': 10 ** res['params']['log_mu'],
            'neg_mse': res['target']
        })

    history_path = os.path.join(step2_dir, 'optimization_history.xlsx')
    pd.DataFrame(history).to_excel(history_path, index=False)
    print(f"\n✓ Optimization history saved: {history_path}")

    # Visualization of optimization progress
    if MAKE_PLOTS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        iters = [h['iteration'] for h in history]

        axes[0, 0].plot(iters, [h['lambda'] for h in history], 'o-')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Lambda')
        axes[0, 0].set_title('Lambda Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(best_lambda, color='r', linestyle='--', label='Best')
        axes[0, 0].legend()

        axes[0, 1].plot(iters, [h['p'] for h in history], 'o-')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('p')
        axes[0, 1].set_title('p Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(best_p, color='r', linestyle='--', label='Best')
        axes[0, 1].legend()

        axes[1, 0].plot(iters, [h['mu0'] for h in history], 'o-')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('mu0')
        axes[1, 0].set_title('mu0 Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(best_mu, color='r', linestyle='--', label='Best')
        axes[1, 0].legend()

        axes[1, 1].plot(iters, [-h['neg_mse'] for h in history], 'o-')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('MSE Evolution (lower is better)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(step2_dir, 'optimization_progress.png'), dpi=200)
        plt.close()

    return best_lambda, best_p, best_mu


# ==================== Step 3: MSBC baseline correction ====================
def step3_msbc_baseline_correction(df_clean, df_orig, signal_cols,
                                   best_lambda, best_p, best_mu):
    """Step 3: Apply MSBC using the optimized parameters."""

    print("\n" + "=" * 70)
    print("Step 3: MSBC multi-spectra baseline correction")
    print("=" * 70)

    step3_dir = os.path.join(OUTPUT_DIR, 'step3_baseline_correction')
    os.makedirs(step3_dir, exist_ok=True)

    x = df_clean[WAVELENGTH_COL].values
    y0 = df_orig[REFERENCE_COL].values
    y0_safe = np.where(finite_mask(y0) & (np.abs(y0) > 1e-10), y0, 1.0)

    # Prepare spectra lists
    spectra_clean = []
    spectra_orig = []
    valid_cols = []

    for col in signal_cols:
        y_clean = df_clean[col].values
        y_orig = df_orig[col].values

        if np.sum(finite_mask(y_clean)) < 10:
            print(f"Warning: too few valid points in {col}, skipped")
            continue

        spectra_clean.append(y_clean)
        spectra_orig.append(y_orig)
        valid_cols.append(col)

    spectra_clean = np.array(spectra_clean)
    spectra_orig = np.array(spectra_orig)

    print(f"\nUsing parameters:")
    print(f"  lambda = {best_lambda:.3e}")
    print(f"  p = {best_p:.6f}")
    print(f"  mu0 = {best_mu:.3e}")
    print(f"\nProcessing {len(valid_cols)} spectra...")

    # Run MSBC
    start_time = time.time()
    baselines, converged = msbc(
        spectra_clean, best_lambda, best_mu, best_p,
        max_iter=MSBC_MAX_ITER, tol=ALS_TOL
    )
    elapsed = time.time() - start_time

    print(f"✓ MSBC finished! Elapsed {elapsed:.3f} s")
    print(f"  Converged: {'Yes' if converged else 'No'}")

    # Compute results
    corrected = spectra_orig - baselines
    normalized = corrected / y0_safe

    # Save results (three separate files)
    results = {
        WAVELENGTH_COL: x,
        REFERENCE_COL: y0
    }

    for i, col in enumerate(valid_cols):
        results[f'{col}_baseline'] = baselines[i]
        results[f'{col}_corrected'] = corrected[i]
        results[f'{col}_normalized'] = normalized[i]

    baseline_path = os.path.join(step3_dir, 'baselines.xlsx')
    corrected_path = os.path.join(step3_dir, 'corrected_spectra.xlsx')
    normalized_path = os.path.join(step3_dir, 'normalized_spectra.xlsx')

    # Baselines
    baseline_df = pd.DataFrame({WAVELENGTH_COL: x})
    for i, col in enumerate(valid_cols):
        baseline_df[f'{col}_baseline'] = baselines[i]
    baseline_df.to_excel(baseline_path, index=False)

    # Corrected spectra
    corrected_df = pd.DataFrame({WAVELENGTH_COL: x})
    for i, col in enumerate(valid_cols):
        corrected_df[f'{col}_corrected'] = corrected[i]
    corrected_df.to_excel(corrected_path, index=False)

    # Normalized spectra
    normalized_df = pd.DataFrame({WAVELENGTH_COL: x, REFERENCE_COL: y0})
    for i, col in enumerate(valid_cols):
        normalized_df[f'{col}_normalized'] = normalized[i]
    normalized_df.to_excel(normalized_path, index=False)

    print(f"\n✓ Baselines saved: {baseline_path}")
    print(f"✓ Corrected spectra saved: {corrected_path}")
    print(f"✓ Normalized spectra saved: {normalized_path}")

    return baselines, corrected, normalized, valid_cols, x, y0


# ==================== Step 4: Visualization ====================
def step4_visualization(x, baselines, corrected, normalized, valid_cols,
                        spectra_orig, y0, bg_region, peak_region):
    """Step 4: Generate consolidated diagnostic plots."""

    print("\n" + "=" * 70)
    print("Step 4: Generate visualizations")
    print("=" * 70)

    step4_dir = os.path.join(OUTPUT_DIR, 'step4_visualization')
    os.makedirs(step4_dir, exist_ok=True)

    m = len(valid_cols)
    colors = plt.cm.viridis(np.linspace(0, 1, m))

    # Figure 1: overview
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # (a) Original spectra + baselines
    for i, col in enumerate(valid_cols):
        axes[0].plot(x, spectra_orig[i], alpha=0.5, color=colors[i], label=col)
        axes[0].plot(x, baselines[i], '--', color=colors[i], linewidth=2)
    axes[0].axvspan(bg_region[0], bg_region[1], alpha=0.1, color='green', label='BG region')
    axes[0].axvspan(peak_region[0], peak_region[1], alpha=0.1, color='red', label='Peak region')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('(a) Original Spectra and Baselines')
    axes[0].legend(loc='best', fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    # (b) Corrected spectra
    for i, col in enumerate(valid_cols):
        axes[1].plot(x, corrected[i], color=colors[i], label=col)
    axes[1].axvspan(bg_region[0], bg_region[1], alpha=0.1, color='green')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title('(b) Baseline-Corrected Spectra')
    axes[1].legend(loc='best', fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)

    # (c) Normalized spectra
    for i, col in enumerate(valid_cols):
        axes[2].plot(x, normalized[i], color=colors[i], label=col)
    axes[2].axvspan(bg_region[0], bg_region[1], alpha=0.1, color='green')
    axes[2].set_xlabel('Wavelength (nm)')
    axes[2].set_ylabel('Normalized Intensity')
    axes[2].set_title('(c) Normalized: (Original - Baseline) / y0')
    axes[2].legend(loc='best', fontsize=8, ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(step4_dir, 'comprehensive_results.png'), dpi=200)
    plt.close()

    # Figure 2: residual histograms in background region
    bg_mask = (x >= bg_region[0]) & (x <= bg_region[1])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        if i >= 8:
            break
        residuals = corrected[i][bg_mask]
        residuals = residuals[finite_mask(residuals)]

        if residuals.size > 0:
            axes[i].hist(residuals, bins=30, density=True, alpha=0.7, color=colors[i])
            mu, sigma = np.mean(residuals), np.std(residuals, ddof=1)
            xpdf = np.linspace(residuals.min(), residuals.max(), 200)
            axes[i].plot(xpdf, norm.pdf(xpdf, mu, sigma), 'r-', lw=2)

            # Normality test
            if residuals.size >= 8:
                _, p_val = normaltest(residuals)
                axes[i].set_title(f'{col}\nμ={mu:.2e}, σ={sigma:.2e}\np={p_val:.3g}', fontsize=9)
            else:
                axes[i].set_title(f'{col}\nμ={mu:.2e}, σ={sigma:.2e}', fontsize=9)

            axes[i].set_xlabel('Residual')
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(valid_cols), 8):
        axes[i].axis('off')

    plt.suptitle('Background Region Residual Histograms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(step4_dir, 'residual_histograms.png'), dpi=200)
    plt.close()

    print(f"✓ Overview and residual histogram figures saved.")


# ==================== Main pipeline ====================
def main():
    """Main pipeline: integrate peak removal, Bayesian optimization, and MSBC baseline correction."""

    print("\n" + "=" * 70)
    print("EBIT spectra multi-baseline correction — Full pipeline")
    print("=" * 70)
    print(f"\nInput file: {INPUT_XLSX}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nPeak region: {PEAK_REGION[0]:.1f}–{PEAK_REGION[1]:.1f} nm")
    print(f"Background region: {BG_REGION[0]:.1f}–{BG_REGION[1]:.1f} nm")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read original data
    print("\nReading input data...")
    try:
        df_orig = pd.read_excel(INPUT_XLSX)
        print(f"✓ Loaded successfully: {df_orig.shape[0]} rows × {df_orig.shape[1]} columns")
    except Exception as e:
        print(f"✗ Failed to read input: {e}")
        return

    # Step 1: Peak removal
    df_clean, signal_cols = step1_peak_removal(
        df_orig, PEAK_REGION, CHAUVENET_TAU,
        MAX_ITER_CHAUVENET, NORMALITY_PVAL
    )

    # Step 2: Parameter selection — two modes available
    print("\n" + "=" * 70)
    print("Parameter selection mode")
    print("=" * 70)
    print("Options:")
    print("  1. Use recommended fixed parameters (fast, robust)")
    print("  2. Use Bayesian optimization (time-consuming, may overfit)")

    use_bayesian = True  # Set True to enable Bayesian optimization

    if use_bayesian:
        print("\nUsing Bayesian optimization...")
        best_lambda, best_p, best_mu = step2_bayesian_optimization(
            df_clean, df_orig, signal_cols, BG_REGION
        )
    else:
        print("\nUsing recommended fixed parameters (tuned for EBIT X-ray spectra)")
        # Empirically robust defaults
        best_lambda = 1e7  # strong smoothing
        best_p = 0.01  # conservative asymmetry
        best_mu = 5e8  # strong smoothness for MSBC

        print(f"\nRecommended parameters:")
        print(f"  lambda = {best_lambda:.3e}  (fit vs smoothness)")
        print(f"  p = {best_p:.6f}        (asymmetric weight)")
        print(f"  mu0 = {best_mu:.3e}     (MSBC smoothness)")
        print(f"\nThese parameters are tuned for typical EBIT spectra.")
        print(f"Adjust values in code if needed.")

    # Step 3: MSBC baseline correction
    baselines, corrected, normalized, valid_cols, x, y0 = step3_msbc_baseline_correction(
        df_clean, df_orig, signal_cols, best_lambda, best_p, best_mu
    )

    # Prepare original spectra array for visualization
    spectra_orig = np.array([df_orig[col].values for col in valid_cols])

    # Step 4: Visualization
    if MAKE_PLOTS:
        step4_visualization(
            x, baselines, corrected, normalized, valid_cols,
            spectra_orig, y0, BG_REGION, PEAK_REGION
        )

    # Save final parameter summary
    summary = {
        'parameter': ['lambda', 'p', 'mu0', 'peak_region_left', 'peak_region_right',
                      'bg_region_left', 'bg_region_right', 'chauvenet_tau',
                      'max_iter_chauvenet', 'msbc_max_iter', 'method'],
        'value': [best_lambda, best_p, best_mu, PEAK_REGION[0], PEAK_REGION[1],
                  BG_REGION[0], BG_REGION[1], CHAUVENET_TAU,
                  MAX_ITER_CHAUVENET, MSBC_MAX_ITER,
                  'Bayesian' if use_bayesian else 'Fixed']
    }
    summary_path = os.path.join(OUTPUT_DIR, 'processing_summary.xlsx')
    pd.DataFrame(summary).to_excel(summary_path, index=False)

    print("\n" + "=" * 70)
    print("🎉 Processing complete!")
    print("=" * 70)
    print(f"\nOutputs located at: {OUTPUT_DIR}/")
    print("\nDirectory structure:")
    print("  step1_peak_removal/")
    print("    ├─ cleaned_data.xlsx          # data after peak removal (NaNs for removed points)")
    print("    ├─ removal_stats.xlsx         # statistics of removal")
    print("    └─ y*_removal.png             # visualizations per column")
    if use_bayesian:
        print("\n  step2_optimization/")
        print("    ├─ optimization_history.xlsx  # Bayesian optimization history")
        print("    └─ optimization_progress.png  # parameter evolution plots")
    print("\n  step3_baseline_correction/")
    print("    ├─ baselines.xlsx             # fitted baselines")
    print("    ├─ corrected_spectra.xlsx     # baseline-corrected spectra")
    print("    └─ normalized_spectra.xlsx    # normalized spectra")
    print("\n  step4_visualization/")
    print("    ├─ comprehensive_results.png  # overview figure")
    print("    └─ residual_histograms.png    # residual distributions")
    print("\n  processing_summary.xlsx        # summary of parameters")

    print("\n" + "=" * 70)
    print("Parameter summary:")
    print("=" * 70)
    print(f"  Final lambda: {best_lambda:.3e}")
    print(f"  Final p:      {best_p:.6f}")
    print(f"  Final mu0:    {best_mu:.3e}")
    print(f"  Method:       {'Bayesian optimization' if use_bayesian else 'Fixed parameters'}")
    print("=" * 70)


if __name__ == '__main__':
    # Usage banner
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║       EBIT Spectra Multi-Baseline Correction — User Guide         ║
╚═══════════════════════════════════════════════════════════════════╝

[Data requirements]
  The Excel file must contain the following columns:
    - x0:       Wavelength/Energy (1st column)
    - y0:       Reference spectrum (4th column, used for normalization)
    - y4–y10:   Spectra under different beam energies (5th–11th columns)

[Parameter notes]
  1) Peak removal parameters:
     - PEAK_REGION:         region in nm containing peaks
     - CHAUVENET_TAU:       removal threshold (recommended 2–3)
     - MAX_ITER_CHAUVENET:  max iterations (recommended 10–30)

  2) Bayesian optimization parameters:
     - BG_REGION:           background region for optimization
     - LOGLAM_BOUNDS:       search range for lambda (log10)
     - P_BOUNDS:            search range for p
     - LOGMU_BOUNDS:        search range for mu0 (log10)
     - BO_N_ITER:           optimization iterations (recommend 20–50)

  3) MSBC parameters:
     - MSBC_MAX_ITER:       maximum MSBC iterations
     - ALS_TOL:             convergence tolerance

[Quick start]
  1) Set INPUT_XLSX to your file name
  2) Adjust PEAK_REGION and BG_REGION to your spectra
  3) Run: python script.py
  4) Check results in OUTPUT_DIR

[Recommended workflow]
  Step 1: Run with default settings once
  Step 2: Inspect step1_peak_removal/ to confirm peak removal
  Step 3: Inspect step2_optimization/ to confirm parameter trends
  Step 4: If results are not satisfactory, adjust parameters and rerun

[Cautions]
  • Peak and background regions should not overlap
  • Choose a flat, featureless background region
  • If optimization doesn’t converge, increase BO_N_ITER
  • For noisy data, increase CHAUVENET_TAU

════════════════════════════════════════════════════════════════════
""")

    # Execute main pipeline
    main()
