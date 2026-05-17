import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm, normaltest, skew

import odatse
from odatse.algorithm import choose_algorithm


# ==================== User configuration ====================
# You can also move these into TOML later if needed.
INPUT_XLSX = "spectra.xlsx"
OUTPUT_DIR_FALLBACK = "./output_msbc_odat"

PEAK_REGION = (20.0, 40.0)      # Region for Chauvenet-based peak cleaning
BG_REGION = (10.0, 20.0)        # Background region used for objective evaluation

CHAUVENET_TAU = 2.5
MAX_ITER_CHAUVENET = 20
NORMALITY_PVAL = 0.05

MSBC_MAX_ITER = 10
ALS_TOL = 1e-6

WAVELENGTH_COL = "x0"
REFERENCE_COL = "y0"
SIGNAL_COLS = [f"y{i}" for i in range(4, 11)]

MAKE_FINAL_PLOTS = True


# ==================== Utilities ====================
def finite_mask(arr):
    return np.isfinite(arr)


def chauvenet_iterative(y, tau=2.5, max_iter=20, p_thresh=0.05, verbose=False):
    y = np.asarray(y, dtype=float)
    keep = finite_mask(y).copy()
    p_val = np.nan
    it = 0

    for it in range(1, max_iter + 1):
        if np.sum(keep) < 3:
            break

        mu = np.mean(y[keep])
        sigma = np.std(y[keep], ddof=1)

        if not np.isfinite(sigma) or sigma == 0:
            break

        deviation = np.abs(y - mu) / sigma
        new_keep = keep & (deviation < tau)

        if np.sum(new_keep) >= 8:
            _, p_val = normaltest(y[new_keep])

        if verbose:
            n_removed = np.sum(keep & ~new_keep)
            print(f"  Iter {it}: removed {n_removed}, kept {np.sum(new_keep)}, p={p_val:.3g}")

        if (p_val == p_val and p_val > p_thresh) or np.array_equal(new_keep, keep):
            keep = new_keep
            break

        keep = new_keep

    stats = {
        "mu": np.mean(y[keep]) if np.any(keep) else np.nan,
        "sigma": np.std(y[keep], ddof=1) if np.sum(keep) > 1 else np.nan,
        "p_value": p_val,
        "iterations": it,
        "n_kept": int(np.sum(keep)),
        "n_removed": int(len(y) - np.sum(keep)),
    }
    return keep, stats


# ==================== MSBC core ====================
def msbc(data, lambda_param, mu0, p=0.001, max_iter=10, tol=1e-6):
    m, n = data.shape

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

    if np.isscalar(mu0):
        mu = np.ones(m) * mu0
    else:
        mu = np.array(mu0, dtype=float)

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    DTD = D.T @ D

    w0 = np.ones((m, n))
    z0 = np.percentile(data_clean, 10, axis=1).reshape(-1, 1) * np.ones((1, n))
    a0 = np.ones(m)

    z = np.zeros((m, n))
    z_prev = np.copy(z0)
    a0_prev = np.ones(m)
    converged = False

    for _ in range(max_iter):
        for k in range(m):
            try:
                w_k = np.clip(w0[k, :], 1e-8, 1.0)
                Q_k = sparse.diags(w_k, 0, shape=(n, n))
                g_k = np.clip(a0[k] * (2 - a0[k]), 0.1, 2.0)
                regularization = 1e-8 * sparse.eye(n)

                A = lambda_param * Q_k + mu[k] * DTD + (m - g_k) * sparse.eye(n) + regularization
                b = (
                    m * data_clean[k, :]
                    - g_k * z0[k, :]
                    - g_k * (data_clean - z0).sum(axis=0)
                    + lambda_param * data_clean[k, :] * w_k
                )

                b = np.nan_to_num(b, nan=0.0, posinf=1e10, neginf=-1e10)
                z[k, :] = spsolve(A.tocsr(), b)
                z[k, :] = np.nan_to_num(z[k, :], nan=np.nanmin(data_clean[k, :]))

            except Exception as e:
                print(f"Warning: spectrum {k} solve failed, fallback baseline used: {e}")
                window = max(3, n // 50)
                z[k, :] = pd.Series(data_clean[k, :]).rolling(
                    window, center=True, min_periods=1
                ).min().values

            w0[k, :] = (1 - p) * (data_clean[k, :] <= z[k, :]) + p * (data_clean[k, :] > z[k, :])
            w0[k, :] = np.clip(w0[k, :], 1e-8, 1.0)
            z0[k, :] = z[k, :]

        theta = (data_clean - z).mean(axis=0)
        theta_norm = np.dot(theta, theta)
        if theta_norm > 1e-12:
            a0 = ((data_clean - z) @ theta) / theta_norm
            a0 = np.clip(a0, 0.5, 1.5)
        else:
            a0 = np.ones(m)

        z_change = np.linalg.norm(z - z_prev) / max(np.linalg.norm(z_prev), 1e-12)
        a0_change = np.linalg.norm(a0 - a0_prev) / max(np.linalg.norm(a0_prev), 1e-12)
        if z_change < tol and a0_change < tol:
            converged = True
            break
        z_prev = np.copy(z)
        a0_prev = a0.copy()

    return z, converged


# ==================== Preprocessing ====================
def step1_peak_removal(df, peak_region, tau, max_iter, p_thresh):
    x = df[WAVELENGTH_COL].values
    peak_mask = (x >= peak_region[0]) & (x <= peak_region[1])

    signal_cols = [c for c in SIGNAL_COLS if c in df.columns]
    cleaned_data = {WAVELENGTH_COL: x}

    for col in signal_cols:
        y = df[col].values.astype(float)
        keep = np.ones_like(y, dtype=bool)

        if np.any(peak_mask):
            keep_peak, _ = chauvenet_iterative(
                y[peak_mask],
                tau=tau,
                max_iter=max_iter,
                p_thresh=p_thresh,
                verbose=False,
            )
            keep[peak_mask] = keep_peak

        y_clean = y.copy()
        y_clean[~keep] = np.nan
        cleaned_data[col] = y_clean
        cleaned_data[f"{col}_mask"] = keep.astype(int)

    return pd.DataFrame(cleaned_data), signal_cols


# ==================== ODAT-SE Solver ====================
class MSBCSolver(odatse.solver.SolverBase):
    """
    ODAT-SE solver for MSBC baseline parameter search.

    xs[0] = log10(lambda)
    xs[1] = p
    xs[2] = log10(mu0)
    """

    def __init__(self, info):
        super().__init__(info)

        self.output_dir = info.base.get("output_dir", OUTPUT_DIR_FALLBACK)

        # Read original data
        self.df_orig = pd.read_excel(INPUT_XLSX)

        # Step 1: preprocessing only once
        self.df_clean, self.signal_cols = step1_peak_removal(
            self.df_orig,
            PEAK_REGION,
            CHAUVENET_TAU,
            MAX_ITER_CHAUVENET,
            NORMALITY_PVAL,
        )

        self.x = self.df_clean[WAVELENGTH_COL].values
        self.bg_mask = (self.x >= BG_REGION[0]) & (self.x <= BG_REGION[1])

        if np.sum(self.bg_mask) < 5:
            raise ValueError(f"Too few data points in background region {BG_REGION}")

        spectra_data = []
        valid_cols = []
        for col in self.signal_cols:
            y = self.df_clean[col].values
            valid = finite_mask(y)
            if np.sum(valid & self.bg_mask) < 3:
                continue
            spectra_data.append(y)
            valid_cols.append(col)

        if len(spectra_data) == 0:
            raise ValueError("No valid spectra available after preprocessing.")

        self.spectra_data = np.array(spectra_data)
        self.valid_cols = valid_cols
        self.m, self.n = self.spectra_data.shape

        self.eval_count = 0

    def evaluate(self, xs, args=None, nprocs=1, nthreads=1):
        self.eval_count += 1

        log_lam = float(xs[0])
        p = float(xs[1])
        log_mu = float(xs[2])

        lam = 10.0 ** log_lam
        mu = 10.0 ** log_mu

        if not (0.0 < p < 1.0):
            return 1e30
        if p < 0.008 or p > 0.08:
            return 1e20

        try:
            z, converged = msbc(
                self.spectra_data,
                lam,
                mu,
                p,
                max_iter=MSBC_MAX_ITER,
                tol=ALS_TOL,
            )

            residuals = (self.spectra_data - z)[:, self.bg_mask]
            residuals = residuals[finite_mask(residuals)]

            if residuals.size < 3:
                return 1e30

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
                data_median = np.median(self.spectra_data[k, :])
                baseline_mean = np.mean(z[k, :])
                if abs(data_median) > 1e-12:
                    ratio = baseline_mean / data_median
                    if ratio > 0.8:
                        penalties += mse_ref * (ratio - 0.8) * 5.0
                    if ratio < 0.0:
                        penalties += mse_ref * abs(ratio) * 5.0

            total_loss = mse + penalties

            if not np.isfinite(total_loss):
                return 1e30

            return float(total_loss)

        except Exception as e:
            print(
                f"Evaluation failed: eval={self.eval_count}, "
                f"log_lam={log_lam:.4f}, p={p:.6f}, log_mu={log_mu:.4f}, err={e}"
            )
            return 1e30


# ==================== Final post-processing ====================
def save_final_results(solver, best_x, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    log_lam = float(best_x[0])
    p = float(best_x[1])
    log_mu = float(best_x[2])

    lam = 10.0 ** log_lam
    mu = 10.0 ** log_mu

    z, converged = msbc(
        solver.spectra_data,
        lam,
        mu,
        p,
        max_iter=MSBC_MAX_ITER,
        tol=ALS_TOL,
    )

    x = solver.x
    y0 = solver.df_orig[REFERENCE_COL].values
    y0_safe = np.where(finite_mask(y0) & (np.abs(y0) > 1e-10), y0, 1.0)

    spectra_orig = np.array([solver.df_orig[col].values for col in solver.valid_cols])

    corrected = spectra_orig - z
    normalized = corrected / y0_safe

    baseline_df = pd.DataFrame({WAVELENGTH_COL: x})
    corrected_df = pd.DataFrame({WAVELENGTH_COL: x})
    normalized_df = pd.DataFrame({WAVELENGTH_COL: x, REFERENCE_COL: y0})

    for i, col in enumerate(solver.valid_cols):
        baseline_df[f"{col}_baseline"] = z[i]
        corrected_df[f"{col}_corrected"] = corrected[i]
        normalized_df[f"{col}_normalized"] = normalized[i]

    baseline_df.to_excel(os.path.join(output_dir, "baselines.xlsx"), index=False)
    corrected_df.to_excel(os.path.join(output_dir, "corrected_spectra.xlsx"), index=False)
    normalized_df.to_excel(os.path.join(output_dir, "normalized_spectra.xlsx"), index=False)

    summary = pd.DataFrame(
        {
            "parameter": [
                "log10_lambda",
                "p",
                "log10_mu0",
                "lambda",
                "mu0",
                "converged",
                "peak_region_left",
                "peak_region_right",
                "bg_region_left",
                "bg_region_right",
            ],
            "value": [
                log_lam,
                p,
                log_mu,
                lam,
                mu,
                int(converged),
                PEAK_REGION[0],
                PEAK_REGION[1],
                BG_REGION[0],
                BG_REGION[1],
            ],
        }
    )
    summary.to_excel(os.path.join(output_dir, "final_summary.xlsx"), index=False)

    if MAKE_FINAL_PLOTS:
        colors = plt.cm.viridis(np.linspace(0, 1, len(solver.valid_cols)))

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        for i, col in enumerate(solver.valid_cols):
            axes[0].plot(x, spectra_orig[i], alpha=0.5, color=colors[i], label=col)
            axes[0].plot(x, z[i], "--", color=colors[i], linewidth=2)
        axes[0].axvspan(BG_REGION[0], BG_REGION[1], alpha=0.1, color="green", label="BG region")
        axes[0].axvspan(PEAK_REGION[0], PEAK_REGION[1], alpha=0.1, color="red", label="Peak region")
        axes[0].set_ylabel("Intensity")
        axes[0].set_title("(a) Original Spectra and Baselines")
        axes[0].legend(loc="best", fontsize=8, ncol=2)
        axes[0].grid(True, alpha=0.3)

        for i, col in enumerate(solver.valid_cols):
            axes[1].plot(x, corrected[i], color=colors[i], label=col)
        axes[1].axvspan(BG_REGION[0], BG_REGION[1], alpha=0.1, color="green")
        axes[1].set_ylabel("Intensity")
        axes[1].set_title("(b) Baseline-Corrected Spectra")
        axes[1].legend(loc="best", fontsize=8, ncol=2)
        axes[1].grid(True, alpha=0.3)

        for i, col in enumerate(solver.valid_cols):
            axes[2].plot(x, normalized[i], color=colors[i], label=col)
        axes[2].axvspan(BG_REGION[0], BG_REGION[1], alpha=0.1, color="green")
        axes[2].set_xlabel("Wavelength (nm)")
        axes[2].set_ylabel("Normalized Intensity")
        axes[2].set_title("(c) Normalized: (Original - Baseline) / y0")
        axes[2].legend(loc="best", fontsize=8, ncol=2)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comprehensive_results.png"), dpi=200)
        plt.close()

    print("\nBest parameters:")
    print(f"  log10(lambda) = {log_lam:.6f}")
    print(f"  p             = {p:.6f}")
    print(f"  log10(mu0)    = {log_mu:.6f}")
    print(f"  lambda        = {lam:.6e}")
    print(f"  mu0           = {mu:.6e}")
    print(f"  converged     = {converged}")


# ==================== Main ====================
def main():
    info, run_mode = odatse.initialize()
    print("algorithm =", info.algorithm["name"])
    output_dir = info.base.get("output_dir", OUTPUT_DIR_FALLBACK)
    os.makedirs(output_dir, exist_ok=True)

    solver = MSBCSolver(info)
    runner = odatse.Runner(solver, info)

    alg_module = choose_algorithm(info.algorithm["name"])
    alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
    result = alg.main()

    print("\nODAT-SE finished.")
    print("Raw result:")
    print(result)

    # Most ODAT-SE algorithms return best x in result["x"]
    if isinstance(result, dict) and "x" in result:
        best_x = result["x"]
    else:
        raise RuntimeError("Cannot find best parameter vector 'x' in ODAT-SE result.")

    save_final_results(solver, best_x, output_dir)



if __name__ == "__main__":
    main()