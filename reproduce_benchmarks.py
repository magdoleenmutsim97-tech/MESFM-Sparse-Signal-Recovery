#!/usr/bin/env python3
import argparse
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon


class MagdoleenExactSubspaceModel:
    def __init__(self, rho=0.25, p=2.5, eta=0.20, n_iterations=400, eps=1e-8):
        self.rho = rho
        self.p = p
        self.eta = eta
        self.n_iterations = n_iterations
        self.eps = eps
        self.weights_stage1 = None
        self.weights = None
        self.active_support = set()
        self.raw_support = set()
        self.cost_history = []
        self.tau = None

    def _adaptive_thresholding(self, z, tau):
        abs_z = np.abs(z)
        mask = abs_z > tau
        adaptive_shift = tau / (1.0 + (abs_z / (tau + self.eps)) ** self.p)
        return np.where(mask, np.sign(z) * (abs_z - adaptive_shift), 0.0)

    def fit(self, X, y):
        _, n = X.shape
        w = np.zeros(n)
        self.cost_history = []
        L = np.linalg.norm(X, ord=2) ** 2
        alpha = 1.0 / L
        self.tau = self.rho * alpha * np.max(np.abs(X.T @ y))

        for _ in range(self.n_iterations):
            residual = X @ w - y
            self.cost_history.append(0.5 * np.linalg.norm(residual) ** 2)
            z = w - alpha * (X.T @ residual)
            w = self._adaptive_thresholding(z, self.tau)

        self.weights_stage1 = w.copy()
        self.raw_support = set(np.where(np.abs(w) > 1e-8)[0])
        max_peak = np.max(np.abs(w))
        supp = np.where(np.abs(w) > self.eta * max_peak)[0] if max_peak > 0 else np.array([], dtype=int)
        self.active_support = set(supp)
        w_refined = np.zeros(n)
        if len(supp) > 0:
            w_refined[supp], _, _, _ = np.linalg.lstsq(X[:, supp], y, rcond=None)
        self.weights = w_refined
        return self


def run_ista(X, y, rho=0.25, eta=0.20, n_iterations=400):
    _, n = X.shape
    w = np.zeros(n)
    L = np.linalg.norm(X, ord=2) ** 2
    alpha = 1.0 / L
    tau = rho * alpha * np.max(np.abs(X.T @ y))
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        z = w - alpha * (X.T @ (X @ w - y))
        w = np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)
    t_stage1 = (time.perf_counter() - t0) * 1000.0
    raw_supp = set(np.where(np.abs(w) > 1e-8)[0])
    max_peak = np.max(np.abs(w))
    supp = np.where(np.abs(w) > eta * max_peak)[0] if max_peak > 0 else np.array([], dtype=int)
    t1 = time.perf_counter()
    w_ls = np.zeros(n)
    if len(supp) > 0:
        w_ls[supp], _, _, _ = np.linalg.lstsq(X[:, supp], y, rcond=None)
    t_full = t_stage1 + (time.perf_counter() - t1) * 1000.0
    return w, w_ls, set(supp), raw_supp, t_stage1, t_full


def run_fista(X, y, rho=0.25, eta=0.20, n_iterations=400):
    _, n = X.shape
    w = np.zeros(n)
    v = np.zeros(n)
    t_mom = 1.0
    L = np.linalg.norm(X, ord=2) ** 2
    alpha = 1.0 / L
    tau = rho * alpha * np.max(np.abs(X.T @ y))
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        z = v - alpha * (X.T @ (X @ v - y))
        w_new = np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom))
        v = w_new + ((t_mom - 1.0) / t_new) * (w_new - w)
        w = w_new
        t_mom = t_new
    t_stage1 = (time.perf_counter() - t0) * 1000.0
    raw_supp = set(np.where(np.abs(w) > 1e-8)[0])
    max_peak = np.max(np.abs(w))
    supp = np.where(np.abs(w) > eta * max_peak)[0] if max_peak > 0 else np.array([], dtype=int)
    t1 = time.perf_counter()
    w_ls = np.zeros(n)
    if len(supp) > 0:
        w_ls[supp], _, _, _ = np.linalg.lstsq(X[:, supp], y, rcond=None)
    t_full = t_stage1 + (time.perf_counter() - t1) * 1000.0
    return w, w_ls, set(supp), raw_supp, t_stage1, t_full


def run_mesfm(X, y, rho=0.25, p=2.5, eta=0.20, n_iterations=400):
    model = MagdoleenExactSubspaceModel(rho=rho, p=p, eta=eta, n_iterations=n_iterations)
    _, n = X.shape
    w = np.zeros(n)
    L = np.linalg.norm(X, ord=2) ** 2
    alpha = 1.0 / L
    tau = rho * alpha * np.max(np.abs(X.T @ y))
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        z = w - alpha * (X.T @ (X @ w - y))
        w = model._adaptive_thresholding(z, tau)
    t_stage1 = (time.perf_counter() - t0) * 1000.0
    w_stage1 = w.copy()
    raw_supp = set(np.where(np.abs(w_stage1) > 1e-8)[0])
    max_peak = np.max(np.abs(w_stage1))
    supp = np.where(np.abs(w_stage1) > eta * max_peak)[0] if max_peak > 0 else np.array([], dtype=int)
    t1 = time.perf_counter()
    w_full = np.zeros(n)
    if len(supp) > 0:
        w_full[supp], _, _, _ = np.linalg.lstsq(X[:, supp], y, rcond=None)
    t_full = t_stage1 + (time.perf_counter() - t1) * 1000.0
    return w_stage1, w_full, set(supp), raw_supp, t_stage1, t_full


def generate_problem(m, n, s, sigma, seed):
    np.random.seed(seed)
    X = np.random.randn(m, n) / np.sqrt(m)
    idx = np.random.choice(n, size=s, replace=False)
    signs = np.random.choice([-1.0, 1.0], size=s)
    mags = np.random.uniform(1.5, 3.2, size=s)
    w_true = np.zeros(n)
    w_true[idx] = signs * mags
    y = X @ w_true + np.random.normal(0.0, sigma, size=m)
    return X, y, w_true, set(idx)


def rel_error_pct(w_est, w_true):
    return 100.0 * np.linalg.norm(w_est - w_true) / np.linalg.norm(w_true)


def bootstrap_ci(diffs, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    samples = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True)
    means = np.mean(samples, axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def run_table1_and_table2(show_table1=True, show_table2=True):
    methods = ["ISTA", "FISTA", "ISTA+LS", "FISTA+LS", "MESFM Stage I", "MESFM Full"]
    all_results = {}
    for sigma in [0.00, 0.05, 0.15]:
        res = {k: {"err": [], "sup": [], "raw": [], "time": []} for k in methods}
        for sd in range(100):
            X, y, wt, St = generate_problem(40, 100, 5, sigma, sd)
            wi, wils, si, ri, ti1, tif = run_ista(X, y)
            wf, wfls, sf, rf, tf1, tff = run_fista(X, y)
            wm1, wmf, sm, rm, tm1, tmf = run_mesfm(X, y)
            for name, est, sup, raw, ms in [
                ("ISTA", wi, si, ri, ti1), ("FISTA", wf, sf, rf, tf1),
                ("ISTA+LS", wils, si, ri, tif), ("FISTA+LS", wfls, sf, rf, tff),
                ("MESFM Stage I", wm1, sm, rm, tm1), ("MESFM Full", wmf, sm, rm, tmf),
            ]:
                res[name]["err"].append(rel_error_pct(est, wt))
                res[name]["sup"].append(sup == St)
                res[name]["raw"].append(raw == St)
                res[name]["time"].append(ms)
        all_results[sigma] = res

    if show_table1:
        print("\nTABLE 1 (1,800 runs across 300 seed-noise combinations, seeds 0-99):")
        for sigma in [0.00, 0.05, 0.15]:
            for name in methods:
                d = all_results[sigma][name]
                errs = np.array(d["err"])
                print(f"sigma={sigma:.2f} | {name:<14} | mean={np.mean(errs):6.2f}% | med={np.median(errs):6.2f}% | sup={sum(d['sup']):>2d}/100")

    if show_table2:
        print("\nTABLE 2 (Paired statistical tests, seeds 0-99):")
        for label, sigma, m_key, b_key in [
            ("MESFM vs ISTA+LS", 0.00, "MESFM Full", "ISTA+LS"),
            ("MESFM vs ISTA+LS", 0.05, "MESFM Full", "ISTA+LS"),
            ("MESFM vs ISTA+LS", 0.15, "MESFM Full", "ISTA+LS"),
            ("MESFM Stg1 vs ISTA", 0.05, "MESFM Stage I", "ISTA"),
        ]:
            diffs = np.array(all_results[sigma][m_key]["err"]) - np.array(all_results[sigma][b_key]["err"])
            lo, hi = bootstrap_ci(diffs)
            w_p = wilcoxon(diffs, zero_method="zsplit").pvalue
            m_plus = sum(m and not b for m, b in zip(all_results[sigma][m_key]["sup"], all_results[sigma][b_key]["sup"]))
            b_plus = sum(b and not m for m, b in zip(all_results[sigma][m_key]["sup"], all_results[sigma][b_key]["sup"]))
            mc_p = binomtest(min(m_plus, b_plus), m_plus + b_plus, 0.5).pvalue
            print(f"{label:<20} | sigma={sigma:.2f} | diff={np.mean(diffs):+6.2f}% | CI=[{lo:+6.2f}%, {hi:+6.2f}%] | W_p={w_p:.4f} | McN_p={mc_p:.4f}")


def run_table3():
    print("\nTABLE 3 (Sparsity seeds 500-549 & Measurement seeds 1000-1049, sigma=0.05):")
    for s in [3, 5, 8, 10]:
        ie, ilse, m1e, mfe, ilss, mfs = [], [], [], [], 0, 0
        for sd in range(500, 550):
            X, y, wt, St = generate_problem(40, 100, s, 0.05, sd)
            wi, wils, si, _, _, _ = run_ista(X, y)
            wm1, wmf, sm, _, _, _ = run_mesfm(X, y)
            ie.append(rel_error_pct(wi, wt)); ilse.append(rel_error_pct(wils, wt))
            m1e.append(rel_error_pct(wm1, wt)); mfe.append(rel_error_pct(wmf, wt))
            ilss += (si == St); mfs += (sm == St)
        print(f"Sparsity s={s:<2} | ISTA={np.mean(ie):6.2f}% | ISTA+LS={np.mean(ilse):6.2f}% | M-Stg1={np.mean(m1e):6.2f}% | M-Full={np.mean(mfe):6.2f}% | Sup={ilss*2}%/{mfs*2}%")
    for m in [25, 40, 60]:
        ie, ilse, m1e, mfe, ilss, mfs = [], [], [], [], 0, 0
        for sd in range(1000, 1050):
            X, y, wt, St = generate_problem(m, 100, 5, 0.05, sd)
            wi, wils, si, _, _, _ = run_ista(X, y)
            wm1, wmf, sm, _, _, _ = run_mesfm(X, y)
            ie.append(rel_error_pct(wi, wt)); ilse.append(rel_error_pct(wils, wt))
            m1e.append(rel_error_pct(wm1, wt)); mfe.append(rel_error_pct(wmf, wt))
            ilss += (si == St); mfs += (sm == St)
        print(f"Measure  m={m:<2} | ISTA={np.mean(ie):6.2f}% | ISTA+LS={np.mean(ilse):6.2f}% | M-Stg1={np.mean(m1e):6.2f}% | M-Full={np.mean(mfe):6.2f}% | Sup={ilss*2}%/{mfs*2}%")


def run_independent_validation():
    print("\nSECTION 5.1 (Seeds 100-199):")
    for label, rho, eta in [("Baseline (0.25, 0.20)", 0.25, 0.20), ("Candidate (0.10, 0.40)", 0.10, 0.40)]:
        errs, sups = [], []
        for sd in range(100, 200):
            X, y, wt, St = generate_problem(40, 100, 5, 0.05, sd)
            _, wf, sp, _, _, _ = run_mesfm(X, y, rho=rho, p=2.5, eta=eta)
            errs.append(rel_error_pct(wf, wt)); sups.append(sp == St)
        print(f"  {label}: mean={np.mean(errs):.2f}%, exact_supp={sum(sups)}/100")


def run_sign_perturbation():
    print("\nSECTION 5.2 (Seed 42):")
    np.random.seed(42)
    X = np.random.randn(40, 100) / np.sqrt(40)
    noise = np.random.normal(0.0, 0.05, size=40)
    S_star = [10, 25, 50, 75, 90]
    mags = np.array([2.5, 1.8, 3.2, 2.0, 1.5])
    for label, rho, p, eta in [("Default (2.5, 0.25, 0.20)", 0.25, 2.5, 0.20), ("Alt Joint (1.5, 0.15, 0.30)", 0.15, 1.5, 0.30)]:
        exact, errs = 0, []
        for signs in product([-1.0, 1.0], repeat=5):
            wt = np.zeros(100); wt[S_star] = np.array(signs) * mags
            _, wf, sp, _, _, _ = run_mesfm(X, X @ wt + noise, rho=rho, p=p, eta=eta)
            exact += (sp == set(S_star)); errs.append(rel_error_pct(wf, wt))
        print(f"  {label}: exact={exact}/32 ({100*exact/32:.0f}%), mean_err={np.mean(errs):.2f}%")


def generate_figure1(save_path="figure1_mesfm_seed42.png"):
    np.random.seed(42)
    X = np.random.randn(40, 100) / np.sqrt(40)
    w_true = np.zeros(100); w_true[[10, 25, 50, 75, 90]] = [2.5, -1.8, 3.2, -2.0, 1.5]
    y = X @ w_true + np.random.normal(0.0, 0.05, size=40)
    model = MagdoleenExactSubspaceModel().fit(X, y)
    err_pct = rel_error_pct(model.weights, w_true)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(model.cost_history, color="darkgreen", linewidth=2.5)
    axes[0].set_title("Lipschitz-Calibrated Convergence (MESFM)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Iterations"); axes[0].set_ylabel("0.5 ||Xw - y||^2"); axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[1].stem(w_true, linefmt="g-", markerfmt="go", basefmt="k-", label="True Sparse Signal")
    axes[1].stem(model.weights, linefmt="r--", markerfmt="rx", basefmt="k-", label="MESFM Recovery (Exact Support)")
    axes[1].set_title(f"Exact Subspace Recovery (Relative Error: {err_pct:.2f}%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Feature Index"); axes[1].set_ylabel("Weight Value"); axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\n[Figure 1] Saved to '{save_path}' (Relative Coefficient Error: {err_pct:.2f}%)")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--table1", action="store_true")
    parser.add_argument("--table2", action="store_true")
    parser.add_argument("--table3", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--signs", action="store_true")
    parser.add_argument("--figure1", action="store_true")
    args, _ = parser.parse_known_args(argv)
    run_all = args.all or not any([args.table1, args.table2, args.table3, args.validation, args.signs, args.figure1])
    if run_all or args.table1 or args.table2:
        run_table1_and_table2(run_all or args.table1, run_all or args.table2)
    if run_all or args.table3:
        run_table3()
    if run_all or args.validation:
        run_independent_validation()
    if run_all or args.signs:
        run_sign_perturbation()
    if run_all or args.figure1:
        generate_figure1()


if __name__ == "__main__":
    main()
