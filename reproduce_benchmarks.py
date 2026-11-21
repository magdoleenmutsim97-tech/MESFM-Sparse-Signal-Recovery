#!/usr/bin/env python3
"""
Reproducibility Script for the Magdoleen Exact Subspace Functional Model (MESFM)
================================================================================
Paper:  "A Lipschitz-Calibrated Functional Optimization Framework with Adaptive
         Thresholding and Subspace Projection for Sparse Signal Recovery"
Author: Dr. Magdoleen Mutseam Mudawi
        Department of Mathematics and Functional Analysis,
        Sudan International University
ORCID:  https://orcid.org/0009-0002-7409-9835
Zenodo: https://doi.org/10.5281/zenodo.24422561

Usage (Command Line / GitHub):
    python reproduce_benchmarks.py --all
    python reproduce_benchmarks.py --table1
    python reproduce_benchmarks.py --table2
    python reproduce_benchmarks.py --table3
    python reproduce_benchmarks.py --validation
    python reproduce_benchmarks.py --signs
    python reproduce_benchmarks.py --figure1

Usage (Kaggle / Jupyter Notebook):
    Run this file directly in a notebook cell (or import and call `main(["--all"])`).
    Only standard scientific Python libraries (numpy, scipy, matplotlib) are required.
"""

import argparse
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon


# ---------------------------------------------------------------------------
# Core Algorithms: MESFM, ISTA, FISTA (with optional Stage III LS debiasing)
# ---------------------------------------------------------------------------

class MagdoleenExactSubspaceModel:
    """Magdoleen Exact Subspace Functional Model (MESFM)."""

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

        # Stage I: Lipschitz-Calibrated Adaptive Thresholding
        for _ in range(self.n_iterations):
            residual = X @ w - y
            self.cost_history.append(0.5 * np.linalg.norm(residual) ** 2)
            z = w - alpha * (X.T @ residual)
            w = self._adaptive_thresholding(z, self.tau)

        self.weights_stage1 = w.copy()
        self.raw_support = set(np.where(np.abs(w) > 1e-8)[0])

        # Stage II & III: Relative Infinity-Norm Support Pruning + Subspace LS
        max_peak = np.max(np.abs(w))
        if max_peak > 0:
            supp = np.where(np.abs(w) > self.eta * max_peak)[0]
        else:
            supp = np.array([], dtype=int)

        self.active_support = set(supp)
        w_refined = np.zeros(n)
        if len(supp) > 0:
            w_refined[supp], _, _, _ = np.linalg.lstsq(X[:, supp], y, rcond=None)
        self.weights = w_refined
        return self


def run_ista(X, y, rho=0.25, eta=0.20, n_iterations=400):
    """Standard ISTA and debiased ISTA+LS."""
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
    """Standard FISTA and debiased FISTA+LS."""
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
    """Run MESFM returning Stage I, Full, supports, and elapsed times (ms)."""
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


# ---------------------------------------------------------------------------
# Synthetic Problem Generator (Exact Legacy NumPy Seeding Protocol)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Benchmark Routines Matching Paper Tables & Sections
# ---------------------------------------------------------------------------

def run_table1_and_table2(show_table1=True, show_table2=True):
    """Reproduce Table 1 (1,800 runs across seeds 0-99, 3 noise regimes) & Table 2."""
    methods = ["ISTA", "FISTA", "ISTA+LS", "FISTA+LS", "MESFM Stage I", "MESFM Full"]
    all_results = {}

    for sigma in [0.00, 0.05, 0.15]:
        res = {k: {"err": [], "sup": [], "raw": [], "time": []} for k in methods}
        for sd in range(100):
            X, y, wt, St = generate_problem(m=40, n=100, s=5, sigma=sigma, seed=sd)
            wi, wils, si, ri, ti1, tif = run_ista(X, y)
            wf, wfls, sf, rf, tf1, tff = run_fista(X, y)
            wm1, wmf, sm, rm, tm1, tmf = run_mesfm(X, y)

            entries = [
                ("ISTA", wi, si, ri, ti1),
                ("FISTA", wf, sf, rf, tf1),
                ("ISTA+LS", wils, si, ri, tif),
                ("FISTA+LS", wfls, sf, rf, tff),
                ("MESFM Stage I", wm1, sm, rm, tm1),
                ("MESFM Full", wmf, sm, rm, tmf),
            ]
            for name, est, sup, raw, ms in entries:
                res[name]["err"].append(rel_error_pct(est, wt))
                res[name]["sup"].append(sup == St)
                res[name]["raw"].append(raw == St)
                res[name]["time"].append(ms)
        all_results[sigma] = res

    if show_table1:
        print("\n" + "=" * 96)
        print("TABLE 1: 100-Seed Comparative Study (1,800 algorithm runs across 300 seed-noise combinations)")
        print("Seeds: 0-99 | m=40, n=100, s=5, K=400, rho=0.25, p=2.5, eta=0.20")
        print("=" * 96)
        print(f"{'Sigma':>6} | {'Method':<15} | {'Mean%':>7} | {'Med%':>7} | {'Std%':>7} | {'ExactSup':>10} | {'RawSup':>8} | {'Time(ms)':>8}")
        print("-" * 96)
        for sigma in [0.00, 0.05, 0.15]:
            for name in methods:
                d = all_results[sigma][name]
                errs = np.array(d["err"])
              print(
                    f"{sigma:6.2f} | {name:<15} | {np.mean(errs):6.2f}% | {np.median(errs):6.2f}% | "
                    f"{np.std(errs, ddof=1):6.2f}% | {sum(d['sup']):>3d}/100    | "
                    f"{sum(d['raw']):>3d}/100  | {np.mean(d['time']):8.2f}"
                )
