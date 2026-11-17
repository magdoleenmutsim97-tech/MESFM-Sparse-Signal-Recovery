"""
Magdoleen Exact Subspace Functional Model (MESFM)
Author: Dr. Magdoleen Mutseam Mudawi

Two-stage sparse signal recovery framework:
1. Lipschitz-calibrated gradient iterations with adaptive thresholding.
2. Relative infinity-norm support selection followed by least-squares refinement.
"""

import numpy as np


def mesfm_solver(X, y, rho=0.25, p=2.5, eta=0.20, max_iter=400, eps=1e-8):
    """
    Solves the underdetermined linear inverse problem y = X w* + noise using MESFM.
    """
    n = X.shape[1]
    w = np.zeros(n)

    # Safe Lipschitz constant calibration (guards against L = 0)
    L = np.linalg.norm(X, 2) ** 2
    if L <= 1e-15:
        return w, w.copy(), np.array([], dtype=int), []
    alpha = 1.0 / L
    tau = rho * alpha * np.max(np.abs(X.T @ y))

    cost_history = []

    # Stage I: Lipschitz-calibrated adaptive thresholding
    for _ in range(max_iter):
        z = w - alpha * (X.T @ (X @ w - y))
        abs_z = np.abs(z)

        # Apply adaptive thresholding rule only on coordinates exceeding tau
        w = np.zeros_like(z)
        mask = abs_z > tau
        if np.any(mask):
            az_m = abs_z[mask]
            w[mask] = np.sign(z[mask]) * (az_m - tau / (1.0 + (az_m / (tau + eps)) ** p))

        # Record least-squares residual cost AFTER updating w
        residual_cost = 0.5 * np.linalg.norm(X @ w - y) ** 2
        cost_history.append(residual_cost)

    w_stage1 = w.copy()

    # Stage II: Support selection via relative infinity-norm criterion
    peak = np.max(np.abs(w_stage1))
    if peak > 0:
        selected_support = np.flatnonzero(np.abs(w_stage1) > eta * peak)
    else:
        selected_support = np.array([], dtype=int)

    # Stage III: Support-restricted least-squares refinement
    w_refined = np.zeros(n)
    if len(selected_support) > 0:
        w_refined[selected_support] = np.linalg.lstsq(X[:, selected_support], y, rcond=None)[0]

    return w_refined, w_stage1, selected_support, cost_history


if __name__ == "__main__":
    true_support = [10, 25, 50, 75, 90]

    # Running both the primary baseline (-1.8) and the sign-sensitivity case (+1.8)
    cases = [
        ("1. Primary Reproducible Baseline (w_25 = -1.8)", np.array([2.5, -1.8, 3.2, -2.0, 1.5])),
        ("2. Sign-Perturbed Sensitivity Case (w_25 = +1.8)", np.array([2.5,  1.8, 3.2, -2.0, 1.5])),
    ]

    for title, true_values in cases:
        np.random.seed(42)
        m, n = 40, 100
        X = np.random.randn(m, n) / np.sqrt(m)
        w_true = np.zeros(n)
        w_true[true_support] = true_values
        y = X @ w_true + 0.05 * np.random.randn(m)

        w_refined, w_stage1, selected_support, cost_history = mesfm_solver(
            X, y, rho=0.25, p=2.5, eta=0.20, max_iter=400
        )

        err_s1 = np.linalg.norm(w_stage1 - w_true) / np.linalg.norm(w_true) * 100.0
        err_ref = np.linalg.norm(w_refined - w_true) / np.linalg.norm(w_true) * 100.0

        print(f"=== {title} ===")
        print(f"Max correlation ||X^T y||_inf : {np.max(np.abs(X.T @ y)):.4f}")
        print(f"True support indices          : {true_support}")
        print(f"Selected support ({len(selected_support)} indices)  : {selected_support.tolist()}")
        print(f"Stage I relative error        : {err_s1:.4f}%")
        print(f"Refined relative error        : {err_ref:.4f}%\n")
