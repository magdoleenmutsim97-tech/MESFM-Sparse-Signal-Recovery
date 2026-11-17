    import numpy as np
import matplotlib.pyplot as plt


def mesfm_solver(X, y, rho=0.25, p=2.5, eta=0.20, max_iter=400, eps=1e-8):
    """
    Magdoleen Exact Subspace Functional Model (MESFM)
    Two-stage adaptive thresholding and subspace least-squares refinement.

    Parameters
    ----------
    X : ndarray of shape (m, n)
        Sensing matrix.
    y : ndarray of shape (m,)
        Observation vector.
    rho : float, default=0.25
        Threshold scaling parameter.
    p : float, default=2.5
        Adaptive thresholding exponent.
    eta : float, default=0.20
        Relative infinity-norm cutoff for active support selection.
    max_iter : int, default=400
        Number of Stage I iterations.
    eps : float, default=1e-8
        Numerical stabilization constant.

    Returns
    -------
    w_refined : ndarray of shape (n,)
        Final coefficient estimate after least-squares refinement on selected support.
    w_stage1 : ndarray of shape (n,)
        Coefficient estimate after Stage I adaptive thresholding.
    selected_support : ndarray
        Indices of the selected active support.
    cost_history : list of float
        Least-squares residual cost (0.5 * ||Xw - y||_2^2) recorded after each update.
    """
    n = X.shape[1]
    w = np.zeros(n)

    # Safe Lipschitz step-size calibration (guards against zero sensing matrix L = 0)
    L = np.linalg.norm(X, 2) ** 2
    if L <= 1e-15:
        return w.copy(), w.copy(), np.array([], dtype=int), []

    alpha = 1.0 / L
    tau = rho * alpha * np.max(np.abs(X.T @ y))

    cost_history = []

    # Stage I: Lipschitz-Calibrated Adaptive Thresholding
    for _ in range(max_iter):
        z = w - alpha * (X.T @ (X @ w - y))
        abs_z = np.abs(z)

        # Evaluate adaptive shrinkage formula strictly on active entries exceeding tau
        w = np.zeros_like(z)
        mask = abs_z > tau
        if np.any(mask):
            abs_m = abs_z[mask]
            w[mask] = np.sign(z[mask]) * (abs_m - tau / (1.0 + (abs_m / (tau + eps)) ** p))

        # Record least-squares data-fidelity cost AFTER updating w
        residual_cost = 0.5 * np.linalg.norm(X @ w - y) ** 2
        cost_history.append(residual_cost)

    w_stage1 = w.copy()

    # Stage II: Relative Infinity-Norm Support Selection
    peak = np.max(np.abs(w_stage1))
    if peak > 0:
        selected_support = np.flatnonzero(np.abs(w_stage1) > eta * peak)
    else:
        selected_support = np.array([], dtype=int)

    # Stage III: Support-Restricted Least-Squares Refinement
    w_refined = np.zeros(n)
    if len(selected_support) > 0:
        w_refined[selected_support] = np.linalg.lstsq(X[:, selected_support], y, rcond=None)[0]

    return w_refined, w_stage1, selected_support, cost_history


def run_synthetic_demo():
    """Reproducible synthetic demonstration matching seed 42 configuration."""
    rng = np.random.RandomState(42)
    m, n = 40, 100
    X = rng.randn(m, n) / np.sqrt(m)

    true_support = np.array([10, 25, 50, 75, 90])
    true_values = np.array([2.5, 1.8, 3.2, -2.0, 1.5])
    w_true = np.zeros(n)
    w_true[true_support] = true_values

    noise = rng.normal(0.0, 0.05, size=m)
    y = X @ w_true + noise

    w_refined, w_stage1, supp_hat, cost_history = mesfm_solver(
        X, y, rho=0.25, p=2.5, eta=0.20, max_iter=400
    )

    err_stage1 = np.linalg.norm(w_stage1 - w_true) / np.linalg.norm(w_true) * 100.0
    err_refined = np.linalg.norm(w_refined - w_true) / np.linalg.norm(w_true) * 100.0

    print("=== MESFM Reproducible Synthetic Demonstration (Seed 42) ===")
    print(f"True Support Indices     : {true_support.tolist()}")
    print(f"Selected Support Indices : {supp_hat.tolist()}")
    print(f"Exact Support Match      : {set(supp_hat) == set(true_support)} ({len(supp_hat)}/{len(true_support)})")
    print(f"Stage I Relative Error   : {err_stage1:.4f}%")
    print(f"Refined Relative Error   : {err_refined:.4f}%")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.stem(w_true, linefmt="k-", markerfmt="ko", basefmt="k-", label="True Signal")
    plt.stem(w_refined, linefmt="r--", markerfmt="rx", basefmt="k-", label="MESFM Refined")
    plt.title("Sparse Signal Reconstruction")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cost_history, color="navy", linewidth=1.5)
    plt.title("Stage I Residual Cost History")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\frac{1}{2}\|Xw^{(k)} - y\|_2^2$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("mesfm_demo_plot.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_synthetic_demo()
    
