    import time
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits


def run_method(X, y, method, K=400, rho=0.25, p=2.5, eta=0.20, eps=1e-8):
    """
    Unified execution of sparse recovery baselines and MESFM variants.
    Methods: 'ISTA', 'FISTA', 'ISTA + LS', 'FISTA + LS', 'MESFM Stage I', 'MESFM'
    """
    t0 = time.perf_counter()
    n = X.shape[1]
    w = np.zeros(n)

    # Guard against zero sensing matrix (L = 0)
    L = np.linalg.norm(X, 2) ** 2
    if L <= 1e-15:
        return w, np.array([], dtype=int), np.array([], dtype=int), 0.0

    a = 1.0 / L
    t = rho * a * np.max(np.abs(X.T @ y))

    z = w.copy()
    q = 1.0
    use_fista = method in ["FISTA", "FISTA + LS"]
    use_adaptive = method in ["MESFM Stage I", "MESFM"]
    use_ls = method in ["ISTA + LS", "FISTA + LS", "MESFM"]

    for _ in range(K):
        old = w.copy()
        u = z if use_fista else w
        v = u - a * (X.T @ (X @ u - y))
        av = np.abs(v)

        if use_adaptive:
            # Evaluate thresholding rule strictly on entries exceeding threshold t
            w = np.zeros_like(v)
            mask = av > t
            if np.any(mask):
                av_m = av[mask]
                w[mask] = np.sign(v[mask]) * (av_m - t / (1.0 + (av_m / (t + eps)) ** p))
        else:
            w = np.sign(v) * np.maximum(av - t, 0.0)

        if use_fista:
            qn = (1.0 + np.sqrt(1.0 + 4.0 * q * q)) / 2.0
            z = w + ((q - 1.0) / qn) * (w - old)
            q = qn

    # Unified support evaluation across all algorithms
    supp_raw_stage1 = np.flatnonzero(np.abs(w) > 1e-8)
    peak = np.max(np.abs(w))
    supp_eta = np.flatnonzero(np.abs(w) > eta * peak) if peak > 0 else np.array([], dtype=int)

    if use_ls:
        w_final = np.zeros(n)
        if len(supp_eta) > 0:
            w_final[supp_eta] = np.linalg.lstsq(X[:, supp_eta], y, rcond=None)[0]
    else:
        w_final = w

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return w_final, supp_eta, supp_raw_stage1, runtime_ms


def run_benchmark_suite(num_seeds=100):
    methods = ["ISTA", "FISTA", "ISTA + LS", "FISTA + LS", "MESFM Stage I", "MESFM"]
    noise_levels = [0.0, 0.05, 0.15]
    rows = []

    with threadpool_limits(limits=1):
        for sigma in noise_levels:
            for seed in range(num_seeds):
                rng = np.random.RandomState(seed)
                m, n, s = 40, 100, 5
                X = rng.randn(m, n) / np.sqrt(m)
                truth = np.zeros(n)
                supp = rng.choice(n, s, replace=False)
                truth[supp] = rng.choice([-1, 1], s) * rng.uniform(1.5, 3.2, s)
                y = X @ truth + rng.normal(0.0, sigma, m)

                for m_name in methods:
                    w_hat, s_eta, s_raw, dt_ms = run_method(X, y, m_name, K=400)
                    err_pct = np.linalg.norm(w_hat - truth) / np.linalg.norm(truth) * 100.0
                    rows.append({
                        "noise": sigma,
                        "seed": seed,
                        "method": m_name,
                        "error_pct": err_pct,
                        "exact_supp_eta": int(set(s_eta) == set(supp)),
                        "exact_supp_raw": int(set(s_raw) == set(supp)),
                        "time_ms": dt_ms,
                    })

    df = pd.DataFrame(rows)
    df.to_csv("MESFM_preliminary_results.csv", index=False)

    summary = df.groupby(["noise", "method"], sort=False).agg(
        mean_err=("error_pct", lambda x: round(np.mean(x), 2)),
        median_err=("error_pct", lambda x: round(np.median(x), 2)),
        std_err=("error_pct", lambda x: round(np.std(x, ddof=1), 2)),
        supp_eta_pct=("exact_supp_eta", lambda x: round(np.mean(x) * 100.0, 1)),
        supp_raw_pct=("exact_supp_raw", lambda x: round(np.mean(x) * 100.0, 1)),
        mean_time_ms=("time_ms", lambda x: round(np.mean(x), 2)),
    )
    print("=== Summary across 100 Seeds per Noise Level ===")
    print(summary)

    print("\n=== Paired Statistical Comparisons (MESFM vs. ISTA + LS) ===")
    for sigma in noise_levels:
        sub = df[df["noise"] == sigma]
        e_mesfm = sub[sub["method"] == "MESFM"]["error_pct"].values
        e_ista_ls = sub[sub["method"] == "ISTA + LS"]["error_pct"].values
        diff = e_mesfm - e_ista_ls

        _, p_wilcoxon = stats.wilcoxon(e_mesfm, e_ista_ls, zero_method="zsplit")
        s_mesfm = sub[sub["method"] == "MESFM"]["exact_supp_eta"].values
        s_ista_ls = sub[sub["method"] == "ISTA + LS"]["exact_supp_eta"].values
        b = int(np.sum((s_mesfm == 1) & (s_ista_ls == 0)))
        c = int(np.sum((s_mesfm == 0) & (s_ista_ls == 1)))
        p_mcnemar = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0

        print(
            f"Noise {sigma:.2f} | Mean Error Diff: {np.mean(diff):+.2f}% | "
            f"Wilcoxon p={p_wilcoxon:.4f} | Support Wins (MESFM+/ISTA-): {b}/{c} (McNemar p={p_mcnemar:.4f})"
        )


if __name__ == "__main__":
    run_benchmark_suite(num_seeds=100)
    
