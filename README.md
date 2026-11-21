# MESFM — Sparse Signal Recovery
**Magdoleen Exact Subspace Functional Model (MESFM)**  
**Researcher:** Dr. Magdoleen Mutseam Mudawi — Department of Mathematics and Functional Analysis, Sudan International University  
**ORCID:** [0009-0002-7409-9835](https://orcid.org/0009-0002-7409-9835)  
**Preprint (Zenodo DOI):** [10.5281/zenodo.24422561](https://doi.org/10.5281/zenodo.24422561)  
**Status:** Reproducible research code and preprint manuscript prepared for peer-reviewed submission; not yet peer-reviewed.

## Overview

This repository investigates sparse signal recovery from underdetermined noisy linear measurements using a two-stage procedure:

1. **Lipschitz-calibrated iterative adaptive thresholding** to estimate a sparse signal (Stage I).
2. **Relative infinity-norm support pruning and support-restricted least-squares refinement** to re-estimate coefficients on the selected support (Stages II–III).

The method is studied in the setting

```math
y = Xw^\star + \varepsilon,\qquad X \in \mathbb{R}^{m\times n},\qquad m < n
```

where $w^\star$ is sparse and $\varepsilon$ represents measurement noise.

The project name contains “Exact,” referring to exact subspace projection on the estimated support, **not** an unconditional guarantee of exact signal reconstruction: noisy experiments exhibit nonzero coefficient error, and general fixed-point convergence and global support-recovery guarantees remain open theoretical questions.

## Method

For $f(w)=\frac{1}{2}\lVert Xw-y\rVert_2^2$, the implementation uses

```math
L=\lVert X\rVert_2^2,\qquad \alpha=1/L,\qquad \tau=\rho\alpha\lVert X^T y\rVert_\infty,
```

and updates

```math
z^{(k)}=w^{(k)}-\alpha X^T(Xw^{(k)}-y),\qquad w^{(k+1)}=T_{\tau,p}(z^{(k)}),
```

with componentwise adaptive thresholding

```math
T_{\tau,p}(z)=
\begin{cases}
0 & \text{if } |z|\leq\tau \\
\mathrm{sgn}(z)\left(|z|-\frac{\tau}{1+(|z|/(\tau+10^{-8}))^p}\right) & \text{if } |z|>\tau
\end{cases}
```

After $K$ iterations, coordinates exceeding $\eta\lVert w^{(K)}\rVert_\infty$ define $\widehat S$, and the final estimate solves

```math
\widehat w_{\widehat S}=\arg\min_v\lVert X_{\widehat S}v-y\rVert_2^2,\qquad \widehat w_{\widehat S^c}=0.
```

The implementation uses `numpy.linalg.lstsq` for this refinement. Because $T_{\tau,p}$ is discontinuous at $|z|=\tau$, the Lipschitz step size alone does not prove convergence of the complete algorithm.

## Illustrative Experiment (Figure 1)

The illustrative example uses a synthetic Gaussian sensing matrix with **40 measurements**, **100 features**, **5 nonzero coefficients** on $\lbrace 10, 25, 50, 75, 90 \rbrace$, noise standard deviation **0.05**, random seed **42**, **400 iterations**, $\rho=0.25$, $p=2.5$, and $\eta=0.20$.

On this instance, MESFM achieves **exact support recovery with approximately 1.25% relative coefficient error**, measured as

```math
100\frac{\lVert\widehat w-w^\star\rVert_2}{\lVert w^\star\rVert_2}\%.
```

This single-seed example is illustrative and is complemented by the multi-seed statistical benchmarks below.

## Repository Files

| File | Purpose |
|---|---|
| `reproduce_benchmarks.py` | Complete CLI & Kaggle-ready script reproducing Tables 1–3, Section 5 analyses, and Figure 1 |
| `magdoleen_exact_subspace_model.py` | Standalone core MESFM class and Seed 42 demonstration |
| `MESFM_preliminary_benchmark.py` | Early exploratory comparison script for ISTA, FISTA, and MESFM |
| `MESFM_preliminary_results.csv` | Archived outputs from the early preliminary benchmark |
| `MESFM_ablation_results.csv` | Archived outputs from the early component-ablation check |

## Running the Project & Reproducing Paper Results

Python 3.8+ with `numpy`, `scipy`, and `matplotlib` is required for `reproduce_benchmarks.py` (`threadpoolctl` is only needed if running the legacy preliminary script).

```bash
python -m pip install numpy scipy matplotlib
```

### Reproduce All Manuscript Results
```bash
python reproduce_benchmarks.py --all
```

### Reproduce Individual Tables, Validations, and Figures
```bash
python reproduce_benchmarks.py --table1       # Table 1: 1,800 runs across 300 seed-noise pairs (seeds 0–99; sigma in {0.00, 0.05, 0.15})
python reproduce_benchmarks.py --table2       # Table 2: Paired Wilcoxon, 95% bootstrap CI, and exact McNemar tests (seeds 0–99)
python reproduce_benchmarks.py --table3       # Table 3: Sparsity scaling (seeds 500–549) & measurement scaling (seeds 1000–1049)
python reproduce_benchmarks.py --validation   # Section 5.1: Independent 100-seed parameter validation (seeds 1
