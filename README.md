# MESFM — Sparse Signal Recovery
**Magdoleen Exact Subspace Functional Model (MESFM)**  
**Researcher:** Dr. Magdoleen Mutseam Mudawi  
**Status:** Exploratory research code and preliminary numerical experiments; not a peer-reviewed publication.

## Overview

This repository investigates sparse signal recovery from underdetermined noisy linear measurements using a two-stage procedure:

1. **Lipschitz-calibrated iterative adaptive thresholding** to estimate a sparse signal.
2. **Support-restricted least-squares refinement** to re-estimate coefficients on the selected support.

The method is studied in the setting

$$y = Xw^\star + \varepsilon, \qquad X\in\mathbb{R}^{m\times n},\quad m<n,$$

where $w^\star$ is sparse and $\varepsilon$ represents measurement noise.

The project name contains “Exact,” but **exact reconstruction is not guaranteed**: the reported experiment has a nonzero reconstruction error, and general convergence and recovery guarantees remain to be established.

## Method

For $f(w)=\frac12\|Xw-y\|_2^2$, the implementation uses

$$L=\|X\|_2^2,\qquad \alpha=1/L,$$

and updates

$$z^{(k)}=w^{(k)}-\alpha X^T(Xw^{(k)}-y),\qquad w^{(k+1)}=T_{\tau,p}(z^{(k)}),$$

with componentwise thresholding

$$T_{\tau,p}(z)=\begin{cases}0,&|z|\le\tau,\\ \operatorname{sign}(z)\left(|z|-\frac{\tau}{1+(|z|/(\tau+10^{-8}))^p}\right),&|z|>\tau.\end{cases}$$

The threshold is $\tau=\rho\alpha\|X^Ty\|_\infty$. Following the iterations, the selected support is

$$\widehat S=\{i:|w_i|>\eta\|w\|_\infty\},$$

and the final estimate solves

$$\widehat w_{\widehat S}=\arg\min_v\|X_{\widehat S}v-y\|_2^2,\qquad \widehat w_{\widehat S^c}=0.$$

The implementation uses `numpy.linalg.lstsq` for this refinement. The thresholding rule is discontinuous at its cutoff; the step-size choice alone does not prove convergence of the complete algorithm.

## Initial reproducible experiment

The supplied notebook uses a synthetic Gaussian sensing matrix with **40 measurements**, **100 features**, **5 nonzero coefficients**, noise standard deviation **0.05**, random seed **42**, **400 iterations**, $\rho=0.25$, $p=2.5$, and $\eta=0.20$.

It reports a **relative reconstruction error of approximately 1.25%** in this single example, calculated as

$$100\,\frac{\|\widehat w-w^\star\|_2}{\|w^\star\|_2}\%.$$

This is an illustrative result, **not** a general performance guarantee.

## Repository files

| File | Purpose |
|---|---|
| `magdoleen_exact_subspace_model.py` | Main Python implementation and synthetic demonstration |
| `MESFM_preliminary_benchmark.py` | Preliminary comparison script for ISTA, FISTA, and MESFM |
| `MESFM_preliminary_results.csv` | Summary of preliminary benchmark outputs |
| `MESFM_ablation_results.csv` | Preliminary component-ablation results |

Additional sensitivity, ablation, and independent-validation scripts and data should be uploaded alongside their corresponding results to make the research auditable.

## Running the project

Python 3 with NumPy and Matplotlib is required for the main example. The preliminary benchmark script also imports `threadpoolctl`.

```bash
python -m pip install numpy matplotlib threadpoolctl
python magdoleen_exact_subspace_model.py
python MESFM_preliminary_benchmark.py
```

Alternatively, open the notebook in Kaggle or Jupyter and run its cells. Verify the notebook's plots and output against a fresh execution.

## Evaluation and limitations

Preliminary experiments examine reconstruction error and exact support identification. Further work is needed to:

- compare against independently tuned ISTA, FISTA, and established support-refinement methods under matched conditions;
- test multiple seeds, sparsity levels, noise regimes, and measurement ratios;
- report uncertainty and runtime, and conduct held-out validation;
- isolate the contributions of adaptive thresholding and least-squares refinement;
- determine whether the adaptive rule admits a variational interpretation and prove any convergence or support-recovery guarantees;
- establish novelty relative to existing sparse recovery literature.

**Research integrity note:** The label “fractal” describes the author's adaptive thresholding construction; no fractal-geometric theorem or formal nonconvex penalty equivalence is established by the current code. The repository should not be cited as a peer-reviewed paper.

## References

- Donoho, D. L. (2006). *Compressed sensing*. IEEE Transactions on Information Theory, 52(4), 1289–1306.
- Daubechies, I., Defrise, M., & De Mol, C. (2004). *An iterative thresholding algorithm for linear inverse problems with a sparsity constraint*. Communications on Pure and Applied Mathematics, 57(11), 1413–1457.
- Beck, A., & Teboulle, M. (2009). *A fast iterative shrinkage-thresholding algorithm for linear inverse problems*. SIAM Journal on Imaging Sciences, 2(1), 183–202.

## Author

**Dr. Magdoleen Mutseam Mudawi**  
Research interests: functional analysis, sparse optimization, compressed sensing, and computational mathematics.
