# CRI: Markovian Criminal Resilience Index

This repository provides a Python workflow for forecasting municipal crime rates and constructing a **Crime Resilience Index (CRI)** in Mexico. It combines regime-specific Markov transition dynamics with a comparative forecasting pipeline and optional spatial smoothing.

The CRI summarizes downward mobility in crime states, position relative to a regime's stationary expected crime rate, and persistence in low or high crime states. Higher scores indicate greater resilience under this operational definition. Municipal coverage and analysis years depend on the input data and configuration.

## Overview & Methodology

Municipalities are grouped into violence regimes, with crime-rate states defined within each regime. Transition matrices describe movement between these states. The CRI uses the municipality's base-year state and its regime's transition dynamics to construct three components.

### Transition dynamics

For a municipality in state $s$ of regime $r$, let $P^{(r)}$ be the transition matrix:

$$D_i = \sum_{j<s} P^{(r)}_{sj}, \qquad U_i = \sum_{j>s} P^{(r)}_{sj}, \qquad \pi_i = P^{(r)}_{ss}.$$

Here, $D_i$ is the probability of moving to a lower crime state, $U_i$ is the probability of moving to a higher state, and $\pi_i$ is the probability of remaining in the current state.

The code computes a stationary distribution $\mu^{(r)}$ satisfying $\mu^{(r)}P^{(r)}=\mu^{(r)}$. With state mean crime rates $m_{r,j}$, the stationary expected rate is:

$$\bar c_r = \sum_j \mu^{(r)}_j m_{r,j}.$$

Regime severity weights are calculated as:

$$\omega_r = 1 - \frac{\log(1+\bar c_r)}{\max_k \log(1+\bar c_k)}.$$

This gives lower-violence regimes larger weights and the regime with the highest stationary expected rate a weight of zero.

### CRI components

| Component | Implemented formula | Interpretation |
| --- | --- | --- |
| `TRS` | $(D_i-U_i)\omega_r$ | Net downward mobility scaled by regime severity. |
| `SCS` | $-G_i$ | Position relative to the regime's stationary expected rate. |
| `PP` | $+\pi_i$ for $s\leq2$; $-\pi_i$ for $s>2$ | Rewards persistence in low states and penalizes persistence in high states. |

The stationary gap is:

$$G_i = \frac{\bar c_r-m_{r,s}}{\max_j m_{r,j}-\min_j m_{r,j}}.$$

The implementation sets $G_i=0$ when the state means have no range. A positive `SCS` means the current state's mean exceeds the stationary expected rate, indicating a downward gap under the model. It does not imply a low current crime rate.

### Composite index

The components are combined using a weighted arithmetic sum:

$$\mathrm{CRI}_{i,\mathrm{raw}} = \alpha\,\mathrm{TRS}_i + \beta\,\mathrm{SCS}_i + \gamma\,\mathrm{PP}_i, \qquad \alpha+\beta+\gamma=1.$$

The weights are configured through `CRI_ALPHA`, `CRI_BETA`, and `CRI_GAMMA` in `src/config.py`. Scores are then min-max normalized across the municipalities included in the calculation:

$$\mathrm{CRI}_i = \frac{\mathrm{CRI}_{i,\mathrm{raw}}-\min_k\mathrm{CRI}_{k,\mathrm{raw}}}{\max_k\mathrm{CRI}_{k,\mathrm{raw}}-\min_k\mathrm{CRI}_{k,\mathrm{raw}}}.$$

### Optional spatial smoothing

When `USE_SPATIAL_LAG` is enabled and a spatial weights matrix is provided, the code row-standardizes the matrix and blends each municipality's normalized score with its neighbors' scores:

$$\widetilde{\mathrm{CRI}}_i = (1-\lambda)\mathrm{CRI}_i + \lambda\sum_j W_{ij}\mathrm{CRI}_j.$$

The blended scores are min-max normalized again and stored as `CRI_spatial`. `main.py` supplies the contiguity matrix returned by `w_contiguity()`; `SPATIAL_LAMBDA` controls the smoothing strength. Spatial smoothing incorporates neighboring scores into the index; it does not estimate a spatial forecasting model.

## Core Analytical Pipeline

1. Load municipal crime rates and construct the spatial contiguity matrix.
2. Generate Markov-chain, fuzzy Markov, and hybrid forecasts.
3. Compute comparison forecasts using ordinary least squares, weighted moving averages, reaction–diffusion, ARIMA, and a persistence baseline.
4. Merge forecasts with observed crime rates by municipal identifier (`CVEGEO`) and call `validate()`.
5. Save the prediction benchmark table.
6. Load regime transition matrices and state metadata; compute stationary distributions and severity weights.
7. Assign base-year states, calculate CRI components, assemble the index, and optionally apply spatial smoothing.
8. Export municipal CRI scores, interpretive categories, and console summaries.

## Getting Started

### Prerequisites

- Python **3.10+**, required by the type annotations used in the supplied CRI code.
- The repository's dependencies and input datasets.
- Municipal identifiers stored consistently as five-character `CVEGEO` strings.

### Installation

```bash
git clone https://github.com/FrancoJPM1991/crime-resilience-index.git
cd crime-resilience-index
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

If the repository includes `requirements.txt`, install its dependencies:

```bash
pip install -r requirements.txt
```

### Configuration and execution

Review `src/config.py` and set the input/output paths, analysis years, regime/state settings, forecasting parameters, and CRI weights before running the pipeline.

| Configuration | Purpose |
| --- | --- |
| `START_YEAR`, `YEAR_BASE`, `YEAR_PREDICT`, `OBSERVED_YEAR` | Historical window, base year, prediction year, and observed comparison year. |
| `N_STEPS`, `N_REGIMES`, `N_QUANTILES` | Markov forecast horizon and regime/state settings. |
| `MIN_MUNI`, `BINS` | Parameters passed to the fuzzy Markov model. |
| `AUX_MODEL`, `STD_THRESHOLD`, `MA_WINDOW` | Hybrid model and moving-average parameters. |
| `CRI_ALPHA`, `CRI_BETA`, `CRI_GAMMA` | CRI component weights, which must sum to one. |
| `USE_SPATIAL_LAG`, `SPATIAL_LAMBDA` | Spatial smoothing switch and blend parameter. |
| `DATA_INTERIM`, `OUTPUT_DIR` | Intermediate data and CRI export locations. |

Ensure the configured output directory and `results/` exist, then run from the repository root:

```bash
python main.py
```

For a single-year benchmark, align the prediction horizon and observed comparison year. The entry point runs forecast validation before constructing the CRI.

### Intermediate inputs required by the CRI

The CRI modules expect these files under `DATA_INTERIM`:

| File | Required content |
| --- | --- |
| `crime_rates_regimes.csv` | `CVEGEO`, `year`, `crime_rate`, and `regimes`. |
| `T_df_regime1.csv` through `T_df_regime5.csv` | Regime transition matrices with integer state labels in rows and columns. |
| `state_meta_df_regime1.csv` through `state_meta_df_regime5.csv` | State metadata containing `state`, `mean`, `bin_lower`, and `bin_upper`. |

The metadata loader currently supports five regimes. Transition matrices and state metadata must describe the same states in matching order. Municipalities with missing regime assignments or unavailable regime metadata are excluded from component calculation.

## Outputs

| Output | Description |
| --- | --- |
| `results/predictions_benchmark.csv` | Municipal forecasts from all comparison models, persistence rates, and observed crime rates. |
| `OUTPUT_DIR/cri.csv` | Municipal identifiers, regime labels, current states, CRI components, raw and normalized scores, spatial scores, and resilience labels. |
| Console summaries | Forecast validation results, CRI statistics by regime/category, and the ten highest- and lowest-scoring municipalities. |

`export_cri()` uses `CRI_spatial` for labels when spatial scores are requested and complete; otherwise it uses `CRI`. Categories follow the implemented thresholds:

| Score range | Label |
| --- | --- |
| $0.75\leq\mathrm{CRI}\leq1$ | High Resilience |
| $0.50\leq\mathrm{CRI}<0.75$ | Moderate Resilience |
| $0.25\leq\mathrm{CRI}<0.50$ | Moderate Vulnerability |
| $0\leq\mathrm{CRI}<0.25$ | High Vulnerability |

## Main Modules

| Module | Role |
| --- | --- |
| `main.py` | Runs the forecasting, validation, and CRI pipeline. |
| `src/config.py` | Defines paths and analysis parameters. |
| `src/data_loader.py` | Provides crime-rate and regime datasets. |
| `src/weights.py` | Provides spatial weights, including contiguity. |
| `src/models/` | Contains Markov, fuzzy Markov, hybrid, ARIMA, OLS, moving-average, and reaction–diffusion models. |
| `src/validation.py` | Validates the merged forecast benchmark table. |
| `src/cri/cri_weights.py` | Loads regime metadata and computes stationary distributions and severity weights. |
| `src/cri/cri_components.py` | Calculates `D`, `U`, `pi`, `G`, `TRS`, `SCS`, and `PP`. |
| `src/cri/cri_index.py` | Aggregates, normalizes, and spatially smooths CRI scores. |
| `src/cri/cri_output.py` | Exports scores and generates interpretive summaries. |

## Interpretation & Reproducibility

The CRI is a relative, model-based indicator of crime-state dynamics. Its values depend on regime/state definitions, the estimated transition matrices, component weights, and spatial smoothing. Min-max normalization makes scores relative to the municipalities in each run, so comparisons across separately normalized years require care.

Forecast validation assesses predictive performance; it does not by itself establish external validity of the resilience index. Category cutoffs are interpretive thresholds used by the exporter.

When reproducing results, record the data version, historical window, base year, state definitions, and configuration. For out-of-sample forecasting, estimate regimes, state boundaries, and transition matrices using only information available before the target year.

Current implementation details to consider:

- The component output column is named `crime_rate_2024` even when another base year is supplied.
- The persistence threshold is fixed at states 1–2 versus states above 2.
- The normalization and severity-weight formulas require nonzero denominators; constant composite scores or all-zero stationary expected rates need handling before interpretation.
- Isolated municipalities and municipalities missing from the spatial matrix receive a zero spatial lag in the current implementation.

## Contact & Support

For questions, collaborations, or suggestions:

**Franco Josué Patiño Morales, M.Sc.**  
Email: [franco.jpm@gmail.com](mailto:franco.jpm@gmail.com)

## Citation

If you use this repository in academic work, please cite:

> Patiño Morales, F. J. (2026). *CRI: Markovian Criminal Resilience Index* [GitHub repository]. https://github.com/FrancoJPM1991/crime-resilience-index
