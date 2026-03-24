# Polymarket Political Market Mispricing Detector — Project Documentation

---

## Project Overview

**Goal:** Build an end-to-end machine learning system that predicts whether Polymarket political markets are mispriced — using a technique called conformal prediction that provides statistically guaranteed confidence intervals around each prediction.

**The end product:** A Streamlit app where you paste a live Polymarket political market slug and receive a signal indicating whether the market is potentially mispriced, backed by a coverage-guaranteed prediction interval.

---

## Tech Stack

- **Python 3.11** — core language
- **Polars** — data loading, filtering, feature engineering
- **XGBoost** — classification model
- **MAPIE** — conformal prediction wrapper (to be implemented)
- **MLflow** — experiment tracking, model registry
- **Streamlit** — live inference app (to be built)
- **Polymarket REST API** — live market data at inference time
- **HuggingFace Datasets** — source of historical trade data

---

## Project Structure
```
/polymarket-conformal
  /data
    /raw
      quant.parquet                   ← 29GB full trade history (HuggingFace)
      markets.parquet                 ← 68MB market metadata (HuggingFace)
    /sample
      quant_sample_political.parquet  ← 1M row sample used on laptop
    /processed
      markets_political.parquet       ← 20,584 filtered political markets
      quant_political.parquet         ← 31.7M trades for political markets
      feature_matrix.parquet          ← 14,443 markets, 17 features
    /model
      train.parquet                   ← 8,665 markets (60%, 2022-2025)
      calibration.parquet             ← 2,889 markets (20%, 2025)
      test.parquet                    ← 2,889 markets (20%, 2026)
  /src
    features.py                       ← feature engineering class
    model.py                          ← training, evaluation, MLflow
  /notebooks
    model-analysis.ipynb              ← overfitting and calibration checks
  mlruns/                             ← MLflow experiment runs
  requirements.txt
  README.md
```

---

## Data Sources

### HuggingFace Dataset — `SII-WANGZJ/Polymarket_data`

Published by researchers from Shanghai Innovation Institute, Westlake University, Shanghai Jiao Tong University, Harbin Institute of Technology, and Fudan University. MIT licensed.

**`markets.parquet` (68MB)**
- 607,561 total markets
- Columns: `id`, `question`, `condition_id`, `closed`, `outcome_prices`, `volume`, `created_at`, `end_date`, `neg_risk`, and more
- Resolution outcome derivable from `outcome_prices` column

**`quant.parquet` (29GB)**
- 293.3 million trade records
- Columns: `timestamp`, `market_id`, `condition_id`, `price`, `usd_amount`, `token_amount`, `side`, `maker`, `taker`
- All trades normalised to YES token perspective
- Full Polymarket history from November 2022 to March 2026

---

## Stage 1 — Data Exploration and Validation

### Markets Dataset

Loaded `markets.parquet` fully into memory (68MB). Applied the following filters to isolate closed political markets:

**Step 1 — Filter to closed markets**
```python
markets.filter(pl.col("closed") == True)
```
Result: 572,547 closed markets

**Step 2 — Political keyword filter on question column**

Applied keywords: `election`, `president`, `congress`, `senate`, `minister`, `vote`, `democrat`, `republican`, `trump`, `biden`, `harris`, `political`, `govern`, `parliament`, `geopolit`, `ukraine`, `israel`, `nato`, `war`, `sanction`

**Step 3 — Sports noise removal**

Removed markets containing: `o/u`, `spread`, `over/under`, `moneyline`, `vs.`, `nfl`, `nba`, `mlb`, `nhl`, `epl`

**Step 4 — Derive target variable**

`outcome_prices` stored as string `"['1', '0']"`. Parsed first value — if > 0.5 the market resolved YES.
```python
def parse_resolved_yes(x):
    parsed = json.loads(x.replace("'", '"'))
    return float(parsed[0]) > 0.5
```

**Final political markets dataset:**
- 20,584 closed political markets
- Date range: October 2020 → March 2026
- Class balance: 68% NO / 32% YES
- Zero nulls

### Quant Dataset

Loaded `quant.parquet` using `pl.scan_parquet` to avoid loading 29GB into memory. Filtered to political market condition IDs only before collecting:
```python
df = (
    pl.scan_parquet('data/raw/quant.parquet')
    .filter(pl.col('condition_id').is_in(m_list))
    .collect()
)
```

Load time: 1 minute 22 seconds. Result: 31,722,655 rows across 20,111 unique markets.

**Findings:**
- 578 markets had zero trades in quant.parquet — dropped silently
- Date range: November 2022 → March 2026
- Median 218 trades per market, max 2.75M (2024 US election market)
- Zero nulls

**Timestamp alignment issue resolved:**
- `quant.parquet` timestamps were Unix epoch integers (seconds)
- `markets.parquet` end_date was `datetime[ms, UTC]`
- Fixed by converting quant timestamps:
```python
quant.with_columns(
    pl.from_epoch(pl.col('timestamp'), time_unit='s').alias('datetime')
).with_columns(
    pl.col('datetime').dt.convert_time_zone('UTC').dt.cast_time_unit('ms')
)
```

---

## Stage 2 — Feature Engineering (`src/features.py`)

### Design Decisions

**Single function for training and inference:** All feature engineering lives in `features.py` and is called identically for both historical training data and live API data at inference time. This prevents training/inference skew — the most common silent failure in production ML.

**Lookback window approach:** Rather than using all available trade history, features are computed from a fixed 30-day window before each market's resolution date. This simulates what would be known at inference time.

**Leakage protection:** Any trade on or after `end_date` is excluded. This is critical — trades on resolution day have prices near 0 or 1 because the outcome is already known.

### Constants
```python
LOOKBACK_DAYS = 30    # days of history used per market
MIN_TRADES = 10       # markets with fewer trades are dropped
```

### Window Logic

For each market:
1. Calculate `window_start = end_date - 30 days`
2. Filter trades to `window_start <= datetime < end_date`
3. If fewer than 10 trades remain, skip the market

### Features Engineered

**From price series (within window):**

| Feature | Description |
|---|---|
| `price_start` | First trade price in window |
| `price_end` | Last trade price before resolution |
| `price_mean` | Average price across all trades |
| `price_min` | Lowest price in window |
| `price_max` | Highest price in window |
| `price_momentum` | `price_end - price_start` — positive means trending YES |
| `price_volatility` | Standard deviation of prices — high means unstable |
| `price_range` | `price_max - price_min` |

**From volume and activity (within window):**

| Feature | Description | Transform |
|---|---|---|
| `log_total_volume` | Total USD traded | log1p applied |
| `log_trade_count` | Number of individual trades | log1p applied |
| `log_avg_trade_size` | Mean USD per trade | log1p applied |

Log1p transformation applied to all volume features because raw volume ranges from $0 to $1.5B — without transformation it would dominate the model.

**Sentiment:**

| Feature | Description |
|---|---|
| `buy_ratio` | Proportion of BUY trades. Above 0.5 = more people buying YES |

**Market metadata:**

| Feature | Description | Transform |
|---|---|---|
| `log_market_volume` | All-time total volume for the market | log1p applied |
| `days_active` | Days between `created_at` and `end_date` | |
| `days_to_resolution` | Days remaining at end of lookback window | |

### Feature Matrix Output

- **14,443 markets** processed successfully (down from 20,111 — remainder had fewer than 10 trades in the 30-day window)
- **17 columns** total (market_id, resolved_yes, 15 features)
- **Zero nulls** across all columns
- **Class balance:** 28% YES / 72% NO

---

## Stage 3 — Time-Based Data Splits (`src/model.py`)

### Why Time-Based Not Random

Political markets behave differently across time — Polymarket in 2022 was a small platform, by 2025 it was a major market with different liquidity and participant dynamics. A random split would mix time periods and overstate how well the model generalises to genuinely future markets.

Splitting chronologically by `end_date` simulates real-world performance: train on old markets, evaluate on newer ones the model has never seen.

### Split Logic
```python
data_copy = feature_matrix.sort('end_date')
n = len(data_copy)
train_cutoff = int(n * 0.60)
calib_cutoff = int(n * 0.80)

train       = data_copy[:train_cutoff]
calibration = data_copy[train_cutoff:calib_cutoff]
test        = data_copy[calib_cutoff:]
```

### Split Results

| Split | Markets | Date Range | YES Rate |
|---|---|---|---|
| Train | 8,665 | Dec 2022 → Nov 2025 | 29.82% |
| Calibration | 2,889 | Nov 2025 → Jan 2026 | 25.34% |
| Test | 2,889 | Jan 2026 → Apr 2026 | 25.16% |

**Note:** YES rate drops from train to test. Recent markets (2025-2026) have more NO resolutions, likely because more speculative short-duration markets were created in that period. Noted as a known limitation.

**Why three splits:** The calibration set is reserved exclusively for MAPIE. MAPIE uses it to calculate what interval width guarantees 90% coverage. If calibration data leaked into training, the coverage guarantee would break silently.

---

## Stage 4 — Model Training (`src/model.py`)

### MLflow Setup

Experiment name: `polymarket-conformal`

All runs logged with:
- Parameters
- Metrics: Brier Score, AUC-ROC, Log Loss
- Artifacts: model file

### Baseline Model
```python
params = {
    'scale_pos_weight': 2.47,  # ratio of NO to YES: 10400/4043
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.8,
    'min_child_weight': 5,
    'eval_metric': 'logloss',
    'random_state': 42,
}
```

**Why `scale_pos_weight = 2.47`:** Dataset is 72% NO / 28% YES. Without correction XGBoost is biased towards predicting NO for everything. This weight tells XGBoost to treat each YES market as 2.47x more important during training.

**Baseline results:**

| Metric | Value |
|---|---|
| Brier Score | 0.0598 |
| AUC-ROC | 0.9724 |
| Log Loss | 0.1909 |

### Hyperparameter Search with Optuna

100 trials, each logged as a nested MLflow child run under a parent `optuna_search` run.

**Search space:**
```python
'n_estimators':     trial.suggest_int('n_estimators', 100, 500)
'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
'max_depth':        trial.suggest_int('max_depth', 3, 5)
'subsample':        trial.suggest_float('subsample', 0.6, 1.0)
'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
'reg_alpha':        trial.suggest_float('reg_alpha', 0.01, 1.0)
'reg_lambda':       trial.suggest_float('reg_lambda', 0.01, 1.0)
'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
```

**Final model results (version 3):**

| Metric | Value |
|---|---|
| Brier Score | 0.0573 |
| AUC-ROC | 0.9723 |

Model registered in MLflow registry as `polymarket-xgboost v3`.

---

---

## Stage 5 Update — Model Fixes and Retraining

### Problem Identified

After initial model analysis, all price level features were found to be highly correlated with the target variable:

| Feature | Correlation with resolved_yes |
|---|---|
| price_end | 0.8399 |
| price_mean | 0.7869 |
| price_min | 0.6380 |
| price_max | 0.6241 |
| price_start | 0.4833 |

This was because the majority of markets in the dataset are measured very close to their resolution date — the price already encodes the answer. The model was reading a nearly-resolved price rather than genuinely predicting from uncertainty.

### Attempted Fixes

**Attempt 1 — Remove price_end only**
Updated `EXCLUDE_COLS` to include `price_end`. Model performance unchanged — Brier 0.0573, AUC 0.9723. The model simply shifted importance to `price_mean` which carries near-identical information.

**Attempt 2 — Change window strategy to 60-30 days before resolution**
Investigated using a window 60-30 days before resolution to force genuinely uncertain predictions. Only 2,621 markets had sufficient trades in this window — too thin for reliable training and MAPIE calibration. Ruled out.

**Attempt 3 — Change window strategy to 45-15 days before resolution**
Even fewer viable markets at 2,389. Ruled out.

**Final fix — Remove all price level features**
Removed `price_start`, `price_end`, `price_mean`, `price_min`, `price_max` from the feature set entirely. Kept only dynamic and metadata features:
```python
EXCLUDE_COLS = [
    'market_id', 'resolved_yes', 'end_date',
    'price_start', 'price_end', 'price_mean',
    'price_min', 'price_max'
]
```

Remaining features:
- `price_momentum` — direction of price movement over the window
- `price_volatility` — stability of the market
- `price_range` — how much the price moved
- `log_total_volume` — total USD traded
- `log_trade_count` — number of trades
- `log_avg_trade_size` — average trade size
- `buy_ratio` — proportion of BUY trades
- `log_market_volume` — all-time market volume
- `days_active` — total market duration
- `days_to_resolution` — days remaining at end of window

Also updated `scale_pos_weight` to reflect the actual class balance in the retrained dataset.

### Retrained Model Results (v4)

**Train vs Test:**

| | Train | Test |
|---|---|---|
| Brier Score | 0.0950 | 0.0790 |
| AUC-ROC | 0.9529 | 0.9399 |

Overfitting gap reduced from 0.028 to 0.013 — model generalises significantly better.

**Feature importance now distributed across all features:**

| Feature | Importance |
|---|---|
| price_momentum | 33% |
| days_to_resolution | 13% |
| days_active | 11% |
| buy_ratio | 11% |
| price_volatility | 9% |
| price_range | 5% |
| log_avg_trade_size | 5% |
| log_market_volume | 5% |
| log_trade_count | 5% |
| log_total_volume | 4% |

No single feature dominates. Model is learning from genuine market dynamics.

**Feature correlations with target (all below 0.55):**

| Feature | Correlation |
|---|---|
| price_momentum | 0.5399 |
| buy_ratio | 0.4953 |
| price_volatility | 0.3488 |
| log_avg_trade_size | 0.3455 |
| price_range | 0.3405 |
| log_market_volume | 0.1535 |
| days_to_resolution | 0.1799 |
| log_trade_count | 0.1169 |
| log_total_volume | 0.2709 |
| days_active | -0.1294 |

**Calibration curve:** Still overconfident in the 0.3-0.6 range. Will be corrected with `CalibratedClassifierCV` before MAPIE.

**Prediction distribution:** No longer strongly bimodal. Predictions are spread more naturally across the probability range with genuine uncertainty expressed in the middle range.

### Model registered in MLflow as `polymarket-xgboost v4`

---

## Updated Metrics Summary

| Model Version | Brier Score | AUC-ROC | Notes |
|---|---|---|---|
| Baseline | 0.0598 | 0.9724 | Default params, all features |
| Optuna v2 | 0.0567 | 0.9717 | max_depth=8, overfitting identified |
| Optuna v3 | 0.0573 | 0.9723 | max_depth capped, price_end removed — no improvement |
| Optuna v4 | 0.0790 | 0.9399 | All price level features removed, genuine dynamics model |

---


## Metrics Summary

| Model Version | Brier Score | AUC-ROC | Notes |
|---|---|---|---|
| Baseline | 0.0598 | 0.9724 | Default params, all features including price levels |
| Optuna v2 | 0.0567 | 0.9717 | max_depth=8, overfitting identified |
| Optuna v3 | 0.0573 | 0.9723 | max_depth capped, price_end removed — no improvement |
| Optuna v4 | 0.0790 | 0.9399 | All price level features removed, genuine dynamics model |
| Optuna v5 | 0.0731 | 0.9521 | Consistency features added, best overall result |
