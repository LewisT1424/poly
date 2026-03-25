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
- **MAPIE 1.3** — conformal prediction wrapper
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
      feature_matrix.parquet          ← 14,443 markets, 22 columns
    /model
      train.parquet                   ← 8,665 markets (60%, 2022-2025)
      calibration.parquet             ← 2,889 markets (20%, 2025)
      test.parquet                    ← 2,889 markets (20%, 2026)
  /src
    features.py                       ← feature engineering class
    model.py                          ← training, evaluation, MLflow
    conformal.py                      ← calibration, MAPIE, signal function
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

**Lookback window approach:** Features are computed from a fixed 30-day window before each market's resolution date. This simulates what would be known at inference time.

**Leakage protection:** Any trade on or after `end_date` is excluded. Trades on resolution day have prices near 0 or 1 because the outcome is already known.

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

**Cross-market consistency features (neg_risk = 1 markets only):**

| Feature | Description |
|---|---|
| `consistency_gap` | How far this market's price deviates from equal-share baseline across siblings |
| `n_siblings` | Number of related markets in the same event |
| `sibling_volume_ratio` | Log of sibling total volume / this market's volume |
| `neg_risk` | Binary flag — mutually exclusive event (1) or independent (0) |

Consistency features only computed for `neg_risk = 1` markets — these are mutually exclusive outcome markets where prices should sum to 1.0. Independent markets (`neg_risk = 0`) get 0 for all consistency features.

Additional filter: only compute consistency where sibling prices sum between 0.7 and 1.3 — outside this range too many siblings are missing.

### Feature Matrix Output

- **14,443 markets** processed
- **22 columns** total
- **14 features** used for training (price level features excluded at model stage)
- **Zero nulls**
- **Class balance:** 28% YES / 72% NO

---

## Stage 3 — Time-Based Data Splits (`src/model.py`)

### Why Time-Based Not Random

Splitting chronologically by `end_date` simulates real-world performance: train on old markets, evaluate on newer ones the model has never seen.

### Split Results

| Split | Markets | Date Range | YES Rate |
|---|---|---|---|
| Train | 8,665 | Dec 2022 → Nov 2025 | 29.82% |
| Calibration | 2,889 | Nov 2025 → Jan 2026 | 25.34% |
| Test | 2,889 | Jan 2026 → Apr 2026 | 25.16% |

**Why three splits:** The calibration set is reserved exclusively for MAPIE. If calibration data leaked into training the coverage guarantee would break silently.

---

## Stage 4 — Model Training (`src/model.py`)

### Feature Set Used for Training

Price level features excluded to prevent the model reading near-resolved prices:
```python
EXCLUDE_COLS = [
    'market_id', 'resolved_yes', 'end_date',
    'price_start', 'price_end', 'price_mean',
    'price_min', 'price_max'
]
```

14 features used: `price_momentum`, `price_volatility`, `price_range`, `log_total_volume`, `log_trade_count`, `log_avg_trade_size`, `buy_ratio`, `log_market_volume`, `days_active`, `days_to_resolution`, `neg_risk`, `n_siblings`, `consistency_gap`, `sibling_volume_ratio`

### Final Model Results (v5)

**Train vs Test:**

| | Train | Test |
|---|---|---|
| Brier Score | 0.0818 | 0.0731 |
| AUC-ROC | 0.9682 | 0.9521 |
| Log Loss | — | 0.2457 |

Overfitting gap: 0.016 — healthy generalisation.

**Feature importance:**

| Feature | Importance |
|---|---|
| price_momentum | 25% |
| consistency_gap | 19% |
| days_to_resolution | 11% |
| days_active | 9% |
| sibling_volume_ratio | 7% |
| buy_ratio | 5% |
| price_volatility | 4% |
| All others | 2-3% each |

**Model registered in MLflow as `polymarket-xgboost v5`**

---

## Stage 5 — Model Iteration History

| Version | Brier | AUC | Notes |
|---|---|---|---|
| Baseline | 0.0598 | 0.9724 | Default params, all features |
| v2 | 0.0567 | 0.9717 | Optuna search, max_depth=8, overfitting identified |
| v3 | 0.0573 | 0.9723 | max_depth capped at 5, price_end removed — no improvement |
| v4 | 0.0790 | 0.9399 | All price level features removed, genuine dynamics model |
| v5 | 0.0731 | 0.9521 | Consistency features added, best overall result |

---

## Stage 6 — Probability Calibration + Conformal Prediction (`src/conformal.py`)

### Overview

`conformal.py` takes the trained XGBoost v5 model and wraps it with:
1. Platt scaling to correct probability miscalibration
2. MAPIE `SplitConformalClassifier` to produce guaranteed prediction sets

### Why Probability Calibration First

XGBoost's raw probability outputs were overconfident in the 0.3-0.6 range — the calibration curve dipped below the diagonal. MAPIE's conformity scores are built directly from probability outputs so miscalibrated probabilities produce unreliable intervals.

### Platt Scaling Implementation

`CalibratedClassifierCV` with `cv='prefit'` was removed in sklearn 1.8. Implemented manually:
```python
# Get raw XGBoost probabilities on calibration set
raw_calib_probs = xgb_model.predict_proba(X_calib)[:, 1].reshape(-1, 1)

# Fit logistic regression correction on top — Platt scaling
platt = LogisticRegression()
platt.fit(raw_calib_probs, y_calib)
```

Wrapped in a `CalibratedModel` class with `predict_proba` and `predict` methods to maintain sklearn compatibility with MAPIE.

**Calibration results:**
- Brier Score before: 0.0731
- Brier Score after: 0.0692
- Curve moved closer to diagonal in 0.3-0.6 range

### Conformal Prediction with MAPIE

Used `SplitConformalClassifier` with LAC conformity score. APS is restricted to multiclass in MAPIE 1.3 — for binary classification LAC produces equivalent results.
```python
mapie = SplitConformalClassifier(
    estimator=calib_model,
    confidence_level=0.9,
    conformity_score='lac',
    prefit=True
)
mapie.conformalize(X_calib, y_calib)
```

### Coverage Verification

| Metric | Value |
|---|---|
| Empirical coverage | 0.9225 ✅ (must be ≥ 0.90) |
| Signal rate | 0.9723 |
| Ambiguity rate | 0.0277 |
| YES signals | 702 |
| NO signals | 2,107 |
| Uncertain | 80 |

Coverage of 92.25% confirms the mathematical guarantee is holding on the test set.

### Alpha Exploration

| Alpha | Confidence | Coverage | Signal Rate | Ambiguity |
|---|---|---|---|---|
| 0.05 | 95% | 0.9533 | 87.8% | 12.2% |
| 0.10 | 90% | 0.9225 | 97.2% | 2.8% |
| 0.20 | 80% | 0.8394 | 100% | 0.0% |

Alpha 0.10 (90% confidence) selected as default — best balance between signal rate and coverage guarantee. Alpha 0.20 drops below 80% coverage which is too low.

### Mispricing Signal Function
```
{YES} prediction set → 🟢 GREEN — model confident YES
{NO}  prediction set → 🔴 RED   — model confident NO
{YES, NO}            → 🟡 YELLOW — model uncertain, no signal
```

### Known Limitations

- Probability calibration and conformalization share the same calibration set — a cross-calibration approach would eliminate the minor leakage at the cost of implementation complexity. Empirical coverage of 92.25% confirms the approach is working acceptably
- Signal rate of 97% is high — the model is confident on almost every market. Combined with some remaining overconfidence, the backtest will determine whether confident signals are actually reliable

**MAPIE model registered in MLflow as `polymarket-mapie v1`**

---

## Current Status

| Stage | Status |
|---|---|
| Data extraction and validation | ✅ Complete |
| Feature engineering | ✅ Complete |
| Consistency features | ✅ Complete |
| Time-based splits | ✅ Complete |
| Model training (XGBoost v5) | ✅ Complete |
| Model analysis and overfitting checks | ✅ Complete |
| Probability calibration | ✅ Complete |
| Conformal prediction (MAPIE) | ✅ Complete |
| Backtest | 🔄 Next |
| Live inference pipeline (api.py) | ⏳ Pending |
| Streamlit app (app.py) | ⏳ Pending |
| README | ⏳ Pending |