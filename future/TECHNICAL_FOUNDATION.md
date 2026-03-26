# Technical Foundation — Existing ML Pipeline

## Purpose of This Document

This document is a complete technical reference for the ML pipeline already built. It is designed to be passed to new AI models as context so they understand what exists before helping build new components. Always read this before building anything new on top of the existing system.

---

## Project Overview

An end-to-end machine learning pipeline that applies conformal prediction to Polymarket political market data to generate statistically guaranteed trading signals.

**The honest finding:** The model achieves 92% overall accuracy but only 40.4% accuracy on genuine market disagreements. The market is efficient on near-resolved markets. The model is a filtering and prioritisation tool, not a standalone trading signal.

---

## Tech Stack

```
Language: Python 3.11
Data processing: Polars (not Pandas)
ML: XGBoost, scikit-learn 1.8
Conformal prediction: MAPIE 1.3
Experiment tracking: MLflow
Application: Streamlit
Data source: HuggingFace dataset SII-WANGZJ/Polymarket_data (MIT licensed)
```

---

## Critical Version Notes

These caused bugs during development. New models must know them:

**sklearn 1.8:**
- `CalibratedClassifierCV(cv='prefit')` was REMOVED
- Solution: Manual Platt scaling implemented via `CalibratedModel` wrapper class
- `CalibratedModel` requires `classes_` attribute passed from XGBoost

**MAPIE 1.3:**
- `MapieClassifier` was REMOVED — use `SplitConformalClassifier`
- APS conformity score is restricted to multiclass — use `lac` for binary
- API changed: `.conformalize()` instead of `.fit()`, `.predict_set()` instead of `.predict()`
- Prediction sets shape: `(n_samples, 2, 1)` — index with `[i][label][0]`

---

## File Structure

```
/polymarket-conformal/
├── data/
│   ├── raw/
│   │   ├── quant.parquet          (29GB — all Polymarket trades)
│   │   └── markets.parquet        (68MB — all market metadata)
│   ├── processed/
│   │   ├── markets_political.parquet  (20,584 filtered political markets)
│   │   ├── quant_political.parquet    (31.7M trades for political markets)
│   │   └── feature_matrix.parquet     (14,443 markets, 22 columns)
│   └── model/
│       ├── train.parquet          (8,665 markets — Dec 2022 to Nov 2025)
│       ├── calibration.parquet    (2,889 markets — Nov 2025 to Jan 2026)
│       └── test.parquet           (2,889 markets — Jan 2026 to Apr 2026)
├── src/
│   ├── features.py               (FeatureEngineer class)
│   ├── model.py                  (Model class)
│   └── conformal.py              (Conformal class)
├── notebooks/
│   └── model-analysis.ipynb
├── mlruns/                        (MLflow experiment tracking)
└── requirements.txt
```

---

## Feature Matrix Schema

```
Column                 | Type    | Description
-----------------------|---------|------------------------------------------
market_id              | str     | condition_id (Polygon wallet address format)
resolved_yes           | bool    | Target variable — did market resolve YES?
end_date               | datetime| Market resolution date (UTC, ms precision)
price_start            | f64     | First trade price in 30-day window (EXCLUDED from model)
price_end              | f64     | Last trade price before resolution (EXCLUDED)
price_mean             | f64     | Mean price in window (EXCLUDED)
price_min              | f64     | Min price in window (EXCLUDED)
price_max              | f64     | Max price in window (EXCLUDED)
price_volatility       | f64     | Std dev of prices in window
price_range            | f64     | price_max - price_min
price_momentum         | f64     | price_end - price_start (directional signal)
log_total_volume       | f64     | Total USD traded, log1p transformed
log_trade_count        | f64     | Number of trades, log1p transformed
log_avg_trade_size     | f64     | Mean USD per trade, log1p transformed
buy_ratio              | f64     | Proportion of BUY trades (0-1)
log_market_volume      | f64     | All-time market volume, log1p transformed
days_active            | f64     | Days between created_at and end_date
days_to_resolution     | f64     | Days remaining at end of lookback window
neg_risk               | u8      | 1=mutually exclusive outcomes, 0=independent
n_siblings             | u32     | Related markets in same event (neg_risk=1 only)
consistency_gap        | f64     | Price deviation from equal-share baseline (neg_risk=1 only)
sibling_volume_ratio   | f64     | Log(sibling volume / this volume) (neg_risk=1 only)
```

**EXCLUDE_COLS (never pass to model):**
```python
EXCLUDE_COLS = [
    'market_id', 'resolved_yes', 'end_date',
    'price_start', 'price_end', 'price_mean',
    'price_min', 'price_max'
]
```

**14 FEATURES USED FOR TRAINING:**
price_volatility, price_range, price_momentum, log_total_volume, log_trade_count, log_avg_trade_size, buy_ratio, log_market_volume, days_active, days_to_resolution, neg_risk, n_siblings, consistency_gap, sibling_volume_ratio

---

## Model Details

### XGBoost Model (polymarket-xgboost v5)

**Best Optuna params:**
```python
{
    'n_estimators': 404,
    'learning_rate': 0.02300,
    'max_depth': 4,
    'subsample': 0.9947,
    'min_child_weight': 2,
    'reg_alpha': 0.01111,
    'reg_lambda': 0.56370,
    'colsample_bytree': 0.75639,
    'scale_pos_weight': 2.5723,
    'random_state': 42,
    'eval_metric': 'logloss'
}
```

**Performance:**
- Train AUC: 0.9682, Train Brier: 0.0818
- Test AUC: 0.9521, Test Brier: 0.0731
- Overfitting gap: 0.016 (healthy)

**Feature importance (top 5):**
1. price_momentum: 25%
2. consistency_gap: 19%
3. days_to_resolution: 11%
4. days_active: 9%
5. sibling_volume_ratio: 7%

### Calibrated Model (CalibratedModel wrapper class)

```python
class CalibratedModel:
    """
    Manual Platt scaling — replaces CalibratedClassifierCV which removed
    prefit support in sklearn 1.8.
    
    Keeps XGBoost fixed and learns sigmoid correction on top.
    Must have classes_ attribute for MAPIE compatibility.
    """
    def __init__(self, base_model, platt_scaler):
        self.base_model = base_model
        self.platt = platt_scaler
        self.classes_ = base_model.classes_

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        calibrated_yes = self.platt.predict_proba(raw)[:, 1]
        calibrated_no = 1 - calibrated_yes
        return np.column_stack([calibrated_no, calibrated_yes])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
```

**Calibration improvement:**
- Brier before: 0.0731
- Brier after: 0.0692

### MAPIE Model (polymarket-mapie v1)

```python
from mapie.classification import SplitConformalClassifier

mapie_model = SplitConformalClassifier(
    estimator=calib_model,     # CalibratedModel wrapper
    confidence_level=0.9,
    conformity_score='lac',    # NOT 'aps' — restricted to multiclass in MAPIE 1.3
    prefit=True
)
mapie_model.conformalize(X_calib.to_numpy(), y_calib.to_numpy().ravel())
```

**Coverage verification:**
- Empirical coverage: 92.25% (target ≥ 90%) ✅
- Signal rate: 97.23% (97% of markets get confident signal)
- Ambiguity rate: 2.77% (only 2.77% get {YES,NO})

**Alpha exploration:**
| Alpha | Coverage | Signal Rate | Ambiguity |
|-------|----------|-------------|-----------|
| 0.05  | 95.33%   | 87.8%       | 12.2%     |
| 0.10  | 92.25%   | 97.2%       | 2.8%      |
| 0.20  | 83.94%   | 100%        | 0%        |

Default alpha: 0.10

---

## Signal Function

```python
def predict_signal(prediction_set, current_price: float) -> dict:
    """
    prediction_set: shape (2, 1) — [NO, YES] as True/False
    current_price: float 0-1 (current YES token price)
    
    Returns dict with keys:
    - signal: 'GREEN', 'RED', or 'YELLOW'
    - description: human-readable explanation
    - current_price: passed through
    - yes_in_set: bool
    - no_in_set: bool
    """
    yes_in_set = bool(prediction_set[1][0])
    no_in_set  = bool(prediction_set[0][0])

    if yes_in_set and not no_in_set:
        return {'signal': 'GREEN', 'description': 'Model confident YES', ...}
    elif no_in_set and not yes_in_set:
        return {'signal': 'RED', 'description': 'Model confident NO', ...}
    else:
        return {'signal': 'YELLOW', 'description': 'Model uncertain', ...}
```

---

## Backtest Results

### Overall Signal Accuracy

| Signal | Count | Accuracy |
|--------|-------|----------|
| GREEN  | 702   | 83.9%    |
| RED    | 2,107 | 94.7%    |
| YELLOW | 80    | N/A      |
| Overall| 2,809 | 92.0%    |

### Disagreement Analysis (Key Finding)

When the model's signal differs from market price direction:

| Price Range | Disagreements | Accuracy |
|-------------|---------------|----------|
| 0.0-0.1     | 6             | 16.7%    |
| 0.1-0.3     | 24            | 29.2%    |
| 0.3-0.5     | 39            | 43.6%    |
| 0.5-0.7     | 36            | **52.8%** |
| 0.7-0.9     | 22            | 40.9%    |
| 0.9-1.0     | 9             | 22.2%    |

**Finding:** Model has no reliable edge over market prices. 40.4% overall disagreement accuracy. Model is most useful as a filtering/scoring tool, not a directional prediction tool.

**Test set price distribution:**
- 60.4% of markets priced below 0.10 (near-certain NO)
- 76.7% priced below 0.30 or above 0.70 (near-resolved)
- Only 10% in genuinely uncertain 0.3-0.7 range

---

## Data Pipeline

### Timestamp Handling (Critical)

```python
# quant.parquet timestamps are Unix epoch integers (seconds)
# markets.parquet end_date is datetime[ms, UTC]
# Must convert quant timestamps:

quant = quant.with_columns(
    pl.from_epoch(pl.col('timestamp'), time_unit='s').alias('datetime')
).with_columns(
    pl.col('datetime').dt.convert_time_zone('UTC').dt.cast_time_unit('ms')
)
```

### Consistency Features — Key Implementation Details

```python
# neg_risk = 1 markets only — verified empirically:
# neg_risk=1 events have median price sum = 1.0 (mutually exclusive)
# neg_risk=0 events have wildly varying price sums (independent questions)

# Threshold filter: only compute for events where sibling prices sum 0.7-1.3
# Outside this range too many siblings are missing from the filtered dataset

# Formula:
consistency_gap = price_end - (1.0 / (n_siblings + 1))
# Positive = overpriced vs equal share
# Negative = underpriced vs equal share

# sibling_volume_ratio is LOG TRANSFORMED:
sibling_volume_ratio = log1p(sibling_volume_sum / this_market_volume)
```

---

## MLflow Registry

```python
# Load models:
xgb_model   = mlflow.xgboost.load_model("models:/polymarket-xgboost/5")
mapie_model  = mlflow.sklearn.load_model("models:/polymarket-mapie/latest")

# MLflow experiment: "polymarket-conformal"
# All runs nested under parent runs for clean UI

# Important: mapie model saved with mlflow.sklearn (pickle)
# Warning about pickle security is expected — not an error
```

---

## Running the Pipeline

```bash
# Full pipeline in order:
python src/features.py     # builds feature_matrix.parquet
python src/model.py        # trains XGBoost, runs Optuna, registers in MLflow
python src/conformal.py    # calibrates, conformalises, backtests, registers MAPIE
streamlit run src/app.py   # launches dashboard (Phase 1 todo)

# MLflow UI:
mlflow ui                  # available at http://localhost:5000
```

---

## Known Limitations

1. Near-resolution bias: 60% of test set priced below 0.10
2. No external information (news, polling, sentiment)
3. Political markets only — most efficient category on Polymarket
4. Short test window (Jan-Apr 2026) — same environment as training
5. Calibration and conformalization share same calibration set (minor leakage)
6. 40.4% disagreement accuracy — model confirms market efficiency not beats it
7. Consistency features: missing siblings for some neg_risk markets (price sum 0.7-1.3 filter)