# Technical Foundation — Existing ML Pipeline

## Last updated: 30 March 2026
## Phase 1 Status: COMPLETE ✅

---

## Purpose

Complete technical reference for the ML pipeline. Pass to any AI assistant
before building new components. Read this before touching any existing code.

---

## Project Overview

End-to-end ML pipeline applying conformal prediction to Polymarket political
market data. Foundation for the **PolySharp** prediction market intelligence
platform (polysharp.substack.com).

**The honest finding:** 92% overall accuracy but only 40.4% on genuine market
disagreements. Model confirms market efficiency. Best used as a
filtering/prioritisation tool not a trading signal.

---

## Tech Stack

```
Language:              Python 3.11
Data processing:       Polars (NOT Pandas)
ML:                    XGBoost, scikit-learn 1.8
Conformal prediction:  MAPIE 1.3
Experiment tracking:   MLflow
Application:           Streamlit
API client:            requests
Data source:           HuggingFace SII-WANGZJ/Polymarket_data (MIT)
```

---

## Critical Version Notes — Read Before Writing Any Code

**sklearn 1.8:**
- CalibratedClassifierCV(cv='prefit') was REMOVED
- Solution: Manual Platt scaling via CalibratedModel wrapper class

**MAPIE 1.3:**
- MapieClassifier was REMOVED — use SplitConformalClassifier
- APS restricted to multiclass — use 'lac' for binary classification
- API: .conformalize() not .fit(), .predict_set() not .predict()
- Prediction sets shape: (n_samples, 2, 1) — index pred_set[label][0]

**Polymarket REST API:**
- Slugs are EVENT slugs — query /events?slug= not /markets?slug=
- Markets are nested inside events in the response
- Data API field names: proxyWallet (not maker/taker), size (not usdcSize)
- Data API 400s at offset 3500 — break pagination gracefully

---

## File Structure

```
/polymarket-conformal/ (GitHub: public)
├── CLAUDE.md              ← AI assistant context file ✅
├── README.md              ← Complete ✅
├── data/
│   ├── raw/               ← gitignored (29GB quant, 68MB markets)
│   ├── processed/         ← feature_matrix.parquet etc
│   └── model/             ← train/calibration/test splits
├── src/
│   ├── features.py        ← FeatureEngineer ✅
│   ├── model.py           ← Model class ✅
│   ├── conformal.py       ← Conformal class ✅
│   ├── api.py             ← Polymarket API client ✅
│   └── app.py             ← Streamlit dashboard ✅
├── docs/
│   ├── PROJECT_NOTES.md
│   ├── disagreement_by_price_range.png ✅
│   └── [other plot images]
├── notebooks/             ← outputs stripped
└── mlruns/                ← gitignored
```

---

## Feature Matrix Schema

```
Column                 | Type    | In model?
-----------------------|---------|----------
market_id              | str     | NO (EXCLUDE)
resolved_yes           | bool    | TARGET
end_date               | datetime| NO (EXCLUDE)
price_start            | f64     | NO (EXCLUDE — leakage)
price_end              | f64     | NO (EXCLUDE — leakage)
price_mean             | f64     | NO (EXCLUDE — leakage)
price_min              | f64     | NO (EXCLUDE — leakage)
price_max              | f64     | NO (EXCLUDE — leakage)
price_volatility       | f64     | YES
price_range            | f64     | YES
price_momentum         | f64     | YES (top feature 25%)
log_total_volume       | f64     | YES
log_trade_count        | f64     | YES
log_avg_trade_size     | f64     | YES
buy_ratio              | f64     | YES
log_market_volume      | f64     | YES
days_active            | f64     | YES
days_to_resolution     | f64     | YES
neg_risk               | u8      | YES
n_siblings             | u32     | YES
consistency_gap        | f64     | YES (2nd feature 19%)
sibling_volume_ratio   | f64     | YES
```

**EXCLUDE_COLS:**
```python
EXCLUDE_COLS = [
    'market_id', 'resolved_yes', 'end_date',
    'price_start', 'price_end', 'price_mean',
    'price_min', 'price_max'
]
```

---

## Model Details

### XGBoost (polymarket-xgboost v5)

```python
params = {
    'n_estimators': 404, 'learning_rate': 0.02300, 'max_depth': 4,
    'subsample': 0.9947, 'min_child_weight': 2, 'reg_alpha': 0.01111,
    'reg_lambda': 0.56370, 'colsample_bytree': 0.75639,
    'scale_pos_weight': 2.5723, 'random_state': 42,
}
# Train AUC: 0.9682 | Test AUC: 0.9521
# Train Brier: 0.0818 | Test Brier: 0.0731
# Overfitting gap: 0.016 (healthy)
```

### CalibratedModel Wrapper

```python
class CalibratedModel:
    def __init__(self, base_model, platt_scaler):
        self.base_model = base_model
        self.platt = platt_scaler
        self.classes_ = base_model.classes_  # Required for MAPIE

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        calibrated_yes = self.platt.predict_proba(raw)[:, 1]
        return np.column_stack([1 - calibrated_yes, calibrated_yes])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
```

### MAPIE (polymarket-mapie v1)

```python
from mapie.classification import SplitConformalClassifier

mapie = SplitConformalClassifier(
    estimator=calib_model,
    confidence_level=0.9,
    conformity_score='lac',  # NOT 'aps'
    prefit=True
)
mapie.conformalize(X_calib, y_calib)
_, pred_sets = mapie.predict_set(X_test)
# pred_sets[i][1][0] = YES in set
# pred_sets[i][0][0] = NO in set
```

**Coverage at alpha=0.10:** 92.25% ✅

---

## Signal Logic

```python
yes_in_set = bool(pred_set[1][0])
no_in_set  = bool(pred_set[0][0])

if yes_in_set and not no_in_set:   signal = 'GREEN'   # confident YES
elif no_in_set and not yes_in_set: signal = 'RED'     # confident NO
else:                              signal = 'YELLOW'  # uncertain
```

---

## Data Splits (never reshuffle)

```
Train:       8,665 markets  Dec 2022 - Nov 2025  YES 29.8%
Calibration: 2,889 markets  Nov 2025 - Jan 2026  YES 25.3%
Test:        2,889 markets  Jan 2026 - Apr 2026  YES 25.2%
```

---

## Backtest Results

```
GREEN:    702 signals  → 83.9% accurate
RED:    2,107 signals  → 94.7% accurate
Overall: 2,809 signals → 92.0% accurate
Coverage: 92.25% (target 90%) ✅

Disagreement: 136 signals → 40.4% accurate
→ Market more informed than model on near-resolved prices
```

---

## Timestamp Handling (Critical)

```python
df = df.with_columns(
    pl.from_epoch(pl.col('timestamp'), time_unit='s')
      .dt.convert_time_zone('UTC')
      .dt.cast_time_unit('ms')
      .alias('datetime')
)
```

---

## Consistency Features

```python
# Only for neg_risk=1, event price sum 0.7-1.3
consistency_gap = price_end - (1.0 / (n_siblings + 1))
sibling_volume_ratio = log1p(sibling_volume_sum / this_market_volume)
```

---

## MLflow Registry

```python
xgb   = mlflow.xgboost.load_model("models:/polymarket-xgboost/5")
mapie = mlflow.sklearn.load_model("models:/polymarket-mapie/latest")
# Experiment: "polymarket-conformal"
# MAPIE saved with mlflow.sklearn — pickle warning is expected
```

---

## Pipeline Order

```bash
python src/features.py    # ~10 min — builds feature_matrix.parquet
python src/model.py       # ~25 min — trains XGBoost, runs Optuna
python src/conformal.py   # ~2 min  — calibrates, conformalises, backtests
streamlit run src/app.py  # localhost:8501
mlflow ui                 # localhost:5000
```

---

## compute_features_for_market() — Live Inference

```python
fe = FeatureEngineer()
features = fe.compute_features_for_market(
    condition_id=market['condition_id'],  # str
    trades=trades,                         # pl.DataFrame from api.py
    market_meta=pl.DataFrame({            # single row
        'condition_id': [market['condition_id']],
        'volume':       [market['volume']],
        'event_id':     [market['event_id']],
        'neg_risk':     [market['neg_risk']],
        'end_date':     [market['end_date']],
    })
)
# Returns pl.DataFrame shape (1, 22)
# Consistency features default to 0.0 — Phase 3 adds live sibling queries
```

---

## Known Limitations

1. Near-resolution bias: 60% test set below 0.10
2. No external information (news, polling, sentiment)
3. Political markets only — most efficient category
4. Short test window Jan-Apr 2026
5. Calibration and conformalization share calibration set (minor leakage)
6. 40.4% disagreement accuracy — confirms efficiency, doesn't beat it
7. Live trades may show NO token prices on neg_risk markets

---

## What We Are Building Next

**Phase 2 — wallet_analysis.py**

Score 3 years of wallet performance from historical blockchain data.

```
Input:  data/processed/quant_political.parquet
        data/processed/markets_political.parquet
Output: data/processed/wallet_scores.parquet

Scoring:
1. Accuracy (40%)   — volume-weighted % of markets bet correctly
2. Consistency (30%) — accuracy uniform across categories
3. Recency (20%)    — last 6 months weighted 3x
4. Volume (10%)     — log-scaled total deployment

Filters:
- Minimum 20 markets traded
- Minimum £500 total volume
- Active within last 12 months
```

Then 90 days paper trading to validate before building product.
See PHASE_2_VALIDATION.md for full specification.

---

## Business Context

```
Brand:     PolySharp
Substack:  polysharp.substack.com ✅ live, first post published
Twitter:   @PolySharpAI ✅ live
Discord:   Polymarket community joined
Domain:    polysharp.app (buy in Phase 3)
Visual:    #0D1117 background, #00D4AA accent, Space Grotesk font
```