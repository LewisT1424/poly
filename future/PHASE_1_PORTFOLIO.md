# Phase 1 — Portfolio Piece Completion

## Context For New Models

The founder has built a complete ML pipeline applying conformal prediction to Polymarket political market data. This phase covers the remaining components needed to complete the portfolio piece. The portfolio piece is the foundation for both career advancement and the future platform business.

---

## What Has Already Been Built

```
src/features.py     ← FeatureEngineer class — complete
src/model.py        ← XGBoost + Optuna + MLflow — complete  
src/conformal.py    ← Platt scaling + MAPIE + backtest — complete
notebooks/          ← model-analysis.ipynb — complete
mlruns/             ← all experiments tracked
```

**Registered MLflow models:**
- `polymarket-xgboost v5` — XGBoost classifier, AUC 0.9521, Brier 0.0731
- `polymarket-mapie v1` — SplitConformalClassifier, 92.25% empirical coverage

**Key technical facts for new models to know:**
- Python 3.11, Polars, XGBoost, MAPIE 1.3, sklearn 1.8, MLflow, Streamlit
- sklearn 1.8 removed `cv='prefit'` from CalibratedClassifierCV — manual Platt scaling implemented instead
- MAPIE 1.3 uses `SplitConformalClassifier` not `MapieClassifier` (removed)
- LAC conformity score used — APS restricted to multiclass in MAPIE 1.3
- `CalibratedModel` wrapper class has `classes_` attribute passed from XGBoost for MAPIE compatibility
- Prediction sets shape: `(n_samples, 2, 1)` — index with `[i][label][0]`
- EXCLUDE_COLS: `['market_id', 'resolved_yes', 'end_date', 'price_start', 'price_end', 'price_mean', 'price_min', 'price_max']`
- 14 features used: price_volatility, price_range, price_momentum, log_total_volume, log_trade_count, log_avg_trade_size, buy_ratio, log_market_volume, days_active, days_to_resolution, neg_risk, n_siblings, consistency_gap, sibling_volume_ratio

---

## Remaining Components

### 1 — `src/api.py` — Live Polymarket API Client

**Purpose:** Fetch live market data at inference time. Must produce features identical to training pipeline.

**Two functions required:**

```python
def get_market_by_slug(slug: str) -> dict:
    """
    Fetch market metadata and current price from Polymarket API.
    
    Returns dict with keys matching markets_political.parquet schema:
    - condition_id, question, slug, volume, end_date, neg_risk, event_id
    - current_price (YES token price, 0-1)
    
    Polymarket REST API base: https://gamma-api.polymarket.com
    Market endpoint: GET /markets?slug={slug}
    
    Handle errors: missing slug, API timeout, market not found,
    market already resolved (raise informative exceptions)
    """

def get_recent_trades(condition_id: str, days: int = 30) -> pl.DataFrame:
    """
    Fetch recent trades for a market.
    
    Returns Polars DataFrame matching quant_political.parquet schema:
    columns: timestamp, condition_id, price, usd_amount, side, maker, taker
    
    Polymarket API: GET /activity?market={condition_id}&limit=1000
    
    CRITICAL: Must convert timestamps to same format as training data
    pl.from_epoch('s').dt.convert_time_zone('UTC').dt.cast_time_unit('ms')
    
    Filter to last {days} days before current time
    Minimum 10 trades required — raise exception if fewer
    """
```

**Training/inference skew prevention:**
After fetching live data, pass through `FeatureEngineer` class from `features.py` using identical logic to training. Test this by fetching a market that exists in the training data and comparing features.

**Polymarket API documentation:** https://docs.polymarket.com

---

### 2 — `src/app.py` — Streamlit Application

**Purpose:** Demonstrate the full pipeline end-to-end with a live interface.

**Section 1 — Model Performance Dashboard**

```python
# Load from MLflow
xgb_model = mlflow.xgboost.load_model("models:/polymarket-xgboost/5")
mapie_model = mlflow.sklearn.load_model("models:/polymarket-mapie/latest")

# Display:
# - Calibration curves (before and after Platt scaling) side by side
# - Coverage verification: empirical vs theoretical at alpha 0.05, 0.10, 0.20
# - Feature importance bar chart (14 features)
# - Backtest results table:
#     GREEN signals: 702 → 83.9% accurate
#     RED signals: 2107 → 94.7% accurate  
#     Overall: 92.0%
#     Disagreement accuracy: 40.4% (honest finding — market is more informed)
# - MLflow run comparison table (model versions and metrics)
```

**Section 2 — Live Market Analyser**

```python
# Input: Polymarket market slug (e.g. "will-trump-win-2024-election")
# Process:
#   1. api.get_market_by_slug(slug) → market metadata
#   2. api.get_recent_trades(condition_id) → trade data
#   3. FeatureEngineer.compute_features() → feature vector
#   4. mapie_model.predict_set(features) → prediction set
#   5. conformal.predict_signal(prediction_set, current_price) → signal

# Display:
# - Current market price
# - Prediction set: {YES}, {NO}, or {YES, NO}
# - Signal: 🟢 GREEN (confident YES) / 🔴 RED (confident NO) / 🟡 YELLOW (uncertain)
# - Feature values (transparency)
# - Alpha slider (0.05 - 0.20) that reruns MAPIE at selected confidence level
# - Honest disclaimer: "This is research not financial advice"
```

**Important:** Load models once at startup using `@st.cache_resource` not on every interaction.

---

### 3 — `README.md`

**Structure:**

```markdown
# Polymarket Political Market Mispricing Detector

## What This Does (non-technical paragraph first)
## Architecture Overview (diagram or description)
## What Is Conformal Prediction (plain English, 3-4 paragraphs)
## How To Interpret Signals (GREEN/RED/YELLOW with examples)
## Installation and Setup
## Running The Pipeline (order: features.py → model.py → conformal.py → app.py)
## Honest Limitations
## Future Work
## Tech Stack
```

**Honest limitations section must include:**
- Model trained on near-resolution markets — 60% of test set priced below 0.10
- No external information (news, polling, social sentiment)
- Backtest disagreement accuracy 40.4% — market currently more efficient than model on genuine mispricings
- Short test window (Jan-Apr 2026) — same general environment as training
- Political markets only — most efficient category on Polymarket
- Calibration and conformalization share same calibration set (minor leakage)

**Future work section:**
- Wallet intelligence layer (Phase 2)
- News sentiment integration (Phase 4)
- Cross-platform Kalshi coverage
- Expansion beyond political markets

---

## Definition of Done

- [ ] `api.py` fetches live market data and produces valid feature vectors
- [ ] `api.py` tested against known training markets with feature comparison
- [ ] `app.py` runs without errors on a live market slug
- [ ] `app.py` displays all dashboard sections correctly
- [ ] `README.md` explains the project clearly to a non-technical reader
- [ ] Repository pushed to GitHub with clean commit history
- [ ] All scripts runnable in sequence: `features.py → model.py → conformal.py → app.py`

---

## Portfolio Positioning

When presenting this project:

**The story:** "I built an end-to-end ML pipeline on 29GB of Polymarket blockchain data, applied conformal prediction to generate statistically guaranteed trading signals, and discovered through honest backtesting that the market is efficient on these signals — which is itself a meaningful finding about prediction market efficiency."

**Key talking points:**
1. End-to-end pipeline from raw blockchain data to live inference
2. Multiple model iterations with honest problem identification (price leakage)
3. Novel application of conformal prediction to prediction markets
4. Honest backtest including uncomfortable findings — not just good numbers
5. Cross-market consistency features derived from blockchain event structure
6. Manual Platt scaling implementation due to sklearn 1.8 API change

**Target roles for this portfolio piece:**
- Data scientist at fintech/crypto companies
- Quantitative analyst at prediction market firms
- ML engineer at financial data companies
- Data scientist at Polymarket, Kalshi, or ecosystem companies