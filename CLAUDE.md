# PolySharp — Context for AI Assistants

## What This Is
End-to-end conformal prediction pipeline for Polymarket political markets.
Foundation for PolySharp — a prediction market intelligence platform.
GitHub repository is public.

## What's Already Built (Phase 1 — COMPLETE)
- src/features.py — FeatureEngineer class with compute_features_for_market()
- src/model.py — XGBoost v5, AUC 0.9521, Brier 0.0731, Optuna tuning, MLflow
- src/conformal.py — Platt scaling + MAPIE, 92.25% empirical coverage
- src/api.py — Live Polymarket REST API client, tested and working
- src/app.py — Streamlit dashboard, both tabs working
- README.md — complete with honest limitations

## Tech Stack
Python 3.11, Polars (NOT Pandas), XGBoost, MAPIE 1.3, sklearn 1.8, MLflow, Streamlit

## Critical Gotchas — Read Before Writing Any Code
- sklearn 1.8: CalibratedClassifierCV cv='prefit' REMOVED — use CalibratedModel wrapper
- MAPIE 1.3: MapieClassifier REMOVED — use SplitConformalClassifier
- MAPIE 1.3: APS restricted to multiclass — use LAC conformity score
- Prediction sets shape: (n_samples, 2, 1) — index with [i][label][0]
- Polars: use pl.from_epoch() for timestamp conversion, NOT pandas methods
- API: slugs are EVENT slugs — query /events?slug= NOT /markets?slug=
- API: proxyWallet maps to both maker and taker columns
- API: size maps to usd_amount, conditionId maps to condition_id
- API: Data API 400s at offset 3500 — break pagination gracefully

## EXCLUDE_COLS (never pass to model)
['market_id', 'resolved_yes', 'end_date', 'price_start', 'price_end',
 'price_mean', 'price_min', 'price_max']

## 14 Features Used
price_volatility, price_range, price_momentum, log_total_volume,
log_trade_count, log_avg_trade_size, buy_ratio, log_market_volume,
days_active, days_to_resolution, neg_risk, n_siblings,
consistency_gap, sibling_volume_ratio

## MLflow Models
- polymarket-xgboost v5
- polymarket-mapie v1 (latest)
Load with:
  mlflow.xgboost.load_model("models:/polymarket-xgboost/5")
  mlflow.sklearn.load_model("models:/polymarket-mapie/latest")

## Data Splits (time-based, never reshuffle)
- train.parquet:        8,665 markets  Dec 2022 - Nov 2025
- calibration.parquet:  2,889 markets  Nov 2025 - Jan 2026
- test.parquet:         2,889 markets  Jan 2026 - Apr 2026

## Honest Model Limitations
- 92% overall accuracy but only 40.4% on genuine disagreements
- 60% of test set priced below 0.10 (near-resolved bias)
- Model confirms market efficiency — does not beat it
- Best used as filtering/prioritisation tool not trading signal

## What We Are Building Next (Phase 2)
wallet_analysis.py — score 3 years of wallet performance
Input:  data/processed/quant_political.parquet
        data/processed/markets_political.parquet
Output: data/processed/wallet_scores.parquet

Scoring dimensions (see PHASE_2_VALIDATION.md for full spec):
1. Accuracy (40%) — volume-weighted % of markets bet correctly
2. Consistency (30%) — accuracy uniform across categories
3. Recency (20%) — last 6 months weighted 3x
4. Volume (10%) — log-scaled total deployment

Then: 90 days paper trading to validate the signal before building product.

## Business Context
Platform name: PolySharp
Substack: polysharp.substack.com (live, first post published)
Twitter: @PolySharpAI
Target: Serious Polymarket traders who want rigorous intelligence
Differentiator: Conformal prediction guarantees + honest published track record

## Never Touch These
- EXCLUDE_COLS list in features.py — training/inference skew risk
- data/model/ parquet files — fixed splits
- MLflow registered model versions