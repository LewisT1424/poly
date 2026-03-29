# Phase 1 — Portfolio Piece Completion

## Status: COMPLETE ✅
**Last updated: 30 March 2026**

```
All components complete and tested:
✅ src/features.py     — FeatureEngineer + compute_features_for_market()
✅ src/model.py        — XGBoost v5, AUC 0.9521, Brier 0.0731
✅ src/conformal.py    — MAPIE, 92.25% empirical coverage
✅ src/api.py          — Polymarket API client, tested live
✅ src/app.py          — Streamlit dashboard, both tabs working
✅ README.md           — Complete with honest limitations
✅ Repository          — Public on GitHub, clean commit history
✅ CLAUDE.md           — AI assistant context file in repo root
✅ Disagreement chart  — Added to Substack post
```

---

## Context For New Models

The portfolio piece is fully complete. Do not rebuild any of these components.
The next phase is Phase 2 — wallet_analysis.py and paper trading.

Platform: PolySharp (polysharp.substack.com)

---

## What Was Built

```
src/features.py     — FeatureEngineer class
  Key method: compute_features_for_market(condition_id, trades, market_meta)
  Returns single-row pl.DataFrame with all 22 columns
  Consistency features default to 0.0 (Phase 3 adds live sibling queries)

src/model.py        — XGBoost + Optuna 100-trial search + MLflow tracking
  Registered: polymarket-xgboost v5
  AUC: 0.9521, Brier: 0.0731, overfitting gap: 0.016

src/conformal.py    — Platt scaling + MAPIE SplitConformalClassifier
  Registered: polymarket-mapie v1
  Coverage: 92.25% empirical (target ≥ 90%) ✅
  Signal: {YES}→GREEN, {NO}→RED, {YES,NO}→YELLOW

src/api.py          — Live Polymarket REST API client
  get_market_by_slug(slug) → dict
  get_recent_trades(condition_id, days=30) → pl.DataFrame
  Tested against democratic-presidential-nominee-2028

src/app.py          — Streamlit dashboard
  Tab 1: Static backtest metrics + live calibration curve
  Tab 2: Slug input → API → features → MAPIE → signal
  Both tabs tested and working on main machine
```

---

## Key Technical Facts

- Python 3.11, Polars, XGBoost, MAPIE 1.3, sklearn 1.8, MLflow, Streamlit
- sklearn 1.8: cv='prefit' removed — manual CalibratedModel wrapper
- MAPIE 1.3: MapieClassifier removed — use SplitConformalClassifier with LAC
- Prediction sets shape: (n_samples, 2, 1) — index [i][label][0]
- API slugs are EVENT slugs — query /events?slug= not /markets?slug=
- API field names: proxyWallet (not maker/taker), size (not usdcSize)
- Data API 400s at offset 3500 — handled gracefully with break

## EXCLUDE_COLS
```python
['market_id', 'resolved_yes', 'end_date', 'price_start', 'price_end',
 'price_mean', 'price_min', 'price_max']
```

## 14 Features Used
price_volatility, price_range, price_momentum, log_total_volume,
log_trade_count, log_avg_trade_size, buy_ratio, log_market_volume,
days_active, days_to_resolution, neg_risk, n_siblings,
consistency_gap, sibling_volume_ratio

---

## Backtest Results

```
Signal    Count    Accuracy
GREEN     702      83.9%
RED       2,107    94.7%
Overall   2,809    92.0%
Coverage  92.25%   (target 90%) ✅

Disagreement accuracy: 40.4% (136 signals)
→ Market is more informed than model
→ Model best used as filtering/prioritisation tool
```

---

## Honest Limitations (published in README and Substack)

1. Near-resolution bias — 60% test set priced below 0.10
2. No external information (news, polling, sentiment)
3. Political markets only — most efficient category
4. Short test window Jan-Apr 2026 — same environment as training
5. Disagreement accuracy 40.4% — model confirms efficiency, doesn't beat it
6. Live trades may show NO token prices on neg_risk markets

---

## Portfolio Positioning

**The story:** Built end-to-end ML pipeline on 29GB of Polymarket blockchain
data, applied conformal prediction for statistically guaranteed signals,
discovered through honest backtesting that the market is efficient — which
is itself a meaningful finding.

**Target roles:**
- Data scientist at fintech/crypto companies
- Quantitative analyst at prediction market firms
- ML engineer at financial data companies
- Data scientist at Polymarket, Kalshi, or ecosystem companies

---

## Definition of Done — All Complete

- [x] api.py — fetches live data, tested against real slug
- [x] app.py — end-to-end test on main machine, both tabs working
- [x] README.md — complete with honest limitations
- [x] Repository — public on GitHub, clean commit history
- [x] compute_features_for_market() — added and tested
- [x] Disagreement bar chart — generated and added to Substack post
- [x] CLAUDE.md — created in repo root
- [x] All phase documents updated