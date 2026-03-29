# Phase 2 — Signal Validation

## Status: NOT STARTED (begins after Phase 1 main machine tasks done)
**Last updated: 27 March 2026**

```
Pre-phase work completed tonight:
✅ PolySharp brand established
✅ polysharp.substack.com live
✅ First Substack post published
✅ @PolySharpAI Twitter claimed and set up
✅ Polymarket Discord joined
✅ Brand assets created (PFP, Twitter header, Substack header)

Next action: Begin wallet_analysis.py on main machine
```

---

## Context For New Models

This phase happens after the portfolio piece is complete. The founder is building **PolySharp** — a prediction market intelligence platform. Before building any product, the wallet intelligence signal must be validated in live conditions via paper trading.

**Brand context:**
- Publication: PolySharp
- Substack: polysharp.substack.com
- Twitter: @PolySharpAI
- Domain reserved: polysharp.app
- Colour: #0D1117 background, #00D4AA accent
- Font: Space Grotesk

**Do not skip this phase. Do not build the product before completing this phase.**

---

## The Single Question This Phase Answers

**Do historically top-performing Polymarket wallets maintain their edge in live 2026 conditions?**

If yes — build the product. If no — the business model does not work as designed.

---

## Background — Why Wallet Intelligence

The existing ML model achieves 92% overall accuracy but only 40.4% when disagreeing with market prices. The market is more informed than the model. Wallet intelligence is different — it tracks people with demonstrated historical accuracy who likely have genuine information advantages.

---

## What To Build

### `src/wallet_analysis.py`

**Input files:**
- `data/processed/quant_political.parquet`
- `data/processed/markets_political.parquet`

**Output file:**
- `data/processed/wallet_scores.parquet`

**Scoring methodology:**

```python
"""
Filters (must pass all):
- Minimum 20 markets traded
- Minimum £500 total volume
- Active within last 12 months
- Exclude markets where wallet was only trader

Scoring dimensions:
1. Accuracy (40% weight)
   Volume-weighted % of markets bet correctly

2. Consistency (30% weight)
   Accuracy uniform across categories, not one lucky trade
   Formula: 1 - (std of accuracy across categories / mean accuracy)

3. Recency (20% weight)
   Last 6 months = 3x weight, 6-12 months = 2x, 12-24 months = 1x

4. Volume (10% weight)
   Log-scaled to prevent whale dominance

Output schema:
wallet_address | accuracy | consistency_score | recency_score |
volume_score | composite_score | n_markets | top_category |
last_active | total_volume | accuracy_by_cat (JSON)
"""
```

---

## Paper Trading Protocol

**Duration:** 90 days minimum. No real money.

**Daily process:**
```
1. Query The Graph for new trades from top 200 wallets
2. For each new position record:
   - Date/time, wallet address, wallet score
   - Market condition_id and question
   - Direction (YES/NO), entry price, detection lag
3. Filter to actionable signals:
   - YES signal: price < 0.40
   - NO signal: price > 0.60
   - Wallet composite score > 0.65
   - Market volume £10k-£500k
   - 14+ days until resolution
   - Price moved < 10% since wallet entry
4. On resolution: record outcome, theoretical P&L
```

**Weekly metrics to track:**
- Overall accuracy by wallet tier (top 10, 50, 100)
- Accuracy by price range
- Average detection lag (target: under 120 seconds)
- Theoretical P&L on £100/trade simulation

---

## Parallel Work During Phase 2

**Substack — already started ✅**

First post published: "Why most Polymarket traders lose — and what the data shows"

Upcoming posts (publish weekly):
```
Post 2: How we score wallets on Polymarket — full methodology
Post 3: What conformal prediction means for traders
Post 4: 90 days of paper trading — honest results (publish at Gate 1)
```

**Community building:**
```
Polymarket Discord: ✅ joined — participate daily, don't mention product yet
Twitter @PolySharpAI: ✅ set up — post prediction market analysis daily
Goal by end of Phase 2:
- 200+ Substack subscribers
- Known presence in Discord
- 300+ Twitter followers
```

**CLAUDE.md setup:**
Before starting wallet_analysis.py — create a `CLAUDE.md` in the repo root.
This gives any AI assistant immediate context without re-explaining everything.

```markdown
# PolySharp — Context for AI Assistants

## What this is
End-to-end conformal prediction pipeline for Polymarket political markets.
Foundation for PolySharp — a prediction market intelligence platform.

## What's built
- Complete ML pipeline (features.py, model.py, conformal.py)
- Live API client (api.py) — tested against Polymarket REST API
- Streamlit dashboard (app.py)

## Tech stack
Python 3.11, Polars, XGBoost, MAPIE 1.3, sklearn 1.8, MLflow, Streamlit

## Critical gotchas
- sklearn 1.8: CalibratedClassifierCV cv='prefit' REMOVED — use CalibratedModel wrapper
- MAPIE 1.3: MapieClassifier REMOVED — use SplitConformalClassifier
- MAPIE 1.3: APS restricted to multiclass — use LAC
- Prediction sets shape: (n_samples, 2, 1) — index [i][label][0]
- API: Slugs are EVENT slugs — query /events?slug= not /markets?slug=
- API: proxyWallet = maker and taker, size = usd_amount

## What we're building next
wallet_analysis.py — score 3 years of wallet performance
Then: 90 days paper trading to validate the signal

## Never modify
src/features.py EXCLUDE_COLS list — training/inference skew risk
data/model/ parquet files — fixed train/calib/test splits
MLflow registered models — use versioned references
```

---

## Gate 1 Decision Criteria

**Continue to Phase 3 if ALL true:**
```
✓ 50+ actionable signals in 90 days
✓ Top 20 wallets show 62%+ accuracy
✓ Average detection lag under 120 seconds
✓ Positive theoretical P&L after 2% fees
✓ At least two categories show consistent edge
✓ Wallets still actively trading in 2026
```

**Do not continue if ANY true:**
```
✗ Overall accuracy below 55%
✗ Negative P&L after fees
✗ Fewer than 20 actionable signals
✗ Most top wallets confirmed as insider traders being investigated
```

---

## Resources

```
The Graph Polymarket subgraph:
https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic

Polymarket Analytics (manual wallet validation):
https://polymarketanalytics.com/traders

Polymarket API:
https://docs.polymarket.com

Polygon explorer:
https://polygonscan.com
```