# Phase 2 — Signal Validation

## Context For New Models

This phase happens after the portfolio piece is complete. The founder is building a prediction market intelligence platform. Before building any product, the wallet intelligence signal must be validated in live conditions. This phase is the most critical gate in the entire business plan — if the signal does not work, the business model does not work.

**Do not skip this phase. Do not build the product before completing this phase.**

---

## The Single Question This Phase Answers

**Do historically top-performing Polymarket wallets maintain their edge in live 2026 conditions?**

Everything else is secondary. The portfolio piece demonstrated the ML pipeline. This phase validates whether the business has a defensible signal.

---

## Background — Why Wallet Intelligence

The existing ML model (Phase 1) achieves 92% overall accuracy but only 40.4% accuracy when disagreeing with market prices. The market is more informed than the model on near-resolved markets. The model is a filtering tool, not a trading signal.

Wallet intelligence is different. It tracks people who have demonstrated genuine historical accuracy across many markets over long time periods. These are likely informed traders — domain experts, political insiders, people with information advantages. Following their positions is following someone who may know something the market hasn't yet priced.

**The edge source:** Information asymmetry, not model superiority.

---

## What To Build

### `src/wallet_analysis.py`

**Input files:**
- `data/processed/quant_political.parquet` — all trades with maker/taker wallet addresses
- `data/processed/markets_political.parquet` — market metadata with resolved_yes outcomes

**Output file:**
- `data/processed/wallet_scores.parquet`

**Scoring methodology:**

```python
"""
For each wallet with sufficient trading history:

FILTERS (must pass all to be scored):
- Minimum 20 markets traded
- Minimum £500 total volume deployed
- Must have traded at least one market after January 2025 (recency filter)
- Exclude markets where wallet was only trader (thin market / self-dealing)

SCORING DIMENSIONS:

1. Accuracy (40% weight)
   - % of markets where wallet bet on the winning side
   - Weight each market by USD amount deployed (larger bets count more)
   - Formula: sum(correct_bets * bet_size) / sum(all_bets * bet_size)

2. Consistency (30% weight)  
   - Accuracy should be consistent across market categories
   - Not just lucky on one big trade (2024 US election)
   - Formula: 1 - (std of accuracy across categories / mean accuracy)
   - Penalise wallets where accuracy is driven by single category

3. Recency (20% weight)
   - Recent activity weighted more heavily than old activity
   - Formula: weighted accuracy where trades in last 6 months = 3x weight,
             6-12 months = 2x weight, 12-24 months = 1x weight
   - A wallet that was great in 2023 but stopped trading scores low here

4. Volume (10% weight)
   - Meaningful capital behind conviction
   - Log-scaled to prevent one whale dominating
   - Formula: log(total_volume) / log(max_volume_in_dataset)

COMPOSITE SCORE:
composite = (accuracy * 0.40) + (consistency * 0.30) + 
            (recency * 0.20) + (volume * 0.10)

OUTPUT SCHEMA:
wallet_address     | str   — Polygon wallet address
accuracy           | f64   — volume-weighted accuracy
consistency_score  | f64   — cross-category consistency  
recency_score      | f64   — time-weighted recent accuracy
volume_score       | f64   — log-scaled volume metric
composite_score    | f64   — weighted composite
n_markets          | i32   — total markets traded
top_category       | str   — category with most trades
last_active        | date  — most recent trade date
total_volume       | f64   — total USD deployed
accuracy_by_cat    | str   — JSON dict of accuracy per category
"""
```

**Validation checks after building:**
```python
# Sanity checks to run
top_wallets = wallet_scores.sort('composite_score', descending=True).head(20)

# Check 1: Top wallets should not all be from same time period
assert top_wallets['last_active'].n_unique() > 10

# Check 2: Top wallet accuracy should be above 60%
assert top_wallets['accuracy'].mean() > 0.60

# Check 3: No wallet with fewer than 20 markets in top 100
assert wallet_scores.head(100)['n_markets'].min() >= 20

# Manual validation: look up top 5 wallet addresses on polymarketanalytics.com
# Do they match known high-performing traders? Cross-reference with leaderboard.
print(top_wallets.select(['wallet_address', 'accuracy', 'n_markets', 'composite_score']).head(20))
```

---

## Paper Trading Protocol

**Duration:** Minimum 90 days (3 months). No exceptions.

**No real money during this phase.**

### Daily Process

```
Each day:
1. Query The Graph for new trades from top 200 wallet addresses
   (see blockchain.py specification in Phase 3 for query structure)
   
2. For each new position detected, record immediately:
   - Date and time
   - Wallet address and composite score
   - Market condition_id and question
   - Direction (YES or NO)
   - Entry price at time of detection
   - How long after wallet's actual trade (detection lag)

3. Apply filters before recording as "actionable":
   - Entry price must be favourable:
     YES signal: price must be < 0.40 (payout asymmetry works in your favour)
     NO signal: price must be > 0.60
   - Wallet composite score must be > 0.65
   - Market volume must be between £10,000 and £500,000
   - Market must have at least 14 days until resolution
   - Price must not have moved more than 10% since wallet's entry

4. When market resolves, record:
   - Outcome (YES or NO)
   - Whether signal was correct
   - Theoretical P&L if traded (after 2% Polymarket fee)
```

### Tracking Spreadsheet / Database

```
columns:
date_detected | wallet_address | wallet_score | market_id | question |
direction | entry_price | detection_lag_seconds | market_volume |
days_to_resolution | category | outcome | correct | theoretical_pnl |
notes
```

### Metrics To Track Weekly

```python
# Overall accuracy
total_signals = len(paper_trades)
correct_signals = paper_trades['correct'].sum()
accuracy = correct_signals / total_signals

# By wallet tier
top10_accuracy = paper_trades[paper_trades['wallet_rank'] <= 10]['correct'].mean()
top50_accuracy = paper_trades[paper_trades['wallet_rank'] <= 50]['correct'].mean()

# By price range
for price_range in [(0.10, 0.30), (0.30, 0.50), (0.50, 0.70)]:
    subset = paper_trades[
        (paper_trades['entry_price'] >= price_range[0]) & 
        (paper_trades['entry_price'] < price_range[1])
    ]
    print(f"{price_range}: {len(subset)} signals, {subset['correct'].mean():.2%} accuracy")

# Detection speed
avg_detection_lag = paper_trades['detection_lag_seconds'].mean()
print(f"Average detection lag: {avg_detection_lag:.0f} seconds")
# Target: below 120 seconds

# Theoretical P&L
total_pnl = paper_trades['theoretical_pnl'].sum()
print(f"Theoretical P&L on £100 per trade: £{total_pnl:.2f}")
```

---

## Gate 1 Decision Criteria

Evaluate honestly at the end of 90 days.

### Continue to Phase 3 if ALL of these are true:

```
✓ At least 50 actionable signals generated (sufficient sample size)
✓ Top 20 wallets show 62%+ accuracy on actionable signals  
✓ Average detection lag under 120 seconds
  (price hasn't moved excessively before you can act)
✓ At least two market categories show consistent positive edge
✓ Theoretical P&L is positive after fees on £100-per-trade simulation
✓ No evidence of systematic wallet behaviour change
  (wallets still active, not just dormant from 2023-2024)
```

### Reassess if:

```
? Fewer than 30 actionable signals in 90 days
  → Market may be too quiet. Extend paper trading or broaden wallet criteria
  
? Accuracy 55-62% range
  → Marginal. Consider extending paper trading to 6 months for larger sample
  
? High accuracy but detection lag > 5 minutes
  → Signal works but execution is too slow. 
  → Invest in faster detection before deploying real capital
```

### Do not continue if ANY of these are true:

```
✗ Overall accuracy below 55% on actionable signals
  → No signal. Market is efficient. Business model does not work as designed.
  
✗ Theoretical P&L negative after fees
  → Even with high accuracy, entry prices are too unfavourable
  
✗ Fewer than 20 actionable signals in 90 days
  → Insufficient volume for the strategy to be viable
  
✗ Evidence that most top wallets are insider traders being investigated
  → Regulatory risk too high. Signal will disappear with enforcement.
```

### If Gate 1 fails:

This is not a failure of the project. The portfolio piece (Phase 1) remains complete and genuinely impressive. The ML pipeline, conformal prediction methodology, and blockchain data skills are valuable regardless of the business outcome.

Options if Gate 1 fails:
1. Take the portfolio piece and pursue career advancement
2. Pivot to cross-market consistency arbitrage detection (mechanical, no signal decay risk)
3. Extend paper trading with modified wallet scoring methodology
4. Build the product purely as a market scoring tool without wallet intelligence claims

---

## Parallel Work During Phase 2

While paper trading runs over 90 days, use the time productively:

### Start Substack Publication

Begin writing immediately. Don't wait for validation results. Publish weekly.

**First four posts:**

**Post 1 — "Why most Polymarket traders lose and what we're building to fix it"**
Sets the problem. Establishes credibility. No product pitch.

**Post 2 — "How we score wallets on Polymarket — our full methodology"**
Explain the four dimensions (accuracy, consistency, recency, volume) in plain English.
Show the rigour. Do not reveal specific wallet addresses.

**Post 3 — "What conformal prediction means for prediction market signals"**
Plain English explanation of the statistical guarantee.
Why prediction sets are more honest than point estimates.

**Post 4 — "90 days of paper trading — honest results"**
Publish at Gate 1. Show everything. Include bad signals. This is the post that converts.

### Community Building

```
Polymarket Discord:
- Join and participate daily
- Answer questions about prediction market mechanics
- Share analysis (not product promotion)
- Build reputation over 2-3 months before any soft launch

Twitter/X:
- Post daily prediction market analysis
- Notable market movements and what they mean
- Methodology snippets
- Engage with @Polymarket and prediction market community

Goal by end of Phase 2:
- 200+ Substack subscribers
- Known presence in Polymarket Discord
- 300+ Twitter followers interested in prediction markets
```

---

## Resources

**The Graph — Polymarket subgraph:**
https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic

**Polymarket Analytics — for manual wallet validation:**
https://polymarketanalytics.com/traders

**Polymarket API documentation:**
https://docs.polymarket.com

**Polygon blockchain explorer:**
https://polygonscan.com