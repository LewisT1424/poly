# Phase 3 — MVP Build

## Status: NOT STARTED (begins after Gate 1 passes)
**Last updated: 27 March 2026**

---

## Context For New Models

This phase builds the minimum viable product. Only begins after Gate 1 (Phase 2 paper trading) validates the wallet signal. The founder has a complete ML pipeline from Phase 1 and validated wallet scores from Phase 2.

**Brand:** PolySharp
**Substack:** polysharp.substack.com (live, building audience during Phase 2)
**Twitter:** @PolySharpAI
**Domain:** polysharp.app ($12.98/yr when ready to buy)

**Do not build any of this before Gate 1 passes.**

---

## MVP Philosophy

Build the minimum that delivers the core value. Goal: five paying subscribers. Not a complete product.

**Core value:** Alert serious traders within 60 seconds when top-performing wallets open new positions, with model signal context.

---

## Technical Stack

```
Data layer:
├── quant_political.parquet (historical, already exists)
├── wallet_scores.parquet (from Phase 2)
├── PostgreSQL — live wallet activity and market scores
└── The Graph — live Polygon blockchain data

Intelligence layer:
├── wallet_analysis.py (Phase 2, extend with live scoring)
├── features.py (extend with wallet activity features)
├── model.py (retrain with new features)
└── conformal.py (existing pipeline, unchanged)

Product layer:
├── Streamlit dashboard V2
├── Telegram bot for real-time alerts
├── FastAPI backend for data serving
└── Stripe for subscriptions

Infrastructure:
├── Hetzner or DigitalOcean VPS — ~£20/month
├── PostgreSQL on same VPS
└── UptimeRobot for monitoring (free tier)
```

---

## New Files To Build

### `src/blockchain.py` — The Graph Integration

```python
GRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic"

WALLET_TRADES_QUERY = """
query GetWalletTrades($wallet: String!, $since: Int!) {
  fpmmTrades(
    where: {
      creator: $wallet,
      creationTimestamp_gt: $since
    }
    orderBy: creationTimestamp
    orderDirection: desc
    first: 50
  ) {
    id
    outcomeIndex
    collateralAmount
    creationTimestamp
    fpmm {
      id
      question { title }
      outcomes
      outcomeTokenPrices
      collateralVolume
    }
  }
}
"""

# Poll top 200 wallets every 30-60 seconds
# Alert when new position detected from top 50 wallets
# Target detection lag: under 60 seconds
```

**PostgreSQL schema:**
```sql
CREATE TABLE wallet_activity (
  id SERIAL PRIMARY KEY,
  detected_at TIMESTAMP DEFAULT NOW(),
  blockchain_timestamp TIMESTAMP,
  wallet_address VARCHAR(42),
  wallet_composite_score FLOAT,
  condition_id VARCHAR(66),
  market_question TEXT,
  direction VARCHAR(3),
  entry_price FLOAT,
  usd_amount FLOAT,
  detection_lag_seconds INT,
  alerted BOOLEAN DEFAULT FALSE
);

CREATE TABLE market_scores (
  condition_id VARCHAR(66) PRIMARY KEY,
  scored_at TIMESTAMP DEFAULT NOW(),
  momentum_signal FLOAT,
  consistency_gap FLOAT,
  smart_money_active BOOLEAN,
  smart_money_direction VARCHAR(3),
  smart_money_score FLOAT,
  mapie_signal VARCHAR(6),
  confidence_level FLOAT,
  current_price FLOAT
);
```

### Extended wallet features for `features.py`

```python
WALLET_FEATURES = {
    'smart_money_active':          bool,
    'smart_money_direction':       float,  # 1.0=YES, -1.0=NO, 0.0=inactive
    'smart_money_entry_price':     float,
    'smart_money_volume':          float,  # log1p transformed
    'top_wallet_composite_score':  float,
    'n_top_wallets_active':        int,
    'smart_money_model_agreement': float,  # 1.0=agree, -1.0=disagree
}
```

---

## Product Tiers

### Free — £0
```
✓ Market scoring feed (24 hour delay)
✓ Top 10 wallet leaderboard (weekly, no addresses)
✓ Basic consistency gap scanner
✓ Weekly Substack newsletter
✗ Real-time data
✗ Telegram alerts
✗ Wallet addresses
```

### Pro — £45/month
```
✓ Real-time market scoring (5-minute updates)
✓ Full wallet leaderboard with category accuracy
✓ Telegram alerts within 60 seconds
✓ Consistency gap alerts on neg_risk markets
✓ Portfolio tracker
✓ Position sizing calculator (Kelly criterion)
✓ Monthly honest performance report
✓ 3 years historical wallet performance data
✓ GREEN/RED/YELLOW model signal with confidence
```

### Alpha — £150/month
```
✓ Everything in Pro
✓ Sub 30-second alerts
✓ Wallet clustering analysis
✓ Custom alert thresholds
✓ Weekly 30-min strategy call with founder
✓ API access to processed intelligence
✓ Priority support
```

---

## Legal Requirements (before first payment)

```
Terms of Service: £300-500 via lawyer
  Must state: research not financial advice,
  not FCA regulated, not UKGC regulated

Privacy Policy: £100-200 (GDPR compliant)

UK Ltd incorporation: £12 via Companies House
  Do this when first paying subscriber acquired

Domain polysharp.app: $12.98/yr when ready
Business email: hello@polysharp.app via Google Workspace £6/month
```

---

## MVP Launch Sequence

```
Week 1 — Infrastructure:
- VPS setup (Hetzner CX21)
- PostgreSQL installation
- blockchain.py deployed and monitoring
- Error alerting configured

Week 2 — Product:
- Streamlit dashboard V2
- Telegram bot built
- Stripe subscriptions configured
- End-to-end test: subscriber → Stripe → Telegram → alerts

Week 3 — Legal and soft launch:
- Terms of Service published
- Privacy Policy published
- UK Ltd incorporated
- Free access to 5-10 Substack subscribers for feedback

Week 4 — First paying subscribers:
- Announce paid tiers to Substack audience
- Offer first 20 subscribers 50% off first 3 months
- Target: 5 paying subscribers by end of week 4
```

---

## Stripe Setup

```python
"""
Products:
- polysharp-pro: £45/month
- polysharp-alpha: £150/month
- polysharp-pro-annual: £450/year

Webhook events:
- customer.subscription.created → add to alert list
- customer.subscription.deleted → remove from alert list
- invoice.payment_failed → send reminder, downgrade access
"""
```

---

## Environment Variables Required

```
TELEGRAM_BOT_TOKEN=
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
POSTGRES_URL=
THEGRAPH_API_KEY=
MLFLOW_TRACKING_URI=
```

---

## Gate 2 Decision Criteria

**Continue to Phase 4 if:**
```
✓ 5+ paying subscribers
✓ 3+ subscribers renew after month 1
✓ Live signal accuracy ≥ 60%
✓ No critical infrastructure failures
```

**Reassess if:**
```
? 2-4 paying subscribers → extend soft launch
? Subscribers churn quickly → product not delivering value
```

**Do not continue if:**
```
✗ No paying subscribers after 90 days
```