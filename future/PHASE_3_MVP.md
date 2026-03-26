# Phase 3 — MVP Build

## Context For New Models

This phase builds the minimum viable product. It starts only after Gate 1 (Phase 2 paper trading) validates the wallet intelligence signal. The founder has a complete ML pipeline from Phase 1 and validated wallet scores from Phase 2. This phase turns those into a product people pay for.

**Do not build any of this before Gate 1 passes.**

---

## MVP Philosophy

Build the minimum that delivers the core value proposition. Nothing extra. The temptation is to keep adding features — resist it. The goal of this phase is five paying subscribers, not a complete product.

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
├── Streamlit dashboard
├── Telegram bot for real-time alerts
├── FastAPI backend for data serving
└── Stripe for subscriptions

Infrastructure:
├── Hetzner or DigitalOcean VPS — £20/month
├── PostgreSQL — on same VPS
├── Cron jobs for wallet score refreshes
└── Simple uptime monitoring (UptimeRobot free tier)
```

---

## New Files To Build

### `src/blockchain.py` — The Graph Integration

**Purpose:** Monitor top wallet addresses in real time on Polygon blockchain.

```python
"""
Core functionality:

1. Query top wallet recent trades
   Uses The Graph GraphQL API
   Polymarket subgraph: https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic
   
2. Detect new positions from monitored wallets
   Poll every 30-60 seconds for each wallet in top 200
   
3. Store detected activity in PostgreSQL
   Alert when new position detected from top 50 wallets
   
4. Calculate detection lag
   Difference between blockchain timestamp and detection time
   Target: under 60 seconds
"""

GRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic"

# Core query pattern
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
    outcomeTokensMinted
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

# PostgreSQL schema for live activity
"""
CREATE TABLE wallet_activity (
  id SERIAL PRIMARY KEY,
  detected_at TIMESTAMP DEFAULT NOW(),
  blockchain_timestamp TIMESTAMP,
  wallet_address VARCHAR(42),
  wallet_composite_score FLOAT,
  condition_id VARCHAR(66),
  market_question TEXT,
  direction VARCHAR(3),  -- YES or NO
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
  mapie_signal VARCHAR(6),  -- GREEN, RED, YELLOW
  confidence_level FLOAT,
  current_price FLOAT
);
"""
```

### Extended `features.py` — Wallet Activity Features

**Add to the existing feature matrix computation:**

```python
# New features to add (compute after existing features):

WALLET_FEATURES = {
    'smart_money_active': bool,
    # Any top-50 wallet traded this market in last 30 days?
    
    'smart_money_direction': float,
    # 1.0 = smart money buying YES, -1.0 = buying NO, 0.0 = not active
    
    'smart_money_entry_price': float,
    # Average entry price of top wallet activity (0 if not active)
    
    'smart_money_volume': float,
    # Total USD from top wallets in this market (log1p transformed)
    
    'top_wallet_composite_score': float,
    # Composite score of most active top wallet (0 if not active)
    
    'n_top_wallets_active': int,
    # How many top-50 wallets have positions (0-50)
    
    'smart_money_model_agreement': float,
    # 1.0 = smart money and momentum model agree, -1.0 = disagree, 0.0 = no signal
}

# After adding these features:
# - Retrain XGBoost model (new model version v6)
# - Rerun conformal calibration
# - Verify coverage still >= 90%
# - Check feature importance — smart_money features should rank highly
```

### `src/monitor.py` — Real-time Alert System

```python
"""
Continuously running process that:
1. Polls blockchain.py for new top wallet activity every 30 seconds
2. For each new position, checks alert criteria
3. Sends Telegram alert if criteria met
4. Logs to PostgreSQL

Alert criteria:
- Wallet composite score > 0.65
- Market volume between £10k-£500k
- Entry price favourable (YES < 0.40 or NO > 0.60)
- Market has 14+ days until resolution
- Not already alerted on this market/wallet combination

Telegram alert format:
🔔 SMART MONEY ALERT

Market: Will [question]?
Direction: [YES/NO]
Entry price: [price]
Wallet accuracy: [score]%
Model signal: [GREEN/RED/YELLOW]
Consistency gap: [gap]

polymarket.com/market/[slug]
"""
```

### `src/telegram_bot.py` — Subscriber Alert Delivery

```python
"""
Telegram bot functionality:

1. Subscriber management
   - /start command — register new subscriber
   - Verify Stripe subscription status before adding to alert list
   - Store chat_id → subscription_tier mapping

2. Alert delivery
   - Free tier: no real-time alerts (weekly digest only)
   - Pro tier: alerts within 60 seconds
   - Alpha tier: alerts within 30 seconds (priority pipeline)

3. Commands
   - /status — show subscription tier and alert settings
   - /top — show top 10 wallets this week
   - /markets — show today's highest scored markets
   - /help — command list

Bot setup:
- Create via @BotFather on Telegram
- Store bot token in environment variable TELEGRAM_BOT_TOKEN
- Use python-telegram-bot library
"""
```

---

## Product Tiers

### Free Tier

```
Purpose: Acquisition, discovery, word of mouth
Cost: £0

Includes:
✓ Market scoring feed — updated daily, not real-time
✓ Top 10 wallet leaderboard — weekly update, no wallet addresses
✓ Basic consistency gap scanner — show events with gaps > 5%
✓ Weekly newsletter (Substack integration)

Excludes:
✗ Real-time anything
✗ Wallet addresses and full history
✗ Telegram alerts
✗ Model signal details
```

### Pro Tier — £45/month

```
Purpose: Core revenue driver
Target: Serious traders deploying £500+ per trade

Includes:
✓ Real-time market scoring (updated every 5 minutes)
✓ Full wallet leaderboard with category accuracy breakdown
✓ Telegram alerts within 60 seconds of top wallet activity
✓ Consistency gap alerts on neg_risk markets > 5% gap
✓ Portfolio tracker (track own positions)
✓ Position sizing calculator (Kelly criterion implementation)
✓ Monthly honest signal performance report
✓ 3 years of historical wallet performance data
✓ Model signal (GREEN/RED/YELLOW) with confidence level

Excludes:
✗ API access
✗ Custom alert thresholds
✗ Direct contact with founder
```

### Alpha Tier — £150/month

```
Purpose: High-value traders, maximum revenue per customer
Target: Traders deploying £5,000+ per trade

Includes:
✓ Everything in Pro
✓ Sub 30-second alerts (priority detection pipeline)
✓ Wallet clustering analysis (same entity multiple addresses)
✓ Custom alert configuration (set own price/score thresholds)
✓ Weekly 30-minute strategy call with founder
✓ API access — processed intelligence endpoints (see below)
✓ Input on product roadmap
✓ Priority support (response within 4 hours)
```

---

## Stripe Integration

```python
"""
Subscription setup:

Products:
- polymarket-pro: £45/month recurring
- polymarket-alpha: £150/month recurring
- polymarket-pro-annual: £450/year (2 months free)

Webhook events to handle:
- customer.subscription.created → add user to alert list
- customer.subscription.deleted → remove from alert list
- invoice.payment_failed → send reminder, downgrade access
- customer.subscription.updated → handle tier changes

User management:
- Store email + Stripe customer ID + subscription tier in PostgreSQL
- Link to Telegram chat_id when user uses /start command
- Check subscription status on every alert send (in case of failed payment)
"""
```

---

## MVP Launch Sequence

```
Week 1 — Infrastructure:
- Set up VPS (Hetzner CX21, £7/month — upgrade later)
- Install PostgreSQL, Python environment
- Deploy blockchain.py with monitoring
- Verify wallet detection is working
- Set up error alerting (email on failure)

Week 2 — Product:
- Build Streamlit dashboard V2 (extend Phase 1 app.py)
- Build Telegram bot
- Set up Stripe subscriptions
- Connect Telegram verification to Stripe
- Test end-to-end: new subscriber → Stripe → Telegram → alerts

Week 3 — Legal and soft launch:
- Publish Terms of Service (use lawyer — £300-500)
- Publish Privacy Policy (GDPR compliant)
- Incorporate as UK Ltd (Companies House — £12)
- Give free access to 5-10 people from Substack audience
- Gather honest feedback before charging

Week 4 — First paying subscribers:
- Announce paid tiers to Substack audience
- Offer first 20 subscribers 50% off first 3 months
- Document everything publicly — subscriber count, signal accuracy
- Target: 5 paying subscribers by end of week 4
```

---

## Gate 2 Decision Criteria

Evaluate 60 days after launch.

### Continue to Phase 4 if:

```
✓ 5+ paying subscribers
✓ At least 3 subscribers renew after first month (not churning immediately)
✓ Live signal accuracy tracking with wallet intelligence >= 60%
✓ Positive qualitative feedback — subscribers finding genuine value
✓ No critical infrastructure failures (alerts going down during active markets)
```

### Reassess if:

```
? 2-4 paying subscribers
  → Extend soft launch, improve free tier discovery, revisit pricing
  
? Subscribers join but churn quickly
  → Product not delivering value. Investigate before scaling.
  
? Infrastructure reliability issues
  → Fix before acquiring more subscribers
```

### Do not continue to Phase 4 if:

```
✗ No paying subscribers after 90 days from launch
  → Reassess product-market fit. Not a signal quality problem — a distribution or
  → positioning problem. Consider manual consulting approach instead.
```

---

## Key Technical Notes for New Models

**PostgreSQL setup:**
```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb polymarket_intelligence
sudo -u postgres createuser polymarket_app --pwprompt
```

**Environment variables required:**
```
TELEGRAM_BOT_TOKEN=
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
POSTGRES_URL=
POLYMARKET_API_KEY=  (if required)
THEGRAPH_API_KEY=    (for higher rate limits)
MLFLOW_TRACKING_URI= (local or remote)
```

**The Graph rate limits:**
Free tier: 1,000 queries/day
Paid tier needed for 200 wallets × 48 polls/day = 9,600 queries/day
Apply for The Graph API key: https://thegraph.com/studio/

**Monitoring top 200 wallets efficiently:**
Batch wallet queries rather than querying one at a time.
Store last_seen_trade_id per wallet to avoid re-processing old trades.
```