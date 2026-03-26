# Business Model and Strategy

## Context For New Models

This document covers the business model, competitive positioning, legal framework, and financial projections for the Polymarket Intelligence Platform. It should be read alongside the phase documents. This is a side project being built by a solo data scientist founder alongside full-time employment.

---

## What The Business Is

A data intelligence platform for serious Polymarket traders providing on-chain wallet tracking, ML market scoring, and smart money alerts.

**What it is:**
- Financial data intelligence and research platform
- Subscription SaaS product
- Built on publicly available blockchain data

**What it is not:**
- A financial adviser or investment service
- A gambling operator
- A trading bot or automated execution service
- A product for casual users who deposit under £200

---

## The Value Proposition

**One sentence:** The only prediction market intelligence platform that combines rigorous on-chain wallet scoring with machine learning signals and mathematically guaranteed confidence intervals.

**For the customer:** "We help you find the 5 markets worth your attention today, alert you when smart money moves, and flag arbitrage opportunities — so you can trade on your knowledge more effectively."

**Why people pay:** They are already deploying real money on Polymarket. A tool that saves them time and improves their research quality is worth £45/month if it helps them make better decisions even once per month.

---

## Customer Segments

### Primary — Serious Informed Retail Traders

The core paying customer. Not casual users. Not professional quants who build their own tools.

**Profile:**
- Deploys £500-5,000 per trade on Polymarket
- Trades 2-5 times per week
- Has genuine domain knowledge in specific areas (politics, economics, geopolitics)
- Makes decisions based on research and conviction
- Currently researches manually — checks news, watches market, follows Twitter
- No ability or desire to build their own data pipeline
- Active year-round not just during elections

**Size:** Approximately 2,000-5,000 people globally fitting this profile.

### Secondary — New Serious Traders

People who have recently joined Polymarket and are transitioning from casual to serious. Going through the discovery phase — lost money, started researching, now looking for tools.

**Why important:** This segment replenishes constantly. Every major news event brings new cohorts of serious traders into the ecosystem. The free tier catches them at discovery.

### Tertiary — Prediction Market Content Producers

Substack writers, political analysts, economists, journalists who use Polymarket data professionally. They want intelligence for their content not for trading.

**Lower willingness to pay** but valuable for distribution — they talk about tools they use.

### Not Your Customer

- Top quant traders and algorithmic firms (build their own)
- Casual users depositing under £200 total (won't pay)
- UK-based traders who cannot access Polymarket (geoblocked)

---

## Revenue Model

### Subscription Tiers

**Free — £0/month**
```
Purpose: Acquisition, discovery, conversion pipeline
What's included:
- Market scoring feed (24 hour delay)
- Top 10 wallet leaderboard (weekly, no addresses)
- Basic consistency gap scanner
- Weekly Substack newsletter
What's excluded: Real-time data, wallet addresses, alerts, model signals
```

**Pro — £45/month**
```
Purpose: Core revenue driver (~70% of MRR)
What's included:
- Real-time market scoring (5-minute updates)
- Full wallet leaderboard with category accuracy
- Telegram alerts within 60 seconds of top wallet activity
- Consistency gap alerts on neg_risk markets
- Portfolio tracker
- Position sizing calculator (Kelly criterion)
- Monthly honest performance report
- 3 years historical wallet performance data
- GREEN/RED/YELLOW model signal with confidence level
```

**Alpha — £150/month**
```
Purpose: High-value segment (~30% of MRR despite fewer subscribers)
What's included:
- Everything in Pro
- Sub 30-second alerts
- Wallet clustering analysis
- Custom alert thresholds
- Weekly 30-minute strategy call with founder
- API access to processed intelligence
- Priority support
```

**Annual plans (add at month 6):**
- Pro Annual: £450/year (2 months free vs monthly)
- Alpha Annual: £1,500/year (2 months free)
- Reduces churn, improves cash flow

### Additional Revenue Streams

**Affiliate revenue (add at month 4):**
Polymarket referral program pays 30% of direct referrals.
If subscribers sign up to Polymarket through your referral link — passive income.
Not primary revenue but covers infrastructure costs at scale.

**Data licensing (add at month 18+):**
Annual contracts with research firms, media organisations, hedge funds.
Historical wallet scores, custom market intelligence feeds, research reports.
Pricing: £5,000-25,000/year per client.
One client = 222 Pro subscribers in revenue equivalent.
Requires 12+ months track record before credible institutional conversations.

---

## Pricing Rationale

**Why £45 for Pro:**
- Above the psychological £40 threshold (signals premium quality)
- Below £50 which starts feeling expensive for a side tool
- £45 is 4.5% of a single £1,000 trade — trivially small if it helps once
- Tested extensively in SaaS — £45 converts better than £40 or £50

**Why £150 for Alpha:**
- 3.3x Pro pricing for genuinely differentiated features
- The weekly strategy call alone justifies the premium
- Direct access to founder is hard to price but highly valued
- Filters for traders with enough capital to justify it

**Why free tier:**
- Prediction market community is sceptical of black boxes
- Free tier proves product quality before asking for money
- Word of mouth from free users drives paid acquisition
- Delayed wallet leaderboard is genuinely useful and builds trust

---

## Unit Economics

```
Customer Acquisition Cost (CAC):
- Organic (Substack, Discord, Twitter): £10-30
- Creator partnership: £189 (35% of year 1 revenue)
- Target blended CAC: below £50

Lifetime Value (LTV):
- Monthly churn target: 5% → average lifetime 20 months
- Pro LTV: £45 × 20 = £900
- Alpha LTV: £150 × 20 = £3,000

LTV:CAC ratio:
- Pro organic: £900 / £30 = 30:1 (excellent)
- Pro creator: £900 / £189 = 4.8:1 (acceptable)
- Target: maintain above 10:1 blended

Payback period:
- Pro organic: less than 1 month
- Pro creator: 4-5 months
```

---

## Financial Projections

### Conservative Scenario

```
Month 6:   10 Pro + 3 Alpha    = £450 + £450  = £900 MRR
Month 12:  40 Pro + 10 Alpha   = £1,800 + £1,500 = £3,300 MRR
Month 18:  100 Pro + 25 Alpha  = £4,500 + £3,750 = £8,250 MRR
Month 24:  200 Pro + 50 Alpha  = £9,000 + £7,500 = £16,500 MRR
           + 1 data license    = £833/month
           Total:                £17,333 MRR
```

### Base Scenario (with creator partnership and grant funding)

```
Month 6:   20 Pro + 5 Alpha    = £900 + £750  = £1,650 MRR
Month 12:  75 Pro + 20 Alpha   = £3,375 + £3,000 = £6,375 MRR
Month 18:  200 Pro + 50 Alpha  = £9,000 + £7,500 = £16,500 MRR
Month 24:  400 Pro + 100 Alpha = £18,000 + £15,000 = £33,000 MRR
           + 3 data licenses   = £2,500/month
           Total:                £35,500 MRR
```

### Running Costs

```
Month 1-6 (bootstrapped):
VPS: £20
Domain/SSL: £5
Stripe fees (~1.4%): £13-23
Total: £38-48/month

Month 6-12 (with news API):
VPS: £50
The Graph API: £50
Stripe fees: £46-89
Total: £146-189/month

Month 12-24 (full feature set):
VPS and infrastructure: £150
APIs (The Graph, news, LLM): £400
Stripe fees: £243-497
Total: £793-1,047/month

Net MRR at month 24 (conservative): ~£16,300
Net MRR at month 24 (base): ~£34,500
```

---

## Competitive Positioning

### Existing Tools

```
polymarketanalytics.com:
  - Real-time trader tracking, P&L history
  - Referenced by WSJ and CoinDesk
  - Powered by Goldsky professional infrastructure
  - Shows raw data — does not process into intelligence

HashDive:
  - Advanced metrics, trader tracking, smart screening
  - Dashboard-focused

PolyAlertHub:
  - Whale tracking, Telegram alerts
  - Alerts when whale moves but no accuracy scoring

PredictFolio:
  - Free P&L analytics
  - No ML signal layer

polymarket-politics.vercel.app:
  - Whale tracking for political markets
  - Free Telegram + premium instant alerts
  - Closest competitor — but no accuracy scoring methodology
```

### Your Differentiation

```
What competitors do: Show you what happened (raw data dashboard)
What you do: Tell you what it means (processed intelligence + statistical guarantee)

Specific advantages:
1. Conformal prediction layer — requires ML expertise, nobody else has built it
2. Rigorous wallet scoring methodology — not just recent P&L, 3 years cross-market analysis
3. Category-specific wallet accuracy — "71% overall but 84% on geopolitical markets"
4. Published honest track record — including bad months, builds trust competitors can't fake
5. Smart money vs model divergence alerts — novel signal type
6. Wallet clustering via on-chain analysis — same entity multiple addresses identified
```

### Positioning Statement

*"Most Polymarket analytics tools show you what happened. We tell you what it means — with mathematically verifiable confidence."*

### Three-Word Brand Positioning

**Rigorous. Transparent. Actionable.**

- Rigorous: Conformal prediction, academic-style wallet scoring, empirical validation
- Transparent: Published methodology, honest track record including failures
- Actionable: Real-time alerts, specific market scoring, clear signals

---

## Open Methodology Commitment

**What to publish publicly (builds trust, drives organic discovery):**

1. Wallet scoring methodology in plain English (four dimensions)
2. Why conformal prediction provides mathematical guarantees
3. Monthly signal performance including bad periods
4. Which market categories show the most interesting activity

**What to never publish (competitive moat):**

1. Specific wallet addresses in top tier
2. Scoring formula weights and exact thresholds
3. Alert detection logic and timing optimisations
4. Feature engineering implementation details
5. Model code or trained weights

**The principle:** Share the intellectual framework. Keep the implementation.

---

## Legal and Compliance Framework

### What You Are (Legally)

A financial data intelligence and research platform. Not a financial adviser. Not a gambling operator. Not an FCA-regulated investment firm.

**The legal distinction that matters:**
Building analytics tools on public blockchain data ≠ operating a gambling service.
Bloomberg analyses stock market transactions without a gambling licence. Same principle applies.

### Immediate Legal Requirements

**Before taking first payment:**

1. **Terms of Service** — £300-500 via lawyer (do not use free templates)

Must include:
```
- "We provide data intelligence and research, not financial advice"
- "Subscribers make their own independent trading decisions"
- "Past signal performance does not guarantee future results"
- "We are not regulated by the Financial Conduct Authority"
- "We are not regulated by the UK Gambling Commission"
- Limitation of liability clause
- Data usage and storage (GDPR compliant)
- Cancellation and refund policy (14-day cooling off period for UK consumers)
```

2. **Privacy Policy** — £100-200 or GDPR generator (required under UK GDPR)

3. **Company formation** — Incorporate as UK Ltd via Companies House (£12)
   Do this when you get first paying subscriber. Separates personal liability.

### What You Do NOT Need

- FCA authorisation (not managing money or providing personal financial advice)
- UKGC licence (not operating a gambling service)
- MiFID II registration (providing research not investment services)

### The UK/Polymarket Question

Polymarket is geoblocked in the UK. This affects traders but not intelligence platform builders.

**Your position:**
- You can build analytics tools on public blockchain data from the UK
- You can sell those tools globally
- You should not market specifically to UK traders to help them circumvent geoblocking
- Market primarily to US and international traders where Polymarket is accessible

**Language rules:**
Never: "Use this to trade on Polymarket from the UK"
Always: "Intelligence for Polymarket traders" (global audience implied)

---

## The Open Methodology Content Calendar

These posts build distribution and trust simultaneously. Start during Phase 2.

```
Post 1 (Phase 2, week 1):
"Why most Polymarket traders lose — and what we're building"
Purpose: Set the problem. Establish credibility. No product pitch.

Post 2 (Phase 2, week 3):
"How we score wallets on Polymarket — full methodology"
Purpose: Show rigour. Build trust with technical audience.

Post 3 (Phase 2, week 5):
"What conformal prediction means for trading signals"
Purpose: Explain the mathematical guarantee in plain English.

Post 4 (Phase 2, month 3 — Gate 1 results):
"90 days of paper trading — honest results"
Purpose: This is the post that converts. Show everything including failures.

Post 5 (Phase 3, launch week):
"We're launching — here's what we built and why"
Purpose: Announce paid tiers to Substack audience.

Monthly thereafter:
"[Month] signal performance report"
Purpose: Build trust with ongoing honest track record.
```

---

## Key Risks

```
Risk 1: Wallet signal decays
Probability: Medium
Impact: Fatal to business model
Mitigation: Gate 1 paper trading validation. Monitor continuously post-launch.

Risk 2: Existing competitor copies methodology
Probability: Medium (if you publish it openly)
Impact: Reduces differentiation but doesn't eliminate trust moat
Mitigation: Published track record they cannot retroactively fake.
           They can copy the approach — not the history.

Risk 3: Polymarket access issues (bans, regulatory)
Probability: Low-Medium (UK already banned, more EU bans possible)
Impact: Reduces accessible customer base
Mitigation: Cover Kalshi alongside Polymarket from day one.
           Market to global audience, primarily US-based traders.

Risk 4: Solo founder burnout
Probability: High if not managed
Impact: Product goes unmaintained, subscribers churn
Mitigation: Clear decision gates allow clean exits.
           Keep infrastructure simple and low-maintenance.
           Set boundaries on support hours.

Risk 5: Can't acquire subscribers
Probability: Medium
Impact: Business doesn't grow beyond early adopters
Mitigation: Build Substack audience BEFORE product launch.
           Creator partnerships for distribution leverage.
           Do not launch without 200+ Substack subscribers.
```

---

## Decision Gates Summary

```
Gate 1 (Phase 2 — Month 3):
Question: Does the wallet signal work in live 2026 conditions?
Pass criteria: 60%+ accuracy, 50+ actionable signals, positive theoretical P&L
Decision: Yes → build MVP. No → portfolio piece, career track, park business.

Gate 2 (Phase 3 — Month 5):
Question: Will people pay for this?
Pass criteria: 5+ paying subscribers, positive retention after month 1
Decision: Yes → growth phase. No → reassess positioning.

Gate 3 (Phase 4 — Month 12):
Question: Is this worth serious commitment?
Pass criteria: 50+ subscribers, £2,000+ MRR, positive monthly growth
Decision: Yes → apply for pre-seed, consider going full-time.
          No → honest assessment, maintain as small side project or exit.
```