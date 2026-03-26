# Polymarket Conformal Prediction Pipeline

An end-to-end machine learning pipeline that applies conformal prediction to Polymarket political markets. Takes raw on-chain trade data and produces statistically guaranteed prediction sets — not point estimates — with an empirically verified 90%+ coverage guarantee.

---

## What It Does

Most prediction models give you a probability. This pipeline gives you a prediction set: `{YES}`, `{NO}`, or `{YES, NO}` — with a mathematical guarantee that the true outcome is inside the set at least 90% of the time, verified empirically on held-out data.

The model learns from market dynamics — momentum, volume, sentiment, and cross-market consistency — without seeing current price levels. This forces genuine pattern learning rather than reading a near-resolved price.

---

## Results

Tested on 2,889 markets from January–April 2026 (unseen during training):

| Metric | Value |
|---|---|
| Test AUC-ROC | 0.9521 |
| Test Brier Score | 0.0731 |
| Empirical coverage | 92.25% (target ≥ 90%) ✅ |
| GREEN signal accuracy | 83.9% (702 signals) |
| RED signal accuracy | 94.7% (2,107 signals) |
| Overall signal accuracy | 92.0% |

**Honest finding:** When the model disagrees with the current market price — the more meaningful test for genuine mispricing — disagreement accuracy drops to 40.4%. The market is more informed than the model on near-resolved markets. The pipeline is best used as a market filtering and prioritisation tool, not a directional trading signal.

---

## Architecture
```
quant.parquet (29GB)          markets.parquet (68MB)
         │                              │
         └──────────┬───────────────────┘
                    ▼
            features.py
      30-day lookback window
      14 features including
      cross-market consistency
                    │
                    ▼
             model.py
         XGBoost + Optuna
         MLflow tracking
                    │
                    ▼
           conformal.py
         Platt calibration
      MAPIE SplitConformalClassifier
      Verified 92.25% coverage
                    │
                    ▼
         {YES} / {NO} / {YES,NO}
```

---

## Features

14 features used for training. Price level features (price_end, price_mean, etc.) were deliberately excluded — they correlated 0.62–0.84 with the target because most markets are measured close to resolution, giving the model an artificial edge. Removing them forces genuine learning from dynamics.

**Price dynamics:** `price_momentum`, `price_volatility`, `price_range`

**Volume:** `log_total_volume`, `log_trade_count`, `log_avg_trade_size`

**Sentiment:** `buy_ratio`

**Metadata:** `log_market_volume`, `days_active`, `days_to_resolution`

**Cross-market consistency** (neg_risk=1 markets only):
`consistency_gap`, `n_siblings`, `sibling_volume_ratio`, `neg_risk`

The consistency features detect mathematically mispriced markets within the same event. For mutually exclusive outcome markets, all outcome prices should sum to 1.0. When they don't, `consistency_gap` measures how far a market deviates from its equal-share baseline.

---

## What Is Conformal Prediction

Standard ML models produce a probability — "73% chance of YES." Conformal prediction wraps any model and produces a guaranteed prediction set instead.

For a chosen error rate α, the prediction set contains the true outcome with probability at least 1-α on new data, with no distributional assumptions. The guarantee is:
```
P(true outcome ∈ prediction set) ≥ 1 - α
```

This pipeline uses split conformal prediction:
1. XGBoost is trained on the training set
2. Conformity scores are computed on a held-out calibration set
3. The α-quantile of those scores becomes the threshold
4. On new markets: outcomes whose score is below the threshold enter the prediction set

The coverage is verified empirically. At α=0.10 (90% confidence), the pipeline achieves 92.25% empirical coverage on the test set.

---

## Signals

| Prediction Set | Signal | Interpretation |
|---|---|---|
| `{YES}` | 🟢 GREEN | Model confident YES |
| `{NO}` | 🔴 RED | Model confident NO |
| `{YES, NO}` | 🟡 YELLOW | Model uncertain — no signal |

Adjusting α changes the tradeoff between signal rate and coverage strength:

| Alpha | Coverage | Signal Rate |
|---|---|---|
| 0.05 | 95.3% | 87.8% |
| 0.10 | 92.3% | 97.2% |
| 0.20 | 83.9% | 100% |

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Data processing | Polars |
| Model | XGBoost |
| Hyperparameter search | Optuna |
| Conformal prediction | MAPIE 1.3 |
| Experiment tracking | MLflow |
| Dashboard | Streamlit |
| Data source | HuggingFace `SII-WANGZJ/Polymarket_data` |

---

## Installation
```bash
git clone https://github.com/your-username/polymarket-conformal
cd polymarket-conformal
conda create -n poly python=3.11
conda activate poly
pip install -r requirements.txt
```

Download the dataset from HuggingFace and place in `data/raw/`:
```
data/raw/quant.parquet      # 29GB
data/raw/markets.parquet    # 68MB
```

---

## Running the Pipeline

Run in order. Each step depends on the previous.

**1. Feature engineering**
```bash
python src/features.py
```
Builds `data/processed/feature_matrix.parquet` from raw trade data.
Runtime: ~10 minutes on the full dataset.

**2. Model training**
```bash
python src/model.py
```
Trains baseline XGBoost, runs 100-trial Optuna search, registers final model in MLflow.
Runtime: ~25 minutes.

**3. Conformal prediction**
```bash
python src/conformal.py
```
Applies Platt scaling, conformalises with MAPIE, runs backtest, registers MAPIE model.
Runtime: ~2 minutes.

**4. Dashboard**
```bash
streamlit run src/app.py
```
Opens at `http://localhost:8501`. Requires MLflow models to be registered.

**View experiments:**
```bash
mlflow ui
```
Opens at `http://localhost:5000`.

---

## Data Splits

Time-based split by `end_date` to simulate real-world future performance. Random splits would mix time periods and overstate generalisation.

| Split | Markets | Date Range | YES Rate |
|---|---|---|---|
| Train | 8,665 | Dec 2022 → Nov 2025 | 29.8% |
| Calibration | 2,889 | Nov 2025 → Jan 2026 | 25.3% |
| Test | 2,889 | Jan 2026 → Apr 2026 | 25.2% |

The calibration set is reserved exclusively for MAPIE conformalisation. It is never used for training. If calibration data leaked into training the coverage guarantee would break silently.

---

## Known Limitations

**Near-resolution bias**
60% of the test set has a market price below 0.10 — effectively already resolved. The model's high overall accuracy partly reflects this. On genuinely uncertain markets (price 0.3–0.7) only 293 test markets exist.

**No external information**
The model sees only trade history. News events, polling data, and social sentiment are not incorporated. A market that hasn't reacted to recent news is invisible to the model.

**Political markets only**
Political markets are among the most efficient on Polymarket. Sophisticated participants price information quickly. The model has a harder task here than it would on less liquid markets.

**Short test window**
The test set covers January–April 2026 — the same general market environment as training. Performance on a materially different environment (different election cycle, global shock) is unknown.

**Calibration set reuse**
Probability calibration (Platt scaling) and MAPIE conformalisation both use the same calibration set. This introduces minor leakage. A cross-calibration approach would eliminate it at the cost of implementation complexity. Empirical coverage of 92.25% confirms the approach is working acceptably.

---

## Implementation Notes

**sklearn 1.8 compatibility**
`CalibratedClassifierCV(cv='prefit')` was removed in sklearn 1.8. A manual `CalibratedModel` wrapper class implements Platt scaling directly via `LogisticRegression` fitted on raw XGBoost probabilities.

**MAPIE 1.3 compatibility**
`MapieClassifier` was removed in MAPIE 1.3. The pipeline uses `SplitConformalClassifier` with `conformity_score='lac'`. APS is restricted to multiclass classification in this version.

**Prediction set indexing**
`predict_set()` returns shape `(n_samples, 2, 1)`. Index with `pred_set[label][0]` not `pred_set[label]`.

---

## Future Work

- **Wallet intelligence layer** — on-chain tracking of historically accurate traders via The Graph. The most promising source of genuine information advantage.
- **News sentiment** — LLM-powered matching of breaking news to open markets. The 30-second to 5-minute information lag window on niche markets is where genuine edge exists.
- **Cross-market consistency arbitrage** — systematic detection of neg_risk events where sibling prices are mathematically inconsistent.
- **Expansion beyond political markets** — geopolitical and economic data markets where information efficiency is lower.

---

## Licence

MIT