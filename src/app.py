'''
Streamlit dashboard for the Polymarket Conformal Prediction pipeline

Two section:
1. Model performance Dashboard - Calibration curves, coverage verification, feature importance, backtest results
2. Live Market Analyser - slug input -> API -> features -> MAPIE -> signal

Runs with: streamlit run src/app.py
'''

import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mlflow
import os
import sys
import logging

# Add src to path so we can import api
sys.path.append(os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title='Polymarket Conformal Predictor',
    layout='wide'
)

# Constratints
EXCLUDE_COLS = [
    'market_id', 'resolved_yes', 'end_date',
    'price_start', 'price_end', 'price_mean',
    'price_min', 'price_max'
]

SIGNAL_COLOURS = {
    'GREEN':  '#28a745',
    'RED':    '#dc3545',
    'YELLOW': '#ffc107',
}

# Model loading (cached)
st.cache_resource
def load_models():
    '''Load XGBoost and MAPIE models from MLflow. Returns (xgb, mapie) or (None, None)'''
    try:
        xgb = mlflow.xgboost.load_model('models:/polymarket-xgboost/5')
        mapie = mlflow.sklearn.load_model('models:/polymarket-mapie/latest')
        return xgb, mapie
    except Exception as e:
        logger.warning(f'Could not load models: {e}')
        return None, None

@st.cache_resource
def load_splits():
    '''Load train/calibration/test parquets. Returns dict or None'''
    try:
        base = os.path.join(os.path.dirname(__name__), '..', 'data', 'model')
        return {
            'train': pl.read_parquet(os.path.join(base, 'train.parquet')),             
            'calibration': pl.read_parquet(os.path.join(base, 'calibration.parquet')),
            'test': pl.read_parquet(os.path.join(base, 'test.parquet'))
        }
    except Exception as e:
        logger.warning(f'Could not load data splits: {e}')
        return None
    

def get_feature_cols(df: pl.DataFrame) -> list:
    return [c for c in df.column if c not in EXCLUDE_COLS]


# Sidebar
st.sidebar.title('Settings')
alpha = st.sidebar.slider(
    'Confidence level (alpha)',
    min_value = 0.05,
    max_value=0.20,
    value=0.10,
    step=0.05,
    help='Alpha = 1 - confidence. Lower alpha = stricter coverage guarantee'
)

st.sidebar.markdown(f'**Confidence level:** {(1-alpha)*100:.0f}%')
st.sidebar.markdown('---')
st.markdown('''
- 🟢 GREEN — Model confident YES
- 🔴 RED — Model confident NO
- 🟡 YELLOW — Model uncertain

**Disclaimer:** This is research not financial advice.
''')

# Main layout
st.title('Polymarket Conformal Prediction Pipeline')
st.markdown(
    'End-to-end ML pipeline applying conformal prediction to Polymarket '
    'political markets. Signals carry a mathematically guaranteed coverage property.'
)

tab1, tab2 = st.tabs(['Model Performance', 'Live Market Analyser'])

# Tab 1 - Model performance dashboard
with tab1:
    st.header('Model Performance Dashboard')

    xgb_model, mapie_model = load_models()
    splits = load_splits()

    if xgb_model is None or splits is None:
        st.warning(
            'Models or data splits not found'
            'Run this app from your main machine with MLflow and data available'
            'Showing static results from the completed backtest below'
        )

    # Static backtest results (always shown)
    st.subheader('Backtest Results')
    st.markdown('Results from the Jan-Apr 2026 test set (2,889 markets)')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Overall Accuracy', '92.0%')
    col2.metric('GREEN Accuracy', '83.9%', '702 signals')
    col3.metric('RED Accuracy', '94.7%', '2,107 signals')
    col4.metric('Coverage (90% target)', '92.5%', '+2.25%')

    st.markdown('---')

    # Disagreement analysis
    st.subheader('Disagreement Analysis')
    st.markdown(
        'When the model signal disagrees with the current market price — '
        'the more meaningful test for genuine mispricing detection.'
    )

    col1, col2, col3 = st.columns(3)
    col1.metric('Disagreement Signals', '136')
    col2.metric('Disagreement Accuracy', '40.4%', '-51.6% vs overall')
    col3.metric('Agreement Accuracy', '94.7%')

    st.info(
        '**Honest finding:** The model confirms market efficiency rather than '
        'beating it on near-resolved markets. The market is more informed than '
        'the model when they disagree. The signal is most useful as a market '
        'filtering and prioritisation tool.'
    )

    # Disagreement by price range 
    st.subheader('Disagreement Accuracy by Price Range')
    price_ranges = ['0.0–0.1', '0.1–0.3', '0.3–0.5', '0.5–0.7', '0.7–0.9', '0.9–1.0']
    accuracies   = [16.7, 29.2, 43.6, 52.8, 40.9, 22.2]
    counts       = [6, 24, 39, 36, 22, 9]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(price_ranges, accuracies, color=[
        '#dc3545' if a < 50 else '#28a745' for a in accuracies
    ], alpha=0.8, edgecolor='black')
    ax.axhline(y=50, linestyle='--', color='grey', label='50% baseline')
    ax.set_xlabel('Market price range at time of signal')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Disagreement Signal Accuracy by Price Range')
    ax.set_ylim(0, 80)
    ax.legend()
    for bar, acc, cnt in zip(bars, accuracies, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{acc}%\n(n={cnt})',
            ha='center', va='bottom', fontsize=8
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Alpha exploration ────────────────────────────────────────────────────
    st.markdown('---')
    st.subheader('Alpha Exploration — Coverage vs Signal Rate')

    alphas_exp    = [0.05, 0.10, 0.20]
    coverages_exp = [95.33, 92.25, 83.94]
    signals_exp   = [87.8, 97.2, 100.0]
    target_cov    = [95, 90, 80]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(alphas_exp, coverages_exp, marker='o', label='Empirical coverage', color='steelblue')
    axes[0].plot(alphas_exp, target_cov, linestyle='--', label='Theoretical target', color='grey')
    axes[0].set_xlabel('Alpha')
    axes[0].set_ylabel('Coverage (%)')
    axes[0].set_title('Empirical vs Theoretical Coverage')
    axes[0].legend()

    axes[1].plot(alphas_exp, signals_exp, marker='o', label='Signal rate', color='#28a745')
    axes[1].set_xlabel('Alpha')
    axes[1].set_ylabel('Signal rate (%)')
    axes[1].set_title('Signal Rate vs Alpha')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Feature importance (static) ──────────────────────────────────────────
    st.markdown('---')
    st.subheader('Feature Importance (v5 Model)')

    features    = [
        'log_total_volume', 'log_market_volume', 'log_avg_trade_size',
        'log_trade_count', 'price_range', 'n_siblings', 'price_volatility',
        'neg_risk', 'buy_ratio', 'price_volatility', 'sibling_volume_ratio',
        'days_active', 'days_to_resolution', 'consistency_gap', 'price_momentum'
    ]
    importances = [0.025, 0.025, 0.027, 0.030, 0.035, 0.035,
                   0.037, 0.030, 0.048, 0.038, 0.065,
                   0.09, 0.11, 0.19, 0.25]

    # Sort ascending for horizontal bar
    sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1])
    feat_sorted, imp_sorted = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat_sorted, imp_sorted, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Feature Importance')
    ax.set_title('XGBoost Feature Importance — v5 (price level features removed)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Live metrics (only when models available) ────────────────────────────
    if xgb_model is not None and splits is not None:
        st.markdown('---')
        st.subheader('Live Model Metrics')

        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.calibration import calibration_curve

        test          = splits['test']
        feature_cols  = get_feature_cols(test)
        X_test        = test.select(feature_cols).to_numpy()
        y_test        = test.select('resolved_yes').to_numpy().ravel().astype(int)

        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        brier  = brier_score_loss(y_test, y_prob)
        auc    = roc_auc_score(y_test, y_prob)

        col1, col2 = st.columns(2)
        col1.metric('Test AUC-ROC', f'{auc:.4f}')
        col2.metric('Test Brier Score', f'{brier:.4f}')

        # Live calibration curve
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(mean_pred, frac_pos, marker='o', label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Curve (Test Set)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE MARKET ANALYSER
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header('Live Market Analyser')
    st.markdown(
        'Enter a Polymarket market slug to fetch live data, '
        'engineer features, and generate a conformal prediction signal.'
    )

    # Slug input
    slug = st.text_input(
        'Market slug',
        placeholder='e.g. democratic-presidential-nominee-2028',
        help='Found in the Polymarket URL: polymarket.com/event/{slug}'
    )

    analyse_btn = st.button('Analyse Market', type='primary')

    if analyse_btn and slug:
        xgb_model, mapie_model = load_models()

        if xgb_model is None or mapie_model is None:
            st.error(
                '❌ Models not available. Run this app from your main machine '
                'with MLflow models loaded.'
            )
        else:
            try:
                from api import (
                    get_market_by_slug, get_recent_trades,
                    MarketNotFoundError, MarketAlreadyResolvedError,
                    InsufficientTradesError, APIError
                )
                from features import FeatureEngineer

                with st.spinner('Fetching live market data...'):
                    market = get_market_by_slug(slug)

                with st.spinner('Fetching recent trades...'):
                    trades = get_recent_trades(market['condition_id'], days=30)

                # Display market info
                st.subheader(market['question'])
                col1, col2, col3, col4 = st.columns(4)
                col1.metric('Current Price (YES)', f"{market['current_price']:.3f}")
                col2.metric('Volume', f"${market['volume']:,.0f}")
                col3.metric('neg_risk', market['neg_risk'])
                col4.metric('Trades (30d)', len(trades))

                st.markdown('---')

                with st.spinner('Engineering features...'):
                    # Build markets dataframe for FeatureEngineer
                    markets_row = pl.DataFrame({
                        'condition_id': [market['condition_id']],
                        'volume':       [market['volume']],
                        'event_id':     [market['event_id']],
                        'neg_risk':     [market['neg_risk']],
                        'end_date':     [market['end_date']],
                    })

                    fe = FeatureEngineer()
                    features_df = fe.compute_features_for_market(
                        condition_id=market['condition_id'],
                        trades=trades,
                        market_meta=markets_row
                    )

                    feature_cols = get_feature_cols(features_df)
                    X_live = features_df.select(feature_cols).to_numpy()

                with st.spinner('Running conformal prediction...'):
                    from mapie.classification import SplitConformalClassifier

                    # Re-conformalize at selected alpha if different from default
                    confidence_level = 1 - alpha

                    # Use stored mapie model directly at default alpha
                    # or re-conformalize at user-selected alpha
                    if abs(alpha - 0.10) < 0.001:
                        # Use the stored model
                        _, pred_sets = mapie_model.predict_set(X_live)
                    else:
                        # Re-conformalize at selected alpha
                        # Load calibration set for this
                        splits = load_splits()
                        if splits:
                            calib       = splits['calib']
                            feat_cols_c = get_feature_cols(calib)
                            X_calib     = calib.select(feat_cols_c).to_numpy()
                            y_calib     = calib.select('resolved_yes').to_numpy().ravel()

                            mapie_alpha = SplitConformalClassifier(
                                estimator=mapie_model.estimator_,
                                confidence_level=confidence_level,
                                conformity_score='lac',
                                prefit=True
                            )
                            mapie_alpha.conformalize(X_calib, y_calib)
                            _, pred_sets = mapie_alpha.predict_set(X_live)
                        else:
                            _, pred_sets = mapie_model.predict_set(X_live)

                    # Extract signal
                    pred_set     = pred_sets[0]
                    yes_in_set   = bool(pred_set[1][0])
                    no_in_set    = bool(pred_set[0][0])

                    if yes_in_set and not no_in_set:
                        signal      = 'GREEN'
                        description = 'Model confident YES'
                        emoji       = '🟢'
                    elif no_in_set and not yes_in_set:
                        signal      = 'RED'
                        description = 'Model confident NO'
                        emoji       = '🔴'
                    else:
                        signal      = 'YELLOW'
                        description = 'Model uncertain — no signal'
                        emoji       = '🟡'

                # Display signal
                st.subheader(f'{emoji} Signal: {signal}')
                st.markdown(f'**{description}**')
                st.markdown(
                    f'Confidence level: **{(1-alpha)*100:.0f}%** '
                    f'(alpha = {alpha}) — '
                    f'empirically verified at **92.25%** coverage on test set.'
                )

                col1, col2 = st.columns(2)
                col1.markdown(f'**YES in prediction set:** {yes_in_set}')
                col2.markdown(f'**NO in prediction set:** {no_in_set}')

                # Show features
                st.markdown('---')
                st.subheader('Feature Values')
                feat_display = features_df.select(feature_cols).to_pandas()
                st.dataframe(feat_display)

                # Disclaimer
                st.markdown('---')
                st.caption(
                    '⚠️ This signal is research output not financial advice. '
                    'The model shows no reliable edge over market prices on '
                    'near-resolved markets. Use as a research and filtering '
                    'tool only. Always make your own trading decisions.'
                )

            except MarketNotFoundError as e:
                st.error(f'❌ Market not found: {e}')
            except MarketAlreadyResolvedError as e:
                st.warning(f'⚠️ Market already resolved: {e}')
            except InsufficientTradesError as e:
                st.warning(f'⚠️ Not enough recent trades: {e}')
            except APIError as e:
                st.error(f'❌ API error: {e}')
            except Exception as e:
                st.error(f'❌ Unexpected error: {e}')
                logger.exception(e)

    elif analyse_btn and not slug:
        st.warning('Please enter a market slug.')