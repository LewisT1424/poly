'''
Feature engineering script.

This will be used by iniital training data (historical data), and inference data. We need to ensure that functions will be the same for both pieces of data
'''
import polars as pl
import logging
from datetime import timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 30 # Days of trade history to use per market
MIN_TRADES = 10 # Drop markets with fewer then 10 trades

def handle_nulls(df):
    # Currently no null values in the dataset so we are going to ignore it
    pass

class FeatureEngineer:
    def __init__(self):
        self.quant = None
        self.markets = None

        self._load_data()

        logger.info(f"Successfully initialized feature engineer")



    def _load_data(self):
        try:
            self.markets = pl.read_parquet('data/processed/markets_political.parquet')
            self.quant = pl.read_parquet('data/processed/quant_filtered.parquet')
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def feature_engineering(self):
        '''
        Main feature engineering function used on both training and inference data.
        '''
        quant_copy = self.quant.clone()
        markets = self.markets.clone()

        # Fix timestamp mismatch between quant and markets
        quant_copy = quant_copy.with_columns(
            pl.from_epoch(pl.col('timestamp'), time_unit='s').alias('datetime')
        ).with_columns(
            pl.col('datetime').dt.convert_time_zone('UTC').dt.cast_time_unit('ms')
        )

        rows = []  # collect one dict per market

        for i, m in enumerate(markets.iter_rows(named=True)):

            # Pull trades for this market and sort chronologically
            id = m['condition_id']
            trades = quant_copy.filter(pl.col('condition_id') == id)
            trades = trades.sort('datetime', descending=False)

            if trades.shape[0] == 0:
                continue # Break if no trades


            # ── Window boundaries ────────────────────────────────────
            # Using latest_trade for sample testing on laptop
            # Swap to end_date when running full dataset at home
            end_date = m['end_date']
            window_start = end_date - timedelta(days=LOOKBACK_DAYS)

            window_trades = trades.filter(
                (pl.col('datetime') >= window_start) &
                (pl.col('datetime') < end_date)
            )

            # Skip markets that don't have enough trades in the window
            if len(window_trades) < MIN_TRADES:
                logger.warning(f"Skipping {id} — not enough trades in window")
                continue

            logger.info(f"Processing market: {id}")

            # ── Price features ───────────────────────────────────────
            price_series = window_trades['price']

            price_start = price_series.first()
            price_end   = price_series.last()
            price_mean  = price_series.mean()
            price_min   = price_series.min()
            price_max   = price_series.max()

            # ── Metadata features ────────────────────────────────────
            # days_active: how long the market ran in total
            days_active = (m['end_date'] - m['created_at']).days

            # days_to_resolution: how many days were left at the end of the window
            days_to_resolution = (m['end_date'] - window_trades['datetime'].max()).days

            # ── Build feature row ────────────────────────────────────
            features = {
                # Identifiers
                'market_id':    id,
                'resolved_yes': m['resolved_yes'],

                # Price features
                'price_start':      price_start,
                'price_end':        price_end,
                'price_mean':       price_mean,
                'price_min':        price_min,
                'price_max':        price_max,
                'price_volatility': price_series.std(),
                'price_range':      price_max - price_min,
                'price_momentum':   price_end - price_start,

                # Volume and activity features
                'log_total_volume':   np.log1p(window_trades['usd_amount'].sum()),
                'log_trade_count':    np.log1p(len(window_trades)),
                'log_avg_trade_size': np.log1p(window_trades['usd_amount'].mean()),

                # Sentiment feature
                'buy_ratio': (window_trades['side'] == 'BUY').mean(),

                # Market metadata features
                'log_market_volume':  np.log1p(float(m['volume'])),
                'days_active':        days_active,
                'days_to_resolution': days_to_resolution,

                'end_date': end_date
            }

            rows.append(features)

        # Convert list of dicts to a single DataFrame — one row per market
        feature_matrix = pl.DataFrame(rows)
        logger.info(f"Feature matrix built: {feature_matrix.shape}")

        return feature_matrix
        


    
    def run(self):
        feature_matrix = self.feature_engineering()

        # Save feature matrix
        feature_matrix.write_parquet('data/processed/feature_matrix.parquet')

if __name__ == '__main__':
    FE = FeatureEngineer()
    FE.run()