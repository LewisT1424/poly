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
    
    def compute_consistency_features(self, feature_matrix):
        '''
        New consistency features, This function will run after the main feature matrix is created to add the additional features onto the feature matrix
        
        Features added:
        - consistency_gap - How far this market's price is from what siblings imply
        - n_siblings - How many related amrekts exist in the same event
        - sibling_volume_ratio - sibling total volume / this markets volume
        '''
        markets = self.markets.clone()

        # Step 1 - Bring event_id and volume into the feature matrix. Join on condition_id (market_id in feature matrix)
        enriched = feature_matrix.join(
            markets.select(['condition_id', 'event_id', 'volume', 'neg_risk']),
            left_on='market_id',
            right_on='condition_id',
            how='left'
        )

        # Step 2 - Compute sibling aggregates using window functions. For each market, sum price_end and volume of ALL markets in same event
        # Then subtract this market's own values to get siblings only
        enriched = enriched.with_columns([
            # Total price_end sum for all markets in this event
            pl.col('price_end').sum().over('event_id').alias('event_price_sum'),

            # Total volume sum for all markets in this event
            pl.col('volume').sum().over('event_id').alias('event_volume_sum'),

            # Count of all markets in this event
            pl.col('market_id').count().over('event_id').alias('event_market_count')
        ])

        # Step 3 - subtract this market's own values to get sibling-only values
        enriched = enriched.with_columns([
            # Sibling price sum = total event price sum minus this market'sprice
            (pl.col('event_price_sum') - pl.col('price_end')).alias('sibling_price_sum'),

            # Sibling volume sum = total event volume minus this market's volume
            (pl.col('event_volume_sum') - pl.col('volume')).alias('sibling_volume_sum'),

            # Number of siblings = total arkets in event minus this one
            (pl.col('event_market_count') - 1).alias('n_siblings')
        ])

        # Step 4 - compute the three final features
        enriched = enriched.with_columns([
            # consistency_gap: Implied price = 1.0 - sibling_price_sum
            # gap = actual price - implied price
            # positive = overpriced vs siblings
            # negative = underpriced vs siblings
            # zero = no siblings
            pl.when(
                (pl.col('neg_risk') == 1) &
                (pl.col('n_siblings') > 0) &
                (pl.col('event_price_sum') >= 0.7) &
                (pl.col('event_price_sum') <= 1.3)
            ).then(
                pl.col('price_end') - (1.0 / (pl.col('n_siblings') + 1))
            ).otherwise(0.0)
            .alias('consistency_gap'),

            # n_siblings - Only meaningful for neg_risk = 1 markets
            pl.when(pl.col('neg_risk') == 1)
            .then(pl.col('n_siblings'))
            .otherwise(0)
            .alias('n_siblings'),
            

            # sibling_volume_ratio: How much more voluime do siblings have vs this market
            # high = siblings are more liquid = consistency signal more trustworthy
            # low = no siblings or this market has zero volume
            pl.when(
                (pl.col('neg_risk') == 1) &
                (pl.col('n_siblings') > 0) &
                (pl.col('volume') > 0)
            ).then(
                (pl.col('sibling_volume_sum') / pl.col('volume')).log1p()
            ).otherwise(0.0)
            .alias('sibling_volume_ratio')
        ])

        # Step 5 - drop intermediate columns, keep only the there new features
        enriched = enriched.drop([
            'event_id', 'volume', 'event_price_sum',
            'event_volume_sum', 'event_market_count',
            'sibling_price_sum', 'sibling_volume_sum'
        ])

        logger.info(f"Consistency features added. Shape: {enriched.shape}")
        logger.info(f"neg_risk = 1 markets: {(enriched['neg_risk'] == 1).sum()}")
        logger.info(f"neg_risk = 0 markets: {(enriched['neg_risk'] == 0).sum()}")
        logger.info(f"Markets with active consistency signal: {(enriched['consistency_gap'] != 0).sum()}")

        return enriched
            
    
    def run(self):
        # Step 1 - Apply existing feature engineering transformations 
        feature_matrix = self.feature_engineering()

        # Step 2 - add consistency features
        feature_matrix = self.compute_consistency_features(feature_matrix) 

        # Save feature matrix
        feature_matrix.write_parquet('data/processed/feature_matrix.parquet')
        logger.info(f"Feature matrix saved: {feature_matrix.shape}")

if __name__ == '__main__':
    FE = FeatureEngineer()
    FE.run()