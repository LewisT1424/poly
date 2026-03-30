'''
Score 3 years of Polymarket wallet performance from historical trade data

Input:
- data/processed/quant_filtered.parquet
- data/processed/markets_political.parquet

Output:
- data/processed/wallet_scores.parquet

Scoring dimensions:
- Accuracy (40%) - volume-weighted % of market bet correctly
- Consistency (30%) - accuracy uniform across categories, not one lucky run
- Recency (20%) - last 6 months 3x weight, 6-12 months 2x, 12-24 months 1x
- Volume (10%) - log-scaled total USD deployed

Filters (must pass all)
- Minimum 20 markets traded
- Minimum $500 total volume
- Active within last 12 months
- Exclude markets where wallet was the only trader
'''

import polars as pl
import numpy as np
from pathlib import Path
import json
import time
import logging

# Config
QUANT_PATH = Path('data/processed/quant_filtered.parquet')
MARKETS_PATH = Path('data/processed/markets_political.parquet')
OUTPUT_PATH = Path('data/processed/wallet_scores.parquet')

MIN_MARKETS = 20
MIN_VOLUME_USD = 500.0
ACCURACY_WEIGHT = 0.40
CONSIST_WEIGHT = 0.30
RECENCY_WEIGHT = 0.20
VOLUME_WEIGHT = 0.10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Recency cutoffs (unix seconds) - computed relative to max timestamp in dataset
RECENCY_0_6_MONTHS = 60 * 60 * 24 * 180
RECENCY_6_12_MONTHS = 60 * 60 * 24 * 365
RECENCY_12_24_MONTHS = 60 * 60 * 24 * 730

# Step 1 - Load and join

def load_data() -> pl.DataFrame:
    logger.info('Loading quant trades...')
    t0 = time.time()
    quant = pl.read_parquet(QUANT_PATH)
    logger.info(f"{len(quant):,} trades loaded ({time.time()-t0:.1f}s)")

    logger.info("Loading markets")
    markets = pl.read_parquet(MARKETS_PATH).filter(
        pl.col('resolved_yes').is_not_null()
    ).select(['condition_id', 'event_id', 'resolved_yes', 'volume', 'end_date', 'neg_risk'])
    logger.info(f"{len(markets):,} resolved markets")

    # We only want taker trades - takers are directional bettors
    # makers are liquidity providers
    logger.info("Joining and filtering to taker trades only")
    df = (
        quant.filter(pl.col('taker').is_not_null())
        .join(markets, on='condition_id', how='inner')
        .rename({'volume': 'market_volume', 'event_id': 'event_id'})
    )

    # Derive correctness
    # BUY = betting YES -> correct if resolved_yes = True
    # SELL = betting NO -> correct if resolved_yes = False
    df = df.with_columns([
            pl.when(pl.col("side") == "BUY")
            .then(pl.col("resolved_yes"))
            .otherwise(~pl.col("resolved_yes"))
            .alias("correct"),

            pl.from_epoch(pl.col("timestamp"), time_unit="s")
            .dt.convert_time_zone("UTC")
            .alias("trade_dt"),
        ])

    # Filter out post-resolution trades — these are arb/cleanup trades
    # on known outcomes and would inflate accuracy scores artificially.
    # Require trade made at least 24 hours before end_date.
    before = len(df)
    df = df.filter(
        pl.col("trade_dt") < (pl.col("end_date") - pl.duration(hours=24))
    )
    logger.info(f"  Removed {before - len(df):,} post-resolution trades "
        f"({100*(before-len(df))/before:.1f}%)")
    logger.info(f"  {len(df):,} taker trades remaining")

    # Filter out near-certainty trades — these show no predictive skill.
    # A wallet buying YES at 0.95+ or selling YES at 0.05- is not predicting,
    # just harvesting basis points on already-decided markets.
    # We only want trades where the market was genuinely uncertain at entry.
    before = len(df)
    df = df.filter(
        (pl.col("price") >= 0.05) & (pl.col("price") <= 0.95)
    )
    logger.info(f"  Removed {before - len(df):,} near-certainty trades "
          f"({100*(before-len(df))/before:.1f}%)")
    logger.info(f"  {len(df):,} trades in uncertain price range remaining")
    return df


# Step 2 - Collapse to one row per walelt x market
def aggregate_to_positions(df: pl.DataFrame) -> pl.DataFrame:
    '''
    A wallet may trade a market multipl times. Collapse to a single position per wallet x market using volume-weighted average entry price 
    and majority-vote correctness (weighted by usd_amount)
    '''
    logger.info("Aggregating to wallet x market positions")
    t0 = time.time()

    positions = (
        df.group_by(['taker', 'condition_id'])
        .agg([
            # Volume-weighted correctness - sum of usd on correct side, divided by total usd in that market
            (pl.col('usd_amount') * pl.col('correct').cast(pl.Float64))
            .sum().alias('usd_correct'),
            pl.col('usd_amount').sum().alias('usd_total'),

            # Average entry price (volume-weighted)
            (pl.col('price') * pl.col('usd_amount')).sum().alias('_price_x_vol'),

            # Market metadata - take first (same for all rows in group)
            pl.col('event_id').first(),
            pl.col('resolved_yes').first(),
            pl.col('market_volume').first(),
            pl.col('neg_risk').first(),
            pl.col('end_date').first(),

            # Most recent trade in this market (for recency scoring)
            pl.col('trade_dt').max().alias('last_trade_dt'),

            # Number of trades in this market
            pl.len().alias('n_trades'),
        ]).with_columns([
            # Position is correct if majority of USD on the right side
            (pl.col('usd_correct') / pl.col('usd_total') >= 0.5).alias('position_correct'),
            (pl.col('_price_x_vol') / pl.col('usd_total')).alias('avg_entry_price')
        ]).drop('_price_x_vol')
    )

    logger.info(f"{len(positions):,} wallet market positions ({time.time()-t0:.1f}s)")
    return positions

# Step 3 - Filter out solo markets
def remove_solo_markets(positions: pl.DataFrame) -> pl.DataFrame:
    '''Remove markets where the wallet was the only traders.'''
    logger.info("Removing solo markets...")
    traders_per_market = (
        positions.group_by('condition_id').agg(
            pl.col('taker').n_unique().alias('n_traders')
        )
    )
    positions = positions.join(traders_per_market, on='condition_id', how='left')
    before = len(positions)
    positions = positions.filter(pl.col('n_traders') > 1).drop('n_traders')
    logger.info(f"Removed {before - len(positions):,} solo-market positions")
    return positions

def compute_baseline(df: pl.DataFrame) -> tuple[float, dict]:
    """
    Compute baseline accuracy (always back market favourite) overall
    and per price bucket. Returns (overall_baseline, bucket_baselines).
    bucket_baselines keys: '0.05-0.20', '0.20-0.35', etc.
    """
    df = df.with_columns(
        pl.when(pl.col("price") > 0.5)
          .then(pl.col("resolved_yes"))
          .otherwise(~pl.col("resolved_yes"))
          .alias("baseline_correct"),

        pl.when(pl.col("price") < 0.20).then(pl.lit("0.05-0.20"))
          .when(pl.col("price") < 0.35).then(pl.lit("0.20-0.35"))
          .when(pl.col("price") < 0.50).then(pl.lit("0.35-0.50"))
          .when(pl.col("price") < 0.65).then(pl.lit("0.50-0.65"))
          .when(pl.col("price") < 0.80).then(pl.lit("0.65-0.80"))
          .otherwise(pl.lit("0.80-0.95"))
          .alias("price_bucket")
    )

    overall = df["baseline_correct"].mean()

    bucket_baselines = (
        df.group_by("price_bucket")
        .agg(pl.col("baseline_correct").mean().alias("bucket_baseline"))
        .to_dicts()
    )
    bucket_map = {r["price_bucket"]: r["bucket_baseline"] for r in bucket_baselines}

    return overall, bucket_map



# Step 4 - Score each wallet
def score_wallets(positions: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Computing wallet scores...")
    t0 = time.time()

    # Max timestamp in data - used to compute recency
    max_ts = positions['last_trade_dt'].max()
    logger.info(f"Data runs to: {max_ts}")

    # Recency cutoffs as datetimes
    cutoff_6m = max_ts - pl.duration(days=180)
    cutoff_12m = max_ts - pl.duration(days=365)
    cutoff_24m = max_ts - pl.duration(days=730)
    
    logger.info("Computing baseline accuracy...")
    baseline_overall, baseline_by_bucket = compute_baseline(df)
    logger.info(f"  Overall baseline: {baseline_overall:.4f} ({baseline_overall*100:.1f}%)")

    # Tag each position with recency weight
    positions = positions.with_columns(
        pl.when(pl.col('last_trade_dt') >= cutoff_6m)
        .then(pl.lit(3.0))
        .when(pl.col('last_trade_dt') >= cutoff_12m)
        .then(pl.lit(2.0))
        .when(pl.col('last_trade_dt') >= cutoff_24m)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.5))
        .alias('recency_weight')
    )

    # Per wallet x event_id: category accuracy
    # Used for cosistency scoring
    cat_accuracy = (
        positions.group_by(['taker', 'event_id']).agg([
            pl.col('position_correct').mean().alias('cat_accuracy'),
            pl.col('usd_total').sum().alias('cat_volume'),
            pl.len().alias('cat_n_markets')
        ])
    )

    # Main wallet aggregation
    wallet_stats = (
        positions.group_by('taker').agg([
            # Core counts
            pl.len().alias('n_markets'),
            pl.col('usd_total').sum().alias('total_volume'),
            pl.col('last_trade_dt').max().alias('last_active'),

            # Accuracy - volume-weighted across all markets
            (
                (pl.col('position_correct').cast(pl.Float64) * pl.col('usd_total')).sum() / pl.col('usd_total').sum()
            ).alias('accuracy'),

            # Recency score - weighted accuracy where recent trades count more
            (
                (pl.col('position_correct').cast(pl.Float64) * pl.col('usd_total') * pl.col('recency_weight')).sum() 
                / (pl.col('usd_total') * pl.col('recency_weight')).sum()
            ).alias('recency_score'),

            # Top cateogy by volume
            pl.col('event_id').sort_by(pl.col('usd_total')).last().alias('top_category'),

            # Accuracy by category - serialise to JSON for the output schema
            pl.struct(['event_id', 'position_correct', 'usd_total']).alias('_cat_data'),
        ])
    )

    # Apply filters
    logger.info("Applying filters")
    before = len(wallet_stats)

    one_year_ago = max_ts - pl.duration(days=365)

    wallet_stats = wallet_stats.filter(
        (pl.col("n_markets") >= MIN_MARKETS) &
        (pl.col("total_volume") >= MIN_VOLUME_USD) &
        (pl.col("last_active") >= one_year_ago)
    )

    # Will be joined with consistency shortly — pre-filter on n_categories
    # happens after the join below

    logger.info(f"Wallets after filterrs: {len(wallet_stats):,} (from {before:,})")


    # Consitency score - std of category accuracy / mean category accuracy, inverted. Only include cateogries with 3+ markets to avoid noise
    MIN_MARKETS_PER_CAT = 5
    MIN_CATEGORIES      = 3

    consistency = (
        cat_accuracy
        .filter(pl.col("cat_n_markets") >= MIN_MARKETS_PER_CAT)
        .group_by("taker")
        .agg([
            pl.col("cat_accuracy").std().alias("cat_std"),
            pl.col("cat_accuracy").mean().alias("cat_mean"),
            pl.col("cat_accuracy").count().alias("n_categories"),
        ])
        .with_columns(
            pl.when(pl.col("cat_mean") > 0)
              .then(1.0 - (pl.col("cat_std") / pl.col("cat_mean")))
              .otherwise(pl.lit(0.5))
              .clip(0.0, 1.0)
              .alias("consistency_score")
        )
        .select(["taker", "consistency_score", "n_categories"])
    )

    wallet_stats = wallet_stats.join(consistency, on="taker", how="left")

    # Wallets with fewer than MIN_CATEGORIES qualifying categories
    # get a penalty consistency score rather than neutral —
    # we can't trust accuracy that isn't spread across multiple domains
    wallet_stats = wallet_stats.with_columns([
        pl.col("consistency_score").fill_null(0.3),
        pl.col("n_categories").fill_null(0),
    ])

    # Require at least MIN_CATEGORIES to remain in the scored set
    before = len(wallet_stats)
    wallet_stats = wallet_stats.filter(pl.col("n_categories") >= MIN_CATEGORIES)
    print(f"  Removed {before - len(wallet_stats):,} wallets with fewer than "
          f"{MIN_CATEGORIES} qualifying categories")

    # Volume score - log-scaled normalised 0-1
    # --- Volume score — log-scaled, normalised 0-1 ---
    log_vol = np.log1p(wallet_stats["total_volume"].to_numpy())
    vol_min, vol_max = log_vol.min(), log_vol.max()
    log_vol_norm = ((log_vol - vol_min) / (vol_max - vol_min + 1e-9)).astype(np.float64)
    wallet_stats = wallet_stats.with_columns(
        pl.Series(name="volume_score", values=log_vol_norm.tolist())
    )

    # Edge above baseline
    # overall edge: simple accuracy minus market-favourite baseline
    wallet_stats = wallet_stats.with_columns(
        (pl.col('accuracy') - baseline_overall).alias('edge_overall')
    )

    # Weighted edge: accuracy minus the baseline for price buckets
    # This wallet actually traded in. Rewards skill in harder buckets
    # We need per-wallet bucket-weighted baseline for the trade-level data
    logger.info("Computing per-wallet weighted edge")

    wallet_bucket_baseline = (
        df.filter(pl.col('taker').is_in(wallet_stats['taker']))
        .with_columns(
            pl.when(pl.col('price') < 0.20).then(pl.lit('0.05-0.20'))
            .when(pl.col('price') < 0.35).then(pl.lit('0.20-0.35'))
            .when(pl.col('price') < 0.50).then(pl.lit('0.35-0.50'))
            .when(pl.col('price') < 0.65).then(pl.lit('0.50-0.65'))
            .when(pl.col('price') < 0.80).then(pl.lit('0.65-0.80'))
            .otherwise(pl.lit('0.80-0.95'))
            .alias('price_bucket')
        ).with_columns(
            pl.col('price_bucket').replace(baseline_by_bucket)
            .cast(pl.Float64).alias('bucket_baseline')
        ).group_by('taker').agg(
            # Volume weighted average of the bucket baselines
            # for the buckets this wallet traded in
            (
                (pl.col('usd_amount') * pl.col('bucket_baseline')).sum() 
                / pl.col('usd_amount').sum()
            ).alias('wallet_baseline')
        )
    )

    wallet_stats = wallet_stats.join(wallet_bucket_baseline, on='taker', how='left')
    wallet_stats = wallet_stats.with_columns([
        pl.col('wallet_baseline').fill_null(baseline_overall),
        (pl.col('accuracy') - pl.col('wallet_baseline')).alias('edge_weighted')
    ])

    # Composite score
    wallet_stats = wallet_stats.with_columns(
        (
            pl.col('accuracy') * ACCURACY_WEIGHT +
            pl.col('consistency_score') * CONSIST_WEIGHT +
            pl.col('recency_score') * RECENCY_WEIGHT + 
            pl.col('volume_score') * VOLUME_WEIGHT
        ).alias('composite_score')
    )

    # Only keep wallets that beat their own bucket-weighted baseline
    before = len(wallet_stats)
    wallet_stats = wallet_stats.filter(pl.col('edge_weighted') > 0)
    logger.info(f"  Removed {before - len(wallet_stats):,} below-baseline wallets")
    logger.info(f"  {len(wallet_stats):,} wallets beat their bucket-weighted baseline")

    # Accuracy by category JSON - Build a simpler per-wallet category accuracy map
    cat_json = (
            cat_accuracy
            .filter(pl.col("taker").is_in(wallet_stats["taker"]))
            .group_by("taker")
            .agg(
                pl.struct(["event_id", "cat_accuracy", "cat_volume"])
                .alias("cats")
            )
        )

    # Serialise to JSON string
    rows = cat_json.to_dicts()
    cat_map = {
        r['taker']: json.dumps({
            c['event_id']: round(c['cat_accuracy'], 4)
            for c in r['cats']
        })
        for r in rows
    }
    wallet_stats = wallet_stats.with_columns(
        pl.col('taker').replace(cat_map).alias('accuracy_by_cat')
    )

    logger.info(f"Scoring complete ({time.time() - t0:.1f}s)")
    return wallet_stats

# Step 5 - Select final output schema and save
def save_output(wallet_stats: pl.DataFrame) -> None:
    output = wallet_stats.select([
        pl.col("taker").alias("wallet_address"),
        "accuracy",
        "edge_overall",
        "edge_weighted",
        "wallet_baseline",
        "consistency_score",
        "recency_score",
        "volume_score",
        "composite_score",
        "n_markets",
        "n_categories",
        "top_category",
        "last_active",
        "total_volume",
        "accuracy_by_cat",
    ]).sort("composite_score", descending=True).with_columns(
        pl.when(pl.col("edge_weighted") > 0.10).then(pl.lit(1))
          .when(pl.col("edge_weighted") > 0.05).then(pl.lit(2))
          .otherwise(pl.lit(3))
          .alias("tier")
    )

    output = output.with_columns(
        pl.when(pl.col("edge_weighted") > 0.10).then(pl.lit(1))
          .when(pl.col("edge_weighted") > 0.05).then(pl.lit(2))
          .otherwise(pl.lit(3))
          .alias("tier")
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.write_parquet(OUTPUT_PATH)

    logger.info(f"Saved {len(output):,} scored wallets → {OUTPUT_PATH}")

    logger.info("\nTop 10 wallets:")
    logger.info(output.select([
        "wallet_address", "tier", "composite_score", "accuracy",
        "edge_overall", "edge_weighted", "n_markets", "total_volume"
    ]).head(10))

    logger.info("\nScore distribution:")
    logger.info(output.select("composite_score").describe())

# Main
if __name__ == '__main__':
    logger.info('=== PolySharp - Wallet Scoring ===\n')
    t_start = time.time()

    df        = load_data()
    positions = aggregate_to_positions(df)
    positions = remove_solo_markets(positions)
    scores    = score_wallets(positions, df)
    save_output(scores)

    logger.info(f"\nTotal runtime: ({time.time()-t_start:.1f}s)")