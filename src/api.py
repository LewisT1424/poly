'''
Live Polymarket API client for infernece pipeline

Two main function:
- get_market_by_slug(slug) -> market metadata + current price
- get_recent_trades(condition_id, days) -> Polars Dataframe matching quant schema
'''

import requests 
import polars as pl
import logging
from datetime import datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Base URLs
GAMMA_API = 'https://gamma-api.polymarket.com'
CLOB_API = 'https://clob.polymarket.com'
DATA_API = 'https://data-api.polymarket.com'

TIMEOUT = 10 # Seconds per request

# Custom exceptions
class MarketNotFound(Exception):
    '''Raised when slug does not match any market'''
    pass

class MarketAlreadyResolvedError(Exception):
    '''Raised when market has already closed and resolved'''
    pass

class InsufficientTradesError(Exception):
    '''Raised when fewer than MIN_TRADES trades exist in the lookback window'''
    pass

class APIError(Exception):
    '''Raised on unexpected API failures'''
    pass


# Core functions
def get_market_by_slug(slug: str) -> dict:
    import json

    # Step 1 — try events endpoint first (slug is always an event slug)
    try:
        resp = requests.get(
            f'{GAMMA_API}/events',
            params={'slug': slug},
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        raise APIError(f'Gamma API timed out fetching slug: {slug}')
    except requests.exceptions.RequestException as e:
        raise APIError(f'Gamma API request failed: {e}')

    # Handle list response
    if isinstance(data, list):
        if len(data) == 0:
            raise MarketNotFoundError(f'No event found for slug: {slug}')
        event = data[0]
    elif isinstance(data, dict):
        event = data
    else:
        raise MarketNotFoundError(f'Unexpected response for slug: {slug}')

    # Step 2 — extract market(s) from the event
    # Events contain a 'markets' array
    markets_in_event = event.get('markets', [])

    if not markets_in_event:
        raise MarketNotFoundError(f'Event found but contains no markets: {slug}')

    # For single-outcome events take the first market
    # For multi-outcome events the user should specify which market —
    # for now take the highest volume one
    if len(markets_in_event) == 1:
        market = markets_in_event[0]
    else:
        # Multiple markets — pick highest volume as default
        market = max(
            markets_in_event,
            key=lambda m: float(m.get('volume', 0) or 0)
        )

    # Step 3 — check market is still active
    if market.get('closed', False):
        raise MarketAlreadyResolvedError(
            f'Market "{slug}" is already closed and resolved.'
        )

    # Step 4 — extract condition_id and token IDs
    condition_id = market.get('conditionId') or market.get('condition_id')
    if not condition_id:
        raise APIError(f'Market data missing conditionId for slug: {slug}')

    clob_token_ids = market.get('clobTokenIds') or market.get('clob_token_ids', '[]')
    if isinstance(clob_token_ids, str):
        try:
            token_ids = json.loads(clob_token_ids)
        except json.JSONDecodeError:
            token_ids = []
    else:
        token_ids = clob_token_ids if clob_token_ids else []

    token_id_yes = token_ids[0] if len(token_ids) > 0 else None

    # Step 5 — fetch current price from CLOB API
    current_price = None
    if token_id_yes:
        try:
            price_resp = requests.get(
                f'{CLOB_API}/price',
                params={'token_id': token_id_yes, 'side': 'BUY'},
                timeout=TIMEOUT
            )
            price_resp.raise_for_status()
            price_data = price_resp.json()
            current_price = float(price_data.get('price', 0.5))
        except Exception as e:
            logger.warning(f'Could not fetch CLOB price: {e}. Falling back to outcomePrices.')
            outcome_prices = market.get('outcomePrices') or market.get('outcome_prices', '[]')
            if isinstance(outcome_prices, str):
                try:
                    prices = json.loads(outcome_prices)
                    current_price = float(prices[0]) if prices else 0.5
                except Exception:
                    current_price = 0.5
            else:
                current_price = 0.5

    # Step 6 — build return dict
    return {
        'condition_id':  condition_id,
        'question':      market.get('question', event.get('title', '')),
        'slug':          slug,
        'volume':        float(market.get('volume', 0) or 0),
        'end_date':      market.get('endDate') or market.get('end_date', ''),
        'neg_risk':      int(market.get('negRisk', 0) or market.get('neg_risk', 0)),
        'event_id':      str(event.get('id', '')),
        'current_price': current_price if current_price is not None else 0.5,
        'token_id_yes':  token_id_yes,
        'n_markets':     len(markets_in_event),
    }

def get_recent_trades(condition_id: str, days: int = 30) -> pl.DataFrame:
    '''
    Fetch recent trades for a market from Polymarket Data API

    Return a Polars DF matching the quant_political_parquet schema so it can be passed
    directly to features.py FeatureEngineer

    Params:
    - condition_id - The market condition ID (from get_market_slug)
    - days - Number of days of history to fetch (default 30, matching training window)

    Returns - pl.DataFrame

    Raises:
    - InsufficientTradesError - Fewer than 10 trades in the window
    - APIError - Unexpected API failure
    '''

    MIN_TRADES = 10
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_ts = int(cutoff_dt.timestamp())

    all_trades = []
    limit = 500
    offset = 0

    while True:
        try:
            resp = requests.get(
                f'{DATA_API}/trades',
                params={
                    'market':  condition_id,
                    'limit':   limit,
                    'offset':  offset,
                },
                timeout=TIMEOUT
            )
            # Stop pagination gracefully on 400 — API has offset limit
            if resp.status_code == 400:
                logger.warning(f'Data API returned 400 at offset {offset} — stopping pagination')
                break
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise APIError(f'Data API timed out fetching trades for {condition_id}')
        except requests.exceptions.RequestException as e:
            raise APIError(f'Data API request failed: {e}')  
        batch = resp.json()

        # Handle both list and dict with 'data' key
        if isinstance(batch, dict):
            batch = batch.get('data', batch.get('trades', []))

        if not batch:
            break

        # Filter to lookback window
        window_trades = [
            t for t in batch
            if int(t.get('timestamp', 0)) >= cutoff_ts
        ]
        all_trades.extend(window_trades)

        # If we get fewer than limit, we have all trades
        if len(batch) < limit:
            break

        # If earliest trade in batch is before uctoff, we have everything in window
        earliest = min(int(t.get('timestamp', 0)) for t in batch)
        if earliest < cutoff_ts:
            break

        offset += limit

    if len(all_trades) < MIN_TRADES:
        raise InsufficientTradesError(
            f'Only {len(all_trades)} trades found in last {days} days'
            f'for market {condition_id} minimum required: {MIN_TRADES}'
        )
    
    # Build DataFrame matching quant_polticial.parquet schema
    df = pl.DataFrame({
        'timestamp':    [int(t.get('timestamp', 0)) for t in all_trades],
        'condition_id': [t.get('conditionId', condition_id) for t in all_trades],
        'price':        [float(t.get('price', 0)) for t in all_trades],
        'usd_amount':   [float(t.get('size', 0) or 0) for t in all_trades],
        'side':         [str(t.get('side', 'BUY')) for t in all_trades],
        'maker':        [str(t.get('proxyWallet', '')) for t in all_trades],
        'taker':        [str(t.get('proxyWallet', '')) for t in all_trades],
    })

    # Convert timestamp to datetime - MUST match training pipeline exactly
    # Training used: pl.from_epoch('s').dt.convert_time_zone('UTC').dt.cast_time_unit('ms')
    df = df.with_columns(
        pl.from_epoch(pl.col('timestamp'), time_unit='s')
        .dt.convert_time_zone('UTC')
        .dt.cast_time_unit('ms')
        .alias('datetime')
    )

    logger.info(
        f'Fetched {len(df)} trades for {condition_id}'
        f'(last {days} days)'
    )

    return df

# CHeck test
if __name__ == '__main__':
    TEST_SLUG = 'democratic-presidential-nominee-2028'

    print(f'\nTesting get_market_by_slug with slug: {TEST_SLUG}')
    print('-' * 60)

    market = get_market_by_slug(TEST_SLUG)
    print(f"Question:      {market['question']}")
    print(f"Condition ID:  {market['condition_id']}")
    print(f"Current price: {market['current_price']:.3f}")

    print(f'\nTesting get_recent_trades')
    print('-' * 60)

    trades = get_recent_trades(market['condition_id'], days=30)
    print(f'Trades fetched: {len(trades)}')
    print(f'Schema: {trades.schema}')
    print(trades.head(3))