"""Aggregate swap-level Uniswap v3 parquet into an hourly panel.

Source: ``~/Projects/defi-rv/data/raw/ethereum/uniswap/<address>.parquet``, which
holds every Swap event with ``block_timestamp, price, amount0, amount1,
sqrtPriceX96, liquidity, tick``. That data is RAW and undecimalized (defi-rv's
fetcher defers decimal scaling), so every conversion here is asserted rather than
assumed: mis-scaled units are exactly the bug that sank the previous paper.

What this produces per pool, on a complete hourly grid:
    price        decimal-adjusted USD price of the base asset (env convention)
    price_raw    (sqrtPriceX96/2**96)**2, token1_raw/token0_raw. Needed to compute
                 our own liquidity in the SAME raw units the pool's `liquidity` uses.
    liquidity    raw in-range L at the last swap of the hour: the LP-share denominator
    volume_usd   summed over the hour
    fees_usd     volume_usd * fee_tier, the gross fee paid to ALL LPs in the hour
    n_swaps      swaps in the hour
    was_imputed  True if the hour had no swaps

Flows (volume, fees, n_swaps) are ZERO-filled on empty hours; state (price,
liquidity, tick) is forward-filled. Never the reverse: forward-filling volume
fabricates fee income, zero-filling price fabricates a crash.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .pools import ETH_USD_POOL, POOLS, Pool

DEFI_RV = Path.home() / "Projects/defi-rv/data/raw/ethereum/uniswap"
OUT_DIR = Path(__file__).resolve().parents[3] / "data/processed"
Q96 = 2 ** 96


def _read_swaps(pool: Pool) -> pd.DataFrame:
    path = DEFI_RV / f"{pool.address}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{pool.key}: no parquet at {path}")
    df = pd.read_parquet(
        path, columns=["block_number", "log_index", "block_timestamp", "price",
                       "amount0", "amount1", "sqrtPriceX96", "liquidity", "tick"]
    )
    # Chain order is (block_number, log_index), NOT block_timestamp: every swap in a
    # block shares one timestamp, and on this pool 65% of swaps sit in a tie group of
    # up to 251. Sorting on the timestamp alone with pandas' default quicksort (which
    # is not stable) leaves 41% of swaps out of true order. That is invisible to the
    # hourly panel, which only takes the hour's last row, but the swap-level fee model
    # chains each swap's price interval off its predecessor, so the order IS the data.
    return df.sort_values(["block_number", "log_index"], kind="stable", ignore_index=True)


def _price_adjusted(price_raw: pd.Series, pool: Pool) -> pd.Series:
    """Raw token1/token0 -> decimal-adjusted token1 per token0."""
    return price_raw * 10.0 ** (pool.dec0 - pool.dec1)


def _assert_sqrt_price_consistent(df: pd.DataFrame, pool: Pool) -> None:
    """`price` must equal (sqrtPriceX96/2**96)**2. Guards against a schema change."""
    s = df["sqrtPriceX96"].to_numpy()[:10_000]
    p = df["price"].to_numpy()[:10_000]
    ok = np.isfinite(s) & np.isfinite(p) & (s > 0)
    recomputed = (s[ok] / Q96) ** 2
    rel = np.abs(recomputed - p[ok]) / np.maximum(np.abs(p[ok]), 1e-30)
    bad = float(np.nanmax(rel)) if rel.size else 0.0
    assert bad < 1e-6, f"{pool.key}: price != (sqrtPriceX96/2^96)^2 (max rel err {bad:.2e})"


def _assert_sides_agree(df: pd.DataFrame, pool: Pool) -> float:
    """The two token legs of a swap must value to the same amount, up to the fee.

    amount0_dec * price_adj == amount1_dec by construction of the pool price, so a
    gross mismatch means wrong decimals or a flipped token0/token1.

    The residual disagreement is not noise: the trader pays the fee on the way in,
    so the median leg gap RECOVERS THE FEE TIER from the raw data. That makes this
    a free, independent check that `fee_tier` in the registry is the tier the pool
    actually charges -- the exact confusion (percent vs fraction) that invalidated
    the previous paper. Returns the median relative disagreement.
    """
    sample = df.iloc[:: max(1, len(df) // 20_000)]
    a0 = sample["amount0"].to_numpy() / 10.0 ** pool.dec0
    a1 = sample["amount1"].to_numpy() / 10.0 ** pool.dec1
    padj = _price_adjusted(sample["price"], pool).to_numpy()
    implied = np.abs(a0 * padj)
    actual = np.abs(a1)
    ok = (actual > 0) & np.isfinite(implied) & np.isfinite(actual)
    rel = np.abs(implied[ok] - actual[ok]) / actual[ok]
    med = float(np.nanmedian(rel))

    assert med < 0.05, (
        f"{pool.key}: token legs disagree by {med:.1%} at the median -> wrong decimals "
        f"or token0/token1 ordering for {pool.token0}({pool.dec0})/{pool.token1}({pool.dec1})"
    )
    # The gap should sit at the fee tier. Loose bound: swaps that cross ticks or
    # route partially leave a wider tail, so this catches a tier off by 5x or 100x,
    # not a few basis points.
    lo, hi = pool.fee_tier * 0.4, pool.fee_tier * 2.5 + 1e-4
    assert lo <= med <= hi, (
        f"{pool.key}: median leg gap {med:.4%} does not match declared fee_tier "
        f"{pool.fee_tier:.4%}. The data says the pool charges ~{med:.4%}."
    )
    return med


def _hourly(df: pd.DataFrame, pool: Pool) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = (df["block_timestamp"] // 3600) * 3600

    # token amounts in human units; volume is the absolute size of the swap
    df["amt0"] = df["amount0"].abs() / 10.0 ** pool.dec0
    df["amt1"] = df["amount1"].abs() / 10.0 ** pool.dec1

    g = df.groupby("hour", sort=True)
    out = g.agg(
        price_raw=("price", "last"),
        sqrt_price_x96=("sqrtPriceX96", "last"),
        liquidity=("liquidity", "last"),
        tick=("tick", "last"),
        n_swaps=("price", "size"),
        vol_token0=("amt0", "sum"),
        vol_token1=("amt1", "sum"),
    )
    return out


def _complete_grid(h: pd.DataFrame) -> pd.DataFrame:
    """Reindex to every hour. Forward-fill state, zero-fill flows."""
    full = np.arange(h.index.min(), h.index.max() + 3600, 3600)
    out = h.reindex(full)
    out["was_imputed"] = out["n_swaps"].isna()

    state = ["price_raw", "sqrt_price_x96", "liquidity", "tick"]
    out[state] = out[state].ffill()
    flows = ["n_swaps", "vol_token0", "vol_token1"]
    out[flows] = out[flows].fillna(0.0)

    out.index.name = "hour"
    return out


def build_pool(key: str, eth_usd: pd.Series | None = None) -> pd.DataFrame:
    pool = POOLS[key]
    swaps = _read_swaps(pool)
    _assert_sqrt_price_consistent(swaps, pool)
    med_err = _assert_sides_agree(swaps, pool)

    h = _complete_grid(_hourly(swaps, pool))
    padj = _price_adjusted(h["price_raw"], pool)  # token1 per token0

    # USD price of the base asset, in the env's convention.
    if pool.base == pool.token0:
        base_in_t1 = padj                      # token1 per token0
        quote = pool.token1
    else:
        base_in_t1 = 1.0 / padj                # token0 per token1
        quote = pool.token0

    if quote in ("USDC", "USDT"):
        h["price"] = base_in_t1                # quote is a dollar already
    elif quote == "WETH":
        if eth_usd is None:
            raise ValueError(f"{key}: needs ETH/USD to price a {quote}-quoted pool")
        h["price"] = base_in_t1 * eth_usd.reindex(h.index).ffill().bfill()
    else:
        raise ValueError(f"{key}: cannot denominate quote {quote} in USD")

    # USD volume: take the leg that is already dollars where one exists, else the
    # WETH leg priced with ETH/USD from the panel's own deepest USDC/WETH pool.
    if pool.usd_side == "token0":
        h["volume_usd"] = h["vol_token0"]
    elif pool.usd_side == "token1":
        h["volume_usd"] = h["vol_token1"]
    elif pool.usd_side == "weth":
        weth_leg = "vol_token1" if pool.token1 == "WETH" else "vol_token0"
        if eth_usd is None:
            raise ValueError(f"{key}: needs ETH/USD for a WETH-denominated pool")
        h["volume_usd"] = h[weth_leg] * eth_usd.reindex(h.index).ffill().bfill()
    else:
        raise ValueError(f"{key}: bad usd_side {pool.usd_side}")

    # Gross fees paid by traders to ALL LPs this hour. The LP's own take is this
    # times its in-range liquidity share, applied in the environment.
    h["fees_usd"] = h["volume_usd"] * pool.fee_tier

    # USD price of each leg, so the environment values a position without having to
    # re-derive which token is the numeraire. Getting this wrong is how a position
    # value silently drifts by a factor of the price.
    def usd_of(token: str) -> pd.Series | float:
        if token == pool.base:
            return h["price"]
        if token in ("USDC", "USDT"):
            return 1.0
        if token == "WETH":
            return eth_usd.reindex(h.index).ffill().bfill()
        raise ValueError(f"{key}: cannot value {token} in USD")

    h["token0_usd"] = usd_of(pool.token0)
    h["token1_usd"] = usd_of(pool.token1)

    h["timestamp"] = pd.to_datetime(h.index, unit="s", utc=True)
    cols = ["timestamp", "price", "price_raw", "sqrt_price_x96", "liquidity", "tick",
            "volume_usd", "fees_usd", "n_swaps", "was_imputed", "token0_usd", "token1_usd"]
    out = h[cols].copy()

    assert out["price"].gt(0).all(), f"{key}: non-positive price after adjustment"
    assert out["liquidity"].ge(0).all(), f"{key}: negative liquidity"
    assert not out.loc[out.was_imputed, "volume_usd"].gt(0).any(), \
        f"{key}: imputed hour carries volume -> flows were forward-filled"
    out.attrs["median_leg_disagreement"] = med_err
    return out


def eth_usd_series() -> pd.Series:
    """ETH/USD hourly from the panel's own deepest USDC/WETH pool."""
    pool = POOLS[ETH_USD_POOL]
    swaps = _read_swaps(pool)
    h = _complete_grid(_hourly(swaps, pool))
    # USDC/WETH: padj = WETH per USDC, so USD per WETH is its reciprocal.
    return 1.0 / _price_adjusted(h["price_raw"], pool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=None, help="pool keys (default: all)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    keys = args.pools or list(POOLS)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"ETH/USD from {ETH_USD_POOL} ...")
    eth = eth_usd_series()
    print(f"  {len(eth):,} hours, ${eth.min():,.0f} to ${eth.max():,.0f}\n")

    print(f"{'pool':<16} {'hours':>8} {'imputed':>8} {'legs':>7} "
          f"{'from':<11} {'to':<11} {'med vol/h':>12} {'fees/h':>9}")
    print("-" * 92)
    for key in keys:
        df = build_pool(key, eth_usd=eth)
        path = args.out / f"{key}_hourly.parquet"
        df.to_parquet(path, index=False)
        imp = df.was_imputed.mean()
        print(f"{key:<16} {len(df):>8,} {imp:>7.1%} "
              f"{df.attrs['median_leg_disagreement']:>6.2%} "
              f"{df.timestamp.min():%Y-%m-%d}  {df.timestamp.max():%Y-%m-%d} "
              f"{df.volume_usd.median():>12,.0f} {df.fees_usd.median():>9,.2f}")
    print(f"\nwrote {len(keys)} pools to {args.out}")


if __name__ == "__main__":
    main()
