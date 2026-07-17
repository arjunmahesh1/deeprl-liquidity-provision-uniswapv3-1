"""Swap-level fee attribution: the flow the hourly panel throws away.

The hourly panel answers "where was the price at the end of the hour"; it cannot
answer "was our band in range while that swap executed". Those differ by a lot. On
`usdc_weth_005` a band crosses in and out many times within one hour, so crediting
the whole hour's fees on an hour-boundary range check charges zero for hours we were
in range for most of, and the error grows as the band narrows. It biases against
concentration, which is the strategy the paper is about.

This module builds the artifact that removes the discretization. Per swap:

    hour        the panel row the swap belongs to
    sqrt_lo     lower end of the sqrt-price interval the swap traversed
    sqrt_hi     upper end of the same interval, RAW (not X96)
    liquidity   the pool's active liquidity, raw, at that swap
    fee_usd     the fee that swap paid to all LPs, in USD
    token0_in   True if token0 was the input leg, which sets the measure the fee is
                apportioned in when a band covers only part of the interval

A Swap event reports the state AFTER the swap, so the interval a swap traversed is
[previous swap's sqrt price, this swap's sqrt price]. A swap therefore has extent in
price, not just a location, and a band that covers only part of that extent earns
only part of the fee.

The fee is taken from the swap's INPUT token, on the GROSS input: Uniswap v3's
`SwapMath.computeSwapStep` scales the remaining input by `(1e6 - feePips)/1e6` and
keeps the difference, so `feeAmount / gross_input == feePips / 1e6` exactly. The
input leg is the one whose signed amount is positive (tokens moving into the pool),
and the event's amount for that leg is the gross figure, because the fee stays in
the pool. Taking the fee off the output leg instead would understate it by the fee
itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import DEFI_RV, OUT_DIR, Q96, _read_swaps, eth_usd_series
from .pools import POOLS, Pool

SWAP_COLS = ["hour", "sqrt_lo", "sqrt_hi", "liquidity", "fee_usd", "token0_in"]


def epoch_seconds(ts: pd.Series) -> np.ndarray:
    """Epoch seconds from a datetime column, whatever its resolution.

    The panel is written as datetime64[ms], so an `astype("int64") // 10**9` reads
    the year 1620 instead of 2021 and every lookup silently misses. Convert through
    the dtype rather than guessing the divisor.
    """
    v = pd.to_datetime(ts, utc=True).dt.tz_localize(None)
    return v.to_numpy().astype("datetime64[s]").astype("int64")


def build_swaps(key: str, panel: pd.DataFrame) -> pd.DataFrame:
    """Per-swap fee and price-interval, aligned to `panel`'s hourly clock.

    `panel` supplies `token0_usd` / `token1_usd`, so a swap is valued with exactly
    the same prices the environment uses to value a position. Pulling a second price
    source in here would let the fee and the position disagree about what a token is
    worth, which is the class of bug this rebuild exists to kill.
    """
    pool: Pool = POOLS[key]
    sw = _read_swaps(pool)

    sqrt_after = sw["sqrtPriceX96"].to_numpy(float) / Q96
    # The Swap event reports post-swap state, so the interval traversed runs from the
    # previous swap's price to this one's. The first swap has no predecessor: give it
    # zero extent rather than inventing a move.
    sqrt_before = np.empty_like(sqrt_after)
    sqrt_before[0] = sqrt_after[0]
    sqrt_before[1:] = sqrt_after[:-1]

    hour = (sw["block_timestamp"].to_numpy() // 3600) * 3600
    a0 = sw["amount0"].to_numpy(float)
    a1 = sw["amount1"].to_numpy(float)

    # Value each leg in USD on the panel's own clock and prices. The panel is a
    # complete hourly grid, so every swap's hour is present exactly once and
    # searchsorted is an exact lookup rather than a nearest-match.
    p_hours = epoch_seconds(panel["timestamp"])
    at = np.searchsorted(p_hours, hour)
    assert at.max() < len(p_hours) and (p_hours[at] == hour).all(), \
        f"{key}: swap hours fall outside the panel's grid"
    t0 = panel["token0_usd"].to_numpy(float)[at]
    t1 = panel["token1_usd"].to_numpy(float)[at]
    usd0 = np.abs(a0) / 10.0 ** pool.dec0 * t0
    usd1 = np.abs(a1) / 10.0 ** pool.dec1 * t1

    # Fee on the gross input leg: the positive amount is what entered the pool.
    gross_in_usd = np.where(a0 > 0, usd0, usd1)
    fee_usd = gross_in_usd * pool.fee_tier

    lo = np.minimum(sqrt_before, sqrt_after)
    hi = np.maximum(sqrt_before, sqrt_after)

    out = pd.DataFrame({"hour": hour, "sqrt_lo": lo, "sqrt_hi": hi,
                        "liquidity": sw["liquidity"].to_numpy(float),
                        "fee_usd": fee_usd, "token0_in": a0 > 0})
    out = out[np.isfinite(out[["sqrt_lo", "sqrt_hi", "liquidity", "fee_usd"]]).all(axis=1)]

    assert (out["sqrt_hi"] >= out["sqrt_lo"]).all(), f"{key}: swap interval inverted"
    assert (out["fee_usd"] >= 0).all(), f"{key}: negative swap fee"
    assert out["hour"].is_monotonic_increasing, f"{key}: swaps not sorted by time"
    return out.reset_index(drop=True)


def check_against_hourly(key: str, swaps: pd.DataFrame, panel: pd.DataFrame) -> float:
    """Per-swap fees must re-aggregate to the hourly panel's `fees_usd`.

    Not a tautology: the panel takes the fee off `volume_usd`, which is one leg's
    absolute size regardless of direction, while this takes it off the gross input
    leg. They agree only if the two legs of a swap differ by about the fee, which is
    the same fact `aggregate._assert_sides_agree` uses to recover the tier from the
    raw data. A large gap means the two disagree about which side pays, so the check
    is worth its cost. Returns the median relative gap over hours with real flow.
    """
    per_hour = swaps.groupby("hour")["fee_usd"].sum()
    ref = panel.set_index(epoch_seconds(panel["timestamp"]))["fees_usd"]
    both = pd.concat([per_hour.rename("swap"), ref.rename("panel")], axis=1).dropna()
    both = both[both["panel"] > 0]
    rel = ((both["swap"] - both["panel"]).abs() / both["panel"])
    med = float(rel.median())
    assert med < 0.05, (
        f"{key}: per-swap fees differ from the hourly panel by {med:.1%} at the "
        f"median. The two disagree about which leg pays the fee."
    )
    return med


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    keys = args.pools or list(POOLS)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{'pool':<16} {'swaps':>12} {'swaps/h':>9} {'vs hourly':>10} {'MB':>7}")
    print("-" * 60)
    for key in keys:
        panel_path = args.out / f"{key}_hourly.parquet"
        if not panel_path.exists():
            raise FileNotFoundError(f"{key}: build the hourly panel first ({panel_path})")
        panel = pd.read_parquet(panel_path)
        sw = build_swaps(key, panel)
        gap = check_against_hourly(key, sw, panel)
        path = args.out / f"{key}_swaps.parquet"
        sw.to_parquet(path, index=False)
        print(f"{key:<16} {len(sw):>12,} {len(sw)/len(panel):>9.1f} "
              f"{gap:>9.2%} {path.stat().st_size/1e6:>6.0f}")
    print(f"\nwrote {len(keys)} pools to {args.out}")


if __name__ == "__main__":
    main()
