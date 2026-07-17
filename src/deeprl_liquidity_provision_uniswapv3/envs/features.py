"""Feature sets for the observation.

Two are available so the negative RL result cannot be blamed on the state:

`legacy` replicates the rejected paper's 13 features exactly, including the TA-Lib
indicators it fed the agent and the 12-hour rolling candles it synthesised for them:
    price, tick, width, liquidity, sigma, ma24, ma168,
    bb_upper, bb_middle, bb_lower, ADXR, BOP, DX

`compact` is the rebuilt set: returns and volatility instead of the price level,
plus the position state (in_position, in_range, where the price sits in the band,
liquidity share) and the fee signal (fee rate, fee per unit of pool liquidity) that
the legacy state never contained.

The legacy state is not strictly larger: it has six technical indicators the compact
set lacks, and lacks the position and fee-income signals the compact set has. Which
is better is an empirical question, so both are run.

One faithful detail worth keeping: the old env's `reset()` put a literal 1 where
`step()` put sigma, so its first observation carried a constant in the volatility
slot. Replicated under `legacy` rather than silently corrected, because the point is
to reproduce the state the agent actually saw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from talib import MA_Type

LEGACY_NAMES = ["price", "tick", "width", "liquidity", "sigma", "ma24", "ma168",
                "bb_upper", "bb_middle", "bb_lower", "adxr", "bop", "dx"]


def rolling_candles(price: np.ndarray, interval: int = 12):
    """The old env's 12-hour rolling candles, rebuilt vectorised.

    open = price `interval-1` bars back, high/low = extremes over the window,
    close = current. Not real candles: a rolling view of a close-only series.
    Our swap data could give true hourly OHLC, but the point here is to reproduce
    the state the previous agent saw.
    """
    s = pd.Series(price)
    w = interval - 1
    op = s.shift(w).fillna(s.iloc[0]).to_numpy()
    hi = s.rolling(interval, min_periods=1).max().to_numpy()
    lo = s.rolling(interval, min_periods=1).min().to_numpy()
    cl = price
    return op, hi, lo, cl


def legacy_series(price: np.ndarray):
    """Precompute the legacy indicators over the whole panel segment."""
    op, hi, lo, cl = rolling_candles(price, 12)
    p = np.asarray(price, dtype=float)
    bb_u, bb_m, bb_l = talib.BBANDS(p, matype=MA_Type.T3)
    out = dict(
        bb_upper=bb_u, bb_middle=bb_m, bb_lower=bb_l,
        adxr=talib.ADX(hi, lo, cl, timeperiod=14),
        bop=talib.BOP(op, hi, lo, cl),
        dx=talib.DX(hi, lo, cl, timeperiod=14),
    )
    # TA-Lib emits NaN over its warmup; the old env sliced them off with a warmup
    # cut. Zero-fill here and rely on the env's own warmup to skip that region.
    return {k: np.nan_to_num(v, nan=0.0) for k, v in out.items()}
