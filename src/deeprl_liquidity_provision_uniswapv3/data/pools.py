"""Pool registry for the Uniswap v3 panel.

Identities come from the authoritative inventory at
``~/Projects/defi-rv/data/pool_inventory.csv`` (geckoterminal-sourced), NOT from
inferring tokens off price magnitude: USDC and USDT both have 6 decimals, so
magnitude cannot tell them apart, and that inference misread the WBTC/USDT 0.30%
pool as WBTC/USDC.

``token0``/``token1`` here are the ON-CHAIN ordering (ascending token address),
which is what ``price = (sqrtPriceX96/2**96)**2 = token1_raw/token0_raw`` refers
to. The inventory's own token0/token1 columns are geckoterminal's base/quote and
do NOT follow that ordering, so they are not reused.

Token addresses that fix the ordering:
    WBTC 0x2260..  <  LINK 0x5149..  <  AAVE 0x7Fc6..  <  USDC 0xA0b8..
    <  WETH 0xC02a..  <  USDT 0xdAC1..   (UNI 0x1f98.. is below all of these)

The core panel is a balanced 3-pairs x 2-tiers factorial (tier varies within pair,
which is what identifies a fee-tier effect free of the asset confound). WBTC/USDT
would have made it 4 pairs but is excluded: both its pools are more empty hours
than data. Extras: a third tier level on WETH/USDT and three alt/WETH pools at a
fixed tier for a volatility contrast.
"""
from dataclasses import dataclass

DECIMALS = {"WETH": 18, "USDC": 6, "USDT": 6, "WBTC": 8, "LINK": 18, "AAVE": 18, "UNI": 18}


@dataclass(frozen=True)
class Pool:
    key: str
    address: str
    token0: str          # on-chain token0 (lower address)
    token1: str          # on-chain token1
    fee_tier: float      # fraction, e.g. 0.0005 for the 0.05% tier
    usd_side: str        # "token0" | "token1" | "weth": which leg denominates USD volume
    base: str            # the risky asset the LP position is denominated against
    group: str           # "core" (balanced factorial) | "extra_tier" | "alt"

    @property
    def dec0(self) -> int:
        return DECIMALS[self.token0]

    @property
    def dec1(self) -> int:
        return DECIMALS[self.token1]

    @property
    def fee_tier_pct(self) -> float:
        """Percent form, e.g. 0.05. Keys tick spacing. NEVER the fee rate."""
        return self.fee_tier * 100

    @property
    def pair(self) -> str:
        return f"{self.token0}/{self.token1}"


# fmt: off
POOLS = {
    # --- core: 4 pairs x 2 tiers, balanced ---
    "usdc_weth_005": Pool("usdc_weth_005", "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
                          "USDC", "WETH", 0.0005, "token0", "WETH", "core"),
    "usdc_weth_030": Pool("usdc_weth_030", "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
                          "USDC", "WETH", 0.0030, "token0", "WETH", "core"),
    "wbtc_weth_005": Pool("wbtc_weth_005", "0x4585fe77225b41b697c938b018e2ac67ac5a20c0",
                          "WBTC", "WETH", 0.0005, "weth",   "WBTC", "core"),
    "wbtc_weth_030": Pool("wbtc_weth_030", "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed",
                          "WBTC", "WETH", 0.0030, "weth",   "WBTC", "core"),
    "weth_usdt_005": Pool("weth_usdt_005", "0x11b815efb8f581194ae79006d24e0d814b7697f6",
                          "WETH", "USDT", 0.0005, "token1", "WETH", "core"),
    "weth_usdt_030": Pool("weth_usdt_030", "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36",
                          "WETH", "USDT", 0.0030, "token1", "WETH", "core"),
    # WBTC/USDT is DROPPED from the core: 52.9% and 22.1% of hours have no swaps at
    # all, and wbtc_usdt_005's median hourly volume is $0. Kept here as documentation
    # of the exclusion rather than deleted, so the panel's composition is auditable.
    "wbtc_usdt_005": Pool("wbtc_usdt_005", "0x56534741cd8b152df6d48adf7ac51f75169a83b2",
                          "WBTC", "USDT", 0.0005, "token1", "WBTC", "excluded_thin"),
    "wbtc_usdt_030": Pool("wbtc_usdt_030", "0x9db9e0e53058c89e5b94e29621a205198648425b",
                          "WBTC", "USDT", 0.0030, "token1", "WBTC", "excluded_thin"),

    # --- extra tier level: only WETH/USDT has a 0.01% pool (created Dec 2022,
    #     so it is unbalanced against the core and must not truncate it) ---
    "weth_usdt_001": Pool("weth_usdt_001", "0xc7bbec68d12a0d1830360f8ec58fa599ba1b0e9b",
                          "WETH", "USDT", 0.0001, "token1", "WETH", "extra_tier"),

    # --- alts at a fixed 0.30% tier: volatility contrast, no tier variation ---
    "link_weth_030": Pool("link_weth_030", "0xa6cc3c2531fdaa6ae1a3ca84c2855806728693e8",
                          "LINK", "WETH", 0.0030, "weth", "LINK", "alt"),
    "aave_weth_030": Pool("aave_weth_030", "0x5ab53ee1d50eef2c1dd3d5402789cd27bb52c1bb",
                          "AAVE", "WETH", 0.0030, "weth", "AAVE", "alt"),
    "uni_weth_030":  Pool("uni_weth_030",  "0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801",
                          "UNI",  "WETH", 0.0030, "weth", "UNI",  "alt"),
}
# fmt: on

# The pool whose price defines ETH/USD on the panel's own clock. Using the panel's
# own deepest USDC/WETH pool keeps ETH/USD consistent with the prices the env sees
# and avoids an external feed on a different clock.
ETH_USD_POOL = "usdc_weth_005"

CORE = [k for k, p in POOLS.items() if p.group == "core"]


def by_group(group: str) -> list[str]:
    return [k for k, p in POOLS.items() if p.group == group]
