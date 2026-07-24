import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deeprl_liquidity_provision_uniswapv3.experiments.retail_frontier import (  # noqa: E402
    adjusted, config_tag,
)


def test_cost_counterfactual_is_exact_linear_accounting():
    result = {"reward": 100.0, "n_rebalances": 3,
              "rebalance_notional": 1_000.0}
    assert adjusted(result, gas=5, rate=0.001) == 84.0


def test_frontier_tag_changes_with_design():
    a = config_tag(["p"], [1_500], [45, 50, 55])
    assert a != config_tag(["p"], [5_000], [45, 50, 55])
    assert a != config_tag(["q"], [1_500], [45, 50, 55])
