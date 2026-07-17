# Active liquidity provision in Uniswap v3

## Research objective

Given that a liquidity provider must quote, does active range management reduce the
loss relative to passive provision, and can a learned policy do it better than the
heuristics practitioners use?

The benchmark is passive LP, not holding. A market maker's business is providing
liquidity, so "withdraw and hold" is not an available strategy. Without that
constraint the problem is degenerate: within a window, fees and impermanent loss
both scale with liquidity at a near-constant ratio, so reward is linear in exposure
with a negative coefficient and the optimum is to provide nothing. That is the
loss-versus-rebalancing result, and it is not what this paper is about.

## Background

An LP cannot eliminate impermanent loss. The claim under test is mitigation.

## Method

Uniswap v3 concentrated liquidity as a Markov Decision Process. The LP holds a
position over a tick range and chooses, at each decision point, whether to hold or
redeploy at one of several widths. Reward is realized profit and loss against a
hold benchmark: the position's share of the pool's actual fee income while in
range, less the change in impermanent loss, less gas and swap costs.

Fee income is the LP's share of real hourly pool fees, `fees_usd * L / (pool_L + L)`,
earned only while in range. This replaces a price-displacement fee model that
captured arbitrage flow only and understated fee income to ~1.5% of impermanent
loss, and it retires the sole-provider assumption in the same expression.

## Data

- Sources: swap-level Uniswap v3 events, Ethereum mainnet, aggregated hourly.
  See `data/raw/PROVENANCE.md`.
- Time period: 2021-05-05 to 2026-06-19. ~44,900 hours per pool.
- Frequency: hourly.
- Panel: 3 asset pairs x 2 fee tiers, balanced. USDC/WETH, WETH/USDT, WBTC/WETH,
  each at 0.05% and 0.30%. Tier varies within pair, which identifies a fee-tier
  effect free of the asset confound.
- Excluded: WBTC/USDT, both tiers (52.9% and 22.1% of hours have no swaps; the
  0.05% pool's median hourly volume is $0). Recorded in the pool registry.
- Variables: see `data/DATA_DICTIONARY.md`.

## Pre-registration (frozen before touching test data)

- **Date frozen:** 2026-07-16
- **Hypotheses:**
  - H1. Active range management mitigates the LP loss relative to passive
    provision, on the core panel.
  - H2. A learned policy (PPO) mitigates the loss by more than the best competitor
    strategy selected under the same protocol.
  - H3. Mitigation varies with the fee tier, identified within asset pair.
  - H4. A policy transfers across pools, and transfer degrades less across asset
    pairs at a fixed fee tier than across fee tiers within a fixed pair. The 3x2
    panel identifies this: USDC/WETH 0.05% -> 0.30% holds the asset fixed and varies
    the tier; USDC/WETH 0.05% -> WBTC/WETH 0.05% holds the tier fixed and varies the
    asset. The previous paper claimed the target pool's fee regime sets the ceiling
    rather than asset similarity, but its two-pool design confounded the two and its
    transfer numbers did not depend on the source policy.
- **Train / out-of-sample split (exact):** windows of 1,500 hours, taken in
  chronological order per pool, non-overlapping. First 50% train, next 25%
  validation, final 25% test. Disjoint by construction and asserted in code
  (`experiments/bakeoff.py::Split`). Test is read once, after selection.
- **Primary metric (single scalar + direction):** mean realized reward per test
  window, in USD, on a $30,000 position. Higher is better. Reported per pool and
  pooled.
- **Independent unit:** the window. Seeds are averaged within a window before any
  test. Five seeds on one price trajectory are not five observations.
- **Statistical test:** Wilcoxon signed-rank, paired by window, given heavy tails.
  Seeded bootstrap confidence intervals. Holm correction across the competitor
  family within each pool.
- **Stopping rule:** one test read per configuration. If a configuration is changed
  after a test read, the change is dated and justified here, and the result is
  reported as exploratory.

### Choices fixed before the test read

Each of these was selected by inspecting training or validation results, so each is
frozen here to keep the test read honest.

- **Decision frequency is a property of the STRATEGY, not the environment.** "I
  rebalance hourly" is a choice a strategy makes. Forcing one schedule on everything
  destroys what distinguishes the competitors, since `ReactiveRecentering` fires on
  a volatility or jump signal, `VolProportionalWidth` recomputes a width, and
  `Passive` never acts. Each competitor is a complete specification including its
  own timing, so **competitors run their native logic on the hourly environment and
  decide for themselves whether to act.**
  A learned agent has no native timing, so one is declared: **daily (every 24h) is
  the primary**, with hourly, weekly, and event-driven as sensitivity arms.
  Daily is not the most favorable choice. On validation, PPO scores -6,502 hourly,
  -3,791 daily, -1,509 weekly, -1,965 event-driven, so daily concedes roughly 2,300
  against weekly. It is chosen because it is the realistic operating frequency for
  an LP: a claim that holds at a realistic cadence is worth more than one that needs
  a weekly cadence to survive. The sensitivity arms are reported whatever they say.
- **Action grid: widths {100, 200, 500, 2000} ticks, plus hold.** The previous
  grid `{0, 45, 50, 55}` was itself selected by maximizing test reward and offers
  only sub-1% bands, in range ~1.2% of the time.
- **Algorithms:** PPO, A2C, DQN. All on the same `Discrete` space and the same
  budget.
- **Training budget: 20,000 agent steps.** Fixed from a convergence check on
  VALIDATION, not chosen to flatter a result: PPO plateaus from 20k to 100k at
  about -1,650 (10k -2,019; 20k -1,671; 50k -1,640; 100k -1,693; 200k -1,839), so
  more training does not help and slightly hurts. 20k is ~22 passes over the train
  span. Recorded because an unstated budget would leave every agent result open to
  "it was under-trained".
- **Agent config grid:** learning rate in {3e-4, 1e-3} x entropy coefficient in
  {0, 0.01}, selected on validation. Deliberately small. The previous paper gave PPO
  10 Optuna trials plus its choice of action space while `ILMinimizer` had no free
  parameters at all; a sprawling agent grid would make "matched budget" a fiction in
  the opposite direction.
- **Competitor set:** the paper's four (PassiveWidthSweep, VolProportionalWidth,
  ILMinimizer, ReactiveRecentering) plus RecentreWhenOut and Passive. Original
  functional forms and parameter grids. Matched selection budget.
- **Capital: $30,000. Gas: flat $5.** Both load-bearing and neither yet defended;
  see LIMITATIONS.md. Sensitivity to both is a robustness exhibit, not a headline.

## Status: verified vs to-confirm

**Verified (tests actually run):**
- NOTE: the H1 numbers below were produced with the competitors consulted hourly
  while PPO was event-driven. They stand as competitor-vs-passive results, but the
  competitor-vs-PPO comparison they imply is void and is being re-run with the
  frequency applied uniformly.
- Fee tier recovered independently from the raw swap data: the median disagreement
  between a swap's two token legs equals the fee tier, on all 12 pools.
- Environment: 17 tests, three verified by mutation (reintroducing the 100x fee
  bug, the sole-provider fee, and the value-manufacturing rebalance each fail the
  test written for them).
- H1 on the competitor set, clean test read: mitigation +2,104 to +3,231 on the four
  WETH-against-stablecoin pools; +335 and +366 on the two WBTC/WETH pools.

- **H1 SUPPORTED.** Clean test read, 48 windows, 6 pools. Active management mitigates
  **+1,754 pooled** (passive -3,151, best competitor -1,397), 56% of the loss.
- **H2 REJECTED for PPO.** Clean test read, one read, budget fixed from a validation
  convergence check. Pooled: PPO -3,137 against competitor -1,397 and passive -3,151.
  **PPO is indistinguishable from doing nothing (+15 over 48 windows)** and loses to
  the best competitor by 1,740. Failure mode differs by pool: on `wbtc_weth_030` it
  learns to never act (seed spread 0, identical to passive); on `weth_usdt_030` it
  acts and is 852 WORSE than passive. Not a budget artifact (converged at 20k), not
  a frequency artifact (daily beats hourly), not a leak artifact (both sides selected
  on validation).
- **H3 SUPPORTED, with the sign REVERSED against the previous paper.** Mitigation by
  pair and tier:

  | pair | 0.05% | 0.30% | tier effect |
  |---|---|---|---|
  | USDC/WETH | +2,104 | +3,231 | +1,127 |
  | WETH/USDT | +2,191 | +2,302 | +111 |
  | WBTC/WETH | +366 | +335 | -31 |

  The **asset pair dominates and the fee tier barely matters**: a 6x gap across
  pairs against a noisy, sign-flipping effect across tiers. The previous abstract
  claimed "the target pool's economic regime, rather than asset similarity, sets the
  absolute reward ceiling". Its two-pool design varied asset and tier together and
  could not have separated them. Caveat: 3 pairs, so the tier effect is weakly
  powered; the pair effect is not.

### OPEN DEFECT, highest priority: hourly fee discretization biases against concentration

The environment checks whether we are in range at the END of each hour and, if so,
credits the WHOLE hour's aggregated fees:

    in_range = sqrtA <= sqrtP[j] <= sqrtB          # hour boundary only
    fee = fees_usd[j] * share if in_range else 0

`usdc_weth_005` sees ~250 swaps/hour, so a narrow band crosses in and out many times
within one hour. Any hour ending out of range earns zero even if we were in range
for most of it. The penalty grows as the band narrows, so the discretization
**systematically penalises concentration**, which is the strategy the paper is about.

This is knife-edge, not a rounding issue. Scaling fee income only, passive LP at a
fixed width, medians over 8 windows on `usdc_weth_005`:

| fee multiplier | best width | fees/IL at +/-0.5% |
|---|---|---|
| **1x (as built)** | **+/-170%, degenerate** | 0.69 |
| **2x** | **+/-5%, realistic** | 1.37 |
| 5x | +/-5% | 3.43 |
| 10x+ | +/-2% | 6.86 |

**Doubling fee income flips the optimum from "barely participate" to a realistic
concentrated position.** Every conclusion below rests on our fee income being right
within a factor of two, and there is a known bias in the direction that would flip it.

**The fix, with data already on disk:** compute the LP's fee PER SWAP, not per hour.
`~/Projects/defi-rv/data/raw/ethereum/uniswap/<addr>.parquet` carries every swap's
own `sqrtPriceX96`, `liquidity`, `amount0/1` and `tick`, so in-range status and the
liquidity share can be evaluated at each swap instead of interpolated across an hour.
That removes the discretization rather than correcting for it.

**Until this lands, treat every "RL loses" result below as provisional.**

**To-confirm (assumed, not yet run):**
- Everything below is conditional on the fee-discretization defect above.

### Results at REALISTIC widths (the wide band is degenerate)

A +/-2000-tick band is -18% to +22%, i.e. approximately a v2 position, and it wins by
avoiding IL rather than earning fees (fees fall 4,177 -> 829 across widths while IL
falls 6,223 -> 1,299). No LP would deploy it, so it is a flawed benchmark. Capping
widths to {100, 200, 500} ticks = +/-1%, +/-2%, +/-5%, which is what real WETH/USDC v3
LPs quote:

| strategy | val | TEST | mitigation | vs best | p (Holm) |
|---|---|---|---|---|---|
| RecentreWhenOut(500) | -1,972 | -2,314 | +837 | - | - |
| ILMinimizer | -1,855 | -2,394 | +758 | -79 | 0.885 |
| VolProportionalWidth | -1,842 | -2,469 | +682 | -155 | 1.000 |
| **PPO (event-driven)** | **-1,623** | -2,743 | +408 | -429 | **0.426** |
| Passive | -1,140 | -3,151 | 0 | -837 | 0.304 |
| PassiveWidthSweep | -1,028 | -3,175 | -23 | -860 | 0.343 |
| DQN | -1,270 | -3,181 | -29 | -866 | 0.293 |
| A2C | -1,074 | -3,360 | -209 | -1,046 | 0.069 |
| ReactiveRecentering | -5,607 | -6,034 | -2,883 | -3,720 | **0.000** |

**After Holm correction NOTHING is distinguishable except ReactiveRecentering being
bad.** PPO trails the best rule by 429 at p=0.43, winning 42% of 48 windows. So "RL
loses" is NOT established at realistic widths; we are underpowered to rank anything.
Mitigation also drops from +2,322 (wide) to +837 (realistic), so the 74% headline was
mostly the degenerate band.

Note PPO has the BEST validation score of all nine (-1,623) and finishes 4th on test.
Heuristics move -40 to +203 from validation to test; every RL algorithm drops 1,100
to 1,850. Simple rules cannot overfit 14 training windows; a neural policy can.

### Features: the previous paper's state is BETTER, and it does not matter

The legacy 13-feature state (price, tick, width, liquidity, sigma, ma24, ma168,
Bollinger x3, ADXR, BOP, DX, on 12h rolling candles) replicated exactly in
`envs/features.py`, selectable via `features="legacy"`:

| features | pooled val | pooled TEST | vs passive |
|---|---|---|---|
| compact (rebuilt) | -1,489 | -2,689 | +462 |
| legacy (the paper's) | -1,485 | -2,308 | +843 |

Legacy is worth +381 to PPO, so the rebuilt state was a mild handicap. Validation is
identical (-1,489 vs -1,485), so legacy does not fit better, it GENERALISES better.
Does not close the gap to the rules. Note the legacy state never contained a fee
signal, because the old fee model made fees a deterministic function of price.

### The rejected paper's result, decomposed and reproduced

On `usdc_weth_005`, isolating each ingredient of the old protocol:

| | |
|---|---|
| PPO, config chosen on validation (honest) | -2,913 |
| PPO, per-seed max on TEST (their exact code) | -2,238 |
| -> inflation bought by the leak | **+675** |
| Best competitor, full action grid | -1,457 (ILMinimizer(H=24,out)) |
| Best competitor, THEIR grid {45,50,55} | -3,596 (PassiveWidthSweep(w=50)) |
| -> handicap imposed on the competitors | **-2,140** |
| **Honest gap** (PPO - competitor) | **-1,456**, PPO loses |
| **Their gap** (leaked PPO - crippled competitor) | **+1,358**, PPO wins |

A 2,815 swing flips the sign. The dominant term is the ACTION GRID, not the leak:
{45,50,55} ticks are +/-0.5% bands, in range ~1% of the time. And that grid was itself
chosen by Optuna maximising test reward, so the leak selected the handicap.
- H4. Not yet tested. The previous transfer result is void: its "same-period" test
  set was a superset of its training set, its reward was an episode sum compared
  across segments of different length, and its saved outputs repeat bit-for-bit
  across different source pools, so the reported 88.5% did not depend on the
  transfer.
- The WBTC/WETH validation-to-test collapse (+302 validation, -1,852 test) is
  unexplained. Those two rows are not trustworthy until it is.
- Gas is a flat constant across 2021 to 2026, when it ran 100+ gwei early and
  collapsed after Dencun. Directionally favors rebalancing in the early sample.

## Milestones

- [x] Panel built and validated
- [x] Environment rebuilt with unit tests
- [x] Pre-registration frozen
- [x] Competitors under the leak-proof protocol (H1)
- [x] PPO under the same protocol (H2): rejected
- [ ] A2C and DQN under the same protocol (H2)
- [ ] Window-level paired statistics
- [x] Fee-tier effect within pair (H3): supported, sign reversed vs the previous paper
- [ ] Cross-pool transfer on the 3x2 design (H4)
- [ ] Paper draft
