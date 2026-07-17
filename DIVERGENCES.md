# What this rebuild does NOT follow from the previous paper

Established by three independent audits (MDP, protocol, data) reading the paper's
`.tex`, the original `rl-code/`, and this rebuild side by side. Every claim below was
verified against the source rather than taken from `SPEC.md`, which was wrong in
several places and is corrected here.

Two categories, and the difference matters:

- **A. We changed the problem.** Comparison to the published numbers is void.
- **B. We fixed a real defect.** Divergence is deliberate; the paper's number was wrong.

A third category, **C**, is where the *paper* diverges from *its own code*, which
neither of us can follow because there is nothing consistent to follow.

---

## A. We changed the problem (unjustified or undocumented)

### A1. The agent searched ONE configuration while the heuristics searched 48

`rolling.py` hardcoded `learning_rate=3e-4, ent_coef=0.01`, computed a validation
score, and **discarded it**. Per arm under `--widths 100 200 500`: Passive 1,
PassiveWidthSweep 3, RecentreWhenOut 3, ILMinimizer 4, VolProportionalWidth 10,
**ReactiveRecentering 27**, agent **1**. Three docstrings asserted the budgets were
matched.

The previous paper had the mirror-image bias: 10 Optuna trials for PPO over a
1,260-point space against `ILMinimizer` with no free parameters. Replacing one rigged
comparison with its reflection is not a fix. **Every "PPO significantly loses" number
this repo has produced was measured under this.**

**FIXED**: `AGENT_GRID` of 8 configs, selected on validation, with tests pinning that
the agent's budget sits inside the heuristics' range. Needs a re-run.

### A2. The agent's data stopped one window before its test; the heuristics' did not

The agent fit on `train` (windows i..i+3, 6,000h) while every heuristic selected on
`val` (window i+4), i.e. right up to the test boundary. The previous paper fit on all
five windows (7,500h). So the agent lost 20% of its fitting data AND was pushed one
window further from test, under this project's own diagnosis that a regime shift
between selection and test is what breaks a learned policy.

**FIXED**: select the config on validation, then refit on train+val (7,500h, matching
the original) before the single test read.

### A3. The paper's policy network is gone

Original: `CustomMLPFeatureExtractor` = `BatchNorm1d(obs_dim, affine=False)` → Linear
→ tanh → Linear → tanh → Linear(128), with `dim_hidden_layers=[4,2]` Optuna-selected.
Rebuild: plain `"MlpPolicy"`, SB3 default `[64,64]`, **no BatchNorm**.

The BatchNorm is load-bearing: the paper's state carries raw prices (~2,000), raw
liquidity (~1e18), and unscaled TA-Lib indicators. Feeding that to a net with no input
normalization is a different agent. **This is why the `legacy` features ablation
(SPEC's "+381") did not measure the paper's agent.**

**FIXED**: `agents/extractors.py::PaperMLPExtractor`, reachable via
`--paper-extractor`. Faithful down to the original indexing only `hidden_dim[0]` and
`[1]`, so a 3-layer config from its own search space silently built 2 layers.

### A4. Early stopping is gone

Original: `StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5)`
+ `EvalCallback(eval_freq=len//3)`, so the budget was data-dependent and capped at
100k. Rebuild: flat `learn(total_timesteps=20_000)`. **Not fixed.** SPEC justifies
20k on a convergence check but never states it replaces an early-stopping rule.

### A5. PPO's coefficients are SB3 defaults, not the paper's

Original: γ=0.999, λ=0.9999, clip=0.2, vf=0.1, ent=1e-4, target_kl=0.3, n_steps=len/3
= 2444, batch=256. Rebuild: γ=0.99, λ=0.95, vf=0.5, target_kl=None, n_steps=2048,
batch=64. γ=0.99 gives an effective horizon ~100 steps against a ~928-step episode.
**Not fixed.**

### A6. `features="legacy"` is not a replication, despite claiming to be

- `mas` are means of **returns**; the original used means of **prices** (~3 orders of
  magnitude apart).
- `vol` is a 24h rolling std of pct_change; the original used `ewm(alpha=0.05).std()`
  of **log** returns.
- `width` is `tick_upper - tick_lower` (~1000); the original passed the raw action
  integer (45).
- `L` is raw pool-scale (~1e18); the original's was x-derived (~1e2).
- `features.py` claims it replicates the reset-time literal `1` in the sigma slot. It
  does not.

**Not fixed.** The "legacy is worth +381" result is withdrawn.

### A7. Two baselines the paper never had, and one of them is our headline winner

`RecentreWhenOut` and `Passive` are ours. `RecentreWhenOut` is the top arm in every
table above and is the arm PPO is reported to lose to. The conclusion survives against
the paper's own four (ILMinimizer still beats PPO), but "the best competitor" is a
strategy the paper never proposed. Also:

- `base_factor` 100 → **1e4**: our `VolProportionalWidth` and `ILMinimizer` produce
  widths **100x** the original's.
- `only_when_out` variants exist in neither original.
- `ILMinimizer` gained a horizon grid {24,168}; the original was fixed at 24.
- `ReactiveRecentering`'s width was fixed at `action_values[1]`; we sweep 3 x 9.

### A8. The paper's second pool does not exist in our panel

The paper used **WETH/USDC 0.05%** and **WBTC/USDC 0.30%** (`0x99ac8ca7...`). Our core
panel has USDC/WETH, WBTC/WETH, WETH/USDT. **`0x99ac8ca7` appears nowhere**, because
the sibling `defi-rv` project never fetched it. Half the paper's evidence base cannot
be reproduced or refuted, only replaced. SPEC documents the WBTC/USDT exclusion and
never mentions this one. The 3x2 panel is presented as a design choice; it is partly a
data-availability accident.

Consequence for H3: our "asset pair" contrast is WBTC/**WETH**, a crypto/crypto pair
with no dollar leg, against the paper's WBTC/**USDC**. And USDC/WETH and WETH/USDT are
both WETH-against-a-dollar, so the 3x2 has 2 distinct economics, not 3. **SPEC's "H3
sign reversed" compares mitigation against a paper sentence about a transfer target's
absolute ceiling: different quantity, different contrast.**

### A9. `rolling.py` is not covered by the pre-registration

SPEC freezes "first 50% train, next 25% validation, final 25% test" and calls
walk-forward "untested here". `rolling.py` is therefore an unregistered protocol, and
`--shaping` / `--schedule` / `--widths` are free parameters outside the stopping rule.
SPEC also pre-registers **daily** as the primary schedule; `rolling.py` defaults to
**event_driven**.

### A10. Statistics pool 6 heterogeneous pools with no clustering

`aggregate` reports one pooled table over 144 window-tests. 24 adjacent windows from
one pool are not independent, and Wilcoxon assumes they are. No per-pool breakdown, so
the largest-scale pool dominates the pooled mean — a reporting regression against the
paper's per-pool tables. SPEC pre-registers bootstrap CIs; none are computed. **Not
fixed.**

---

## B. We fixed a real defect (the paper's number was wrong)

### B1. The fee rate was 105x too high, and fees were zero on every down move

`delta/(1-delta)` = **5.26%** where the 0.05% tier is 0.0005. And `_calculate_fee`
keys both `p` and `p_prime` on `pt`, never `pt_1`, so for any in-range falling price it
returns **exactly 0.0**. Fees accrued only on up-moves, at 105x the true rate. The
down-move half was not in SPEC and is new.

### B2. Rebalancing teleported ±44% of position value

`self.l = xt / (1/sqrt(pt) - 1/sqrt(pu))` sizes new liquidity from the **x leg alone**;
the y leg is re-derived next step. Verified numerically:

| price drift at rebalance | value change |
|---|---|
| -2.0% | **+44.4%** |
| -0.5% | +10.9% |
| 0.0% | -0.0% |
| +0.5% | -11.0% |
| +2.0% | **-43.6%** |

On the paper's ~$40k position that is ±$17,000 per rebalance against a stated $5 gas
cost. **This is the single largest term in the published rewards and the paper never
mentions it.** SPEC misdiagnosed it, pointing at the `xt = xt/2` branch, which is
actually value-conserving.

### B3. The tuning objective was the test set

`optimize_ppo` returns `evaluate_model(rl_model, Monitor(test_env))` and checkpoints on
test improvement. Worth **+675** on our env. This is the one part of SPEC's leak
decomposition that survives.

Additionally, hyperparameters **leaked across rolling steps**: study 1 writes `_r1`,
study 2 writes back to the base YAML, so window *i*'s tuned parameters became window
*i+1*'s defaults. Rolling steps were not independent.

### B4. The transfer result did not depend on the transfer

`train ⊂ same-period test`, `future test ⊂ same-period test`; reward is an episode sum
over segments with a **5.67x length ratio**, which is most of the paper's "20-fold
gap". 40 cells hold **23 distinct values**; `WETH→WBTC_same` and `WBTC→WBTC_same` are
bit-identical. The script's own printed efficiency is **714%**, not 88.5%; the paper's
number uses a different denominator computed off the table by hand.

**We do not reproduce it.** H4 is untested, and the pool it needs is absent (A8).

### B5. PassiveWidthSweep swept exactly one width

`if w not in env.action_values: continue` → `range(20,201,10)` ∩ `[0,45,50,55]` =
**`[50]`**. The paper calls it "fixed-width range, optimized width". It optimized
nothing on WETH.

### B6. Statistics: n=50 was 10 windows replicated 5x

Each competitor has exactly **1 distinct value per window, replicated 5x**. The
independent unit is the window (10). At the window level, "significantly outperforms
three of four" collapses to one at p=0.044 and one at p=0.050 — before the leak or
multiplicity. No correction anywhere across 8 comparisons.

### B7. Sample-period confound

WETH/USDC: 23,995h, 10 windows. WBTC/USDC: 40,801h, 22 windows. **Over half the WBTC
evidence comes from a regime the WETH arm never sees**, yet the paper says the
comparison is "under matched evaluation". Our balanced panel fixes this by
construction.

---

## C. The paper diverges from its own code

1. **The state vector in the text is largely fiction.** Described: one-step return,
   EWMA of returns, `[p_l, p_u]`, relative position within the interval, fee tier.
   **None exist in the code.** Actually present and undisclosed: `w`, `L`, ma24, ma168,
   three Bollinger bands, ADXR, BOP, DX — **9 of 13 features**.
2. **Eq. (4) is sign-inverted as printed.** It pays the agent to lose money. The code
   is right.
3. **Timesteps: claims 1,000,000, runs 100,000.** `total_timesteps: 1000000` sits in
   the YAML and is never read. The paper's explanation for WBTC underperformance
   ("budget is at the lower end") rests on a number never used.
4. **"5 PPO agents under different random initializations" do not exist** in
   `uniswap_test.py`: `seed=256`, fixed, every trial and window. The 5 seeds exist only
   in `baseline_comparison_statistical.py`.
5. **"x_0 = 2 units of the risky token" matches no config.** `x: 10` WETH (~$40k) vs
   `x: 0.15` WBTC (~$6-10k). **The cross-pool reward levels in Tab. IV/V are not
   comparable**, independent of everything else.
6. **A second Optuna stage is hidden.** Tab. II calls γ, λ, ε, c1, c2 "fixed"; study 2
   tunes `ent_coef, gamma, clip_range`, and the YAML the final test env loads carries
   `clip_range: 0.05`, `ent_coef: 1e-5`, contradicting both stated values.
7. **Tab. III's action grid does not describe the WBTC runs.** `pool_wbtc_usdc_030.yaml`
   uses `[0,60,120,180]` with a search space sharing **no entry** with Tab. III.
8. **The WBTC "tick-spacing misalignment" is a 7x understatement.** With `d=60`, action
   45 is 2,700 ticks = a **±31% band**, against ±4.6% on WETH from the identical
   action. That is not misalignment, it is a different strategy, and it can explain the
   cross-pool difference the paper attributes to fee regime.
9. **40,801 WBTC "hours" is a forward-filled grid count**, not observations, on a pool
   averaging 10 swaps/h.

---

## RETRACTED from our own record

**The action grid was never crippled. We misread it by 10x.**

`SPEC` claimed `{0,45,50,55}` are ±0.5% bands "in range ~1.2% of the time" and priced
the competitor handicap at **-2,140**, concluding "the dominant term is the ACTION
GRID, not the leak". The original multiplies by tick spacing:

    self.d = self._fee_to_tickspacing(self.delta)     # 0.05 -> 10
    tl, tu = m - self.d * self.w, m + self.d * self.w

So action 45 = **450 ticks = ±4.60%**, 50 = ±5.13%, 55 = ±5.65%. That is essentially
this rebuild's own `500` (±5.13%), the width we adopted as the "realistic" fix after
calling theirs degenerate. The -2,140 and the 2,815 sign-flip are **artifacts of our
own unit error**. Only the leak (+675) survives.
