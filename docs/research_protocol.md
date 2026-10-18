# Phase 2.5 — Research Protocol Freeze

Version: **phase2.5-v2** (current). Status: **frozen; Phase 3 not started**. Supersedes phase2.5-v1, whose lock is archived at `configs/protocol_lock_phase2.5-v1.json`.

**Phase 2.5 v2 was defined before any real feature–target Alpha Research was executed.** v2 changes only the statistical rules: the primary metric, a secondary robustness metric, explicit inference roles, a second pre-registered hypothesis family and a global sensitivity analysis. The reasons come only from properties of X already observed in Phase 2 EDA: heavy-tailed feature distributions, temporal distribution drift, and missingness/availability structure. No real IC, predictive result, feature ranking, model performance or backtest result informed v2. Splits, purge/embargo, availability thresholds, HAC lag, baselines and the official metric are unchanged from v1.

This stage fixes rules and tests software using artificial observations. It does not repeat real-data EDA, test real X–Y relationships, fit models, generate signals, evaluate real portfolios or optimize scores. Phase 1/2 data, reports, notebooks and artifacts remain preserved.

## Labels and information timing

The authority is the [official data description](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data), the [official evaluation page](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview/evaluation) and the locally acquired gateway source, not participant discussions.

| Field | Verified definition and availability |
| --- | --- |
| `forward_returns` | S&P 500 return from buying on the identified trading day and selling one day later; contemporaneous training outcome |
| `market_forward_excess_returns` | Forward return minus its rolling five-year mean, then MAD-based winsorization with criterion 4; training outcome |
| `risk_free_rate` | Federal funds rate; contemporaneous training outcome/cash-accounting input |
| `lagged_forward_returns` | The corresponding return supplied by inference with one trading-day lag |
| `lagged_market_forward_excess_returns` | The corresponding adjusted target supplied with one trading-day lag |
| `lagged_risk_free_rate` | The corresponding cash-rate field supplied with one trading-day lag |

Use the decision convention `X_t → allocation_t → forward_returns[t]`, with the holding-period return realized one trading day later. At inference call t, the three published lagged fields reveal their t−1 outcomes; current t outcomes are not inputs. Keep the publisher's target on its existing row; never shift an already-forward label a second time.

Symbolically, `y_t = W_MAD,4(r_t − mean_5y(r)_t)`, where W denotes the publisher's MAD winsorization. This verifies the published construction; it does not invent an exact reconstruction. The public description does not specify rolling endpoint/minimum periods, five-year row conversion, MAD normalization or all clipping implementation details. Use the supplied y unchanged. Official wording confirms a one-day holding period but does not certify exact intraday feature cutoffs or a close-to-close executable price convention. Do not infer these from participant comments. The scorer consumes the supplied cash rates without dividing by 252 or replacing them with external rates.

The Phase 1 dictionary is a preserved, dated audit snapshot. Its former verification caveats are superseded by this protocol and the current interface/metric notes; its data statistics are not rewritten.

## Prediction and portfolio evaluation are separate

Primary forecasting target: **market_forward_excess_returns**. Historical-mean adjustment and tail treatment motivate studying its stability; they do not prove easier forecasting or alpha. The official metric does not use this target or an IC.

```text
X → forecast → allocation → realized portfolio return → official metric
```

Prediction metrics belong to the prediction layer. Cash, market returns, bounded allocations and the verified official metric belong to portfolio evaluation. Metric implementation here is a calculator tested on synthetic cases, not a real-data backtest. See [official metric notes](official_metric_notes.md).

## Frozen chronological boundaries

| Area | Inclusive date_id | Rows | Rule |
| --- | --- | --- | --- |
| Walk-Forward Research | 0–7859 | 7860 | Future training and validation only |
| Internal Holdout | 7860–8867 | 1008 | Prospectively reserved; no selection/tuning |
| Competition Simulation | 8868–9047 | 180 | Existing lock; headers/IDs/counts only |

The two internal areas still constitute Development 0–8867 (8868 rows); the outer partition has not changed. Official test.csv remains interface/schema-only, never model evaluation data.

Internal Holdout is **4 × 252 = 1008 trading observations**, approximately four metric-annualization years and 5.6 times the final 180-row simulation. It supplies a longer later evaluation interval while leaving 7860 rows for research. This is a design convention, not a power calculation or guarantee of stable inference. It was chosen using sample length and already-published X missingness, never predictive performance.

Existing Phase 2 missingness records place E7's first observation at 6969 and V10's at 6049. Shorter expanding training histories must not assume these inputs exist. Under the coverage rule below, the last two proposed training windows would have 386/638 E7 observations based solely on the recorded leading interval; an individual feature may therefore have insufficient fold coverage for a discovery claim. Do not move boundaries or lower eligibility thresholds to rescue it.

| Parameter | Frozen value |
| --- | --- |
| `initial_train_size` | 5087 |
| `validation_window` | 252 |
| `step_size` | 252 |
| `purge_size` | 1 |
| `embargo_size` | 0 |
| Fold count | 11 |
| Internal Holdout | 7860–8867 |
| Final Competition Simulation | 8868–9047 |

The initial sample exceeds 20×252 observations and absorbs a 47-row remainder, allowing all 11 later validation blocks to have a full 252 rows. Fold 1 trains 0–5086, purges 5087 and validates 5088–5339. Fold 11 trains 0–7606, purges 7607 and validates 7608–7859. Training expands; validation intervals do not overlap. A model/transform, if later adopted, stays frozen within each block. No model family or hyperparameters are selected now.

Before a future Internal Holdout evaluation, refit only on 0–7858 and purge 7859, then freeze throughout 7860–8867. Internal Holdout is not part of walk-forward, multiple testing, parameter selection or feature-availability decisions. There is no automatic unlock or evaluation command in this stage.

The one-row purge requires training outcome endpoints to precede the first validation decision strictly. With one-day labels, additional post-validation embargo is unnecessary in a forward-only expanding split: no future training rows enter the current fold. Earlier validation labels may join later training once historical and outside that fold's purge. The splitter also supports positive embargo for future protocols: rows immediately after validation are excluded from all later training and validation; step size must skip them. Default remains zero. Longer target horizons require a new protocol before use.

## Feature availability and lagged candidates

Future candidate universe: **94 anonymous fields + 3 verified competition-available lagged outcome fields**. The anonymous-feature count remains 94; no MOM family is created. `date_id`, `is_scored` and current outcomes remain excluded.

For historical training data, the three lagged candidates can be reconstructed by a one-row shift over consecutive IDs. At validation time only past observations may initialize that history. The mock official test file is never copied into research data. Helpers exist and are tested on artificial observations; no real lagged-feature research has run.

A feature enters a particular fold only if its **training X** meets all conditions:

- At least **252 nonmissing observations** and **2 distinct observed values**.
- At least **80% observed coverage from its first observed training row to that training endpoint**.
- At most **25% missing in the trailing 504 training rows** (or all training rows if shorter).

The post-launch denominator distinguishes structural leading gaps from sporadic holes; recent coverage prevents a formerly observed but stale input entering. No fill, backward availability, validation coverage or Y-based selection changes eligibility. Store reasons and coverage for every field/fold. Preserve all source columns.

## Primary and secondary metrics

**Primary Alpha metric: block-wise Spearman Rank IC.** For each of the 11 validation blocks, X and y are ranked separately inside that block, using only complete (X, y) pairs, with average ranks for ties. NaNs receive no rank and keep their time positions. Ranks are divided by (complete pairs in the block + 1) so blocks with different coverage share one scale. The ranked blocks are concatenated chronologically and **never re-ranked**; no other fold enters a block's ranks. The primary IC and its inference are computed on this concatenated rank series. This prevents level drift across folds from entering the Rank IC. Implementation: `blockwise_rank`, `blockwise_rank_pairs`, `blockwise_spearman_hac` in `src/statistics/robust_inference.py`.

**Secondary robustness metric: training-window 1%/99% winsorized Pearson IC.** For each feature and fold, the 1st and 99th percentiles of X are estimated from that fold's training window only. The same bounds clip both training X and validation X. Validation data, pooled folds and future data never estimate bounds. y is **not** winsorized again: `market_forward_excess_returns` is already winsorized by the publisher. This metric is not confirmatory. A feature that is significant under winsorized Pearson but not under the primary Spearman test is **not** a discovery. Implementation: `fit_winsor_bounds`, `apply_winsor_bounds`.

Fold-level reporting keeps fold IC, sample counts, complete-pair coverage, equal-fold mean IC, IC std (ddof=1), sign consistency and positive-fold ratio. A fold counts only with at least **126 validation pairs** and **50% validation coverage**. A hypothesis needs at least **3 eligible folds**. Missing folds stay explicit; no zero IC is fabricated.

## Inference roles

- **Newey–West → significance.** The single primary two-sided p-value comes from Bartlett HAC with **lag 5** on the Pearson-correlation influence function of the block-wise ranked series, with a normal approximation. This is approximate inference on the correlation of block-wise ranked series, not an exact finite-sample Spearman test. NaN gaps keep their time positions. Undefined uncertainty means no rejection, never p=0.
- **Paired moving-block bootstrap → confidence interval and stability only.** Block length **20**, **2000** repetitions, **95%** percentile interval, seed **42**. It resamples the already block-wise ranked (X, y) pairs jointly and preserves time dependence. It produces no p-value and no alternative discovery rule.

Strong nonstationarity and data-dependent missingness can still invalidate nominal uncertainty.

## Hypothesis families

**Family 1 — 97 single-feature hypotheses (primary).** 94 anonymous features plus `lagged_forward_returns`, `lagged_market_forward_excess_returns` and `lagged_risk_free_rate`. A feature that fails fold availability or has fewer than 3 eligible folds is `not_testable` and receives p=1. The denominator is always 97. E7 and other late-starting features follow this rule; thresholds are not relaxed.

**Family 2 — 7 family-composite hypotheses (pre-registered).** One composite each for M, E, I, P, V, S and D, built fold by fold from training information only (`src/features/family_composites.py`):

1. **Members:** family features that pass the existing availability rule in that fold's training window. A feature eligible in another fold is not added to this fold. The three lagged fields belong to no family.
2. **Direction:** the sign of the member's training-window Spearman correlation with y. Positive → +1, negative → −1, zero or undefined → member excluded. Validation, other folds and the full sample never set direction. Qualified members = eligible members with a nonzero direction.
3. **Normalization:** rank-based, not z-score. Each member is mapped to the mid-rank empirical CDF of its training values, p(v) = (#train < v + 0.5 × #train = v) / n_train. Validation values use the same training map; no distribution is refitted on validation. The result is multiplied by the direction.
4. **Weighting:** equal-weight mean of the oriented members observed in that row. No IC, p-value, volatility, regression or optimized weights.
5. **Missing members:** a row needs at least max(2, ceil(0.5 × qualified members)) observed members, otherwise the composite is NaN. With fewer than 2 qualified members the fold composite is NaN.
6. **Testing:** the same framework as Family 1: block-wise Spearman, HAC lag 5, block-bootstrap interval, the same fold eligibility, not testable → p=1. The denominator is always 7.

The training percentile map used to build composites is a different step from the block-wise validation ranking used to test them.

The D family is retained. Interpretation of the D-family composite should be cautious because rank-based aggregation of discrete features may contain limited cross-sectional resolution. The test standard is not changed for it.

## Multiple testing

| Analysis | Hypotheses | Method | Role |
| --- | --- | --- | --- |
| Family 1 | 97 single features | BH, FDR 0.05 | Primary, reported separately |
| Family 2 | 7 family composites | BH, FDR 0.05 | Primary, reported separately |
| Global sensitivity | 104 = 97 + 7 raw primary p-values | BH, FDR 0.05 | Sensitivity only; never replaces family results |
| Dependence sensitivity | each of the above | Benjamini–Yekutieli | Sensitivity only |

BH requires independence or suitable positive dependence. A finding that fails BY is labeled dependence-sensitive. Secondary metrics, family tables and repeated windows are not discovery gates. New horizons or variants need a new registered family before inspection. Implementation: `adjust_fdr`, `family_pvalues`, `protocol_multiplicity` in `src/statistics/multiple_testing.py`.

All tools only compute statistics from caller-supplied arrays and are tested on synthetic data. No CLI applies them to official X/Y.

References: [Benjamini–Hochberg (1995)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x), [Benjamini–Yekutieli (2001)](https://doi.org/10.1214/aos/1013699998), [Newey–West (1987)](https://doi.org/10.2307/1913610), [Künsch (1989)](https://doi.org/10.1214/aos/1176347265).

## Baselines and later selection

Forecast baseline: the arithmetic mean of **market_forward_excess_returns in that fold's purged training rows**, frozen throughout validation; zero forecast is a diagnostic comparison. Do not use an expanding mean that consumes validation outcomes within the block.

Portfolio baseline: **allocation=1.0** (100% S&P 500), with **allocation=0.0** cash as a sanity comparison. No baseline is optimized or evaluated on real returns now. The official score can be undefined for a zero-risk strategy; do not silently replace that with a manufactured score.

Deflated Sharpe belongs to later **model/strategy selection**, particularly multiple strategies, parameter searches and backtest candidates. It is not a gate in single-feature Phase 3 screening and is not implemented or optimized here. Record all later trials before using it; see [Bailey–López de Prado (2014)](https://doi.org/10.3905/jpm.2014.40.5.094).

## Locks, access history and changes

`configs/validation.yaml` is authoritative for boundaries. `configs/research.yaml` fixes eligibility, metrics, inference, hypothesis families, multiple testing and baselines; `configs/official_metric.yaml` records the verified scorer. `configs/protocol_lock.json` (phase2.5-v2) binds these documents/configurations, the relevant source files and the archived v1 lock by hash. Future research must verify it before reading research values. Any change requires an explicit new protocol version and disclosure before examining new predictive outcomes.

Historical Phase 2 outputs and `split_definition.json` retain their original metadata, including the then-undecided future design. They are historical evidence, not current protocol configuration. They are not regenerated in this task.

Internal Holdout values have already appeared in Phase 2 descriptive EDA, and Phase 1/withdrawn work involved earlier inspection. Reservation is **prospective**: it prevents subsequent predictive selection and tuning, but cannot establish a historically untouched independent test. The existing final 180-row firewall continues unchanged. No held-out values or results determine these rules.

After future walk-forward selection: freeze preprocessing, feature set, model, timing and evaluation schedule; authorize one Internal Holdout evaluation; then freeze any complete strategy before separately authorizing the final 180-row simulation. Reusing either later interval for redesign invalidates its selection-independent interpretation. This stage performs none of those evaluations.
