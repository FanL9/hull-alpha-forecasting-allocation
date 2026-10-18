# Phase 2.5 — Research Protocol Freeze

Version: **phase2.5-v1**. Status: **frozen; Phase 3 not started**.

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

## Future single-feature inference and multiplicity

These are rules for Phase 3, not results from this stage:

- Report raw Pearson IC and Spearman Rank IC, sample counts and complete-pair coverage by fold. Require at least **126 validation pairs**, **50% validation coverage** and **3 eligible folds** for a primary discovery test. Missing/unavailable folds stay explicit; no fabricated zero IC.
- Report equal-fold mean IC, sample IC std (ddof=1), sign consistency and positive-fold ratio among defined folds. Orientation is fixed from each fold's training Pearson sign; a zero/undefined training association gives no oriented test for that fold. Never choose a sign from validation or holdout. Report raw and oriented metrics separately.
- The one primary two-sided p-value per feature comes from pooled chronological validation Pearson IC, using X centering/scaling and direction determined inside each corresponding training window. Retain native timestamps and unavailable-row gaps; never concatenate observed pairs as if gaps were adjacent. Eligible validation rows are disjoint. Pooling retains cross-boundary dependence; fold metrics remain essential. This is a future research recipe, not an implemented Alpha pipeline.
- Use **Newey–West/Bartlett HAC with lag 5** on the Pearson-correlation influence function and a normal approximation. Generic mean-HAC and correlation-HAC tools are implemented; no regression is fitted. Undefined uncertainty becomes no rejection, never a zero p-value. Rank-IC HAC is only an approximate secondary diagnostic.
- Add a **paired moving-block bootstrap**, block length **20**, **2000** repetitions, **95%** percentile interval, seed **42**. Resample X/Y jointly on the original grid; record valid repetitions. Never stitch blocks across nonconsecutive eligible segments. This uncertainty resampling is not random train/test splitting. Strong nonstationarity and data-dependent missingness can still invalidate nominal uncertainty.
- Apply **Benjamini–Hochberg at FDR 0.05** to one prespecified family of **97 hypotheses**, one target and one horizon. Ineligible/unestimable hypotheses receive p=1, preserving the denominator. Do not treat secondary Rank IC, family tables or repeated windows as alternative discovery gates. New horizons/variants expand multiplicity and require a new registered family before inspection.
- Report **Benjamini–Yekutieli** adjusted values as dependence sensitivity. BH's control requires independence or suitable positive dependence, not arbitrary financial dependence. A finding sensitive to BY is labeled dependence-sensitive; neither correction proves persistent or investable alpha. No stability threshold is retrofitted after seeing results.

Implemented tools only calculate statistics from caller-supplied arrays. There is no CLI that applies them to official X/Y. The current EDA entry point continues to reject target columns in X analysis.

References: [Benjamini–Hochberg (1995)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x), [Newey–West (1987)](https://doi.org/10.2307/1913610), [Künsch (1989)](https://doi.org/10.1214/aos/1176347265).

## Baselines and later selection

Forecast baseline: the arithmetic mean of **market_forward_excess_returns in that fold's purged training rows**, frozen throughout validation; zero forecast is a diagnostic comparison. Do not use an expanding mean that consumes validation outcomes within the block.

Portfolio baseline: **allocation=1.0** (100% S&P 500), with **allocation=0.0** cash as a sanity comparison. No baseline is optimized or evaluated on real returns now. The official score can be undefined for a zero-risk strategy; do not silently replace that with a manufactured score.

Deflated Sharpe belongs to later **model/strategy selection**, particularly multiple strategies, parameter searches and backtest candidates. It is not a gate in single-feature Phase 3 screening and is not implemented or optimized here. Record all later trials before using it; see [Bailey–López de Prado (2014)](https://doi.org/10.3905/jpm.2014.40.5.094).

## Locks, access history and changes

`configs/validation.yaml` is authoritative. `configs/research.yaml` fixes eligibility, inference, FDR and baselines; `configs/official_metric.yaml` records the verified scorer. A protocol lock binds these documents/configurations and relevant source hashes. Future research must verify it before reading research values. Any change requires an explicit new protocol version and disclosure before examining new predictive outcomes.

Historical Phase 2 outputs and `split_definition.json` retain their original metadata, including the then-undecided future design. They are historical evidence, not current protocol configuration. They are not regenerated in this task.

Internal Holdout values have already appeared in Phase 2 descriptive EDA, and Phase 1/withdrawn work involved earlier inspection. Reservation is **prospective**: it prevents subsequent predictive selection and tuning, but cannot establish a historically untouched independent test. The existing final 180-row firewall continues unchanged. No held-out values or results determine these rules.

After future walk-forward selection: freeze preprocessing, feature set, model, timing and evaluation schedule; authorize one Internal Holdout evaluation; then freeze any complete strategy before separately authorizing the final 180-row simulation. Reusing either later interval for redesign invalidates its selection-independent interpretation. This stage performs none of those evaluations.
