# Protocol v3
# Engineered and Nonlinear Alpha Research

Lifecycle: **DRAFT → FROZEN BEFORE REAL PHASE 4/5 RESULTS**.
Final state: **FROZEN**; annotated tag **protocol-v3** identifies the freeze commit.
Machine authority: [protocol_v3.yaml](../configs/protocol_v3.yaml); integrity record: [protocol_v3_lock.json](../configs/protocol_v3_lock.json).
This is an additive research-design protocol. It does not replace, edit or re-execute [Phase 2.5-v2](research_protocol.md) or Phase 3.

## 1. Question, prior information and scope

> In a limited and non-stationary financial time series, can engineered dynamic alpha signals and nonlinear temporal representations provide incremental out-of-sample predictive value beyond the raw alpha benchmark established in Phase 3?

**RULES FIRST; RESULTS LATER.** This freeze precedes any real execution of the new nested Phase 4/5 pipelines, model fitting or their outer OOS scores. This task reads configuration, source and sealed result-file hashes only; it does not read real market X–Y, run transformations on them, or fit even a baseline model.

Phase 3's 97 singles (24 PASS, 72 FAIL, 1 NOT_TESTABLE), seven composites (M/P/V/S PASS), decisions and all diagnostic snapshots remain immutable. They are historical benchmark/descriptive references, not a global input whitelist. M3/P3/P4 are not automatically removed or promoted.

The researcher has already observed Phase 3, short/long-lag diagnostics and ΔX_1/5/21 diagnostics. In particular, Delta5 and Delta21 are **not** first-ever unseen transformations. The v3 candidate choices are informed subsequent research design; they are not retroactive preregistration of those earlier diagnostics. Phase 4/5 outer results are development/model-selection evidence on a previously studied history, not an untouched confirmatory experiment.

The existing v2 reservation is prospective: Phase 2 descriptive EDA covered Development 0–8867, including the later-reserved Internal Holdout. Do not claim that no human or process has ever descriptively read that interval. No Phase 3/4/5 predictive selection uses it. Here and below, “untouched” means no new research/confirmation access under this protocol; it does not erase the acquisition audit, descriptive EDA or byte-hash history.

## 2. Data, timing and outer windows

Only the official Hull data are permitted. The target remains supplied-row **market_forward_excess_returns**, one trading-row horizon, without another target shift, reconstruction, clipping or winsorization.

| Reservation | Inclusive date_id | Role |
| --- | --- | --- |
| Research | 0–7859 | All Phase 4/5 fitting, nested selection and outer development evaluation |
| Internal Holdout | 7860–8867, 1008 rows | One later frozen-pipeline confirmation; currently locked |
| Final Simulation | 8868–9047, 180 rows | Later final sequential competition simulation; currently locked |

Value-bearing Phase 4/5 access must use the existing `load_research_prefix`, after v2/data identity verification. Do not create a wider reader or use official test.csv for research. A byte hash is not a parsed held-out X–Y observation. The present protocol verifier never requests raw-data verification or calls a research loader.

The frozen outer design is unchanged: **initial_train_size=5087, validation_window=252, step_size=252, purge=1, embargo=0, 11 folds, no shuffle**. Fold 1: train 0–5086, purge 5087, validate 5088–5339. Fold 11: train 0–7606, purge 7607, validate 7608–7859. Fit state remains frozen within each outer block.

Current outcomes, date_id and is_scored are not inputs. The raw universe is exactly 94 anonymous features (M18/E20/I9/P13/V13/S12/D9) plus the three official lagged fields. Reconstruct lagged fields with source offset −1 and missing first row, never from mock test values. No MOM family is invented.

All date shifts preserve the consecutive original trading-row grid and NaN positions. At decision t, a fixed causal feature-state algorithm may consume X_t and already realized t−1 outcomes, including earlier validation observations. That is not permission to refit a selector, threshold, normalization, model, direction or hyperparameter on validation y. Never use a later observation to fill an earlier input.

## 3. Exact engineered universe

Every raw input has these five, and only these five, representations:

| Representation | Frozen definition |
| --- | --- |
| Level | X_t |
| Delta5 | X_t − X_(t−5), exact row endpoints |
| Delta21 | X_t − X_(t−21), exact row endpoints |
| Z252 | (X_t − mean)/(population std, ddof=0), trailing 252 **trading rows including t**, at least 126 finite observations |
| Rank63 | Mid-ECDF of X_t within the last 63 **observed values including t**, at least 32 finite values |

Rank63 is an observed-value queue, not a 63-calendar-row window: missing dates retain their place in the output but do not enter the queue. For n observed values, percentile = (number strictly below X_t + 0.5 × number equal to X_t)/n. Ties use this definition; all equal observations map to 0.5. A missing current X_t produces NaN and no queue update. Z252 with missing current X_t, fewer than 126 observations, or zero standard deviation is NaN. A delta with either endpoint missing is NaN. No pre-transformation imputation, centered window, backward fill or negative/predictive shift is allowed.

This gives **97 × 5 = 485** raw representations. D features receive the same transformations and availability checks; unusable or duplicate results are recorded, not rescued. No Delta1, Delta63, other z/rank windows, EMA, MACD, Fourier, wavelet or polynomial expansion is admitted in v3.

Add exactly seven family-level candidates and seven high-state interactions, for a **maximum of 499 named candidates**. IDs are `raw/<field>/<representation>`, `family/<family>/Level` and `interaction/<family>/HighV`. Keep every ID in the registry, even when unavailable; never expand it after seeing Phase 4/5 results.

## 4. Family composites and two-state V proxy

Rebuild M/E/I/P/V/S/D from eligible raw anonymous **levels** using the unchanged Phase 3 family builder, separately inside every reference training prefix. Lagged outcomes have no family. A member's direction is its training Spearman sign; zero/undefined directions are excluded. Map through its training mid-ECDF and equally average oriented observed members. A row requires max(2, ceil(0.5 × qualified member count)) observations and at least two qualified members overall. Do not reuse historical outer-fold recipes, Phase 3 PASS membership or full-history directions.

The implementation supports safe training-prefix reconstruction, so the frozen regime source is **that reference's V-family composite**, not a variable selected for effectiveness. Its members, signs and percentile maps use only the purged reference's historically observed data. Training y can set the inherited historical composite signs; validation y cannot. The applied regime score is a function of current/past X under those fixed training parameters.

The threshold is the finite **training median (50th percentile)** of this V score. `High_V_proxy` means score strictly above the median; `Low_V_proxy` means score less than or equal to it. Missing score means unknown state. Require at least 252 observed reference scores, 126 reference rows in each state, and a nonconstant score; otherwise disable all seven interactions. Do not switch to another V summary when the data fail these rules.

The inherited V composite is signed toward historical alpha, not calibrated physical volatility. “High/low volatility” is therefore a V-family **proxy-state label**, not a proved ordering of realized risk. This caveat prevents supervised orientation from being described as an identified economic volatility measure.

The entire interaction universe is `Composite_F × 1[High_V_proxy]` for F in **M,E,I,P,V,S,D**. A known low state gives zero; unknown regime or missing parent composite gives NaN. There are no low-state duplicates, raw×raw products, transformed×transformed products, alternative thresholds or regime counts. Parent construction must be available before transformed-value eligibility is considered.

## 5. Availability and nested temporal selector

Apply v2's training-only availability tests independently to **each constructed candidate**, after its warm-up: at least **252** observed values, **80%** coverage since first observation, no more than **25%** missing in the trailing **504** rows, and at least **2** distinct values. Raw eligibility does not imply engineered eligibility. Do not fill NaNs before assessing it.

For every purged reference prefix [0,T], use its **last two chronological 252-row blocks** as screening/inner validation. If s1=T−503 and s2=T−251, block j is [sj,sj+251], its fit prefix is [0,sj−2], and sj−1 is purged. Reference endpoints and all source dates must be inside 0–7859. There is no random split and no finite-data exception changing these lengths.

For each candidate and each inner block, rebuild fitted construction parameters (notably family signs, member maps and V threshold) only on that block's earlier purged prefix. Compute complete-pair Spearman within the block using average ties. Require training eligibility in both inner fit prefixes, at least 126 validation pairs and 50% coverage in each, finite nonzero ICs of the **same sign in both blocks**, and eligibility again in the final reference prefix. These are inner training-selection conditions, never outer validation feature filters.

Rank surviving candidates by descending **absolute arithmetic mean of the two inner OOS Rank ICs**, then ascending candidate ID (literal Unicode/ASCII lexicographic order). The selected direction is the common sign. There is **no absolute IC/noise threshold**.

Greedily scan this full ordering, keeping at most **Top-K=16**. Compare each candidate against those already retained using pairwise-complete **reference-training Spearman**, at least 252 common observations. Reject the lower-ranked candidate when |rho| **≥0.90**, or when a pair correlation cannot be estimated. Continue beyond the original top 16 until 16 survive or the list ends. This is a training-only redundancy rule, not reuse of the post-hoc clustering outcome.

Freeze selected IDs, directions and a final reference-training mid-ECDF per selected candidate. Contribution = direction × (2F_train(candidate value)−1). The signal is the fixed equal-weight sum divided by selected count; a missing mapped contribution is zero, with no row-wise weight optimization. A row still needs max(2, ceil(0.5 × selected count)) observed contributions. Fewer than two selected candidates yields constant zero with explicit `no_alpha` status; no substitute candidate or target-dependent rescue is allowed. A constant/undefined validation IC remains undefined.

## 6. A/B benchmark definition and genuine outer evaluation

**A = Raw Alpha:** the same v3 nested selector and scoring rule, restricted to the **97 Level candidates**. **B = Engineered Alpha:** that rule over the fixed 499-candidate registry, including Levels. This matched comparison isolates the additional representation universe without a Phase 3 PASS prefilter.

There was no uniquely frozen Phase 3 aggregate A forecast. The archived Phase 3 benchmark remains exactly what it was: individual/composite metrics and formal decisions. A is an explicitly new **v3 raw-only benchmark pipeline**, fixed now using Phase 3 methods; do not relabel its later results as previously sealed Phase 3 results or recompute the 97/7 BH exercise.

Each outer fold creates candidates and chooses inputs/directions only inside its own training history, then evaluates its frozen A/B signal on the 252 validation rows. Save candidate registry, eligibility reasons, both inner ICs, ranking, selected count/IDs, directions, training map/threshold hashes, outer pair counts, Rank IC and descriptive Pearson. No outer y chooses a representation, selector cutoff, regime, K or correlation cut.

## 7. Shared temporal input design for Ridge/GRU/TCN

Use the 97 raw Level inputs, not an uncontrolled 94×63 network. In **each fitting reference separately**, apply raw eligibility, compute training pairwise Spearman correlations (minimum 252 pairs), and average-linkage clustering on distance `1−|rho|`, with frozen cut **0.10**; undefined pair distances are 1. Deterministic equal-distance merge ties use ascending current node IDs. No Phase 3 post-hoc cluster memberships are reused.

Within each cluster choose the raw Level member with the highest stable two-inner-block score from section 5, tie by feature ID. Drop clusters without a qualifying member. Order representatives by score then feature ID; retain at most **16**. A selector fitted for an inner model may use only that inner model's reference prefix, not the parent's whole outer training window. This finite extra screening layer is mandatory: reusing a final outer selector while scoring an earlier inner model would leak its future labels.

All three models receive **the same representatives, same 63 rows t−62…t, and same 16 channels** in a given fitting/reference context. Pad unused channels with zero, not extra learned features; no missingness indicator channels. Require 63 real consecutive history rows, dropping training endpoints before 62; do not invent pre-date-0 history. Sequences overlap but endpoint labels remain strictly within the appropriate purged fit interval.

Fit observed mean/std (ddof=0) of each channel on the reference training X only. Impute missing inputs with that training mean (standardized zero), standardize and clip X to **[−5,5]**. Undefined/std≤1e−12 channels are unavailable and remain padded; no forward/backward fill. Target mean/std uses admissible training endpoint labels only, with **no target clipping**. Invert target scaling before predictions are averaged or compared. No scaler or input selection sees inner stopping/validation or outer validation labels or X outside its fitting prefix.

## 8. Exact small-model candidates and budgets

| Model | Frozen candidate settings | Size at 16 channels |
| --- | --- | --- |
| Ridge | Flatten 63×16; alpha ∈ {0.1,1,10}; unpenalized intercept; deterministic SVD float64 | 1008 slopes + 1 intercept |
| GRU | One layer, input 16, hidden 8, two bias vectors per gate; last hidden → linear 8→1; dropout 0 | 633 trainable parameters per seed |
| TCN | Two residual blocks; two causal Conv1d per block, channels 8, kernel 3, block dilations [1,2], biases; first residual projection 16→8 with bias; dropout 0 | 1137 trainable parameters per seed |

Ridge minimizes **mean squared error + alpha × squared slope norm**. If an implementation uses an SSE objective, multiply alpha by the fitting sample count. Select alpha by highest mean IC of the two chronological inner validation blocks, with both finite; numerical ties within 1e−12 prefer larger alpha. No extra alpha search is allowed.

TCN uses left-only padding d×(k−1), ReLU between its two convolutions and after residual addition, no batch norm or weight norm. Mean-pool all **63 causal output positions** and use a linear 8→1 head. Pooling lets every input date contribute despite a short convolutional receptive field; no future position is used. Identity residuals have no parameters. Parameter accounting: GRU = 3hp+3h²+6h+h+1 = 633; TCN = 392+200+136+200+200+9 = 1137.

All models must pass a **≤10,000 actual trainable-parameter** preflight before future fitting. GRU and TCN have **one setting each**, not an architecture grid. The five-seed GRU+TCN collection has **8850** trainable parameters; Ridge has 1009 fitted coefficients. Do not add LSTM, Transformer, Informer, TFT, N-BEATS, TabNet, tree boosters or larger networks in v3.

## 9. Training, early stopping and seeds

Neural training is fixed: **AdamW**, learning rate **0.001**, weight decay **0.001**, standardized-target MSE, gradient norm clip **1.0**, batch size **64**, chronological no-shuffle batches, reset hidden state per 63-row sequence. Use CPU float32, deterministic algorithms and one computational thread; record actual future software versions. No portfolio loss, outer-IC loss selection or target clipping is added.

Every model-fitting reference [0,R] gets its own stopping block [R−251,R], purge R−252 and weight-training prefix [0,R−253]. Fit selector/scaler/target scaling on that earlier weight-training prefix only. Train up to **100 epochs**, eligible checkpoints from **epoch 10**, patience **10**, improvement only if stopping MSE falls by **more than 0.0001**, earliest checkpoint on ties. Nonfinite loss makes that seed unavailable.

After choosing the epoch count, reinitialize with the **same seed**, refit the selector/scalers on the full purged fitting reference [0,R], and train for exactly that many epochs. Never early-stop on the subsequent inner model-validation block or outer validation. This happens again for the final outer refit: stopping rows remain inside outer training, and the outer 252 rows stay untouched.

Use exactly **[11,22,33,44,55]** for each neural architecture. Average all five inverse-scaled predictions, never select a seed. If any seed fails, the architecture/fold is unavailable; do not silently drop seeds or replace it with a lucky run. Runtime fixes require logged synthetic reproduction and a new implementation freeze before real re-execution; no result-driven architecture/setting change is permitted.

## 10. C, nonlinear comparisons and incremental information

**C = fixed 0.5 × GRU five-seed mean + 0.5 × TCN five-seed mean**, in original target units. Both architectures must be available; there is no Ridge fallback or post-result winner selection. Ridge, GRU and TCN also remain separately named, predefined secondary comparisons on identical information and outer folds. This gives one fixed nonlinear pipeline rather than choosing the best nonlinear architecture after observing outer results.

For each outer reference, produce A/B/GRU/TCN inner OOS scores over the two 252-row inner blocks. Every inner forecast must be generated by a complete fit restricted to that block's earlier purged reference, including its own selector/scaler and internal stopping split. Never substitute in-sample fitted predictions or an outer-fitted selector. Subsequent inner blocks may use earlier realized blocks only once historical.

For **each of GRU and TCN**, fit residualization only on common inner OOS rows inside the outer training history (minimum **252**). Normalize these inner OOS component scores with their mean/std (ddof=0), fit `z_deep = beta0 + betaA*z_A + betaB*z_B + residual` by OLS pseudoinverse **rcond=1e−12**, and retain the normalization and coefficients. No y is used as the residualization regressand. Apply those fixed coefficients to normalized outer forecasts, then report IC(deep,y) and IC(residualized_deep,y) on matched outer dates.

This is a descriptive incremental-information diagnostic, not a formal discovery gate, architecture selector, weighting rule or proof of causality. Undefined normalizers or insufficient inner common pairs make the residual diagnostic unavailable, never authorize fitting on outer data.

## 11. D and normalization

**D = (z_A+z_B+z_C)/3**, equal weights **1/3 each**. Fit each score mean/std (ddof=0) on common A/B/C **inner OOS** forecasts inside the current reference, minimum **252** rows, then freeze and apply to outer forecasts. C's inner forecast is its same fixed GRU/TCN seed ensemble. No training-IC weights, regression stacker, outer tuning, holdout fitting or data-dependent reweighting is allowed. A missing component, insufficient calibration rows or zero/undefined scale makes D unavailable, with no two-component rescue.

## 12. Metrics and development comparison

For A/B/C/D and the predefined Ridge/GRU/TCN comparisons, preserve v2 block-wise complete-pair average-rank conventions. A fold requires at least **126 pairs and 50% coverage**; an undefined/constant correlation remains undefined. Save pooled block-rank IC, equal-fold mean and median, **IC std ddof=1**, positive-fold ratio, worst-fold IC and coverage. The primary selection score is the **equal arithmetic mean of the 11 outer-fold Rank ICs**, not pooled global Spearman or RMSE.

For paired primary comparisons, first determine structural availability from fitted state/runtime and predictions without consulting outer y: a candidate must have prediction coverage and variation in every block. Use common finite predictions across these candidates and finite target dates within each fold, then compute all paired primary metrics there. Record individual-coverage metrics separately. Do not reinterpret missing/constant folds as zero or remove bad folds to improve a mean.

Pearson is descriptive: estimate 1%/99% bounds on the score's **inner OOS training forecasts**, use those fixed bounds on outer predictions, and never clip validation y. No new HAC p-values, bootstrap significance, BH/BY families or hundreds of candidate PASS assertions are authorized by v3. Existing Phase 3 inference is unchanged. Development IC comparisons, residual ICs and inner selection are exploratory/model-selection evidence, with serial dependence and prior history inspection explicitly disclosed.

## 13. Phase 6 primary selection: four pipelines only

The only primary candidates are **A Raw, B Engineered, C Nonlinear, D Combined**. Historical mean (fit on purged training labels), zero forecast, and separately named Ridge/GRU/TCN remain secondary benchmarks; undefined zero-variance ICs are reported, not rescued. There is no fifth primary after opening held-out data.

An admissible primary has **11 finite paired fold ICs**, **positive IC in at least 7/11**, and **positive mean IC**. Let M be the highest admissible mean. Form the near-best set with **M−mean < 0.01** (strict; exactly 0.01 is not near-best). Choose the **simplest** member using frozen order **A < B < C < D**. Any same-complexity tie is broken by lower IC std, then better worst-fold IC, then ascending ID. This is one pass anchored to the global M: do not chain successive margins. The absolute simplicity margin is **0.01 Rank IC**, not a significance threshold.

If none qualifies, record **no admissible primary** and stop; do not unlock the holdout, relax 7/11, alter the margin, add a model or invent an allocation rescue. Selecting a primary on outer development results is planned model selection, not an untouched confirmation. Commit the selected ID, full fitted algorithm, implementation version and secondary reporting list before any held-out predictive evaluation.

## 14. Internal Holdout and Final Simulation remain locked

Opening the Internal Holdout requires all of: Phase 4 complete; Phase 5 complete; a selected, committed primary; a separately frozen Phase 6 forecast/allocation execution protocol; and **separate explicit authorization**. v3 does not implement a loader, unlock, fit a model or authorize this run. Keep existing loader/lock reservations unchanged.

The one confirmation run fits on **0–7858**, purges **7859**, and freezes learned state throughout **7860–8867**. Deterministic feature state may consume currently available X and previously realized official lagged information under the same frozen online algorithm; it may not refit on holdout outcomes. Report primary plus A/B/C/D/Ridge/GRU/TCN, training historical mean and zero forecast as predefined secondaries. A better secondary **never replaces** the primary. Holdout outcomes cannot select a feature, epoch, hyperparameter, seed, combination weight, model or allocation.

Final **8868–9047** is one later Kaggle-style sequential simulation, after confirmation completion and separate authorization. It cannot feed back into selection or training design. Exact forecast-update, transaction-cost, risk and position policies must be fixed by a separate Phase 6 execution/allocation protocol before confirmation; v3 freezes no optimized position strategy. Official exposure remains **[0,2]**. A violated reservation is disclosed as exploratory and cannot be repaired by relabeling the same data untouched.

## 15. Forbidden adaptation and execution governance

Do not add transformations, change any window or regime threshold, alter K/correlation cuts, choose cluster representatives with outer performance, change architectures, pick a best seed, tune on holdout, return from Final to training, or describe prior exploratory diagnostics as preregistered. Validation y may score the frozen output, never fit its generating state. Technical failures may not silently change rules or sample membership.

Future implementations must record all candidate and fitted-state inventories, fit/source date bounds, selected input and direction maps, correlations, scaling/normalization hashes, epoch choices, all seeds, exact parameter counts, predictions, runtime environment, code commits and consumed reservations. A config test is a design contract, not proof that an as-yet unwritten Phase 4/5 implementation satisfies it. Before real execution, implement and synthetically verify the contract, freeze its code separately, verify this v3 lock and preserve all old result hashes.

## 16. Freeze and metadata-only verification

The separate v3 lock binds this document, YAML, metadata validator and its tests, the old v2 lock, frozen Phase 3 code identity, and all existing tracked Phase 3 result/diagnostic files. Its canonical text hashes use UTF-8/LF; sealed Phase 3 snapshot bytes remain bound in original form. It does not add fields to or rewrite Phase 2.5-v2. Local ignored artifacts are additionally compared with the pre-task baseline; no raw data are opened.

```powershell
.\.venv\Scripts\python.exe -m src.validation.protocol_v3 --check-config
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m src.validation.protocol_v3 --check
.\.venv\Scripts\python.exe -m src.validation.protocol_v3 --check-tag
```

The full suite, old v2 lock, Phase 3 snapshots, both post-hoc snapshots and code hashes must pass before the independent freeze commit and annotated **protocol-v3** tag. The tag points to that commit, not to a later experiment. No Phase 4/5 command or model is implemented/executed here. Future scientific changes require a new independent protocol version before new results, not editing this frozen design after inspection.

Protocol v3 was frozen before any real Phase 4 or Phase 5 engineered/deep-alpha results were observed.

Internal Holdout 7860–8867 and Final Simulation 8868–9047 remain untouched.

The closing statements refer to the new nested v3 experiments and prospective confirmation/simulation reservation; the prior Phase 3/post-hoc observations and historical descriptive EDA disclosures in section 1 remain part of the record.
