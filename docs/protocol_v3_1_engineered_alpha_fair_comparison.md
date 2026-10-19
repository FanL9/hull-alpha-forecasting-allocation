# Protocol v3.1 — X-only Regime and Fair Phase 4 Calibration

State: **FROZEN BEFORE REAL PHASE 4 RESULTS**. Tag: **protocol-v3.1**.
Parent: [Protocol v3](protocol_v3_engineered_and_deep_alpha.md), unchanged commit `c3e35d62cfaa2e8ccff231511ebe803b7656e34b` / annotated `protocol-v3`.
Machine authority: [protocol_v3_1.yaml](../configs/protocol_v3_1.yaml). Independent lock: `configs/protocol_v3_1_lock.json`.

## Authorized amendment and inherited rules

The user explicitly requested a new frozen version whose regime does not depend on y. v3's family-alpha V composite uses training-target signs; it is safe from validation-label fitting but is not X-only. This amendment separates the volatility-state proxy from that supervised alpha composite. The old version and tag are preserved, not overwritten or reinterpreted.

All other scientific settings inherit v3 unchanged: the target, 0–7859 research boundary, 11 outer folds, two inner 252-row blocks, purge=1, eligibility, transformations, Top-K=16, |Spearman|≥0.90 greedy redundancy, aligned training-mid-ECDF normalization, equal member weights and paired outer evaluation. Deep architectures, lookback, seeds, parameter limits, primary selection, holdout/final boundaries and allocation constraints are not redesigned.

## X-only V-state proxy

For each purged reference prefix, select eligible raw V1…V13 by the unchanged X-only availability rules. There are no member signs, target correlations or target-derived weights: every member has direction **+1**. Map each through the training mid-ECDF `F(v)=(#train<v + 0.5×#train=v)/n`; average observed member percentiles with equal weights. At least two eligible members are required, and a row needs `max(2,ceil(0.5×member_count))` observed values.

Fit the median on this training proxy only. High means strictly above the median; Low includes equality. Preserve v3's minimum 252 observed training scores, 126 rows in each state and nonconstant-score rule. Failure disables all seven interactions, with no alternative proxy or threshold. The fitted regime API accepts X and reference dates only, **not y**. Mutating training, validation or future y cannot change its members, maps, score or threshold.

The ordinary seven **alpha** family composites still use the inherited training-only supervised signs. They are candidates, not regime definitions. The seven interactions remain signed `Composite_F × 1[High_V_proxy]` for M/E/I/P/V/S/D. Thus candidate counts remain exactly 97×5+7+7=499. Anonymous V feature orientation does not certify actual realized-volatility ordering; retain the proxy interpretation.

## Identical-selector Raw-Control

**A Raw-Control:** 97 raw Level candidates (94 anonymous and three official lagged fields); no deltas, z-scores, ranks, family levels or interactions in its input universe.

**B Engineered:** at most 499 candidates: five representations per raw input, seven alpha family levels and seven High-V interactions.

Both invoke the **same** selection/composite/evaluation algorithm: training eligibility; two earlier-purged 252-row inner blocks; two finite, nonzero, same-sign Spearman ICs; rank by absolute mean inner IC; training-only greedy |rho|≥0.90 redundancy; retain at most 16; training-fitted normalization; fixed direction alignment and equal weights; untouched outer validation. Only the candidate universe is the intended primary difference.

The question is: **Does the engineered candidate universe provide incremental out-of-sample predictive information relative to an identically selected raw-feature universe?** Phase 3 remains the immutable interpretable raw individual/composite benchmark. The nested Raw-Control composite, not a best historical individual IC, is the fair Phase 4 comparator. No Phase 3 PASS whitelist is imposed.

## Prospective synthetic calibration

Two 252-row blocks give noisy IC estimates; choosing 16 from up to 499 candidates can overfit inner selection. Same-sign screening, redundancy and an outer block reduce but do not eliminate this risk. Freeze the following calibration before implementation/results, without changing K or screening thresholds.

Every case uses 7860 artificial consecutive rows, the frozen 11 outer folds and both full selectors. Raw inputs combine within-family AR(1) factors (coefficient 0.8), independent innovations, continuous within-family loading giving approximately 0.5 contemporaneous correlation, and Bernoulli D fields. The target under the null is independent IID standard normal; official lagged fields are reconstructed with offset −1. Outcome fields other than target are independent artificial series.

**Pure noise:** 20 fixed seeds: 101,211,307,401,503,601,701,809,907,1009,1103,1201,1301,1409,1511,1601,1709,1801,1901,2003. Record each pipeline's outer pooled Rank IC, mean/median fold IC, positive-fold ratio, worst/best fold, best apparent inner IC and selected count. Summarize across seeds with mean, sample std, p05/p50/p95, min/max.

The engineering alarm is `abs(mean of seed mean-fold ICs) > max(0.02, 3×sample_std/sqrt(20))`, checked separately for A/B. This is a prospective software-bias investigation trigger, **not** a feature IC cutoff, p-value, market discovery standard or proof of nominal error control. If triggered, investigate implementation leakage; if no bug explains it, stop without freezing the pipeline or changing the selector. Do not select/drop seeds or retune K, thresholds or windows to clear it.

**Planted cases:** seeds [11,22,33], with independent target noise std=1:

- Raw: y=0.35×M1−0.25×E1+0.25×P1+noise. Recover planted sources with the correct direction in some outer folds and positive mean OOS IC. Engineered must not systematically destroy the signal.
- Transformed: replace M1 with `0.02×date_id+epsilon_t`, epsilon IID N(0,1); y=`0.6×(epsilon_t−epsilon_(t−21))/sqrt(2)+noise`. Early missing endpoints remain missing. Test Delta21 recovery and average engineered advantage over Raw-Control across all three fixed seeds; do not claim that finite raw IC must be exactly zero.
- Interaction: latent V follows AR(1)=0.97, V members have loading 1 and independent noise std=0.5. P members share an IID signal plus independent noise std=0.2. y=`0.8×P_common×1[latent_V>0]+noise`. Recover the P/HighV interaction in some folds with positive OOS IC, and prove regime invariance to y mutations. The planted latent state is not supplied to the fitted pipeline.

Calibration outputs are **synthetic only**. Record generator assumptions, all seeds, versions/hashes and failures. Recovery tests describe implementation behavior, not real-alpha efficacy. No synthetic result can change the frozen research universe or selection rules; only documented code bugs may be corrected and revalidated.

## Leakage, audit and execution requirements

Before real execution, tests must cover future-X mutation for rolling features, outer-y selection isolation, X-only regime invariance even under training-y mutation, training-only maps, eligibility after transformation, missingness, deterministic tie ordering, paired A/B evaluation, sign alignment and rejected-pair audit trails. Constant/undefined correlations and unavailable regimes retain the inherited policies, not ad hoc fallbacks.

Phase 4's implementation must freeze separately at `phase4-engineered-pipeline-v1` after full tests and calibration. Real entry requires a clean checkout at that code tag, verified protocol-v3.1 tag/hash, bounded existing research loader and an atomic canonical single-run state independent of output path. No real Phase 4/5 data are read by this protocol task. No model is trained. Holdout and Final remain prospectively locked under the v3 caveats about historical descriptive EDA and prior post-hoc observations.

Protocol v3.1 was frozen before any real Phase 4 engineered-alpha results were observed.

The Raw-Control and Engineered pipelines use the same nested selector; the candidate universe is the intended primary difference.

Internal Holdout 7860–8867 and Final Simulation 8868–9047 remain untouched by this amendment or any new predictive confirmation.
