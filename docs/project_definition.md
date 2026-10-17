# Project definition

Regime-Aware Alpha Forecasting and Dynamic S&P 500 Allocation is an independent research project using only official Hull Tactical data. Its future goal is anonymous-feature alpha research → expected-return forecasting → market regimes → signals → bounded S&P 500 exposure → portfolio and risk evaluation. “Jane Street-style” refers to the intended emphasis on alpha discovery/stability, not proprietary methods or endorsement.

## Current accepted scope

Phase 1 is complete and preserved: official acquisition, safe extraction, hashes, audit, dictionary, tests and credential protection. Phase 2 has been rebuilt as Data Processing + EDA + Research Split Design. Earlier Phase 2 alpha/model code, notebooks, outputs and claims were withdrawn.

Development date_id 0..8867 contains 8868 rows. Competition Simulation date_id 8868..9047 contains 180 locked rows. Only count, identifiers and header compatibility may be checked in the latter. Official test.csv is inference-schema/API material only. Neither source is an independent model test set for this phase.

Phase 2 preserves 94 anonymous X columns, date_id, all three outcomes, original encodings and NaNs; reports development quality, distributions, missing intervals, family scales, temporal drift, X–X structure and separate target-only descriptions. No predictive relationship, model, signal, allocation or backtest is computed. The source dataset and historical Phase 1 outputs remain unchanged.

## Future scope and open decisions

Alpha stability, regime-aware forecasting and dynamic allocation remain future work. Their models, fold count, internal holdout length and tuning rules are not chosen now. Future development uses expanding-window walk-forward only inside development, followed by a defined internal holdout and strategy freeze before separately authorized competition simulation. Existing feature publication timing, target construction, scoring and calendar semantics still require verification.

All evidence must come from actual official data and an appropriate future evaluation protocol. Anonymous feature numbers have no inferred economic definition. Descriptive X–X correlation and target distribution are not alpha, causality or performance. No Kaggle submission, automated rules acceptance, external dataset, live execution or GitHub push is included in this rebuild.
