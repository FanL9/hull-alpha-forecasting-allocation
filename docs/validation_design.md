# Research split design

Status: Phase 2.5 protocol frozen. Split boundaries are implemented and software-tested; no real-data alpha/model validation has been executed. See [research protocol](research_protocol.md) for the authoritative rules and rationale.

| Area | Inclusive date_id | Rows | Current permitted use |
| --- | --- | --- | --- |
| Walk-Forward Research | 0..7859 | 7860 | Future chronological research |
| Internal Holdout | 7860..8867 | 1008 | Prospectively reserved evaluation |
| Competition Simulation | 8868..9047 | 180 | Headers/IDs/count metadata only |

Research and Internal Holdout together retain the original 8868-row Development Area. The outer split and all Phase 2 outputs are unchanged. configs/split_policy.json and artifacts/split_definition.json preserve the final 180-row firewall. configs/validation.yaml adds the frozen inner design. The future research loader reads only nrows=7860; historical EDA used nrows=8868 and is not repeated now.

The final 180 rows cannot inform missing-value handling, feature engineering, selection, EDA, regimes, model/tuning/allocation choices or backtest optimization. No simulation dtype inference, missingness or target summary is allowed. Shared train headers provide schema compatibility; value checks wait for a future authorized simulation. Official test.csv is inference-schema-only and never a model test set.

Frozen future sequence, not research results:

1. Expand training from 5087 initial rows through 11 validation blocks of 252, with step 252, purge 1 and embargo 0, inside 0..7859. Models/hyperparameters remain unselected.
2. Enforce training-only availability and the frozen statistical/multiplicity protocol. No real X–Y calculation has run.
3. Fit preprocessing, feature selection, regime thresholds and models inside each historical training window only; evaluate strictly later rows. No random split.
4. After selection freeze, separately authorize one Internal Holdout evaluation on 7860..8867. Refit can use 0..7858; purge 7859 and freeze throughout holdout.
5. Freeze the strategy, implementation and any sequential update schedule.
6. Separately authorize and run the 180-row Kaggle-style competition simulation under that frozen protocol.

The annualization convention is 252 trading observations; the 1008-row holdout comprises four such blocks, longer than the 180-row simulation. The initial history absorbs a 47-row remainder, so no short final validation fold is created. The 252-row validation length is a research-design choice, independent of Phase 2 descriptive rolling settings. One-row purge follows verified one-day labels; forward-only training needs no additional embargo. The outer Development/Simulation partition itself still has no gap.

For a nonzero embargo in a future version, the splitter permanently excludes post-validation embargo IDs from subsequent training and skips them in subsequent validation. Software tests verify this behavior. The current embargo is zero.

The original Phase 1 full-file audit and withdrawn previous analyses predate this rebuild. Preserving them as history does not make the new simulation historically unseen. New work enforces a prospective firewall and discloses prior access. Final independent-performance claims will require acknowledging that history.
