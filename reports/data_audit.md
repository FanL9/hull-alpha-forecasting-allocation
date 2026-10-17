# Official data audit

Status: **complete**

Generated (UTC): 2026-10-16T09:00:31.345011+00:00

Source: [Kaggle — Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data).

## Dataset shape

| File | Rows | Columns |
| --- | --- | --- |
| train | 9048 | 98 |
| test | 10 | 99 |

## Missingness overview

| File | Columns with missing values | Missing cells |
| --- | --- | --- |
| train | 85 | 137675 |
| test | 0 | 0 |

## train: feature families

Family labels follow the project taxonomy; individual fields remain anonymous features.

| Prefix | Family | Count | Actual columns |
| --- | --- | --- | --- |
| M | Market dynamics / technical | 18 | M1, M10, M11, M12, M13, M14, M15, M16, M17, M18, M2, M3, M4, M5, M6, M7, M8, M9 |
| E | Macro economic | 20 | E1, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E2, E20, E3, E4, E5, E6, E7, E8, E9 |
| I | Interest rate | 9 | I1, I2, I3, I4, I5, I6, I7, I8, I9 |
| P | Price / valuation | 13 | P1, P10, P11, P12, P13, P2, P3, P4, P5, P6, P7, P8, P9 |
| V | Volatility | 13 | V1, V10, V11, V12, V13, V2, V3, V4, V5, V6, V7, V8, V9 |
| S | Sentiment | 12 | S1, S10, S11, S12, S2, S3, S4, S5, S6, S7, S8, S9 |
| MOM | Momentum | 0 | none |
| D | Dummy / binary | 9 | D1, D2, D3, D4, D5, D6, D7, D8, D9 |

## train: full column and missingness audit

| Column | dtype | Role | Missing | Missing % | Inf |
| --- | --- | --- | --- | --- | --- |
| date_id | int64 | time index | 0 | 0.0000 | 0 |
| D1 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D2 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D3 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D4 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D5 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D6 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D7 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D8 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D9 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| E1 | float64 | anonymous feature | 1784 | 19.7171 | 0 |
| E10 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E11 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E12 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E13 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E14 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E15 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E16 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E17 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E18 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E19 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E2 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E20 | float64 | anonymous feature | 1616 | 17.8603 | 0 |
| E3 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E4 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E5 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E6 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E7 | float64 | anonymous feature | 6969 | 77.0225 | 0 |
| E8 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| E9 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I1 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I2 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I3 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I4 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I5 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I6 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I7 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I8 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| I9 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M1 | float64 | anonymous feature | 5547 | 61.3064 | 0 |
| M10 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M11 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M12 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M13 | float64 | anonymous feature | 5540 | 61.2290 | 0 |
| M14 | float64 | anonymous feature | 5540 | 61.2290 | 0 |
| M15 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M16 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M17 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M18 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M2 | float64 | anonymous feature | 3217 | 35.5548 | 0 |
| M3 | float64 | anonymous feature | 2018 | 22.3033 | 0 |
| M4 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M5 | float64 | anonymous feature | 3283 | 36.2843 | 0 |
| M6 | float64 | anonymous feature | 5043 | 55.7361 | 0 |
| M7 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M8 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| M9 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P1 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P10 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P11 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P12 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P13 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P2 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P3 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P4 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P5 | float64 | anonymous feature | 1574 | 17.3961 | 0 |
| P6 | float64 | anonymous feature | 1638 | 18.1034 | 0 |
| P7 | float64 | anonymous feature | 1616 | 17.8603 | 0 |
| P8 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| P9 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| S1 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| S10 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| S11 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| S12 | float64 | anonymous feature | 3537 | 39.0915 | 0 |
| S2 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| S3 | float64 | anonymous feature | 5733 | 63.3621 | 0 |
| S4 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| S5 | float64 | anonymous feature | 1511 | 16.6998 | 0 |
| S6 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| S7 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| S8 | float64 | anonymous feature | 3009 | 33.2560 | 0 |
| S9 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V1 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V10 | float64 | anonymous feature | 6049 | 66.8546 | 0 |
| V11 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V12 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V13 | float64 | anonymous feature | 1511 | 16.6998 | 0 |
| V2 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V3 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V4 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V5 | float64 | anonymous feature | 1512 | 16.7109 | 0 |
| V6 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V7 | float64 | anonymous feature | 1511 | 16.6998 | 0 |
| V8 | float64 | anonymous feature | 1006 | 11.1185 | 0 |
| V9 | float64 | anonymous feature | 4539 | 50.1658 | 0 |
| forward_returns | float64 | outcome / unavailable as contemporaneous predictor | 0 | 0.0000 | 0 |
| risk_free_rate | float64 | outcome / unavailable as contemporaneous predictor | 0 | 0.0000 | 0 |
| market_forward_excess_returns | float64 | outcome / unavailable as contemporaneous predictor | 0 | 0.0000 | 0 |

## train: missingness by family and input-order block

| Family | Missing % | Block missing percentages (chronological input order) |
| --- | --- | --- |
| M | 25.330336968268004 | 0:2262 = 65.47%, 2262:4524 = 27.08%, 4524:6786 = 8.78%, 6786:9048 = 0.00% |
| E | 15.180702917771882 | 0:2262 = 50.32%, 2262:4524 = 5.00%, 4524:6786 = 5.00%, 6786:9048 = 0.40% |
| I | 11.118479221927497 | 0:2262 = 44.47%, 2262:4524 = 0.00%, 4524:6786 = 0.00%, 6786:9048 = 0.00% |
| P | 12.6572808270421 | 0:2262 = 50.63%, 2262:4524 = 0.00%, 4524:6786 = 0.00%, 6786:9048 = 0.00% |
| V | 19.69836087873223 | 0:2262 = 58.17%, 2262:4524 = 15.38%, 4524:6786 = 5.24%, 6786:9048 = 0.00% |
| S | 20.113100501031536 | 0:2262 = 60.22%, 2262:4524 = 15.78%, 4524:6786 = 4.45%, 6786:9048 = 0.00% |
| MOM | None |  |
| D | 0.0 | 0:2262 = 0.00%, 2262:4524 = 0.00%, 4524:6786 = 0.00%, 6786:9048 = 0.00% |

## train: time structure

| Check | Observed |
| --- | --- |
| present | True |
| missing_count | 0 |
| nonnumeric_or_nonfinite_count | 0 |
| duplicate_count | 0 |
| monotonic_increasing | True |
| integer_like | True |
| minimum | 0 |
| maximum | 9047 |
| non_unit_step_count | 0 |

## train: observed D-family encodings

These are observed value sets, not inferred economic meanings. A two-state -1/0 field is binary and is not recoded to match a 0/1 assumption.

| Column | Observed values (up to 20) | Unique nonmissing |
| --- | --- | --- |
| D1 | [0, 1] | 2 |
| D2 | [0, 1] | 2 |
| D3 | [0, 1] | 2 |
| D4 | [0, 1] | 2 |
| D5 | [0, 1] | 2 |
| D6 | [-1, 0] | 2 |
| D7 | [0, 1] | 2 |
| D8 | [0, 1] | 2 |
| D9 | [0, 1] | 2 |

## test: feature families

Family labels follow the project taxonomy; individual fields remain anonymous features.

| Prefix | Family | Count | Actual columns |
| --- | --- | --- | --- |
| M | Market dynamics / technical | 18 | M1, M10, M11, M12, M13, M14, M15, M16, M17, M18, M2, M3, M4, M5, M6, M7, M8, M9 |
| E | Macro economic | 20 | E1, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E2, E20, E3, E4, E5, E6, E7, E8, E9 |
| I | Interest rate | 9 | I1, I2, I3, I4, I5, I6, I7, I8, I9 |
| P | Price / valuation | 13 | P1, P10, P11, P12, P13, P2, P3, P4, P5, P6, P7, P8, P9 |
| V | Volatility | 13 | V1, V10, V11, V12, V13, V2, V3, V4, V5, V6, V7, V8, V9 |
| S | Sentiment | 12 | S1, S10, S11, S12, S2, S3, S4, S5, S6, S7, S8, S9 |
| MOM | Momentum | 0 | none |
| D | Dummy / binary | 9 | D1, D2, D3, D4, D5, D6, D7, D8, D9 |

## test: full column and missingness audit

| Column | dtype | Role | Missing | Missing % | Inf |
| --- | --- | --- | --- | --- | --- |
| date_id | int64 | time index | 0 | 0.0000 | 0 |
| D1 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D2 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D3 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D4 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D5 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D6 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D7 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D8 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| D9 | int64 | anonymous feature | 0 | 0.0000 | 0 |
| E1 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E10 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E11 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E12 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E13 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E14 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E15 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E16 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E17 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E18 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E19 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E2 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E20 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E3 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E4 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E5 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E6 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E7 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E8 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| E9 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I1 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I2 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I3 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I4 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I5 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I6 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I7 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I8 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| I9 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M1 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M10 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M11 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M12 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M13 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M14 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M15 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M16 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M17 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M18 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M2 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M3 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M4 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M5 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M6 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M7 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M8 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| M9 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P1 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P10 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P11 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P12 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P13 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P2 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P3 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P4 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P5 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P6 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P7 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P8 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| P9 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S1 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S10 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S11 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S12 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S2 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S3 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S4 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S5 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S6 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S7 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S8 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| S9 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V1 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V10 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V11 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V12 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V13 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V2 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V3 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V4 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V5 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V6 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V7 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V8 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| V9 | float64 | anonymous feature | 0 | 0.0000 | 0 |
| is_scored | bool | evaluation metadata | 0 | 0.0000 | 0 |
| lagged_forward_returns | float64 | lagged field; availability requires official API verification | 0 | 0.0000 | 0 |
| lagged_risk_free_rate | float64 | lagged field; availability requires official API verification | 0 | 0.0000 | 0 |
| lagged_market_forward_excess_returns | float64 | lagged field; availability requires official API verification | 0 | 0.0000 | 0 |

## test: missingness by family and input-order block

| Family | Missing % | Block missing percentages (chronological input order) |
| --- | --- | --- |
| M | 0.0 | 0:2 = 0.00%, 2:5 = 0.00%, 5:7 = 0.00%, 7:10 = 0.00% |
| E | 0.0 | 0:2 = 0.00%, 2:5 = 0.00%, 5:7 = 0.00%, 7:10 = 0.00% |
| I | 0.0 | 0:2 = 0.00%, 2:5 = 0.00%, 5:7 = 0.00%, 7:10 = 0.00% |
| P | 0.0 | 0:2 = 0.00%, 2:5 = 0.00%, 5:7 = 0.00%, 7:10 = 0.00% |
| V | 0.0 | 0:2 = 0.00%, 2:5 = 0.00%, 5:7 = 0.00%, 7:10 = 0.00% |
| S | 0.0 | 0:2 = 0.00%, 2:5 = 0.00%, 5:7 = 0.00%, 7:10 = 0.00% |
| MOM | None |  |
| D | 0.0 | 0:2 = 0.00%, 2:5 = 0.00%, 5:7 = 0.00%, 7:10 = 0.00% |

## test: time structure

| Check | Observed |
| --- | --- |
| present | True |
| missing_count | 0 |
| nonnumeric_or_nonfinite_count | 0 |
| duplicate_count | 0 |
| monotonic_increasing | True |
| integer_like | True |
| minimum | 8980 |
| maximum | 8989 |
| non_unit_step_count | 0 |

## test: observed D-family encodings

These are observed value sets, not inferred economic meanings. A two-state -1/0 field is binary and is not recoded to match a 0/1 assumption.

| Column | Observed values (up to 20) | Unique nonmissing |
| --- | --- | --- |
| D1 | [0] | 1 |
| D2 | [0] | 1 |
| D3 | [0, 1] | 2 |
| D4 | [0] | 1 |
| D5 | [0, 1] | 2 |
| D6 | [0] | 1 |
| D7 | [0, 1] | 2 |
| D8 | [0, 1] | 2 |
| D9 | [0, 1] | 2 |

## Train/test relationship

| Check | Observed |
| --- | --- |
| overlapping_date_ids | 10 |
| test_strictly_after_train | False |

| Schema difference | Observed columns |
| --- | --- |
| Train only | forward_returns, market_forward_excess_returns, risk_free_rate |
| Test only | is_scored, lagged_forward_returns, lagged_market_forward_excess_returns, lagged_risk_free_rate |

## Official evaluation/API files

These files are inventoried, not executed. File presence does not establish a scoring formula or label availability; see docs/official_api_notes.md for the separate source review.

- `kaggle_evaluation/__init__.py`
- `kaggle_evaluation/core/__init__.py`
- `kaggle_evaluation/core/base_gateway.py`
- `kaggle_evaluation/core/generated/__init__.py`
- `kaggle_evaluation/core/generated/kaggle_evaluation_pb2.py`
- `kaggle_evaluation/core/generated/kaggle_evaluation_pb2_grpc.py`
- `kaggle_evaluation/core/kaggle_evaluation.proto`
- `kaggle_evaluation/core/relay.py`
- `kaggle_evaluation/core/templates.py`
- `kaggle_evaluation/default_gateway.py`
- `kaggle_evaluation/default_inference_server.py`

## Errors

None.

## Warnings and research safeguards

- train: D-family encoding is not 0/1: {'D6': [-1, 0]}; preserve observed values. Two distinct values still constitute a binary encoding.
- test: constant nonmissing columns: ['D1', 'D2', 'D4', 'D6', 'S7']
- No MOM-prefixed features are present in the actual training CSV; do not invent a momentum family or infer momentum meaning from another anonymous field.
- Official test rows are not strictly after training rows; do not treat the supplied test file as an independent final holdout.
- Official test.csv is an API/demo input until official evaluation files establish its meaning; final research holdout must come from the chronological tail of labelled training history.

A `date_id` is an ordered identifier; no calendar mapping or daily continuity is inferred. Missing features are not filled during the audit. Outcome fields are never predictors. No split, target-performance analysis or final-holdout evaluation has run.
