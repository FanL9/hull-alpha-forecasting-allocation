# Data dictionary

Source: [official competition data](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data).

Individual numbered predictors are **anonymous features**. The project brief supplies family labels; exact economic definitions and prediction-time availability require official documentation. No field-level meaning is inferred from a number.

| Prefix | Family taxonomy |
| --- | --- |
| M | Market dynamics / technical |
| E | Macro economic |
| I | Interest rate |
| P | Price / valuation |
| V | Volatility |
| S | Sentiment |
| MOM | Momentum |
| D | Dummy / binary |

Observed shape: train 9048 × 98; test 10 × 99.

| Observed column | Train dtype | Test dtype | Role | Definition |
| --- | --- | --- | --- | --- |
| date_id | int64 | int64 | time index | Official definition / availability review pending |
| D1 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| D2 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| D3 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| D4 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| D5 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| D6 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| D7 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| D8 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| D9 | int64 | int64 | anonymous feature | anonymous feature; Dummy / binary |
| E1 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E10 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E11 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E12 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E13 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E14 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E15 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E16 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E17 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E18 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E19 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E2 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E20 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E3 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E4 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E5 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E6 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E7 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E8 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| E9 | float64 | float64 | anonymous feature | anonymous feature; Macro economic |
| I1 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| I2 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| I3 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| I4 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| I5 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| I6 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| I7 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| I8 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| I9 | float64 | float64 | anonymous feature | anonymous feature; Interest rate |
| M1 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M10 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M11 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M12 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M13 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M14 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M15 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M16 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M17 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M18 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M2 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M3 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M4 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M5 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M6 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M7 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M8 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| M9 | float64 | float64 | anonymous feature | anonymous feature; Market dynamics / technical |
| P1 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P10 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P11 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P12 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P13 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P2 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P3 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P4 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P5 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P6 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P7 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P8 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| P9 | float64 | float64 | anonymous feature | anonymous feature; Price / valuation |
| S1 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S10 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S11 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S12 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S2 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S3 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S4 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S5 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S6 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S7 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S8 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| S9 | float64 | float64 | anonymous feature | anonymous feature; Sentiment |
| V1 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V10 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V11 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V12 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V13 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V2 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V3 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V4 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V5 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V6 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V7 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V8 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| V9 | float64 | float64 | anonymous feature | anonymous feature; Volatility |
| forward_returns | float64 | absent | outcome / unavailable as contemporaneous predictor | Official definition / availability review pending |
| risk_free_rate | float64 | absent | outcome / unavailable as contemporaneous predictor | Official definition / availability review pending |
| market_forward_excess_returns | float64 | absent | outcome / unavailable as contemporaneous predictor | Official definition / availability review pending |
| is_scored | absent | bool | evaluation metadata | Official definition / availability review pending |
| lagged_forward_returns | absent | float64 | lagged field; availability requires official API verification | Official definition / availability review pending |
| lagged_risk_free_rate | absent | float64 | lagged field; availability requires official API verification | Official definition / availability review pending |
| lagged_market_forward_excess_returns | absent | float64 | lagged field; availability requires official API verification | Official definition / availability review pending |

## Observed schema qualifications

A missing family is not reconstructed from assumptions. MOM-family count in this training file: 0.

| D field | Observed training values | Unique nonmissing |
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

`market_forward_excess_returns` is the planned forecasting label. Its precise construction must be verified against official documentation; it is **not assumed equal** to `forward_returns - risk_free_rate`.

`forward_returns` and `risk_free_rate` are retained for later economic accounting. Their row alignment, units, timing and financing interpretation remain to be verified. Lagged labels, when present, are excluded until the official API establishes their availability.

The official metric, its volatility penalty and precise field-availability semantics remain unverified. A 1.20 reference in configuration is a research requirement, not a reproduced official score. The downloaded inference gateway is reviewed separately in `docs/official_api_notes.md` without being executed.
