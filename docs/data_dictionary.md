# Data Dictionary

Source: [official competition data description](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data).
train.csv has 9048 rows and 98 columns: date_id, 94 anonymous features and three outcomes; test.csv has 10 rows and 99 columns.
The mock test is for schema/interface checks, not independent research evaluation.

## Anonymous features

These seven observed families contain all 94 anonymous inputs; broad family labels do not reveal the exact economic meaning of any numbered field.

| Prefix | Official family label | Count | Observed columns |
| --- | --- | --- | --- |
| M | Market dynamics / technical | 18 | M1, M10, M11, M12, M13, M14, M15, M16, M17, M18, M2, M3, M4, M5, M6, M7, M8, M9 |
| E | Macro economic | 20 | E1, E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E2, E20, E3, E4, E5, E6, E7, E8, E9 |
| I | Interest rate | 9 | I1, I2, I3, I4, I5, I6, I7, I8, I9 |
| P | Price / valuation | 13 | P1, P10, P11, P12, P13, P2, P3, P4, P5, P6, P7, P8, P9 |
| V | Volatility | 13 | V1, V10, V11, V12, V13, V2, V3, V4, V5, V6, V7, V8, V9 |
| S | Sentiment | 12 | S1, S10, S11, S12, S2, S3, S4, S5, S6, S7, S8, S9 |
| D | Dummy / binary | 9 | D1, D2, D3, D4, D5, D6, D7, D8, D9 |

D fields have integer encodings and the other families use floating-point values; missing values are retained.
D6 uses the binary values {-1,0}, while the other observed D fields use {0,1}; no recoding is implied.

## Identifiers and outcomes

| Field | Type | Definition | Availability |
| --- | --- | --- | --- |
| date_id | int64 | Ordered identifier for one trading day | Train and inference metadata; not a predictor |
| forward_returns | float64 | S&P 500 return from buying on the identified day and selling one trading day later | Current training outcome |
| market_forward_excess_returns | float64 | Forward return minus its rolling five-year mean, then MAD-based winsorization with criterion 4 | Primary training target |
| risk_free_rate | float64 | Supplied federal funds rate used directly in official cash-return accounting | Current training outcome/accounting input |
| lagged_forward_returns | float64 | Corresponding forward return with one trading-day lag | Inference-available candidate |
| lagged_market_forward_excess_returns | float64 | Corresponding adjusted target with one trading-day lag | Inference-available candidate |
| lagged_risk_free_rate | float64 | Corresponding cash-rate field with one trading-day lag | Inference-available candidate |
| is_scored | bool | Whether an inference row enters competition scoring | Inference metadata; not a predictor |

## Timing and use

At decision t, current inputs support allocation t and the associated forward return is realized one trading day later; the API's three lagged outcomes reveal t−1, not t.
Use the supplied target on its original row without shifting again or replacing it with forward_returns minus risk_free_rate.
The published target definition specifies a five-year adjustment and MAD criterion, but does not specify exact window endpoints, MAD normalization or intraday executable price cutoffs; no reconstruction is assumed.
Lagged outcomes are allowed future candidates under the [research protocol](research_protocol.md), reconstructed causally from training history rather than copied from mock test values.
Current outcomes and scoring metadata remain outside predictors; keep supplied cash-rate units unchanged when using the [official metric](official_metric_notes.md).
