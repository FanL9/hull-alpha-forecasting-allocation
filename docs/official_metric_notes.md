# Official Hull metric verification

Authority: [competition Evaluation](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview/evaluation) links to [Kaggle's Hull Competition Sharpe](https://www.kaggle.com/code/metric/hull-competition-sharpe). The latter was retrieved directly using the official CLI, from the `metric` account (kernel ID 92174585). The code was inspected before executing it on artificial inputs only. No competition data was downloaded or scored.

| Evidence | SHA-256 |
| --- | --- |
| Retrieved notebook | `a0f9633183ac52b912b30439fbf91df2899c843016c9c8f00c1b8af7410f2e20` |
| Extracted code cells | `19e64d96a713bf31c63cfea45a4416906dbd87ce3e770ed0b2e658460b257d13` |

The original notebook remains ignored in `work/phase25/official_metric/`; it contains no data. Portable provenance is in `configs/official_metric.yaml`. There is no reliance on participant reimplementations. This pins the retrieved implementation rather than claiming it can never change.

## Exact arithmetic

For already-selected scored rows, let a be the submitted `prediction`, r be `forward_returns`, and f be `risk_free_rate`:

```text
R_strategy[t] = f[t] × (1 − a[t]) + a[t] × r[t]
e_strategy[t] = R_strategy[t] − f[t]
e_market[t]   = r[t] − f[t]
g_strategy   = product(1 + e_strategy)^(1/n) − 1
g_market     = product(1 + e_market)^(1/n) − 1
sigma_strategy = sample_std(R_strategy, ddof=1)
sigma_market   = sample_std(r, ddof=1)
Sharpe = sqrt(252) × g_strategy / sigma_strategy
V_strategy = 100 × sqrt(252) × sigma_strategy
V_market   = 100 × sqrt(252) × sigma_market
P_vol = 1 + max(0, V_strategy / V_market − 1.20)
gap = max(0, (g_market − g_strategy) × 100 × 252)
P_return = 1 + gap^2 / 100
score = min(Sharpe / (P_vol × P_return), 1,000,000)
```

These are geometric means of **excess** daily returns, not total-wealth CAGR or an arithmetic-mean Sharpe. The denominator uses total strategy-return std, not excess-return std. The return gap is scaled into annualized percentage points before squaring. All those distinctions are preserved.

## Bounds, cash and exceptions

- Numeric allocations must lie in **[0,2]**; out-of-bounds values raise, not clip. The 1.20 ratio is a **soft penalty threshold**, not a hard rejection or a volatility target to optimize now.
- Cash earns the provided rate; allocation above one gives a negative cash weight at that same rate. The official metric adds no transaction costs or separate borrowing spread. Later economic evaluation must report those separately, without calling a modified score official.
- Exactly zero strategy or market std raises `ParticipantVisibleError`. At allocation=0 with constant cash rates, the official score is undefined; with varying cash rates and positive market std, excess mean is zero and the score is zero. Do not add an epsilon or redefine these cases.
- Only the upper score is capped. Negative scores and the publisher's penalty behavior are preserved.
- The official function ignores its `row_id_column_name` argument and does not mask `is_scored`. Future evaluation must select scored rows and ensure chronological alignment before calling it.

`src/evaluation/kaggle_metric.py::score` preserves official valid-input numeric arithmetic and pandas label alignment. It does not mutate its arguments, unlike the published function. `metric_components` exposes intermediate quantities for diagnostics. `evaluate_allocations` is a stricter local adapter rejecting mismatched lengths and nonfinite inputs. That adapter's validation is an explicit addition; the official implementation has permissive NaN/alignment behavior and is not redefined as robust to malformed input.

## Verification

`tests/fixtures/kaggle_metric_synthetic_cases.json` contains **42 entirely artificial cases**, with expected scores/errors obtained from the pinned official code. Cases cover constant exposure, cash, leverage, negative returns, the upper cap, pandas alignment, bounds and random artificial arrays. Additional tests independently check the arithmetic, volatility/return penalties and undefined cash case. No real strategy score, forecast accuracy, holding-period return or final-period statistic is generated.

`market_forward_excess_returns` is absent from the metric's arithmetic. Changing it must not change the score. Forecast research and portfolio scoring remain separate layers.
