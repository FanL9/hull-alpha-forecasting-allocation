# Research questions and evidence plan

Status: research design only. No empirical conclusion is available before acquisition and chronological validation.

| Question | Planned evidence | Interpretation safeguard |
| --- | --- | --- |
| Q1: Are future excess returns predictable? | Development walk-forward correlations, directional accuracy and univariate forecasts versus historical mean | In-sample correlation is exploratory, not alpha; account for feature search |
| Q2: Is candidate alpha stable? | Rolling correlations, coefficient signs, fold-level predictive direction, temporal family summaries | Report sign reversals and uncertainty; overlapping windows are dependent |
| Q3: Does effectiveness depend on regimes? | Signal and portfolio outcomes conditional on causal volatility and trend states | Thresholds use earlier history; small conditional samples cannot establish a regime effect |
| Q4: Does ML beat linear baselines? | Identical development folds, labels, eligible fields, tuning budgets and forecast metrics | Compare Ridge/ElasticNet before nonlinear complexity; freeze selection before final test |
| Q5: Do improved forecasts improve investment results? | Matched prediction metrics and net portfolio outcomes under fixed sizing rules | Exposure and risk differences can explain returns; a lower RMSE is insufficient |
| Q6: Does volatility-aware sizing improve risk-adjusted outcomes? | Linear, threshold and volatility-adjusted mappings with cost and leverage sensitivities | Compare realized risk, turnover, drawdown and financing assumptions, not Sharpe alone |

## Alpha investigations

For each observed feature, use `X[j,t]` to forecast the verified future label attached to row `t`. Establish the row/label alignment from official documentation before implementing shifts. Distinguish delayed signals from longer-horizon compounded future returns. Only construct the latter when the return sequence and availability permit it; otherwise report that the longer-horizon question is not identifiable.

Report all examined families, candidate selection procedures, fold sample sizes and unsuccessful findings. Family labels do not establish the economic meaning of any numbered predictor. A feature may remain a statistical candidate without an interpretable economic story.

## Reporting standard

Negative and unstable results are valid outcomes. Answer each question using actual out-of-sample evidence with uncertainty, competing explanations and implementation limits. No results will be inserted to fit the project narrative.
