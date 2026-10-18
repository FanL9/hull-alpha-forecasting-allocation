# Project Flow

This is the execution order from official data to the final competition simulation; statuses distinguish completed work from future research.

```text
0 ---------------- 7859 | 7860 -------- 8867 | 8868 ---- 9047
   Research / WF          Internal Holdout       Final Simulation
```

1. **Official Data Acquisition — Done**
   Acquire the official Kaggle files and verify their integrity with SHA-256 hashes.

2. **Data Audit — Done**
   Check schema, chronological order, missingness, duplicates and outcome fields without changing the source data.

3. **Development Split — Done**
   Use a chronological tail reservation to lock the last 180 rows and assign the earlier 8868 rows to development.

4. **Descriptive EDA — Done**
   Describe development missingness, distributions, temporal drift, families, X–X correlations and the target separately without testing X–Y relationships.

5. **Research Protocol Freeze — Frozen**
   Fix the inner holdout, expanding windows, purge, feature availability, inference, FDR and evaluation rules in executable configuration before research.

6. **Walk-Forward Alpha Research — Not started**
   Evaluate 97 single features and 7 pre-registered family composites with training-only eligibility and direction, using block-wise Spearman Rank IC (primary) and winsorized Pearson IC (secondary) across 11 expanding chronological folds.

7. **Statistical Validation — Not started**
   Assess out-of-sample associations and fold stability using HAC p-values, block-bootstrap intervals, family-specific BH-FDR, 104-hypothesis global BH and BY sensitivity.

8. **Forecasting Model — Not started**
   Compare forecasts of market_forward_excess_returns against the historical-training-mean baseline using only research-window information.

9. **Internal Holdout Test — Not started**
   Evaluate the selected, frozen forecasting pipeline once on date_id 7860–8867 without further tuning.

10. **Signal and Allocation — Not started**
    Calibrate forecasts into allocations in [0,2] using research data while keeping holdout outcomes outside design decisions.

11. **Official Evaluation — Not started**
    Compare research-period strategy and baseline returns with the reproduced Kaggle metric under frozen allocation rules.

12. **Competition Simulation — Not started**
    Run the frozen strategy sequentially on date_id 8868–9047 using only information available at each inference call.
