# Final Simulation (date_id 8868–9047): published files

One run under Protocol v5. A was refit on 0–8866 (purge 8867) and frozen. Rows were processed in order, and forecasts were sealed before any Final outcome was attached. Interpretation: [Final Project Report](../final_project_report.md).

| File | Content |
| --- | --- |
| `final_summary.json` | Official score of the research track, competition track, `constant_1` and `volatility_only` |
| `final_scores.csv` | The same, with mean/min/max position, turnover and 0/1/5 bp cost sensitivity |
| `final_rank_ic.csv` | Descriptive 180-day Rank IC of A (no test) |
| `final_forecasts_sealed.csv` | Sealed forecasts: date_id, pipeline, prediction |
| `final_daily_positions.csv` | Volatility state c and daily positions (no returns) |
| `final_fit_audit.json` | Refit state: selected features, signs, map hashes |
| `final_run_manifest.json`, `phase7_final_state.json`, `snapshot_seal.json` | Run identity, state and output hashes |

Task caches and the pending audit copy stay in the local archive and are not published.
