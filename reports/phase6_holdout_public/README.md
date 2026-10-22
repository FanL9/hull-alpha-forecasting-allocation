# Internal Holdout: approved public evidence

This is a publication subset of the unchanged, complete local archive
`reports/phase6_holdout/`. It includes aggregate forecast/allocation metrics,
block scores, summary, fit audit, the original manifest/state and forecasts.
`holdout_forecasts_sealed.csv` contains only date_id, pipeline and prediction.

Task caches and files containing daily realized forward returns, risk-free rates
or market excess returns are not published. The complete local archive remains
unchanged. Its original files and state remain the authority for the single run.

`snapshot_seal.json` lists SHA-256 for ALL 163 files in that full local archive,
including omitted files and its original seal. The public seal is a separate
document; it does not attempt an impossible self-referential SHA-256.

Every published original is byte-identical to the full local copy. The unchanged
run manifest/state still reference hashes of omitted files. Full verification of
those files requires the retained local archive; no data were removed from it.

The original summary and metrics follow frozen Protocol v4.1. No model, mapping,
confirmation test or selection rule was changed for publication. Final Simulation
8868-9047 remains unopened by this run.
