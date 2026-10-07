# Result summaries

This directory contains compact, manuscript-facing summaries rather than complete per-seed raw outputs.

- `rbf_svr_100seed_stability.csv` — 100-seed q40 RBF-SVR stability and interval-score statistics.
- `clipiqa_20seed_summary.csv` — 20-seed frozen base CLIP-IQA representation-sensitivity summary.
- `mos_distribution_summary.csv` — normalized MOS distribution summaries for the three benchmark databases.

The two manuscript-facing uncertainty tables are generated deterministically from seed-level outputs by `make_results_tables.py`; they are not intended to be independent hand-edited result sources. The same builder also writes `interval_score_robustness_tests.csv`, which adds exact sign/binomial tests alongside the Wilcoxon signed-rank diagnostics.

The source-size-matched audit is implemented by `run_source_size_audit.py`; its outputs are written to the user-selected output directory because the full per-seed audit is larger than the compact snapshots retained here.

The 100-seed primary evidence and 20-seed robustness/sensitivity evidence are intentionally distinguished. No raw benchmark images are redistributed here.
