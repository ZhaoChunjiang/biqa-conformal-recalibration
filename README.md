# Cross-database BIQA conformal recalibration — reproducibility package

This repository accompanies the manuscript:

**Few-Shot Conformal Recalibration under Cross-Database Shift: When Does Coverage Recovery Remain Informative in Blind Image Quality Assessment?**

## What this repository reproduces

This is an **analysis-level reproducibility package** for the paper's cross-database calibration and robustness experiments.

It includes:
- the frozen RBF-SVR and ExtraTrees analysis code operating on the 37-D BRISQUE36+NIQE feature caches;
- the base CLIP-IQA score extractor and source-only affine-head sensitivity analysis;
- q40, scale40, affine20+q20, label-budget, conditional-coverage, and seed-level diagnostics;
- the frozen protocol, seed rules, software environment record, and representative summary tables;
- a unit test that locks the interval-score definition and q40 finite-sample rank.

It does **not** redistribute KonIQ-10k, CID2013, or SPAQ images.

## Evidence hierarchy

- Main RBF-SVR / ExtraTrees q40 experiments: seeds `0..99`.
- Mechanism, budget, conditional-coverage, and CLIP-IQA sensitivity analyses: fixed seeds `0..19`.
- Target permutation for seed `s`: `np.random.default_rng(s + 202604)`.
- Nominal coverage: 0.90 (`alpha = 0.10`).

The 100-seed and 20-seed analyses should be run into separate output directories.

## Core files

- `run_strengthening.py` — classical predictors, q40 recalibration, equal-budget diagnostics, label-budget and conditional-coverage analyses.
- `extract_clipiqa_scores.py` — frozen base CLIP-IQA score extraction.
- `FROZEN_PROTOCOL.md` — protocol frozen before the formal robustness results were inspected.
- `SEEDS.md` — random-seed and target-permutation rules.
- `test_interval_score.py` — interval-score, clipping, and q40-rank unit tests.
- `environment.json` / `requirements.txt` — environment record and install requirements.
- `REPRODUCIBILITY_SCOPE.md` — explicit scope and known provenance limitations.
- `results/` — representative summary tables only.

## Critical interval-score definition

For miscoverage level `alpha`:

`IS = (U-L) + (2/alpha)(L-y) 1[y<L] + (2/alpha)(y-U) 1[y>U]`

There is **no outer division by alpha**. Therefore the full-range interval `[0,1]` has score 1 whenever `y in [0,1]`.

Run:

```bash
python test_interval_score.py
```

Expected output:

```text
All reproducibility unit tests passed.
```

## Expected feature-cache schema

The classical analysis expects three CSV files with:
- `y`: normalized MOS in `[0,1]`;
- `brisque_00` ... `brisque_35`: 36 BRISQUE NSS features;
- `niqe`: one NIQE feature;
- optionally `name` or `image` for sample identifiers.

By default the script looks for:

```text
data/KonIQ_BRISQUE36_NIQE.csv
data/CID2013_BRISQUE36_NIQE.csv
data/SPAQ_BRISQUE36_NIQE.csv
```

You can override all paths on the command line.

## Reproducing the reported analyses

### Structural self-test

```bash
python run_strengthening.py --mode selftest --outdir outputs/selftest
```

### Main 100-seed q40 experiment

```bash
python run_strengthening.py \
  --mode formal \
  --koniq-csv data/KonIQ_BRISQUE36_NIQE.csv \
  --cid-csv data/CID2013_BRISQUE36_NIQE.csv \
  --spaq-csv data/SPAQ_BRISQUE36_NIQE.csv \
  --seeds 100 \
  --outdir outputs/primary_100seed
```

This command also computes the diagnostics for 100 seeds. The paper's lower-cost diagnostic analyses use the separate 20-seed run below.

### 20-seed diagnostics

```bash
python run_strengthening.py \
  --mode formal \
  --koniq-csv data/KonIQ_BRISQUE36_NIQE.csv \
  --cid-csv data/CID2013_BRISQUE36_NIQE.csv \
  --spaq-csv data/SPAQ_BRISQUE36_NIQE.csv \
  --seeds 20 \
  --outdir outputs/diagnostics_20seed
```

### CLIP-IQA sensitivity analysis

First extract one CLIP-IQA cache per database with `extract_clipiqa_scores.py`, then run:

```bash
python run_strengthening.py \
  --mode formal \
  --koniq-csv data/KonIQ_BRISQUE36_NIQE.csv \
  --cid-csv data/CID2013_BRISQUE36_NIQE.csv \
  --spaq-csv data/SPAQ_BRISQUE36_NIQE.csv \
  --clip-koniq data/KonIQ_CLIPIQA.csv \
  --clip-cid data/CID2013_CLIPIQA.csv \
  --clip-spaq data/SPAQ_CLIPIQA.csv \
  --seeds 20 \
  --outdir outputs/clipiqa_20seed
```

## Data and feature-cache boundary

Raw benchmark images are intentionally omitted. The 37-D BRISQUE36+NIQE caches used by the classical analysis are also not redistributed in this repository; see `REPRODUCIBILITY_SCOPE.md` for the resulting provenance boundary.

## PLCC

Reported PLCC is raw PLCC without target-domain nonlinear/logistic fitting. Cross-database point-transfer conclusions emphasize SRCC and KRCC.

## Citation

See `CITATION.cff`.

## License

No explicit software license has been granted in this repository. The code is public for scholarly inspection and verification; contact the author for reuse permissions unless and until a license is added.
