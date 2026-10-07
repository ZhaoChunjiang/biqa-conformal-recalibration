# Reproducibility scope and provenance boundaries

This repository is intentionally transparent about what is and is not reproduced end to end.

## Fully specified in this repository

The following analysis logic is executable from the required caches:

- source 60/20/20 split and target permutation rules;
- RBF-SVR and ExtraTrees configurations;
- split-conformal finite-sample residual quantile;
- 40-label q-only recalibration;
- global residual-scale baseline;
- 20+20 affine-plus-conformal diagnostic;
- 10/20/40/80 target-label sensitivity;
- MOS- and prediction-quintile conditional-coverage diagnostics;
- proper interval score and the `[0,1]` uninformative reference;
- base CLIP-IQA score extraction and source-only nonnegative affine head;
- seed-level summaries and Wilcoxon tests.

## External benchmark data

KonIQ-10k, CID2013, and SPAQ are third-party benchmark datasets. Raw images are not redistributed here. Users should obtain them from the original providers and comply with the applicable licenses and terms.

## BRISQUE36+NIQE feature-cache boundary

The classical analysis consumes frozen 37-D CSV caches (`36 BRISQUE NSS + 1 NIQE`) with normalized MOS.

Those caches are not redistributed in this repository, and this repository does not currently include a complete raw-image-to-BRISQUE36+NIQE cache-generation pipeline. Therefore the public package provides **analysis-level reproducibility from the frozen feature caches**, not full end-to-end reconstruction of the classical representation from raw images.

This boundary is stated explicitly rather than inferring or recreating an unverified feature-extraction implementation.

## CLIP-IQA environment provenance

The historical CLIP-IQA extraction used the frozen base `clipiqa` model through IQA-PyTorch on a Tesla T4, with all images resized to 512×384 by bicubic interpolation.

The exact historical `pyiqa`, `torch`, and `torchvision` package versions were not captured in the original extraction log. No version is guessed here. The current `extract_clipiqa_scores.py` records these versions automatically for future reruns.

## Result tables

The `results/` directory contains compact summaries supporting manuscript verification. It intentionally omits raw benchmark images and large per-seed/intermediate files.

## Randomness

- Main experiments: seeds 0–99.
- Diagnostic/sensitivity experiments: seeds 0–19.
- Target permutation: `np.random.default_rng(seed + 202604)`.

No direction or seed is dropped based on observed results.
