# Frozen robustness protocol

This protocol is frozen **before** the formal robustness-audit results are inspected.

## 1. Scope

The robustness audit addresses a pre-specified set of high-value reliability questions. It does not expand to a fourth database and does not attempt to turn the study into a new conformal-method benchmark.

The manuscript claim is downgraded from **failure boundary** to **informative operating regime** unless new evidence justifies stronger language.

Recommended revised title:

**Few-Shot Conformal Recalibration under Cross-Database Shift: When Does Coverage Recovery Remain Informative in Blind Image Quality Assessment?**

## 2. Existing predictors retained

- RBF-SVR: the frozen configuration used in the primary experiments.
- ExtraTrees: the frozen sensitivity configuration used before formal results were inspected.
- Same 37-D BRISQUE36+NIQE representation.
- Same six directed transfers.
- Same source split and target permutation rule.
- Same 90% nominal level.
- No target predictor retraining for the primary q-only method.

## 3. New modern learned representation

Use **base CLIP-IQA** (AAAI 2023), not CLIP-IQA+.

Reasons for this choice:
- modern vision-language NR-IQA;
- zero-shot / no IQA-label task-specific training in the base method;
- avoids using a KonIQ-trained checkpoint that could leak one of the study databases into the representation;
- can run on CPU, with GPU optional.

The frozen CLIP-IQA score is converted to a database-specific point predictor only through a **source-training-only nonnegative affine head**:
`y_hat = a * clipiqa_score + b`, with `a >= 0`.

The source head is fitted on the same 60% source-training split. Conformal source calibration uses the same 20% source-calibration split. Target q40 uses only the first 40 target MOS values. No target MOS is used to fit the CLIP-IQA head.

### Interpretation rule
Exact 4-PASS/2-FAIL replication is **not required**.

- If at least one modern-model direction has coverage recovery but IS >= 1, this supports the existence of a non-informative operating regime beyond the hand-crafted representation.
- If all six modern-model directions are informative, the manuscript will say the hand-crafted-predictor failures are not universal and that a stronger learned representation can move transfers into the informative regime.
- Either result is retained; no model or prompt is changed after inspection.

## 4. New fair 40-label baselines

### 4.1 Robust global-scale baseline
Use the same source q, multiplied by:
`median(target_40 absolute residual) / median(source-calibration absolute residual)`.

This isolates whether the main gain is essentially residual-scale correction.

### 4.2 Affine + conformal, same total label budget
Use exactly 40 target labels:
- first 20: fit affine point-score calibration;
- next 20: estimate conformal residual q;
- common evaluation begins after target index 40.

This tests whether MOS location/scale mismatch explains the apparent uncertainty failure.

The primary q40 method still uses all first 40 labels for q estimation, so the affine baseline is deliberately conservative while respecting the same total label budget.

## 5. MOS and residual diagnostics

Report:
- normalized MOS histogram and quantiles for all three databases;
- source-calibration residual quantiles vs full-target residual quantiles;
- q inflation;
- affine slope/intercept for the 20+20 baseline.

No claim that [0,1] normalization makes subjective scales exchangeable is permitted.

## 6. Raw vs clipped intervals

For y in [0,1], clipping an interval to [0,1] does not change the coverage event:
`y in [L,U]  <=>  y in [L,U] ∩ [0,1]`.

The robustness audit must verify raw and clipped coverage numerically for every seed/direction, while reporting both raw and clipped widths.

## 7. Conditional coverage

For primary q40, report conditional coverage, width and interval score by:
- target MOS quintile;
- predicted-score quintile.

The manuscript will describe q40 as restoring **marginal** coverage unless conditional results also support stronger language.

## 8. Seed dispersion and score-vs-baseline testing

For every direction report:
- median;
- IQR;
- empirical 2.5–97.5 percentile range;
- seed rate with IS < 1;
- seed operational PASS rate;
- one-sided paired/signed Wilcoxon test of seed-level IS against 1.

The operational criterion remains descriptive and pre-specified, not a universal statistical boundary.

## 9. Label-budget sensitivity

For RBF-SVR and ExtraTrees, run 10/20/40/80 target labels on a **common evaluation set after the first 80 target observations**.

The original 40-label main result remains primary and is not replaced.

## 10. Source-size-matched intervention audit

The source-size audit is implemented in `run_source_size_audit.py`.

- Every source domain is matched to n=474 per seed.
- CID2013 uses all 474 images and therefore serves as an **identical-by-construction anchor** for B/E.
- KonIQ-10k uses sampling without replacement with RNG offset 61001.
- SPAQ uses sampling without replacement with RNG offset 62003.
- Each 474-image source pool is split into 284 train / 95 calibration / 95 source-test samples.
- Targets remain full and use the same target permutation offset 202604.
- The primary target recalibration budget remains q40.
- Inference about source-size intervention is based on A/C/D/F, the four directions whose large source is actually downsampled.
- The optional `--compare-full` mode reruns the full-source counterparts and writes paired comparisons.

## 11. Frozen after formal robustness-audit results

- changing CLIP-IQA to another modern model;
- changing CLIP prompts;
- switching to CLIP-IQA+;
- retuning RBF-SVR or ExtraTrees;
- changing the affine/scale baselines;
- changing alpha;
- changing target permutations;
- dropping seeds or directions;
- redefining the informative criterion.
