#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build manuscript-facing result tables from seed-level outputs.

This closes the provenance chain:
run_strengthening.py -> seed-level CSVs -> manuscript-facing CSVs in results/.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

NOMINAL=0.90

TRANSFER={
    "A_KonIQ_to_CID2013":"KonIQ-10k → CID2013",
    "B_CID2013_to_KonIQ":"CID2013 → KonIQ-10k",
    "C_KonIQ_to_SPAQ":"KonIQ-10k → SPAQ",
    "D_SPAQ_to_KonIQ":"SPAQ → KonIQ-10k",
    "E_CID2013_to_SPAQ":"CID2013 → SPAQ",
    "F_SPAQ_to_CID2013":"SPAQ → CID2013",
}
ORDER=list(TRANSFER)


def _wilcoxon(x, alternative):
    x=np.asarray(x,float)-1.0
    try:
        return float(wilcoxon(x,alternative=alternative,zero_method="wilcox").pvalue)
    except Exception:
        return np.nan


def _sign_test(x, alternative):
    d=np.asarray(x,float)-1.0
    d=d[np.isfinite(d)]
    d=d[np.abs(d)>1e-15]
    if len(d)==0:
        return 1.0
    if alternative=="less":
        k=int(np.sum(d<0))
    elif alternative=="greater":
        k=int(np.sum(d>0))
    else:
        raise ValueError(alternative)
    return float(binomtest(k,n=len(d),p=.5,alternative="greater").pvalue)


def _require_seed_count(g, expected, label):
    n=int(g.seed.nunique())
    if expected is not None and n!=expected:
        raise ValueError(f"{label}: found {n} unique seeds, expected {expected}")
    return n


def build_rbf(classical_seed, expected_seeds=100):
    df=pd.read_csv(classical_seed)
    gdf=df[(df.predictor=="rbf_svr")&(df.method=="q40")].copy()
    rows=[]
    for direction in ORDER:
        g=gdf[gdf.direction==direction].copy()
        _require_seed_count(g,expected_seeds,direction)
        if g.empty:
            raise ValueError(f"missing {direction}")
        isv=g.interval_score_mean.to_numpy(float)
        ce=g.coverage_error.to_numpy(float)
        w=g.clipped_width_mean.to_numpy(float)
        rows.append(dict(
            direction=direction,
            transfer=TRANSFER[direction],
            n_eval=int(np.median(g.n_eval)),
            coverage_median=float(np.median(g.coverage)),
            coverage_q25=float(np.quantile(g.coverage,.25)),
            coverage_q75=float(np.quantile(g.coverage,.75)),
            coverage_p2_5=float(np.quantile(g.coverage,.025)),
            coverage_p97_5=float(np.quantile(g.coverage,.975)),
            coverage_error_median=float(np.median(ce)),
            seed_coverage_error_le_0_05_rate=float(np.mean(ce<=.05)),
            seed_coverage_ge_0_90_rate=float(np.mean(g.coverage.to_numpy(float)>=.90)),
            width_median=float(np.median(w)),
            interval_score_median=float(np.median(isv)),
            interval_score_q25=float(np.quantile(isv,.25)),
            interval_score_q75=float(np.quantile(isv,.75)),
            interval_score_p2_5=float(np.quantile(isv,.025)),
            interval_score_p97_5=float(np.quantile(isv,.975)),
            seed_IS_lt_1_rate=float(np.mean(isv<1)),
            seed_operational_pass_rate=float(np.mean((ce<=.05)&(w<1)&(isv<1))),
            wilcoxon_IS_less_than_1_p=_wilcoxon(isv,"less"),
            wilcoxon_IS_greater_than_1_p=_wilcoxon(isv,"greater"),
            sign_test_IS_less_than_1_p=_sign_test(isv,"less"),
            sign_test_IS_greater_than_1_p=_sign_test(isv,"greater"),
        ))
    out=pd.DataFrame(rows)
    out=out.rename(columns={"seed_coverage_error_le_0_05_rate":"seed_coverage_error_le_0.05_rate"})
    return out


def build_clipiqa(clip_seed, expected_seeds=20):
    df=pd.read_csv(clip_seed)
    if "method" in df.columns:
        df=df[df.method=="clip_q40"].copy()
    rows=[]
    for direction in ORDER:
        g=df[df.direction==direction].copy()
        _require_seed_count(g,expected_seeds,direction)
        if g.empty:
            raise ValueError(f"missing {direction}")
        isv=g.interval_score_mean.to_numpy(float)
        ce=g.coverage_error.to_numpy(float)
        w=g.clipped_width_mean.to_numpy(float)
        med_cov=float(np.median(g.coverage))
        med_ce=float(np.median(ce))
        med_w=float(np.median(w))
        med_is=float(np.median(isv))
        rows.append({
            "Transfer":TRANSFER[direction],
            "Target SRCC":float(np.median(g.target_srcc)),
            "Raw PLCC":float(np.median(g.target_plcc_raw)),
            "KRCC":float(np.median(g.target_krcc_tau_b)),
            "40-MOS coverage":med_cov,
            "Coverage error":med_ce,
            "Clipped width":med_w,
            "Interval score":med_is,
            "Seed IS<1 rate":float(np.mean(isv<1)),
            "Seed operational PASS rate":float(np.mean((ce<=.05)&(w<1)&(isv<1))),
            "Wilcoxon p (IS<1)":_wilcoxon(isv,"less"),
            "Sign-test p (IS<1)":_sign_test(isv,"less"),
            "Direction-level informative":bool((med_ce<=.05)&(med_w<1)&(med_is<1)),
        })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--classical-seed",type=Path,required=True,
                    help="100-seed classical_seed_metrics.csv")
    ap.add_argument("--clip-seed",type=Path,required=True,
                    help="20-seed CLIPIQA_seed_metrics.csv")
    ap.add_argument("--outdir",type=Path,default=Path("results_generated"))
    ap.add_argument("--expect-classical-seeds",type=int,default=100)
    ap.add_argument("--expect-clip-seeds",type=int,default=20)
    args=ap.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)

    rbf=build_rbf(args.classical_seed,args.expect_classical_seeds)
    clip=build_clipiqa(args.clip_seed,args.expect_clip_seeds)

    rbf_path=args.outdir/"rbf_svr_100seed_stability.csv"
    clip_path=args.outdir/"clipiqa_20seed_summary.csv"
    rbf.to_csv(rbf_path,index=False)
    clip.to_csv(clip_path,index=False)
    print("[DONE]",rbf_path)
    print("[DONE]",clip_path)


if __name__=="__main__":
    main()
