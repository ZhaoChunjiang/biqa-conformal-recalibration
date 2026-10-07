#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source-size-matched intervention audit.

All source domains are restricted to n=474 per seed:
- CID2013: all 474 images, preserving original order (anchor by construction).
- KonIQ-10k: 474 images sampled without replacement using seed + 61001.
- SPAQ: 474 images sampled without replacement using seed + 62003.

Each matched pool is split 284/95/95 (train/calibration/source-test) by the
same split_source() rule used in the primary analysis. Target databases remain
full. The primary target recalibration budget is q40.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

from run_strengthening import (
    FEATURE_NAMES, DIRECTIONS, SOURCE_TO_TARGETS,
    split_source, target_perm, finite_q, make_predictor,
    interval_metrics, point_metrics, load_feature_data
)

MATCH_N=474
POOL_OFFSETS={"KonIQ-10k":61001,"SPAQ":62003,"CID2013":63007}
PRIMARY_DIRECTIONS={"A_KonIQ_to_CID2013","C_KonIQ_to_SPAQ",
                    "D_SPAQ_to_KonIQ","F_SPAQ_to_CID2013"}


def select_pool(source,n,seed):
    if source=="CID2013":
        if n!=MATCH_N:
            raise ValueError("CID2013 anchor must contain exactly 474 images")
        return np.arange(n,dtype=int)
    rng=np.random.default_rng(seed+POOL_OFFSETS[source])
    return rng.choice(n,size=MATCH_N,replace=False).astype(int)


def fit_one_source(df, source, seed, matched=True):
    X=df[FEATURE_NAMES].to_numpy(float)
    y=df.y.to_numpy(float)
    if matched:
        pool=select_pool(source,len(df),seed)
    else:
        pool=np.arange(len(df),dtype=int)

    Xp=X[pool]; yp=y[pool]
    tr,cal,st=split_source(len(pool),seed)
    model=make_predictor("rbf_svr",seed)
    model.fit(Xp[tr],yp[tr])
    pcal=model.predict(Xp[cal])
    source_q=finite_q(np.abs(yp[cal]-pcal))
    pst=model.predict(Xp[st])
    source_test=point_metrics(yp[st],pst)

    membership=pd.DataFrame({
        "source":source,
        "seed":seed,
        "dataset_index":pool,
        "role":""
    })
    membership.loc[tr,"role"]="train"
    membership.loc[cal,"role"]="cal"
    membership.loc[st,"role"]="source_test"
    membership["matched_pool"]=matched
    return model,source_q,source_test,membership


def summarize(seed_df):
    rows=[]
    for direction,g in seed_df.groupby("direction"):
        ce=g.coverage_error.to_numpy(float)
        w=g.clipped_width_mean.to_numpy(float)
        isv=g.interval_score_mean.to_numpy(float)
        pass_seed=(ce<=.05)&(w<1)&(isv<1)
        row=dict(
            direction=direction,
            source=g.source.iloc[0],
            target=g.target.iloc[0],
            n_seeds=int(g.seed.nunique()),
            source_test_srcc_median=float(np.median(g.source_test_srcc)),
            target_srcc_median=float(np.median(g.target_srcc)),
            source_q_median=float(np.median(g.source_q)),
            q40_median=float(np.median(g.q)),
            coverage_median=float(np.median(g.coverage)),
            coverage_error_median=float(np.median(ce)),
            clipped_width_mean_median=float(np.median(w)),
            interval_score_mean_median=float(np.median(isv)),
            seed_operational_pass_rate=float(np.mean(pass_seed)),
            direction_level_informative=bool(
                (np.median(ce)<=.05)&(np.median(w)<1)&(np.median(isv)<1)
            ),
            anchor_only=bool(g.source.iloc[0]=="CID2013"),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def interpretation(summary):
    m={r.direction:bool(r.direction_level_informative)
       for r in summary.itertuples()}
    retained=[m[d] for d in sorted(PRIMARY_DIRECTIONS)]
    if all(retained):
        label="source_size_only_explanation_not_supported"
    elif not any(retained):
        label="strong_source_size_only_pattern"
    else:
        label="mixed_source_size_and_domain_effect"
    return {
        "primary_directions":sorted(PRIMARY_DIRECTIONS),
        "n_preserved_pass":int(sum(retained)),
        "n_lost":int(len(retained)-sum(retained)),
        "interpretation":label,
        "CID2013_source_rows":"anchor_only_identical_by_construction",
    }


def paired_compare(matched_df, full_df):
    rows=[]
    metrics=["target_srcc","coverage","clipped_width_mean","interval_score_mean"]
    for direction in sorted(set(matched_df.direction)&set(full_df.direction)):
        m=matched_df[matched_df.direction==direction].sort_values("seed")
        f=full_df[full_df.direction==direction].sort_values("seed")
        if list(m.seed)!=list(f.seed):
            raise ValueError(f"seed mismatch {direction}")
        for metric in metrics:
            delta=m[metric].to_numpy(float)-f[metric].to_numpy(float)
            try:
                p=float(wilcoxon(delta,alternative="two-sided",zero_method="wilcox").pvalue)
            except Exception:
                p=np.nan
            rows.append(dict(
                direction=direction,metric=metric,
                matched_median=float(np.median(m[metric])),
                full_median=float(np.median(f[metric])),
                paired_delta_median=float(np.median(delta)),
                wilcoxon_two_sided_p=p,
                anchor_only=bool(m.source.iloc[0]=="CID2013"),
            ))
    return pd.DataFrame(rows)


def run(data,seeds,outdir,compare_full=False):
    matched_rows=[]; full_rows=[]; memberships=[]
    for source,targets in SOURCE_TO_TARGETS.items():
        src=data[source]
        for seed in tqdm(range(seeds),desc=f"matched474|{source}"):
            model,sq,sm,mem=fit_one_source(src,source,seed,matched=True)
            memberships.append(mem)
            for direction,target in targets:
                tgt=data[target]
                Xt=tgt[FEATURE_NAMES].to_numpy(float); yt=tgt.y.to_numpy(float)
                pt=model.predict(Xt)
                pm=point_metrics(yt,pt)
                perm=target_perm(len(tgt),seed)
                cal40=perm[:40]; ev=perm[40:]
                q40=finite_q(np.abs(yt[cal40]-pt[cal40]))
                im=interval_metrics(yt[ev],pt[ev],q40)
                matched_rows.append(dict(
                    direction=direction,source=source,target=target,seed=seed,
                    source_n=MATCH_N,train_n=284,cal_n=95,source_test_n=95,
                    source_q=sq,source_test_srcc=sm["srcc"],
                    target_srcc=pm["srcc"],target_plcc_raw=pm["plcc_raw"],
                    target_krcc_tau_b=pm["krcc_tau_b"],**im
                ))

            if compare_full:
                if source=="CID2013":
                    # Exact anchor: matched and full are the same experiment by construction.
                    for row in [r for r in matched_rows if r["source"]==source and r["seed"]==seed]:
                        full_rows.append(dict(row))
                else:
                    fm,fsq,fsm,_=fit_one_source(src,source,seed,matched=False)
                    for direction,target in targets:
                        tgt=data[target]
                        Xt=tgt[FEATURE_NAMES].to_numpy(float); yt=tgt.y.to_numpy(float)
                        pt=fm.predict(Xt)
                        pm=point_metrics(yt,pt)
                        perm=target_perm(len(tgt),seed)
                        cal40=perm[:40]; ev=perm[40:]
                        q40=finite_q(np.abs(yt[cal40]-pt[cal40]))
                        im=interval_metrics(yt[ev],pt[ev],q40)
                        full_rows.append(dict(
                            direction=direction,source=source,target=target,seed=seed,
                            source_n=len(src),train_n=len(split_source(len(src),seed)[0]),
                            cal_n=len(split_source(len(src),seed)[1]),
                            source_test_n=len(split_source(len(src),seed)[2]),
                            source_q=fsq,source_test_srcc=fsm["srcc"],
                            target_srcc=pm["srcc"],target_plcc_raw=pm["plcc_raw"],
                            target_krcc_tau_b=pm["krcc_tau_b"],**im
                        ))

    mdf=pd.DataFrame(matched_rows)
    mdf.to_csv(outdir/"source_size_matched_seed_metrics.csv",index=False)
    msum=summarize(mdf)
    msum.to_csv(outdir/"source_size_matched_summary.csv",index=False)
    pd.concat(memberships,ignore_index=True).to_csv(
        outdir/"source_size_matched_membership.csv",index=False
    )
    interp=interpretation(msum)
    (outdir/"source_size_interpretation.json").write_text(
        json.dumps(interp,indent=2),encoding="utf-8"
    )

    if compare_full:
        fdf=pd.DataFrame(full_rows)
        fdf.to_csv(outdir/"source_size_full_seed_metrics.csv",index=False)
        summarize(fdf).to_csv(outdir/"source_size_full_summary.csv",index=False)
        paired_compare(mdf,fdf).to_csv(
            outdir/"source_size_full_vs_matched_paired.csv",index=False
        )

    return msum,interp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--koniq-csv",type=Path,default=Path("data/KonIQ_BRISQUE36_NIQE.csv"))
    ap.add_argument("--cid-csv",type=Path,default=Path("data/CID2013_BRISQUE36_NIQE.csv"))
    ap.add_argument("--spaq-csv",type=Path,default=Path("data/SPAQ_BRISQUE36_NIQE.csv"))
    ap.add_argument("--seeds",type=int,default=100)
    ap.add_argument("--outdir",type=Path,default=Path("outputs/source_size_matched"))
    ap.add_argument("--compare-full",action="store_true",
                    help="also rerun full-source counterparts and paired comparisons")
    args=ap.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise RuntimeError(f"{args.outdir} is non-empty; use a fresh output directory")
    args.outdir.mkdir(parents=True,exist_ok=True)
    cfg={
        "seeds":args.seeds,
        "matched_source_n":MATCH_N,
        "pool_offsets":POOL_OFFSETS,
        "target_permutation_offset":202604,
        "compare_full":bool(args.compare_full),
    }
    (args.outdir/"run_config.json").write_text(
        json.dumps(cfg,indent=2),encoding="utf-8"
    )
    data=load_feature_data(args.koniq_csv,args.cid_csv,args.spaq_csv)
    summary,interp=run(data,args.seeds,args.outdir,args.compare_full)
    print(summary.to_string(index=False))
    print(json.dumps(interp,indent=2))


if __name__=="__main__":
    main()
