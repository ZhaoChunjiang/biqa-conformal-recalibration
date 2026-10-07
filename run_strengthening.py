#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-database BIQA conformal recalibration — robustness analyses

Runs only from existing feature caches (no image download required):
1) Re-run RBF-SVR and ExtraTrees on the frozen six-direction protocol.
2) Add 10/20/40/80 label-budget curves on a common evaluation set.
3) Add two fair 40-label baselines:
   - robust global-scale correction of source q;
   - 20-label affine score calibration + 20-label conformal residual calibration.
4) Add MOS and source/target residual-distribution diagnostics.
5) Add conditional coverage/width by MOS quintile and predicted-score quintile.
6) Verify raw-vs-clipped coverage identity for y in [0,1].
7) Add seed dispersion, IS<1 rates and one-sided Wilcoxon tests against IS=1.
8) Optionally evaluate CLIP-IQA score caches as a modern representation sensitivity:
   source train fits ONLY a nonnegative affine head from frozen CLIP-IQA score to MOS.

No hyperparameters or decision rules may be changed after formal results are seen.
"""

from __future__ import annotations
import argparse, json, math, platform, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, pearsonr, kendalltau, wilcoxon, binomtest
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from tqdm import tqdm

ALPHA=0.10
NOMINAL=0.90
DEFAULT_SEEDS=100
FEATURE_NAMES=[f"brisque_{i:02d}" for i in range(36)] + ["niqe"]
BUDGETS=[10,20,40,80]

DIRECTIONS=[
    ("A_KonIQ_to_CID2013","KonIQ-10k","CID2013"),
    ("B_CID2013_to_KonIQ","CID2013","KonIQ-10k"),
    ("C_KonIQ_to_SPAQ","KonIQ-10k","SPAQ"),
    ("D_SPAQ_to_KonIQ","SPAQ","KonIQ-10k"),
    ("E_CID2013_to_SPAQ","CID2013","SPAQ"),
    ("F_SPAQ_to_CID2013","SPAQ","CID2013"),
]
SOURCE_TO_TARGETS={}
for d,s,t in DIRECTIONS:
    SOURCE_TO_TARGETS.setdefault(s,[]).append((d,t))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)


def split_source(n, seed):
    ids=np.arange(n)
    tr,tmp=train_test_split(ids,test_size=.40,random_state=seed,shuffle=True)
    cal,st=train_test_split(tmp,test_size=.50,random_state=seed+10000,shuffle=True)
    return tr,cal,st


def target_perm(n, seed):
    return np.random.default_rng(seed+202604).permutation(n)


def prepare_formal_outdir(outdir, args, clip_enabled):
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    cfg={
        "mode":"formal",
        "seeds":int(args.seeds),
        "alpha":ALPHA,
        "koniq_csv":str(args.koniq_csv),
        "cid_csv":str(args.cid_csv),
        "spaq_csv":str(args.spaq_csv),
        "clip_enabled":bool(clip_enabled),
        "clip_koniq":None if args.clip_koniq is None else str(args.clip_koniq),
        "clip_cid":None if args.clip_cid is None else str(args.clip_cid),
        "clip_spaq":None if args.clip_spaq is None else str(args.clip_spaq),
    }
    cfg_path=outdir/"run_config.json"
    existing=[p for p in outdir.iterdir() if p.name!="run_config.json"]
    if cfg_path.exists():
        old=json.loads(cfg_path.read_text(encoding="utf-8"))
        if old!=cfg:
            raise RuntimeError(
                f"Output directory {outdir} already belongs to a different run configuration. "
                "Use a fresh output directory to avoid mixing 100-seed and 20-seed evidence."
            )
    elif existing:
        raise RuntimeError(
            f"Output directory {outdir} is non-empty but has no run_config.json. "
            "Use a fresh output directory to prevent accidental result mixing."
        )
    cfg_path.write_text(json.dumps(cfg,indent=2),encoding="utf-8")


def finite_q(abs_resid, alpha=ALPHA):
    x=np.asarray(abs_resid,float)
    x=x[np.isfinite(x)]
    if len(x)==0: raise ValueError("no residuals")
    k=min(max(int(math.ceil((len(x)+1)*(1-alpha))),1),len(x))
    return float(np.sort(x)[k-1])


def safe_srcc(y,p):
    return float(spearmanr(y,p,nan_policy="omit").statistic)


def safe_plcc(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    m=np.isfinite(y)&np.isfinite(p)
    y=y[m]; p=p[m]
    if len(y)<2 or np.ptp(y)<=1e-15 or np.ptp(p)<=1e-15:
        return np.nan
    return float(pearsonr(y,p).statistic)


def safe_krcc(y,p):
    return float(kendalltau(y,p,variant="b",nan_policy="omit").statistic)


def point_metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    e=y-p
    return dict(
        srcc=safe_srcc(y,p), plcc_raw=safe_plcc(y,p), krcc_tau_b=safe_krcc(y,p),
        mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e*e))),
        pred_outside_01_rate=float(np.mean((p<0)|(p>1)))
    )


def interval_arrays(y,p,q):
    y=np.asarray(y,float); p=np.asarray(p,float); q=float(q)
    rawL=p-q; rawU=p+q
    L=np.clip(rawL,0,1); U=np.clip(rawU,0,1)
    raw_inside=(y>=rawL)&(y<=rawU)
    clip_inside=(y>=L)&(y<=U)
    width=U-L
    penalty=(2/ALPHA)*(L-y)*(y<L)+(2/ALPHA)*(y-U)*(y>U)
    iscore=width+penalty
    return rawL,rawU,L,U,raw_inside,clip_inside,width,iscore,penalty


def interval_metrics(y,p,q):
    arr=interval_arrays(y,p,q)
    rawL,rawU,L,U,ri,ci,width,iscore,pen=arr
    cov=float(np.mean(ci))
    return dict(
        n_eval=int(len(y)), q=float(q),
        raw_coverage=float(np.mean(ri)),
        clipped_coverage=cov,
        raw_clipped_coverage_absdiff=float(abs(np.mean(ri)-cov)),
        coverage=cov, coverage_error=float(abs(cov-NOMINAL)),
        raw_width_mean=float(np.mean(rawU-rawL)),
        clipped_width_mean=float(np.mean(width)),
        full_span_rate=float(np.mean(width>=1-1e-12)),
        near_full_08_rate=float(np.mean(width>=.8-1e-12)),
        interval_score_mean=float(np.mean(iscore)),
        miss_penalty_mean=float(np.mean(pen)),
    )


def make_predictor(name,seed):
    if name=="rbf_svr":
        return Pipeline([
            ("scaler",StandardScaler()),
            ("svr",SVR(kernel="rbf",C=64.,gamma="scale",epsilon=.02,cache_size=2048))
        ])
    if name=="extra_trees":
        return ExtraTreesRegressor(
            n_estimators=200,min_samples_leaf=3,max_features=1.0,
            bootstrap=False,random_state=seed,n_jobs=-1
        )
    raise ValueError(name)


def validate_feature_cache(df,name,n):
    req={"y",*FEATURE_NAMES}
    miss=req-set(df.columns)
    if miss: raise ValueError(f"{name}: missing {sorted(miss)}")
    if len(df)!=n: raise ValueError(f"{name}: n={len(df)}, expected {n}")
    if not np.all(np.isfinite(df[FEATURE_NAMES+["y"]].to_numpy(float))):
        raise ValueError(f"{name}: nonfinite values")
    if not np.all((df.y.to_numpy(float)>=0)&(df.y.to_numpy(float)<=1)):
        raise ValueError(f"{name}: y outside [0,1]")


def load_feature_data(koniq,cid,spaq):
    data={
        "KonIQ-10k":pd.read_csv(koniq),
        "CID2013":pd.read_csv(cid),
        "SPAQ":pd.read_csv(spaq)
    }
    validate_feature_cache(data["KonIQ-10k"],"KonIQ-10k",10073)
    validate_feature_cache(data["CID2013"],"CID2013",474)
    validate_feature_cache(data["SPAQ"],"SPAQ",11125)
    return data


def load_clip_data(paths):
    out={}
    for name,path,n in [
        ("KonIQ-10k",paths.get("KonIQ-10k"),10073),
        ("CID2013",paths.get("CID2013"),474),
        ("SPAQ",paths.get("SPAQ"),11125),
    ]:
        if path is None: continue
        z=pd.read_csv(path)
        if len(z)!=n or not {"y","clipiqa_raw"}.issubset(z.columns):
            raise ValueError(f"bad CLIP-IQA cache for {name}")
        if not np.isfinite(z[["y","clipiqa_raw"]].to_numpy(float)).all():
            raise ValueError(f"nonfinite CLIP-IQA cache {name}")
        out[name]=z
    return out


def residual_quantiles(x):
    x=np.asarray(x,float)
    qs=[.10,.25,.50,.75,.90,.95]
    return {f"q{int(q*100):02d}":float(np.quantile(x,q)) for q in qs}


def quintile_rows(y,p,q,direction,predictor,seed,bin_by):
    y=np.asarray(y,float); p=np.asarray(p,float)
    rawL,rawU,L,U,ri,ci,width,iscore,pen=interval_arrays(y,p,q)
    v=y if bin_by=="mos" else p
    # rank-based qcut is robust to repeated values
    ranks=pd.Series(v).rank(method="first")
    bins=pd.qcut(ranks,5,labels=False,duplicates="drop").to_numpy()
    rows=[]
    for b in sorted(np.unique(bins)):
        m=bins==b
        rows.append(dict(
            predictor=predictor,direction=direction,seed=seed,
            bin_by=bin_by,quintile=int(b)+1,n=int(m.sum()),
            bin_value_min=float(np.min(v[m])),bin_value_max=float(np.max(v[m])),
            coverage=float(np.mean(ci[m])),
            clipped_width_mean=float(np.mean(width[m])),
            interval_score_mean=float(np.mean(iscore[m])),
        ))
    return rows


def summarize_seed_table(df):
    rows=[]
    grpcols=["predictor","direction","method"]
    for keys,g in df.groupby(grpcols):
        pred,direction,method=keys
        row=dict(predictor=pred,direction=direction,method=method,n_seeds=int(g.seed.nunique()))
        for c in [
            "target_srcc","target_plcc_raw","target_krcc_tau_b","target_mae","target_rmse",
            "coverage","coverage_error","raw_coverage","clipped_coverage",
            "raw_clipped_coverage_absdiff","raw_width_mean","clipped_width_mean",
            "interval_score_mean","full_span_rate","near_full_08_rate","q"
        ]:
            x=g[c].dropna().to_numpy(float)
            if len(x):
                row[c+"_median"]=float(np.median(x))
                row[c+"_q25"]=float(np.quantile(x,.25))
                row[c+"_q75"]=float(np.quantile(x,.75))
                row[c+"_p2_5"]=float(np.quantile(x,.025))
                row[c+"_p97_5"]=float(np.quantile(x,.975))
        if method in {"q40","scale40","affine20_q20","clip_q40"}:
            isv=g.interval_score_mean.to_numpy(float)
            ce=g.coverage_error.to_numpy(float)
            w=g.clipped_width_mean.to_numpy(float)
            row["seed_IS_lt_1_rate"]=float(np.mean(isv<1))
            row["seed_coverage_error_le_0.05_rate"]=float(np.mean(ce<=.05))
            row["seed_operational_pass_rate"]=float(np.mean((ce<=.05)&(w<1)&(isv<1)))
            diff=isv-1.0
            nonzero=diff[np.abs(diff)>1e-15]
            try:
                row["wilcoxon_IS_less_than_1_p"]=float(
                    wilcoxon(diff,alternative="less",zero_method="wilcox").pvalue
                )
                row["wilcoxon_IS_greater_than_1_p"]=float(
                    wilcoxon(diff,alternative="greater",zero_method="wilcox").pvalue
                )
            except Exception:
                row["wilcoxon_IS_less_than_1_p"]=np.nan
                row["wilcoxon_IS_greater_than_1_p"]=np.nan
            if len(nonzero):
                nneg=int(np.sum(nonzero<0))
                npos=int(np.sum(nonzero>0))
                row["sign_test_IS_less_than_1_p"]=float(
                    binomtest(nneg,n=len(nonzero),p=.5,alternative="greater").pvalue
                )
                row["sign_test_IS_greater_than_1_p"]=float(
                    binomtest(npos,n=len(nonzero),p=.5,alternative="greater").pvalue
                )
                row["sign_test_nontie_n"]=int(len(nonzero))
            else:
                row["sign_test_IS_less_than_1_p"]=1.0
                row["sign_test_IS_greater_than_1_p"]=1.0
                row["sign_test_nontie_n"]=0
        rows.append(row)
    return pd.DataFrame(rows)


def run_classical(data,seeds,outdir):
    seedrows=[]; residrows=[]; qrows=[]; budgetrows=[]
    for pname in ["rbf_svr","extra_trees"]:
        for source,targets in SOURCE_TO_TARGETS.items():
            src=data[source]
            Xs=src[FEATURE_NAMES].to_numpy(float); ys=src.y.to_numpy(float)
            for seed in tqdm(range(seeds),desc=f"{pname}|{source}"):
                set_seed(seed)
                tr,cal,st=split_source(len(src),seed)
                model=make_predictor(pname,seed)
                model.fit(Xs[tr],ys[tr])
                pcal=model.predict(Xs[cal])
                qs=finite_q(np.abs(ys[cal]-pcal))
                med_src_res=float(np.median(np.abs(ys[cal]-pcal)))
                for direction,target in targets:
                    tgt=data[target]
                    Xt=tgt[FEATURE_NAMES].to_numpy(float); yt=tgt.y.to_numpy(float)
                    pt=model.predict(Xt)
                    pm=point_metrics(yt,pt)
                    perm=target_perm(len(tgt),seed)

                    # Primary 40-label evaluation, original common eval starts at 40
                    c40=perm[:40]; ev40=perm[40:]
                    target_res40=np.abs(yt[c40]-pt[c40])
                    q40=finite_q(target_res40)
                    scale=np.median(target_res40)/med_src_res if med_src_res>0 else 1.0
                    qscale=qs*scale

                    for method,q,p_eval in [
                        ("q40",q40,pt),
                        ("scale40",qscale,pt),
                    ]:
                        im=interval_metrics(yt[ev40],p_eval[ev40],q)
                        seedrows.append(dict(
                            predictor=pname,direction=direction,source=source,target=target,
                            seed=seed,method=method,
                            **{f"target_{k}":v for k,v in pm.items()},**im
                        ))

                    # Fair 40-label affine+conformal baseline: 20 fit + 20 q
                    fit20=perm[:20]; cal20=perm[20:40]; ev=ev40
                    aff=LinearRegression()
                    aff.fit(pt[fit20,None],yt[fit20])
                    p_aff=aff.predict(pt[:,None])
                    q_aff=finite_q(np.abs(yt[cal20]-p_aff[cal20]))
                    pm_aff=point_metrics(yt,p_aff)
                    im=interval_metrics(yt[ev],p_aff[ev],q_aff)
                    seedrows.append(dict(
                        predictor=pname,direction=direction,source=source,target=target,
                        seed=seed,method="affine20_q20",
                        affine_slope=float(aff.coef_[0]),affine_intercept=float(aff.intercept_),
                        **{f"target_{k}":v for k,v in pm_aff.items()},**im
                    ))

                    # Residual diagnostic quantiles
                    rr=dict(predictor=pname,direction=direction,seed=seed)
                    rr.update({"source_"+k:v for k,v in residual_quantiles(np.abs(ys[cal]-pcal)).items()})
                    rr.update({"target_"+k:v for k,v in residual_quantiles(np.abs(yt-pt)).items()})
                    residrows.append(rr)

                    # Conditional coverage for primary q40
                    qrows += quintile_rows(yt[ev40],pt[ev40],q40,direction,pname,seed,"mos")
                    qrows += quintile_rows(yt[ev40],pt[ev40],q40,direction,pname,seed,"prediction")

                    # 10/20/40/80 budget curve on SAME eval set after 80
                    ev80=perm[80:]
                    for m in BUDGETS:
                        cm=perm[:m]
                        qm=finite_q(np.abs(yt[cm]-pt[cm]))
                        im=interval_metrics(yt[ev80],pt[ev80],qm)
                        budgetrows.append(dict(
                            predictor=pname,direction=direction,source=source,target=target,
                            seed=seed,budget=m,**im
                        ))

    seed_df=pd.DataFrame(seedrows)
    seed_df.to_csv(outdir/"classical_seed_metrics.csv",index=False)
    summarize_seed_table(seed_df).to_csv(outdir/"classical_summary.csv",index=False)
    pd.DataFrame(residrows).to_csv(outdir/"residual_quantiles.csv",index=False)

    qdf=pd.DataFrame(qrows)
    qdf.to_csv(outdir/"conditional_coverage_seed.csv",index=False)
    qsum=qdf.groupby(["predictor","direction","bin_by","quintile"]).agg(
        coverage_median=("coverage","median"),
        coverage_q25=("coverage",lambda x:np.quantile(x,.25)),
        coverage_q75=("coverage",lambda x:np.quantile(x,.75)),
        width_median=("clipped_width_mean","median"),
        interval_score_median=("interval_score_mean","median"),
        n_median=("n","median"),
    ).reset_index()
    qsum.to_csv(outdir/"conditional_coverage_summary.csv",index=False)

    bdf=pd.DataFrame(budgetrows)
    bdf.to_csv(outdir/"budget_curve_seed.csv",index=False)
    bsum_rows=[]
    for (predictor,direction,budget),g in bdf.groupby(["predictor","direction","budget"]):
        bsum_rows.append(dict(
            predictor=predictor,
            direction=direction,
            budget=int(budget),
            coverage_median=float(np.median(g.coverage)),
            coverage_error_median=float(np.median(g.coverage_error)),
            width_median=float(np.median(g.clipped_width_mean)),
            interval_score_median=float(np.median(g.interval_score_mean)),
            IS_lt1_seed_rate=float(np.mean(g.interval_score_mean.to_numpy(float)<1)),
            operational_seed_pass_rate=float(np.mean(
                (g.coverage_error.to_numpy(float)<=.05)&
                (g.clipped_width_mean.to_numpy(float)<1)&
                (g.interval_score_mean.to_numpy(float)<1)
            )),
            n_eval=int(np.median(g.n_eval)),
        ))
    bsum=pd.DataFrame(bsum_rows)
    bsum.to_csv(outdir/"budget_curve_summary.csv",index=False)

    # MOS distribution itself does not depend on seed
    mosrows=[]
    for name,df in data.items():
        y=df.y.to_numpy(float)
        mosrows.append(dict(
            database=name,n=len(y),mean=float(np.mean(y)),std=float(np.std(y,ddof=1)),
            min=float(np.min(y)),q05=float(np.quantile(y,.05)),q25=float(np.quantile(y,.25)),
            median=float(np.median(y)),q75=float(np.quantile(y,.75)),q95=float(np.quantile(y,.95)),
            max=float(np.max(y))
        ))
    pd.DataFrame(mosrows).to_csv(outdir/"MOS_distribution_summary.csv",index=False)

    # figures
    plt.figure(figsize=(7.2,5.0))
    for name,df in data.items():
        plt.hist(df.y.to_numpy(float),bins=30,density=True,histtype="step",linewidth=1.5,label=name)
    plt.xlabel("Normalized MOS")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"Figure_Normalized_MOS_Distributions.png",dpi=300)
    plt.close()

    # budget aggregate plot across directions: median of direction medians
    plt.figure(figsize=(7.2,5.0))
    for pname in ["rbf_svr","extra_trees"]:
        z=bsum[bsum.predictor==pname]
        zz=z.groupby("budget").agg(
            cov=("coverage_median","median"),
            width=("width_median","median"),
            IS=("interval_score_median","median")
        ).reset_index()
        plt.plot(zz.budget,zz.IS,marker="o",label=pname)
    plt.axhline(1.0,linestyle="--",linewidth=1)
    plt.xlabel("Target-label budget")
    plt.ylabel("Median-of-directions interval score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"Figure_LabelBudget_IntervalScore.png",dpi=300)
    plt.close()

    return seed_df


def run_clipiqa(feature_data,clip_data,seeds,outdir):
    if set(clip_data)!=set(feature_data):
        raise ValueError("Need CLIP-IQA caches for all three databases.")
    rows=[]; qrows=[]
    for source,targets in SOURCE_TO_TARGETS.items():
        src=feature_data[source]
        clip_src=clip_data[source]
        if len(src)!=len(clip_src):
            raise ValueError(f"CLIP row-count mismatch {source}")
        # The extractor writes rows in feature-cache order. Verify identifiers
        # explicitly when both caches expose them; otherwise retain the frozen
        # row-order assumption used in the formal run.
        src_name_col = "name" if "name" in src.columns else ("image" if "image" in src.columns else None)
        clip_name_col = "name" if "name" in clip_src.columns else None
        if src_name_col is not None and clip_name_col is not None:
            src_names = src[src_name_col].astype(str).map(lambda x: Path(x).name).reset_index(drop=True)
            clip_names = clip_src[clip_name_col].astype(str).map(lambda x: Path(x).name).reset_index(drop=True)
            if not src_names.equals(clip_names):
                raise ValueError(f"CLIP row-identifier mismatch {source}")
        ys=src.y.to_numpy(float)
        zs=clip_src.clipiqa_raw.to_numpy(float)
        for seed in tqdm(range(seeds),desc=f"clipiqa|{source}"):
            tr,cal,st=split_source(len(src),seed)
            # Frozen modern predictor: zero-shot CLIP-IQA score + SOURCE-ONLY nonnegative affine head
            head=LinearRegression(positive=True)
            head.fit(zs[tr,None],ys[tr])
            ps=head.predict(zs[:,None])
            qs=finite_q(np.abs(ys[cal]-ps[cal]))
            for direction,target in targets:
                yt=feature_data[target].y.to_numpy(float)
                zt=clip_data[target].clipiqa_raw.to_numpy(float)
                pt=head.predict(zt[:,None])
                pm=point_metrics(yt,pt)
                perm=target_perm(len(yt),seed)
                c40=perm[:40]; ev=perm[40:]
                q40=finite_q(np.abs(yt[c40]-pt[c40]))
                im=interval_metrics(yt[ev],pt[ev],q40)
                rows.append(dict(
                    predictor="clipiqa_affine_head",direction=direction,source=source,target=target,
                    seed=seed,method="clip_q40",
                    head_slope=float(head.coef_[0]),head_intercept=float(head.intercept_),
                    source_q=qs,**{f"target_{k}":v for k,v in pm.items()},**im
                ))
                qrows += quintile_rows(yt[ev],pt[ev],q40,direction,"clipiqa_affine_head",seed,"mos")
                qrows += quintile_rows(yt[ev],pt[ev],q40,direction,"clipiqa_affine_head",seed,"prediction")
    df=pd.DataFrame(rows)
    df.to_csv(outdir/"CLIPIQA_seed_metrics.csv",index=False)
    summarize_seed_table(df).to_csv(outdir/"CLIPIQA_summary.csv",index=False)
    qdf=pd.DataFrame(qrows)
    qdf.to_csv(outdir/"CLIPIQA_conditional_coverage_seed.csv",index=False)
    return df


def selftest(outdir):
    rng=np.random.default_rng(1062026)
    data={}
    for j,(name,n) in enumerate([("KonIQ-10k",300),("CID2013",180),("SPAQ",330)]):
        X=rng.normal(loc=.1*j,size=(n,len(FEATURE_NAMES)))
        beta=rng.normal(scale=.05,size=len(FEATURE_NAMES))
        y=np.clip(.5+X@beta+rng.normal(scale=.08,size=n),0,1)
        z=np.clip(y+rng.normal(scale=.12,size=n),0,1)
        df=pd.DataFrame(X,columns=FEATURE_NAMES); df["y"]=y
        data[name]=df
    # lightweight structural checks only
    tr,cal,st=split_source(300,0)
    assert len(tr)+len(cal)+len(st)==300
    q=finite_q(np.abs(rng.normal(size=40)))
    assert np.isfinite(q)
    y=np.clip(rng.normal(.5,.2,size=100),0,1); p=y+rng.normal(0,.1,size=100)
    im=interval_metrics(y,p,.15)
    assert im["raw_clipped_coverage_absdiff"]<1e-12
    outdir.mkdir(parents=True,exist_ok=True)
    with open(outdir/"SELFTEST_PASS.txt","w") as f:
        f.write("Structural self-test PASS\n")
    print("[SELFTEST PASS] diagnostics/baselines structure is valid.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["formal","selftest"],default="formal")
    ap.add_argument("--koniq-csv",type=Path,default=Path("data/KonIQ_BRISQUE36_NIQE.csv"))
    ap.add_argument("--cid-csv",type=Path,default=Path("data/CID2013_BRISQUE36_NIQE.csv"))
    ap.add_argument("--spaq-csv",type=Path,default=Path("data/SPAQ_BRISQUE36_NIQE.csv"))
    ap.add_argument("--clip-koniq",type=Path,default=None)
    ap.add_argument("--clip-cid",type=Path,default=None)
    ap.add_argument("--clip-spaq",type=Path,default=None)
    ap.add_argument("--seeds",type=int,default=DEFAULT_SEEDS)
    ap.add_argument("--outdir",type=Path,default=Path("outputs"))
    args=ap.parse_args()
    if args.mode=="selftest":
        args.outdir.mkdir(parents=True,exist_ok=True)
        selftest(args.outdir); return
    clip_paths={"KonIQ-10k":args.clip_koniq,"CID2013":args.clip_cid,"SPAQ":args.clip_spaq}
    clip_enabled=all(v is not None and v.exists() for v in clip_paths.values())
    prepare_formal_outdir(args.outdir,args,clip_enabled)
    data=load_feature_data(args.koniq_csv,args.cid_csv,args.spaq_csv)
    run_classical(data,args.seeds,args.outdir)
    if clip_enabled:
        cdata=load_clip_data(clip_paths)
        run_clipiqa(data,cdata,args.seeds,args.outdir)
    else:
        print("[INFO] CLIP-IQA caches not all supplied; modern representation stage skipped.")
    env=dict(
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        sklearn=sklearn.__version__,
        matplotlib=plt.matplotlib.__version__,
    )
    (args.outdir/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    print(f"[DONE] outputs: {args.outdir}")

if __name__=="__main__":
    main()
