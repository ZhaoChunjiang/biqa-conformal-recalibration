#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frozen zero-shot CLIP-IQA scores using IQA-PyTorch/pyiqa.

Rationale:
- CLIP-IQA (AAAI 2023) is a modern vision-language NR-IQA model.
- Base CLIP-IQA is used, NOT CLIP-IQA+ trained on KonIQ.
- No target MOS is used during score extraction.
- Images are explicitly resized to 512x384 before inference to match the
  main study's canonical image resolution.
- GPU is optional; CPU is supported but will be slower.

The output cache is later combined with a SOURCE-ONLY nonnegative affine head
inside run_strengthening.py.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}


def build_index(root:Path):
    idx={}
    duplicates=[]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if p.name in idx and idx[p.name] != p:
                duplicates.append((p.name, str(idx[p.name]), str(p)))
            else:
                idx[p.name]=p
    if duplicates:
        preview=duplicates[:10]
        raise RuntimeError(
            f"duplicate image basenames under {root}; first duplicates={preview}. "
            "Use a root with unique basenames to avoid ambiguous membership."
        )
    return idx


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feature-csv",type=Path,required=True,
                    help="Existing BRISQUE36+NIQE cache; provides name/y membership.")
    ap.add_argument("--image-root",type=Path,required=True)
    ap.add_argument("--out-csv",type=Path,required=True)
    ap.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    ap.add_argument("--batch-size",type=int,default=8)
    args=ap.parse_args()

    import torch
    import torchvision
    import torchvision.transforms.functional as TF
    import pyiqa
    import importlib.metadata as importlib_metadata

    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    print("[CLIP-IQA] device =",device)
    metric=pyiqa.create_metric("clipiqa",device=device)
    metric.eval()

    meta=pd.read_csv(args.feature_csv)
    if "y" not in meta.columns:
        raise ValueError("feature cache must contain y")
    name_col="name" if "name" in meta.columns else None
    if name_col is None:
        if "image" in meta.columns:
            meta["name"]=meta["image"].map(lambda x:Path(str(x)).name)
            name_col="name"
        else:
            raise ValueError("feature cache needs name or image column")

    idx=build_index(args.image_root)
    missing=[n for n in meta[name_col].astype(str) if n not in idx]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing under {args.image_root}; first={missing[:10]}")
    print("[membership]",len(meta),"images matched")

    args.out_csv.parent.mkdir(parents=True,exist_ok=True)
    partial=args.out_csv.with_suffix(".partial.csv")
    done={}
    if partial.exists():
        pold=pd.read_csv(partial)
        done={str(r["name"]):float(r["clipiqa_raw"]) for _,r in pold.iterrows()}
        print("[resume]",len(done))

    rows=[]
    for _,r in meta.iterrows():
        n=str(r[name_col])
        if n in done:
            rows.append({"name":n,"y":float(r["y"]),"clipiqa_raw":done[n]})

    pending=[(str(r[name_col]),float(r["y"])) for _,r in meta.iterrows() if str(r[name_col]) not in done]
    t0=time.time()
    for start in tqdm(range(0,len(pending),args.batch_size),desc="CLIP-IQA"):
        batch=pending[start:start+args.batch_size]
        ims=[]
        for n,y in batch:
            with Image.open(idx[n]) as im:
                im=im.convert("RGB").resize((512,384),Image.Resampling.BICUBIC)
                t=TF.to_tensor(im)
                ims.append(t)
        x=torch.stack(ims,dim=0).to(device)
        with torch.inference_mode():
            score=metric(x).detach().cpu().numpy().reshape(-1)
        for (n,y),s in zip(batch,score):
            rows.append({"name":n,"y":y,"clipiqa_raw":float(s)})
        if len(rows)%200 < args.batch_size:
            pd.DataFrame(rows).drop_duplicates("name",keep="last").to_csv(partial,index=False)

    out=pd.DataFrame(rows).drop_duplicates("name",keep="last")
    out=meta[[name_col,"y"]].rename(columns={name_col:"name"}).merge(out[["name","clipiqa_raw"]],on="name",how="left")
    if out.clipiqa_raw.isna().any():
        raise RuntimeError("missing CLIP-IQA scores")
    out.to_csv(args.out_csv,index=False)
    if partial.exists(): partial.unlink()
    try:
        pyiqa_version = importlib_metadata.version("pyiqa")
    except Exception:
        pyiqa_version = getattr(pyiqa, "__version__", "unknown")
    info={
        "metric":"clipiqa",
        "pyiqa_model":"base CLIP-IQA (not CLIP-IQA+)",
        "device":device,
        "canonical_input_resize":"512x384 bicubic before pyiqa inference",
        "python":sys.version,
        "torch":torch.__version__,
        "torchvision":torchvision.__version__,
        "pyiqa":pyiqa_version,
        "n":len(out),
        "score_min":float(out.clipiqa_raw.min()),
        "score_max":float(out.clipiqa_raw.max()),
        "elapsed_minutes":(time.time()-t0)/60,
    }
    args.out_csv.with_suffix(".json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print("[DONE]",args.out_csv)
    print(json.dumps(info,indent=2))

if __name__=="__main__":
    main()
