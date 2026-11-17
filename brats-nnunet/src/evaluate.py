"""Score a folder of predicted NIfTI segmentations against ground truth.

Usage:
  python -m src.evaluate --pred results/predictions/nnunet_3d_fullres_fold0 \
      --gt $nnUNet_raw/Dataset001_BrainTumour/labelsTr --name nnunet_3d --out results/tables
Works for any model that writes <case>.nii.gz label maps, so the baseline and
nnU-Net are scored by exactly the same code.
"""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from .metrics import case_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()

    rows = []
    for f in tqdm(sorted(Path(a.pred).glob("*.nii.gz"))):
        pimg = nib.load(str(f))
        pred = np.asarray(pimg.dataobj).astype(np.int16)
        gt = np.asarray(nib.load(str(Path(a.gt) / f.name)).dataobj).astype(np.int16)
        spacing = tuple(float(z) for z in pimg.header.get_zooms()[:3])
        rows.append({"case": f.name.replace(".nii.gz", ""), **case_metrics(pred, gt, spacing)})

    df = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / f"{a.name}_per_case.csv", index=False)

    summ = {}
    for r in ["WT", "TC", "ET"]:
        summ[f"Dice {r}"] = f"{df[f'dice_{r}'].mean():.3f} +/- {df[f'dice_{r}'].std():.3f}"
        h = df[f"hd95_{r}"]
        summ[f"HD95 {r} (mm)"] = f"{h.mean():.2f} +/- {h.std():.2f} (n_empty_pred={int(h.isna().sum())})"
    pd.Series(summ, name=a.name).to_csv(Path(a.out) / f"{a.name}_summary.csv")
    print(f"\n{a.name}: {len(df)} cases")
    print(pd.Series(summ).to_string())


if __name__ == "__main__":
    main()
