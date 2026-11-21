"""Score a folder of predicted NIfTI label maps against the reference segmentations.

Usage:
  python -m src.evaluate --pred results/predictions/nnunet_3d_fullres_fold0 \
      --gt $nnUNet_raw/Dataset501_BrainTumour/labelsTs_heldout --name nnunet_3d_fullres_fold0

Writes results/tables/<name>_per_case.csv (one row per patient, small, committable)
and results/tables/<name>_summary.csv (per region and metric: n, mean, sd, median, 95% CI).
"""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from .metrics import case_metrics, REGIONS
from .stats import bootstrap_ci

METRICS = {"dice": "Dice", "hd95": "HD95 (mm)", "sens": "Sensitivity", "prec": "Precision"}


def summarize(df, n_boot=5000):
    rows = []
    for r in REGIONS:
        for key, label in METRICS.items():
            v = df[f"{key}_{r}"].astype(float)
            ok = v.dropna()
            lo, hi = bootstrap_ci(ok.values, n_boot)
            rows.append({"region": r, "metric": label, "n": len(ok), "n_undefined": int(v.isna().sum()),
                         "mean": ok.mean(), "sd": ok.std(), "median": ok.median(),
                         "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()

    pred_files = sorted(Path(a.pred).glob("*.nii.gz"))
    assert pred_files, f"no predictions found in {a.pred}"
    gt_names = {p.name for p in Path(a.gt).glob("*.nii.gz")}
    missing = sorted(gt_names - {p.name for p in pred_files})
    assert not missing, f"{len(missing)} reference cases have no prediction, e.g. {missing[:3]}"

    rows = []
    for f in tqdm(pred_files):
        pimg = nib.load(str(f))
        pred = np.asarray(pimg.dataobj).astype(np.int16)
        gt = np.asarray(nib.load(str(Path(a.gt) / f.name)).dataobj).astype(np.int16)
        assert pred.shape == gt.shape, f"shape mismatch for {f.name}: {pred.shape} vs {gt.shape}"
        spacing = tuple(float(z) for z in pimg.header.get_zooms()[:3])
        rows.append({"case": f.name[:-7], **case_metrics(pred, gt, spacing)})

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{a.name}_per_case.csv", index=False)
    summ = summarize(df, a.n_boot)
    summ.to_csv(out / f"{a.name}_summary.csv", index=False)
    print(f"\n{a.name}: {len(df)} patients")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
