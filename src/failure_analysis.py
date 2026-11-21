"""Failure analysis from a per-case CSV: worst cases and Dice vs tumor volume.

  python -m src.failure_analysis --csv results/tables/nnunet_3d_fullres_fold0_per_case.csv --k 10
Outputs: results/tables/<name>_worst<k>.csv and results/figures/<name>_dice_vs_volume.png
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .metrics import REGIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fig-dir", default="results/figures"); ap.add_argument("--tab-dir", default="results/tables")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    name = Path(a.csv).stem.replace("_per_case", "")
    Path(a.fig_dir).mkdir(parents=True, exist_ok=True); Path(a.tab_dir).mkdir(parents=True, exist_ok=True)

    worst = df.sort_values("dice_WT").head(a.k)
    cols = ["case"] + [f"dice_{r}" for r in REGIONS] + [f"hd95_{r}" for r in REGIONS] + ["gt_vol_mm3_WT", "gt_vol_mm3_ET"]
    worst[cols].to_csv(Path(a.tab_dir) / f"{name}_worst{a.k}.csv", index=False)
    print(worst[cols].round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for i, r in enumerate(REGIONS):
        v = df[f"gt_vol_mm3_{r}"] / 1000.0   # mm^3 -> mL
        ax[i].scatter(v.clip(lower=0.01), df[f"dice_{r}"], s=14, alpha=.7)
        ax[i].set_xscale("log"); ax[i].set_title(r); ax[i].set_xlabel("reference volume (mL, log)")
        n0 = int((v == 0).sum())
        if n0:
            ax[i].text(.02, .04, f"{n0} case(s) with empty reference\n(plotted at 0.01 mL)", transform=ax[i].transAxes, fontsize=7)
    ax[0].set_ylabel("Dice")
    plt.tight_layout(); plt.savefig(Path(a.fig_dir) / f"{name}_dice_vs_volume.png", dpi=150)


if __name__ == "__main__":
    main()
