"""Qualitative panel for one patient: FLAIR | reference | prediction | disagreement.

  python -m src.overlay --case BRATS_123 --images $RAW/imagesTs_heldout --gt $RAW/labelsTs_heldout \
      --pred results/predictions/nnunet_3d_fullres_fold0
Disagreement colours (tumor = any label>0): green = both agree, red = model only (false positive), blue = expert only (missed).
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--channel", type=int, default=0, help="0 = FLAIR in MSD (verify in dataset.json)")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    img = np.asarray(nib.load(f"{a.images}/{a.case}_{a.channel:04d}.nii.gz").dataobj)
    gt = np.asarray(nib.load(f"{a.gt}/{a.case}.nii.gz").dataobj).astype(int)
    pr = np.asarray(nib.load(f"{a.pred}/{a.case}.nii.gz").dataobj).astype(int)
    z = int(np.argmax(((gt > 0) | (pr > 0)).sum((0, 1))))
    g, p = gt[..., z] > 0, pr[..., z] > 0
    dis = np.zeros(g.shape + (3,))
    dis[g & p] = (0, .8, 0); dis[p & ~g] = (1, 0, 0); dis[g & ~p] = (0, .3, 1)
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    base = np.rot90(img[..., z])
    for k, (t, m) in enumerate([("MRI", None), ("Expert", gt[..., z]), ("Model", pr[..., z])]):
        ax[k].imshow(base, cmap="gray"); ax[k].set_title(t)
        if m is not None:
            ax[k].imshow(np.ma.masked_equal(np.rot90(m), 0), cmap="autumn", alpha=.6, vmin=1, vmax=3)
    ax[3].imshow(base, cmap="gray"); ax[3].imshow(np.rot90(dis), alpha=.6); ax[3].set_title("Disagreement")
    for x in ax: x.axis("off")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(Path(a.out) / f"overlay_{a.case}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
