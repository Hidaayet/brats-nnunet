"""Segmentation metrics shared by every model.

MSD Task01 labels: 0 background, 1 edema, 2 non-enhancing tumor, 3 enhancing tumor.
Evaluation regions (BraTS convention, MSD label numbering):
  WT = {1,2,3}, TC = {2,3}, ET = {3}
VERIFY against the "labels" entry of dataset.json before trusting any score.

Empty-region conventions (decided in advance, reported in the results):
  Dice:        both empty -> 1.0; exactly one empty -> 0.0
  HD95:        both empty -> 0.0; exactly one empty -> NaN (counted, excluded from mean)
  Sensitivity: NaN when the reference is empty
  Precision:   NaN when the prediction is empty
"""
import numpy as np
from scipy import ndimage as ndi

REGIONS = {"WT": (1, 2, 3), "TC": (2, 3), "ET": (3,)}


def region_mask(seg, labels):
    return np.isin(seg, labels)


def dice(p, g):
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / s)


def sensitivity(p, g):
    return float(np.logical_and(p, g).sum() / g.sum()) if g.sum() else float("nan")


def precision(p, g):
    return float(np.logical_and(p, g).sum() / p.sum()) if p.sum() else float("nan")


def hd95(p, g, spacing=(1.0, 1.0, 1.0)):
    """95th percentile symmetric surface distance in mm (spacing = voxel size in mm)."""
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("nan")
    ps = p & ~ndi.binary_erosion(p)
    gs = g & ~ndi.binary_erosion(g)
    d_to_g = ndi.distance_transform_edt(~gs, sampling=spacing)
    d_to_p = ndi.distance_transform_edt(~ps, sampling=spacing)
    d = np.concatenate([d_to_g[ps], d_to_p[gs]])
    return float(np.percentile(d, 95))


def case_metrics(pred, gt, spacing=(1.0, 1.0, 1.0)):
    voxel_mm3 = float(np.prod(spacing))
    out = {}
    for name, labels in REGIONS.items():
        p, g = region_mask(pred, labels), region_mask(gt, labels)
        out[f"dice_{name}"] = dice(p, g)
        out[f"hd95_{name}"] = hd95(p, g, spacing)
        out[f"sens_{name}"] = sensitivity(p, g)
        out[f"prec_{name}"] = precision(p, g)
        out[f"gt_vol_mm3_{name}"] = float(g.sum() * voxel_mm3)
        out[f"pred_vol_mm3_{name}"] = float(p.sum() * voxel_mm3)
    return out
