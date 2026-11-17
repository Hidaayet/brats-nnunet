# Brain Tumor Segmentation with nnU-Net (MSD Task01 / BraTS)

> **Headline result:** _TODO (Week 3): "nnU-Net 3D full-res reaches WT/TC/ET Dice of x/x/x vs y/y/y for a 2D U-Net baseline (fold 0, 250 epochs)."_

_TODO: hero figure (best / median / worst case: MRI | ground truth | prediction)_

## Why this matters
_TODO: one paragraph on treatment planning, radiotherapy targeting, longitudinal tumor monitoring._

## Reproduce
```bash
pip install -r requirements.txt
source scripts/env.sh
bash scripts/01_prepare_data.sh               # download, convert, preprocess
bash scripts/02_train_nnunet.sh 3d_fullres 0  # also run 2d for a 2D-vs-3D comparison
bash scripts/03_predict_evaluate.sh 3d_fullres 0

python -m src.baseline train   --raw $nnUNet_raw/Dataset001_BrainTumour
python -m src.baseline predict --raw $nnUNet_raw/Dataset001_BrainTumour
python -m src.evaluate --pred results/predictions/unet2d_fold0 \
    --gt $nnUNet_raw/Dataset001_BrainTumour/labelsTr --name unet2d_fold0
```
Baseline and nnU-Net use the same train/val split (fold 0) and the same evaluation code.

## Results
_TODO: table (Dice + HD95 for WT/TC/ET; baseline vs nnU-Net 2D vs 3D), failure cases, Seg-Grad-CAM panel._

## Limitations
- Single dataset (MSD Task01, 484 cases); no external validation.
- Shortened schedule (250 epochs vs nnU-Net's default 1000) because of free-GPU limits.
- One fold evaluated, so no variance estimate across resamples.
- The baseline is a lightly trained 2D U-Net, not a tuned competitor.
- Seg-Grad-CAM is a qualitative aid, not evidence of causal reasoning by the model.
