# Brain Tumor Segmentation with nnU-Net (MSD Task01 BrainTumour)

Automatic segmentation of brain tumor regions from multimodal MRI using
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), compared against a 2D baseline
and evaluated per tumor region with Dice and HD95.

> **Status: work in progress.** The pipeline code is written. Training and evaluation
> have not been completed yet, so there are **no results in this repository yet**.
> This file will be updated with real numbers once they exist.

## Why this matters

Radiologists outline brain tumors by hand on MRI. This is slow and differs between readers.
Accurate outlines are used for radiotherapy planning, surgical planning and tracking a tumor
over time. This project automates the outlining step and measures how close the result
gets to expert outlines on patients the model has never seen.
It is a research prototype, **not** a clinical tool.

## Data

- **Dataset:** Medical Segmentation Decathlon, Task01_BrainTumour (484 labeled cases,
  derived from BraTS data). The 266 unlabeled test cases in the download are not used.
- **Input:** four MRI channels per patient (FLAIR, T1w, T1gd, T2w; order to be verified in `dataset.json`).
- **Labels:** 0 background, 1 edema, 2 non-enhancing tumor, 3 enhancing tumor.
- **Evaluated regions:** Whole Tumor WT = {1,2,3}, Tumor Core TC = {2,3}, Enhancing Tumor ET = {3}.
- **License:** the dataset terms must be respected (check the Medical Segmentation Decathlon
  page for the current license). Raw patient scans are never committed to this repository.

## Approach

| Part | Choice |
|---|---|
| Main model | nnU-Net v2, `3d_fullres`, one fold, shortened training |
| Baseline | 2D model on the same patients (nnU-Net `2d` configuration; a custom 2D U-Net is also included) |
| Split | By patient, never by slice |
| Metrics | Dice and HD95 (mm) per region; mean ± SD per region |
| Planned | Median, bootstrap 95% CI, paired Wilcoxon test, sensitivity/precision, failure analysis |

## Progress

- [x] Repository structure, scripts and notebooks
- [x] Metrics (Dice, HD95) and shared evaluation script
- [x] 2D baseline code
- [ ] Frozen patient-level split and held-out test set
- [ ] Preprocessing completed
- [ ] 3D nnU-Net trained
- [ ] 2D baseline trained
- [ ] Predictions and metrics
- [ ] Statistics and failure analysis
- [ ] Two-page summary and video

## Repository layout

```
notebooks/   Colab / Kaggle notebooks (setup, EDA, training)
scripts/     one script per stage: prepare data, train, predict + evaluate
src/         metrics.py, evaluate.py, unet2d.py, baseline.py
requirements.txt
```

## Reproduce

Requires a GPU. Free options (Kaggle, Colab) work but have session and disk limits.

```bash
pip install -r requirements.txt
source scripts/env.sh                         # set BASE to a persistent folder first
bash scripts/01_prepare_data.sh               # download, convert, preprocess
bash scripts/02_train_nnunet.sh 3d_fullres 0  # resumes automatically after a cut-off
bash scripts/03_predict_evaluate.sh 3d_fullres 0
```

Optional custom 2D baseline:

```bash
python -m src.baseline train   --raw $nnUNet_raw/Dataset001_BrainTumour
python -m src.baseline predict --raw $nnUNet_raw/Dataset001_BrainTumour
python -m src.evaluate --pred results/predictions/unet2d_fold0 \
    --gt $nnUNet_raw/Dataset001_BrainTumour/labelsTr --name unet2d_fold0
```

## Results

_No results yet. This section will contain a table of Dice and HD95 for WT, TC and ET
(3D nnU-Net vs 2D baseline), the number of test patients, the worst cases and a
prediction-vs-expert figure. Numbers will only be added from real runs._

## Limitations

- Single dataset (MSD Task01, 484 cases); no external validation.
- Shortened training schedule instead of nnU-Net's default 1000 epochs, because of free-GPU limits.
- One fold only, so no variance estimate across folds.
- The 2D baseline is lightly trained and not tuned.
- Expert outlines also vary between readers.
- Research prototype, not a clinical or regulated medical device.

## References

- Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods, 2021.
- Simpson et al., *A large annotated medical image dataset for the development and evaluation of segmentation algorithms*, 2019 (Medical Segmentation Decathlon).
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015.

## License

Code: MIT (see `LICENSE`). The dataset has its own license and is not included.