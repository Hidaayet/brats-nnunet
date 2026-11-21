# Brain Tumor Segmentation with nnU-Net (MSD Task01 BrainTumour)

Automatic segmentation of brain tumor regions from multimodal MRI using
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), compared against a 2D baseline and evaluated
per tumor region on **held-out patients** with Dice, HD95, sensitivity and precision.

> **Status: work in progress.** The pipeline is written and tested on synthetic data.
> Training on real data has not been completed yet, so there are **no results in this repository yet**.

## Why this matters
Radiologists outline brain tumors by hand on MRI. This is slow and differs between readers.
Outlines are used for radiotherapy planning, surgical planning and tracking a tumor over time.
This project automates the outlining step and measures how close it gets to expert outlines on
patients the model has never seen. It is a research prototype, **not** a clinical tool.

## Data
- Medical Segmentation Decathlon, Task01_BrainTumour: 484 labeled cases (the 266 unlabeled cases are not used).
- Input: four MRI channels (FLAIR, T1w, T1gd, T2w; order verified from `dataset.json`).
- Labels: 0 background, 1 edema, 2 non-enhancing tumor, 3 enhancing tumor.
- Regions: WT = {1,2,3}, TC = {2,3}, ET = {3}.
- Respect the dataset license (see the Medical Segmentation Decathlon page). No scans are committed here.

## Method
| Part | Choice |
|---|---|
| Split | By patient. About 18% held out as a **frozen test set** (`splits/split_v1.json`, seed 42); the rest split into train/val for fold 0. |
| Main model | nnU-Net v2 `3d_fullres`, one fold, shortened training (epoch count set in `scripts/env.sh`, chosen after a timing test) |
| Baseline | nnU-Net `2d` on the identical split |
| Metrics | Dice, HD95 (mm), sensitivity, precision per region; mean, SD, median, bootstrap 95% CI over patients |
| Comparison | Paired Wilcoxon signed-rank test, 3D vs 2D, same patients |
| Failure analysis | Worst 10 patients, Dice vs tumor volume, expert/model/disagreement overlays |

The test patients are moved out of the training folders **before** preprocessing and are only used once, at the end.

## Progress
- [x] Pipeline code: split/hold-out, preprocessing, training with resume, prediction, evaluation, statistics, failure analysis
- [x] Tested on synthetic volumes (metrics, split, statistics, plots)
- [ ] Real-data preprocessing run
- [ ] 3D nnU-Net trained
- [ ] 2D baseline trained
- [ ] Held-out evaluation, statistics, failure analysis
- [ ] Two-page summary and video

## Reproduce
**On Kaggle (free GPU):** run `notebooks/kaggle_01_prepare.ipynb` (no GPU needed), then `notebooks/kaggle_02_train.ipynb` (GPU).
**From a terminal with a GPU:**
```bash
pip install -r requirements.txt            # install the right torch build first
source scripts/env.sh                      # set BASE to a folder with ~40 GB free
bash scripts/01_prepare_data.sh "3d_fullres 2d"   # download, convert, hold out test set, preprocess
python scripts/install_trainer.py          # optional: checkpoints every 10 epochs
bash scripts/02_train_nnunet.sh 3d_fullres 0      # resumes automatically if a checkpoint exists
bash scripts/02_train_nnunet.sh 2d 0
bash scripts/03_predict_evaluate.sh 3d_fullres 0  # held-out patients, run ONCE at the end
bash scripts/03_predict_evaluate.sh 2d 0
python -m src.compare --a results/tables/nnunet_3d_fullres_fold0_per_case.csv --name-a 3D \
                      --b results/tables/nnunet_2d_fold0_per_case.csv --name-b 2D
python -m src.failure_analysis --csv results/tables/nnunet_3d_fullres_fold0_per_case.csv
```

## Results
_No results yet. This section will contain the Dice/HD95 table for WT, TC and ET (3D vs 2D), the number of
test patients, the worst cases and a prediction-vs-expert figure. Numbers are added only from real runs._

## Limitations
- Single dataset (484 cases); no external validation.
- Shortened training instead of nnU-Net's default 1000 epochs, because of free-GPU limits.
- One fold only: no variance estimate across folds.
- Expert outlines also vary between readers.
- Research prototype, not a clinical or regulated medical device.

## Repository layout
```
notebooks/  Kaggle notebooks (prepare, train)     scripts/  one script per stage
src/        metrics, evaluation, statistics, plots splits/   frozen patient split
extras/     unused earlier code (see extras/README.md)
```

## References
- Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods, 2021.
- Simpson et al., *A large annotated medical image dataset for the development and evaluation of segmentation algorithms*, 2019.
- Ronneberger et al., *U-Net*, 2015.

## License
Code: MIT (`LICENSE`). The dataset has its own license and is not included.
