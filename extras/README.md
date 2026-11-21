Not used by the main pipeline.

- `custom_unet2d/`: an earlier hand-written PyTorch 2D U-Net baseline. It was written for the old split
  (nnU-Net fold-0 validation set) and does NOT know about the held-out test set. The baseline now used is
  nnU-Net's own `2d` configuration (same framework, same split, only 2D vs 3D differs).
- `old_notebooks/`: earlier Colab/Kaggle notebooks, replaced by `notebooks/kaggle_01_prepare.ipynb` and `kaggle_02_train.ipynb`.
