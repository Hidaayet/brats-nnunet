# usage: source scripts/env.sh
# Set BASE to a folder with plenty of space BEFORE sourcing
#   Kaggle:  BASE=/kaggle/temp/nnunet      Colab: BASE=/content/nnunet      Local: default ./data
BASE=${BASE:-$PWD/data}
export BASE
export nnUNet_raw=$BASE/nnUNet_raw
export nnUNet_preprocessed=$BASE/nnUNet_preprocessed
# Results (checkpoints) should live somewhere that survives a session cut-off
export nnUNet_results=${RESULTS_DIR:-$BASE/nnUNet_results}
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

export DATASET_ID=501
export DATASET_NAME=Dataset501_BrainTumour
export RAW_DS=$nnUNet_raw/$DATASET_NAME
# One place to change the training length. Built-in trainers: nnUNetTrainer_5epochs (smoke test),
# nnUNetTrainer_100epochs, nnUNetTrainer_250epochs.
# Kaggle/Colab-safe variants that save a resumable checkpoint every 10 epochs (instead of 50):
# nnUNetTrainer_100ep_s10, nnUNetTrainer_250ep_s10  (run: python scripts/install_trainer.py once)
export TRAINER=${TRAINER:-nnUNetTrainer_100ep_s10}
