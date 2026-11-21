#!/usr/bin/env bash
# Predict the validation fold with nnU-Net, then score with the shared evaluator.
# usage: bash scripts/03_predict_evaluate.sh 3d_fullres 0 [trainer]
set -e
source scripts/env.sh
CONFIG=${1:-3d_fullres}; FOLD=${2:-0}; TRAINER=${3:-nnUNetTrainer_250epochs}
RAW=$nnUNet_raw/Dataset001_BrainTumour
OUT=results/predictions/nnunet_${CONFIG}_fold${FOLD}
mkdir -p $OUT /tmp/val_imgs && rm -f /tmp/val_imgs/*
python - <<PY
import json, os, shutil
s = json.load(open(os.environ["nnUNet_preprocessed"] + "/Dataset001_BrainTumour/splits_final.json"))[$FOLD]["val"]
for c in s:
    for m in range(4):
        shutil.copy(f"$RAW/imagesTr/{c}_{m:04d}.nii.gz", "/tmp/val_imgs/")
PY
nnUNetv2_predict -i /tmp/val_imgs -o $OUT -d 1 -c $CONFIG -f $FOLD -tr $TRAINER
python -m src.evaluate --pred $OUT --gt $RAW/labelsTr --name nnunet_${CONFIG}_fold${FOLD}
