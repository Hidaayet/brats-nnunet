#!/usr/bin/env bash
# Predict the HELD-OUT test patients and score them. Run once, at the very end.
# usage: bash scripts/03_predict_evaluate.sh 3d_fullres 0
set -euo pipefail
source scripts/env.sh
CONFIG=${1:-3d_fullres}; FOLD=${2:-0}
NAME=nnunet_${CONFIG}_fold${FOLD}
OUT=results/predictions/$NAME
mkdir -p "$OUT"
nnUNetv2_predict -i "$RAW_DS/imagesTs_heldout" -o "$OUT" -d $DATASET_ID -c $CONFIG -f $FOLD -tr $TRAINER
python -m src.evaluate --pred "$OUT" --gt "$RAW_DS/labelsTs_heldout" --name "$NAME"
