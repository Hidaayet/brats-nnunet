#!/usr/bin/env bash
# usage: bash scripts/02_train_nnunet.sh 3d_fullres 0        (TRAINER from env.sh, override: TRAINER=nnUNetTrainer_5epochs bash ...)
# Resumes ONLY if a checkpoint exists. It never silently restarts after an error
# (so an out-of-memory crash can not wipe a half-trained model).
set -euo pipefail
source scripts/env.sh
CONFIG=${1:-3d_fullres}; FOLD=${2:-0}
CK=$nnUNet_results/$DATASET_NAME/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}/checkpoint_latest.pth
if [ -f "$CK" ]; then
  echo "Resuming from $CK"; RESUME=--c
else
  echo "Starting a new training"; RESUME=""
fi
nnUNetv2_train $DATASET_ID $CONFIG $FOLD -tr $TRAINER $RESUME
