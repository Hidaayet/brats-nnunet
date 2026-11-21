#!/usr/bin/env bash
# usage: bash scripts/02_train_nnunet.sh 3d_fullres 0 [trainer]
# Shortened schedule is the honest default on free GPUs; state it in the README.
set -e
source scripts/env.sh
CONFIG=${1:-3d_fullres}; FOLD=${2:-0}; TRAINER=${3:-nnUNetTrainer_250epochs}
# --c resumes from the last checkpoint if a session died; falls back to a fresh start
nnUNetv2_train 1 $CONFIG $FOLD -tr $TRAINER --c || nnUNetv2_train 1 $CONFIG $FOLD -tr $TRAINER
