#!/usr/bin/env bash
# Download MSD Task01, convert to nnU-Net v2 (ID 501), hold out test patients, preprocess.
# Safe to re-run: every step is skipped if already done.
# usage: bash scripts/01_prepare_data.sh [configs]      e.g.  "3d_fullres"  or  "2d"  or  "3d_fullres 2d"
set -euo pipefail
source scripts/env.sh
CONFIGS=${1:-"3d_fullres 2d"}
TAR=$BASE/Task01_BrainTumour.tar

echo "Free disk space at BASE:"; df -h "$BASE" | tail -1

if [ ! -d "$BASE/Task01_BrainTumour/imagesTr" ]; then
  wget -c -O "$TAR" https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
  tar -xf "$TAR" -C "$BASE"
  rm -f "$TAR"                                   # frees ~7 GB
  find "$BASE/Task01_BrainTumour" -name '._*' -delete   # macOS junk files break nnU-Net
fi

if [ ! -d "$RAW_DS/imagesTr" ]; then
  nnUNetv2_convert_MSD_dataset -i "$BASE/Task01_BrainTumour" -overwrite_id $DATASET_ID
  rm -rf "$BASE/Task01_BrainTumour"              # converted copy exists; frees ~7 GB
fi

# Hold out test patients BEFORE preprocessing (idempotent; re-uses splits/split_v1.json)
python scripts/00_split_holdout.py --raw "$RAW_DS" --drop-unlabeled

nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -c $CONFIGS

# Make fold 0 use the frozen train/val lists
python scripts/00_split_holdout.py --raw "$RAW_DS" --write-nnunet-split
echo "Preparation done."
