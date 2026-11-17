#!/usr/bin/env bash
# Download MSD Task01_BrainTumour, convert to nnU-Net v2 format, plan + preprocess.
set -e
source scripts/env.sh
if [ ! -d "$BASE/Task01_BrainTumour" ]; then
  wget -c -O $BASE/Task01_BrainTumour.tar \
    https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
  tar -xf $BASE/Task01_BrainTumour.tar -C $BASE
fi
nnUNetv2_convert_MSD_dataset -i $BASE/Task01_BrainTumour     # -> Dataset001_BrainTumour
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 2d 3d_fullres
