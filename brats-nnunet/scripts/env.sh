# usage: source scripts/env.sh   (set BASE to a persistent dir first, e.g. Google Drive on Colab)
BASE=${BASE:-$PWD/data}
export BASE
export nnUNet_raw=$BASE/nnUNet_raw
export nnUNet_preprocessed=$BASE/nnUNet_preprocessed
export nnUNet_results=$BASE/nnUNet_results
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
