"""Copy the Kaggle-safe trainers into the installed nnunetv2 package so `-tr nnUNetTrainer_100ep_s10` works.
Run once per session after `pip install nnunetv2`:  python scripts/install_trainer.py
(nnU-Net finds trainers by scanning its own package folder, so the file has to live there.)"""
import shutil
from pathlib import Path
import nnunetv2

dst = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "training_length"
src = Path(__file__).resolve().parent.parent / "src" / "trainers" / "nnUNetTrainer_s10.py"
shutil.copy(src, dst / src.name)
print("Installed", dst / src.name)
