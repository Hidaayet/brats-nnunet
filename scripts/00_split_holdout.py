#!/usr/bin/env python
"""Freeze a patient-level split and hold out the test patients.

Run ONCE, right after `nnUNetv2_convert_MSD_dataset` and BEFORE preprocessing.

What it does
  1. Creates splits/split_v1.json with three disjoint lists of patient IDs:
       test  (~18% of patients)  -> never used for training or tuning
       val   (~20% of the rest)  -> monitoring set for fold 0
       train (the remaining)     -> training set for fold 0
     The split is seeded, so it is reproducible. If the file already exists it
     is re-used and never overwritten ("frozen").
  2. Moves the test patients out of imagesTr/labelsTr into imagesTs_heldout/
     and labelsTs_heldout/ (same file names).
  3. Updates numTraining in dataset.json (nnU-Net's integrity check needs it).
  4. Optionally deletes the 266 unlabeled MSD test cases (--drop-unlabeled) to save disk.

After preprocessing, run with --write-nnunet-split so nnU-Net's fold 0 uses
exactly the train/val lists above (otherwise it makes its own random split
and your test patients would not be the only ones excluded).

Usage
  python scripts/00_split_holdout.py --raw $nnUNet_raw/Dataset501_BrainTumour --drop-unlabeled
  # ... nnUNetv2_plan_and_preprocess ...
  python scripts/00_split_holdout.py --raw $nnUNet_raw/Dataset501_BrainTumour --write-nnunet-split
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

SUFFIX = ".nii.gz"


def ids_in(folder: Path):
    return sorted(p.name[: -len(SUFFIX)] for p in folder.glob(f"*{SUFFIX}"))


def make_split(ids, test_frac, val_frac, seed):
    ids = sorted(ids)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_test = round(len(ids) * test_frac)
    test, rest = shuffled[:n_test], shuffled[n_test:]
    n_val = round(len(rest) * val_frac)
    val, train = rest[:n_val], rest[n_val:]
    return {"test": sorted(test), "val": sorted(val), "train": sorted(train)}


def check_split(split, all_ids):
    t, v, tr = set(split["test"]), set(split["val"]), set(split["train"])
    assert not (t & v) and not (t & tr) and not (v & tr), "split lists overlap"
    assert t | v | tr == set(all_ids), "split does not cover exactly the dataset's patients"


def move(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.move(str(src), str(dst_dir / src.name))


def holdout(raw: Path, a):
    ds_json_path = raw / "dataset.json"
    ds = json.load(open(ds_json_path))
    n_ch = len(ds["channel_names"])
    labels_tr, images_tr = raw / "labelsTr", raw / "imagesTr"
    labels_ho, images_ho = raw / "labelsTs_heldout", raw / "imagesTs_heldout"

    all_ids = sorted(set(ids_in(labels_tr)) | set(ids_in(labels_ho)))
    split_file = Path(a.split_file)
    if split_file.exists():
        split = json.load(open(split_file))
        print(f"Re-using frozen split {split_file}")
    else:
        if ids_in(labels_ho):
            raise SystemExit("Held-out cases exist but no split file: refusing to guess. Restore the split file.")
        split = make_split(all_ids, a.test_frac, a.val_frac, a.seed)
        split.update({"seed": a.seed, "test_frac": a.test_frac, "val_frac": a.val_frac,
                      "dataset": raw.name, "n_total": len(all_ids)})
        split_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(split, open(split_file, "w"), indent=1)
        print(f"Wrote {split_file}")
    check_split(split, all_ids)

    for cid in split["test"]:
        move(labels_tr / f"{cid}{SUFFIX}", labels_ho)
        for ch in range(n_ch):
            move(images_tr / f"{cid}_{ch:04d}{SUFFIX}", images_ho)

    # sanity: nothing from the test list may remain in the training folders
    left = [c for c in split["test"] if (labels_tr / f"{c}{SUFFIX}").exists()]
    assert not left, f"test cases still in labelsTr: {left[:3]}"
    ds["numTraining"] = len(ids_in(labels_tr))
    json.dump(ds, open(ds_json_path, "w"), indent=2)

    if a.drop_unlabeled and (raw / "imagesTs").exists():
        shutil.rmtree(raw / "imagesTs")
        print("Deleted unlabeled imagesTs (not used).")
    print(f"train {len(split['train'])} | val {len(split['val'])} | test (held out) {len(split['test'])} "
          f"| numTraining in dataset.json = {ds['numTraining']}")


def write_nnunet_split(raw: Path, a):
    split = json.load(open(a.split_file))
    pre = Path(os.environ["nnUNet_preprocessed"]) / raw.name
    assert pre.exists(), f"{pre} not found: run nnUNetv2_plan_and_preprocess first"
    out = pre / "splits_final.json"
    json.dump([{"train": split["train"], "val": split["val"]}], open(out, "w"), indent=1)
    print(f"Wrote {out} (fold 0 = frozen train/val; test patients excluded)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="e.g. $nnUNet_raw/Dataset501_BrainTumour")
    ap.add_argument("--split-file", default="splits/split_v1.json")
    ap.add_argument("--test-frac", type=float, default=0.18)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-unlabeled", action="store_true")
    ap.add_argument("--write-nnunet-split", action="store_true")
    a = ap.parse_args()
    raw = Path(a.raw)
    write_nnunet_split(raw, a) if a.write_nnunet_split else holdout(raw, a)
