"""Train + predict with the 2D U-Net baseline on the SAME fold as nnU-Net.

  python -m src.baseline train   --raw $nnUNet_raw/Dataset001_BrainTumour --epochs 20
  python -m src.baseline predict --raw $nnUNet_raw/Dataset001_BrainTumour

Slices are cached to disk as float16 (use --stride 2 on Kaggle's limited disk).
Images are center-cropped 240->192; predictions are padded back to 240.
"""
import argparse, json, os
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm
from .unet2d import UNet2D

CROP, OFF = 192, 24
CACHE = Path("data/slice_cache")
CKPT = Path("results/unet2d_fold0.pth")


def get_split(raw: Path, fold=0):
    pre = Path(os.environ.get("nnUNet_preprocessed", "")) / raw.name / "splits_final.json"
    if pre.exists():
        s = json.load(open(pre))[fold]
        return s["train"], s["val"]
    # Same procedure nnU-Net v2 uses when it creates splits_final.json
    ids = sorted(f.name[:-7] for f in (raw / "labelsTr").glob("*.nii.gz"))
    kf = KFold(n_splits=5, shuffle=True, random_state=12345)
    tr, va = list(kf.split(ids))[fold]
    return [ids[i] for i in tr], [ids[i] for i in va]


def load_volume(raw: Path, cid):
    chans = []
    for m in range(4):
        v = np.asarray(nib.load(str(raw / "imagesTr" / f"{cid}_{m:04d}.nii.gz")).dataobj, dtype=np.float32)
        nz = v > 0
        if nz.any():
            v = np.where(nz, (v - v[nz].mean()) / (v[nz].std() + 1e-8), 0)
        chans.append(v)
    return np.stack(chans)  # (4,H,W,Z)


def cache_case(raw, cid, stride):
    out = CACHE / f"{cid}.npz"
    if out.exists():
        return
    x = load_volume(raw, cid)[:, OFF:OFF + CROP, OFF:OFF + CROP]
    y = np.asarray(nib.load(str(raw / "labelsTr" / f"{cid}.nii.gz")).dataobj, dtype=np.uint8)[OFF:OFF + CROP, OFF:OFF + CROP]
    keep = [z for z in range(0, x.shape[-1], stride) if (x[..., z] != 0).any()]
    np.savez(out, x=x[..., keep].transpose(3, 0, 1, 2).astype(np.float16), y=y[..., keep].transpose(2, 0, 1))


class SliceDS(torch.utils.data.Dataset):
    def __init__(self, ids):
        data = [np.load(CACHE / f"{c}.npz") for c in ids]
        self.x = [d["x"] for d in data]
        self.y = [d["y"] for d in data]
        self.idx = [(i, z) for i, a in enumerate(self.x) for z in range(len(a))]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i, z = self.idx[k]
        x = torch.from_numpy(self.x[i][z].astype(np.float32))
        y = torch.from_numpy(self.y[i][z].astype(np.int64))
        if np.random.rand() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def loss_fn(logits, y):
    ce = F.cross_entropy(logits, y)
    p = F.softmax(logits.float(), 1)
    oh = F.one_hot(y, 4).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (den + 1))[1:].mean()


def train(a):
    raw = Path(a.raw)
    CACHE.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(exist_ok=True, parents=True)
    tr, _ = get_split(raw)
    for c in tqdm(tr, desc="caching train slices"):
        cache_case(raw, c, a.stride)
    dl = torch.utils.data.DataLoader(SliceDS(tr), batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet2D().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    for ep in range(a.epochs):
        net.train()
        tot = 0
        for x, y in tqdm(dl, desc=f"epoch {ep+1}/{a.epochs}"):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                l = loss_fn(net(x), y)
            opt.zero_grad()
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            tot += l.item()
        sched.step()
        print(f"epoch {ep+1} loss {tot/len(dl):.4f}")
        torch.save(net.state_dict(), CKPT)  # checkpoint every epoch (sessions die)


@torch.no_grad()
def predict(a):
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _, va = get_split(raw)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet2D().to(dev)
    net.load_state_dict(torch.load(CKPT, map_location=dev))
    net.eval()
    for cid in tqdm(va, desc="predicting val fold"):
        ref = nib.load(str(raw / "imagesTr" / f"{cid}_0000.nii.gz"))
        x = load_volume(raw, cid)[:, OFF:OFF + CROP, OFF:OFF + CROP]
        full = np.zeros(ref.shape[:3], dtype=np.uint8)
        xs = torch.from_numpy(x.transpose(3, 0, 1, 2)).float()
        pred = torch.cat([net(xs[i:i + 16].to(dev)).argmax(1).cpu() for i in range(0, len(xs), 16)])
        full[OFF:OFF + CROP, OFF:OFF + CROP] = pred.numpy().transpose(1, 2, 0)
        nib.save(nib.Nifti1Image(full, ref.affine, ref.header), str(out / f"{cid}.nii.gz"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="results/predictions/unet2d_fold0")
    a = ap.parse_args()
    train(a) if a.mode == "train" else predict(a)
