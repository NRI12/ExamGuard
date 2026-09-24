"""Shared clip dataset + GPU-side augmentation for X3D / TSM training.

Workers only decode (cv2) and return uint8 frames; crops, flips, colour jitter and normalisation
run batched on the GPU, which keeps the 12 vCPUs from being the bottleneck.
"""
import csv
import json
import math
import os
import random
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_split(root, split, drop_related=False):
    """Rows of (path, label, kind) for one split, joined with meta.csv."""
    meta = {r["path"]: r for r in csv.DictReader(open(os.path.join(root, "meta.csv")))}
    rows = []
    for line in open(os.path.join(root, f"{split}.csv")):
        path, label = line.split()
        kind = meta[path]["kind"]
        if drop_related and kind == "r":
            continue
        rows.append((path, int(label), kind))
    return rows


def clip_len_of(root):
    info = os.path.join(root, "clips_info.json")
    return json.load(open(info))["clip_len"] if os.path.exists(info) else 80


def read_classes(root):
    return [line.split(maxsplit=1)[1].strip() for line in open(os.path.join(root, "classes.txt"))]


class ClipDataset(Dataset):
    """Returns (uint8 tensor [T, H, W, 3] RGB, label).

    sampling="dense": num_frames frames `stride` apart (X3D).
    sampling="segment": num_frames segments over the clip, one frame per segment (TSM / TSN).
    """

    def __init__(self, root, rows, num_frames, stride=1, sampling="dense", train=True, views=1):
        self.root, self.rows = root, rows
        self.num_frames, self.stride, self.sampling = num_frames, stride, sampling
        self.train, self.views = train, views
        self.clip_len = clip_len_of(root)

    def __len__(self):
        return len(self.rows)

    def _indices(self, n, view=0):
        if self.sampling == "dense":
            span = (self.num_frames - 1) * self.stride + 1
            max_start = max(0, n - span)
            if self.train:
                start = random.randint(0, max_start)
            elif self.views == 1:
                start = max_start // 2
            else:
                start = round(view * max_start / (self.views - 1))
            return [min(n - 1, start + i * self.stride) for i in range(self.num_frames)]
        seg = n / self.num_frames
        if self.train:
            return [min(n - 1, int(seg * i + random.random() * seg)) for i in range(self.num_frames)]
        offset = (view + 0.5) / self.views
        return [min(n - 1, int(seg * (i + offset))) for i in range(self.num_frames)]

    def _decode(self, path, wanted):
        cap = cv2.VideoCapture(os.path.join(self.root, path))
        need = Counter(wanted)
        last = max(wanted)
        frames, i, prev = {}, 0, None
        while i <= last:
            if i in need:
                ok, img = cap.read()
                if not ok:
                    break
                prev = frames[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif not cap.grab():
                break
            i += 1
        cap.release()
        # a truncated clip repeats its last decoded frame instead of crashing the epoch
        return np.stack([frames.get(k, prev) for k in wanted])

    def __getitem__(self, idx):
        path, label, _ = self.rows[idx]
        n = self.clip_len
        if self.views == 1:
            x = self._decode(path, self._indices(n))
        else:  # [V*T, H, W, 3], unpacked on the GPU side
            x = np.concatenate([self._decode(path, self._indices(n, v)) for v in range(self.views)])
        return torch.from_numpy(x), label


def balanced_sampler(rows, hard_negative_boost=3.0, num_samples=None):
    """Each class drawn equally often; hard negatives oversampled inside the normal class."""
    counts = Counter(label for _, label, _ in rows)
    w = []
    for _, label, kind in rows:
        wi = 1.0 / counts[label]
        if kind == "h":
            wi *= hard_negative_boost
        w.append(wi)
    return WeightedRandomSampler(w, num_samples or len(rows), replacement=True)


def worker_init(_):
    cv2.setNumThreads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- GPU transforms
def _norm(x, mean=KINETICS_MEAN, std=KINETICS_STD):
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1, 1)
    return (x - mean) / std


def gpu_train_transform(frames, size, scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3), flip=0.5, jitter=0.3,
                        mean=KINETICS_MEAN, std=KINETICS_STD):
    """frames uint8 [B, T, H, W, 3] on GPU -> float [B, 3, T, size, size]."""
    b, t, h, w, _ = frames.shape
    x = frames.permute(0, 4, 1, 2, 3).float().div_(255)  # B,3,T,H,W
    out = torch.empty(b, 3, t, size, size, device=x.device)
    for i in range(b):
        area = h * w * random.uniform(*scale)
        ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw = min(w, int(round(math.sqrt(area * ar))))
        ch = min(h, int(round(math.sqrt(area / ar))))
        y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
        crop = x[i, :, :, y0:y0 + ch, x0:x0 + cw].transpose(0, 1)  # T,3,ch,cw
        crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        if random.random() < flip:
            crop = crop.flip(-1)
        out[i] = crop.transpose(0, 1)
    if jitter > 0:  # per-clip brightness / contrast / saturation, identical across frames
        bri = 1 + (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        con = 1 + (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        sat = 1 + (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        out = out * bri
        avg = out.mean(dim=(1, 2, 3, 4), keepdim=True)
        out = (out - avg) * con + avg
        gray = out.mean(dim=1, keepdim=True)
        out = ((out - gray) * sat + gray).clamp_(0, 1)
    return _norm(out, mean, std)


def gpu_eval_transform(frames, size, resize=None, mean=KINETICS_MEAN, std=KINETICS_STD):
    """frames uint8 [B, T, H, W, 3] -> float [B, 3, T, size, size]: short-side resize + centre crop."""
    b, t, h, w, _ = frames.shape
    x = frames.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w).float().div_(255)
    resize = resize or int(round(size * 256 / 224))
    if min(h, w) != resize:
        s = resize / min(h, w)
        x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False, antialias=True)
    hh, ww = x.shape[-2:]
    y0, x0 = (hh - size) // 2, (ww - size) // 2
    x = x[:, :, y0:y0 + size, x0:x0 + size]
    x = x.reshape(b, t, 3, size, size).permute(0, 2, 1, 3, 4)
    return _norm(x, mean, std)


def mixup(x, y_onehot, alpha):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]
