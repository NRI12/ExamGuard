"""Clip dataset, balanced sampler and GPU-side augmentation.

Workers only decode (cv2) and return uint8 frames; crops, flips, colour jitter and normalisation run batched
on the GPU so the CPU is not the bottleneck.
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

NORMS = {"kinetics": ((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),   # X3D (pytorchvideo)
         "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}  # TSM (OpenMMLab)


def read_split(clips_dir, split, drop_related=False):
    """Rows (path, label, kind) of one split, joined with meta.csv."""
    meta = {r["path"]: r for r in csv.DictReader(open(os.path.join(clips_dir, "meta.csv")))}
    rows = []
    for line in open(os.path.join(clips_dir, f"{split}.csv")):
        path, label = line.split()
        kind = meta[path]["kind"]
        if drop_related and kind == "r":
            continue
        rows.append((path, int(label), kind))
    return rows


def read_classes(clips_dir):
    return [line.split(maxsplit=1)[1].strip() for line in open(os.path.join(clips_dir, "classes.txt"))]


def clip_len_of(clips_dir):
    return json.load(open(os.path.join(clips_dir, "clips_info.json")))["clip_len"]


class ClipDataset(Dataset):
    """Returns (uint8 [T, H, W, 3] RGB, label).

    sampling "dense"  : num_frames frames `stride` apart (X3D)
    sampling "segment": num_frames segments over the clip, one frame each (TSM)
    views > 1 stacks evenly spaced temporal views along T for multi-view testing."""

    def __init__(self, clips_dir, rows, num_frames, stride=1, train=True, views=1, sampling="dense"):
        self.root, self.rows = clips_dir, rows
        self.num_frames, self.stride, self.train, self.views = num_frames, stride, train, views
        self.sampling = sampling
        self.clip_len = clip_len_of(clips_dir)

    def __len__(self):
        return len(self.rows)

    def _indices(self, n, view=0):
        if self.sampling == "segment":
            seg = n / self.num_frames
            if self.train:
                return [min(n - 1, int(seg * i + random.random() * seg)) for i in range(self.num_frames)]
            return [min(n - 1, int(seg * (i + (view + 0.5) / self.views))) for i in range(self.num_frames)]
        span = (self.num_frames - 1) * self.stride + 1
        max_start = max(0, n - span)
        if self.train:
            start = random.randint(0, max_start)
        elif self.views == 1:
            start = max_start // 2
        else:
            start = round(view * max_start / (self.views - 1))
        return [min(n - 1, start + i * self.stride) for i in range(self.num_frames)]

    def _decode(self, path, wanted):
        cap = cv2.VideoCapture(os.path.join(self.root, path))
        need, last = set(wanted), max(wanted)
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
        return np.stack([frames.get(k, prev) for k in wanted])  # truncated clip: repeat last frame

    def __getitem__(self, idx):
        path, label, _ = self.rows[idx]
        views = [self._decode(path, self._indices(self.clip_len, v)) for v in range(self.views)]
        return torch.from_numpy(np.concatenate(views)), label


def balanced_sampler(rows, hard_negative_boost=3.0):
    """Every class drawn equally often; hard negatives oversampled inside the normal class."""
    counts = Counter(label for _, label, _ in rows)
    w = [(1.0 / counts[label]) * (hard_negative_boost if kind == "h" else 1.0) for _, label, kind in rows]
    return WeightedRandomSampler(w, len(rows), replacement=True)


def worker_init(_):
    cv2.setNumThreads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _norm(x, norm="kinetics"):
    mean, std = (torch.tensor(v, device=x.device).view(1, 3, 1, 1, 1) for v in NORMS[norm])
    return (x - mean) / std


def gpu_train_transform(frames, size, norm="kinetics", scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3), flip=0.5,
                        jitter=0.3):
    """uint8 [B, T, H, W, 3] on GPU -> float [B, 3, T, size, size]: random resized crop, flip, colour jitter."""
    b, t, h, w, _ = frames.shape
    x = frames.permute(0, 4, 1, 2, 3).float().div_(255)
    out = torch.empty(b, 3, t, size, size, device=x.device)
    for i in range(b):
        area = h * w * random.uniform(*scale)
        ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = min(w, int(round(math.sqrt(area * ar)))), min(h, int(round(math.sqrt(area / ar))))
        y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
        crop = x[i, :, :, y0:y0 + ch, x0:x0 + cw].transpose(0, 1)
        crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        out[i] = (crop.flip(-1) if random.random() < flip else crop).transpose(0, 1)
    if jitter > 0:  # per-clip brightness / contrast / saturation, same for all frames
        rnd = lambda: 1 + (torch.rand(b, 1, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        out = out * rnd()
        avg = out.mean(dim=(1, 2, 3, 4), keepdim=True)
        out = (out - avg) * rnd() + avg
        gray = out.mean(dim=1, keepdim=True)
        out = ((out - gray) * rnd() + gray).clamp_(0, 1)
    return _norm(out, norm)


def gpu_eval_transform(frames, size, norm="kinetics"):
    """uint8 [B, T, H, W, 3] -> float [B, 3, T, size, size]: short-side resize (256/224 ratio) + centre crop."""
    b, t, h, w, _ = frames.shape
    x = frames.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w).float().div_(255)
    resize = int(round(size * 256 / 224))
    if min(h, w) != resize:
        s = resize / min(h, w)
        x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False, antialias=True)
    hh, ww = x.shape[-2:]
    y0, x0 = (hh - size) // 2, (ww - size) // 2
    x = x[:, :, y0:y0 + size, x0:x0 + size].reshape(b, t, 3, size, size).permute(0, 2, 1, 3, 4)
    return _norm(x, norm)


def mixup(x, y_onehot, alpha):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]
