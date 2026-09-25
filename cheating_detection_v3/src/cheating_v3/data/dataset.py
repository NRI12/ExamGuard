"""Clip dataset (RGB and / or optical flow), balanced sampler and GPU-side augmentation.

Workers only decode (cv2 / npz) and return uint8 arrays; crops, flips, jitter and normalisation run batched on the
GPU. RGB and flow use the same frame indices: `frames` frames `stride` apart; flow frame f is the flow f-stride -> f,
so the window never starts before frame `stride` when flow is used.
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

from ..models.raft_flow import INPUT_DIV, Q_SCALE, Q_ZERO
from .flow import flow_path

KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)
EMPTY = torch.zeros(0, dtype=torch.uint8)


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
    """Returns (rgb uint8 [V*T, H, W, 3] or empty, flow uint8 [V*T, h, w, 2] or empty, label).
    views > 1 stacks evenly spaced temporal views along T for multi-view testing."""

    def __init__(self, clips_dir, rows, num_frames, stride, train=True, views=1, rgb=True, flow_dir=None):
        self.root, self.rows, self.flow_dir, self.rgb = clips_dir, rows, flow_dir, rgb
        self.num_frames, self.stride, self.train, self.views = num_frames, stride, train, views
        self.clip_len = clip_len_of(clips_dir)
        self.first = stride if flow_dir else 0  # flow[f] needs frame f - stride

    def __len__(self):
        return len(self.rows)

    def _indices(self, view=0):
        span = (self.num_frames - 1) * self.stride + 1
        lo, hi = self.first, max(self.first, self.clip_len - span)
        if self.train:
            start = random.randint(lo, hi)
        elif self.views == 1:
            start = (lo + hi) // 2
        else:
            start = lo + round(view * (hi - lo) / (self.views - 1))
        return [min(self.clip_len - 1, start + i * self.stride) for i in range(self.num_frames)]

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
        views = [self._indices(v) for v in range(self.views)]
        rgb = torch.from_numpy(np.concatenate([self._decode(path, w) for w in views])) if self.rgb else EMPTY
        flow = EMPTY
        if self.flow_dir:
            f = np.load(flow_path(self.flow_dir, path))["flow"]
            flow = torch.from_numpy(np.concatenate([f[np.minimum(w, len(f) - 1)] for w in views]))
        return rgb, flow, label


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


def _norm(x):
    mean = torch.tensor(KINETICS_MEAN, device=x.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD, device=x.device).view(1, 3, 1, 1, 1)
    return (x - mean) / std


def _random_box(h, w, scale, ratio):
    area = h * w * random.uniform(*scale)
    ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
    cw, ch = min(w, int(round(math.sqrt(area * ar)))), min(h, int(round(math.sqrt(area / ar))))
    return random.randint(0, h - ch), random.randint(0, w - cw), ch, cw


def gpu_train_transform(frames, size, scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3), flip=0.5, jitter=0.3):
    """RGB uint8 [B, T, H, W, 3] on GPU -> float [B, 3, T, size, size]: random resized crop, flip, colour jitter."""
    b, t, h, w, _ = frames.shape
    x = frames.permute(0, 4, 1, 2, 3).float().div_(255)
    out = torch.empty(b, 3, t, size, size, device=x.device)
    for i in range(b):
        y0, x0, ch, cw = _random_box(h, w, scale, ratio)
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
    return _norm(out)


def gpu_eval_transform(frames, size):
    """RGB uint8 [B, T, H, W, 3] -> float [B, 3, T, size, size]: short-side resize (256/224 ratio) + centre crop."""
    b, t, h, w, _ = frames.shape
    x = frames.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w).float().div_(255)
    resize = int(round(size * 256 / 224))
    if min(h, w) != resize:
        s = resize / min(h, w)
        x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False, antialias=True)
    hh, ww = x.shape[-2:]
    y0, x0 = (hh - size) // 2, (ww - size) // 2
    x = x[:, :, y0:y0 + size, x0:x0 + size].reshape(b, t, 3, size, size).permute(0, 2, 1, 3, 4)
    return _norm(x)


def _flow_px(flow_u8):
    return (flow_u8.float() - Q_ZERO) / Q_SCALE  # uint8 -> pixels at stored resolution


def gpu_flow_train_transform(flow_u8, size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3), flip=0.5):
    """Flow uint8 [B, T, H, W, 2] -> float [B, 2, T, size, size]. Resizing rescales the vectors; a horizontal
    flip negates the x component."""
    b, t, h, w, _ = flow_u8.shape
    x = _flow_px(flow_u8).permute(0, 4, 1, 2, 3)
    out = torch.empty(b, 2, t, size, size, device=x.device)
    for i in range(b):
        y0, x0, ch, cw = _random_box(h, w, scale, ratio)
        crop = x[i, :, :, y0:y0 + ch, x0:x0 + cw].transpose(0, 1)  # T, 2, ch, cw
        crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
        crop[:, 0] *= size / cw
        crop[:, 1] *= size / ch
        if random.random() < flip:
            crop = crop.flip(-1)
            crop[:, 0] = -crop[:, 0]
        out[i] = crop.transpose(0, 1)
    return out / INPUT_DIV


def gpu_flow_eval_transform(flow_u8, size):
    """Flow uint8 [B, T, H, W, 2] -> float [B, 2, T, size, size] (whole field, resized if needed)."""
    b, t, h, w, _ = flow_u8.shape
    x = _flow_px(flow_u8).permute(0, 1, 4, 2, 3).reshape(b * t, 2, h, w)
    if (h, w) != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x[:, 0] *= size / w
        x[:, 1] *= size / h
    return x.reshape(b, t, 2, size, size).permute(0, 2, 1, 3, 4) / INPUT_DIV


def mixup(xs, y_onehot, alpha):
    """Mix every input tensor of the batch (rgb and / or flow) with the same permutation and weight."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(y_onehot.size(0), device=y_onehot.device)
    mixed = [None if x is None else lam * x + (1 - lam) * x[perm] for x in xs]
    return mixed, lam * y_onehot + (1 - lam) * y_onehot[perm]
