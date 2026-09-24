"""Fine-tune TSM (Temporal Shift Module, ResNet-50, 8 segments) on the per-person exam-cheating clips.

TSM is a 2D ResNet-50 whose residual branches shift 1/8 of the channels one frame forward and 1/8 one frame
backward in time ("blockres", shift_div=8): temporal modelling at 2D-CNN cost (~33 GFLOPs / 8-frame clip).
Weights: OpenMMLab TSM-R50 Kinetics-400 checkpoint, remapped onto torchvision's ResNet-50.

Same sampler / GPU augmentation / EMA / metrics / outputs as train_x3d.py, so runs compare 1:1.

  python train_tsm.py --data /workspace/data/clips_hr
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_x3d import evaluate
from video_data import (IMAGENET_MEAN, IMAGENET_STD, ClipDataset, balanced_sampler, gpu_train_transform, mixup,
                        read_classes, read_split, worker_init)

NORM = dict(mean=IMAGENET_MEAN, std=IMAGENET_STD)
K400_URL = ("https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/"
            "tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/"
            "tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/clips_hr")
    p.add_argument("--out", default="/workspace/runs")
    p.add_argument("--pretrained", default="/workspace/pretrained/tsm_r50_k400_seg8.pth",
                   help="OpenMMLab TSM-R50 K400 checkpoint (downloaded if missing); 'imagenet' for torchvision weights")
    p.add_argument("--segments", type=int, default=8)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--shift-div", type=int, default=8)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3, help="head LR; backbone uses lr * backbone_lr_mult")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--hard-neg-boost", type=float, default=3.0)
    p.add_argument("--drop-related", action="store_true")
    p.add_argument("--test-views", type=int, default=3)
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    return p.parse_args()


class TemporalShift(nn.Module):
    """Wraps a conv: shifts channel folds along time before it (identical to mmaction2 / original TSM)."""

    def __init__(self, net, n_segment, shift_div):
        super().__init__()
        self.net, self.n_segment, self.shift_div = net, n_segment, shift_div

    def forward(self, x):
        nt, c, h, w = x.shape
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w)
        fold = c // self.shift_div
        out = torch.empty_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                     # pull from the next frame
        out[:, -1, :fold] = 0
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]     # pull from the previous frame
        out[:, 0, fold:2 * fold] = 0
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return self.net(out.view(nt, c, h, w))


class TSM(nn.Module):
    """Input [B, 3, T, H, W] (same as X3D) -> logits [B, C], average consensus over segments."""

    def __init__(self, num_classes, n_segment=8, shift_div=8, dropout=0.5):
        super().__init__()
        self.n_segment = n_segment
        r = torchvision.models.resnet50()
        for layer in (r.layer1, r.layer2, r.layer3, r.layer4):
            for blk in layer:
                blk.conv1 = TemporalShift(blk.conv1, n_segment, shift_div)
        r.fc = nn.Identity()
        self.backbone = r
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        b, c, t, h, w = x.shape
        assert t == self.n_segment, f"TSM built for {self.n_segment} segments, got {t}"
        x = x.transpose(1, 2).reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        f = self.backbone(x)
        return self.fc(self.dropout(f)).view(b, t, -1).mean(1)


def mmaction_to_torchvision(sd):
    """backbone.layer1.0.conv1.conv.net.weight -> backbone.layer1.0.conv1.net.weight, bn -> bnK, etc."""
    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        k = k[len("backbone."):]
        k = re.sub(r"^conv1\.conv\.", "conv1.", k)
        k = re.sub(r"^conv1\.bn\.", "bn1.", k)
        k = re.sub(r"\.downsample\.conv\.", ".downsample.0.", k)
        k = re.sub(r"\.downsample\.bn\.", ".downsample.1.", k)
        k = re.sub(r"\.conv(\d)\.bn\.", r".bn\1.", k)
        k = re.sub(r"\.conv1\.conv\.net\.", ".conv1.net.", k)
        k = re.sub(r"\.conv(\d)\.conv\.", r".conv\1.", k)
        out["backbone." + k] = v
    return out


def load_pretrained(model, path):
    if path == "imagenet":
        sd = torchvision.models.resnet50(weights="IMAGENET1K_V2").state_dict()
        sd = {("backbone." + (k.replace("conv1.", "conv1.net.", 1) if re.match(r"layer\d\.\d+\.conv1\.", k) else k)): v
              for k, v in sd.items() if not k.startswith("fc.")}
    else:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.hub.download_url_to_file(K400_URL, path)
        sd = mmaction_to_torchvision(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"fc.weight", "fc.bias"} and not unexpected, (missing[:6], unexpected[:6])
    return len(sd)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    classes = read_classes(args.data)
    nc = len(classes)
    src = "in1k" if args.pretrained == "imagenet" else "k400"
    tag = f"tsm_r50_{src}_seg{args.segments}_{os.path.basename(args.data.rstrip('/'))}"
    run = os.path.join(args.out, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run, "args.json"), "w"), indent=1)
    tb = SummaryWriter(run)

    train_rows = read_split(args.data, "train", args.drop_related)
    val_rows = read_split(args.data, "val", args.drop_related)
    ds_kw = dict(num_frames=args.segments, sampling="segment")
    dl_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                 worker_init_fn=worker_init)
    train_dl = DataLoader(ClipDataset(args.data, train_rows, train=True, **ds_kw), batch_size=args.bs,
                          sampler=balanced_sampler(train_rows, args.hard_neg_boost), drop_last=True, **dl_kw)
    val_dl = DataLoader(ClipDataset(args.data, val_rows, train=False, **ds_kw), batch_size=args.bs,
                        shuffle=False, **dl_kw)

    model = TSM(nc, args.segments, args.shift_div, args.dropout)
    n_loaded = load_pretrained(model, args.pretrained)
    model = model.cuda().to(memory_format=torch.channels_last)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None

    head = set(id(p) for p in model.fc.parameters())
    groups = []
    for is_head, lr in ((True, args.lr), (False, args.lr * args.backbone_lr_mult)):
        ps = [p for p in model.parameters() if (id(p) in head) == is_head]
        groups += [dict(params=[p for p in ps if p.ndim > 1], lr=lr, weight_decay=args.wd),
                   dict(params=[p for p in ps if p.ndim <= 1], lr=lr, weight_decay=0.0)]
    opt = torch.optim.AdamW(groups)
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = args.max_steps or len(train_dl)
    total_steps, warm_steps = args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)

    def set_lr(step):
        f = (step + 1) / warm_steps if step < warm_steps else \
            0.5 * (1 + math.cos(math.pi * (step - warm_steps) / max(1, total_steps - warm_steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    print(f"run dir: {run}\ntrain {len(train_rows)} clips, val {len(val_rows)} clips, classes {classes}\n"
          f"TSM-R50 ({src}, {n_loaded} tensors loaded): {args.segments} segments @ {args.size}px, "
          f"{steps_per_epoch} steps/epoch", flush=True)

    best, step, history = -1.0, 0, []
    epoch_bar = tqdm(range(args.epochs), desc="epochs", unit="ep", mininterval=5)
    for epoch in epoch_bar:
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        bar = tqdm(train_dl, desc=f"train {epoch + 1}/{args.epochs}", total=steps_per_epoch, leave=False, mininterval=2)
        for i, (frames, y) in enumerate(bar):
            if args.max_steps and i >= args.max_steps:
                break
            set_lr(step)
            frames, y = frames.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x = gpu_train_transform(frames, args.size, **NORM)
            target = F.one_hot(y, nc).float() * (1 - args.label_smoothing) + args.label_smoothing / nc
            if args.mixup > 0 and np.random.rand() < args.mixup_prob:
                x, target = mixup(x, target, args.mixup)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = torch.sum(-target * F.log_softmax(logits.float(), -1), -1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
            step += 1
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            seen += y.size(0)
            if step % 10 == 0:
                tb.add_scalar("train/loss", loss.item(), step)
                tb.add_scalar("train/grad_norm", gn.item(), step)
            bar.set_postfix(loss=f"{loss_sum / seen:.3f}", acc=f"{correct / seen:.3f}")
        train_time = time.time() - t0

        eval_model = ema.module if ema is not None else model
        m = evaluate(eval_model, val_dl, args.size, 1, nc, f"val {epoch + 1}", norm=NORM)
        rec = dict(epoch=epoch + 1, train_loss=loss_sum / seen, train_acc=correct / seen, clips_per_s=seen / train_time,
                   **{k: v for k, v in m.items() if k not in ("confusion", "per_class")}, val_f1=m["per_class"]["f1"])
        history.append(rec)
        for k in ("acc", "macro_f1", "cheat_auroc", "cheat_recall", "cheat_precision"):
            tb.add_scalar(f"val/{k}", m[k], epoch + 1)
        for c, f in zip(classes, m["per_class"]["f1"]):
            tb.add_scalar(f"val_f1/{c}", f, epoch + 1)

        state = dict(model=eval_model.state_dict(), classes=classes, args=vars(args), epoch=epoch + 1, metrics=m)
        torch.save(state, os.path.join(run, "last.pt"))
        tag = ""
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(state, os.path.join(run, "best.pt"))
            tag = "  *best*"
        tqdm.write(f"ep {epoch + 1:3d} | loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} "
                   f"({rec['clips_per_s']:.0f} clip/s) | val acc {m['acc']:.3f} macroF1 {m['macro_f1']:.3f} "
                   f"cheat AUROC {m['cheat_auroc']:.3f} R {m['cheat_recall']:.3f} P {m['cheat_precision']:.3f} | "
                   f"F1 {' '.join(f'{f:.2f}' for f in m['per_class']['f1'])}{tag}")
        epoch_bar.set_postfix(best_f1=f"{best:.3f}")
        json.dump(history, open(os.path.join(run, "history.json"), "w"), indent=1)

    final_model = ema.module if ema is not None else model
    final_model.load_state_dict(torch.load(os.path.join(run, "best.pt"), weights_only=False)["model"])
    test_dl = DataLoader(ClipDataset(args.data, val_rows, train=False, views=args.test_views, **ds_kw),
                         batch_size=max(1, args.bs // args.test_views), shuffle=False,
                         **{**dl_kw, "persistent_workers": False})
    m = evaluate(final_model, test_dl, args.size, args.test_views, nc, "final test", norm=NORM)
    json.dump(m, open(os.path.join(run, "final_val_multiview.json"), "w"), indent=1)
    print(f"\nBEST epoch macroF1 {best:.3f}  |  {args.test_views}-view: acc {m['acc']:.3f} macroF1 {m['macro_f1']:.3f} "
          f"cheat AUROC {m['cheat_auroc']:.3f} R {m['cheat_recall']:.3f} P {m['cheat_precision']:.3f}")
    print("classes   :", classes)
    print("F1        :", [round(f, 3) for f in m["per_class"]["f1"]])
    print("recall    :", [round(f, 3) for f in m["per_class"]["recall"]])
    print("confusion (rows=true):")
    for c, r in zip(classes, m["confusion"]):
        print(f"  {c:17s} {r}")
    tb.close()


if __name__ == "__main__":
    main()
