"""Fine-tune V-JEPA 2 (Meta, HF transformers) on the per-person exam-cheating clips.

Default recipe for small data: frozen ViT encoder, train the attentive pooler (initialised from the
SSv2 classifier, a motion-centric dataset) + a new linear head. --unfreeze-last N also trains the last
N encoder blocks at a lower LR.

Same data pipeline, sampler, augmentation, EMA, metrics and outputs as train_x3d.py, so runs compare 1:1.

  python train_vjepa2.py --data /workspace/data/clips_hr
  python train_vjepa2.py --data /workspace/data/clips_hr --unfreeze-last 4
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForVideoClassification
from transformers.utils import logging as hf_logging

from train_x3d import evaluate
from video_data import (IMAGENET_MEAN, IMAGENET_STD, ClipDataset, balanced_sampler, gpu_train_transform, mixup,
                        read_classes, read_split, worker_init)

hf_logging.set_verbosity_error()
NORM = dict(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/clips_hr")
    p.add_argument("--out", default="/workspace/runs")
    p.add_argument("--hf-model", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--stride", type=int, default=5, help="16x5 = 3.0 s at 25 fps")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--unfreeze-last", type=int, default=0, help="train the last N encoder blocks too")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr-head", type=float, default=1e-3, help="new linear classifier")
    p.add_argument("--lr-pooler", type=float, default=1e-4, help="attentive pooler (pretrained)")
    p.add_argument("--lr-encoder", type=float, default=1e-5, help="unfrozen encoder blocks")
    p.add_argument("--wd", type=float, default=0.05)
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


class VJEPA2Classifier(nn.Module):
    """Takes [B, 3, T, H, W] like X3D; frozen encoder runs without autograd."""

    def __init__(self, name, num_classes, unfreeze_last):
        super().__init__()
        self.m = AutoModelForVideoClassification.from_pretrained(
            name, num_labels=num_classes, ignore_mismatched_sizes=True, attn_implementation="sdpa")
        self.unfreeze_last = unfreeze_last
        enc = self.m.vjepa2
        for p in enc.parameters():
            p.requires_grad_(False)
        if unfreeze_last:
            for blk in enc.encoder.layer[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for p in enc.encoder.layernorm.parameters():
                p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.m.vjepa2.eval()  # frozen blocks stay deterministic; unfrozen ones have no dropout either
        return self

    def forward(self, x):
        v = x.transpose(1, 2)  # B,T,3,H,W
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.unfreeze_last > 0):
            h = self.m.vjepa2(pixel_values_videos=v, skip_predictor=True).last_hidden_state
        return self.m.classifier(self.m.pooler(h))

    def trainable_state_dict(self):
        """Everything except frozen encoder weights (the 1.3 GB encoder is re-downloaded from the hub)."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items()
                if k in keep or not k.startswith("m.vjepa2.")}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    classes = read_classes(args.data)
    nc = len(classes)
    short = args.hf_model.split("/")[-1].replace("vjepa2-", "vjepa2_")
    tag = f"{short}_u{args.unfreeze_last}_s{args.stride}_{os.path.basename(args.data.rstrip('/'))}"
    run = os.path.join(args.out, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run, "args.json"), "w"), indent=1)
    tb = SummaryWriter(run)

    train_rows = read_split(args.data, "train", args.drop_related)
    val_rows = read_split(args.data, "val", args.drop_related)
    dl_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                 worker_init_fn=worker_init)
    train_dl = DataLoader(ClipDataset(args.data, train_rows, args.frames, args.stride, train=True), batch_size=args.bs,
                          sampler=balanced_sampler(train_rows, args.hard_neg_boost), drop_last=True, **dl_kw)
    val_dl = DataLoader(ClipDataset(args.data, val_rows, args.frames, args.stride, train=False), batch_size=args.bs,
                        shuffle=False, **dl_kw)

    model = VJEPA2Classifier(args.hf_model, nc, args.unfreeze_last).cuda()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None

    def group(params, lr):
        params = [p for p in params if p.requires_grad]
        return [dict(params=[p for p in params if p.ndim > 1], lr=lr, weight_decay=args.wd),
                dict(params=[p for p in params if p.ndim <= 1], lr=lr, weight_decay=0.0)]
    groups = group(model.m.classifier.parameters(), args.lr_head) + group(model.m.pooler.parameters(), args.lr_pooler)
    if args.unfreeze_last:
        groups += group(model.m.vjepa2.parameters(), args.lr_encoder)
    opt = torch.optim.AdamW([g for g in groups if g["params"]])
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = args.max_steps or len(train_dl)
    total_steps, warm_steps = args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)

    def set_lr(step):
        f = (step + 1) / warm_steps if step < warm_steps else \
            0.5 * (1 + math.cos(math.pi * (step - warm_steps) / max(1, total_steps - warm_steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"run dir: {run}\ntrain {len(train_rows)} clips, val {len(val_rows)} clips, classes {classes}\n"
          f"{args.hf_model}: {args.frames}x{args.stride} frames @ {args.size}px, trainable {n_train:.1f}M params, "
          f"{steps_per_epoch} steps/epoch", flush=True)

    best, step, history = -1.0, 0, []
    trainable = [p for p in model.parameters() if p.requires_grad]
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
            gn = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
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

        state = dict(model=eval_model.trainable_state_dict(), hf_model=args.hf_model, classes=classes,
                     args=vars(args), epoch=epoch + 1, metrics=m)
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
    final_model.load_state_dict(torch.load(os.path.join(run, "best.pt"), weights_only=False)["model"], strict=False)
    test_dl = DataLoader(ClipDataset(args.data, val_rows, args.frames, args.stride, train=False, views=args.test_views),
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
