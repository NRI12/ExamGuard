"""Fine-tune X3D (pytorchvideo, Kinetics-400 pretrained) on the per-person exam-cheating clips.

Optimisations: bf16 autocast, cudnn.benchmark + TF32, uint8 transfer with GPU-side
augmentation, class-balanced sampling with hard-negative boost, AdamW with separate head/backbone LR,
warmup + cosine, label smoothing + mixup, gradient clipping, weight EMA,
best checkpoint by val macro-F1, multi-view test at the end. Logs to TensorBoard (portal tab).

  python train_x3d.py --data /workspace/data/clips --epochs 30
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
from pytorchvideo.models import hub
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from video_data import (ClipDataset, balanced_sampler, gpu_eval_transform, gpu_train_transform, mixup,
                        read_classes, read_split, worker_init)

# variant -> (frames, crop size); stride tuned for 25 fps clips
VARIANTS = {"x3d_xs": (4, 160), "x3d_s": (13, 160), "x3d_m": (16, 224), "x3d_l": (16, 312)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data/clips")
    p.add_argument("--out", default="/workspace/runs")
    p.add_argument("--model", default="x3d_m", choices=list(VARIANTS))
    p.add_argument("--stride", type=int, default=4, help="frame stride (16x4 = 2.4 s at 25 fps)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup", type=float, default=2.0, help="warmup epochs")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="head LR; backbone uses lr * backbone_lr_mult")
    p.add_argument("--backbone-lr-mult", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2, help="beta alpha; 0 disables")
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--ema", type=float, default=0.99, help="per-step EMA decay; 0 disables")
    p.add_argument("--hard-neg-boost", type=float, default=3.0)
    p.add_argument("--drop-related", action="store_true", help="drop passive related-actor clips")
    p.add_argument("--test-views", type=int, default=3)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--compile", action="store_true", help="~5%% faster on X3D, slow first epoch")
    p.add_argument("--channels-last", action="store_true",
                   help="off by default: 2x slower for X3D depthwise conv3d on RTX 3090")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    return p.parse_args()


def build_model(name, num_classes, dropout):
    m = getattr(hub, name)(pretrained=True)
    head = m.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, num_classes)
    head.dropout = nn.Dropout(dropout)
    return m


def param_groups(model, lr, backbone_mult, wd):
    head_ids = {id(p) for p in model.blocks[-1].proj.parameters()}
    groups = {k: [] for k in ("head_decay", "head_no", "bb_decay", "bb_no")}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "bb"
        decay = "decay" if p.ndim > 1 else "no"  # no weight decay on BN / bias
        groups[f"{part}_{decay}"].append(p)
    return [
        dict(params=groups["head_decay"], lr=lr, weight_decay=wd),
        dict(params=groups["head_no"], lr=lr, weight_decay=0.0),
        dict(params=groups["bb_decay"], lr=lr * backbone_mult, weight_decay=wd),
        dict(params=groups["bb_no"], lr=lr * backbone_mult, weight_decay=0.0),
    ]


@torch.no_grad()
def evaluate(model, loader, size, views, num_classes, desc, mem_fmt=torch.contiguous_format, norm=None):
    model.eval()
    probs, labels = [], []
    for frames, y in tqdm(loader, desc=desc, leave=False, mininterval=2):
        frames = frames.cuda(non_blocking=True)
        b = frames.size(0)
        t = frames.size(1) // views
        frames = frames.view(b * views, t, *frames.shape[2:])
        x = gpu_eval_transform(frames, size, **(norm or {})).contiguous(memory_format=mem_fmt)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
        probs.append(logits.softmax(-1).view(b, views, -1).mean(1).cpu())
        labels.append(y)
    p = torch.cat(probs).numpy()
    y = torch.cat(labels).numpy()
    pred = p.argmax(1)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=range(num_classes), zero_division=0)
    cheat_true = y != 0
    cheat_score = 1 - p[:, 0]
    cheat_pred = pred != 0
    tp = (cheat_true & cheat_pred).sum()
    return dict(
        acc=float((pred == y).mean()), macro_f1=float(f1.mean()),
        per_class=dict(precision=prec.tolist(), recall=rec.tolist(), f1=f1.tolist(), support=sup.tolist()),
        cheat_auroc=float(roc_auc_score(cheat_true, cheat_score)) if 0 < cheat_true.sum() < len(y) else float("nan"),
        cheat_precision=float(tp / max(1, cheat_pred.sum())), cheat_recall=float(tp / max(1, cheat_true.sum())),
        confusion=confusion_matrix(y, pred, labels=range(num_classes)).tolist(),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    frames_n, size = VARIANTS[args.model]
    classes = read_classes(args.data)
    nc = len(classes)
    tag = f"{args.model}_s{args.stride}_{os.path.basename(args.data.rstrip('/'))}"
    run = os.path.join(args.out, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run, "args.json"), "w"), indent=1)
    tb = SummaryWriter(run)

    train_rows = read_split(args.data, "train", args.drop_related)
    val_rows = read_split(args.data, "val", args.drop_related)
    dl_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                 worker_init_fn=worker_init)
    train_dl = DataLoader(ClipDataset(args.data, train_rows, frames_n, args.stride, train=True), batch_size=args.bs,
                          sampler=balanced_sampler(train_rows, args.hard_neg_boost), drop_last=True, **dl_kw)
    val_dl = DataLoader(ClipDataset(args.data, val_rows, frames_n, args.stride, train=False), batch_size=args.bs,
                        shuffle=False, **dl_kw)

    model = build_model(args.model, nc, args.dropout).cuda()
    mem_fmt = torch.channels_last_3d if args.channels_last else torch.contiguous_format
    model = model.to(memory_format=mem_fmt)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None
    fwd = torch.compile(model) if args.compile else model

    opt = torch.optim.AdamW(param_groups(model, args.lr, args.backbone_lr_mult, args.wd), betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = args.max_steps or len(train_dl)
    total_steps, warm_steps = args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)

    def set_lr(step):
        if step < warm_steps:
            f = (step + 1) / warm_steps
        else:
            f = 0.5 * (1 + math.cos(math.pi * (step - warm_steps) / max(1, total_steps - warm_steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    print(f"run dir: {run}\ntrain {len(train_rows)} clips, val {len(val_rows)} clips, classes {classes}\n"
          f"{args.model}: {frames_n}x{args.stride} frames @ {size}px, {steps_per_epoch} steps/epoch", flush=True)

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
            x = gpu_train_transform(frames, size)
            target = F.one_hot(y, nc).float()
            target = target * (1 - args.label_smoothing) + args.label_smoothing / nc
            if args.mixup > 0 and np.random.rand() < args.mixup_prob:
                x, target = mixup(x, target, args.mixup)
            x = x.contiguous(memory_format=mem_fmt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(x)
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
                tb.add_scalar("train/lr_head", opt.param_groups[0]["lr"], step)
            bar.set_postfix(loss=f"{loss_sum / seen:.3f}", acc=f"{correct / seen:.3f}")
        train_time = time.time() - t0

        eval_model = ema.module if ema is not None else model
        m = evaluate(eval_model, val_dl, size, 1, nc, f"val {epoch + 1}", mem_fmt)
        rec = dict(epoch=epoch + 1, train_loss=loss_sum / seen, train_acc=correct / seen, clips_per_s=seen / train_time,
                   **{k: v for k, v in m.items() if k not in ("confusion", "per_class")}, val_f1=m["per_class"]["f1"])
        history.append(rec)
        for k in ("acc", "macro_f1", "cheat_auroc", "cheat_recall", "cheat_precision"):
            tb.add_scalar(f"val/{k}", m[k], epoch + 1)
        for c, f in zip(classes, m["per_class"]["f1"]):
            tb.add_scalar(f"val_f1/{c}", f, epoch + 1)
        tb.add_scalar("train/epoch_loss", rec["train_loss"], epoch + 1)
        tb.add_scalar("train/clips_per_s", rec["clips_per_s"], epoch + 1)

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

    # final multi-view test with the best checkpoint
    final_model = ema.module if ema is not None else model
    final_model.load_state_dict(torch.load(os.path.join(run, "best.pt"), weights_only=False)["model"])
    test_dl = DataLoader(ClipDataset(args.data, val_rows, frames_n, args.stride, train=False, views=args.test_views),
                         batch_size=max(1, args.bs // args.test_views), shuffle=False,
                         **{**dl_kw, "persistent_workers": False})
    m = evaluate(final_model, test_dl, size, args.test_views, nc, "final test", mem_fmt)
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
