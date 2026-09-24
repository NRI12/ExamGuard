"""Distil a trained V-JEPA 2 teacher into a real-time X3D student.

Online distillation: every step the teacher and the student see the *same* augmented (and mixed-up) clip,
so the soft targets always match what the student is looking at. Teacher runs frozen in bf16, no grads.

  loss = alpha * T^2 * KL(softmax(t/T) || softmax(s/T)) + (1 - alpha) * CE(label-smoothed / mixup target)

  python train_distill.py --teacher /workspace/runs/<vjepa2 run>/best.pt --student x3d_l --bs 12
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_vjepa2 import VJEPA2Classifier
from train_x3d import VARIANTS, build_model, evaluate, param_groups
from video_data import (IMAGENET_MEAN, IMAGENET_STD, KINETICS_MEAN, KINETICS_STD, ClipDataset, _norm,
                        balanced_sampler, gpu_train_transform, mixup, read_classes, read_split, worker_init)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, help="best.pt from train_vjepa2.py")
    p.add_argument("--data", default="/workspace/data/clips_hr")
    p.add_argument("--out", default="/workspace/runs")
    p.add_argument("--student", default="x3d_l", choices=list(VARIANTS))
    p.add_argument("--stride", type=int, default=5, help="must match the teacher's frames (16x5)")
    p.add_argument("--alpha", type=float, default=0.7, help="weight of the distillation term")
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--hard-neg-boost", type=float, default=3.0)
    p.add_argument("--test-views", type=int, default=3)
    p.add_argument("--workers", type=int, default=11)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    return p.parse_args()


def load_teacher(path, num_classes):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    targs = ck["args"]
    t = VJEPA2Classifier(ck["hf_model"], num_classes, targs.get("unfreeze_last", 0))
    missing, unexpected = t.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all(k.startswith("m.vjepa2.") for k in missing), (missing[:5], unexpected[:5])
    t = t.cuda().to(torch.bfloat16).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t, targs, ck["metrics"]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    classes = read_classes(args.data)
    nc = len(classes)
    teacher, targs, tmetrics = load_teacher(args.teacher, nc)
    frames_n, size = VARIANTS[args.student]
    assert (frames_n, args.stride) == (targs["frames"], targs["stride"]), \
        f"student samples {frames_n}x{args.stride}, teacher was trained on {targs['frames']}x{targs['stride']}"
    t_size = targs["size"]

    tag = f"distill_{args.student}_s{args.stride}_a{args.alpha}_T{args.temp}_{os.path.basename(args.data.rstrip('/'))}"
    run = os.path.join(args.out, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run, "args.json"), "w"), indent=1)
    tb = SummaryWriter(run)

    train_rows = read_split(args.data, "train")
    val_rows = read_split(args.data, "val")
    dl_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                 worker_init_fn=worker_init)
    train_dl = DataLoader(ClipDataset(args.data, train_rows, frames_n, args.stride, train=True), batch_size=args.bs,
                          sampler=balanced_sampler(train_rows, args.hard_neg_boost), drop_last=True, **dl_kw)
    val_dl = DataLoader(ClipDataset(args.data, val_rows, frames_n, args.stride, train=False), batch_size=args.bs,
                        shuffle=False, **dl_kw)

    model = build_model(args.student, nc, args.dropout).cuda()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.backbone_lr_mult, args.wd))
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = args.max_steps or len(train_dl)
    total_steps, warm_steps = args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)

    def set_lr(step):
        f = (step + 1) / warm_steps if step < warm_steps else \
            0.5 * (1 + math.cos(math.pi * (step - warm_steps) / max(1, total_steps - warm_steps)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    print(f"run dir: {run}\nteacher {args.teacher} (val macroF1 {tmetrics['macro_f1']:.3f}, "
          f"cheat AUROC {tmetrics['cheat_auroc']:.3f})\nstudent {args.student}: {frames_n}x{args.stride} @ {size}px, "
          f"alpha {args.alpha}, T {args.temp}, {steps_per_epoch} steps/epoch", flush=True)

    best, step, history = -1.0, 0, []
    epoch_bar = tqdm(range(args.epochs), desc="epochs", unit="ep", mininterval=5)
    for epoch in epoch_bar:
        model.train()
        t0, sums, correct, seen, agree = time.time(), dict(loss=0.0, kd=0.0, ce=0.0), 0, 0, 0
        bar = tqdm(train_dl, desc=f"train {epoch + 1}/{args.epochs}", total=steps_per_epoch, leave=False, mininterval=2)
        for i, (frames, y) in enumerate(bar):
            if args.max_steps and i >= args.max_steps:
                break
            set_lr(step)
            frames, y = frames.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # one augmented view in [0, 1], shared by teacher and student
            raw = gpu_train_transform(frames, size, mean=(0, 0, 0), std=(1, 1, 1))
            target = F.one_hot(y, nc).float() * (1 - args.label_smoothing) + args.label_smoothing / nc
            if args.mixup > 0 and np.random.rand() < args.mixup_prob:
                raw, target = mixup(raw, target, args.mixup)
            b, c, t, h, w = raw.shape
            t_in = raw.transpose(1, 2).reshape(b * t, c, h, w)
            if h != t_size:
                t_in = F.interpolate(t_in, size=(t_size, t_size), mode="bilinear", align_corners=False, antialias=True)
            t_in = _norm(t_in.reshape(b, t, c, t_size, t_size).transpose(1, 2), IMAGENET_MEAN, IMAGENET_STD)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logits = teacher(t_in.to(torch.bfloat16)).float()
            x = _norm(raw, KINETICS_MEAN, KINETICS_STD)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            kd = F.kl_div(F.log_softmax(logits / args.temp, -1), F.softmax(t_logits / args.temp, -1),
                          reduction="batchmean") * args.temp ** 2
            ce = torch.sum(-target * F.log_softmax(logits, -1), -1).mean()
            loss = args.alpha * kd + (1 - args.alpha) * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
            step += 1
            for k, v in (("loss", loss), ("kd", kd), ("ce", ce)):
                sums[k] += v.item() * b
            correct += (logits.argmax(-1) == y).sum().item()
            agree += (logits.argmax(-1) == t_logits.argmax(-1)).sum().item()
            seen += b
            if step % 10 == 0:
                for k, v in (("loss", loss), ("kd", kd), ("ce", ce), ("grad_norm", gn)):
                    tb.add_scalar(f"train/{k}", v.item(), step)
            bar.set_postfix(loss=f"{sums['loss'] / seen:.3f}", kd=f"{sums['kd'] / seen:.3f}",
                            agree=f"{agree / seen:.2f}")
        train_time = time.time() - t0

        eval_model = ema.module if ema is not None else model
        m = evaluate(eval_model, val_dl, size, 1, nc, f"val {epoch + 1}")
        rec = dict(epoch=epoch + 1, **{f"train_{k}": v / seen for k, v in sums.items()}, train_acc=correct / seen,
                   teacher_agreement=agree / seen, clips_per_s=seen / train_time,
                   **{k: v for k, v in m.items() if k not in ("confusion", "per_class")}, val_f1=m["per_class"]["f1"])
        history.append(rec)
        for k in ("acc", "macro_f1", "cheat_auroc", "cheat_recall", "cheat_precision"):
            tb.add_scalar(f"val/{k}", m[k], epoch + 1)
        tb.add_scalar("train/teacher_agreement", agree / seen, epoch + 1)
        state = dict(model=eval_model.state_dict(), classes=classes, args=vars(args), epoch=epoch + 1, metrics=m,
                     student=args.student)
        torch.save(state, os.path.join(run, "last.pt"))
        tag = ""
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(state, os.path.join(run, "best.pt"))
            tag = "  *best*"
        tqdm.write(f"ep {epoch + 1:3d} | loss {rec['train_loss']:.3f} kd {rec['train_kd']:.3f} ce {rec['train_ce']:.3f} "
                   f"agree {agree / seen:.2f} ({rec['clips_per_s']:.0f} clip/s) | val acc {m['acc']:.3f} "
                   f"macroF1 {m['macro_f1']:.3f} cheat AUROC {m['cheat_auroc']:.3f} R {m['cheat_recall']:.3f} "
                   f"P {m['cheat_precision']:.3f} | F1 {' '.join(f'{f:.2f}' for f in m['per_class']['f1'])}{tag}")
        epoch_bar.set_postfix(best_f1=f"{best:.3f}")
        json.dump(history, open(os.path.join(run, "history.json"), "w"), indent=1)

    del teacher
    torch.cuda.empty_cache()
    final_model = ema.module if ema is not None else model
    final_model.load_state_dict(torch.load(os.path.join(run, "best.pt"), weights_only=False)["model"])
    test_dl = DataLoader(ClipDataset(args.data, val_rows, frames_n, args.stride, train=False, views=args.test_views),
                         batch_size=max(1, args.bs // args.test_views), shuffle=False,
                         **{**dl_kw, "persistent_workers": False})
    m = evaluate(final_model, test_dl, size, args.test_views, nc, "final test")
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
