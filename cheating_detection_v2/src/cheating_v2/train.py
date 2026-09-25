"""Fine-tune X3D (Kinetics-400) on the per-person clips.

bf16 autocast, cudnn.benchmark + TF32, uint8 transfer with GPU augmentation, class-balanced sampling with
hard-negative boost, AdamW (separate head / backbone LR, no decay on BN / bias), warmup + cosine, label smoothing
+ mixup, gradient clipping, weight EMA, best checkpoint by val macro-F1, multi-view test at the end.
Runs go to outputs/runs/<project>_<time>/ with TensorBoard events.

  python -m cheating_v2 train [--set train.epochs=15]
"""
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import config_parser, load_config
from .data.dataset import (ClipDataset, balanced_sampler, gpu_train_transform, mixup, read_classes, read_split,
                           worker_init)
from .evaluate import evaluate, print_report
from .models.x3d import VARIANTS, build_x3d, save_checkpoint


def param_groups(model, lr, backbone_mult, wd):
    head = {id(p) for p in model.blocks[-1].proj.parameters()}
    groups = []
    for is_head, glr in ((True, lr), (False, lr * backbone_mult)):
        ps = [p for p in model.parameters() if p.requires_grad and (id(p) in head) == is_head]
        groups += [dict(params=[p for p in ps if p.ndim > 1], lr=glr, weight_decay=wd),
                   dict(params=[p for p in ps if p.ndim <= 1], lr=glr, weight_decay=0.0)]
    return groups


def main(argv=None):
    args = config_parser(__doc__.splitlines()[0]).parse_args(argv)
    cfg = load_config(args.config, args.set)
    tc, mc = cfg["train"], cfg["model"]
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    clips = cfg["paths"]["clips"]
    classes = read_classes(clips)
    assert classes == cfg["data"]["classes"], f"clips have {classes}, config expects {cfg['data']['classes']}"
    nc = len(classes)
    frames_n, size = VARIANTS[mc["name"]]
    run = os.path.join(cfg["paths"]["outputs"], "runs", f"{cfg['project']}_{mc['name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run, exist_ok=True)
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"), indent=1)
    tb = SummaryWriter(run)

    train_rows = read_split(clips, "train", tc["drop_related"])
    val_rows = read_split(clips, "val", tc["drop_related"])
    dl_kw = dict(num_workers=tc["workers"], pin_memory=True, persistent_workers=True, prefetch_factor=4,
                 worker_init_fn=worker_init)
    train_dl = DataLoader(ClipDataset(clips, train_rows, frames_n, mc["stride"]), batch_size=tc["batch_size"],
                          sampler=balanced_sampler(train_rows, tc["hard_neg_boost"]), drop_last=True, **dl_kw)
    val_dl = DataLoader(ClipDataset(clips, val_rows, frames_n, mc["stride"], train=False),
                        batch_size=tc["batch_size"], **dl_kw)

    model = build_x3d(mc["name"], nc, mc["dropout"]).cuda()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(tc["ema"]), use_buffers=True) if tc["ema"] else None
    opt = torch.optim.AdamW(param_groups(model, tc["lr"], tc["backbone_lr_mult"], tc["weight_decay"]))
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps = tc["max_steps"] or len(train_dl)
    total, warm = tc["epochs"] * steps, int(tc["warmup"] * steps)

    def set_lr(step):
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    print(f"run dir: {run}\ntrain {len(train_rows)} clips, val {len(val_rows)} clips, classes {classes}\n"
          f"{mc['name']}: {frames_n}x{mc['stride']} frames @ {size}px, {steps} steps/epoch", flush=True)
    best, step, history = -1.0, 0, []
    epoch_bar = tqdm(range(tc["epochs"]), desc="epochs", unit="ep", mininterval=5)
    for epoch in epoch_bar:
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        bar = tqdm(train_dl, desc=f"train {epoch + 1}/{tc['epochs']}", total=steps, leave=False, mininterval=2)
        for i, (frames, y) in enumerate(bar):
            if tc["max_steps"] and i >= tc["max_steps"]:
                break
            set_lr(step)
            frames, y = frames.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x = gpu_train_transform(frames, size)
            target = F.one_hot(y, nc).float() * (1 - tc["label_smoothing"]) + tc["label_smoothing"] / nc
            if tc["mixup"] > 0 and np.random.rand() < tc["mixup_prob"]:
                x, target = mixup(x, target, tc["mixup"])
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
        eval_model = ema.module if ema is not None else model
        m = evaluate(eval_model, val_dl, size, 1, nc, f"val {epoch + 1}")
        history.append(dict(epoch=epoch + 1, train_loss=loss_sum / seen, train_acc=correct / seen,
                            clips_per_s=seen / (time.time() - t0),
                            **{k: v for k, v in m.items() if k not in ("confusion", "per_class")}))
        for k in ("acc", "macro_f1", "cheat_auroc", "cheat_recall", "cheat_precision"):
            tb.add_scalar(f"val/{k}", m[k], epoch + 1)
        save_checkpoint(os.path.join(run, "last.pt"), eval_model, classes, cfg, epoch + 1, m)
        tag = ""
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            shutil.copy(os.path.join(run, "last.pt"), os.path.join(run, "best.pt"))
            tag = "  *best*"
        tqdm.write(f"ep {epoch + 1:3d} | loss {loss_sum / seen:.3f} acc {correct / seen:.3f} | val acc {m['acc']:.3f} "
                   f"macroF1 {m['macro_f1']:.3f} cheat AUROC {m['cheat_auroc']:.3f} R {m['cheat_recall']:.3f} "
                   f"P {m['cheat_precision']:.3f}{tag}")
        epoch_bar.set_postfix(best_f1=f"{best:.3f}")
        json.dump(history, open(os.path.join(run, "history.json"), "w"), indent=1)

    final = ema.module if ema is not None else model
    final.load_state_dict(torch.load(os.path.join(run, "best.pt"), weights_only=False)["model"])
    views = tc["test_views"]
    test_dl = DataLoader(ClipDataset(clips, val_rows, frames_n, mc["stride"], train=False, views=views),
                         batch_size=max(1, tc["batch_size"] // views), **{**dl_kw, "persistent_workers": False})
    m = evaluate(final, test_dl, size, views, nc, "final test")
    json.dump(m, open(os.path.join(run, "final_val_multiview.json"), "w"), indent=1)
    print(f"\nbest epoch macroF1 {best:.3f} | {views}-view test:")
    print_report(m, classes)
    print(f"checkpoint: {os.path.join(run, 'best.pt')}  (copy it to {cfg['paths']['weights']} to use it by default)")
    tb.close()


if __name__ == "__main__":
    main()
