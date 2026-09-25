"""Clip-level evaluation of a checkpoint (RGB, flow or fusion) on the val split (multi-view test).

  python -m cheating_v3 evaluate                                        # weights/fusion_best.pt
  python -m cheating_v3 evaluate --checkpoint weights/rgb_x3d_l_best.pt  # RGB stream alone (= v1)
  python -m cheating_v3 evaluate --checkpoint weights/flow_x3d_s_best.pt # flow stream alone
"""
import json
import os

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import config_parser, load_config
from .data.dataset import (ClipDataset, gpu_eval_transform, gpu_flow_eval_transform, gpu_flow_train_transform,
                           gpu_train_transform, read_split, worker_init)
from .models.registry import forward, load_model, uses_flow, uses_rgb


def make_dataset(cfg, info, rows, train=False, views=1):
    return ClipDataset(cfg["paths"]["clips"], rows, info["frames"], info["stride"], train=train, views=views,
                       rgb=uses_rgb(info), flow_dir=cfg["paths"]["flow"] if uses_flow(info) else None)


def to_inputs(info, rgb, flow, train):
    """uint8 batches from the loader -> normalised GPU tensors (None for an unused stream)."""
    x_rgb = x_flow = None
    if uses_rgb(info):
        rgb = rgb.cuda(non_blocking=True)
        x_rgb = gpu_train_transform(rgb, info["rgb_size"]) if train else gpu_eval_transform(rgb, info["rgb_size"])
    if uses_flow(info):
        flow = flow.cuda(non_blocking=True)
        x_flow = (gpu_flow_train_transform(flow, info["flow_size"]) if train
                  else gpu_flow_eval_transform(flow, info["flow_size"]))
    return x_rgb, x_flow


@torch.no_grad()
def evaluate(model, info, loader, views, num_classes, desc="eval"):
    """Class 0 is 'normal'; cheat_score = 1 - P(normal) gives the cheating-vs-normal metrics."""
    model.eval()
    probs, labels = [], []
    for rgb, flow, y in tqdm(loader, desc=desc, leave=False, mininterval=2):
        b = y.size(0)
        split = lambda t: t.view(b * views, t.size(1) // views, *t.shape[2:]) if t.numel() else t
        x_rgb, x_flow = to_inputs(info, split(rgb), split(flow), train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward(model, info, x_rgb, x_flow).float()
        probs.append(logits.softmax(-1).view(b, views, -1).mean(1).cpu())
        labels.append(y)
    p, y = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    pred = p.argmax(1)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=range(num_classes), zero_division=0)
    cheat_true, cheat_pred = y != 0, pred != 0
    tp = (cheat_true & cheat_pred).sum()
    return dict(
        acc=float((pred == y).mean()), macro_f1=float(f1.mean()),
        per_class=dict(precision=prec.tolist(), recall=rec.tolist(), f1=f1.tolist(), support=sup.tolist()),
        cheat_auroc=float(roc_auc_score(cheat_true, 1 - p[:, 0])) if 0 < cheat_true.sum() < len(y) else float("nan"),
        cheat_precision=float(tp / max(1, cheat_pred.sum())), cheat_recall=float(tp / max(1, cheat_true.sum())),
        confusion=confusion_matrix(y, pred, labels=range(num_classes)).tolist())


def print_report(m, classes):
    print(f"acc {m['acc']:.3f}  macroF1 {m['macro_f1']:.3f}  cheat AUROC {m['cheat_auroc']:.3f}  "
          f"recall {m['cheat_recall']:.3f}  precision {m['cheat_precision']:.3f}")
    print("F1    :", dict(zip(classes, [round(f, 3) for f in m["per_class"]["f1"]])))
    print("confusion (rows = true):")
    for c, r in zip(classes, m["confusion"]):
        print(f"  {c:17s} {r}")


def multiview_loader(cfg, info, rows, views):
    return DataLoader(make_dataset(cfg, info, rows, views=views), batch_size=max(1, cfg["train"]["batch_size"] // views),
                      num_workers=cfg["train"]["workers"], pin_memory=True, worker_init_fn=worker_init)


def main(argv=None):
    p = config_parser(__doc__.splitlines()[0])
    p.add_argument("--checkpoint", help="default: paths.weights")
    p.add_argument("--views", type=int, help="default: train.test_views")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    ckpt = args.checkpoint or cfg["paths"]["weights"]
    views = args.views or cfg["train"]["test_views"]
    model, ck, info = load_model(ckpt, cfg["data"])
    rows = read_split(cfg["paths"]["clips"], "val", cfg["train"]["drop_related"])
    m = evaluate(model, info, multiview_loader(cfg, info, rows, views), views, len(ck["classes"]), f"val x{views}")
    print(f"{os.path.basename(ckpt)} ({info['arch']}):")
    print_report(m, ck["classes"])
    out = os.path.join(cfg["paths"]["outputs"], "eval")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{os.path.splitext(os.path.basename(ckpt))[0]}_val_{views}view.json")
    json.dump(m, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
