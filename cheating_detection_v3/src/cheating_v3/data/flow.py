"""Pre-compute optical flow for every clip of paths.clips (RAFT-small on GPU) into paths.flow.

For clip videos/<split>/<class>/<name>.mp4 writes <name>.npz under paths.flow with
  flow: uint8 [clip_len, flow_size, flow_size, 2]; flow[f] = flow from frame f - stride to frame f
        (frames < stride hold zero motion), computed at data.flow_compute_size px then resized.

  python -m cheating_v3 flow            # ~10 min on an RTX 3090 for the 1735 clips
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config import config_parser, load_config
from ..models.raft_flow import FlowEstimator, quantize, resize_flow


def flow_path(flow_dir, clip_path):
    return os.path.join(flow_dir, os.path.splitext(clip_path)[0] + ".npz")


class Frames(Dataset):
    def __init__(self, clips_dir, paths):
        self.clips_dir, self.paths = clips_dir, paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        cap, frames = cv2.VideoCapture(os.path.join(self.clips_dir, self.paths[i])), []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cap.release()
        return torch.from_numpy(np.stack(frames)), i


def main(argv=None):
    args = config_parser(__doc__.splitlines()[0]).parse_args(argv)
    cfg = load_config(args.config, args.set)
    clips, out = cfg["paths"]["clips"], cfg["paths"]["flow"]
    stride, size, work = cfg["model"]["stride"], cfg["model"]["flow_size"], cfg["data"]["flow_compute_size"]
    paths = [line.split()[0] for split in ("train", "val") for line in open(os.path.join(clips, f"{split}.csv"))]
    todo = [p for p in paths if not os.path.exists(flow_path(out, p))]
    print(f"{len(paths)} clips, {len(todo)} without flow -> {out}")
    est = FlowEstimator()
    dl = DataLoader(Frames(clips, todo), batch_size=1, num_workers=cfg["train"]["workers"])
    for frames, i in tqdm(dl, desc="flow", unit="clip", mininterval=5):
        x = frames[0].cuda().permute(0, 3, 1, 2).float()  # T, 3, H, W
        x = F.interpolate(x, size=(work, work), mode="bilinear", align_corners=False, antialias=True)
        n = x.shape[0]
        flows = []
        for s in range(stride, n, 48):
            idx = torch.arange(s, min(n, s + 48), device=x.device)
            flows.append(resize_flow(est(x[idx - stride], x[idx]), size))
        q = torch.full((n, 2, size, size), 128, dtype=torch.uint8, device=x.device)
        q[stride:] = quantize(torch.cat(flows))
        dst = flow_path(out, todo[i.item()])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst, flow=q.permute(0, 2, 3, 1).cpu().numpy())


if __name__ == "__main__":
    main()
