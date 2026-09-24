"""Ver 2 dataset: same clips as clips_hr, labels collapsed to 0 normal / 1 abnormal.

looking, interaction, phone_cheatsheet -> abnormal; normal (incl. hard negatives) stays normal.
Videos are shared through a symlink, so this only writes the label files.

  python make_v2_binary.py --src /workspace/data/clips_hr --dst /workspace/data/clips_hr_v2
"""
import argparse
import csv
import os
import shutil
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/data/clips_hr")
    ap.add_argument("--dst", default="/workspace/data/clips_hr_v2")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    link = os.path.join(args.dst, "videos")
    if not os.path.exists(link):
        os.symlink(os.path.join(os.path.abspath(args.src), "videos"), link)
    shutil.copy(os.path.join(args.src, "clips_info.json"), args.dst)
    with open(os.path.join(args.dst, "classes.txt"), "w") as f:
        f.write("0 normal\n1 abnormal\n")
    for split in ("train", "val"):
        rows = [line.split() for line in open(os.path.join(args.src, f"{split}.csv"))]
        with open(os.path.join(args.dst, f"{split}.csv"), "w") as f:
            f.writelines(f"{p} {int(int(l) > 0)}\n" for p, l in rows)
        print(split, dict(Counter("abnormal" if int(l) else "normal" for _, l in rows)))
    with open(os.path.join(args.src, "meta.csv")) as fi, open(os.path.join(args.dst, "meta.csv"), "w", newline="") as fo:
        rd = csv.DictReader(fi)
        wr = csv.DictWriter(fo, fieldnames=rd.fieldnames + ["cls_v1"])
        wr.writeheader()
        for r in rd:
            r["cls_v1"] = r["cls"]
            r["label"] = int(int(r["label"]) > 0)
            r["cls"] = "abnormal" if r["label"] else "normal"
            wr.writerow(r)


if __name__ == "__main__":
    main()
