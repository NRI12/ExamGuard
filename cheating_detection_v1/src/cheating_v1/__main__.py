"""Command line entry point: python -m cheating_v1 <command> [options]

commands:
  download     fetch the raw DOAN1 dataset from Google Drive          (data.download)
  prepare      extract session videos and cut per-person clips      (data.prepare)
  train        fine-tune X3D on the clips                            (train)
  evaluate     clip-level multi-view evaluation of a checkpoint      (evaluate)
  infer        run the detector on a video / webcam / stream         (app.infer)
  eval-videos  end-to-end evaluation on full val videos              (app.eval_videos)
  tune         threshold sweep on eval-videos logs                   (app.tune_thresholds)
  calibrate    fit app.box_map from YOLO vs annotated boxes          (app.calibrate_boxes)

Every command accepts --config and --set KEY=VALUE; add -h after a command for its options.
"""
import importlib
import sys

COMMANDS = {
    "download": "data.download", "prepare": "data.prepare", "train": "train", "evaluate": "evaluate",
    "infer": "app.infer", "eval-videos": "app.eval_videos", "tune": "app.tune_thresholds",
    "calibrate": "app.calibrate_boxes",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    module = importlib.import_module(f"{__package__}.{COMMANDS[sys.argv[1]]}")
    module.main(sys.argv[2:])


if __name__ == "__main__":
    main()
