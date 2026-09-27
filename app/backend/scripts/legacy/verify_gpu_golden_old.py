"""Xác minh golden regression E3 bằng single và GPU batch inference."""

import argparse
import json
from pathlib import Path

import torch

from app.ai.predictor import ExamBehaviorPredictor
from app.ai.preprocessing import preprocess_video
from app.ai.runtime import runtime_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xác minh golden regression E3 trên CUDA")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tolerance <= 0:
        raise ValueError("Tolerance phải lớn hơn 0")

    golden_samples = json.loads(args.golden.read_text(encoding="utf-8"))["samples"]
    predictor = ExamBehaviorPredictor(
        args.artifact_dir,
        device="cuda",
        require_cuda=True,
    )
    preprocessing = predictor.config.preprocessing
    clips = []
    singles = []
    for golden in golden_samples:
        sample_path = args.sample_dir / golden["file"]
        prediction = predictor.predict(sample_path)
        if prediction.class_idx != golden["class_idx"]:
            raise AssertionError(
                f"{golden['file']}: class={prediction.class_idx}, "
                f"golden={golden['class_idx']}"
            )
        expected_probabilities = list(golden["probabilities"].values())
        actual_probabilities = list(prediction.probabilities.values())
        max_golden_diff = max(
            abs(actual - expected)
            for actual, expected in zip(
                actual_probabilities, expected_probabilities, strict=True
            )
        )
        if max_golden_diff > args.tolerance:
            raise AssertionError(
                f"{golden['file']}: sai khác golden {max_golden_diff:.8f} "
                f"vượt tolerance {args.tolerance}"
            )
        clip, _ = preprocess_video(
            sample_path,
            num_segments=preprocessing.num_segments,
            input_size=preprocessing.input_size,
            mean=preprocessing.mean,
            std=preprocessing.std,
        )
        clips.append(clip)
        singles.append(actual_probabilities)

    batch_probabilities = predictor.predict_batch(torch.stack(clips))
    if tuple(batch_probabilities.shape) != (len(golden_samples), 5):
        raise AssertionError(f"Batch output shape không hợp lệ: {batch_probabilities.shape}")

    results = []
    for index, golden in enumerate(golden_samples):
        batch_values = batch_probabilities[index].tolist()
        batch_class = int(batch_probabilities[index].argmax().item())
        max_single_batch_diff = max(
            abs(single - batch)
            for single, batch in zip(singles[index], batch_values, strict=True)
        )
        if batch_class != golden["class_idx"]:
            raise AssertionError(
                f"{golden['file']}: batch class={batch_class}, "
                f"golden={golden['class_idx']}"
            )
        if max_single_batch_diff > args.tolerance:
            raise AssertionError(
                f"{golden['file']}: single/batch sai khác {max_single_batch_diff:.8f}"
            )
        results.append(
            {
                "file": golden["file"],
                "predicted_class": predictor.config.classes[batch_class],
                "confidence": batch_values[batch_class],
                "max_single_batch_diff": max_single_batch_diff,
            }
        )

    report = {
        "runtime": runtime_info(
            device="cuda", require_cuda=True, model_loaded=True
        ).as_dict(),
        "input_shape": list(torch.stack(clips).shape),
        "output_shape": list(batch_probabilities.shape),
        "tolerance": args.tolerance,
        "samples": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
