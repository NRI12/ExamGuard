"""Profile the frozen R3 checkpoint on real ROI frames without changing production math."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from app.ai.action_recognition.adapter import R3_CLASS_NAMES, _safe_load
from app.ai.action_recognition.model import KineticsTSMClassifier
from app.ai.action_recognition.preprocessing import preprocess_rgb_clips, sample_segment_indices


def _percentile(values: list[float], point: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), point))


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
    }


def _decode_real_clip(path: Path) -> tuple[np.ndarray, ...]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    indices = sample_segment_indices(len(frames), 8)
    return tuple(frames[int(index)] for index in indices)


def _load_model(checkpoint_path: Path, device: torch.device, precision: str) -> torch.nn.Module:
    checkpoint = _safe_load(checkpoint_path)
    model_config = checkpoint["config"]["model"]
    model = KineticsTSMClassifier(
        num_classes=len(R3_CLASS_NAMES),
        num_segments=8,
        fold_div=8,
        dropout=float(model_config.get("dropout", 0.6)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    if precision == "fp16":
        model.half()
    model.eval()
    return model


def _run_batch(
    *,
    model: torch.nn.Module,
    frames: tuple[np.ndarray, ...],
    batch_size: int,
    precision: str,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], list[float], list[float]]:
    preprocess_values: list[float] = []
    h2d_values: list[float] = []
    forward_values: list[float] = []
    postprocess_values: list[float] = []
    total_values: list[float] = []
    last_logits: torch.Tensor | None = None
    last_probabilities: torch.Tensor | None = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for iteration in range(warmup + iterations):
        total_started = time.perf_counter()
        started = time.perf_counter()
        cpu_tensor = preprocess_rgb_clips([frames] * batch_size)
        preprocess_ms = (time.perf_counter() - started) * 1000

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        tensor = cpu_tensor.to(device)
        if precision == "fp16":
            tensor = tensor.half()
        torch.cuda.synchronize(device)
        h2d_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(tensor)
        torch.cuda.synchronize(device)
        forward_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        probabilities = torch.softmax(logits, dim=1).float().cpu()
        logits_cpu = logits.float().cpu()
        torch.cuda.synchronize(device)
        postprocess_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        if iteration >= warmup:
            preprocess_values.append(preprocess_ms)
            h2d_values.append(h2d_ms)
            forward_values.append(forward_ms)
            postprocess_values.append(postprocess_ms)
            total_values.append(total_ms)
        last_logits = logits_cpu[0]
        last_probabilities = probabilities[0]
    assert last_logits is not None and last_probabilities is not None
    mean_total_ms = statistics.fmean(total_values)
    result = {
        "precision": precision,
        "batch_size": batch_size,
        "preprocess_ms": _summary(preprocess_values),
        "h2d_ms": _summary(h2d_values),
        "forward_ms": _summary(forward_values),
        "postprocess_ms": _summary(postprocess_values),
        "total_ms": _summary(total_values),
        "clips_per_second": batch_size / (mean_total_ms / 1000),
        "vram_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "vram_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "vram_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "top1": R3_CLASS_NAMES[int(last_probabilities.argmax())],
        "logits_finite": bool(torch.isfinite(last_logits).all()),
        "probabilities_finite": bool(torch.isfinite(last_probabilities).all()),
        "logits": [
            float(value) if np.isfinite(value) else None for value in last_logits.tolist()
        ],
        "probabilities": [
            float(value) if np.isfinite(value) else None
            for value in last_probabilities.tolist()
        ],
    }
    return result, last_logits.tolist(), last_probabilities.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or any(value <= 0 for value in args.batch_sizes):
        parser.error("iterations/batch sizes must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the action model profiler")
    device = torch.device("cuda:0")
    frames = _decode_real_clip(args.clip)
    results: list[dict[str, Any]] = []
    comparison: dict[int, dict[str, tuple[list[float], list[float]]]] = {}
    for precision in ("fp32", "fp16"):
        model = _load_model(args.checkpoint, device, precision)
        for batch_size in args.batch_sizes:
            try:
                result, logits, probabilities = _run_batch(
                    model=model,
                    frames=frames,
                    batch_size=batch_size,
                    precision=precision,
                    device=device,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                results.append(
                    {"precision": precision, "batch_size": batch_size, "status": "oom"}
                )
                continue
            result["status"] = "ok"
            results.append(result)
            comparison.setdefault(batch_size, {})[precision] = (logits, probabilities)
        del model
        torch.cuda.empty_cache()
    numeric = []
    for batch_size, rows in sorted(comparison.items()):
        if "fp32" not in rows or "fp16" not in rows:
            continue
        fp32_logits, fp32_probabilities = rows["fp32"]
        fp16_logits, fp16_probabilities = rows["fp16"]
        finite = bool(
            np.isfinite(fp32_logits).all()
            and np.isfinite(fp32_probabilities).all()
            and np.isfinite(fp16_logits).all()
            and np.isfinite(fp16_probabilities).all()
        )
        numeric.append(
            {
                "batch_size": batch_size,
                "numerically_comparable": finite,
                "max_absolute_logit_difference": (
                    max(
                        abs(left - right)
                        for left, right in zip(fp32_logits, fp16_logits, strict=True)
                    )
                    if finite
                    else None
                ),
                "max_probability_difference": (
                    max(
                        abs(left - right)
                        for left, right in zip(
                            fp32_probabilities, fp16_probabilities, strict=True
                        )
                    )
                    if finite
                    else None
                ),
                "top1_agreement": (
                    int(np.argmax(fp32_probabilities))
                    == int(np.argmax(fp16_probabilities))
                    if finite
                    else None
                ),
            }
        )
    output = {
        "clip": str(args.clip),
        "checkpoint": str(args.checkpoint),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": results,
        "fp32_fp16": numeric,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.quiet:
        print(json.dumps({"output": str(args.output), "results": len(results)}))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
