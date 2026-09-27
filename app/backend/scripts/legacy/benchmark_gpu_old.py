"""Benchmark batch inference E3 trên CUDA, không dùng dữ liệu final test."""

import argparse
import json
import time
from pathlib import Path

import torch

from app.ai.predictor import ExamBehaviorPredictor
from app.ai.runtime import runtime_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CUDA batch inference E3")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def benchmark_batch(
    predictor: ExamBehaviorPredictor, batch_size: int, iterations: int
) -> dict[str, float | int | str]:
    shape = (batch_size, 8, 3, 224, 224)
    try:
        clips = torch.zeros(shape, dtype=torch.float32)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            predictor.predict_batch(clips)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - started) / iterations
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return {
            "batch_size": batch_size,
            "status": "ok",
            "latency_ms": latency * 1000,
            "clips_per_second": batch_size / latency,
            "peak_vram_mb": peak_vram,
        }
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.OutOfMemoryError):
            raise
        return {"batch_size": batch_size, "status": "oom"}
    finally:
        if "clips" in locals():
            del clips
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("Số vòng benchmark phải lớn hơn 0")
    predictor = ExamBehaviorPredictor(
        args.artifact_dir,
        device="cuda",
        require_cuda=True,
    )
    warmup = torch.zeros((1, 8, 3, 224, 224), dtype=torch.float32)
    for _ in range(3):
        predictor.predict_batch(warmup)
    torch.cuda.synchronize()

    diagnostics = runtime_info(device="cuda", require_cuda=True, model_loaded=True)
    report = {
        "runtime": diagnostics.as_dict(),
        "iterations": args.iterations,
        "results": [
            benchmark_batch(predictor, batch_size, args.iterations)
            for batch_size in (1, 4, 8, 16, 32)
        ],
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
