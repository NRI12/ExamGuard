"""Download the pinned YOLO11n artifact and verify it before publishing atomically."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
MODEL_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == MODEL_SHA256:
        print(f"YOLO11n đã sẵn sàng: {destination}")
        return
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix="yolo11n-",
        suffix=".partial",
        delete=False,
    ) as temporary_file:
        temporary = Path(temporary_file.name)
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=300) as response:  # noqa: S310
                shutil.copyfileobj(response, temporary_file)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    if sha256(temporary) != MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checksum YOLO11n tải về không hợp lệ")
    temporary.replace(destination)
    print(f"Đã tải YOLO11n: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    download(arguments.destination.resolve())


if __name__ == "__main__":
    main()
