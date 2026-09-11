#!/usr/bin/env python3
"""Acquire hash-pinned GGUF subjects into the ignored content-addressed cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, artifact: dict) -> None:
    if path.stat().st_size != artifact["bytes"]:
        raise ValueError(f"{artifact['id']}: byte length mismatch")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"{artifact['id']}: SHA-256 mismatch")


def acquire(cache_root: Path, artifact: dict) -> Path:
    destination = cache_root / artifact["sha256"] / artifact["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        verify(destination, artifact)
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        artifact["downloadUrl"],
        headers={"User-Agent": "tensor-quantization-metadata-study/1.1"},
    )
    digest = hashlib.sha256()
    byte_count = 0
    with urllib.request.urlopen(request) as response, partial.open("wb") as output:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
    if byte_count != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
        raise ValueError(f"{artifact['id']}: downloaded identity mismatch; partial file retained")
    os.replace(partial, destination)
    return destination


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-index",
        type=Path,
        default=root / "experiments/gguf-imatrix/artifacts.json",
    )
    parser.add_argument("--cache-root", type=Path, default=root / "cache/gguf-imatrix")
    parser.add_argument("--id", action="append", dest="ids")
    args = parser.parse_args()
    index = json.loads(args.artifact_index.read_text(encoding="utf-8"))
    selected = [
        item for item in index["artifacts"]
        if not args.ids or item["id"] in args.ids
    ]
    unknown = set(args.ids or []) - {item["id"] for item in index["artifacts"]}
    if unknown:
        raise ValueError(f"Unknown artifact ids: {sorted(unknown)}")
    for artifact in selected:
        print(acquire(args.cache_root, artifact).relative_to(root))


if __name__ == "__main__":
    main()
