#!/usr/bin/env python3
"""Measure an outcome-neutral pair of revision-pinned GGUF artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_measurement_module():
    source = ROOT / "scripts/measure-gguf-imatrix.py"
    spec = importlib.util.spec_from_file_location("gguf_primary_measurement", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load primary GGUF measurement module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    artifact_index_path = experiment / "artifacts.json"
    artifact_index_bytes = artifact_index_path.read_bytes()
    artifact_index = json.loads(artifact_index_bytes)
    artifacts = artifact_index["artifacts"]
    if len(artifacts) != 2:
        raise ValueError("paired replication requires exactly two artifacts")
    primary = load_measurement_module()
    left = primary.measure(args.left, artifacts[0], False)
    right = primary.measure(args.right, artifacts[1], False)
    comparison = primary.compare(left, right)
    if comparison["assignmentHashesDiffer"]:
        comparison["observation"] = "The two pinned Q4_K_M artifacts have different serialized tensor-to-encoding assignments."
    else:
        comparison["observation"] = "The two pinned Q4_K_M artifacts have the same serialized tensor-to-encoding assignment under this hash basis."
    comparison["interpretationBoundary"] = (
        "This pair-specific result neither proves nor disproves that the Q4_K_M label uniquely determines assignments in other releases."
    )
    results = experiment / "results"
    write(results / "left-tensor-inventory.json", left)
    write(results / "right-tensor-inventory.json", right)
    write(results / "tensor-assignment-comparison.json", comparison)
    run = {
        "recordType": "tensor-quantization-metadata-study.gguf-paired-replication-run.v1",
        "artifactIndexSha256": hashlib.sha256(artifact_index_bytes).hexdigest(),
        "artifactIds": [item["id"] for item in artifacts],
        "primaryReader": left["measurementTool"],
        "resultFiles": ["left-tensor-inventory.json", "right-tensor-inventory.json", "tensor-assignment-comparison.json"],
    }
    write(results / "measurement-run.json", run)


if __name__ == "__main__":
    main()
