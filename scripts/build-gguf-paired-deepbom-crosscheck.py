#!/usr/bin/env python3
"""Run and compare an isolated secondary implementation for a GGUF pair."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def run_secondary(node: str, cli: Path, artifact: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [node, str(cli), "gguf", str(artifact), "--tensors", "--compact"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.stderr.strip():
        raise RuntimeError(f"secondary implementation emitted stderr: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def secondary_version(node: str, cli: Path) -> str:
    completed = subprocess.run([node, str(cli), "--version"], check=True, capture_output=True, text=True, encoding="utf-8")
    if completed.stderr.strip():
        raise RuntimeError(f"secondary version command emitted stderr: {completed.stderr.strip()}")
    return completed.stdout.strip()


def compare(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, bool]:
    expected = [{
        "index": row["index"],
        "name": row["name"],
        "encoding": row["encoding"],
        "shape": row["shape"],
        "byte_length": row["byteLength"],
    } for row in primary["tensors"]]
    observed = [{
        "index": row["index"],
        "name": row["name"],
        "encoding": row["encoding"],
        "shape": row["shape"],
        "byte_length": row["byte_length"],
    } for row in secondary["tensors"]]
    return {
        "artifact_sha256": primary["source"]["sha256"] == secondary["artifact"]["sha256"],
        "artifact_byte_length": primary["source"]["bytes"] == secondary["artifact"]["byte_length"],
        "tensor_count": primary["summary"]["tensorCount"] == secondary["tensor_count"],
        "normalized_tensor_rows": expected == observed,
        "descriptor_projection_sha256": (
            primary["summary"]["serializedDescriptorProjectionSha256"]
            == secondary["tensor_encoding_assignment_sha256"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--deepbom-cli", type=Path, required=True)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    crosscheck = experiment / "crosschecks/deepbom"
    primary_left = load(experiment / "results/left-tensor-inventory.json")
    primary_right = load(experiment / "results/right-tensor-inventory.json")
    secondary_left = run_secondary(args.node, args.deepbom_cli.resolve(), args.left.resolve())
    secondary_right = run_secondary(args.node, args.deepbom_cli.resolve(), args.right.resolve())
    write(crosscheck / "raw/left-tensor-table.json", secondary_left)
    write(crosscheck / "raw/right-tensor-table.json", secondary_right)
    checks = {
        "left": compare(primary_left, secondary_left),
        "right": compare(primary_right, secondary_right),
        "pair_assignment_relation": (
            (primary_left["summary"]["serializedDescriptorProjectionSha256"] == primary_right["summary"]["serializedDescriptorProjectionSha256"])
            == (secondary_left["tensor_encoding_assignment_sha256"] == secondary_right["tensor_encoding_assignment_sha256"])
        ),
    }
    flat = list(checks["left"].values()) + list(checks["right"].values()) + [checks["pair_assignment_relation"]]
    write(crosscheck / "normalized-comparison.json", {
        "recordType": "tensor-quantization-metadata-study.gguf-secondary-implementation-comparison.v1",
        "primaryImplementation": "upstream gguf-py 0.19.0",
        "secondaryImplementation": secondary_version(args.node, args.deepbom_cli.resolve()),
        "secondaryMaintainerRelationship": "maintained by the study author",
        "checks": checks,
        "allChecksPass": all(flat),
        "interpretationBoundary": "The secondary implementation tests reproducibility only; no study conclusion depends solely on it.",
    })
    if not all(flat):
        raise SystemExit("secondary implementation comparison failed")


if __name__ == "__main__":
    main()
