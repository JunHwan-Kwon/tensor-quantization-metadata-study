#!/usr/bin/env python3
"""Verify a committed GGUF paired-replication package without model bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def assignment_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["name"].encode("utf-8")):
        digest.update(row["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(row["encoding"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_inventory(document: dict[str, Any], artifact: dict[str, Any]) -> None:
    rows = document["tensors"]
    summary = document["summary"]
    if document["measurementTool"]["reader"] != "gguf-py" or document["measurementTool"]["ggufVersion"] != "0.19.0":
        raise ValueError("primary reader is not pinned gguf-py 0.19.0")
    if document["source"]["sha256"] != artifact["sha256"] or document["source"]["bytes"] != artifact["bytes"]:
        raise ValueError("inventory is not bound to the artifact index")
    if len(rows) != summary["tensorCount"] or sum(row["elementCount"] for row in rows) != summary["elementCount"]:
        raise ValueError("tensor or element conservation failed")
    if sum(row["byteLength"] for row in rows) != summary["serializedTensorBytes"]:
        raise ValueError("serialized tensor byte conservation failed")
    projection = [{"index": row["index"], "name": row["name"], "dtype": row["encoding"],
                   "shape": row["shape"], "byte_length": row["byteLength"]} for row in rows]
    if canonical_sha(projection) != summary["serializedDescriptorProjectionSha256"]:
        raise ValueError("descriptor projection hash mismatch")
    if assignment_sha(rows) != summary["nameEncodingAssignmentSha256"]:
        raise ValueError("name/encoding assignment hash mismatch")
    counts = Counter(row["encoding"] for row in rows)
    if counts != Counter({row["encoding"]: row["tensorCount"] for row in summary["encodingInventory"]}):
        raise ValueError("encoding inventory conservation failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if list(experiment.rglob("*.gguf")):
        raise ValueError("GGUF artifact bytes must not be stored in the evidence directory")
    index = load(experiment / "artifacts.json")
    left = load(experiment / "results/left-tensor-inventory.json")
    right = load(experiment / "results/right-tensor-inventory.json")
    comparison = load(experiment / "results/tensor-assignment-comparison.json")
    verify_inventory(left, index["artifacts"][0])
    verify_inventory(right, index["artifacts"][1])
    left_by_name = {row["name"]: row for row in left["tensors"]}
    right_by_name = {row["name"]: row for row in right["tensors"]}
    names = set(left_by_name) | set(right_by_name)
    changed = sum(1 for name in names if name in left_by_name and name in right_by_name
                  and left_by_name[name]["encoding"] != right_by_name[name]["encoding"])
    if changed != comparison["changedEncodingCount"]:
        raise ValueError("changed-encoding count cannot be reconstructed")
    hashes_differ = left["summary"]["nameEncodingAssignmentSha256"] != right["summary"]["nameEncodingAssignmentSha256"]
    if hashes_differ != comparison["assignmentHashesDiffer"]:
        raise ValueError("assignment hash comparison is inconsistent")
    for path in list((experiment / "results").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        if "deepbom" in text.lower():
            raise ValueError(f"primary result is tool-branded: {path.name}")
        if re.search(r"[A-Za-z]:\\\\", text):
            raise ValueError(f"primary result contains an absolute Windows path: {path.name}")
    crosscheck_path = experiment / "crosschecks/deepbom/normalized-comparison.json"
    if crosscheck_path.exists() and not load(crosscheck_path).get("allChecksPass"):
        raise ValueError("secondary implementation comparison contains a mismatch")
    print(json.dumps({
        "status": "pass",
        "subjectCount": 2,
        "tensorCounts": [left["summary"]["tensorCount"], right["summary"]["tensorCount"]],
        "sameTensorNameSet": comparison["sameTensorNameSet"],
        "shapeMismatchCount": comparison["shapeMismatchCount"],
        "changedEncodingCount": changed,
        "assignmentHashesDiffer": hashes_differ,
    }, indent=2))


if __name__ == "__main__":
    main()
