#!/usr/bin/env python3
"""Measure revision-pinned GGUF artifacts with the upstream gguf-py reader."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    return value


def dtype_name(tensor_type: Any) -> str:
    return str(getattr(tensor_type, "name", tensor_type))


def name_encoding_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["name"].encode("utf-8")):
        name = row["name"].encode("utf-8")
        encoding = row["encoding"].encode("utf-8")
        if b"\x00" in name:
            raise ValueError("Tensor name contains NUL and cannot use this hash basis")
        if b"\n" in encoding:
            raise ValueError("Tensor encoding contains LF and cannot use this hash basis")
        digest.update(name)
        digest.update(b"\x00")
        digest.update(encoding)
        digest.update(b"\n")
    return digest.hexdigest()


def relevant_metadata(reader: GGUFReader) -> tuple[list[str], dict[str, Any]]:
    keys = sorted(name for name in reader.fields if not name.startswith("GGUF."))
    selected = {}
    prefixes = (
        "general.name",
        "general.version",
        "general.description",
        "general.author",
        "general.organization",
        "general.license",
        "general.file_type",
        "general.base_model",
        "general.datasets",
        "general.tags",
        "quantize.imatrix",
    )
    for name in keys:
        if not name.startswith(prefixes):
            continue
        try:
            selected[name] = plain(reader.fields[name].contents())
        except Exception as exc:
            selected[name] = {"decodeError": f"{type(exc).__name__}: {exc}"}
    return keys, selected


def scan_f32(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(flat)
    finite_values = flat[finite_mask]
    exact_zero_count = int(np.count_nonzero(flat == 0))
    return {
        "valueCount": int(flat.size),
        "nonfiniteValueCount": int(flat.size - np.count_nonzero(finite_mask)),
        "exactZeroValueCount": exact_zero_count,
        "allZero": bool(flat.size and exact_zero_count == flat.size),
        "constant": bool(flat.size and np.all(flat == flat[0])),
        "finiteMinimum": float(np.min(finite_values)) if finite_values.size else None,
        "finiteMaximum": float(np.max(finite_values)) if finite_values.size else None,
    }


def measure(path: Path, artifact: dict[str, Any], scan_values: bool) -> dict[str, Any]:
    path = path.resolve()
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != artifact["bytes"]:
        raise ValueError(f"{artifact['id']}: byte length mismatch")
    if actual_sha256 != artifact["sha256"]:
        raise ValueError(f"{artifact['id']}: SHA-256 mismatch")

    reader = GGUFReader(str(path), "r")
    metadata_keys, metadata = relevant_metadata(reader)
    tensors = []
    encoding_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensorCount": 0, "elementCount": 0, "byteLength": 0}
    )
    layer_encodings: dict[int, Counter[str]] = defaultdict(Counter)
    numerical_rows = []
    duplicate_names = Counter()

    for index, tensor in enumerate(reader.tensors):
        name = str(tensor.name)
        encoding = dtype_name(tensor.tensor_type)
        row = {
            "index": index,
            "name": name,
            "encoding": encoding,
            "shape": [int(item) for item in tensor.shape.tolist()],
            "elementCount": int(tensor.n_elements),
            "byteLength": int(tensor.n_bytes),
            "dataOffset": int(tensor.data_offset),
        }
        tensors.append(row)
        duplicate_names[name] += 1
        totals = encoding_totals[encoding]
        totals["tensorCount"] += 1
        totals["elementCount"] += row["elementCount"]
        totals["byteLength"] += row["byteLength"]
        layer = re.match(r"^blk\.([0-9]+)\.", name)
        if layer:
            layer_encodings[int(layer.group(1))][encoding] += 1
        if scan_values:
            if encoding != "F32":
                numerical_rows.append({
                    "index": index,
                    "name": name,
                    "encoding": encoding,
                    "status": "not-assessed-non-f32",
                })
            else:
                numerical_rows.append({
                    "index": index,
                    "name": name,
                    "encoding": encoding,
                    "status": "assessed",
                    **scan_f32(tensor.data),
                })

    descriptor_projection = [
        {
            "index": row["index"],
            "name": row["name"],
            "dtype": row["encoding"],
            "shape": row["shape"],
            "byte_length": row["byteLength"],
        }
        for row in tensors
    ]
    total_elements = sum(row["elementCount"] for row in tensors)
    total_bytes = sum(row["byteLength"] for row in tensors)
    inventory = [
        {"encoding": encoding, **encoding_totals[encoding]}
        for encoding in sorted(encoding_totals, key=lambda item: item.encode("utf-8"))
    ]
    summary = {
        "tensorCount": len(tensors),
        "uniqueTensorNameCount": len(duplicate_names),
        "duplicateTensorNames": sorted(
            (name for name, count in duplicate_names.items() if count > 1),
            key=lambda item: item.encode("utf-8"),
        ),
        "elementCount": total_elements,
        "serializedTensorBytes": total_bytes,
        "effectiveStorageBitsPerElement": (
            format(total_bytes * 8 / total_elements, ".6f") if total_elements else None
        ),
        "encodingInventory": inventory,
        "serializedDescriptorProjectionSha256": canonical_sha256(descriptor_projection),
        "nameEncodingAssignmentSha256": name_encoding_sha256(tensors),
    }
    result: dict[str, Any] = {
        "recordType": "tensor-quantization-metadata-study.gguf-tensor-inventory.v1",
        "evidenceClass": "serialized-artifact observation",
        "measurementTool": {
            "reader": "gguf-py",
            "distribution": "gguf",
            "ggufVersion": importlib.metadata.version("gguf"),
            "numpyVersion": np.__version__,
            "pythonVersion": platform.python_version(),
        },
        "source": {
            key: artifact[key]
            for key in ("id", "sourceUri", "revision", "filename", "releaseLabel", "bytes", "sha256")
        },
        "format": {
            "ggufVersion": int(reader.fields["GGUF.version"].contents()),
            "metadataKeys": metadata_keys,
            "selectedMetadata": metadata,
        },
        "hashDefinitions": {
            "serializedDescriptorProjectionSha256": (
                "SHA-256 over compact UTF-8 JSON with sorted object keys; rows remain in "
                "serialized tensor-index order and project index/name/dtype/shape/byte_length."
            ),
            "nameEncodingAssignmentSha256": (
                "SHA-256 over UTF8(name) || NUL || UTF8(encoding) || LF records sorted "
                "by raw UTF-8 name bytes; no Unicode normalization."
            ),
        },
        "summary": summary,
        "layerNameObservation": {
            "pattern": "^blk\\.([0-9]+)\\.",
            "interpretation": "serialized name grouping only; not an architecture-semantic layer assertion",
            "layers": [
                {
                    "index": layer,
                    "tensorCount": sum(layer_encodings[layer].values()),
                    "encodings": dict(sorted(layer_encodings[layer].items())),
                }
                for layer in sorted(layer_encodings)
            ],
        },
        "tensors": tensors,
    }
    if scan_values:
        assessed = [row for row in numerical_rows if row["status"] == "assessed"]
        result["numericalScan"] = {
            "scope": "all serialized F32 tensors",
            "assessedTensorCount": len(assessed),
            "unassessedTensorCount": len(numerical_rows) - len(assessed),
            "decodedValueCount": sum(row["valueCount"] for row in assessed),
            "nonfiniteValueCount": sum(row["nonfiniteValueCount"] for row in assessed),
            "exactZeroValueCount": sum(row["exactZeroValueCount"] for row in assessed),
            "allZeroTensorCount": sum(1 for row in assessed if row["allZero"]),
            "constantTensorCount": sum(1 for row in assessed if row["constant"]),
            "tensors": numerical_rows,
        }
    return result


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_by_name = {row["name"]: row for row in left["tensors"]}
    right_by_name = {row["name"]: row for row in right["tensors"]}
    names = sorted(set(left_by_name) | set(right_by_name), key=lambda item: item.encode("utf-8"))
    transitions: Counter[str] = Counter()
    changed = []
    left_only = []
    right_only = []
    shape_mismatches = []
    for name in names:
        a = left_by_name.get(name)
        b = right_by_name.get(name)
        if a is None:
            right_only.append(name)
            continue
        if b is None:
            left_only.append(name)
            continue
        transitions[f"{a['encoding']}->{b['encoding']}"] += 1
        if a["shape"] != b["shape"]:
            shape_mismatches.append({"name": name, "left": a["shape"], "right": b["shape"]})
        if a["encoding"] != b["encoding"]:
            changed.append({
                "name": name,
                "shape": a["shape"],
                "leftEncoding": a["encoding"],
                "rightEncoding": b["encoding"],
                "leftByteLength": a["byteLength"],
                "rightByteLength": b["byteLength"],
            })
    release_labels = [left["source"]["releaseLabel"], right["source"]["releaseLabel"]]
    return {
        "recordType": "tensor-quantization-metadata-study.gguf-assignment-comparison.v1",
        "evidenceClass": "comparison of serialized tensor descriptors",
        "subjects": [left["source"], right["source"]],
        "sameDeclaredReleaseLabel": len(set(release_labels)) == 1,
        "declaredReleaseLabels": release_labels,
        "sameTensorNameSet": not left_only and not right_only,
        "leftOnlyTensorNames": left_only,
        "rightOnlyTensorNames": right_only,
        "shapeMismatchCount": len(shape_mismatches),
        "shapeMismatches": shape_mismatches,
        "transitionCounts": dict(sorted(transitions.items())),
        "changedEncodingCount": len(changed),
        "changedTensors": changed,
        "assignmentHashesDiffer": (
            left["summary"]["nameEncodingAssignmentSha256"]
            != right["summary"]["nameEncodingAssignmentSha256"]
        ),
        "observation": (
            "For these two pinned artifacts, the shared release label does not uniquely "
            "identify the serialized tensor-to-encoding assignment."
        ),
        "nonClaims": [
            "No conversion command or calibration input is inferred from the assignment difference.",
            "No ecosystem prevalence or performance effect is estimated.",
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-index", type=Path, default=root / "experiments/gguf-imatrix/artifacts.json")
    parser.add_argument("--ibm-q4km", type=Path, required=True)
    parser.add_argument("--bartowski-q4km", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "experiments/gguf-imatrix/results")
    args = parser.parse_args()

    index_bytes = args.artifact_index.read_bytes()
    index = json.loads(index_bytes)
    artifacts = {item["id"]: item for item in index["artifacts"]}
    ibm = measure(args.ibm_q4km, artifacts["ibm-granite-4.2-3b-q4-k-m"], False)
    bartowski = measure(args.bartowski_q4km, artifacts["bartowski-granite-4.2-3b-q4-k-m"], False)
    imatrix = measure(args.imatrix, artifacts["bartowski-granite-4.2-3b-imatrix"], True)

    write_json(args.output_dir / "ibm-q4km-tensor-inventory.json", ibm)
    write_json(args.output_dir / "bartowski-q4km-tensor-inventory.json", bartowski)
    write_json(args.output_dir / "tensor-assignment-comparison.json", compare(ibm, bartowski))
    write_json(args.output_dir / "imatrix-measurement.json", imatrix)
    write_json(args.output_dir / "measurement-run.json", {
        "recordType": "tensor-quantization-metadata-study.gguf-measurement-run.v1",
        "artifactIndexSha256": hashlib.sha256(index_bytes).hexdigest(),
        "artifactIds": list(artifacts),
        "resultFiles": [
            "ibm-q4km-tensor-inventory.json",
            "bartowski-q4km-tensor-inventory.json",
            "tensor-assignment-comparison.json",
            "imatrix-measurement.json",
        ],
    })


if __name__ == "__main__":
    main()
