#!/usr/bin/env python3
"""Verify committed cross-format model-card coverage evidence without model bytes."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/model-card-artifact-coverage"
RESULT = EXPERIMENT / "results/artifact-coverage.json"
MANIFEST = EXPERIMENT / "sources/SOURCE_MANIFEST.json"
MAPPINGS = (
    "Model Details",
    "Developed By",
    "License",
    "Supported Tasks",
    "Dataset Identity and Licensing",
    "Intended Use",
    "Technical Limitations",
    "Ethical Considerations",
)
EXPECTED_COHORTS = {"gguf": 2, "tflite": 50, "onnx": 15}
MODEL_EXTENSIONS = {".gguf", ".onnx", ".tflite", ".safetensors", ".bin"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_format_summary(subjects: list[dict[str, Any]], fmt: str) -> dict[str, Any]:
    group = [subject for subject in subjects if subject["format"] == fmt]
    mapping_rows = []
    for name in MAPPINGS:
        observations = [
            next(mapping for mapping in subject["mappings"] if mapping["mapping"] == name)
            for subject in group
        ]
        counts = Counter(mapping["artifact_presence"] for mapping in observations)
        mapping_rows.append({
            "mapping": name,
            "artifact_count": len(group),
            "strict_complete_count": counts["complete_candidate"],
            "partial_overlap_count": counts["partial"],
            "absent_count": counts["absent"],
            "any_relevant_presence_count": counts["complete_candidate"] + counts["partial"],
        })
    observation_count = len(group) * len(MAPPINGS)
    strict_count = sum(row["strict_complete_count"] for row in mapping_rows)
    return {
        "artifact_count": len(group),
        "mapping_count": len(MAPPINGS),
        "artifact_mapping_observation_count": observation_count,
        "strict_complete_observation_count": strict_count,
        "not_strict_complete_observation_count": observation_count - strict_count,
        "strict_complete_fraction": {"numerator": strict_count, "denominator": observation_count},
        "mappings": mapping_rows,
    }


def main() -> None:
    result = load(RESULT)
    manifest = load(MANIFEST)
    if manifest["artifact_bytes_included"] is not False:
        raise ValueError("manifest must state that artifact bytes are excluded")
    for record in manifest["inputs"]:
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"source binding mismatch: {record['path']}")
    if canonical_sha(result) != manifest["result_canonical_sha256"]:
        raise ValueError("result canonical SHA-256 does not match the source manifest")
    if result["record_type"] != "tensor_quantization_metadata_study.model_card_artifact_coverage.v1":
        raise ValueError("unexpected coverage result type")
    if result["scope"]["cohorts"] != EXPECTED_COHORTS:
        raise ValueError("cohort counts differ from the pinned study scope")
    subjects = result["subjects"]
    if len(subjects) != sum(EXPECTED_COHORTS.values()):
        raise ValueError("subject conservation failed")
    if len({(subject["format"], subject["id"]) for subject in subjects}) != len(subjects):
        raise ValueError("duplicate format/subject identity")
    for subject in subjects:
        names = tuple(mapping["mapping"] for mapping in subject["mappings"])
        if names != MAPPINGS:
            raise ValueError(f"mapping coverage/order mismatch for {subject['format']}:{subject['id']}")
        for mapping in subject["mappings"]:
            if mapping["artifact_presence"] not in {"complete_candidate", "partial", "absent"}:
                raise ValueError("unknown artifact-presence classification")
            if mapping["candidate_strict_complete"] != (mapping["artifact_presence"] == "complete_candidate"):
                raise ValueError("strict-completion flag disagrees with artifact-presence classification")
            if mapping["partial_overlap"] != (mapping["artifact_presence"] == "partial"):
                raise ValueError("partial-overlap flag disagrees with artifact-presence classification")
    for fmt in EXPECTED_COHORTS:
        if result["summary_by_format"][fmt] != expected_format_summary(subjects, fmt):
            raise ValueError(f"summary conservation failed for {fmt}")
    model_files = [path for path in EXPERIMENT.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS]
    if model_files:
        raise ValueError(f"model bytes are present in the evidence package: {model_files}")
    result_text = RESULT.read_text(encoding="utf-8")
    if "deepbom" in result_text.lower():
        raise ValueError("primary coverage result must remain implementation-neutral")
    if re.search(r'"[A-Za-z]:\\\\', result_text):
        raise ValueError("primary coverage result contains an absolute Windows path")
    print(json.dumps({
        "status": "pass",
        "artifact_count": len(subjects),
        "mapping_count": len(MAPPINGS),
        "observation_count": len(subjects) * len(MAPPINGS),
        "strict_complete_observations": {
            fmt: result["summary_by_format"][fmt]["strict_complete_observation_count"]
            for fmt in EXPECTED_COHORTS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
