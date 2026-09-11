#!/usr/bin/env python3
"""Verify committed GGUF/iMatrix measurements without downloading artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def name_encoding_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["name"].encode("utf-8")):
        digest.update(row["name"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(row["encoding"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_inventory(document: dict[str, Any]) -> None:
    rows = document["tensors"]
    summary = document["summary"]
    if len(rows) != summary["tensorCount"]:
        fail("Tensor count mismatch")
    if sum(row["elementCount"] for row in rows) != summary["elementCount"]:
        fail("Element count mismatch")
    if sum(row["byteLength"] for row in rows) != summary["serializedTensorBytes"]:
        fail("Tensor byte count mismatch")
    projection = [{
        "index": row["index"],
        "name": row["name"],
        "dtype": row["encoding"],
        "shape": row["shape"],
        "byte_length": row["byteLength"],
    } for row in rows]
    if canonical_sha256(projection) != summary["serializedDescriptorProjectionSha256"]:
        fail("Serialized descriptor projection hash mismatch")
    if name_encoding_sha256(rows) != summary["nameEncodingAssignmentSha256"]:
        fail("Name/encoding assignment hash mismatch")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "experiments/gguf-imatrix"
    results = experiment / "results"
    if list(experiment.rglob("*.gguf")):
        fail("GGUF artifact bytes must not be committed in the evidence directory")

    index = load(experiment / "artifacts.json")
    expected = {item["id"]: item for item in index["artifacts"]}
    ibm = load(results / "ibm-q4km-tensor-inventory.json")
    bart = load(results / "bartowski-q4km-tensor-inventory.json")
    matrix = load(results / "imatrix-measurement.json")
    comparison = load(results / "tensor-assignment-comparison.json")
    for document in (ibm, bart, matrix):
        verify_inventory(document)
        artifact = expected[document["source"]["id"]]
        if document["source"]["sha256"] != artifact["sha256"] or document["source"]["bytes"] != artifact["bytes"]:
            fail("Result is not bound to the artifact index")

    def counts(document: dict[str, Any]) -> dict[str, int]:
        return {row["encoding"]: row["tensorCount"] for row in document["summary"]["encodingInventory"]}

    if (ibm["summary"]["tensorCount"], ibm["summary"]["elementCount"], counts(ibm)) != (
        363, 3659737600, {"F32": 81, "Q4_K": 241, "Q6_K": 41}
    ):
        fail("Unexpected IBM Q4_K_M inventory")
    if (bart["summary"]["tensorCount"], bart["summary"]["elementCount"], counts(bart)) != (
        363, 3659737600, {"F32": 81, "Q4_K": 141, "Q5_K": 80, "Q6_K": 61}
    ):
        fail("Unexpected Bartowski Q4_K_M inventory")
    expected_transitions = {
        "F32->F32": 81,
        "Q4_K->Q4_K": 141,
        "Q4_K->Q5_K": 80,
        "Q4_K->Q6_K": 20,
        "Q6_K->Q6_K": 41,
    }
    if comparison["transitionCounts"] != expected_transitions:
        fail("Unexpected tensor-encoding transition counts")
    if not comparison["sameDeclaredReleaseLabel"] or comparison["declaredReleaseLabels"] != ["Q4_K_M", "Q4_K_M"]:
        fail("Expected the two comparison subjects to carry the same Q4_K_M label")
    if not comparison["sameTensorNameSet"] or comparison["shapeMismatchCount"] != 0:
        fail("Tensor names or shapes differ unexpectedly")
    if comparison["changedEncodingCount"] != 100 or not comparison["assignmentHashesDiffer"]:
        fail("Expected 100 changed encoding assignments")

    scan = matrix["numericalScan"]
    expected_scan = (560, 0, 942360, 0, 0, 0, 280)
    actual_scan = (
        scan["assessedTensorCount"], scan["unassessedTensorCount"], scan["decodedValueCount"],
        scan["nonfiniteValueCount"], scan["exactZeroValueCount"], scan["allZeroTensorCount"],
        scan["constantTensorCount"],
    )
    if actual_scan != expected_scan:
        fail(f"Unexpected importance-matrix scan: {actual_scan}")
    assessed = [row for row in scan["tensors"] if row["status"] == "assessed"]
    if sum(row["valueCount"] for row in assessed) != scan["decodedValueCount"]:
        fail("Per-tensor matrix value counts do not sum to the aggregate")
    if sum(row["nonfiniteValueCount"] for row in assessed) != scan["nonfiniteValueCount"]:
        fail("Per-tensor non-finite counts do not sum to the aggregate")

    crosscheck = load(experiment / "crosschecks/deepbom/normalized-comparison.json")
    if not crosscheck["allChecksPass"]:
        fail("Secondary implementation comparison contains a mismatch")
    field_observations = load(experiment / "standards-mapping/pr-1067/model-card-artifact-field-observations.json")
    if field_observations["source"]["requiredMappingCount"] != 8 or len(field_observations["rows"]) != 8:
        fail("Required mapping observation count is not eight")
    if any(key in field_observations for key in ("pass", "fail", "complete", "score")):
        fail("Field observations must not declare a completeness verdict")
    metadata_observations = load(results / "metadata-cross-source-observations.json")
    matrix_metadata = metadata_observations["importanceMatrixMetadata"]
    if matrix_metadata["ibmSerializedValues"]:
        fail("IBM sample unexpectedly contains quantize.imatrix metadata")
    if matrix_metadata["bartowskiSerializedValues"].get("quantize.imatrix.entries_count") != 280:
        fail("Bartowski sample imatrix entry metadata mismatch")
    lineage = metadata_observations["lineageMetadata"]
    if lineage["bartowskiArtifactSerializedBaseModel"].get("general.base_model.0.name") != "Granite 4.1 3b Base":
        fail("Serialized base-model observation changed")
    if lineage["bartowskiRepositoryCardBaseModel"] != "ibm-granite/granite-4.2-3b":
        fail("Repository-card base-model observation changed")

    primary_paths = list(results.glob("*.json")) + list((experiment / "standards-mapping").rglob("*.json"))
    for path in primary_paths:
        text = path.read_text(encoding="utf-8")
        if "deepbom" in text.lower():
            fail(f"Primary evidence is tool-branded: {path.relative_to(root)}")
        if re.search(r"[A-Za-z]:\\\\", text):
            fail(f"Absolute Windows path in public result: {path.relative_to(root)}")
    print(json.dumps({
        "status": "pass",
        "q4SubjectCount": 2,
        "tensorCountPerQ4Subject": 363,
        "changedEncodingCount": 100,
        "imatrixAssessedTensorCount": 560,
        "imatrixDecodedValueCount": 942360,
        "secondaryImplementationChecksPass": True,
    }, indent=2))


if __name__ == "__main__":
    main()
