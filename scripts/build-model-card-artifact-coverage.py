#!/usr/bin/env python3
"""Build a bounded cross-format model-card artifact evidence coverage study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/model-card-artifact-coverage"
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


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def observation(
    mapping: dict[str, Any],
    capability: str,
    fields: list[str],
    values: dict[str, Any],
    strict: bool,
    relationship: str,
    external_needed: list[str],
) -> dict[str, Any]:
    present = {key: value for key, value in values.items() if nonempty(value)}
    partial = bool(present) and not strict
    return {
        "mapping": mapping["nativeName"],
        "catalog_expression": mapping["expression"],
        "catalog_via": mapping.get("via", []),
        "format_capability": capability,
        "standardized_relevant_fields": fields,
        "semantic_relationship": relationship,
        "artifact_presence": "complete_candidate" if strict else "partial" if partial else "absent",
        "primary_extraction": present,
        "candidate_strict_complete": strict,
        "partial_overlap": partial,
        "external_evidence_needed": [] if strict else external_needed,
    }


def tflite_rows(mapping_by_name: dict[str, Any], record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = record.get("model_metadata") or {}
    details = {name: metadata.get(name) for name in ("name", "version", "description")}
    rows = [
        observation(mapping_by_name["Model Details"], "direct_fields", list(details), details,
                    all(nonempty(value) for value in details.values()),
                    "TFLite Model Metadata defines name, version, and description fields; the strict candidate requires all three.",
                    ["one or more non-empty model detail fields"]),
        observation(mapping_by_name["Developed By"], "partial_semantic_overlap", ["author"], {"author": metadata.get("author")}, False,
                    "Model Metadata author is not a CycloneDX party with an asserted supplier role.",
                    ["publisher-bound party identity", "supplier role assertion"]),
        observation(mapping_by_name["License"], "direct_field", ["license"], {"license": metadata.get("license")}, nonempty(metadata.get("license")),
                    "TFLite Model Metadata defines an artifact-contained free-text license field.",
                    ["non-empty artifact-contained license declaration"]),
    ]
    absent = {
        "Supported Tasks": ([], "No standardized TFLite Model Metadata field directly encodes the CycloneDX task collection.", ["publisher task declaration"]),
        "Dataset Identity and Licensing": (["associated_files"], "Associated files do not by themselves identify and license training datasets.", ["training dataset identity", "dataset version or identifiers", "dataset license"]),
        "Intended Use": ([], "No standardized TFLite Model Metadata field directly establishes intended use.", ["publisher intended-use declaration"]),
        "Technical Limitations": (["description"], "A general description is not treated as a structured limitations declaration.", ["publisher limitations declaration"]),
        "Ethical Considerations": ([], "No standardized TFLite Model Metadata risk object corresponds to this mapping.", ["externally bound ethical-risk evidence"]),
    }
    for name, (fields, relationship, needed) in absent.items():
        rows.append(observation(mapping_by_name[name], "absent" if not fields else "partial_schema_overlap", fields, {}, False, relationship, needed))
    return sorted(rows, key=lambda row: MAPPINGS.index(row["mapping"]))


def onnx_rows(mapping_by_name: dict[str, Any], model: onnx.ModelProto) -> list[dict[str, Any]]:
    details = {
        "graph.name": model.graph.name,
        "model_version": model.model_version if model.model_version != 0 else None,
        "doc_string": model.doc_string,
    }
    rows = [
        observation(mapping_by_name["Model Details"], "candidate_direct_fields", list(details), details,
                    all(nonempty(value) for value in details.values()),
                    "The candidate maps graph.name, non-zero model_version, and ModelProto doc_string to the three selected detail fields.",
                    ["one or more non-empty model detail fields", "publisher confirmation that graph.name identifies the model"]),
        observation(mapping_by_name["Developed By"], "partial_semantic_overlap", ["producer_name", "producer_version"],
                    {"producer_name": model.producer_name, "producer_version": model.producer_version}, False,
                    "ONNX producer identity describes the producing tool or framework, not necessarily a supplier party.",
                    ["publisher-bound party identity", "supplier role assertion"]),
    ]
    absent = {
        "License": ([], "ONNX ModelProto has no standardized license field; metadata_props keys are producer-defined.", ["publisher license declaration"]),
        "Supported Tasks": ([], "ONNX ModelProto has no standardized supported-task field.", ["publisher task declaration"]),
        "Dataset Identity and Licensing": ([], "ONNX ModelProto has no standardized training-dataset identity and license object.", ["training dataset identity", "dataset version or identifiers", "dataset license"]),
        "Intended Use": ([], "ONNX ModelProto has no standardized intended-use field.", ["publisher intended-use declaration"]),
        "Technical Limitations": (["doc_string"], "A general model doc_string is not treated as a structured limitations declaration.", ["publisher limitations declaration"]),
        "Ethical Considerations": ([], "ONNX ModelProto has no standardized ethical-risk object corresponding to this mapping.", ["externally bound ethical-risk evidence"]),
    }
    for name, (fields, relationship, needed) in absent.items():
        rows.append(observation(mapping_by_name[name], "absent" if not fields else "partial_schema_overlap", fields, {}, False, relationship, needed))
    return sorted(rows, key=lambda row: MAPPINGS.index(row["mapping"]))


def gguf_rows(mapping_by_name: dict[str, Any], source_rows: list[dict[str, Any]], subject_key: str) -> list[dict[str, Any]]:
    output = []
    for source in source_rows:
        name = source["mapping"]
        values = source[subject_key]
        if name == "Model Details":
            strict = all(nonempty(values.get(key)) for key in ("general.name", "general.version", "general.description"))
            needed = ["one or more non-empty model detail fields"]
            capability = "direct_fields"
        elif name == "License":
            strict = any(nonempty(values.get(key)) for key in ("general.license", "general.license.name", "general.license.link"))
            needed = ["non-empty standardized GGUF license declaration"]
            capability = "direct_field"
        elif name == "Developed By":
            strict = False
            needed = ["publisher-bound party identity", "supplier role assertion"]
            capability = "partial_semantic_overlap"
        elif name == "Supported Tasks":
            strict = False
            needed = ["mapping from tags to a publisher-asserted task declaration"]
            capability = "partial_semantic_overlap"
        elif name == "Dataset Identity and Licensing":
            strict = False
            needed = ["dataset version or identifiers", "dataset license", "binding to the training relationship"]
            capability = "partial_schema_overlap"
        elif name == "Technical Limitations":
            strict = False
            needed = ["publisher limitations declaration"]
            capability = "partial_schema_overlap"
        else:
            strict = False
            needed = ["publisher intended-use declaration"] if name == "Intended Use" else ["externally bound ethical-risk evidence"]
            capability = "absent"
        output.append(observation(mapping_by_name[name], capability, source["ggufStandardizedRelevantKeys"], values,
                                  strict, source["semanticRelationship"], needed))
    return sorted(output, key=lambda row: MAPPINGS.index(row["mapping"]))


def locate_onnx(cache: Path, expected: dict[str, Any], by_hash: dict[str, Path]) -> Path:
    target = by_hash.get(expected["sha256"])
    if target is None:
        raise FileNotFoundError(f"ONNX cache does not contain {expected['id']} ({expected['sha256']})")
    if target.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"ONNX byte length mismatch for {expected['id']}")
    return target


def summarize(subjects: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for fmt in ("gguf", "tflite", "onnx"):
        group = [row for row in subjects if row["format"] == fmt]
        mapping_rows = []
        for name in MAPPINGS:
            observations = [next(item for item in subject["mappings"] if item["mapping"] == name) for subject in group]
            counts = Counter(item["artifact_presence"] for item in observations)
            mapping_rows.append({
                "mapping": name,
                "artifact_count": len(group),
                "strict_complete_count": counts["complete_candidate"],
                "partial_overlap_count": counts["partial"],
                "absent_count": counts["absent"],
                "any_relevant_presence_count": counts["complete_candidate"] + counts["partial"],
            })
        total = len(group) * len(MAPPINGS)
        strict_total = sum(row["strict_complete_count"] for row in mapping_rows)
        result[fmt] = {
            "artifact_count": len(group),
            "mapping_count": len(MAPPINGS),
            "artifact_mapping_observation_count": total,
            "strict_complete_observation_count": strict_total,
            "not_strict_complete_observation_count": total - strict_total,
            "strict_complete_fraction": {"numerator": strict_total, "denominator": total},
            "mappings": mapping_rows,
        }
    return result


def build(onnx_cache: Path) -> dict[str, Any]:
    catalog_path = ROOT / "experiments/gguf-imatrix/sources/pr-1067/model-card-perspective.json"
    catalog = load(catalog_path)["perspectives"][0]
    required = [row for row in catalog["mappings"] if row.get("relevance") == "required"]
    if tuple(row["nativeName"] for row in required) != MAPPINGS:
        raise ValueError("Pinned required mapping names or order changed")
    mapping_by_name = {row["nativeName"]: row for row in required}

    subjects: list[dict[str, Any]] = []
    gguf_path = ROOT / "experiments/gguf-imatrix/standards-mapping/pr-1067/model-card-artifact-field-observations.json"
    gguf = load(gguf_path)
    for subject_id, key in ((gguf["subjects"][0], "ibmSerializedValues"), (gguf["subjects"][1], "bartowskiSerializedValues")):
        subjects.append({"id": subject_id, "format": "gguf", "primary_reader": "gguf-py 0.19.0",
                         "mappings": gguf_rows(mapping_by_name, gguf["rows"], key)})

    tflite_path = ROOT / "data/tflite-metadata-audit.json"
    tflite = load(tflite_path)
    for record in tflite["artifacts"]:
        subjects.append({"id": record["qualified_id"], "format": "tflite",
                         "artifact_sha256": record["artifact_sha256"], "artifact_size_bytes": record["artifact_size_bytes"],
                         "primary_reader": f"{tflite['tool']} using ai-edge-litert 2.1.2 generated schema",
                         "mappings": tflite_rows(mapping_by_name, record)})

    onnx_manifest_path = ROOT / "data/onnx-pilot-manifest.json"
    onnx_manifest = load(onnx_manifest_path)
    by_hash = {}
    for path in onnx_cache.rglob("*.onnx"):
        digest = sha256_file(path)
        if digest in by_hash:
            raise ValueError(f"ONNX cache repeats SHA-256 {digest}")
        by_hash[digest] = path
    for record in onnx_manifest["artifacts"]:
        path = locate_onnx(onnx_cache, record, by_hash)
        model = onnx.load_model(str(path), load_external_data=False)
        subjects.append({"id": record["id"], "format": "onnx", "artifact_sha256": record["sha256"],
                         "artifact_size_bytes": record["size_bytes"], "primary_reader": f"onnx {onnx.__version__}",
                         "producer_defined_metadata_property_names": sorted(item.key for item in model.metadata_props),
                         "mappings": onnx_rows(mapping_by_name, model)})

    summary = summarize(subjects)
    return {
        "record_type": "tensor_quantization_metadata_study.model_card_artifact_coverage.v1",
        "normative_status": "bounded observations under an explicitly stated candidate profile; not CycloneDX conformance verdicts",
        "scope": {
            "cyclonedx_pull_request": 1067,
            "cyclonedx_head": "7a7d2dd599968e528349ca3cf262120da2831ca8",
            "required_mapping_count": len(MAPPINGS),
            "cohorts": {name: summary[name]["artifact_count"] for name in ("gguf", "tflite", "onnx")},
            "representativeness": "fixed public cohorts only; no population estimate",
        },
        "candidate_profile": {
            "complete_candidate": "all selected direct fields required by the mapping-specific rule are non-empty in the artifact",
            "partial": "at least one standardized or explicitly identified semantically related artifact field is present, but the strict rule is not met",
            "absent": "no qualifying artifact-contained field is observed",
            "arbitrary_extensions": "producer-defined extension capacity is not counted as standardized format capability",
            "compound_model_details": "name, version, and description must all be non-empty for strict completion",
            "developed_by": "author or producer fields alone do not establish a CycloneDX supplier party role",
        },
        "summary_by_format": summary,
        "subjects": subjects,
        "bounded_conclusion": "In the pinned GGUF, TFLite, and ONNX cohorts, standardized artifact-contained fields alone did not establish most model-card mappings under the stated candidate completeness profile.",
        "boundary": "Publisher declarations, repository metadata, arbitrary custom metadata, inferred task labels, and externally bound evidence are not silently promoted to artifact-contained standardized facts.",
    }


def manifest(result: dict[str, Any], onnx_cache: Path) -> dict[str, Any]:
    inputs = [
        "data/tflite-metadata-audit.json",
        "data/onnx-pilot-manifest.json",
        "experiments/gguf-imatrix/standards-mapping/pr-1067/model-card-artifact-field-observations.json",
        "experiments/gguf-imatrix/sources/pr-1067/model-card-perspective.json",
        "scripts/build-model-card-artifact-coverage.py",
    ]
    return {
        "record_type": "tensor_quantization_metadata_study.model_card_artifact_coverage_source_manifest.v1",
        "artifact_bytes_included": False,
        "inputs": [{"path": name, "sha256": sha256_file(ROOT / name)} for name in inputs],
        "runtime": {"python": platform.python_version(), "onnx": importlib.metadata.version("onnx")},
        "onnx_cache": {"required": True, "committed": False, "verified_artifact_count": result["scope"]["cohorts"]["onnx"]},
        "result_canonical_sha256": sha256_bytes(canonical_bytes(result)),
    }


def render_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Model-card artifact-contained evidence coverage",
        "",
        "This bounded study compares eight required mappings from CycloneDX #1067 head `7a7d2dd` with fields actually contained in fixed public GGUF, TFLite, and ONNX cohorts. It does not define CycloneDX completeness or conformance.",
        "",
        "| Format | Artifacts | Strict complete observations | Mapping observations |",
        "| --- | ---: | ---: | ---: |",
    ]
    for fmt in ("gguf", "tflite", "onnx"):
        row = result["summary_by_format"][fmt]
        lines.append(f"| {fmt.upper()} | {row['artifact_count']} | {row['strict_complete_observation_count']} | {row['artifact_mapping_observation_count']} |")
    lines += [
        "",
        "Counts are reported per format rather than pooled as an ecosystem prevalence estimate. A strict Model Details observation requires non-empty name, version, and description. Producer-defined extension fields and repository metadata are excluded from standardized artifact capability.",
        "",
        "> " + result["bounded_conclusion"],
        "",
        "See `protocol.md` for the interpretation profile and `results/artifact-coverage.json` for every subject/mapping observation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-cache", type=Path, default=ROOT / "cache/onnx-pilot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(args.onnx_cache.resolve())
    source_manifest = manifest(result, args.onnx_cache.resolve())
    outputs = {
        EXPERIMENT / "results/artifact-coverage.json": json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        EXPERIMENT / "sources/SOURCE_MANIFEST.json": json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        EXPERIMENT / "summary.md": render_summary(result),
    }
    if args.write:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        return
    for path, expected in outputs.items():
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"stale or missing generated evidence: {path.relative_to(ROOT)}")
    for fmt, row in result["summary_by_format"].items():
        if row["strict_complete_observation_count"] * 2 >= row["artifact_mapping_observation_count"]:
            raise SystemExit(f"bounded conclusion is not supported for {fmt}")
    print("Model-card artifact coverage evidence is reproducible and internally consistent.")


if __name__ == "__main__":
    main()
