import argparse
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path


SCHEMA = "tensor_quantization_metadata_study.tflite_metadata_flatc_crosscheck.v1"
SCHEMA_SHA256 = "2d3386ba124690ba1195bfc1d51ac814843bd675a7c845afab7c001c7891449e"
FLATC_RELEASE = {
    "version": "25.9.23",
    "archive_name": "Windows.flatc.binary.zip",
    "archive_sha256": "3d6383193ecd274f5de544a6e03464a87f581befb9fc1dda9cf508fa3cce3127",
    "url": "https://github.com/google/flatbuffers/releases/download/v25.9.23/Windows.flatc.binary.zip",
}

CONTENT_TYPES = {
    None: "NONE",
    "NONE": "NONE",
    "FeatureProperties": "FEATURE",
    "ImageProperties": "IMAGE",
    "BoundingBoxProperties": "BOUNDING_BOX",
    "AudioProperties": "AUDIO",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_auditor(root):
    path = root / "scripts/audit-tflite-metadata.py"
    spec = importlib.util.spec_from_file_location("tflite_metadata_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalization_projection(tensor_metadata):
    units = tensor_metadata.get("process_units", [])
    rows = [
        unit.get("options", {})
        for unit in units
        if unit.get("options_type") == "NormalizationOptions"
    ]
    if not rows:
        return None
    return [
        {
            "mean": [float(value) for value in row.get("mean", [])],
            "std": [float(value) for value in row.get("std", [])],
        }
        for row in rows
    ]


def tensor_projection(tensor_metadata):
    content = tensor_metadata.get("content", {})
    content_type = content.get("content_properties_type")
    properties = content.get("content_properties", {})
    associated = [
        {
            "name": row.get("name"),
            "type": row.get("type", "UNKNOWN"),
        }
        for row in tensor_metadata.get("associated_files", [])
    ]
    return {
        "name": tensor_metadata.get("name"),
        "content": {
            "type": CONTENT_TYPES.get(content_type, f"UNKNOWN_{content_type}"),
            "color_space": (
                properties.get("color_space")
                if content_type == "ImageProperties"
                else None
            ),
        },
        "normalizations": normalization_projection(tensor_metadata),
        "associated_files": associated,
    }


def compare_artifact(audit_artifact, audit_parameters, flatc_document):
    mismatches = []
    model_metadata = audit_artifact["model_metadata"]
    for field in (
        "name",
        "description",
        "version",
        "author",
        "license",
        "min_parser_version",
    ):
        if flatc_document.get(field) != model_metadata.get(field):
            mismatches.append({
                "scope": "model",
                "field": field,
                "audit": model_metadata.get(field),
                "flatc": flatc_document.get(field),
            })

    subgraphs = flatc_document.get("subgraph_metadata", [])
    if len(subgraphs) != audit_artifact["metadata_subgraph_count"]:
        mismatches.append({
            "scope": "model",
            "field": "subgraph_count",
            "audit": audit_artifact["metadata_subgraph_count"],
            "flatc": len(subgraphs),
        })
        return mismatches
    if not subgraphs:
        return mismatches

    main = subgraphs[0]
    for direction in ("input", "output"):
        parameters = sorted(
            [row for row in audit_parameters if row["direction"] == direction],
            key=lambda row: row["ordinal"],
        )
        tensors = main.get(f"{direction}_tensor_metadata", [])
        if len(parameters) != len(tensors):
            mismatches.append({
                "scope": direction,
                "field": "tensor_metadata_count",
                "audit": len(parameters),
                "flatc": len(tensors),
            })
            continue
        for parameter, tensor_metadata in zip(parameters, tensors):
            flatc = tensor_projection(tensor_metadata)
            expected_normalization = parameter["normalization"]
            audit_normalizations = None
            if expected_normalization["status"].startswith("PRESENT_"):
                audit_normalizations = [{
                    "mean": expected_normalization["mean"],
                    "std": expected_normalization["std"],
                }]
            expected = {
                "name": parameter["tensor_metadata_name"],
                "content": parameter["content"],
                "normalizations": audit_normalizations,
                "associated_files": [
                    {"name": row["name"], "type": row["type"]}
                    for row in parameter["declared_associated_files"]
                ],
            }
            for field in (
                "name",
                "content",
                "normalizations",
                "associated_files",
            ):
                if flatc[field] != expected[field]:
                    mismatches.append({
                        "scope": f"{direction}:{parameter['ordinal']}",
                        "field": field,
                        "audit": expected[field],
                        "flatc": flatc[field],
                    })
    return mismatches


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=root / "data/tflite-metadata-audit.json",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=root / "data/artifacts.json",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=root / "cache/tflite-metadata-audit",
    )
    parser.add_argument(
        "--flatc",
        type=Path,
        default=root / "cache/flatc-crosscheck/bin/flatc.exe",
    )
    parser.add_argument(
        "--metadata-schema",
        type=Path,
        default=root / "cache/flatc-crosscheck/metadata_schema.fbs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/tflite-metadata-flatc-crosscheck.json",
    )
    args = parser.parse_args()

    if sha256_file(args.metadata_schema) != SCHEMA_SHA256:
        raise ValueError("Pinned metadata_schema.fbs SHA-256 mismatch")
    flatc_version = subprocess.run(
        [str(args.flatc), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if flatc_version != f"flatc version {FLATC_RELEASE['version']}":
        raise ValueError(f"Unexpected flatc version: {flatc_version}")

    auditor = load_auditor(root)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.artifacts.read_text(encoding="utf-8"))
    source_artifacts = {
        row["qualified_id"]: row for row in manifest["artifacts"]
    }
    parameters = {}
    for row in audit["parameters"]:
        parameters.setdefault(row["qualified_id"], []).append(row)

    results = []
    with tempfile.TemporaryDirectory(
        dir=root / "cache", prefix="metadata-flatc-"
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        for artifact in audit["artifacts"]:
            if artifact["metadata_status"] != "PRESENT_PARSEABLE":
                continue
            qualified_id = artifact["qualified_id"]
            model_path = auditor.ensure_artifact(
                args.model_cache, source_artifacts[qualified_id], False
            )
            model = auditor.tflite.Model.GetRootAsModel(
                model_path.read_bytes(), 0
            )
            entries = [
                row for row in auditor.metadata_entries(model)
                if row["name"] == "TFLITE_METADATA"
            ]
            if len(entries) != 1:
                raise ValueError(
                    f"Expected one TFLITE_METADATA entry: {qualified_id}"
                )
            stem = artifact["artifact_sha256"]
            binary_path = temporary / f"{stem}.bin"
            binary_path.write_bytes(entries[0]["raw"])
            subprocess.run(
                [
                    str(args.flatc),
                    "--json",
                    "--strict-json",
                    "--defaults-json",
                    "-o",
                    str(temporary),
                    str(args.metadata_schema),
                    "--",
                    str(binary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            document = json.loads(
                (temporary / f"{stem}.json").read_text(encoding="utf-8")
            )
            mismatches = compare_artifact(
                artifact, parameters[qualified_id], document
            )
            results.append({
                "qualified_id": qualified_id,
                "artifact_sha256": artifact["artifact_sha256"],
                "external_parameter_count": len(parameters[qualified_id]),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
            })

    output = {
        "schema": SCHEMA,
        "tool": "scripts/crosscheck-tflite-metadata-flatc.py",
        "source_audit_sha256": sha256_file(args.audit),
        "metadata_schema_sha256": sha256_file(args.metadata_schema),
        "flatc": {
            **FLATC_RELEASE,
            "executable_sha256": sha256_file(args.flatc),
        },
        "artifact_count": len(results),
        "external_parameter_count": sum(
            row["external_parameter_count"] for row in results
        ),
        "mismatch_count": sum(row["mismatch_count"] for row in results),
        "results": results,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass" if output["mismatch_count"] == 0 else "fail",
        "artifact_count": output["artifact_count"],
        "external_parameter_count": output["external_parameter_count"],
        "mismatch_count": output["mismatch_count"],
    }, indent=2))
    if output["mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
