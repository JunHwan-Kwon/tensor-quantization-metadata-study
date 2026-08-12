import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEMA = "tensor_quantization_metadata_study.tflite_metadata_audit.v1"


def fail(message):
    raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    artifacts_path = root / "data/artifacts.json"
    interfaces_path = root / "data/interface-contracts.json"
    audit_path = root / "data/tflite-metadata-audit.json"
    source_artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))[
        "artifacts"
    ]
    source_interfaces = json.loads(
        interfaces_path.read_text(encoding="utf-8")
    )["parameters"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    if audit["schema"] != SCHEMA:
        fail("Unexpected metadata-audit schema")
    if audit["source_artifact_manifest_sha256"] != sha256_file(artifacts_path):
        fail("Metadata audit is not bound to the current artifact manifest")
    if audit["source_interface_ledger_sha256"] != sha256_file(interfaces_path):
        fail("Metadata audit is not bound to the current interface ledger")

    expected_artifacts = {row["qualified_id"]: row for row in source_artifacts}
    observed_artifacts = {row["qualified_id"]: row for row in audit["artifacts"]}
    if set(expected_artifacts) != set(observed_artifacts):
        fail("Metadata-audit artifact identifiers differ from the source manifest")
    for qualified_id, expected in expected_artifacts.items():
        observed = observed_artifacts[qualified_id]
        if observed["artifact_sha256"] != expected["sha256"]:
            fail(f"Artifact hash mismatch: {qualified_id}")
        if observed["artifact_size_bytes"] != expected["size_bytes"]:
            fail(f"Artifact size mismatch: {qualified_id}")
        if observed["metadata_status"] == "PRESENT_PARSE_FAILED":
            if not observed["metadata_parse_error"]:
                fail(f"Missing metadata parse error: {qualified_id}")
        elif observed["metadata_parse_error"] is not None:
            fail(f"Unexpected metadata parse error: {qualified_id}")

    source_parameters = {
        (row["qualified_id"], row["direction"], row["ordinal"]): row
        for row in source_interfaces
    }
    observed_parameters = {
        (row["qualified_id"], row["direction"], row["ordinal"]): row
        for row in audit["parameters"]
    }
    if set(source_parameters) != set(observed_parameters):
        fail("Metadata-audit parameter keys differ from the interface ledger")
    for key, expected in source_parameters.items():
        observed = observed_parameters[key]
        for field in ("parameter_id", "tensor_index", "dtype", "shape"):
            if observed[field] != expected[field]:
                fail(f"Parameter mismatch for {key}/{field}")
        if observed["affine_contract_status"] != expected["quantization"]["status"]:
            fail(f"Affine status mismatch: {key}")
        status = observed["normalization"]["status"]
        mean = observed["normalization"]["mean"]
        std = observed["normalization"]["std"]
        if status == "ABSENT" and (mean is not None or std is not None):
            fail(f"Absent normalization carries values: {key}")
        if status == "NOT_ASSESSABLE" and (mean is not None or std is not None):
            fail(f"Unassessed normalization carries values: {key}")
        if status == "PRESENT_VALID" and (not mean or not std):
            fail(f"Valid normalization has empty values: {key}")
        if observed["full_preprocessing_pipeline_status"] != "NOT_ASSESSABLE":
            fail(f"Full preprocessing was overclaimed: {key}")

    artifact_status = Counter(
        row["metadata_status"] for row in audit["artifacts"]
    )
    input_rows = [row for row in audit["parameters"] if row["direction"] == "input"]
    mapping_status = Counter(
        row["metadata_mapping_status"] for row in audit["parameters"]
    )
    normalization_status = Counter(
        row["normalization"]["status"] for row in input_rows
    )
    image_inputs = [
        row for row in input_rows if row["content"]["type"] == "IMAGE"
    ]
    quantized_inputs = [
        row for row in input_rows if row["affine_contract_status"] == "complete"
    ]
    cohorts = {}
    for cohort in sorted({row["cohort"] for row in audit["artifacts"]}):
        cohort_artifacts = [
            row for row in audit["artifacts"] if row["cohort"] == cohort
        ]
        cohort_inputs = [row for row in input_rows if row["cohort"] == cohort]
        cohorts[cohort] = {
            "artifact_count": len(cohort_artifacts),
            "parseable_tflite_metadata_count": sum(
                row["metadata_status"] == "PRESENT_PARSEABLE"
                for row in cohort_artifacts
            ),
            "input_parameter_count": len(cohort_inputs),
            "quantized_input_count": sum(
                row["affine_contract_status"] == "complete"
                for row in cohort_inputs
            ),
            "mapped_input_metadata_count": sum(
                row["metadata_mapping_status"]
                == "MAPPED_BY_DIRECTION_AND_ORDINAL"
                for row in cohort_inputs
            ),
            "valid_input_normalization_count": sum(
                row["normalization"]["status"] == "PRESENT_VALID"
                for row in cohort_inputs
            ),
        }
    model_fields = {
        field: sum(
            row["model_metadata"] is not None
            and row["model_metadata"].get(field) is not None
            for row in audit["artifacts"]
        )
        for field in (
            "name",
            "description",
            "version",
            "author",
            "license",
            "min_parser_version",
        )
    }
    expected_summary = {
        "artifact_count": len(audit["artifacts"]),
        "artifact_metadata_status_counts": dict(sorted(artifact_status.items())),
        "artifact_with_parseable_tflite_metadata_count": sum(
            row["metadata_status"] == "PRESENT_PARSEABLE"
            for row in audit["artifacts"]
        ),
        "external_parameter_count": len(audit["parameters"]),
        "input_parameter_count": len(input_rows),
        "parameter_mapping_status_counts": dict(sorted(mapping_status.items())),
        "input_normalization_status_counts": dict(
            sorted(normalization_status.items())
        ),
        "mapped_image_input_count": len(image_inputs),
        "mapped_image_input_with_valid_normalization_count": sum(
            row["normalization"]["status"] == "PRESENT_VALID"
            for row in image_inputs
        ),
        "quantized_input_count": len(quantized_inputs),
        "quantized_input_with_valid_normalization_count": sum(
            row["normalization"]["status"] == "PRESENT_VALID"
            for row in quantized_inputs
        ),
        "artifact_with_valid_input_normalization_count": len({
            row["qualified_id"]
            for row in input_rows
            if row["normalization"]["status"] == "PRESENT_VALID"
        }),
        "model_metadata_field_presence_counts": model_fields,
        "artifact_with_missing_packaged_associated_file_count": sum(
            bool(row["missing_packaged_associated_file_names"])
            for row in audit["artifacts"]
        ),
        "cohort_results": cohorts,
    }
    for key, value in expected_summary.items():
        if audit["summary"].get(key) != value:
            fail(f"Metadata-audit summary mismatch: {key}")

    frozen_expected = {
        "artifact_metadata_status_counts": {
            "ABSENT": 30,
            "PRESENT_PARSEABLE": 20,
        },
        "input_normalization_status_counts": {
            "ABSENT": 7,
            "NOT_ASSESSABLE": 30,
            "PRESENT_VALID": 15,
        },
        "artifact_with_parseable_tflite_metadata_count": 20,
        "artifact_with_valid_input_normalization_count": 15,
        "mapped_image_input_count": 15,
        "mapped_image_input_with_valid_normalization_count": 15,
        "quantized_input_count": 32,
        "quantized_input_with_valid_normalization_count": 3,
        "artifact_with_missing_packaged_associated_file_count": 0,
        "model_metadata_field_presence_counts": {
            "author": 18,
            "description": 20,
            "license": 6,
            "min_parser_version": 20,
            "name": 20,
            "version": 18,
        },
    }
    for key, value in frozen_expected.items():
        if audit["summary"].get(key) != value:
            fail(f"Frozen 50-file result changed for {key}")

    expected_cohorts = {
        "google-legacy-quantized": (4, 0, 4, 0),
        "litert-modern-static-int8": (15, 0, 15, 0),
        "mcunet-curated-ptq": (11, 0, 10, 0),
        "mediapipe-public": (20, 20, 3, 15),
    }
    for cohort, expected in expected_cohorts.items():
        row = audit["summary"]["cohort_results"].get(cohort)
        if row is None:
            fail(f"Missing frozen cohort: {cohort}")
        actual = (
            row["artifact_count"],
            row["parseable_tflite_metadata_count"],
            row["quantized_input_count"],
            row["valid_input_normalization_count"],
        )
        if actual != expected:
            fail(f"Frozen cohort result changed: {cohort}")

    if audit["artifact_count"] != len(source_artifacts):
        fail("Metadata-audit artifact count mismatch")
    if audit["parameter_count"] != len(source_interfaces):
        fail("Metadata-audit parameter count mismatch")

    print(json.dumps({
        "status": "pass",
        "artifact_count": audit["artifact_count"],
        "parameter_count": audit["parameter_count"],
        "parseable_metadata_count": audit["summary"][
            "artifact_with_parseable_tflite_metadata_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
