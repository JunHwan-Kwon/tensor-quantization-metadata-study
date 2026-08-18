import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_COHORTS = {
    "google-legacy-quantized": 4,
    "litert-modern-static-int8": 15,
    "mcunet-curated-ptq": 11,
    "mediapipe-public": 20,
}


def fail(message):
    raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_files(root):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            relative.name == "SHA256SUMS"
            or ".git" in relative.parts
            or ".venv" in relative.parts
            or "cache" in relative.parts
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
        ):
            continue
        yield relative


def write_sha256s(root):
    lines = [
        f"{sha256_file(root / relative)}  {relative.as_posix()}"
        for relative in public_files(root)
    ]
    (root / "SHA256SUMS").write_bytes(
        ("\n".join(lines) + "\n").encode("ascii")
    )


def verify_sha256s(root):
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        fail("SHA256SUMS is missing")
    recorded = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        recorded[relative] = digest
    actual_paths = {path.as_posix() for path in public_files(root)}
    if set(recorded) != actual_paths:
        missing = sorted(actual_paths - set(recorded))
        extra = sorted(set(recorded) - actual_paths)
        fail(f"SHA256SUMS path mismatch: missing={missing}, extra={extra}")
    for relative, digest in recorded.items():
        actual = sha256_file(root / relative)
        if actual != digest:
            fail(f"SHA-256 mismatch: {relative}")


def verify_public_ledger(root):
    artifacts = json.loads(
        (root / "data/artifacts.json").read_text(encoding="utf-8")
    )
    contracts = json.loads(
        (root / "data/interface-contracts.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads(
        (root / "data/interface-summary.json").read_text(encoding="utf-8")
    )
    corpus_summary = json.loads(
        (root / "data/corpus-summary.json").read_text(encoding="utf-8")
    )
    crosscheck = json.loads(
        (root / "data/litert-interface-crosscheck.json").read_text(
            encoding="utf-8"
        )
    )

    expected_schemas = {
        "artifacts": "tensor_quantization_metadata_study.public_artifacts.v1",
        "contracts": (
            "tensor_quantization_metadata_study.interface_contract_ledger.v1"
        ),
        "summary": "tensor_quantization_metadata_study.interface_summary.v1",
        "corpus_summary": "tensor_quantization_metadata_study.corpus_summary.v1",
        "crosscheck": (
            "tensor_quantization_metadata_study."
            "litert_interface_verification.v1"
        ),
    }
    documents = {
        "artifacts": artifacts,
        "contracts": contracts,
        "summary": summary,
        "corpus_summary": corpus_summary,
        "crosscheck": crosscheck,
    }
    for name, schema in expected_schemas.items():
        if documents[name].get("schema") != schema:
            fail(f"Unexpected schema for {name}")
    artifacts_sha256 = sha256_file(root / "data/artifacts.json")
    contracts_sha256 = sha256_file(root / "data/interface-contracts.json")
    if summary.get("source_interface_contract_ledger_sha256") != contracts_sha256:
        fail("Interface summary is not bound to the interface ledger")
    if corpus_summary.get("source_artifact_manifest_sha256") != artifacts_sha256:
        fail("Corpus summary is not bound to the artifact manifest")
    nested_summary = corpus_summary.get(
        "interface_quantization_contract_summary", {}
    )
    if (
        nested_summary.get("source_interface_contract_ledger_sha256")
        != contracts_sha256
    ):
        fail("Corpus summary is not bound to the interface ledger")
    if (
        crosscheck.get("source_interface_contract_ledger_sha256")
        != contracts_sha256
    ):
        fail("LiteRT cross-check is not bound to the interface ledger")

    artifact_rows = artifacts["artifacts"]
    if len(artifact_rows) != 50:
        fail("Artifact count is not 50")
    if len({row["qualified_id"] for row in artifact_rows}) != 50:
        fail("Artifact identifiers are not unique")
    if len({row["sha256"] for row in artifact_rows}) != 50:
        fail("Artifact SHA-256 values are not unique")
    cohort_counts = Counter(row["cohort"] for row in artifact_rows)
    if dict(sorted(cohort_counts.items())) != EXPECTED_COHORTS:
        fail(f"Unexpected cohort counts: {dict(cohort_counts)}")

    parameters = contracts["parameters"]
    if len(parameters) != 114:
        fail("Interface parameter count is not 114")
    known_hashes = {row["sha256"] for row in artifact_rows}
    if any(row["artifact_sha256"] not in known_hashes for row in parameters):
        fail("Interface ledger references an unknown artifact")

    status_counts = Counter(
        row["quantization"]["status"] for row in parameters
    )
    if status_counts != {"complete": 62, "not_quantized": 52}:
        fail(f"Unexpected interface status counts: {status_counts}")
    complete = [
        row for row in parameters
        if row["quantization"]["status"] == "complete"
    ]
    granularity_counts = Counter(
        row["quantization"]["granularity"] for row in complete
    )
    if granularity_counts != {"per_tensor": 62}:
        fail(f"Unexpected external granularity: {granularity_counts}")

    by_artifact = defaultdict(list)
    for row in complete:
        by_artifact[row["artifact_sha256"]].append(row)
    if len(by_artifact) != 32:
        fail("Artifacts with complete interface contracts is not 32")
    multiple = sum(
        len({
            row["quantization"]["contract_sha256"] for row in rows
        }) > 1
        for rows in by_artifact.values()
    )
    if multiple != 30:
        fail("Artifacts with multiple complete contracts is not 30")

    ambiguity = defaultdict(list)
    for row in complete:
        key = (
            row["direction"],
            row["dtype"],
            tuple(row["shape"]),
        )
        ambiguity[key].append(row)
    ambiguous_groups = {
        key: rows for key, rows in ambiguity.items()
        if len({
            row["quantization"]["contract_sha256"] for row in rows
        }) > 1
    }
    if len(ambiguous_groups) != 6:
        fail("Ambiguous direction/dtype/shape group count is not 6")
    int8_output = ambiguous_groups.get(("output", "INT8", (1, 1000)))
    if int8_output is None or len(int8_output) != 22:
        fail("INT8 [1,1000] output group does not contain 22 artifacts")
    if len({
        row["quantization"]["contract_sha256"] for row in int8_output
    }) != 22:
        fail("INT8 [1,1000] output contracts are not all distinct")

    by_parameter = {
        (
            row["qualified_id"],
            row["direction"],
            row["ordinal"],
        ): row
        for row in complete
    }
    output_examples = {
        "google-legacy-quantized/mobilenet-v1-1.0-224-quant": 0.5,
        "google-legacy-quantized/mobilenet-v2-1.0-224-quant":
            6.9224777817726135,
        "google-legacy-quantized/inception-v3-quant":
            5.698123946785927,
    }
    for qualified_id, expected_real in output_examples.items():
        row = by_parameter[(qualified_id, "output", 0)]
        quantization = row["quantization"]
        scale = quantization["scales"][0]
        zero_point = quantization["zero_points"][0]
        actual_real = (128 - zero_point) * scale
        if not math.isclose(
            actual_real, expected_real, rel_tol=1e-12, abs_tol=1e-12
        ):
            fail(f"q=128 output example mismatch for {qualified_id}")

    mobilenet_input = by_parameter[(
        "google-legacy-quantized/mobilenet-v2-1.0-224-quant",
        "input",
        0,
    )]["quantization"]
    efficientnet_input = by_parameter[(
        "mediapipe-public/efficientnet-lite0-int8",
        "input",
        0,
    )]["quantization"]
    mobile_scale = mobilenet_input["scales"][0]
    mobile_zero_point = mobilenet_input["zero_points"][0]
    efficient_scale = efficientnet_input["scales"][0]
    efficient_zero_point = efficientnet_input["zero_points"][0]
    mobile_unsaturated = round(1.0 / mobile_scale + mobile_zero_point)
    if mobile_unsaturated != 256:
        fail("MobileNet x=1 unsaturated code is not 256")
    mobile_saturated = min(255, max(0, mobile_unsaturated))
    efficient_read = (
        mobile_saturated - efficient_zero_point
    ) * efficient_scale
    efficient_code = round(1.0 / efficient_scale + efficient_zero_point)
    efficient_roundtrip = (
        efficient_code - efficient_zero_point
    ) * efficient_scale
    if mobile_saturated != 255 or efficient_code != 211:
        fail("Input-contract code example mismatch")
    if not math.isclose(
        efficient_read,
        1.5581861063838005,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        fail("EfficientNet interpretation of MobileNet code mismatch")
    if not math.isclose(
        efficient_roundtrip,
        1.0052813589572906,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        fail("EfficientNet x=1 round-trip mismatch")

    expected_summary = {
        "public_artifact_count": 50,
        "parameter_count": 114,
        "complete_quantized_parameter_count": 62,
        "unquantized_parameter_count": 52,
        "invalid_or_incomplete_parameter_count": 0,
        "per_tensor_parameter_count": 62,
        "per_axis_parameter_count": 0,
        "artifacts_with_complete_quantized_interface": 32,
        "artifacts_with_multiple_complete_interface_contracts": 30,
        "dtype_shape_signatures_with_multiple_affine_contracts": 6,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            fail(f"Summary mismatch for {key}")
    if corpus_summary.get("repeat_count") != 2:
        fail("Corpus repeat count is not 2")
    if corpus_summary.get("passed_public_artifact_count") != 50:
        fail("Passed public artifact count is not 50")
    if corpus_summary.get("failed_public_artifact_count") != 0:
        fail("Failed public artifact count is not 0")
    kernel_counts = corpus_summary.get(
        "kernel_quantization_granularity_counts"
    )
    if kernel_counts != {
        "not_assessed": 9,
        "not_quantized": 7,
        "per_axis_kernel_observed": 30,
        "per_tensor_kernel_observed": 4,
    }:
        fail(f"Unexpected kernel granularity counts: {kernel_counts}")

    expected_crosscheck = {
        "expected_artifact_count": 50,
        "located_artifact_count": 50,
        "verified_artifact_count": 50,
        "verified_parameter_count": 114,
        "missing_artifact_count": 0,
        "mismatch_count": 0,
    }
    for key, value in expected_crosscheck.items():
        if crosscheck.get(key) != value:
            fail(f"LiteRT cross-check mismatch for {key}")


def verify_experiments(root):
    command = [
        sys.executable,
        str(root / "scripts/check-affine-interface-mismatch.py"),
        str(root / "experiments/imagenetv2"),
    ]
    subprocess.run(command, check=True)


def verify_supplementary_results(root):
    commands = [
        [
            sys.executable,
            str(root / "scripts/test-kaggle-snapshot.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/test-kaggle-frame-stability.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/test-kaggle-acquisition-pipeline.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/test-kaggle-tflite-audit.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/test-kaggle-revision-comparison.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/check-kaggle-cohort-results.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/check-tflite-metadata-audit.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/check-tflite-metadata-flatc-crosscheck.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/check-onnx-pilot-results.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/check-supplementary-results.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/check-cyclonedx-candidate-examples.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/check-quantization-fixture-evidence.py"),
        ],
        [
            sys.executable,
            str(root / "scripts/check-affine-all-pairs.py"),
            str(root / "experiments/imagenetv2-all-pairs"),
        ],
        [
            sys.executable,
            str(root / "scripts/build-paper-tables.py"),
            "--check",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-sha256s", action="store_true")
    args = parser.parse_args()
    if args.write_sha256s:
        write_sha256s(root)
    verify_public_ledger(root)
    verify_experiments(root)
    verify_supplementary_results(root)
    verify_sha256s(root)
    print(json.dumps({
        "status": "pass",
        "artifact_count": 50,
        "interface_parameter_count": 114,
        "all_pairs_comparison_count": 6,
        "all_pairs_image_count": 1000,
    }, indent=2))


if __name__ == "__main__":
    main()
