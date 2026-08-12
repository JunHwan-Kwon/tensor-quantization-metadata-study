import hashlib
import json
from pathlib import Path


SCHEMA = "tensor_quantization_metadata_study.tflite_metadata_flatc_crosscheck.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message):
    raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    audit_path = root / "data/tflite-metadata-audit.json"
    path = root / "data/tflite-metadata-flatc-crosscheck.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["schema"] != SCHEMA:
        fail("Unexpected flatc cross-check schema")
    if result["source_audit_sha256"] != sha256_file(audit_path):
        fail("flatc cross-check is not bound to the current metadata audit")
    if result["metadata_schema_sha256"] != (
        "2d3386ba124690ba1195bfc1d51ac814843bd675a7c845afab7c001c7891449e"
    ):
        fail("Unexpected metadata schema SHA-256")
    if result["artifact_count"] != len(result["results"]):
        fail("flatc artifact count mismatch")
    if result["artifact_count"] != 20:
        fail("flatc cross-check did not cover 20 metadata-bearing artifacts")
    if result["external_parameter_count"] != sum(
        row["external_parameter_count"] for row in result["results"]
    ):
        fail("flatc external-parameter count mismatch")
    if result["external_parameter_count"] != 49:
        fail("flatc cross-check did not cover 49 external parameters")
    if result["mismatch_count"] != sum(
        row["mismatch_count"] for row in result["results"]
    ):
        fail("flatc mismatch count is not conserved")
    if result["mismatch_count"] != 0:
        fail("flatc and manual metadata decoders disagree")
    if any(row["mismatches"] for row in result["results"]):
        fail("A flatc result contains uncounted mismatches")
    print(json.dumps({
        "status": "pass",
        "artifact_count": result["artifact_count"],
        "external_parameter_count": result["external_parameter_count"],
        "mismatch_count": result["mismatch_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
