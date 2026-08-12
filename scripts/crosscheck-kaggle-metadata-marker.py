import argparse
import hashlib
import json
from pathlib import Path


MARKER = b"TFLITE_METADATA"
SCHEMA = "tensor_quantization_metadata_study.kaggle_metadata_marker_crosscheck.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--materialization",
        type=Path,
        default=root / "data/kaggle-tflite-materialization.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=root / "data/kaggle-tflite-audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/kaggle-tflite-metadata-marker-crosscheck.json",
    )
    args = parser.parse_args()
    materialization = json.loads(
        args.materialization.read_text(encoding="utf-8")
    )
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    audit_by_digest = {
        row["artifact_sha256"]: row for row in audit["artifacts"]
    }
    rows = []
    mismatches = []
    for version in materialization["versions"]:
        for file in version["files"]:
            if file["status"] != "ASSESSED_DOWNLOAD":
                continue
            digest = file["artifact_sha256"]
            path = root / file["cache_path"]
            if sha256_file(path) != digest:
                raise ValueError(f"Cached artifact hash mismatch: {digest}")
            marker_count = path.read_bytes().count(MARKER)
            audit_status = audit_by_digest[digest]["metadata"]["status"]
            marker_status = "PRESENT" if marker_count else "ABSENT"
            row = {
                "artifact_sha256": digest,
                "marker_count": marker_count,
                "marker_status": marker_status,
                "audit_metadata_status": audit_status,
            }
            rows.append(row)
            if marker_status != audit_status:
                mismatches.append(row)
    document = {
        "schema": SCHEMA,
        "method": (
            "Exact byte search for the FlatBuffer metadata name "
            "TFLITE_METADATA; no model execution."
        ),
        "source_materialization_sha256": sha256_file(args.materialization),
        "source_audit_sha256": sha256_file(args.audit),
        "assessed_artifact_count": len(rows),
        "marker_present_count": sum(row["marker_count"] > 0 for row in rows),
        "marker_absent_count": sum(row["marker_count"] == 0 for row in rows),
        "mismatch_count": len(mismatches),
        "artifacts": sorted(rows, key=lambda row: row["artifact_sha256"]),
        "mismatches": mismatches,
    }
    args.output.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "assessed_artifact_count": len(rows),
        "marker_present_count": document["marker_present_count"],
        "marker_absent_count": document["marker_absent_count"],
        "mismatch_count": len(mismatches),
    }, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
