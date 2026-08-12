import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "tensor_quantization_metadata_study.kaggle_revision_pairs.v1"
SNAPSHOT_SCHEMA = "tensor_quantization_metadata_study.kaggle_tflite_snapshot.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tflite_paths(version):
    return {
        row["logical_path"]: row
        for row in version["files"]
        if row["is_tflite"]
    }


def build_pairs(snapshot):
    if snapshot["schema"] != SNAPSHOT_SCHEMA:
        raise ValueError("Unexpected Kaggle snapshot schema")
    if snapshot["snapshot_status"] != "COMPLETE":
        raise ValueError("Revision pairs require a complete snapshot")
    if snapshot["acquisition"]["history_scope"] != "all":
        raise ValueError("Revision pairs require history_scope=all")
    pairs = []
    for variation in snapshot["variations"]:
        versions = sorted(
            variation["versions"], key=lambda row: row["version_number"]
        )
        if len(versions) != variation["enumerated_version_count"]:
            raise ValueError(
                f"Version history is incomplete: {variation['variation_ref']}"
            )
        for old, new in zip(versions, versions[1:]):
            old_paths = tflite_paths(old)
            new_paths = tflite_paths(new)
            for logical_path in sorted(set(old_paths) | set(new_paths)):
                old_file = old_paths.get(logical_path)
                new_file = new_paths.get(logical_path)
                if old_file is None or new_file is None:
                    evidence_class = "PATH_ADDED_OR_REMOVED"
                else:
                    evidence_class = "PENDING_ARTIFACT_ASSESSMENT"
                pairs.append({
                    "variation_ref": variation["variation_ref"],
                    "logical_path": logical_path,
                    "old_version_number": old["version_number"],
                    "new_version_number": new["version_number"],
                    "old_listed_size_bytes": (
                        old_file["size_bytes"] if old_file else None
                    ),
                    "new_listed_size_bytes": (
                        new_file["size_bytes"] if new_file else None
                    ),
                    "evidence_class": evidence_class,
                })
    return pairs


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=root / "data/kaggle-tflite-snapshot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/kaggle-tflite-revision-pairs.json",
    )
    parser.add_argument(
        "--frame-stability",
        type=Path,
        default=root / "data/kaggle-tflite-frame-stability.json",
    )
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    stability = json.loads(args.frame_stability.read_text(encoding="utf-8"))
    if (
        stability.get("status") != "STABLE"
        or stability.get("after_snapshot_sha256") != sha256_file(args.snapshot)
    ):
        raise ValueError(
            "Revision pairs require a stable frame bound to this snapshot"
        )
    pairs = build_pairs(snapshot)
    result = {
        "schema": SCHEMA,
        "source_snapshot_sha256": sha256_file(args.snapshot),
        "source_frame_stability_sha256": sha256_file(args.frame_stability),
        "pair_count": len(pairs),
        "same_path_pair_count": sum(
            row["evidence_class"] == "PENDING_ARTIFACT_ASSESSMENT"
            for row in pairs
        ),
        "path_added_or_removed_count": sum(
            row["evidence_class"] == "PATH_ADDED_OR_REMOVED"
            for row in pairs
        ),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "pair_count": result["pair_count"],
        "same_path_pair_count": result["same_path_pair_count"],
        "path_added_or_removed_count": result[
            "path_added_or_removed_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
