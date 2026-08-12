import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "tensor_quantization_metadata_study.kaggle_frame_stability.v1"
SNAPSHOT_SCHEMA = "tensor_quantization_metadata_study.kaggle_tflite_snapshot.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_projection(snapshot):
    if snapshot["schema"] != SNAPSHOT_SCHEMA:
        raise ValueError("Unexpected Kaggle snapshot schema")
    if snapshot["snapshot_status"] != "COMPLETE":
        raise ValueError("Frame comparison requires complete snapshots")
    population = snapshot.get("population_frame")
    if not isinstance(population, dict):
        raise ValueError("Snapshot is missing its population frame")
    stable_population = {
        "frame_type": population.get("frame_type"),
        "history_scope": snapshot.get("acquisition", {}).get(
            "history_scope", "UNKNOWN"
        ),
        "model_id_min_inclusive": population.get("model_id_min_inclusive"),
        "model_id_max_exclusive": population.get("model_id_max_exclusive"),
        "lower_boundary_covered": population.get("lower_boundary_covered"),
        "upper_boundary_covered": population.get("upper_boundary_covered"),
        "selected_model_count": population.get("selected_model_count"),
    }
    rows = {}
    for variation in snapshot["variations"]:
        latest_count = sum(row["is_latest"] for row in variation["versions"])
        if latest_count > 1:
            raise ValueError(
                f"Expected at most one latest version: {variation['variation_ref']}"
            )
        rows[variation["variation_ref"]] = {
            "current_version_status": variation.get(
                "current_version_status", "UNKNOWN"
            ),
            "version_enumeration_status": variation.get(
                "version_enumeration_status", "UNKNOWN"
            ),
            "versions": [
                {
                    "version_number": version["version_number"],
                    "is_latest": version["is_latest"],
                    "file_enumeration_status": version.get(
                        "file_enumeration_status", "ASSESSED"
                    ),
                    "files": sorted(
                        (row["logical_path"], row["size_bytes"])
                        for row in version["files"]
                    ),
                }
                for version in sorted(
                    variation["versions"],
                    key=lambda row: row["version_number"],
                )
            ],
        }
    return {
        "population_frame": stable_population,
        "models": {
            row["model_ref"]: row.get(
                "instance_enumeration_status", "ASSESSED"
            )
            for row in snapshot["models"]
        },
        "variations": rows,
    }


def compare_frames(before, after):
    first = frame_projection(before)
    second = frame_projection(after)
    first_population = first["population_frame"]
    second_population = second["population_frame"]
    first_models = first["models"]
    second_models = second["models"]
    first_model_keys = set(first_models)
    second_model_keys = set(second_models)
    common_models = first_model_keys & second_model_keys
    changed_models = sorted(
        key for key in common_models
        if first_models[key] != second_models[key]
    )
    first = first["variations"]
    second = second["variations"]
    first_keys = set(first)
    second_keys = set(second)
    common = first_keys & second_keys
    changed = sorted(key for key in common if first[key] != second[key])
    return {
        "status": (
            "STABLE"
            if (
                first_population == second_population
                and first == second
                and first_models == second_models
            )
            else "CHANGED_REQUIRES_NEW_ENUMERATION"
        ),
        "population_frame_changed": first_population != second_population,
        "before_population_frame": first_population,
        "after_population_frame": second_population,
        "before_variation_count": len(first),
        "after_variation_count": len(second),
        "added_models": sorted(second_model_keys - first_model_keys),
        "removed_models": sorted(first_model_keys - second_model_keys),
        "changed_model_enumeration_statuses": [{
            "model_ref": key,
            "before": first_models[key],
            "after": second_models[key],
        } for key in changed_models],
        "added_variations": sorted(second_keys - first_keys),
        "removed_variations": sorted(first_keys - second_keys),
        "changed_variations": [{
            "variation_ref": key,
            "before": first[key],
            "after": second[key],
        } for key in changed],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--before",
        type=Path,
        default=root / "data/kaggle-tflite-snapshot-a.json",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=root / "data/kaggle-tflite-snapshot-b.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/kaggle-tflite-frame-stability.json",
    )
    args = parser.parse_args()
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    comparison = compare_frames(before, after)
    document = {
        "schema": SCHEMA,
        "before_snapshot_sha256": sha256_file(args.before),
        "after_snapshot_sha256": sha256_file(args.after),
        **comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["status"] == "STABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
