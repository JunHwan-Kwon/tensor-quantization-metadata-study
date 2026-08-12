import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_SCHEMA = "tensor_quantization_metadata_study.kaggle_tflite_snapshot.v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def verify_raw_pages(snapshot, raw_root):
    root = (ROOT / raw_root).resolve()
    records = snapshot["raw_response_pages"]
    paths = [row["raw_response_path"] for row in records]
    require(len(paths) == len(set(paths)), f"Duplicate raw path in {raw_root}")
    disk_paths = {
        path.relative_to(root).as_posix()
        for path in root.glob("*.json")
    }
    require(set(paths) == disk_paths, f"Raw page path mismatch in {raw_root}")
    for row in records:
        path = (root / row["raw_response_path"]).resolve()
        require(path.parent == root, f"Raw page escapes root: {path}")
        document = json.loads(path.read_text(encoding="ascii"))
        require(
            canonical_sha256(document) == row["response_sha256"],
            f"Raw response hash mismatch: {path.name}",
        )


def verify_snapshot(relative, raw_root, expected_history_scope):
    snapshot = read_json(relative)
    require(snapshot["schema"] == SNAPSHOT_SCHEMA, f"Schema mismatch: {relative}")
    require(snapshot["snapshot_status"] == "COMPLETE", f"Incomplete: {relative}")
    require(
        snapshot["acquisition"]["history_scope"] == expected_history_scope,
        f"History scope mismatch: {relative}",
    )
    frame = snapshot["population_frame"]
    require(
        frame["frame_type"] == "MODEL_ID_BOUNDED_PUBLIC_REGISTRY_COHORT",
        f"Frame type mismatch: {relative}",
    )
    require(frame["model_id_min_inclusive"] == 700000, "Lower bound changed")
    require(frame["model_id_max_exclusive"] == 725000, "Upper bound changed")
    require(frame["lower_boundary_covered"], "Lower bound not covered")
    require(frame["upper_boundary_covered"], "Upper bound not covered")
    require(frame["listing_ids_strictly_descending"], "Listing order failed")
    require(
        frame["listing_record_count"]
        == frame["listing_unique_model_count"]
        + frame["identical_duplicate_model_record_count"],
        "Listing deduplication does not conserve records",
    )

    models = snapshot["models"]
    variations = snapshot["variations"]
    model_ids = [int(row["model_id"]) for row in models]
    require(len(model_ids) == len(set(model_ids)), "Selected model IDs repeat")
    require(all(700000 <= value < 725000 for value in model_ids), "ID escaped cohort")
    require(len(models) == frame["selected_model_count"], "Model count mismatch")
    require(len(models) == snapshot["summary"]["enumerated_model_count"], "Model summary mismatch")
    require(len(variations) == snapshot["summary"]["enumerated_variation_count"], "Variation summary mismatch")
    require(len({row["variation_ref"] for row in variations}) == len(variations), "Variation refs repeat")
    model_refs = {row["model_ref"] for row in models}
    require(all(row["model_ref"] in model_refs for row in variations), "Variation lacks model")

    crosschecks = Counter(
        row["embedded_instance_crosscheck_status"] for row in models
    )
    require(crosschecks["MATCH"] == 100, "Embedded instance cross-check changed")
    declared = [row for row in variations if row["declared_tflite_framework"]]
    latest = [row for row in declared if row["latest_version_contains_tflite"]]
    files = [
        file
        for variation in latest
        for version in variation["versions"]
        if version["is_latest"]
        for file in version["files"]
        if file["is_tflite"]
    ]
    require(len(declared) == snapshot["summary"]["declared_tflite_variation_count"], "Declared TFLite count mismatch")
    require(len(latest) == snapshot["summary"]["latest_tflite_variation_count"], "Latest TFLite count mismatch")
    require(len(files) == snapshot["summary"]["latest_published_tflite_file_count"], "TFLite file count mismatch")
    require(len(snapshot["subject_enumeration_failures"]) == snapshot["summary"]["subject_enumeration_failure_count"], "Failure count mismatch")
    require(len(snapshot["raw_response_pages"]) == snapshot["summary"]["raw_response_page_count"], "Raw page count mismatch")
    verify_raw_pages(snapshot, raw_root)
    return snapshot


def main():
    before = verify_snapshot(
        "data/kaggle-tflite-snapshot-a.json",
        "data/kaggle-snapshot-pages-a",
        "latest",
    )
    after = verify_snapshot(
        "data/kaggle-tflite-snapshot-b.json",
        "data/kaggle-snapshot-pages-b",
        "latest",
    )
    expected_snapshot = {
        "selected_model_count": 1966,
        "enumerated_variation_count": 1985,
        "declared_tflite_variation_count": 3,
        "latest_tflite_variation_count": 2,
        "latest_published_tflite_file_count": 9,
        "embedded_instance_crosscheck_match_count": 100,
        "subject_enumeration_failure_count": 1,
    }
    for snapshot in (before, after):
        for key, value in expected_snapshot.items():
            require(snapshot["summary"][key] == value, f"Snapshot metric changed: {key}")

    stability = read_json("data/kaggle-tflite-frame-stability.json")
    require(stability["status"] == "STABLE", "Kaggle frame is not stable")
    require(
        stability["before_snapshot_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-snapshot-a.json"),
        "Stability ledger is not bound to snapshot A",
    )
    require(
        stability["after_snapshot_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-snapshot-b.json"),
        "Stability ledger is not bound to snapshot B",
    )
    for key in (
        "added_models",
        "removed_models",
        "changed_model_enumeration_statuses",
        "added_variations",
        "removed_variations",
        "changed_variations",
    ):
        require(not stability[key], f"Stable frame has differences: {key}")

    materialization = read_json("data/kaggle-tflite-materialization.json")
    require(materialization["published_tflite_file_count"] == 9, "Materialized file count changed")
    require(materialization["status_counts"] == {"ASSESSED_DOWNLOAD": 9}, "Download coverage changed")
    require(
        materialization["source_snapshot_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-snapshot-b.json"),
        "Materialization is not bound to snapshot B",
    )
    require(
        materialization["source_frame_stability_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-frame-stability.json"),
        "Materialization is not bound to frame stability",
    )

    audit = read_json("data/kaggle-tflite-audit.json")
    expected_audit = {
        "published_file_count": 9,
        "assessed_published_file_count": 9,
        "assessed_unique_artifact_count": 9,
        "integer_affine_interface_published_file_count": 1,
        "parseable_tflite_metadata_published_file_count": 0,
        "external_parameter_count": 18,
    }
    for key, value in expected_audit.items():
        require(audit["summary"][key] == value, f"Audit metric changed: {key}")
    require(audit["summary"]["artifact_status_counts"] == {"ASSESSED": 9}, "Audit coverage changed")
    require(audit["summary"]["affine_status_counts"] == {"COMPLETE": 2, "NOT_QUANTIZED": 16}, "Affine conservation changed")
    require(
        audit["source_materialization_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-materialization.json"),
        "Audit is not bound to materialization",
    )

    crosscheck = read_json("data/kaggle-litert-interface-crosscheck.json")
    require(crosscheck["expected_artifact_count"] == 9, "Cross-check artifact count changed")
    require(crosscheck["verified_artifact_count"] == 9, "Cross-check coverage changed")
    require(crosscheck["verified_parameter_count"] == 18, "Cross-check parameter count changed")
    require(crosscheck["missing_artifact_count"] == 0, "Cross-check has missing artifacts")
    require(crosscheck["mismatch_count"] == 0, "Cross-check has mismatches")
    require(
        crosscheck["source_interface_contract_ledger_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-audit.json"),
        "Cross-check is not bound to audit",
    )
    marker = read_json(
        "data/kaggle-tflite-metadata-marker-crosscheck.json"
    )
    require(marker["assessed_artifact_count"] == 9, "Marker coverage changed")
    require(marker["marker_present_count"] == 0, "Metadata marker appeared")
    require(marker["marker_absent_count"] == 9, "Marker absence count changed")
    require(marker["mismatch_count"] == 0, "Metadata marker mismatch")
    require(
        marker["source_materialization_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-materialization.json"),
        "Marker cross-check is not bound to materialization",
    )
    require(
        marker["source_audit_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-audit.json"),
        "Marker cross-check is not bound to audit",
    )

    history_before = verify_snapshot(
        "data/kaggle-tflite-history-a.json",
        "data/kaggle-history-pages-a",
        "all",
    )
    history_after = verify_snapshot(
        "data/kaggle-tflite-history-b.json",
        "data/kaggle-history-pages-b",
        "all",
    )
    expected_history = {
        "selected_model_count": 1966,
        "enumerated_variation_count": 1985,
        "declared_tflite_variation_count": 3,
        "latest_tflite_variation_count": 2,
        "latest_published_tflite_file_count": 9,
        "embedded_instance_crosscheck_match_count": 100,
        "foreign_version_record_count": 1,
        "subject_enumeration_failure_count": 1,
        "raw_response_page_count": 308,
    }
    for snapshot in (history_before, history_after):
        for key, value in expected_history.items():
            require(
                snapshot["summary"][key] == value,
                f"Historical snapshot metric changed: {key}",
            )

    history_stability_path = ROOT / "data/kaggle-tflite-history-stability.json"
    history_stability = read_json("data/kaggle-tflite-history-stability.json")
    require(history_stability["status"] == "STABLE", "Historical frame is not stable")
    require(
        history_stability["before_snapshot_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-history-a.json"),
        "Historical stability is not bound to snapshot A",
    )
    require(
        history_stability["after_snapshot_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-history-b.json"),
        "Historical stability is not bound to snapshot B",
    )
    for key in (
        "added_models",
        "removed_models",
        "changed_model_enumeration_statuses",
        "added_variations",
        "removed_variations",
        "changed_variations",
    ):
        require(
            not history_stability[key],
            f"Stable historical frame has differences: {key}",
        )

    pairs_path = ROOT / "data/kaggle-tflite-revision-pairs.json"
    pairs = read_json("data/kaggle-tflite-revision-pairs.json")
    require(pairs["pair_count"] == 9, "Historical pair count changed")
    require(pairs["same_path_pair_count"] == 1, "Same-path pair count changed")
    require(
        pairs["path_added_or_removed_count"] == 8,
        "Path addition/removal count changed",
    )
    require(
        pairs["source_snapshot_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-history-b.json"),
        "Revision pairs are not bound to historical snapshot B",
    )
    require(
        pairs["source_frame_stability_sha256"]
        == sha256_file(history_stability_path),
        "Revision pairs are not bound to historical stability",
    )

    all_materialization_path = ROOT / "data/kaggle-tflite-materialization-all.json"
    all_materialization = read_json("data/kaggle-tflite-materialization-all.json")
    require(all_materialization["scope"] == "all", "Historical materialization scope changed")
    require(all_materialization["version_count"] == 5, "Historical version count changed")
    require(
        all_materialization["published_tflite_file_count"] == 15,
        "Historical materialized file count changed",
    )
    require(
        all_materialization["status_counts"] == {"ASSESSED_DOWNLOAD": 15},
        "Historical download coverage changed",
    )
    require(
        all_materialization["source_snapshot_sha256"]
        == sha256_file(ROOT / "data/kaggle-tflite-history-b.json"),
        "Historical materialization is not bound to snapshot B",
    )
    require(
        all_materialization["source_frame_stability_sha256"]
        == sha256_file(history_stability_path),
        "Historical materialization is not bound to stability",
    )

    all_audit_path = ROOT / "data/kaggle-tflite-audit-all-versions.json"
    all_audit = read_json("data/kaggle-tflite-audit-all-versions.json")
    expected_all_audit = {
        "published_file_count": 15,
        "assessed_published_file_count": 15,
        "assessed_unique_artifact_count": 15,
        "integer_affine_interface_published_file_count": 5,
        "parseable_tflite_metadata_published_file_count": 0,
        "external_parameter_count": 30,
    }
    for key, value in expected_all_audit.items():
        require(
            all_audit["summary"][key] == value,
            f"Historical audit metric changed: {key}",
        )
    require(
        all_audit["summary"]["artifact_status_counts"] == {"ASSESSED": 15},
        "Historical audit coverage changed",
    )
    require(
        all_audit["summary"]["affine_status_counts"]
        == {"COMPLETE": 10, "NOT_QUANTIZED": 20},
        "Historical affine conservation changed",
    )
    require(
        all_audit["source_materialization_sha256"]
        == sha256_file(all_materialization_path),
        "Historical audit is not bound to materialization",
    )

    comparison = read_json("data/kaggle-tflite-revision-comparison.json")
    require(comparison["comparison_count"] == 9, "Revision comparison count changed")
    require(
        comparison["classification_counts"]
        == {"INTERFACE_UNCHANGED": 1, "PATH_ADDED_OR_REMOVED": 8},
        "Revision classifications changed",
    )
    require(
        comparison["source_pair_ledger_sha256"] == sha256_file(pairs_path),
        "Revision comparison is not bound to pair ledger",
    )
    require(
        comparison["source_audit_sha256"] == sha256_file(all_audit_path),
        "Revision comparison is not bound to historical audit",
    )
    unchanged = [
        row for row in comparison["comparisons"]
        if row["classification"] == "INTERFACE_UNCHANGED"
    ]
    require(len(unchanged) == 1, "Expected one unchanged-interface revision")
    unchanged = unchanged[0]
    require(
        unchanged["old_artifact_sha256"] != unchanged["new_artifact_sha256"],
        "Same-path revision artifact hashes unexpectedly match",
    )
    require(not unchanged["dtype_shape_deltas"], "Unexpected dtype/shape revision")
    require(not unchanged["affine_deltas"], "Unexpected affine revision")
    initializer = unchanged["initializer_comparison"]
    require(initializer["old_initializer_count"] == 107, "Old initializer count changed")
    require(initializer["new_initializer_count"] == 107, "New initializer count changed")
    require(initializer["shared_initializer_count"] == 107, "Shared initializer count changed")
    require(
        initializer["changed_shared_initializer_count"] == 106,
        "Changed initializer count changed",
    )
    require(not initializer["sets_and_bytes_identical"], "Initializer change disappeared")
    require(unchanged["causal_attribution"] == "NOT_INFERRED", "Causality was inferred")

    all_crosscheck = read_json(
        "data/kaggle-litert-interface-crosscheck-all-versions.json"
    )
    require(all_crosscheck["expected_artifact_count"] == 15, "Historical cross-check count changed")
    require(all_crosscheck["verified_artifact_count"] == 15, "Historical cross-check coverage changed")
    require(all_crosscheck["verified_parameter_count"] == 30, "Historical parameter coverage changed")
    require(all_crosscheck["missing_artifact_count"] == 0, "Historical cross-check has missing artifacts")
    require(all_crosscheck["mismatch_count"] == 0, "Historical cross-check has mismatches")
    require(
        all_crosscheck["source_interface_contract_ledger_sha256"]
        == sha256_file(all_audit_path),
        "Historical cross-check is not bound to audit",
    )

    all_marker = read_json(
        "data/kaggle-tflite-metadata-marker-crosscheck-all-versions.json"
    )
    require(all_marker["assessed_artifact_count"] == 15, "Historical marker coverage changed")
    require(all_marker["marker_present_count"] == 0, "Historical metadata marker appeared")
    require(all_marker["marker_absent_count"] == 15, "Historical marker absence changed")
    require(all_marker["mismatch_count"] == 0, "Historical marker mismatch")
    require(
        all_marker["source_materialization_sha256"]
        == sha256_file(all_materialization_path),
        "Historical marker check is not bound to materialization",
    )
    require(
        all_marker["source_audit_sha256"] == sha256_file(all_audit_path),
        "Historical marker check is not bound to audit",
    )
    print(json.dumps({
        "status": "pass",
        "selected_model_count": 1966,
        "variation_count": 1985,
        "published_tflite_file_count": 9,
        "historical_published_tflite_file_count": 15,
        "same_path_revision_pair_count": 1,
        "observed_affine_revision_count": 0,
        "raw_response_page_count": (
            len(before["raw_response_pages"])
            + len(after["raw_response_pages"])
            + len(history_before["raw_response_pages"])
            + len(history_after["raw_response_pages"])
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
