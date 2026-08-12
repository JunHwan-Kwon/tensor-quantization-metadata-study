import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def csv_payload(fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def cohort_table(metadata):
    rows = []
    for cohort, values in sorted(metadata["summary"]["cohort_results"].items()):
        rows.append({"cohort": cohort, **values})
    fields = [
        "cohort",
        "artifact_count",
        "input_parameter_count",
        "quantized_input_count",
        "parseable_tflite_metadata_count",
        "mapped_input_metadata_count",
        "valid_input_normalization_count",
    ]
    return csv_payload(fields, rows)


def interface_table(artifacts, contracts, summary, metadata):
    complete = [
        row for row in contracts["parameters"]
        if row["quantization"]["status"] == "complete"
    ]
    signatures = defaultdict(set)
    signature_artifacts = defaultdict(set)
    for row in complete:
        key = (row["direction"], row["dtype"], tuple(row["shape"]))
        signatures[key].add(row["quantization"]["contract_sha256"])
        signature_artifacts[key].add(row["artifact_sha256"])
    multi_artifact = {
        key for key, values in signature_artifacts.items() if len(values) > 1
    }
    ambiguous = {
        key for key in multi_artifact if len(signatures[key]) > 1
    }
    values = [
        ("curated_artifacts", len(artifacts["artifacts"]), 50, "artifact"),
        (
            "external_parameters",
            len(contracts["parameters"]),
            114,
            "external_parameter",
        ),
        (
            "complete_affine_external_parameters",
            len(complete),
            len(contracts["parameters"]),
            "external_parameter",
        ),
        (
            "unquantized_external_parameters",
            sum(
                row["quantization"]["status"] == "not_quantized"
                for row in contracts["parameters"]
            ),
            len(contracts["parameters"]),
            "external_parameter",
        ),
        (
            "per_tensor_affine_external_parameters",
            sum(
                row["quantization"]["granularity"] == "per_tensor"
                for row in complete
            ),
            len(complete),
            "complete_affine_external_parameter",
        ),
        (
            "per_axis_affine_external_parameters",
            sum(
                row["quantization"]["granularity"] == "per_axis"
                for row in complete
            ),
            len(complete),
            "complete_affine_external_parameter",
        ),
        (
            "artifacts_with_complete_affine_interface",
            summary["artifacts_with_complete_quantized_interface"],
            len(artifacts["artifacts"]),
            "artifact",
        ),
        (
            "multi_artifact_signature_groups_with_multiple_contracts",
            len(ambiguous),
            len(multi_artifact),
            "direction_dtype_shape_group",
        ),
        (
            "artifacts_with_parseable_tflite_metadata",
            metadata["summary"]["artifact_with_parseable_tflite_metadata_count"],
            metadata["summary"]["artifact_count"],
            "artifact",
        ),
        (
            "mapped_image_inputs_with_valid_normalization",
            metadata["summary"]["mapped_image_input_with_valid_normalization_count"],
            metadata["summary"]["mapped_image_input_count"],
            "mapped_image_input",
        ),
        (
            "quantized_inputs_with_valid_normalization",
            metadata["summary"]["quantized_input_with_valid_normalization_count"],
            metadata["summary"]["quantized_input_count"],
            "quantized_input",
        ),
    ]
    rows = [
        {
            "metric": metric,
            "count": count,
            "denominator": denominator,
            "analysis_unit": unit,
            "proportion": round(count / denominator, 10) if denominator else "",
        }
        for metric, count, denominator, unit in values
    ]
    return csv_payload(
        ["metric", "count", "denominator", "analysis_unit", "proportion"],
        rows,
    )


def all_pairs_table(measurement):
    aliases = {
        row["contract_id"]: ";".join(sorted(row["model_aliases"]))
        for row in measurement["contract_catalog"]
    }
    rows = []
    for comparison in sorted(
        measurement["comparisons"],
        key=lambda row: (row["target_model_id"], row["source_contract_id"]),
    ):
        metrics = comparison["metrics"]
        delta = metrics["paired_top1_accuracy_delta"]
        rows.append({
            "target_model_id": comparison["target_model_id"],
            "target_contract_id": comparison["target_contract_id"],
            "source_contract_id": comparison["source_contract_id"],
            "source_contract_models": aliases[comparison["source_contract_id"]],
            "image_count": metrics["image_count"],
            "baseline_top1": metrics["correct_top1_accuracy"],
            "substituted_top1": metrics["wrong_top1_accuracy"],
            "paired_delta_percentage_points": round(100 * delta["point"], 6),
            "bootstrap_95_lower_percentage_points": round(
                100 * delta["lower_95"], 6
            ),
            "bootstrap_95_upper_percentage_points": round(
                100 * delta["upper_95"], 6
            ),
            "input_clip_fraction": round(
                metrics["wrong_input_clip_fraction"], 10
            ),
            "mcnemar_exact_two_sided_p": (
                metrics["mcnemar_exact"]["two_sided_p_value"]
            ),
            "holm_adjusted_p": comparison["mcnemar_holm"]["adjusted_p_value"],
            "holm_reject_0_05": comparison["mcnemar_holm"]["reject_at_0_05"],
        })
    return csv_payload(list(rows[0]), rows)


def format_pilot_table(artifacts, contracts, onnx):
    tflite_status = Counter(
        row["quantization"]["status"] for row in contracts["parameters"]
    )
    onnx_status = onnx["summary"]["contract_status_counts"]
    rows = [
        {
            "format": "TFLite",
            "study_design": "curated_benchmark",
            "artifact_count": len(artifacts["artifacts"]),
            "external_parameter_count": len(contracts["parameters"]),
            "complete_affine_external_parameter_count": tflite_status["complete"],
            "not_quantized_external_parameter_count": tflite_status["not_quantized"],
            "claim_boundary": "Criterion-based corpus; not a prevalence estimate.",
        },
        {
            "format": "ONNX",
            "study_design": "targeted_pilot",
            "artifact_count": len(onnx["artifacts"]),
            "external_parameter_count": onnx["summary"]["parameter_count"],
            "complete_affine_external_parameter_count": onnx_status.get(
                "COMPLETE", 0
            ),
            "not_quantized_external_parameter_count": onnx_status.get(
                "NOT_QUANTIZED", 0
            ),
            "claim_boundary": "Targeted pilot; not a prevalence estimate.",
        },
    ]
    return csv_payload(list(rows[0]), rows)


def kaggle_cohort_table(
    snapshot, stability, materialization, audit, crosscheck, marker
):
    snapshot_path = ROOT / "data/kaggle-tflite-snapshot-b.json"
    stability_path = ROOT / "data/kaggle-tflite-frame-stability.json"
    materialization_path = ROOT / "data/kaggle-tflite-materialization.json"
    audit_path = ROOT / "data/kaggle-tflite-audit.json"
    if stability["status"] != "STABLE":
        raise ValueError("Kaggle cohort frame is not stable")
    if stability["after_snapshot_sha256"] != sha256_file(snapshot_path):
        raise ValueError("Kaggle stability ledger is not bound to snapshot B")
    if materialization["source_snapshot_sha256"] != sha256_file(snapshot_path):
        raise ValueError("Kaggle materialization is not bound to snapshot B")
    if materialization["source_frame_stability_sha256"] != sha256_file(
        stability_path
    ):
        raise ValueError("Kaggle materialization is not bound to stability")
    if audit["source_materialization_sha256"] != sha256_file(
        materialization_path
    ):
        raise ValueError("Kaggle audit is not bound to materialization")
    if crosscheck["source_interface_contract_ledger_sha256"] != sha256_file(
        audit_path
    ):
        raise ValueError("Kaggle LiteRT cross-check is not bound to audit")
    if crosscheck["mismatch_count"] or crosscheck["missing_artifact_count"]:
        raise ValueError("Kaggle LiteRT cross-check is incomplete")
    if marker["mismatch_count"] or marker["marker_present_count"]:
        raise ValueError("Kaggle metadata marker cross-check failed")
    if marker["source_materialization_sha256"] != sha256_file(
        materialization_path
    ) or marker["source_audit_sha256"] != sha256_file(audit_path):
        raise ValueError("Kaggle metadata marker cross-check is not source-bound")

    declared = [
        row for row in snapshot["variations"]
        if row["declared_tflite_framework"]
    ]
    assessed_declared = [
        row for row in declared
        if row["versions"]
        and row["versions"][-1]["file_enumeration_status"] == "ASSESSED"
    ]
    row = {
        "frame_type": snapshot["population_frame"]["frame_type"],
        "model_id_min_inclusive": snapshot["population_frame"][
            "model_id_min_inclusive"
        ],
        "model_id_max_exclusive": snapshot["population_frame"][
            "model_id_max_exclusive"
        ],
        "stable_two_crawl_frame": True,
        "selected_model_count": snapshot["summary"]["selected_model_count"],
        "variation_count": snapshot["summary"]["enumerated_variation_count"],
        "declared_tflite_variation_count": len(declared),
        "file_enumerable_tflite_variation_count": len(assessed_declared),
        "unassessed_tflite_variation_count": len(declared) - len(assessed_declared),
        "published_tflite_file_count": audit["summary"]["published_file_count"],
        "assessed_unique_artifact_count": audit["summary"][
            "assessed_unique_artifact_count"
        ],
        "integer_affine_interface_file_count": audit["summary"][
            "integer_affine_interface_published_file_count"
        ],
        "parseable_tflite_metadata_file_count": audit["summary"][
            "parseable_tflite_metadata_published_file_count"
        ],
        "tflite_metadata_marker_present_file_count": marker[
            "marker_present_count"
        ],
        "embedded_instance_crosscheck_match_count": snapshot["summary"][
            "embedded_instance_crosscheck_match_count"
        ],
        "litert_interface_crosscheck_mismatch_count": crosscheck[
            "mismatch_count"
        ],
        "claim_boundary": (
            "Descriptive census of the frozen identifier cohort; not a "
            "Kaggle-wide prevalence estimate."
        ),
    }
    return csv_payload(list(row), [row])


def kaggle_revision_table(pairs, audit, comparison, crosscheck, marker):
    pairs_path = ROOT / "data/kaggle-tflite-revision-pairs.json"
    audit_path = ROOT / "data/kaggle-tflite-audit-all-versions.json"
    materialization_path = ROOT / "data/kaggle-tflite-materialization-all.json"
    history_snapshot_path = ROOT / "data/kaggle-tflite-history-b.json"
    history_stability_path = ROOT / "data/kaggle-tflite-history-stability.json"
    if pairs["source_snapshot_sha256"] != sha256_file(history_snapshot_path):
        raise ValueError("Kaggle revision pairs are not bound to snapshot")
    if pairs["source_frame_stability_sha256"] != sha256_file(
        history_stability_path
    ):
        raise ValueError("Kaggle revision pairs are not bound to stability")
    if audit["source_materialization_sha256"] != sha256_file(
        materialization_path
    ):
        raise ValueError("Historical audit is not bound to materialization")
    if comparison["source_pair_ledger_sha256"] != sha256_file(pairs_path):
        raise ValueError("Kaggle revision comparison is not bound to pairs")
    if comparison["source_audit_sha256"] != sha256_file(audit_path):
        raise ValueError("Kaggle revision comparison is not bound to audit")
    if crosscheck["source_interface_contract_ledger_sha256"] != sha256_file(
        audit_path
    ):
        raise ValueError("Historical LiteRT cross-check is not bound to audit")
    if crosscheck["mismatch_count"] or crosscheck["missing_artifact_count"]:
        raise ValueError("Historical LiteRT cross-check is incomplete")
    if marker["source_materialization_sha256"] != sha256_file(
        materialization_path
    ) or marker["source_audit_sha256"] != sha256_file(audit_path):
        raise ValueError("Historical metadata marker check is not source-bound")
    if marker["mismatch_count"]:
        raise ValueError("Historical metadata marker check has mismatches")
    if comparison["comparison_count"] != pairs["pair_count"]:
        raise ValueError("Kaggle revision comparison does not conserve pairs")

    rows = []
    for result in comparison["comparisons"]:
        initializer = result.get("initializer_comparison", {})
        old_hash = result.get("old_artifact_sha256", "")
        new_hash = result.get("new_artifact_sha256", "")
        dtype_shape_deltas = result.get("dtype_shape_deltas")
        affine_deltas = result.get("affine_deltas")
        rows.append({
            "variation_ref": result["variation_ref"],
            "logical_path": result["logical_path"],
            "old_version_number": result["old_version_number"],
            "new_version_number": result["new_version_number"],
            "classification": result["classification"],
            "old_listed_size_bytes": result.get("old_listed_size_bytes", ""),
            "new_listed_size_bytes": result.get("new_listed_size_bytes", ""),
            "old_artifact_sha256": old_hash,
            "new_artifact_sha256": new_hash,
            "artifact_hash_changed": (
                old_hash != new_hash if old_hash and new_hash else ""
            ),
            "dtype_shape_delta_count": (
                len(dtype_shape_deltas)
                if dtype_shape_deltas is not None else ""
            ),
            "affine_delta_count": (
                len(affine_deltas) if affine_deltas is not None else ""
            ),
            "shared_initializer_count": initializer.get(
                "shared_initializer_count", ""
            ),
            "changed_shared_initializer_count": initializer.get(
                "changed_shared_initializer_count", ""
            ),
            "initializer_sets_and_bytes_identical": initializer.get(
                "sets_and_bytes_identical", ""
            ),
            "causal_attribution": result.get("causal_attribution", ""),
        })
    return csv_payload(list(rows[0]), rows)


def claim_register(snapshot, audit, revision_comparison):
    unchanged_revisions = [
        row for row in revision_comparison["comparisons"]
        if row["classification"] == "INTERFACE_UNCHANGED"
    ]
    if len(unchanged_revisions) != 1:
        raise ValueError("Expected exactly one unchanged-interface revision")
    unchanged_revision = unchanged_revisions[0]
    initializer = unchanged_revision["initializer_comparison"]
    if unchanged_revision["dtype_shape_deltas"] or unchanged_revision["affine_deltas"]:
        raise ValueError("Unchanged-interface revision contains interface deltas")
    if unchanged_revision["old_artifact_sha256"] == unchanged_revision["new_artifact_sha256"]:
        raise ValueError("Unchanged-interface revision is byte-identical")
    rows = [
        {
            "claim_id": "C01",
            "status": "SUPPORTED",
            "evidence_grade": "OBSERVED_CROSS_CHECKED",
            "scope": "curated_50_file_tflite_benchmark",
            "claim": "62 of 114 external parameters carried complete affine contracts.",
            "primary_source": "data/interface-contracts.json;data/litert-interface-crosscheck.json",
            "excluded_inference": "No TFLite ecosystem prevalence estimate.",
        },
        {
            "claim_id": "C02",
            "status": "SUPPORTED",
            "evidence_grade": "DERIVED_STATIC",
            "scope": "curated_50_file_tflite_benchmark",
            "claim": "Six multi-artifact direction/dtype/shape groups contained multiple affine contracts.",
            "primary_source": "data/interface-summary.json",
            "excluded_inference": "No deployed harness mismatch prevalence estimate.",
        },
        {
            "claim_id": "C03",
            "status": "SUPPORTED",
            "evidence_grade": "CONTROLLED_PAIRED_EXPERIMENT",
            "scope": "three_models_and_1000_selected_imagenetv2_images",
            "claim": "Two of six substitutions remained significant after Holm correction; both clipped at least 31.9% of encoded input elements.",
            "primary_source": "experiments/imagenetv2-all-pairs/measurement.json",
            "excluded_inference": "No claim that every mismatch reduces accuracy.",
        },
        {
            "claim_id": "C04",
            "status": "SUPPORTED",
            "evidence_grade": "OBSERVED_CROSS_CHECKED",
            "scope": "curated_50_file_tflite_benchmark",
            "claim": "20 artifacts contained parseable TFLITE_METADATA; all were in the MediaPipe subcohort.",
            "primary_source": "data/tflite-metadata-audit.json;data/tflite-metadata-flatc-crosscheck.json",
            "excluded_inference": "No claim that metadata describes the complete preprocessing pipeline.",
        },
        {
            "claim_id": "C05",
            "status": "SUPPORTED",
            "evidence_grade": "TARGETED_FORMAT_PILOT",
            "scope": "15_file_onnx_pilot",
            "claim": "All 39 external ONNX parameters in the pilot were not quantized, although 10 artifacts contained internal quantization patterns.",
            "primary_source": "data/onnx-pilot-results.json",
            "excluded_inference": "No ONNX ecosystem prevalence estimate.",
        },
        {
            "claim_id": "C06",
            "status": "SUPPORTED",
            "evidence_grade": "OBSERVED_REVISION_PAIR",
            "scope": "one_same_repository_same_path_onnx_revision_pair",
            "claim": "Four external parameters changed dtype while 131 serialized initializers remained byte-identical.",
            "primary_source": "data/onnx-revision-comparison.json",
            "excluded_inference": "Not an affine revision and no converter causality claim.",
        },
        {
            "claim_id": "C07",
            "status": "SUPPORTED",
            "evidence_grade": "OBSERVED_STABLE_COHORT_CROSS_CHECKED",
            "scope": "kaggle_model_id_700000_724999_registry_cohort",
            "claim": (
                f"The frozen cohort contained {snapshot['summary']['selected_model_count']} "
                f"models and {snapshot['summary']['enumerated_variation_count']} "
                "variations. Three variations were declared as TF Lite; two "
                f"enumerable variations yielded {audit['summary']['published_file_count']} "
                "assessed TFLite files, one with a complete integer-affine "
                "external interface and none with parseable TFLITE_METADATA."
            ),
            "primary_source": (
                "data/kaggle-tflite-frame-stability.json;"
                "data/kaggle-tflite-audit.json;"
                "data/kaggle-litert-interface-crosscheck.json;"
                "data/kaggle-tflite-metadata-marker-crosscheck.json"
            ),
            "excluded_inference": (
                "No Kaggle-wide prevalence estimate; one declared TF Lite "
                "variation was not publicly enumerable."
            ),
        },
        {
            "claim_id": "C08",
            "status": "NOT_ASSESSED",
            "evidence_grade": "NOT_ASSESSABLE",
            "scope": "deployed_systems",
            "claim": "Prevalence of application/model affine-contract mismatches.",
            "primary_source": "docs/paper-claim-boundary.md",
            "excluded_inference": "Requires independently observed harness contracts.",
        },
        {
            "claim_id": "C09",
            "status": "NOT_ASSESSED",
            "evidence_grade": "NOT_ASSESSABLE",
            "scope": "kaggle_wide_or_global_tflite_ecosystem",
            "claim": "Kaggle-wide or global prevalence of integer interfaces or missing preprocessing metadata.",
            "primary_source": "docs/paper-claim-boundary.md",
            "excluded_inference": "The completed run is an identifier-bounded registry cohort, not a probability sample of the full ecosystem.",
        },
        {
            "claim_id": "C10",
            "status": "SUPPORTED",
            "evidence_grade": "OBSERVED_REVISION_PAIR_CROSS_CHECKED",
            "scope": "one_same_variation_same_path_tflite_revision_pair",
            "claim": (
                "In the only assessable same-path adjacent-version TFLite "
                f"pair, the artifact hash and {initializer['changed_shared_initializer_count']} "
                f"of {initializer['shared_initializer_count']} shared serialized "
                "initializers changed while the external dtype, shape, scale, "
                "and zero-point contract remained unchanged. "
                f"{revision_comparison['classification_counts']['PATH_ADDED_OR_REMOVED']} "
                "other adjacent-version path events were additions or removals."
            ),
            "primary_source": (
                "data/kaggle-tflite-history-stability.json;"
                "data/kaggle-tflite-revision-pairs.json;"
                "data/kaggle-tflite-audit-all-versions.json;"
                "data/kaggle-tflite-revision-comparison.json;"
                "data/kaggle-litert-interface-crosscheck-all-versions.json"
            ),
            "excluded_inference": (
                "No affine revision was observed; no converter causality or "
                "revision-frequency inference from one same-path pair."
            ),
        },
    ]
    if revision_comparison["classification_counts"] != {
        "INTERFACE_UNCHANGED": 1,
        "PATH_ADDED_OR_REMOVED": 8,
    }:
        raise ValueError("Kaggle revision claim no longer matches evidence")
    return csv_payload(list(rows[0]), rows)


def build_outputs():
    artifacts = read_json("data/artifacts.json")
    contracts = read_json("data/interface-contracts.json")
    summary = read_json("data/interface-summary.json")
    metadata = read_json("data/tflite-metadata-audit.json")
    onnx = read_json("data/onnx-pilot-results.json")
    all_pairs = read_json("experiments/imagenetv2-all-pairs/measurement.json")
    kaggle_snapshot = read_json("data/kaggle-tflite-snapshot-b.json")
    kaggle_stability = read_json("data/kaggle-tflite-frame-stability.json")
    kaggle_materialization = read_json(
        "data/kaggle-tflite-materialization.json"
    )
    kaggle_audit = read_json("data/kaggle-tflite-audit.json")
    kaggle_crosscheck = read_json(
        "data/kaggle-litert-interface-crosscheck.json"
    )
    kaggle_marker = read_json(
        "data/kaggle-tflite-metadata-marker-crosscheck.json"
    )
    kaggle_revision_pairs = read_json(
        "data/kaggle-tflite-revision-pairs.json"
    )
    kaggle_all_audit = read_json(
        "data/kaggle-tflite-audit-all-versions.json"
    )
    kaggle_revision_comparison = read_json(
        "data/kaggle-tflite-revision-comparison.json"
    )
    kaggle_all_crosscheck = read_json(
        "data/kaggle-litert-interface-crosscheck-all-versions.json"
    )
    kaggle_all_marker = read_json(
        "data/kaggle-tflite-metadata-marker-crosscheck-all-versions.json"
    )

    outputs = {
        "tables/curated-cohorts.csv": cohort_table(metadata),
        "tables/external-interface-summary.csv": interface_table(
            artifacts, contracts, summary, metadata
        ),
        "tables/affine-contract-substitutions.csv": all_pairs_table(all_pairs),
        "tables/format-pilot-summary.csv": format_pilot_table(
            artifacts, contracts, onnx
        ),
        "tables/kaggle-identifier-cohort.csv": kaggle_cohort_table(
            kaggle_snapshot,
            kaggle_stability,
            kaggle_materialization,
            kaggle_audit,
            kaggle_crosscheck,
            kaggle_marker,
        ),
        "tables/kaggle-revision-comparisons.csv": kaggle_revision_table(
            kaggle_revision_pairs,
            kaggle_all_audit,
            kaggle_revision_comparison,
            kaggle_all_crosscheck,
            kaggle_all_marker,
        ),
        "claim-register.csv": claim_register(
            kaggle_snapshot,
            kaggle_audit,
            kaggle_revision_comparison,
        ),
    }
    sources = [
        "data/artifacts.json",
        "data/interface-contracts.json",
        "data/interface-summary.json",
        "data/litert-interface-crosscheck.json",
        "data/tflite-metadata-audit.json",
        "data/tflite-metadata-flatc-crosscheck.json",
        "data/onnx-pilot-results.json",
        "data/onnx-revision-comparison.json",
        "data/kaggle-tflite-snapshot-a.json",
        "data/kaggle-tflite-snapshot-b.json",
        "data/kaggle-tflite-frame-stability.json",
        "data/kaggle-tflite-materialization.json",
        "data/kaggle-tflite-audit.json",
        "data/kaggle-litert-interface-crosscheck.json",
        "data/kaggle-tflite-metadata-marker-crosscheck.json",
        "data/kaggle-tflite-history-a.json",
        "data/kaggle-tflite-history-b.json",
        "data/kaggle-tflite-history-stability.json",
        "data/kaggle-tflite-revision-pairs.json",
        "data/kaggle-tflite-materialization-all.json",
        "data/kaggle-tflite-audit-all-versions.json",
        "data/kaggle-tflite-revision-comparison.json",
        "data/kaggle-litert-interface-crosscheck-all-versions.json",
        "data/kaggle-tflite-metadata-marker-crosscheck-all-versions.json",
        "docs/paper-claim-boundary.md",
        "docs/research-population-protocol.md",
        "experiments/imagenetv2-all-pairs/measurement.json",
        "scripts/build-paper-tables.py",
    ]
    provenance = {
        "schema": "tensor_quantization_metadata_study.paper_tables.v1",
        "sources": {path: sha256_file(ROOT / path) for path in sources},
        "outputs": {
            path: hashlib.sha256(payload).hexdigest()
            for path, payload in outputs.items()
        },
    }
    outputs["table-provenance.json"] = (
        json.dumps(provenance, indent=2) + "\n"
    ).encode("utf-8")
    return outputs


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()

    if args.write:
        for relative, payload in outputs.items():
            path = OUTPUT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    else:
        for relative, payload in outputs.items():
            path = OUTPUT / relative
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"Generated paper output is stale: {relative}")
    print(json.dumps({
        "status": "pass",
        "mode": "write" if args.write else "check",
        "output_count": len(outputs),
    }, indent=2))


if __name__ == "__main__":
    main()
