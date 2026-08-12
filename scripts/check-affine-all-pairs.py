import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "scripts" / "check-affine-interface-mismatch.py"
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "affine_interface_checker", CHECKER_PATH
)
BASE = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(BASE)
SCHEMA = "tensor_quantization_metadata_study.affine_all_pairs.v1"


def fail(message):
    raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_source(path, expected_sha256):
    if path.is_file() and sha256_file(path) == expected_sha256:
        return path
    recorded = (
        ROOT / "data/recorded-run-sources" / f"{expected_sha256}.json"
    )
    if recorded.is_file() and sha256_file(recorded) == expected_sha256:
        return recorded
    fail(
        "No current or recorded-run source matches provenance hash: "
        f"{path} / {expected_sha256}"
    )


def sha256_json(value):
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def close(actual, expected, label, tolerance=1e-12):
    if not math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    ):
        fail(f"{label}: {actual!r} != {expected!r}")


def read_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            rows.append({
                "target_model_id": raw["target_model_id"],
                "source_contract_id": raw["source_contract_id"],
                "relative_path": raw["relative_path"],
                "wnid": raw["wnid"],
                "label": int(raw["label"]),
                "correct_prediction": int(raw["correct_prediction"]),
                "wrong_prediction": int(raw["wrong_prediction"]),
                "correct_top1_correct": BASE.parse_bool(
                    raw["correct_top1_correct"]
                ),
                "wrong_top1_correct": BASE.parse_bool(
                    raw["wrong_top1_correct"]
                ),
                "correct_top5_correct": BASE.parse_bool(
                    raw["correct_top5_correct"]
                ),
                "wrong_top5_correct": BASE.parse_bool(
                    raw["wrong_top5_correct"]
                ),
                "correct_confidence": float(raw["correct_confidence"]),
                "wrong_confidence": float(raw["wrong_confidence"]),
                "js_divergence": float(raw["js_divergence"]),
                "identity_reencode_exact": BASE.parse_bool(
                    raw["identity_reencode_exact"]
                ),
                "input_element_count": int(raw["input_element_count"]),
                "wrong_input_low_clip_count": int(
                    raw["wrong_input_low_clip_count"]
                ),
                "wrong_input_high_clip_count": int(
                    raw["wrong_input_high_clip_count"]
                ),
            })
    return rows


def verify_bootstrap(rows, recorded, prefix, label):
    correct = [row[f"correct_{prefix}_correct"] for row in rows]
    wrong = [row[f"wrong_{prefix}_correct"] for row in rows]
    point, lower, upper = BASE.bootstrap_paired_delta(
        correct,
        wrong,
        recorded["iterations"],
        recorded["seed"],
    )
    close(point, recorded["point"], f"{label} bootstrap point")
    close(lower, recorded["lower_95"], f"{label} bootstrap lower")
    close(upper, recorded["upper_95"], f"{label} bootstrap upper")


def verify_comparison(rows, comparison):
    label = (
        f"{comparison['target_model_id']} <- "
        f"{comparison['source_contract_id']}"
    )
    metrics = comparison["metrics"]
    if len(rows) != metrics["image_count"]:
        fail(f"{label}: image count mismatch")
    if len({row["relative_path"] for row in rows}) != len(rows):
        fail(f"{label}: duplicate image rows")
    if not all(row["identity_reencode_exact"] for row in rows):
        fail(f"{label}: identity re-encoding failure")

    correct_top1 = [row["correct_top1_correct"] for row in rows]
    wrong_top1 = [row["wrong_top1_correct"] for row in rows]
    correct_top5 = [row["correct_top5_correct"] for row in rows]
    wrong_top5 = [row["wrong_top5_correct"] for row in rows]
    correct_conf = [row["correct_confidence"] for row in rows]
    wrong_conf = [row["wrong_confidence"] for row in rows]
    divergences = [row["js_divergence"] for row in rows]
    count = len(rows)

    close(sum(correct_top1) / count, metrics["correct_top1_accuracy"], f"{label}: baseline top-1")
    close(sum(wrong_top1) / count, metrics["wrong_top1_accuracy"], f"{label}: substituted top-1")
    close(sum(correct_top5) / count, metrics["correct_top5_accuracy"], f"{label}: baseline top-5")
    close(sum(wrong_top5) / count, metrics["wrong_top5_accuracy"], f"{label}: substituted top-5")
    verify_bootstrap(rows, metrics["paired_top1_accuracy_delta"], "top1", label)
    verify_bootstrap(rows, metrics["paired_top5_accuracy_delta"], "top5", label)

    lost = sum(c and not w for c, w in zip(correct_top1, wrong_top1))
    gained = sum(not c and w for c, w in zip(correct_top1, wrong_top1))
    mcnemar = metrics["mcnemar_exact"]
    if lost != mcnemar["correct_to_incorrect"]:
        fail(f"{label}: McNemar loss count mismatch")
    if gained != mcnemar["incorrect_to_correct"]:
        fail(f"{label}: McNemar gain count mismatch")
    if lost + gained != mcnemar["discordant_pair_count"]:
        fail(f"{label}: McNemar discordant count mismatch")
    close(
        BASE.exact_binomial_two_sided(min(lost, gained), lost + gained),
        mcnemar["two_sided_p_value"],
        f"{label}: exact McNemar p",
    )

    close(
        sum(
            row["correct_prediction"] == row["wrong_prediction"]
            for row in rows
        ) / count,
        metrics["top1_prediction_agreement"],
        f"{label}: prediction agreement",
    )
    close(sum(divergences) / count, metrics["mean_js_divergence"], f"{label}: mean JS")
    close(BASE.quantile(divergences, 0.5), metrics["median_js_divergence"], f"{label}: median JS")
    close(BASE.quantile(divergences, 0.95), metrics["p95_js_divergence"], f"{label}: p95 JS")
    close(sum(correct_conf) / count, metrics["correct_mean_confidence"], f"{label}: baseline confidence")
    close(sum(wrong_conf) / count, metrics["wrong_mean_confidence"], f"{label}: substituted confidence")
    close(
        BASE.expected_calibration_error(correct_conf, correct_top1),
        metrics["correct_ece_15_bin"],
        f"{label}: baseline ECE",
    )
    close(
        BASE.expected_calibration_error(wrong_conf, wrong_top1),
        metrics["wrong_ece_15_bin"],
        f"{label}: substituted ECE",
    )

    low = sum(row["wrong_input_low_clip_count"] for row in rows)
    high = sum(row["wrong_input_high_clip_count"] for row in rows)
    elements = sum(row["input_element_count"] for row in rows)
    if low != metrics["wrong_input_low_clip_count"]:
        fail(f"{label}: low clip count mismatch")
    if high != metrics["wrong_input_high_clip_count"]:
        fail(f"{label}: high clip count mismatch")
    close((low + high) / elements, metrics["wrong_input_clip_fraction"], f"{label}: clip fraction")


def expected_holm(comparisons):
    ordered = sorted(
        comparisons,
        key=lambda row: (
            row["metrics"]["mcnemar_exact"]["two_sided_p_value"],
            row["target_model_id"],
            row["source_contract_id"],
        ),
    )
    previous = 0.0
    result = {}
    for ordinal, comparison in enumerate(ordered):
        raw = comparison["metrics"]["mcnemar_exact"]["two_sided_p_value"]
        adjusted = min(1.0, max(previous, (len(ordered) - ordinal) * raw))
        key = (comparison["target_model_id"], comparison["source_contract_id"])
        result[key] = (ordinal + 1, adjusted)
        previous = adjusted
    return result


def verify_baseline_consistency(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["target_model_id"], row["relative_path"])].append(row)
    fields = (
        "label",
        "correct_prediction",
        "correct_top1_correct",
        "correct_top5_correct",
        "correct_confidence",
        "input_element_count",
    )
    for key, members in grouped.items():
        for field in fields:
            if len({row[field] for row in members}) != 1:
                fail(f"Baseline differs across source contracts: {key}/{field}")


def verify_two_model_overlap(root, result, rows):
    previous = BASE.read_prediction_rows(
        root / "experiments" / "imagenetv2" / "predictions.csv"
    )
    catalog = {
        alias: contract["contract_id"]
        for contract in result["contract_catalog"]
        for alias in contract["model_aliases"]
    }
    mappings = [
        (
            "google-mobilenet-v2-1.0-224-quant",
            "google-legacy-quantized/mobilenet-v2-1.0-224-quant",
            catalog["mediapipe-public/efficientnet-lite0-int8"],
        ),
        (
            "mediapipe-efficientnet-lite0-int8",
            "mediapipe-public/efficientnet-lite0-int8",
            catalog["google-legacy-quantized/mobilenet-v2-1.0-224-quant"],
        ),
    ]
    exact_fields = (
        "label",
        "correct_prediction",
        "wrong_prediction",
        "correct_top1_correct",
        "wrong_top1_correct",
        "correct_top5_correct",
        "wrong_top5_correct",
        "input_element_count",
        "wrong_input_low_clip_count",
        "wrong_input_high_clip_count",
    )
    float_fields = (
        "correct_confidence",
        "wrong_confidence",
        "js_divergence",
    )
    for old_model, target, source in mappings:
        old_by_path = {
            row["relative_path"]: row
            for row in previous if row["model_id"] == old_model
        }
        new_by_path = {
            row["relative_path"]: row
            for row in rows
            if row["target_model_id"] == target
            and row["source_contract_id"] == source
        }
        if set(old_by_path) != set(new_by_path):
            fail(f"Overlapping two-model image set differs: {target}")
        for path, old in old_by_path.items():
            new = new_by_path[path]
            for field in exact_fields:
                if old[field] != new[field]:
                    fail(f"Overlapping result differs: {target}/{path}/{field}")
            for field in float_fields:
                close(old[field], new[field], f"Overlapping result: {target}/{path}/{field}")


def verify_result(directory):
    measurement = directory / "measurement.json"
    result = json.loads(measurement.read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA:
        fail(f"Unsupported schema: {result.get('schema')}")

    unsigned = dict(result)
    recorded_ledger = unsigned.pop("ledger_sha256")
    if sha256_json(unsigned) != recorded_ledger:
        fail("Measurement ledger SHA-256 mismatch")

    provenance = result["provenance"]
    path_fields = (
        ("script", "script_sha256", ROOT / "scripts"),
        ("base_script", "base_script_sha256", ROOT / "scripts"),
        ("model_manifest", "model_manifest_sha256", ROOT),
        ("artifact_ledger", "artifact_ledger_sha256", ROOT),
        ("interface_ledger", "interface_ledger_sha256", ROOT),
    )
    for path_key, hash_key, base in path_fields:
        path = base / provenance[path_key]
        provenance_source(path, provenance[hash_key])

    predictions = directory / result["outputs"]["predictions_csv"]
    if sha256_file(predictions) != result["outputs"]["predictions_csv_sha256"]:
        fail("Predictions CSV SHA-256 mismatch")
    rows = read_rows(predictions)
    comparisons = result["comparisons"]
    if len(rows) != result["dataset"]["image_count"] * len(comparisons):
        fail("Prediction row count mismatch")

    model_ids = {row["qualified_id"] for row in result["models"]}
    contract_ids = {row["contract_id"] for row in result["contract_catalog"]}
    if len(model_ids) != len(result["models"]):
        fail("Duplicate target model IDs")
    if len(contract_ids) != len(result["contract_catalog"]):
        fail("Duplicate contract IDs")
    target_contract = {
        row["qualified_id"]: row["input_contract_id"] for row in result["models"]
    }
    expected_pairs = {
        (target, source)
        for target in model_ids
        for source in contract_ids
        if source != target_contract[target]
    }
    recorded_pairs = {
        (row["target_model_id"], row["source_contract_id"])
        for row in comparisons
    }
    if recorded_pairs != expected_pairs:
        fail("Comparison matrix is not the complete non-identity set")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["target_model_id"], row["source_contract_id"])].append(row)
    if set(grouped) != expected_pairs:
        fail("Prediction CSV pair set mismatch")
    comparison_by_pair = {
        (row["target_model_id"], row["source_contract_id"]): row
        for row in comparisons
    }
    for pair, members in grouped.items():
        verify_comparison(members, comparison_by_pair[pair])
    verify_baseline_consistency(rows)
    verify_two_model_overlap(ROOT, result, rows)

    first_pair = sorted(grouped)[0]
    image_rows = sorted(grouped[first_pair], key=lambda row: row["relative_path"])
    image_ledger = [{
        "relative_path": row["relative_path"],
        "wnid": row["wnid"],
        "label": row["label"],
    } for row in image_rows]
    if sha256_json(image_ledger) != result["dataset"]["image_ledger_sha256"]:
        fail("Image ledger SHA-256 mismatch")
    expected_paths = {row["relative_path"] for row in image_rows}
    for pair, members in grouped.items():
        if {row["relative_path"] for row in members} != expected_paths:
            fail(f"Image selection differs across comparison: {pair}")

    holm = expected_holm(comparisons)
    for pair, comparison in comparison_by_pair.items():
        rank, adjusted = holm[pair]
        recorded = comparison["mcnemar_holm"]
        if recorded["family_size"] != len(comparisons):
            fail(f"Holm family size mismatch: {pair}")
        if recorded["rank"] != rank:
            fail(f"Holm rank mismatch: {pair}")
        close(adjusted, recorded["adjusted_p_value"], f"Holm adjusted p: {pair}")
        if recorded["reject_at_0_05"] != (adjusted < 0.05):
            fail(f"Holm decision mismatch: {pair}")

    text = measurement.read_text(encoding="utf-8") + predictions.read_text(encoding="utf-8")
    if ":\\Users\\" in text or "/Users/" in text or "file://" in text:
        fail("Public result contains a local absolute path")
    return {
        "status": "pass",
        "target_model_count": len(model_ids),
        "distinct_contract_count": len(contract_ids),
        "comparison_count": len(comparisons),
        "image_count": result["dataset"]["image_count"],
        "prediction_row_count": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(verify_result(Path(args.directory)), indent=2))


if __name__ == "__main__":
    main()
