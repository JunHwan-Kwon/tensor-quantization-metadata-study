import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEMA = "tensor_quantization_metadata_study.kaggle_revision_comparison.v1"
PAIR_SCHEMA = "tensor_quantization_metadata_study.kaggle_revision_pairs.v1"
AUDIT_SCHEMA = "tensor_quantization_metadata_study.kaggle_tflite_audit.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_id(variation_ref, version_number, logical_path):
    return f"{variation_ref}/{version_number}:{logical_path}"


def affine_projection(parameter):
    return {
        "status": parameter["affine"]["status"],
        "granularity": parameter["affine"]["granularity"],
        "scales": parameter["scales"],
        "zero_points": parameter["zero_points"],
        "quantized_dimension": parameter["quantized_dimension"],
    }


def initializer_comparison(old, new):
    def keyed_entries(artifact):
        entries = artifact["initializers"]["entries"]
        keys = [row["match_key"] for row in entries]
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"Duplicate initializer match_key: {artifact['published_file_id']}"
            )
        return {row["match_key"]: row["data_sha256"] for row in entries}

    old_entries = keyed_entries(old)
    new_entries = keyed_entries(new)
    shared = set(old_entries) & set(new_entries)
    changed = sorted(
        key for key in shared if old_entries[key] != new_entries[key]
    )
    return {
        "old_initializer_count": len(old_entries),
        "new_initializer_count": len(new_entries),
        "shared_initializer_count": len(shared),
        "changed_shared_initializer_count": len(changed),
        "old_only_initializer_count": len(set(old_entries) - set(new_entries)),
        "new_only_initializer_count": len(set(new_entries) - set(old_entries)),
        "sets_and_bytes_identical": old_entries == new_entries,
    }


def compare_pair(pair, artifacts, parameters):
    if pair["evidence_class"] == "PATH_ADDED_OR_REMOVED":
        return {**pair, "classification": "PATH_ADDED_OR_REMOVED"}
    old_id = public_id(
        pair["variation_ref"],
        pair["old_version_number"],
        pair["logical_path"],
    )
    new_id = public_id(
        pair["variation_ref"],
        pair["new_version_number"],
        pair["logical_path"],
    )
    old_artifact = artifacts.get(old_id)
    new_artifact = artifacts.get(new_id)
    if (
        old_artifact is None
        or new_artifact is None
        or old_artifact["status"] != "ASSESSED"
        or new_artifact["status"] != "ASSESSED"
    ):
        return {
            **pair,
            "classification": "NOT_ASSESSABLE",
            "old_status": (
                None if old_artifact is None else old_artifact["status"]
            ),
            "new_status": (
                None if new_artifact is None else new_artifact["status"]
            ),
        }
    def keyed_parameters(identifier):
        rows = parameters.get(identifier, [])
        keys = [(row["direction"], row["ordinal"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate external parameter key: {identifier}")
        return {key: row for key, row in zip(keys, rows)}

    old_parameters = keyed_parameters(old_id)
    new_parameters = keyed_parameters(new_id)
    if set(old_parameters) != set(new_parameters):
        return {
            **pair,
            "classification": "EXTERNAL_PARAMETER_SET_CHANGED",
            "old_parameter_keys": [list(key) for key in sorted(old_parameters)],
            "new_parameter_keys": [list(key) for key in sorted(new_parameters)],
        }

    dtype_shape_deltas = []
    affine_deltas = []
    for key in sorted(old_parameters):
        old = old_parameters[key]
        new = new_parameters[key]
        for field in ("dtype", "shape"):
            if old[field] != new[field]:
                dtype_shape_deltas.append({
                    "direction": key[0],
                    "ordinal": key[1],
                    "field": field,
                    "old": old[field],
                    "new": new[field],
                })
        old_affine = affine_projection(old)
        new_affine = affine_projection(new)
        if old_affine != new_affine:
            affine_deltas.append({
                "direction": key[0],
                "ordinal": key[1],
                "old": old_affine,
                "new": new_affine,
            })
    initializers = initializer_comparison(old_artifact, new_artifact)
    if old_artifact["artifact_sha256"] == new_artifact["artifact_sha256"]:
        if (
            dtype_shape_deltas
            or affine_deltas
            or not initializers["sets_and_bytes_identical"]
        ):
            raise ValueError(
                "Byte-identical artifacts produced inconsistent audit facts: "
                f"{old_id} vs {new_id}"
            )
        classification = "ARTIFACT_IDENTICAL"
    elif dtype_shape_deltas:
        classification = "DTYPE_OR_SHAPE_CHANGED"
    elif affine_deltas:
        classification = (
            "AFFINE_CHANGED_INITIALIZERS_IDENTICAL"
            if initializers["sets_and_bytes_identical"]
            else "AFFINE_CHANGED_INITIALIZERS_CHANGED"
        )
    else:
        classification = "INTERFACE_UNCHANGED"
    return {
        **pair,
        "classification": classification,
        "old_artifact_sha256": old_artifact["artifact_sha256"],
        "new_artifact_sha256": new_artifact["artifact_sha256"],
        "dtype_shape_deltas": dtype_shape_deltas,
        "affine_deltas": affine_deltas,
        "initializer_comparison": initializers,
        "causal_attribution": "NOT_INFERRED",
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=root / "data/kaggle-tflite-revision-pairs.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=root / "data/kaggle-tflite-audit-all-versions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/kaggle-tflite-revision-comparison.json",
    )
    args = parser.parse_args()
    pair_document = json.loads(args.pairs.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if pair_document["schema"] != PAIR_SCHEMA:
        raise ValueError("Unexpected revision-pair schema")
    if audit["schema"] != AUDIT_SCHEMA or audit["scope"] != "all":
        raise ValueError("Revision comparison requires an all-version audit")
    artifacts = {
        row["published_file_id"]: row for row in audit["artifacts"]
    }
    parameters = {}
    for row in audit["parameters"]:
        parameters.setdefault(row["published_file_id"], []).append(row)
    results = [
        compare_pair(pair, artifacts, parameters)
        for pair in pair_document["pairs"]
    ]
    counts = dict(sorted(Counter(
        row["classification"] for row in results
    ).items()))
    document = {
        "schema": SCHEMA,
        "source_pair_ledger_sha256": sha256_file(args.pairs),
        "source_audit_sha256": sha256_file(args.audit),
        "comparison_count": len(results),
        "classification_counts": counts,
        "comparisons": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
