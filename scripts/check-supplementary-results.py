import json
import math
from collections import Counter
from pathlib import Path


def fail(message):
    raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    artifacts = json.loads(
        (root / "data/artifacts.json").read_text(encoding="utf-8")
    )["artifacts"]
    parameters = json.loads(
        (root / "data/interface-contracts.json").read_text(encoding="utf-8")
    )["parameters"]
    known_artifacts = {row["qualified_id"]: row for row in artifacts}
    known_parameters = {
        (row["qualified_id"], row["parameter_id"]): row
        for row in parameters
    }

    alignment = json.loads(
        (root / "data/candidate-transform-alignment.json").read_text(
            encoding="utf-8"
        )
    )
    rows = alignment["inputs"]
    if len(rows) != 32 or alignment["summary"]["assessed_input_count"] != 32:
        fail("Candidate transform input count mismatch")
    if len({
        (row["qualified_id"], row["parameter_id"]) for row in rows
    }) != len(rows):
        fail("Duplicate candidate transform parameter")
    exact_counts = Counter()
    closest_counts = Counter()
    no_exact = 0
    for row in rows:
        source = known_parameters.get((row["qualified_id"], row["parameter_id"]))
        if source is None:
            fail(f"Unknown candidate transform parameter: {row['parameter_id']}")
        if row["qualified_id"] not in known_artifacts:
            fail(f"Unknown candidate transform artifact: {row['qualified_id']}")
        quantization = source["quantization"]
        if row["scale"] != quantization["scales"][0]:
            fail(f"Candidate transform scale mismatch: {row['parameter_id']}")
        if row["zero_point"] != quantization["zero_points"][0]:
            fail(f"Candidate transform zero-point mismatch: {row['parameter_id']}")
        if set(row["candidates"]) != {
            "raw_storage",
            "unit_interval",
            "minus_one_to_one_255",
            "center_128_div_128",
            "center_127_5_div_127_5",
        }:
            fail(f"Candidate transform set mismatch: {row['parameter_id']}")
        if set(row["candidate_ranking"]) != set(row["candidates"]):
            fail(f"Candidate ranking is not a permutation: {row['parameter_id']}")
        if row["closest_candidate_by_rms_direct_byte_error"] != row["candidate_ranking"][0]:
            fail(f"Closest candidate differs from ranking: {row['parameter_id']}")
        exact = []
        for name, candidate in row["candidates"].items():
            count = candidate["clipping_source_value_count"]
            if not math.isclose(
                candidate["clipping_fraction"], count / 256.0,
                rel_tol=1e-12, abs_tol=1e-12,
            ):
                fail(f"Candidate clipping fraction mismatch: {row['parameter_id']}/{name}")
            if candidate["alignment_class"] == "exact_lattice_alignment":
                if candidate["exact_storage_code_count"] != 256:
                    fail(f"Invalid exact candidate: {row['parameter_id']}/{name}")
                exact.append(name)
        if sorted(exact) != sorted(row["exact_candidates"]):
            fail(f"Exact candidate list mismatch: {row['parameter_id']}")
        if row["exact_candidate_count"] != len(exact):
            fail(f"Exact candidate count mismatch: {row['parameter_id']}")
        exact_counts.update(exact)
        closest_counts[row["closest_candidate_by_rms_direct_byte_error"]] += 1
        no_exact += not exact
    summary = alignment["summary"]
    if summary["input_count_without_exact_candidate"] != no_exact:
        fail("Candidate no-exact count mismatch")
    if summary["exact_alignment_counts"] != dict(sorted(exact_counts.items())):
        fail("Candidate exact summary mismatch")
    if summary["closest_candidate_counts"] != dict(sorted(closest_counts.items())):
        fail("Candidate closest summary mismatch")

    revision_manifest = json.loads(
        (root / "data/onnx-revision-pairs.json").read_text(encoding="utf-8")
    )
    revision_result = json.loads(
        (root / "data/onnx-revision-comparison.json").read_text(
            encoding="utf-8"
        )
    )
    expected_pairs = {row["id"]: row for row in revision_manifest["pairs"]}
    observed_pairs = {row["id"]: row for row in revision_result["pairs"]}
    if set(expected_pairs) != set(observed_pairs):
        fail("Revision pair identifiers differ")
    for pair_id, expected in expected_pairs.items():
        observed = observed_pairs[pair_id]
        for key in ("repo", "path", "before", "after"):
            if observed[key] != expected[key]:
                fail(f"Revision manifest mismatch: {pair_id}/{key}")
        file_changed = expected["before"]["sha256"] != expected["after"]["sha256"]
        if observed["file_changed"] != file_changed:
            fail(f"Revision file-change mismatch: {pair_id}")
        actual_interface_change = observed["interface_before"] != observed["interface_after"]
        if observed["interface_changed"] != actual_interface_change:
            fail(f"Revision interface-change mismatch: {pair_id}")
        interface_summary = observed["interface_change_summary"]
        if interface_summary["dtype_change_count"] != 4:
            fail(f"Revision dtype-change count mismatch: {pair_id}")
        if interface_summary["shape_change_count"] != 0:
            fail(f"Revision shape-change count mismatch: {pair_id}")
        if interface_summary["affine_contract_change_count"] != 0:
            fail(f"Revision affine-change count mismatch: {pair_id}")
        if interface_summary["unchanged_common_parameter_count"] != 1:
            fail(f"Revision unchanged-parameter count mismatch: {pair_id}")
        initializers = observed["initializer_comparison"]
        if initializers["before_count"] != 131:
            fail(f"Revision before-initializer count mismatch: {pair_id}")
        if initializers["after_count"] != 131:
            fail(f"Revision after-initializer count mismatch: {pair_id}")
        if not initializers["key_sets_equal"]:
            fail(f"Revision initializer key set changed: {pair_id}")
        if not initializers["all_common_serialized_tensors_identical"]:
            fail(f"Revision initializer payload changed: {pair_id}")
        graph = observed["graph_comparison"]
        if graph["before_graph_count"] != 31 or graph["after_graph_count"] != 31:
            fail(f"Revision recursive graph count mismatch: {pair_id}")
        if graph["before_node_count"] != 160 or graph["after_node_count"] != 164:
            fail(f"Revision recursive node count mismatch: {pair_id}")
        if graph["operator_count_delta"] != {"ai.onnx::Cast": 4}:
            fail(f"Revision operator delta mismatch: {pair_id}")
        if graph["boundary_cast_count_delta"] != 4:
            fail(f"Revision boundary-Cast count mismatch: {pair_id}")
        if len(graph["after_boundary_casts"]) != 4:
            fail(f"Revision boundary-Cast evidence mismatch: {pair_id}")
        if observed["causal_attribution"] != "not_assessed":
            fail(f"Revision causal attribution overclaim: {pair_id}")
    if revision_result["pair_count"] != len(observed_pairs):
        fail("Revision pair count mismatch")
    if revision_result["changed_file_count"] != sum(
        row["file_changed"] for row in observed_pairs.values()
    ):
        fail("Revision changed-file count mismatch")
    if revision_result["changed_interface_count"] != sum(
        row["interface_changed"] for row in observed_pairs.values()
    ):
        fail("Revision changed-interface count mismatch")

    print(json.dumps({
        "status": "pass",
        "candidate_transform_input_count": len(rows),
        "revision_pair_count": len(observed_pairs),
        "changed_revision_interface_count": revision_result["changed_interface_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
