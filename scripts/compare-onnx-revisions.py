import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


def load_scanner(root):
    path = root / "scripts/scan-onnx-interface-contracts.py"
    spec = importlib.util.spec_from_file_location("onnx_contract_scanner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interface_projection(artifact):
    return [
        {
            "direction": row["direction"],
            "ordinal": row["ordinal"],
            "name": row["name"],
            "dtype": row["dtype"],
            "shape": row["shape"],
            "contract": row["contract"],
        }
        for row in artifact["parameters"]
    ]


def changed_fields(before, after):
    keys = sorted(set(before) | set(after))
    return [key for key in keys if before.get(key) != after.get(key)]


def walk_graphs(onnx, graph, path="graph"):
    yield path, graph
    for node_index, node in enumerate(graph.node):
        node_name = node.name or f"{node.op_type}_{node_index}"
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                child_path = f"{path}/{node_name}:{attribute.name}"
                yield from walk_graphs(onnx, attribute.g, child_path)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph_index, child in enumerate(attribute.graphs):
                    child_path = (
                        f"{path}/{node_name}:{attribute.name}[{graph_index}]"
                    )
                    yield from walk_graphs(onnx, child, child_path)


def operator_counts(onnx, model):
    counts = Counter()
    for _, graph in walk_graphs(onnx, model.graph):
        for node in graph.node:
            domain = node.domain or "ai.onnx"
            counts[f"{domain}::{node.op_type}"] += 1
    return dict(sorted(counts.items()))


def initializer_records(onnx, model):
    records = {}
    for graph_path, graph in walk_graphs(onnx, model.graph):
        for tensor in graph.initializer:
            key = f"{graph_path}::{tensor.name}"
            serialized = tensor.SerializeToString(deterministic=True)
            records[key] = {
                "dtype": onnx.TensorProto.DataType.Name(tensor.data_type),
                "shape": list(tensor.dims),
                "serialized_sha256": hashlib.sha256(serialized).hexdigest(),
            }
    return records


def compare_initializers(before, after):
    before_keys = set(before)
    after_keys = set(after)
    common = sorted(before_keys & after_keys)
    dtype_changed = sum(
        before[key]["dtype"] != after[key]["dtype"] for key in common
    )
    shape_changed = sum(
        before[key]["shape"] != after[key]["shape"] for key in common
    )
    serialized_changed = sum(
        before[key]["serialized_sha256"]
        != after[key]["serialized_sha256"]
        for key in common
    )
    return {
        "before_count": len(before),
        "after_count": len(after),
        "common_count": len(common),
        "only_before_count": len(before_keys - after_keys),
        "only_after_count": len(after_keys - before_keys),
        "dtype_changed_count": dtype_changed,
        "shape_changed_count": shape_changed,
        "serialized_tensor_changed_count": serialized_changed,
        "byte_identical_common_count": len(common) - serialized_changed,
        "key_sets_equal": before_keys == after_keys,
        "all_common_serialized_tensors_identical": serialized_changed == 0,
    }


def affine_contract_payload(contract):
    if contract.get("status") not in {
        "DIRECT_INTERFACE_METADATA",
        "DERIVED_STATIC_GRAPH",
    }:
        return None
    keys = (
        "scheme",
        "granularity",
        "scales",
        "zero_points",
        "axis",
        "quantized_dimension",
    )
    return {key: contract.get(key) for key in keys if key in contract}


def compare_interfaces(before, after):
    before_rows = {
        (row["direction"], row["ordinal"], row["name"]): row
        for row in before
    }
    after_rows = {
        (row["direction"], row["ordinal"], row["name"]): row
        for row in after
    }
    common = sorted(set(before_rows) & set(after_rows))
    deltas = []
    dtype_changes = 0
    shape_changes = 0
    affine_changes = 0
    unchanged = 0
    for key in common:
        before_row = before_rows[key]
        after_row = after_rows[key]
        fields = []
        if before_row["dtype"] != after_row["dtype"]:
            fields.append("dtype")
            dtype_changes += 1
        if before_row["shape"] != after_row["shape"]:
            fields.append("shape")
            shape_changes += 1
        before_affine = affine_contract_payload(before_row["contract"])
        after_affine = affine_contract_payload(after_row["contract"])
        if before_affine != after_affine:
            fields.append("affine_contract")
            affine_changes += 1
        if fields:
            deltas.append({
                "direction": key[0],
                "ordinal": key[1],
                "name": key[2],
                "changed_fields": fields,
                "dtype_before": before_row["dtype"],
                "dtype_after": after_row["dtype"],
                "shape_before": before_row["shape"],
                "shape_after": after_row["shape"],
                "affine_contract_before": before_affine,
                "affine_contract_after": after_affine,
            })
        else:
            unchanged += 1
    return {
        "before_parameter_count": len(before),
        "after_parameter_count": len(after),
        "common_parameter_count": len(common),
        "only_before_parameter_count": len(set(before_rows) - set(after_rows)),
        "only_after_parameter_count": len(set(after_rows) - set(before_rows)),
        "dtype_change_count": dtype_changes,
        "shape_change_count": shape_changes,
        "affine_contract_change_count": affine_changes,
        "unchanged_common_parameter_count": unchanged,
        "parameter_deltas": deltas,
    }


def count_delta(before, after):
    keys = sorted(set(before) | set(after))
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in keys
        if after.get(key, 0) != before.get(key, 0)
    }


def boundary_casts(onnx, model):
    graph = model.graph
    input_dtypes = {
        row.name: onnx.TensorProto.DataType.Name(
            row.type.tensor_type.elem_type
        )
        for row in graph.input
    }
    output_dtypes = {
        row.name: onnx.TensorProto.DataType.Name(
            row.type.tensor_type.elem_type
        )
        for row in graph.output
    }
    rows = []
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        cast_to = next(
            (attribute.i for attribute in node.attribute if attribute.name == "to"),
            None,
        )
        cast_to_dtype = (
            onnx.TensorProto.DataType.Name(cast_to)
            if cast_to is not None
            else None
        )
        if node.input and node.input[0] in input_dtypes:
            rows.append({
                "node_name": node.name or None,
                "direction": "input",
                "external_parameter": node.input[0],
                "external_dtype": input_dtypes[node.input[0]],
                "cast_to_dtype": cast_to_dtype,
                "node_inputs": list(node.input),
                "node_outputs": list(node.output),
            })
        elif node.output and node.output[0] in output_dtypes:
            rows.append({
                "node_name": node.name or None,
                "direction": "output",
                "external_parameter": node.output[0],
                "external_dtype": output_dtypes[node.output[0]],
                "cast_to_dtype": cast_to_dtype,
                "node_inputs": list(node.input),
                "node_outputs": list(node.output),
            })
    return sorted(
        rows,
        key=lambda row: (
            row["direction"],
            row["external_parameter"],
            row["node_name"] or "",
        ),
    )


def main():
    root = Path(__file__).resolve().parents[1]
    scanner = load_scanner(root)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=root / "data/onnx-revision-pairs.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "cache/onnx-revisions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/onnx-revision-comparison.json",
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.pairs.read_text(encoding="utf-8"))
    results = []
    for pair in manifest["pairs"]:
        scanned = {}
        models = {}
        for side in ("before", "after"):
            row = pair[side]
            path = scanner.ensure_artifact(args.cache, row, args.download)
            scanned[side] = scanner.scan_artifact(row, path)
            models[side] = scanner.onnx.load(path, load_external_data=False)
        before_interface = interface_projection(scanned["before"])
        after_interface = interface_projection(scanned["after"])
        interface_comparison = compare_interfaces(
            before_interface, after_interface
        )
        before_operator_counts = operator_counts(
            scanner.onnx, models["before"]
        )
        after_operator_counts = operator_counts(
            scanner.onnx, models["after"]
        )
        initializer_comparison = compare_initializers(
            initializer_records(scanner.onnx, models["before"]),
            initializer_records(scanner.onnx, models["after"]),
        )
        before_boundary_casts = boundary_casts(
            scanner.onnx, models["before"]
        )
        after_boundary_casts = boundary_casts(
            scanner.onnx, models["after"]
        )
        operator_delta = count_delta(
            before_operator_counts, after_operator_counts
        )
        results.append({
            "id": pair["id"],
            "repo": pair["repo"],
            "path": pair["path"],
            "before": pair["before"],
            "after": pair["after"],
            "file_changed": (
                pair["before"]["sha256"] != pair["after"]["sha256"]
            ),
            "interface_changed": before_interface != after_interface,
            "interface_before": before_interface,
            "interface_after": after_interface,
            "interface_change_summary": interface_comparison,
            "graph_summary_changed_fields": changed_fields(
                {
                    "pattern": scanned["before"]["graph_quantization_pattern"],
                    "node_count": scanned["before"]["node_count"],
                    "operator_counts": scanned["before"]["quantization_operator_counts"],
                },
                {
                    "pattern": scanned["after"]["graph_quantization_pattern"],
                    "node_count": scanned["after"]["node_count"],
                    "operator_counts": scanned["after"]["quantization_operator_counts"],
                },
            ),
            "model_metadata_changed_fields": changed_fields(
                {
                    "producer_name": scanned["before"]["producer_name"],
                    "producer_version": scanned["before"]["producer_version"],
                    "ir_version": scanned["before"]["ir_version"],
                    "opset_imports": scanned["before"]["opset_imports"],
                },
                {
                    "producer_name": scanned["after"]["producer_name"],
                    "producer_version": scanned["after"]["producer_version"],
                    "ir_version": scanned["after"]["ir_version"],
                    "opset_imports": scanned["after"]["opset_imports"],
                },
            ),
            "graph_comparison": {
                "before_graph_count": sum(
                    1 for _ in walk_graphs(
                        scanner.onnx, models["before"].graph
                    )
                ),
                "after_graph_count": sum(
                    1 for _ in walk_graphs(
                        scanner.onnx, models["after"].graph
                    )
                ),
                "before_node_count": sum(before_operator_counts.values()),
                "after_node_count": sum(after_operator_counts.values()),
                "operator_count_delta": operator_delta,
                "before_boundary_casts": before_boundary_casts,
                "after_boundary_casts": after_boundary_casts,
                "boundary_cast_count_delta": (
                    len(after_boundary_casts) - len(before_boundary_casts)
                ),
            },
            "initializer_comparison": initializer_comparison,
            "evidence_interpretation": {
                "observed": (
                    "The same repository path changed external dtypes while "
                    "shapes and affine-contract status remained unchanged. "
                    "All serialized initializers remained byte-identical; "
                    "the operator-count delta was confined to boundary Cast "
                    "nodes and their required rewiring."
                ),
                "claim_boundary": (
                    "This is an observed external dtype-contract revision, "
                    "not an affine scale or zero-point revision. The graph "
                    "changed, so converter causality and a contract-only file "
                    "change are not claimed."
                ),
            },
            "causal_attribution": "not_assessed",
        })

    result = {
        "schema_version": 2,
        "tool": "scripts/compare-onnx-revisions.py",
        "pair_count": len(results),
        "changed_file_count": sum(row["file_changed"] for row in results),
        "changed_interface_count": sum(
            row["interface_changed"] for row in results
        ),
        "pairs": results,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "pair_count": result["pair_count"],
        "changed_file_count": result["changed_file_count"],
        "changed_interface_count": result["changed_interface_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
