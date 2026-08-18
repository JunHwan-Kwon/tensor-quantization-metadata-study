import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from ai_edge_litert import schema_py_generated as tflite


WEIGHT_INPUT_ROLES = {
    "CONV_2D": {1: "weight", 2: "bias"},
    "DEPTHWISE_CONV_2D": {1: "weight", 2: "bias"},
    "FULLY_CONNECTED": {1: "weight", 2: "bias"},
    "TRANSPOSE_CONV": {1: "weight", 3: "bias"},
}

# These axes are specified by the public LiteRT 8-bit operator contract.
LITERT_PER_AXIS_WEIGHT_AXES = {
    "CONV_2D": 0,
    "DEPTHWISE_CONV_2D": 3,
    "FULLY_CONNECTED": 0,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_type_names():
    return {
        value: name
        for name, value in vars(tflite.TensorType).items()
        if name.isupper() and isinstance(value, int)
    }


def builtin_operator_names():
    return {
        value: name
        for name, value in vars(tflite.BuiltinOperator).items()
        if name.isupper() and isinstance(value, int)
    }


def decode_flatbuffer_string(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def operator_name(model, operator, operator_names):
    code = model.OperatorCodes(operator.OpcodeIndex())
    # TFLite kept an int8 opcode for backward compatibility and added an
    # int32 field when the enum outgrew it. The reference accessor resolves
    # the effective opcode as the maximum of both serialized fields.
    builtin_code = max(
        int(code.BuiltinCode()), int(code.DeprecatedBuiltinCode())
    )
    if builtin_code == tflite.BuiltinOperator.CUSTOM:
        custom = decode_flatbuffer_string(code.CustomCode())
        return f"CUSTOM:{custom or 'unnamed'}"
    return operator_names.get(builtin_code, f"BUILTIN_{builtin_code}")


def locate_models(cache_root, artifacts):
    expected = {row["sha256"]: row for row in artifacts}
    located = {}
    for path in cache_root.resolve().rglob("*.tflite"):
        digest = sha256_file(path)
        if digest in expected and digest not in located:
            located[digest] = path
    return expected, located


def analyze_artifact(path, artifact, type_names, operator_names):
    payload = path.read_bytes()
    if len(payload) != artifact["size_bytes"]:
        raise ValueError(f"Artifact size mismatch: {artifact['qualified_id']}")
    if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
        raise ValueError(f"Artifact SHA-256 mismatch: {artifact['qualified_id']}")

    model = tflite.Model.GetRootAsModel(bytearray(payload), 0)
    counts = Counter()
    axes = Counter()
    scale_lengths = Counter()
    weight_contract_counts = Counter()
    weight_consumer_binding_op_axis_counts = Counter()
    weight_tensor_op_axis_counts = Counter()
    weight_role_counts = Counter()
    violations = []
    external_rows = []
    known_weight_rows = []
    for subgraph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(subgraph_index)
        consumers = {}
        for operator_index in range(graph.OperatorsLength()):
            operator = graph.Operators(operator_index)
            op_name = operator_name(model, operator, operator_names)
            roles = WEIGHT_INPUT_ROLES.get(op_name, {})
            for input_slot in range(operator.InputsLength()):
                tensor_index = int(operator.Inputs(input_slot))
                if tensor_index < 0:
                    continue
                consumers.setdefault(tensor_index, []).append({
                    "operator": operator_index,
                    "op": op_name,
                    "input_slot": input_slot,
                    "role": roles.get(input_slot, "operand"),
                })
        external = {
            int(graph.Inputs(index))
            for index in range(graph.InputsLength())
        } | {
            int(graph.Outputs(index))
            for index in range(graph.OutputsLength())
        }
        for tensor_index in range(graph.TensorsLength()):
            tensor = graph.Tensors(tensor_index)
            shape = [
                int(tensor.Shape(index))
                for index in range(tensor.ShapeLength())
            ]
            dtype = type_names.get(tensor.Type(), str(tensor.Type()))
            quantization = tensor.Quantization()
            if quantization is None:
                if tensor_index in external:
                    external_rows.append({
                        "subgraph": subgraph_index,
                        "tensor": tensor_index,
                        "dtype": dtype,
                        "shape": shape,
                        "status": "not_quantized",
                    })
                continue
            scale_count = quantization.ScaleLength()
            zero_point_count = quantization.ZeroPointLength()
            if scale_count == 0 and zero_point_count == 0:
                if tensor_index in external:
                    external_rows.append({
                        "subgraph": subgraph_index,
                        "tensor": tensor_index,
                        "dtype": dtype,
                        "shape": shape,
                        "status": "not_quantized",
                    })
                continue
            if scale_count == 0 or scale_count != zero_point_count:
                violations.append({
                    "subgraph": subgraph_index,
                    "tensor": tensor_index,
                    "kind": "scale-zero-point-count-mismatch",
                    "scale_count": scale_count,
                    "zero_point_count": zero_point_count,
                })
                continue

            axis = int(quantization.QuantizedDimension())
            buffer = model.Buffers(tensor.Buffer())
            inline_buffer = buffer is not None and buffer.DataLength() > 0
            external_offset = int(buffer.Offset()) if buffer is not None else 0
            external_size = int(buffer.Size()) if buffer is not None else 0
            external_buffer = external_offset > 0 or external_size > 0
            if external_buffer and (
                external_offset <= 0
                or external_size <= 0
                or external_offset + external_size > len(payload)
            ):
                violations.append({
                    "subgraph": subgraph_index,
                    "tensor": tensor_index,
                    "kind": "external-buffer-range-invalid",
                    "offset": external_offset,
                    "size": external_size,
                    "file_size": len(payload),
                })
            has_buffer = inline_buffer or external_buffer
            if tensor_index in external:
                scope = "external"
            elif has_buffer:
                scope = "constant"
            else:
                scope = "internal_nonconstant"
            granularity = "per_tensor" if scale_count == 1 else "per_axis"
            counts[f"{scope}:{granularity}"] += 1
            counts[f"{scope}:{granularity}:{dtype}"] += 1
            if granularity == "per_tensor":
                counts[f"{scope}:per_tensor:serialized_axis:{axis}"] += 1
            else:
                axes[f"{scope}:{dtype}:{axis}"] += 1
                scale_lengths[scale_count] += 1
                if axis < 0 or axis >= len(shape):
                    violations.append({
                        "subgraph": subgraph_index,
                        "tensor": tensor_index,
                        "kind": "axis-out-of-range",
                        "axis": axis,
                        "shape": shape,
                    })
                elif shape[axis] >= 0 and shape[axis] != scale_count:
                    violations.append({
                        "subgraph": subgraph_index,
                        "tensor": tensor_index,
                        "kind": "scale-axis-cardinality-mismatch",
                        "axis": axis,
                        "shape": shape,
                        "scale_count": scale_count,
                    })

            tensor_consumers = consumers.get(tensor_index, [])
            weight_bindings = [
                binding
                for binding in tensor_consumers
                if binding["role"] == "weight"
            ]
            bias_bindings = [
                binding
                for binding in tensor_consumers
                if binding["role"] == "bias"
            ]
            if scope == "constant":
                if weight_bindings:
                    role = "known_weight"
                elif bias_bindings:
                    role = "known_bias"
                elif tensor_consumers:
                    role = "other_consumed_constant"
                else:
                    role = "unconsumed_constant"
                weight_role_counts[f"{role}:{dtype}:{granularity}"] += 1

            if scope == "constant" and weight_bindings:
                semantic_axis = axis if granularity == "per_axis" else None
                signature = (
                    "per-tensor"
                    if semantic_axis is None
                    else f"per-channel:axis={semantic_axis}"
                )
                weight_contract_counts[signature] += 1
                axis_label = (
                    "not-applicable"
                    if semantic_axis is None
                    else str(semantic_axis)
                )
                for op_name in sorted({
                    binding["op"] for binding in weight_bindings
                }):
                    weight_tensor_op_axis_counts[
                        f"{op_name}:{granularity}:axis={axis_label}"
                    ] += 1
                for binding in weight_bindings:
                    weight_consumer_binding_op_axis_counts[
                        f"{binding['op']}:{granularity}:axis={axis_label}"
                    ] += 1
                    expected_axis = LITERT_PER_AXIS_WEIGHT_AXES.get(
                        binding["op"]
                    )
                    if (
                        semantic_axis is not None
                        and expected_axis is not None
                        and semantic_axis != expected_axis
                    ):
                        violations.append({
                            "subgraph": subgraph_index,
                            "tensor": tensor_index,
                            "kind": "litert-weight-axis-contract-mismatch",
                            "op": binding["op"],
                            "expected_axis": expected_axis,
                            "actual_axis": semantic_axis,
                        })
                known_weight_rows.append({
                    "subgraph": subgraph_index,
                    "tensor": tensor_index,
                    "name": decode_flatbuffer_string(tensor.Name()),
                    "dtype": dtype,
                    "shape": shape,
                    "semantic_granularity": granularity,
                    "semantic_axis": semantic_axis,
                    "serialized_quantized_dimension": axis,
                    "scale_count": scale_count,
                    "zero_point_count": zero_point_count,
                    "contract_signature": signature,
                    "weight_consumers": weight_bindings,
                    "other_consumer_count": (
                        len(tensor_consumers) - len(weight_bindings)
                    ),
                })
            if tensor_index in external:
                external_rows.append({
                    "subgraph": subgraph_index,
                    "tensor": tensor_index,
                    "dtype": dtype,
                    "shape": shape,
                    "status": "complete",
                    "scale_count": scale_count,
                    "zero_point_count": zero_point_count,
                    "serialized_quantized_dimension": axis,
                    "semantic_granularity": granularity,
                })

    projection_signatures = dict(sorted(weight_contract_counts.items()))
    per_channel_axes = sorted({
        int(signature.rsplit("=", 1)[1])
        for signature in projection_signatures
        if signature.startswith("per-channel:axis=")
    })
    granularities = sorted({
        "per-channel" if signature.startswith("per-channel:") else signature
        for signature in projection_signatures
    })
    if not projection_signatures:
        projection_status = "not_assessed_no_known_weight_binding"
    elif len(projection_signatures) == 1:
        projection_status = "lossless_single_object_projection_available"
    else:
        projection_status = "no_lossless_single_object_projection"

    projection_candidates = []
    weight_tensor_count = sum(weight_contract_counts.values())
    for signature, covered in sorted(weight_contract_counts.items()):
        projection_candidates.append({
            "selected_contract_signature": signature,
            "exactly_represented_weight_tensors": covered,
            "omitted_or_misrepresented_weight_tensors": (
                weight_tensor_count - covered
            ),
        })

    return {
        "qualified_id": artifact["qualified_id"],
        "artifact_sha256": artifact["sha256"],
        "subgraph_count": model.SubgraphsLength(),
        "counts": dict(sorted(counts.items())),
        "axis_counts": dict(sorted(axes.items())),
        "scale_vector_length_counts": {
            str(key): value for key, value in sorted(scale_lengths.items())
        },
        "constant_role_counts": dict(sorted(weight_role_counts.items())),
        "known_weight_contract_counts": projection_signatures,
        "known_weight_tensor_op_axis_counts": dict(
            sorted(weight_tensor_op_axis_counts.items())
        ),
        "known_weight_consumer_binding_op_axis_counts": dict(
            sorted(weight_consumer_binding_op_axis_counts.items())
        ),
        "known_weight_contracts": known_weight_rows,
        "model_level_single_object_projection": {
            "status": projection_status,
            "known_weight_tensor_count": weight_tensor_count,
            "distinct_contract_signature_count": len(projection_signatures),
            "observed_granularities": granularities,
            "observed_per_channel_axes": per_channel_axes,
            "candidate_information_loss": projection_candidates,
            "axis_omission_effect": (
                "axis semantics omitted for every per-channel weight tensor"
                if per_channel_axes else "not_applicable"
            ),
        },
        "external_parameters": external_rows,
        "violation_count": len(violations),
        "violations": violations,
    }


def sum_nested_counts(rows, key):
    total = Counter()
    for row in rows:
        total.update(row[key])
    return dict(sorted(total.items()))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=root / "data/artifacts.json",
    )
    parser.add_argument(
        "--contracts",
        type=Path,
        default=root / "data/interface-contracts.json",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=root / "cache/tflite-metadata-audit",
    )
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact_document = json.loads(args.artifacts.read_text(encoding="utf-8"))
    contract_document = json.loads(args.contracts.read_text(encoding="utf-8"))
    artifacts = artifact_document["artifacts"]
    expected, located = locate_models(args.cache_root, artifacts)
    missing = sorted(set(expected) - set(located))
    if args.require_all and missing:
        raise ValueError(f"Missing {len(missing)} hash-identified artifacts")

    type_names = tensor_type_names()
    operator_names = builtin_operator_names()
    rows = [
        analyze_artifact(
            located[digest], expected[digest], type_names, operator_names
        )
        for digest in sorted(located)
    ]
    all_counts = sum_nested_counts(rows, "counts")
    all_axes = sum_nested_counts(rows, "axis_counts")
    all_scale_lengths = sum_nested_counts(rows, "scale_vector_length_counts")
    all_constant_roles = sum_nested_counts(rows, "constant_role_counts")
    all_weight_contracts = sum_nested_counts(
        rows, "known_weight_contract_counts"
    )
    all_weight_tensor_op_axes = sum_nested_counts(
        rows, "known_weight_tensor_op_axis_counts"
    )
    all_weight_consumer_binding_op_axes = sum_nested_counts(
        rows, "known_weight_consumer_binding_op_axis_counts"
    )
    violations = sum(row["violation_count"] for row in rows)
    external_rows = [
        parameter
        for row in rows
        for parameter in row["external_parameters"]
    ]
    contract_complete = [
        row
        for row in contract_document["parameters"]
        if row["quantization"]["status"] == "complete"
    ]
    if len(external_rows) != len(contract_document["parameters"]):
        raise ValueError("External parameter count differs from interface ledger")
    if sum(
        row.get("semantic_granularity") == "per_tensor"
        for row in external_rows
    ) != len(contract_complete):
        raise ValueError("External per-tensor count differs from interface ledger")
    if violations:
        raise ValueError(f"Found {violations} quantization metadata violations")

    per_axis_artifacts = sum(
        any(
            key.startswith("constant:per_axis")
            or key.startswith("internal_nonconstant:per_axis")
            for key in row["counts"]
        )
        for row in rows
    )
    constant_axis_sets = Counter()
    for row in rows:
        axes = tuple(sorted({
            int(key.rsplit(":", 1)[1])
            for key in row["axis_counts"]
            if key.startswith("constant:")
        }))
        if axes:
            constant_axis_sets[",".join(map(str, axes))] += 1
    projection_status_counts = Counter(
        row["model_level_single_object_projection"]["status"]
        for row in rows
    )
    mixed_weight_axis_artifact_count = sum(
        len(
            row["model_level_single_object_projection"]
            ["observed_per_channel_axes"]
        ) > 1
        for row in rows
    )
    no_lossless_projection_rows = [
        row
        for row in rows
        if row["model_level_single_object_projection"]["status"]
        == "no_lossless_single_object_projection"
    ]
    best_candidate_losses = [
        min(
            candidate["omitted_or_misrepresented_weight_tensors"]
            for candidate in row["model_level_single_object_projection"]
            ["candidate_information_loss"]
        )
        for row in no_lossless_projection_rows
    ]
    result = {
        "schema": (
            "tensor_quantization_metadata_study."
            "tflite_granularity_evidence.v1"
        ),
        "parser": f"ai-edge-litert {version('ai-edge-litert')}",
        "method": (
            "Hash-verify each TFLite file, read every subgraph tensor with "
            "the generated LiteRT FlatBuffer schema, classify a tensor as "
            "per-tensor when scale_count=zero_point_count=1 and per-axis "
            "when the equal counts exceed one, and check scale_count "
            "against shape[quantized_dimension]. Treat both inline "
            "Buffer.data and file-local Buffer.offset/size ranges as "
            "serialized constants, rejecting out-of-file ranges. Resolve "
            "the effective builtin opcode from both compatibility fields "
            "and bind constants to operator input slots. No allocation or "
            "inference."
        ),
        "source_artifact_manifest_sha256": sha256_file(args.artifacts),
        "source_interface_contract_ledger_sha256": sha256_file(args.contracts),
        "expected_artifact_count": len(expected),
        "verified_artifact_count": len(rows),
        "missing_artifact_count": len(missing),
        "missing_artifact_sha256": missing,
        "external_parameter_count": len(external_rows),
        "complete_affine_external_parameter_count": len(contract_complete),
        "per_axis_artifact_count": per_axis_artifacts,
        "constant_per_axis_artifact_count": sum(constant_axis_sets.values()),
        "multiple_constant_axis_artifact_count": sum(
            value
            for axes, value in constant_axis_sets.items()
            if "," in axes
        ),
        "constant_axis_set_artifact_counts": dict(
            sorted(constant_axis_sets.items())
        ),
        "known_weight_tensor_count": sum(all_weight_contracts.values()),
        "known_weight_contract_counts": all_weight_contracts,
        "known_weight_tensor_op_axis_counts": all_weight_tensor_op_axes,
        "known_weight_consumer_binding_op_axis_counts": (
            all_weight_consumer_binding_op_axes
        ),
        "constant_role_counts": all_constant_roles,
        "model_level_projection_status_counts": dict(
            sorted(projection_status_counts.items())
        ),
        "mixed_weight_axis_artifact_count": (
            mixed_weight_axis_artifact_count
        ),
        "no_lossless_projection_known_weight_tensor_count": sum(
            row["model_level_single_object_projection"]
            ["known_weight_tensor_count"]
            for row in no_lossless_projection_rows
        ),
        "best_single_candidate_omitted_weight_tensor_count": sum(
            best_candidate_losses
        ),
        "best_single_candidate_omitted_weight_tensor_range": [
            min(best_candidate_losses),
            max(best_candidate_losses),
        ],
        "counts": all_counts,
        "axis_counts": all_axes,
        "scale_vector_length_counts": all_scale_lengths,
        "minimum_scale_vector_length": min(map(int, all_scale_lengths)),
        "maximum_scale_vector_length": max(map(int, all_scale_lengths)),
        "distinct_scale_vector_length_count": len(all_scale_lengths),
        "violation_count": violations,
        "artifact_rows": rows,
        "interpretation_boundary": (
            "The 50 artifacts are a curated, hash-identified benchmark and "
            "not a probability sample. External model parameters and internal "
            "serialized tensors are separate analysis units. A serialized "
            "quantized_dimension value on a per-tensor contract is a format "
            "field, not evidence that an axis is semantically applicable. "
            "Model-level projection is assessed only for constant tensors "
            "bound to known operator weight input slots; bias tensors and "
            "unclassified constants are excluded. A failed single-object "
            "projection means an exact tensor-level weight contract cannot "
            "be represented losslessly by one shared granularity/axis pair; "
            "it does not prevent a deliberately coarse model-level summary."
        ),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
