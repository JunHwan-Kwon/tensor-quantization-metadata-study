import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


STATUS_VALUES = {
    "NOT_QUANTIZED",
    "DIRECT_INTERFACE_METADATA",
    "DERIVED_STATIC_GRAPH",
    "DYNAMIC_OR_UNBOUND",
    "AMBIGUOUS_MULTIPLE_CONTRACTS",
    "UNSUPPORTED",
}

QUANTIZED_DTYPES = {
    "INT8",
    "UINT8",
    "INT16",
    "UINT16",
    "INT32",
    "UINT32",
    "INT4",
    "UINT4",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_url(row):
    quoted_path = urllib.parse.quote(row["path"], safe="/")
    return (
        f"https://huggingface.co/{row['repo']}/resolve/"
        f"{row['revision']}/{quoted_path}?download=true"
    )


def cache_path(cache_root, row):
    safe_id = row["id"].replace("/", "__")
    return cache_root / f"{safe_id}.onnx"


def ensure_artifact(cache_root, row, allow_download):
    target = cache_path(cache_root, row)
    if target.is_file():
        if target.stat().st_size == row["size_bytes"]:
            if sha256_file(target) == row["sha256"]:
                return target
        target.unlink()
    if not allow_download:
        raise FileNotFoundError(f"Missing verified cache file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    request = urllib.request.Request(
        artifact_url(row),
        headers={"User-Agent": "tensor-quantization-metadata-study"},
    )
    with urllib.request.urlopen(request) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    if temporary.stat().st_size != row["size_bytes"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Size mismatch for {row['id']}")
    actual_hash = sha256_file(temporary)
    if actual_hash != row["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {row['id']}")
    temporary.replace(target)
    return target


def dtype_name(value_info):
    element_type = value_info.type.tensor_type.elem_type
    try:
        return TensorProto.DataType.Name(element_type)
    except ValueError:
        return f"UNKNOWN_{element_type}"


def shape_value(value_info):
    result = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param") and dim.dim_param:
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def attribute_value(node, name, default=None):
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def static_values(model):
    values = {}
    external = []
    for tensor in model.graph.initializer:
        if tensor.data_location == TensorProto.EXTERNAL:
            external.append(tensor.name)
            continue
        try:
            values[tensor.name] = numpy_helper.to_array(tensor)
        except (TypeError, ValueError):
            continue
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        tensor = attribute_value(node, "value")
        if tensor is None or tensor.data_location == TensorProto.EXTERNAL:
            continue
        try:
            values[node.output[0]] = numpy_helper.to_array(tensor)
        except (TypeError, ValueError):
            continue
    return values, external


def json_numbers(array):
    values = np.asarray(array).reshape(-1).tolist()
    return [value.item() if hasattr(value, "item") else value for value in values]


def contract_from_scale(
    node,
    scale_name,
    zero_point_name,
    constants,
    parameter_shape,
):
    if scale_name not in constants:
        return {
            "status": "DYNAMIC_OR_UNBOUND",
            "reason": f"Scale input {scale_name!r} is not a static initializer or Constant.",
        }
    scale = np.asarray(constants[scale_name])
    if scale.size == 0 or not np.all(np.isfinite(scale)):
        return {
            "status": "UNSUPPORTED",
            "reason": "Scale is empty or non-finite.",
        }
    if np.any(scale <= 0):
        return {
            "status": "UNSUPPORTED",
            "reason": "Scale contains a non-positive value.",
        }
    if zero_point_name:
        if zero_point_name not in constants:
            return {
                "status": "DYNAMIC_OR_UNBOUND",
                "reason": (
                    f"Zero-point input {zero_point_name!r} is not a static "
                    "initializer or Constant."
                ),
            }
        zero_point = np.asarray(constants[zero_point_name])
    else:
        zero_point = np.zeros(scale.shape or (), dtype=np.int64)
    if zero_point.size != scale.size:
        return {
            "status": "UNSUPPORTED",
            "reason": "Scale and zero-point cardinalities differ.",
        }

    block_size = int(attribute_value(node, "block_size", 0) or 0)
    axis_source = int(attribute_value(node, "axis", 1))
    if block_size > 0:
        granularity = "blocked"
    elif scale.size == 1:
        granularity = "per_tensor"
    else:
        granularity = "per_axis"

    axis = None
    cardinality = "not_applicable"
    if granularity != "per_tensor":
        rank = len(parameter_shape)
        axis = axis_source + rank if axis_source < 0 else axis_source
        if axis < 0 or axis >= rank:
            cardinality = "invalid_axis"
        elif isinstance(parameter_shape[axis], int):
            expected = parameter_shape[axis]
            if granularity == "per_axis":
                cardinality = (
                    "valid" if scale.size == expected else "invalid_length"
                )
            else:
                expected_blocked = (expected + block_size - 1) // block_size
                cardinality = (
                    "valid"
                    if scale.shape[axis] == expected_blocked
                    else "invalid_length"
                )
        else:
            cardinality = "not_assessable_dynamic_shape"

    return {
        "status": "DERIVED_STATIC_GRAPH",
        "evidence_class": "DERIVED_SERIALIZED_GRAPH",
        "consumer_or_producer": {
            "domain": node.domain or "ai.onnx",
            "op_type": node.op_type,
            "name": node.name or None,
        },
        "scheme": "affine",
        "granularity": granularity,
        "scales": json_numbers(scale),
        "zero_points": json_numbers(zero_point),
        "axis": axis,
        "source_axis": axis_source if granularity != "per_tensor" else None,
        "block_size": block_size or None,
        "cardinality_status": cardinality,
    }


def input_contract(value_info, consumers, constants):
    dtype = dtype_name(value_info)
    shape = shape_value(value_info)
    if dtype not in QUANTIZED_DTYPES:
        return {
            "status": "NOT_QUANTIZED",
            "reason": "External input dtype is not a quantized integer storage type.",
        }

    contracts = []
    unrecognized = []
    for node in consumers:
        if node.op_type == "DequantizeLinear" and len(node.input) >= 2:
            contracts.append(contract_from_scale(
                node,
                node.input[1],
                node.input[2] if len(node.input) >= 3 else None,
                constants,
                shape,
            ))
        elif node.op_type in {"QLinearConv", "QLinearMatMul"}:
            contracts.append(contract_from_scale(
                node,
                node.input[1],
                node.input[2] if len(node.input) >= 3 else None,
                constants,
                shape,
            ))
        else:
            unrecognized.append(f"{node.domain or 'ai.onnx'}::{node.op_type}")

    if not contracts:
        return {
            "status": "UNSUPPORTED",
            "reason": "No supported direct consumer exposes an affine contract.",
            "direct_consumers": sorted(set(unrecognized)),
        }
    if unrecognized:
        return {
            "status": "AMBIGUOUS_MULTIPLE_CONTRACTS",
            "reason": "The input has both affine-aware and unrecognized direct consumers.",
            "candidate_contracts": contracts,
            "direct_consumers": sorted(set(unrecognized)),
        }
    canonical = {json.dumps(item, sort_keys=True) for item in contracts}
    if len(canonical) != 1:
        return {
            "status": "AMBIGUOUS_MULTIPLE_CONTRACTS",
            "reason": "Direct consumers expose different affine contracts.",
            "candidate_contracts": contracts,
        }
    return contracts[0]


def output_contract(value_info, producer, constants):
    dtype = dtype_name(value_info)
    shape = shape_value(value_info)
    if dtype not in QUANTIZED_DTYPES:
        return {
            "status": "NOT_QUANTIZED",
            "reason": "External output dtype is not a quantized integer storage type.",
        }
    if producer is None:
        return {
            "status": "UNSUPPORTED",
            "reason": "No producer was found for the integer graph output.",
        }
    if producer.op_type == "QuantizeLinear" and len(producer.input) >= 2:
        return contract_from_scale(
            producer,
            producer.input[1],
            producer.input[2] if len(producer.input) >= 3 else None,
            constants,
            shape,
        )
    if producer.op_type in {"QLinearConv", "QLinearMatMul"}:
        if len(producer.input) < 8:
            return {
                "status": "UNSUPPORTED",
                "reason": "QLinear producer has no static output scale inputs.",
            }
        return contract_from_scale(
            producer,
            producer.input[6],
            producer.input[7],
            constants,
            shape,
        )
    return {
        "status": "UNSUPPORTED",
        "reason": "The direct producer is not a supported affine quantization operator.",
        "direct_producer": f"{producer.domain or 'ai.onnx'}::{producer.op_type}",
    }


def walk_nodes(graph):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from walk_nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from walk_nodes(child)


def quantization_pattern(model):
    counts = Counter(node.op_type for node in walk_nodes(model.graph))
    categories = []
    if counts["QuantizeLinear"] or counts["DequantizeLinear"]:
        categories.append("QDQ")
    if any(
        name.startswith("QLinear") or name in {"ConvInteger", "MatMulInteger"}
        for name in counts
    ):
        categories.append("QOPERATOR")
    if counts["DynamicQuantizeLinear"]:
        categories.append("DYNAMIC")
    if any(
        name in {"MatMulNBits", "GatherBlockQuantized"}
        for name in counts
    ):
        categories.append("WEIGHT_ONLY")
    if not categories:
        return "NONE", counts
    if len(categories) == 1:
        return categories[0], counts
    return "MIXED:" + "+".join(categories), counts


def scan_artifact(row, path):
    model = onnx.load(path, load_external_data=False)
    constants, external_initializers = static_values(model)
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    graph_inputs = [
        value for value in model.graph.input
        if value.name not in initializer_names
    ]
    consumers = defaultdict(list)
    producers = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)
        for name in node.output:
            if name:
                producers[name] = node

    parameters = []
    for direction, values in (
        ("input", graph_inputs),
        ("output", model.graph.output),
    ):
        for ordinal, value in enumerate(values):
            if direction == "input":
                contract = input_contract(
                    value, consumers.get(value.name, []), constants
                )
            else:
                contract = output_contract(
                    value, producers.get(value.name), constants
                )
            if contract["status"] not in STATUS_VALUES:
                raise ValueError(f"Unknown status: {contract['status']}")
            parameters.append({
                "direction": direction,
                "ordinal": ordinal,
                "name": value.name,
                "dtype": dtype_name(value),
                "shape": shape_value(value),
                "contract": contract,
            })

    pattern, op_counts = quantization_pattern(model)
    interesting_ops = {
        name: count for name, count in sorted(op_counts.items())
        if (
            "Quant" in name
            or name.startswith("QLinear")
            or name in {
                "ConvInteger",
                "MatMulInteger",
                "MatMulNBits",
                "GatherBlockQuantized",
            }
        )
    }
    return {
        "id": row["id"],
        "repo": row["repo"],
        "revision": row["revision"],
        "path": row["path"],
        "variant": row["variant"],
        "download_url": artifact_url(row),
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "producer_name": model.producer_name or None,
        "producer_version": model.producer_version or None,
        "ir_version": model.ir_version,
        "opset_imports": {
            item.domain or "ai.onnx": item.version
            for item in model.opset_import
        },
        "top_level_node_count": len(model.graph.node),
        "node_count": sum(1 for _ in walk_nodes(model.graph)),
        "initializer_count": len(model.graph.initializer),
        "external_initializer_count": len(external_initializers),
        "external_initializers": sorted(external_initializers),
        "graph_quantization_pattern": pattern,
        "quantization_operator_counts": interesting_ops,
        "parameters": parameters,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/onnx-pilot-manifest.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "cache/onnx-pilot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/onnx-pilot-results.json",
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    artifacts = []
    for row in manifest["artifacts"]:
        path = ensure_artifact(args.cache, row, args.download)
        artifacts.append(scan_artifact(row, path))

    status_counts = Counter(
        parameter["contract"]["status"]
        for artifact in artifacts
        for parameter in artifact["parameters"]
    )
    pattern_counts = Counter(
        artifact["graph_quantization_pattern"] for artifact in artifacts
    )
    result = {
        "schema_version": 1,
        "tool": {
            "script": "scripts/scan-onnx-interface-contracts.py",
            "onnx_version": onnx.__version__,
        },
        "scope": {
            "artifact_count": len(artifacts),
            "model_family_count": manifest["selection"]["model_family_count"],
            "selection_rule": manifest["selection"]["selection_rule"],
            "population_inference": "not_permitted",
        },
        "summary": {
            "parameter_count": sum(
                len(artifact["parameters"]) for artifact in artifacts
            ),
            "contract_status_counts": dict(sorted(status_counts.items())),
            "graph_quantization_pattern_counts": dict(
                sorted(pattern_counts.items())
            ),
        },
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
