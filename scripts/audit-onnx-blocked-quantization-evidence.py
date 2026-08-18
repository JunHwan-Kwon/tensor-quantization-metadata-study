import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ONNX_VERSION = "1.22.0"
SOURCE_URL = (
    "https://github.com/onnx/onnx/blob/v1.22.0/"
    "onnx/backend/test/case/node/quantizelinear.py"
)
OPERATOR_URL = "https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html"
FIXTURES = {
    "test_quantizelinear_blocked_asymmetric": (
        "d2ab1e36f03087e91468adc97590757e07b183f4e46bfeefa53899a0d3c0ed87"
    ),
    "test_quantizelinear_blocked_symmetric": (
        "a02c7614276ebe6029db1519a0c779a4c96f646f6f8bf8ad2103488a400a86fd"
    ),
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def tensor_shape(value_info):
    return [
        int(dimension.dim_value)
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def load_tensor(path):
    tensor = TensorProto()
    tensor.ParseFromString(path.read_bytes())
    return numpy_helper.to_array(tensor)


def attributes(node):
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def normalize_axis(axis, rank):
    effective = axis if axis >= 0 else axis + rank
    if effective < 0 or effective >= rank:
        raise ValueError(f"Axis {axis} is invalid for rank {rank}")
    return effective


def quantized_dtype(model, attrs, zero_point):
    if zero_point is not None:
        return zero_point.dtype
    output_dtype = int(attrs.get("output_dtype", 0))
    if output_dtype == TensorProto.INT16:
        return np.dtype(np.int16)
    raise ValueError("Cannot derive quantized dtype from fixture")


def assess_fixture(root, name, expected_sha256):
    directory = root / name
    model_path = directory / "model.onnx"
    payload = model_path.read_bytes()
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        raise ValueError(f"Unexpected official ONNX fixture SHA-256: {name}")
    model = onnx.load_from_string(payload)
    if len(model.graph.node) != 1:
        raise ValueError(f"Expected one node in {name}")
    node = model.graph.node[0]
    if node.op_type != "QuantizeLinear" or node.domain not in ("", "ai.onnx"):
        raise ValueError(f"Unexpected operator in {name}")
    attrs = attributes(node)
    axis = int(attrs["axis"])
    block_size = int(attrs["block_size"])
    input_shapes = {
        value_info.name: tensor_shape(value_info)
        for value_info in model.graph.input
    }
    x_shape = input_shapes[node.input[0]]
    scale_shape = input_shapes[node.input[1]]
    rank = len(x_shape)
    effective_axis = normalize_axis(axis, rank)
    if len(scale_shape) != rank:
        raise ValueError(f"Scale rank mismatch in {name}")
    if any(
        x_shape[index] != scale_shape[index]
        for index in range(rank)
        if index != effective_axis
    ):
        raise ValueError(f"Non-axis shape mismatch in {name}")
    expected_scale_axis = math.ceil(
        x_shape[effective_axis] / block_size
    )
    if scale_shape[effective_axis] != expected_scale_axis:
        raise ValueError(f"Blocked scale shape mismatch in {name}")
    minimum_block_size = math.ceil(
        x_shape[effective_axis] / scale_shape[effective_axis]
    )
    maximum_block_size = (
        math.ceil(
            x_shape[effective_axis]
            / (scale_shape[effective_axis] - 1)
        )
        - 1
    )
    if not minimum_block_size <= block_size <= maximum_block_size:
        raise ValueError(f"Block size outside ONNX range in {name}")

    dataset = directory / "test_data_set_0"
    x = load_tensor(dataset / "input_0.pb")
    scale = load_tensor(dataset / "input_1.pb")
    zero_point = (
        load_tensor(dataset / "input_2.pb")
        if (dataset / "input_2.pb").is_file()
        else None
    )
    expected_output = load_tensor(dataset / "output_0.pb")
    expanded_scale = np.repeat(
        scale, repeats=block_size, axis=effective_axis
    )
    axis_slice = [slice(None)] * rank
    axis_slice[effective_axis] = slice(0, x_shape[effective_axis])
    expanded_scale = expanded_scale[tuple(axis_slice)]
    expanded_zero_point = None
    if zero_point is not None:
        expanded_zero_point = np.repeat(
            zero_point, repeats=block_size, axis=effective_axis
        )[tuple(axis_slice)]
    dtype = quantized_dtype(model, attrs, zero_point)
    limits = np.iinfo(dtype)
    calculated = np.rint(x / expanded_scale)
    if expanded_zero_point is not None:
        calculated = calculated + expanded_zero_point
    calculated = np.clip(calculated, limits.min, limits.max).astype(dtype)
    numerical_exact = bool(np.array_equal(calculated, expected_output))
    if not numerical_exact:
        raise ValueError(f"Numerical output mismatch in {name}")

    negative_source_axis = effective_axis - rank
    if normalize_axis(negative_source_axis, rank) != effective_axis:
        raise ValueError("Negative-axis normalization mismatch")
    return {
        "id": name,
        "model_sha256": digest,
        "model_size_bytes": len(payload),
        "opset_imports": {
            item.domain or "ai.onnx": int(item.version)
            for item in model.opset_import
        },
        "operator": node.op_type,
        "source_axis": axis,
        "equivalent_negative_source_axis": negative_source_axis,
        "effective_axis": effective_axis,
        "block_size": block_size,
        "input_shape": x_shape,
        "scale_shape": scale_shape,
        "zero_point_shape": (
            list(zero_point.shape) if zero_point is not None else None
        ),
        "output_shape": list(expected_output.shape),
        "block_size_accepted_range": [
            minimum_block_size,
            maximum_block_size,
        ],
        "derived_scale_axis_length": expected_scale_axis,
        "scale_replication_conservation": (
            scale_shape[effective_axis] * block_size
            >= x_shape[effective_axis]
        ),
        "numerical_output_element_count": int(expected_output.size),
        "numerical_output_exact": numerical_exact,
        "cyclonedx_normalized_projection": {
            "method": "onnx-quantize-linear",
            "scheme": "affine" if zero_point is not None else "symmetric",
            "granularity": "per-group",
            "groupSize": block_size,
            "axis": effective_axis,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if onnx.__version__ != ONNX_VERSION:
        raise ValueError(
            f"Expected onnx {ONNX_VERSION}, found {onnx.__version__}"
        )
    package_root = Path(onnx.__file__).resolve().parent
    fixture_root = package_root / "backend/test/data/node"
    rows = [
        assess_fixture(fixture_root, name, digest)
        for name, digest in FIXTURES.items()
    ]
    result = {
        "schema": (
            "tensor_quantization_metadata_study."
            "onnx_blocked_quantization_evidence.v1"
        ),
        "parser": f"onnx {onnx.__version__}",
        "source": {
            "repository": "onnx/onnx",
            "tag": f"v{ONNX_VERSION}",
            "generator_url": SOURCE_URL,
            "operator_specification_url": OPERATOR_URL,
            "fixture_distribution": "onnx PyPI wheel backend test data",
        },
        "fixture_count": len(rows),
        "numerically_exact_fixture_count": sum(
            row["numerical_output_exact"] for row in rows
        ),
        "blocked_shape_conservation_pass_count": sum(
            row["scale_replication_conservation"] for row in rows
        ),
        "fixtures": rows,
        "axis_normalization_assessment": {
            "formula": (
                "effective_axis = source_axis if source_axis >= 0 "
                "else source_axis + tensor_rank"
            ),
            "rank": 2,
            "source_axis": -1,
            "effective_axis": 1,
            "status": "exact_for_both_official_fixtures",
        },
        "interpretation_boundary": (
            "These are official ONNX backend conformance fixtures and prove "
            "that blocked QuantizeLinear uses an axis and a positive block "
            "size with the recorded shape relation. The CycloneDX objects "
            "are normalized projections of that source contract. This does "
            "not establish that GGUF encoding block widths have identical "
            "semantics, or that every per-group method is affine."
        ),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
