import importlib.util
import json
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper


def load_scanner(root):
    path = root / "scripts/scan-onnx-interface-contracts.py"
    spec = importlib.util.spec_from_file_location("onnx_contract_scanner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def main():
    root = Path(__file__).resolve().parents[1]
    scanner = load_scanner(root)
    uint8_input = helper.make_tensor_value_info(
        "x", TensorProto.UINT8, [1, 3]
    )
    float_input = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [1, 3]
    )
    uint8_output = helper.make_tensor_value_info(
        "y", TensorProto.UINT8, [1, 3]
    )
    scale = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    zero_point = np.asarray([0, 0, 0], dtype=np.uint8)
    constants = {"s": scale, "z": zero_point}

    dq = helper.make_node(
        "DequantizeLinear", ["x", "s", "z"], ["y"], axis=1
    )
    derived = scanner.input_contract(uint8_input, [dq], constants)
    expect("per-axis status", derived["status"], "DERIVED_STATIC_GRAPH")
    expect("per-axis granularity", derived["granularity"], "per_axis")
    expect("per-axis axis", derived["axis"], 1)
    expect("per-axis cardinality", derived["cardinality_status"], "valid")

    dynamic = scanner.input_contract(uint8_input, [dq], {"z": zero_point})
    expect("dynamic scale", dynamic["status"], "DYNAMIC_OR_UNBOUND")

    identity = helper.make_node("Identity", ["x"], ["other"])
    ambiguous = scanner.input_contract(
        uint8_input, [dq, identity], constants
    )
    expect(
        "mixed direct consumers",
        ambiguous["status"],
        "AMBIGUOUS_MULTIPLE_CONTRACTS",
    )

    unsupported = scanner.input_contract(uint8_input, [identity], constants)
    expect("unsupported consumer", unsupported["status"], "UNSUPPORTED")

    not_quantized = scanner.input_contract(float_input, [dq], constants)
    expect("float graph input", not_quantized["status"], "NOT_QUANTIZED")

    q = helper.make_node(
        "QuantizeLinear", ["real", "s", "z"], ["y"], axis=1
    )
    output = scanner.output_contract(uint8_output, q, constants)
    expect("quantized output", output["status"], "DERIVED_STATIC_GRAPH")
    expect("quantized output axis", output["axis"], 1)

    bad_scale = np.asarray([0.1, 0.2], dtype=np.float32)
    bad_zero = np.asarray([0, 0], dtype=np.uint8)
    invalid = scanner.input_contract(
        uint8_input,
        [dq],
        {"s": bad_scale, "z": bad_zero},
    )
    expect(
        "invalid vector length",
        invalid["cardinality_status"],
        "invalid_length",
    )

    dq_other = helper.make_node(
        "DequantizeLinear", ["x", "s2", "z"], ["y2"], axis=1
    )
    distinct = scanner.input_contract(
        uint8_input,
        [dq, dq_other],
        {"s": scale, "s2": scale * 2, "z": zero_point},
    )
    expect(
        "different direct contracts",
        distinct["status"],
        "AMBIGUOUS_MULTIPLE_CONTRACTS",
    )

    print(json.dumps({
        "status": "pass",
        "case_count": 8,
        "covered_statuses": [
            "NOT_QUANTIZED",
            "DERIVED_STATIC_GRAPH",
            "DYNAMIC_OR_UNBOUND",
            "AMBIGUOUS_MULTIPLE_CONTRACTS",
            "UNSUPPORTED",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
