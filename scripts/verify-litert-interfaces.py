import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_name(value):
    dtype = np.dtype(value)
    return "STRING" if dtype.kind in ("O", "S", "U") else dtype.name.upper()


def interpreter_parameters(filename):
    interpreter = Interpreter(model_path=str(filename))
    rows = []
    for direction, details in (
        ("input", interpreter.get_input_details()),
        ("output", interpreter.get_output_details()),
    ):
        for ordinal, detail in enumerate(details):
            quantization = detail.get("quantization_parameters") or {}
            rows.append({
                "direction": direction,
                "ordinal": ordinal,
                "tensor_index": int(detail["index"]),
                "dtype": dtype_name(detail["dtype"]),
                "shape": [
                    int(value) for value in detail["shape"].tolist()
                ],
                "scales": [
                    float(value)
                    for value in quantization.get("scales", [])
                ],
                "zero_points": [
                    int(value)
                    for value in quantization.get("zero_points", [])
                ],
                "quantized_dimension": int(
                    quantization.get("quantized_dimension", 0)
                ),
            })
    return rows


def expected_parameters(parameters):
    return [{
        "direction": parameter["direction"],
        "ordinal": parameter["ordinal"],
        "tensor_index": parameter["tensor_index"],
        "dtype": parameter["dtype"],
        "shape": parameter["shape"],
        "scales": parameter["quantization"].get("scales", []),
        "zero_points": parameter["quantization"].get("zero_points", []),
        "quantized_dimension":
            parameter["quantization"].get("quantized_dimension") or 0,
    } for parameter in parameters]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Cross-check the public interface ledger with LiteRT without "
            "allocating tensors or invoking inference."
        )
    )
    parser.add_argument(
        "--contracts",
        type=Path,
        default=root / "data/interface-contracts.json",
    )
    parser.add_argument("--cache-root", type=Path, default=root / "cache")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ledger = json.loads(args.contracts.read_text(encoding="utf-8"))
    expected = {}
    for parameter in ledger["parameters"]:
        expected.setdefault(parameter["artifact_sha256"], {
            "qualified_id": parameter["qualified_id"],
            "parameters": [],
        })["parameters"].append(parameter)
    for artifact in expected.values():
        artifact["parameters"].sort(
            key=lambda row: (row["direction"], row["ordinal"])
        )

    located = {}
    for filename in args.cache_root.resolve().rglob("*.tflite"):
        digest = sha256_file(filename)
        if digest in expected and digest not in located:
            located[digest] = filename

    verified = []
    mismatches = []
    for digest, filename in sorted(located.items()):
        artifact = expected[digest]
        actual = interpreter_parameters(filename)
        wanted = expected_parameters(artifact["parameters"])
        if actual == wanted:
            verified.append({
                "qualified_id": artifact["qualified_id"],
                "artifact_sha256": digest,
                "parameter_count": len(actual),
            })
        else:
            mismatches.append({
                "qualified_id": artifact["qualified_id"],
                "artifact_sha256": digest,
                "expected": wanted,
                "actual": actual,
            })

    missing = sorted(set(expected) - set(located))
    result = {
        "schema": "deepbom.public_litert_interface_verification.v1",
        "parser": f"ai-edge-litert {version('ai-edge-litert')}",
        "method": (
            "Interpreter.get_input_details/get_output_details only; "
            "no allocation or inference."
        ),
        "expected_artifact_count": len(expected),
        "located_artifact_count": len(located),
        "verified_artifact_count": len(verified),
        "verified_parameter_count": sum(
            row["parameter_count"] for row in verified
        ),
        "missing_artifact_count": len(missing),
        "mismatch_count": len(mismatches),
        "missing_artifact_sha256": missing,
        "verified": verified,
        "mismatches": mismatches,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if mismatches or (args.require_all and missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
