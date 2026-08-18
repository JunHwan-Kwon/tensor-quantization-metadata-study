import argparse
import hashlib
import json
import urllib.request
from importlib.metadata import version
from pathlib import Path

from jsonschema import Draft202012Validator


SPECIFICATION_COMMIT = "49a945618811213e55686a23fa63b287940071c6"
SPECIFICATION_PATH = "schema/2.0/model/cyclonedx-ai-ml-2.0.schema.json"
SPECIFICATION_SHA256 = (
    "9b346b22f6115c08a6d63cbb04b0774b93aa769dec08015e303b3956363dfa53"
)
SPECIFICATION_URL = (
    "https://raw.githubusercontent.com/CycloneDX/specification/"
    f"{SPECIFICATION_COMMIT}/{SPECIFICATION_PATH}"
)


CASES = [
    {
        "id": "empty-object",
        "instance": {},
        "current_expected_valid": True,
        "class": "cross-field-policy-gap",
    },
    {
        "id": "per-tensor-with-axis",
        "instance": {"granularity": "per-tensor", "axis": 0},
        "current_expected_valid": True,
        "class": "cross-field-policy-gap",
    },
    {
        "id": "per-tensor-with-group-size",
        "instance": {"granularity": "per-tensor", "groupSize": 32},
        "current_expected_valid": True,
        "class": "cross-field-policy-gap",
    },
    {
        "id": "per-channel-without-axis",
        "instance": {"granularity": "per-channel"},
        "current_expected_valid": True,
        "class": "cross-field-policy-gap",
    },
    {
        "id": "per-channel-with-group-size",
        "instance": {
            "granularity": "per-channel",
            "axis": 0,
            "groupSize": 32,
        },
        "current_expected_valid": True,
        "class": "cross-field-policy-gap",
    },
    {
        "id": "per-group-without-axis-or-group-size",
        "instance": {"granularity": "per-group"},
        "current_expected_valid": True,
        "class": "cross-field-policy-gap",
    },
    {
        "id": "zero-bits",
        "instance": {"bits": 0},
        "current_expected_valid": False,
        "class": "range-check",
    },
    {
        "id": "negative-axis",
        "instance": {"granularity": "per-channel", "axis": -1},
        "current_expected_valid": False,
        "class": "framework-compatibility-question",
    },
    {
        "id": "zero-group-size",
        "instance": {"granularity": "per-group", "groupSize": 0},
        "current_expected_valid": False,
        "class": "range-check",
    },
    {
        "id": "unknown-predefined-scheme",
        "instance": {"scheme": "affine_asymmetric"},
        "current_expected_valid": False,
        "class": "closed-enum-check",
    },
    {
        "id": "custom-scheme-object",
        "instance": {"scheme": {"name": "affine_asymmetric"}},
        "current_expected_valid": True,
        "class": "extension-check",
    },
    {
        "id": "fractional-bits",
        "instance": {"bits": 1.58},
        "current_expected_valid": True,
        "class": "numeric-domain-check",
    },
]


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def ensure_schema(path, allow_download):
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"Pinned schema is missing: {path}. Use --download."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            SPECIFICATION_URL,
            headers={"User-Agent": "tensor-quantization-metadata-study/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if sha256_bytes(payload) != SPECIFICATION_SHA256:
            raise ValueError("Downloaded CycloneDX schema SHA-256 mismatch")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if sha256_bytes(payload) != SPECIFICATION_SHA256:
        raise ValueError("CycloneDX schema SHA-256 mismatch")
    return payload


def validate_cases(definition):
    validator = Draft202012Validator(definition)
    rows = []
    for case in CASES:
        errors = sorted(
            validator.iter_errors(case["instance"]),
            key=lambda error: list(error.absolute_path),
        )
        valid = not errors
        if valid != case["current_expected_valid"]:
            raise ValueError(f"Unexpected validation result: {case['id']}")
        rows.append({
            **case,
            "current_valid": valid,
            "first_error": errors[0].message if errors else None,
        })
    return rows


def validate_context_cases(schema):
    definitions = schema["$defs"]
    contexts = {
        "model-parameter": {
            "$defs": definitions,
            "$ref": "#/$defs/modelParameter",
        },
        "model-properties": {
            "$defs": definitions,
            "$ref": "#/$defs/modelProperties",
        },
    }
    probes = [
        {
            "id": "parameter-negative-source-axis-with-static-rank",
            "context": "model-parameter",
            "instance": {
                "name": "image",
                "shape": [1, 3, 224, 224],
                "quantization": {
                    "granularity": "per-channel",
                    "axis": -1,
                },
            },
            "current_expected_valid": False,
        },
        {
            "id": "parameter-normalized-effective-axis-with-static-rank",
            "context": "model-parameter",
            "instance": {
                "name": "image",
                "shape": [1, 3, 224, 224],
                "quantization": {
                    "granularity": "per-channel",
                    "axis": 3,
                },
            },
            "current_expected_valid": True,
        },
        {
            "id": "parameter-axis-without-shape",
            "context": "model-parameter",
            "instance": {
                "name": "image",
                "quantization": {
                    "granularity": "per-channel",
                    "axis": 0,
                },
            },
            "current_expected_valid": True,
        },
        {
            "id": "model-properties-negative-source-axis",
            "context": "model-properties",
            "instance": {
                "quantization": {
                    "granularity": "per-channel",
                    "axis": -1,
                }
            },
            "current_expected_valid": False,
        },
        {
            "id": "model-properties-nonnegative-axis-without-weight-rank",
            "context": "model-properties",
            "instance": {
                "quantization": {
                    "granularity": "per-channel",
                    "axis": 0,
                }
            },
            "current_expected_valid": True,
        },
    ]
    rows = []
    for probe in probes:
        validator = Draft202012Validator(contexts[probe["context"]])
        errors = sorted(
            validator.iter_errors(probe["instance"]),
            key=lambda error: list(error.absolute_path),
        )
        valid = not errors
        if valid != probe["current_expected_valid"]:
            raise ValueError(
                f"Unexpected context validation result: {probe['id']}"
            )
        rows.append({
            **probe,
            "current_valid": valid,
            "first_error": errors[0].message if errors else None,
        })
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        type=Path,
        default=(
            root
            / "cache/cyclonedx-specification"
            / SPECIFICATION_COMMIT
            / "cyclonedx-ai-ml-2.0.schema.json"
        ),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    schema = json.loads(
        ensure_schema(args.schema.resolve(), args.download).decode("utf-8")
    )
    definition = schema["$defs"]["quantization"]
    cases = validate_cases(definition)
    context_cases = validate_context_cases(schema)
    class_counts = {}
    for row in cases:
        key = f"{row['class']}:{'valid' if row['current_valid'] else 'invalid'}"
        class_counts[key] = class_counts.get(key, 0) + 1

    result = {
        "schema": (
            "tensor_quantization_metadata_study."
            "cyclonedx_quantization_schema_audit.v1"
        ),
        "source": {
            "repository": "CycloneDX/specification",
            "commit": SPECIFICATION_COMMIT,
            "path": SPECIFICATION_PATH,
            "url": SPECIFICATION_URL,
            "sha256": SPECIFICATION_SHA256,
        },
        "validator": f"jsonschema {version('jsonschema')}",
        "quantization_definition_sha256": sha256_bytes(
            canonical_json_bytes(definition)
        ),
        "quantization_definition": definition,
        "case_count": len(cases),
        "valid_case_count": sum(row["current_valid"] for row in cases),
        "invalid_case_count": sum(not row["current_valid"] for row in cases),
        "class_result_counts": dict(sorted(class_counts.items())),
        "cases": cases,
        "quantization_usage_sites": [
            {
                "json_pointer": (
                    "/$defs/modelProperties/properties/quantization"
                ),
                "scope": "model weights as distributed",
                "reference": schema["$defs"]["modelProperties"]
                ["properties"]["quantization"]["$ref"],
                "direct_shape_property_available": False,
            },
            {
                "json_pointer": (
                    "/$defs/modelParameter/properties/quantization"
                ),
                "scope": "one named input or output parameter",
                "reference": schema["$defs"]["modelParameter"]
                ["properties"]["quantization"]["$ref"],
                "direct_shape_property_available": True,
                "shape_required": (
                    "shape"
                    in schema["$defs"]["modelParameter"].get(
                        "required", []
                    )
                ),
            },
        ],
        "normalization_probe_count": len(context_cases),
        "normalization_probes": context_cases,
        "axis_normalization_assessment": {
            "source_axis_formula_when_rank_known": (
                "effective_axis = source_axis if source_axis >= 0 "
                "else source_axis + tensor_rank"
            ),
            "shared_quantization_definition": True,
            "model_parameter_can_supply_rank_when_shape_present": True,
            "model_parameter_shape_is_required": False,
            "model_properties_has_direct_weight_tensor_shape": False,
            "finding": (
                "The draft rejects a negative source axis in both scopes. "
                "A model parameter can be normalized when its optional "
                "shape supplies rank; the model-level weight summary has "
                "no direct tensor rank and may aggregate heterogeneous "
                "weight tensors. The draft does not state whether axis is "
                "source-faithful or a normalized effective value."
            ),
        },
        "interpretation_boundary": (
            "The matrix records behavior of the pinned draft. It does not "
            "declare every accepted combination semantically valid. In "
            "particular, negative axis is rejected by this draft while ONNX "
            "defines negative axes for per-axis and blocked quantization."
        ),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
