import json
import math
from pathlib import Path


QUANTIZATION_PREFIX = "cdx:ai-ml:model:parameter:quantization:"


def fail(message):
    raise ValueError(message)


def normalize_dtype(value):
    return value.replace("32", "32").upper()


def parameter_key(row):
    return row["direction"], row["ordinal"]


def property_map(parameter):
    if "properties" in parameter["format"]:
        fail("Quantization properties must be siblings of format")
    rows = parameter.get("properties", [])
    if any(not isinstance(row.get("value"), str) for row in rows):
        fail("CycloneDX property values must be strings")
    if any(not row.get("name", "").startswith(QUANTIZATION_PREFIX) for row in rows):
        fail("Unexpected candidate property namespace")
    names = [row["name"].removeprefix(QUANTIZATION_PREFIX) for row in rows]
    if len(names) != len(set(names)):
        fail("Duplicate quantization property")
    return {name: row["value"] for name, row in zip(names, rows)}


def verify_example(root, filename, qualified_id):
    example = json.loads(filename.read_text(encoding="utf-8"))
    if example["status"] != "candidate-not-schema-validated":
        fail(f"Candidate status missing: {filename.name}")
    if example["evidenceGrade"] != "OBSERVED_CROSS_CHECKED":
        fail(f"Candidate evidence grade mismatch: {filename.name}")

    artifacts = json.loads(
        (root / "data/artifacts.json").read_text(encoding="utf-8")
    )["artifacts"]
    artifact = next(row for row in artifacts if row["qualified_id"] == qualified_id)
    if example["artifact"]["sha256"] != artifact["sha256"]:
        fail(f"Candidate artifact SHA-256 mismatch: {filename.name}")
    if example["artifact"]["source"] != artifact["download"]["url"]:
        fail(f"Candidate artifact source mismatch: {filename.name}")

    ledger = json.loads(
        (root / "data/interface-contracts.json").read_text(encoding="utf-8")
    )["parameters"]
    observed = {
        parameter_key(row): row
        for row in ledger if row["qualified_id"] == qualified_id
    }
    candidate_rows = []
    for direction, plural in (("input", "inputs"), ("output", "outputs")):
        for ordinal, row in enumerate(example["candidateModelParameters"][plural]):
            candidate_rows.append((direction, ordinal, row))
    if len(candidate_rows) != len(observed):
        fail(f"Candidate parameter count mismatch: {filename.name}")

    for direction, ordinal, candidate in candidate_rows:
        source = observed[(direction, ordinal)]
        if candidate["name"] != source["tensor_name"]:
            fail(f"Candidate parameter name mismatch: {filename.name}/{direction}")
        if candidate["shape"] != source["shape"]:
            fail(f"Candidate shape mismatch: {filename.name}/{direction}")
        if normalize_dtype(candidate["format"]["dataType"]) != source["dtype"]:
            fail(f"Candidate dtype mismatch: {filename.name}/{direction}")

        properties = property_map(candidate)
        quantization = source["quantization"]
        if quantization["status"] == "not_quantized":
            if properties:
                fail(f"Float candidate carries affine properties: {filename.name}/{direction}")
            continue
        expected_names = {"scheme", "granularity", "scale", "zeroPoint"}
        if set(properties) != expected_names:
            fail(f"Candidate property set mismatch: {filename.name}/{direction}")
        if properties["scheme"] != "affine_asymmetric":
            fail(f"Candidate scheme mismatch: {filename.name}/{direction}")
        expected_granularity = quantization["granularity"].replace("_", "-")
        if properties["granularity"] != expected_granularity:
            fail(f"Candidate granularity mismatch: {filename.name}/{direction}")
        scale = float(properties["scale"])
        if not math.isfinite(scale) or scale <= 0:
            fail(f"Candidate scale is not positive finite: {filename.name}/{direction}")
        if scale != quantization["scales"][0]:
            fail(f"Candidate scale mismatch: {filename.name}/{direction}")
        zero_point = int(properties["zeroPoint"])
        if str(zero_point) != properties["zeroPoint"]:
            fail(f"Candidate zero-point is not canonical decimal: {filename.name}/{direction}")
        if zero_point != quantization["zero_points"][0]:
            fail(f"Candidate zero-point mismatch: {filename.name}/{direction}")


def main():
    root = Path(__file__).resolve().parents[1]
    examples = [
        (
            root / "examples/cyclonedx/efficientnet-lite0-float32.candidate.json",
            "mediapipe-public/efficientnet-lite0-f32",
        ),
        (
            root / "examples/cyclonedx/efficientnet-lite0-int8-affine.candidate.json",
            "mediapipe-public/efficientnet-lite0-int8",
        ),
    ]
    for filename, qualified_id in examples:
        verify_example(root, filename, qualified_id)
    print(json.dumps({
        "status": "pass",
        "candidate_example_count": len(examples),
    }, indent=2))


if __name__ == "__main__":
    main()
