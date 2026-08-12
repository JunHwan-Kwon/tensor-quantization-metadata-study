import json
from collections import Counter
from pathlib import Path


def fail(message):
    raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "data/onnx-pilot-manifest.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (root / "data/onnx-pilot-results.json").read_text(encoding="utf-8")
    )
    expected = {row["id"]: row for row in manifest["artifacts"]}
    observed = {row["id"]: row for row in result["artifacts"]}
    if len(expected) != 15 or len(observed) != 15:
        fail("The ONNX pilot must contain exactly 15 artifacts")
    if set(expected) != set(observed):
        fail("Manifest and result artifact identifiers differ")
    for artifact_id, source in expected.items():
        row = observed[artifact_id]
        for key in ("repo", "revision", "path", "variant", "size_bytes", "sha256"):
            if row[key] != source[key]:
                fail(f"Manifest mismatch for {artifact_id}: {key}")
        if row["external_initializer_count"] != 0:
            fail(f"Unresolved external data in pilot artifact {artifact_id}")

    parameters = [
        parameter
        for artifact in result["artifacts"]
        for parameter in artifact["parameters"]
    ]
    status_counts = Counter(
        parameter["contract"]["status"] for parameter in parameters
    )
    pattern_counts = Counter(
        artifact["graph_quantization_pattern"]
        for artifact in result["artifacts"]
    )
    if status_counts != {"NOT_QUANTIZED": 39}:
        fail(f"Unexpected interface status counts: {status_counts}")
    if pattern_counts != {"NONE": 5, "MIXED:QOPERATOR+DYNAMIC": 10}:
        fail(f"Unexpected graph pattern counts: {pattern_counts}")
    if result["summary"]["parameter_count"] != len(parameters):
        fail("ONNX summary parameter count mismatch")
    if result["summary"]["contract_status_counts"] != dict(status_counts):
        fail("ONNX summary status counts mismatch")
    if result["summary"]["graph_quantization_pattern_counts"] != dict(pattern_counts):
        fail("ONNX summary graph pattern counts mismatch")
    print(json.dumps({
        "status": "pass",
        "artifact_count": len(observed),
        "parameter_count": len(parameters),
        "contract_status_counts": dict(status_counts),
        "graph_quantization_pattern_counts": dict(pattern_counts),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
