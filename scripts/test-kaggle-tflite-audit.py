import importlib.util
from pathlib import Path


def load(root):
    path = root / "scripts/audit-kaggle-tflite-snapshot.py"
    spec = importlib.util.spec_from_file_location("kaggle_tflite_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter(scales, zero_points, dimension=0, shape=None):
    return {
        "dtype": "INT8",
        "shape": [1, 4] if shape is None else shape,
        "scales": scales,
        "zero_points": zero_points,
        "quantized_dimension": dimension,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    module = load(root)
    assert module.affine_assessment(parameter([], []))["status"] == (
        "NOT_QUANTIZED"
    )
    per_tensor = module.affine_assessment(parameter([0.25], [0]))
    assert per_tensor["status"] == "COMPLETE"
    assert per_tensor["granularity"] == "PER_TENSOR"
    per_axis = module.affine_assessment(
        parameter([0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0], 1)
    )
    assert per_axis["status"] == "COMPLETE"
    assert per_axis["granularity"] == "PER_AXIS"
    negative_axis = module.affine_assessment(
        parameter([0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0], -1)
    )
    assert negative_axis["status"] == "INVALID_AXIS"
    assert module.affine_assessment(
        parameter([0.1, 0.2], [0, 0], 1)
    )["status"] == "INVALID_AXIS_CARDINALITY"
    assert module.affine_assessment(
        parameter([0.0], [0])
    )["status"] == "INVALID_SCALE"
    rows = []
    for digest, scale in (("a", 0.1), ("b", 0.2), ("c", 0.2)):
        row = parameter([scale], [0])
        row.update({
            "artifact_sha256": digest * 64,
            "direction": "input",
            "affine": module.affine_assessment(row),
        })
        rows.append(row)
    groups = module.signature_groups(rows)
    assert len(groups) == 1
    assert groups[0]["unique_artifact_count"] == 3
    assert groups[0]["unique_contract_count"] == 2
    assert groups[0]["ambiguous"]
    print("Kaggle TFLite audit unit checks passed")


if __name__ == "__main__":
    main()
