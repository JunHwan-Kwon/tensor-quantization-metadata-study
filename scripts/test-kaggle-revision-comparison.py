import importlib.util
from pathlib import Path


def load(root):
    path = root / "scripts/compare-kaggle-tflite-revisions.py"
    spec = importlib.util.spec_from_file_location("revision_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact(identifier, digest, initializer_digest="a"):
    return {
        "published_file_id": identifier,
        "status": "ASSESSED",
        "artifact_sha256": digest,
        "initializers": {
            "entries": [{"match_key": "weight:0", "data_sha256": initializer_digest}]
        },
    }


def parameter(identifier, scale, dtype="INT8"):
    return {
        "published_file_id": identifier,
        "direction": "input",
        "ordinal": 0,
        "dtype": dtype,
        "shape": [1, 4],
        "scales": [scale],
        "zero_points": [0],
        "quantized_dimension": 0,
        "affine": {"status": "COMPLETE", "granularity": "PER_TENSOR"},
    }


def main():
    root = Path(__file__).resolve().parents[1]
    module = load(root)
    variation = "owner/model/MODEL_FRAMEWORK_TF_LITE/int8"
    old_id = f"{variation}/1:model.tflite"
    new_id = f"{variation}/2:model.tflite"
    pair = {
        "variation_ref": variation,
        "logical_path": "model.tflite",
        "old_version_number": 1,
        "new_version_number": 2,
        "evidence_class": "PENDING_ARTIFACT_ASSESSMENT",
    }
    artifacts = {
        old_id: artifact(old_id, "0" * 64),
        new_id: artifact(new_id, "1" * 64),
    }
    parameters = {
        old_id: [parameter(old_id, 0.1)],
        new_id: [parameter(new_id, 0.2)],
    }
    row = module.compare_pair(pair, artifacts, parameters)
    assert row["classification"] == "AFFINE_CHANGED_INITIALIZERS_IDENTICAL"
    artifacts[new_id] = artifact(new_id, "1" * 64, "b")
    row = module.compare_pair(pair, artifacts, parameters)
    assert row["classification"] == "AFFINE_CHANGED_INITIALIZERS_CHANGED"
    parameters[new_id] = [parameter(new_id, 0.1, "UINT8")]
    row = module.compare_pair(pair, artifacts, parameters)
    assert row["classification"] == "DTYPE_OR_SHAPE_CHANGED"
    artifacts[new_id] = artifact(new_id, "0" * 64)
    parameters[new_id] = [parameter(new_id, 0.1)]
    row = module.compare_pair(pair, artifacts, parameters)
    assert row["classification"] == "ARTIFACT_IDENTICAL"

    parameters[new_id] = [
        parameter(new_id, 0.1),
        parameter(new_id, 0.2),
    ]
    try:
        module.compare_pair(pair, artifacts, parameters)
    except ValueError as error:
        assert "Duplicate external parameter key" in str(error)
    else:
        raise AssertionError("Duplicate external parameter key was accepted")

    parameters[new_id] = [parameter(new_id, 0.1)]
    artifacts[new_id]["initializers"]["entries"].append(
        {"match_key": "weight:0", "data_sha256": "b"}
    )
    try:
        module.compare_pair(pair, artifacts, parameters)
    except ValueError as error:
        assert "Duplicate initializer match_key" in str(error)
    else:
        raise AssertionError("Duplicate initializer match_key was accepted")
    print("Kaggle revision-comparison unit checks passed")


if __name__ == "__main__":
    main()
