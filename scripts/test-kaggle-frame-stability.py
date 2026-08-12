import importlib.util
from copy import deepcopy
from pathlib import Path


def load(root):
    path = root / "scripts/compare-kaggle-snapshot-frames.py"
    spec = importlib.util.spec_from_file_location("frame_stability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot():
    return {
        "schema": "tensor_quantization_metadata_study.kaggle_tflite_snapshot.v1",
        "snapshot_status": "COMPLETE",
        "acquisition": {"history_scope": "latest"},
        "population_frame": {
            "frame_type": "MODEL_ID_BOUNDED_PUBLIC_REGISTRY_COHORT",
            "model_id_min_inclusive": 700000,
            "model_id_max_exclusive": 725000,
            "listing_record_count": 10000,
            "listing_unique_model_count": 9999,
            "identical_duplicate_model_record_count": 1,
            "listing_min_model_id": 573836,
            "listing_max_model_id": 734017,
            "listing_ids_strictly_descending": True,
            "lower_boundary_covered": True,
            "upper_boundary_covered": True,
            "selected_model_count": 1,
        },
        "models": [{
            "model_ref": "owner/model",
            "instance_enumeration_status": "ASSESSED",
        }],
        "variations": [{
            "variation_ref": "owner/model/MODEL_FRAMEWORK_TF_LITE/int8",
            "current_version_status": "PRESENT",
            "version_enumeration_status": (
                "NOT_REQUESTED_CURRENT_FROM_INSTANCE"
            ),
            "versions": [{
                "version_number": 2,
                "is_latest": True,
                "file_enumeration_status": "ASSESSED",
                "files": [{"logical_path": "model.tflite", "size_bytes": 10}],
            }],
        }],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    module = load(root)
    before = snapshot()
    after = deepcopy(before)
    assert module.compare_frames(before, after)["status"] == "STABLE"
    after["variations"][0]["versions"][0]["files"][0]["size_bytes"] = 11
    changed = module.compare_frames(before, after)
    assert changed["status"] == "CHANGED_REQUIRES_NEW_ENUMERATION"
    assert len(changed["changed_variations"]) == 1
    after = deepcopy(before)
    after["models"][0]["instance_enumeration_status"] = (
        "NOT_PUBLICLY_ENUMERABLE"
    )
    changed = module.compare_frames(before, after)
    assert changed["status"] == "CHANGED_REQUIRES_NEW_ENUMERATION"
    assert len(changed["changed_model_enumeration_statuses"]) == 1
    after = deepcopy(before)
    after["population_frame"]["model_id_max_exclusive"] = 724999
    changed = module.compare_frames(before, after)
    assert changed["status"] == "CHANGED_REQUIRES_NEW_ENUMERATION"
    assert changed["population_frame_changed"]
    after = deepcopy(before)
    after["population_frame"]["listing_max_model_id"] += 10
    assert module.compare_frames(before, after)["status"] == "STABLE"
    history_before = deepcopy(before)
    history_before["acquisition"]["history_scope"] = "all"
    old_version = {
        "version_number": 1,
        "is_latest": False,
        "file_enumeration_status": "ASSESSED",
        "files": [{"logical_path": "model.tflite", "size_bytes": 8}],
    }
    history_before["variations"][0]["versions"].insert(0, old_version)
    history_after = deepcopy(history_before)
    history_after["variations"][0]["versions"][0]["files"][0][
        "size_bytes"
    ] = 9
    changed = module.compare_frames(history_before, history_after)
    assert changed["status"] == "CHANGED_REQUIRES_NEW_ENUMERATION"
    assert len(changed["changed_variations"]) == 1
    print("Kaggle frame-stability unit checks passed")


if __name__ == "__main__":
    main()
