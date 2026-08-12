import importlib.util
import io
import json
import tarfile
import tempfile
from pathlib import Path


def load(root, name):
    path = root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_snapshot():
    def version(number, paths, latest=False):
        return {
            "version_number": number,
            "is_latest": latest,
            "files": [{
                "logical_path": path,
                "size_bytes": len(path.encode("ascii")),
                "is_tflite": path.lower().endswith(".tflite"),
            } for path in paths],
        }
    return {
        "schema": "tensor_quantization_metadata_study.kaggle_tflite_snapshot.v1",
        "snapshot_status": "COMPLETE",
        "acquisition": {"history_scope": "all"},
        "variations": [{
            "variation_ref": "owner/model/MODEL_FRAMEWORK_TF_LITE/int8",
            "enumerated_version_count": 3,
            "versions": [
                version(1, ["a.tflite"]),
                version(2, ["a.tflite", "b.tflite"]),
                version(3, ["b.tflite"], True),
            ],
        }],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    pairs_module = load(root, "build-kaggle-revision-pairs.py")
    materialize = load(root, "materialize-kaggle-tflite-snapshot.py")
    pairs = pairs_module.build_pairs(fixture_snapshot())
    assert len(pairs) == 4
    assert sum(
        row["evidence_class"] == "PENDING_ARTIFACT_ASSESSMENT"
        for row in pairs
    ) == 2
    assert sum(
        row["evidence_class"] == "PATH_ADDED_OR_REMOVED"
        for row in pairs
    ) == 2
    assert materialize.normalized_member_name("./models/a.tflite") == (
        "models/a.tflite"
    )
    try:
        materialize.normalized_member_name("../escape.tflite")
    except ValueError:
        pass
    else:
        raise AssertionError("Archive path traversal was accepted")

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        archive = directory / "fixture.tar.gz"
        payload = b"model-bytes"
        with tarfile.open(archive, "w:gz") as bundle:
            info = tarfile.TarInfo("./model.tflite")
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
        rows = materialize.extract_expected(
            archive,
            [{"logical_path": "model.tflite", "size_bytes": len(payload)}],
            directory / "cache" / "models",
        )
        assert rows[0]["status"] == "ASSESSED_DOWNLOAD"
        assert rows[0]["artifact_size_bytes"] == len(payload)
        assert len(rows[0]["artifact_sha256"]) == 64
    print("Kaggle acquisition pipeline unit checks passed")


if __name__ == "__main__":
    main()
