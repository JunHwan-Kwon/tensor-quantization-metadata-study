import importlib.util
import json
import tempfile
from pathlib import Path


def load_module(root):
    path = root / "scripts/snapshot-kaggle-tflite-population.py"
    spec = importlib.util.spec_from_file_location("kaggle_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Item:
    def __init__(self, value):
        self.value = value


class Response:
    def __init__(self, values, next_page_token=""):
        self.values = [Item(value) for value in values]
        self.next_page_token = next_page_token

    def to_dict(self, ignore_defaults=False):
        return {
            "values": [row.value for row in self.values],
            "nextPageToken": self.next_page_token,
        }


class HttpResponse:
    def __init__(self, status_code, retry_after=None):
        self.status_code = status_code
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = str(retry_after)


class HttpError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__(f"HTTP {status_code}")
        self.response = HttpResponse(status_code, retry_after)


class Version:
    def __init__(self, instance_id, variation_slug, number):
        self.owner_slug = "owner"
        self.model_slug = "model"
        self.model_instance_id = instance_id
        self.variation_slug = variation_slug
        self.version_number = number


class Instance:
    id = 10
    slug = "target"


class Framework:
    name = "MODEL_FRAMEWORK_TENSOR_FLOW_LITE"
    value = 17


class PytorchFramework:
    name = "MODEL_FRAMEWORK_PY_TORCH"
    value = 2


class SnapshotResponse:
    def __init__(self, **values):
        self.next_page_token = ""
        for key, value in values.items():
            setattr(self, key, value)

    def to_dict(self, ignore_defaults=False):
        return {
            key: len(value) if isinstance(value, list) else value
            for key, value in self.__dict__.items()
            if key != "next_page_token"
        } | {"nextPageToken": ""}


class SnapshotModel:
    def __init__(self, ref="owner/model", model_id=1):
        self.ref = ref
        self.id = model_id
        self.title = "Model"
        self.is_private = False
        self.publish_time = None
        self.update_time = None


class SnapshotInstance:
    id = 10
    slug = "target"
    framework = Framework()
    version_number = 2
    version_id = 222


class PytorchInstance(SnapshotInstance):
    framework = PytorchFramework()


class SnapshotFile:
    name = "model.tflite"
    size = 123
    creation_date = None


class LatestBackend:
    def list_models(self, token, page_size):
        model = SnapshotModel()
        model.instances = [SnapshotInstance()]
        return SnapshotResponse(models=[model])

    def list_instances(self, owner, model, token, page_size):
        return SnapshotResponse(instances=[SnapshotInstance()])

    def list_versions(self, owner, model, instance, token, page_size):
        raise AssertionError("Latest snapshot called the version-list endpoint")

    def list_files(
        self, owner, model, instance, version_number, token, page_size
    ):
        assert version_number == 2
        return SnapshotResponse(files=[SnapshotFile()])


class ForbiddenFileBackend(LatestBackend):
    def list_files(
        self, owner, model, instance, version_number, token, page_size
    ):
        raise HttpError(403)


class NonTfliteBackend(LatestBackend):
    def list_models(self, token, page_size):
        model = SnapshotModel()
        model.instances = [PytorchInstance()]
        return SnapshotResponse(models=[model])

    def list_instances(self, owner, model, token, page_size):
        return SnapshotResponse(instances=[PytorchInstance()])

    def list_files(
        self, owner, model, instance, version_number, token, page_size
    ):
        raise AssertionError("Non-TFLite framework requested a file listing")


class CheckpointBackend(LatestBackend):
    def __init__(self, fail_second):
        self.fail_second = fail_second

    def list_models(self, token, page_size):
        models = [
            SnapshotModel("owner/first", 1),
            SnapshotModel("owner/second", 2),
        ]
        for model in models:
            model.instances = [SnapshotInstance()]
        return SnapshotResponse(models=models)

    def list_instances(self, owner, model, token, page_size):
        if model == "second" and self.fail_second:
            raise HttpError(401)
        return SnapshotResponse(instances=[SnapshotInstance()])


def main():
    root = Path(__file__).resolve().parents[1]
    module = load_module(root)
    responses = {
        "": Response([1, 2], "second"),
        "second": Response([3], ""),
    }
    rows, pages = module.collect_pages(
        lambda token, size: responses[token],
        lambda response: response.values,
        200,
    )
    assert [row.value for row in rows] == [1, 2, 3]
    assert [row["item_count"] for row in pages] == [2, 1]
    assert all(len(row["response_sha256"]) == 64 for row in pages)
    assert module.is_tflite_path("model.TFLITE")
    assert not module.is_tflite_path("model.tflite.zip")

    looping = {"": Response([], "again"), "again": Response([], "again")}
    try:
        module.collect_pages(
            lambda token, size: looping[token],
            lambda response: response.values,
            200,
        )
    except ValueError as error:
        assert "repeated" in str(error)
    else:
        raise AssertionError("Repeated pagination token was accepted")

    versions, foreign = module.partition_versions(
        [Version(10, "target", 1), Version(11, "foreign", 1)],
        "owner",
        "model",
        Instance(),
    )
    assert [row.variation_slug for row in versions] == ["target"]
    assert [row.variation_slug for row in foreign] == ["foreign"]

    selected, frame = module.select_model_candidates(
        [
            SnapshotModel("owner/newer", 725001),
            SnapshotModel("owner/upper", 725000),
            SnapshotModel("owner/in-two", 724999),
            SnapshotModel("owner/in-one", 700000),
            SnapshotModel("owner/older", 699999),
        ],
        700000,
        725000,
        None,
    )
    assert [row["model_id"] for row in selected] == [724999, 700000]
    assert frame["lower_boundary_covered"]
    assert frame["upper_boundary_covered"]
    assert frame["selected_model_count"] == 2

    duplicate = SnapshotModel("owner/in-two", 724999)
    selected, duplicate_frame = module.select_model_candidates(
        [
            SnapshotModel("owner/newer", 725001),
            SnapshotModel("owner/in-two", 724999),
            duplicate,
            SnapshotModel("owner/in-one", 700000),
            SnapshotModel("owner/older", 699999),
        ],
        700000,
        725000,
        None,
    )
    assert [row["model_id"] for row in selected] == [724999, 700000]
    assert duplicate_frame["identical_duplicate_model_record_count"] == 1
    conflicting = SnapshotModel("owner/conflict", 724999)
    try:
        module.select_model_candidates(
            [
                SnapshotModel("owner/newer", 725001),
                SnapshotModel("owner/in-two", 724999),
                conflicting,
                SnapshotModel("owner/older", 699999),
            ],
            700000,
            725000,
            None,
        )
    except ValueError as error:
        assert "conflicting facts" in str(error)
    else:
        raise AssertionError("Conflicting duplicate model facts were accepted")

    with tempfile.TemporaryDirectory() as directory:
        writer = module.SnapshotWriter(Path(directory))
        models, variations, failures, frame = module.snapshot(
            LatestBackend(), 200, "latest", None, writer
        )
        assert len(models) == 1
        assert len(variations) == 1
        assert not failures
        assert frame["frame_type"] == "SERVICE_LISTING_WINDOW"
        variation = variations[0]
        assert variation["enumerated_version_count"] == 1
        assert variation["foreign_version_record_count"] == 0
        assert variation["latest_version_contains_tflite"]
        assert variation["versions"][0]["version_id"] == 222
        assert variation["versions"][0]["version_source"] == (
            "model_list_embedded_current_version"
        )

    with tempfile.TemporaryDirectory() as directory:
        writer = module.SnapshotWriter(Path(directory))
        _, variations, failures, _ = module.snapshot(
            ForbiddenFileBackend(), 200, "latest", None, writer
        )
        assert len(failures) == 1
        assert failures[0]["status"] == "NOT_PUBLICLY_ENUMERABLE"
        version = variations[0]["versions"][0]
        assert version["file_enumeration_status"] == (
            "NOT_PUBLICLY_ENUMERABLE"
        )
        assert version["file_count"] is None
        assert not variations[0]["latest_version_contains_tflite"]

    with tempfile.TemporaryDirectory() as directory:
        writer = module.SnapshotWriter(Path(directory))
        _, variations, failures, _ = module.snapshot(
            NonTfliteBackend(), 200, "latest", None, writer
        )
        assert not failures
        variation = variations[0]
        assert not variation["declared_tflite_framework"]
        assert variation["latest_version_contains_tflite"] is None
        version = variation["versions"][0]
        assert version["file_enumeration_status"] == (
            "NOT_REQUESTED_NON_TFLITE_FRAMEWORK"
        )
        assert version["file_count"] is None

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        raw_pages = directory / "raw"
        checkpoint = directory / "checkpoint.json"
        writer = module.SnapshotWriter(raw_pages)
        try:
            module.snapshot(
                CheckpointBackend(fail_second=True),
                200,
                "latest",
                None,
                writer,
                checkpoint_path=checkpoint,
                checkpoint_context={"test": True},
                checkpoint_interval=1,
            )
        except HttpError as error:
            assert error.response.status_code == 401
        else:
            raise AssertionError("Checkpoint interruption was not raised")
        state = json.loads(checkpoint.read_text(encoding="ascii"))
        assert state["next_model_index"] == 1
        restored_writer = module.SnapshotWriter(
            raw_pages, state["raw_response_pages"]
        )
        models, variations, failures, _ = module.snapshot(
            CheckpointBackend(fail_second=False),
            200,
            "latest",
            None,
            restored_writer,
            checkpoint_path=checkpoint,
            checkpoint_context={"test": True},
            resume_state=state,
            checkpoint_interval=1,
        )
        assert [row["model_ref"] for row in models] == [
            "owner/first",
            "owner/second",
        ]
        assert len(variations) == 2
        assert not failures

    calls = []
    sleeps = []

    def rate_limited_operation():
        calls.append(len(calls) + 1)
        if len(calls) < 3:
            raise HttpError(429, retry_after=3)
        return "complete"

    controller = module.RequestController(
        request_interval_seconds=0,
        max_retries=3,
        retry_base_seconds=2,
        retry_cap_seconds=10,
        clock=lambda: 0,
        sleeper=sleeps.append,
    )
    assert controller.call("test", rate_limited_operation) == "complete"
    assert controller.request_count == 3
    assert [row["http_status"] for row in controller.retry_events] == [429, 429]
    assert sleeps == [3.0, 4.0]

    no_retry = module.RequestController(
        request_interval_seconds=0,
        max_retries=3,
        retry_base_seconds=2,
        retry_cap_seconds=10,
        clock=lambda: 0,
        sleeper=lambda delay: None,
    )
    try:
        no_retry.call("test", lambda: (_ for _ in ()).throw(HttpError(401)))
    except HttpError:
        assert no_retry.request_count == 1
        assert not no_retry.retry_events
    else:
        raise AssertionError("Non-retryable status was retried")
    print("Kaggle snapshot unit checks passed")


if __name__ == "__main__":
    main()
