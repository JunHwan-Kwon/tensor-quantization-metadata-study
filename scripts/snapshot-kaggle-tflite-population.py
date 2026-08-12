import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "tensor_quantization_metadata_study.kaggle_tflite_snapshot.v1"
CHECKPOINT_SCHEMA = (
    "tensor_quantization_metadata_study.kaggle_tflite_checkpoint.v1"
)
KAGGLE_VERSION = "2.2.4"


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_tflite_path(name):
    return isinstance(name, str) and name.lower().endswith(".tflite")


def collect_pages(fetch_page, select_items, page_size):
    pages = []
    items = []
    token = ""
    seen_tokens = set()
    while True:
        if token in seen_tokens:
            raise ValueError(f"Pagination token repeated: {token!r}")
        seen_tokens.add(token)
        response = fetch_page(token, page_size)
        response_dict = response.to_dict(ignore_defaults=False)
        page_items = list(select_items(response) or [])
        next_token = response.next_page_token or ""
        pages.append({
            "request_page_token": token,
            "response_sha256": sha256_text(canonical_json(response_dict)),
            "item_count": len(page_items),
            "next_page_token": next_token,
            "raw_response": response_dict,
        })
        items.extend(page_items)
        if not next_token:
            break
        token = next_token
    return items, pages


def enum_name(value):
    return value.name if hasattr(value, "name") else str(value)


def instance_fact(instance):
    framework = getattr(instance, "framework", None)
    return {
        "model_instance_id": getattr(instance, "id", None),
        "variation_slug": getattr(instance, "slug", None),
        "framework": enum_name(framework),
        "framework_value": (
            int(framework.value) if hasattr(framework, "value") else None
        ),
        "version_number": getattr(instance, "version_number", None),
        "version_id": getattr(instance, "version_id", None),
    }


def instance_identity(instance):
    return tuple(instance[key] for key in (
        "model_instance_id",
        "variation_slug",
        "framework",
        "framework_value",
        "version_number",
        "version_id",
    ))


def is_tflite_framework(value):
    return value in {
        "MODEL_FRAMEWORK_TF_LITE",
        "MODEL_FRAMEWORK_TENSOR_FLOW_LITE",
    }


def stable_object(value):
    return value.to_dict(ignore_defaults=False)


def model_fact(model):
    return {
        "model_ref": model.ref,
        "model_id": model.id,
        "title": model.title,
        "is_private": model.is_private,
        "publish_time": (
            model.publish_time.isoformat()
            if model.publish_time is not None else None
        ),
        "update_time": (
            model.update_time.isoformat()
            if model.update_time is not None else None
        ),
        "embedded_instances": [
            instance_fact(instance)
            for instance in (getattr(model, "instances", None) or [])
        ],
    }


def select_model_candidates(
    models, model_id_min, model_id_max_exclusive, model_limit
):
    listed = [model_fact(model) for model in models]
    if not listed:
        raise ValueError("Model listing is empty")
    listed_ids = [row["model_id"] for row in listed]
    if any(value is None for value in listed_ids):
        raise ValueError("Model listing contains a missing model ID")
    candidates = []
    by_id = {}
    identical_duplicate_count = 0
    for row in listed:
        model_id = int(row["model_id"])
        previous = by_id.get(model_id)
        if previous is None:
            by_id[model_id] = row
            candidates.append(row)
        elif previous == row:
            identical_duplicate_count += 1
        else:
            raise ValueError(
                f"Model listing contains conflicting facts for ID {model_id}"
            )
    ids = [int(row["model_id"]) for row in candidates]
    if any(value is None for value in ids):
        raise ValueError("Model listing contains a missing model ID")
    strictly_descending = all(
        ids[index] > ids[index + 1]
        for index in range(len(ids) - 1)
    )
    bounded = (
        model_id_min is not None and model_id_max_exclusive is not None
    )
    if bounded:
        if model_id_min >= model_id_max_exclusive:
            raise ValueError("Model ID lower bound must be below upper bound")
        if not strictly_descending:
            raise ValueError(
                "Model listing is not strictly descending by model ID"
            )
        lower_covered = min(ids) < model_id_min
        upper_covered = max(ids) >= model_id_max_exclusive
        if not lower_covered or not upper_covered:
            raise ValueError(
                "Model listing does not bracket the requested ID cohort"
            )
        selected = [
            row for row in candidates
            if model_id_min <= int(row["model_id"]) < model_id_max_exclusive
        ]
        if not selected:
            raise ValueError("Requested ID cohort contains no listed models")
        frame_type = "MODEL_ID_BOUNDED_PUBLIC_REGISTRY_COHORT"
    else:
        selected = candidates
        lower_covered = None
        upper_covered = None
        frame_type = "SERVICE_LISTING_WINDOW"
    if model_limit is not None:
        selected = selected[:model_limit]
    frame = {
        "frame_type": frame_type,
        "model_id_min_inclusive": model_id_min,
        "model_id_max_exclusive": model_id_max_exclusive,
        "listing_record_count": len(listed),
        "listing_unique_model_count": len(candidates),
        "identical_duplicate_model_record_count": identical_duplicate_count,
        "listing_min_model_id": min(ids),
        "listing_max_model_id": max(ids),
        "listing_ids_strictly_descending": strictly_descending,
        "lower_boundary_covered": lower_covered,
        "upper_boundary_covered": upper_covered,
        "selected_model_count": len(selected),
    }
    return selected, frame


def write_json_atomic(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def partition_versions(versions, owner, model_slug, instance):
    instance_id = (
        instance["model_instance_id"]
        if isinstance(instance, dict) else instance.id
    )
    variation_slug = (
        instance["variation_slug"]
        if isinstance(instance, dict) else instance.slug
    )
    matched = []
    foreign = []
    for version in versions:
        identity = {
            "owner_slug": getattr(version, "owner_slug", None),
            "model_slug": getattr(version, "model_slug", None),
            "model_instance_id": getattr(version, "model_instance_id", None),
            "variation_slug": getattr(version, "variation_slug", None),
        }
        if (
            identity["model_instance_id"] is None
            and identity["variation_slug"] is None
        ):
            raise ValueError(
                "Version response lacks model-instance identity fields"
            )
        checks = [
            identity["owner_slug"] in (None, owner),
            identity["model_slug"] in (None, model_slug),
            identity["model_instance_id"] in (None, instance_id),
            identity["variation_slug"] in (None, variation_slug),
        ]
        (matched if all(checks) else foreign).append(version)
    return matched, foreign


def http_status(error):
    current = error
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None:
            return int(status), response
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    text = str(error)
    for candidate in (429, 500, 502, 503, 504):
        if str(candidate) in text:
            return candidate, None
    return None, None


def public_subject_failure(error, stage, subject):
    status, _ = http_status(error)
    codes = {
        403: "NOT_PUBLICLY_ENUMERABLE",
        404: "NOT_FOUND_DURING_ENUMERATION",
        500: "TRANSIENT_ENUMERATION_FAILED",
        502: "TRANSIENT_ENUMERATION_FAILED",
        503: "TRANSIENT_ENUMERATION_FAILED",
        504: "TRANSIENT_ENUMERATION_FAILED",
    }
    code = codes.get(status)
    if code is None:
        return None
    message = str(error).encode("utf-8", errors="replace")
    return {
        "stage": stage,
        "subject": subject,
        "status": code,
        "http_status": status,
        "error_class": type(error).__name__,
        "message_sha256": hashlib.sha256(message).hexdigest(),
    }


class RequestController:
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        request_interval_seconds,
        max_retries,
        retry_base_seconds,
        retry_cap_seconds,
        clock=time.monotonic,
        sleeper=time.sleep,
    ):
        self.request_interval_seconds = float(request_interval_seconds)
        self.max_retries = int(max_retries)
        self.retry_base_seconds = float(retry_base_seconds)
        self.retry_cap_seconds = float(retry_cap_seconds)
        self.clock = clock
        self.sleeper = sleeper
        self.last_request_at = None
        self.request_count = 0
        self.retry_events = []

    def _pace(self):
        if self.last_request_at is not None:
            remaining = (
                self.request_interval_seconds
                - (self.clock() - self.last_request_at)
            )
            if remaining > 0:
                self.sleeper(remaining)
        self.last_request_at = self.clock()

    @staticmethod
    def _retry_after(response):
        if response is None:
            return 0.0
        value = response.headers.get("Retry-After")
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    def call(self, endpoint, operation):
        for failed_attempt in range(self.max_retries + 1):
            self._pace()
            self.request_count += 1
            try:
                return operation()
            except Exception as error:
                status, response = http_status(error)
                if (
                    status not in self.RETRYABLE_STATUS
                    or failed_attempt >= self.max_retries
                ):
                    raise
                exponential = min(
                    self.retry_cap_seconds,
                    self.retry_base_seconds * (2 ** failed_attempt),
                )
                delay = max(exponential, self._retry_after(response))
                self.retry_events.append({
                    "endpoint": endpoint,
                    "failed_attempt": failed_attempt + 1,
                    "http_status": status,
                    "delay_seconds": delay,
                    "observed_at_utc": utc_now(),
                })
                self.sleeper(delay)
        raise AssertionError("Retry loop exited without a result")


class KaggleBackend:
    def __init__(self, request_controller):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            from kagglesdk.models.types.model_api_service import (
                ApiListModelInstancesRequest,
                ApiListModelInstanceVersionFilesRequest,
                ApiListModelInstanceVersionsRequest,
                ApiListModelsRequest,
            )
            from kagglesdk.models.types.model_enums import ListModelsOrderBy
            from kagglesdk.models.types.model_enums import ModelFramework
        except ImportError as error:
            raise RuntimeError(
                "Kaggle CLI 2.2.4 is required; install requirements-kaggle.txt"
            ) from error
        self.ApiListModelsRequest = ApiListModelsRequest
        self.ApiListModelInstancesRequest = ApiListModelInstancesRequest
        self.ApiListModelInstanceVersionsRequest = (
            ApiListModelInstanceVersionsRequest
        )
        self.ApiListModelInstanceVersionFilesRequest = (
            ApiListModelInstanceVersionFilesRequest
        )
        self.ListModelsOrderBy = ListModelsOrderBy
        self.ModelFramework = ModelFramework
        self.requests = request_controller
        self.api = KaggleApi()
        try:
            self.api.authenticate()
            self._client_context = self.api.build_kaggle_client()
            self.client = self._client_context.__enter__()
        except BaseException as error:
            raise RuntimeError(
                "Kaggle authentication failed. Run `kaggle auth login` or "
                "set KAGGLE_API_TOKEN. No snapshot was written."
            ) from error

    def close(self):
        if hasattr(self, "_client_context"):
            self._client_context.__exit__(None, None, None)

    @staticmethod
    def _paging(request, page_size, token):
        request.page_size = int(page_size)
        request.page_token = token or ""

    def list_models(self, token, page_size):
        request = self.ApiListModelsRequest()
        request.search = ""
        request.owner = ""
        request.sort_by = self.ListModelsOrderBy.LIST_MODELS_ORDER_BY_CREATE_TIME
        self._paging(request, page_size, token)
        return self.requests.call(
            "list_models",
            lambda: self.client.models.model_api_client.list_models(request),
        )

    def list_instances(self, owner, model, token, page_size):
        request = self.ApiListModelInstancesRequest()
        request.owner_slug = owner
        request.model_slug = model
        self._paging(request, page_size, token)
        return self.requests.call(
            "list_model_instances",
            lambda: self.client.models.model_api_client.list_model_instances(
                request
            ),
        )

    def list_versions(self, owner, model, instance, token, page_size):
        request = self.ApiListModelInstanceVersionsRequest()
        request.owner_slug = owner
        request.model_slug = model
        request.instance_slug = instance["variation_slug"]
        request.framework = self.ModelFramework[instance["framework"]]
        self._paging(request, page_size, token)
        api_client = self.client.models.model_api_client
        operation = api_client.list_model_instance_versions
        return self.requests.call(
            "list_model_instance_versions",
            lambda: operation(request),
        )

    def list_files(
        self, owner, model, instance, version_number, token, page_size
    ):
        request = self.ApiListModelInstanceVersionFilesRequest()
        request.owner_slug = owner
        request.model_slug = model
        request.instance_slug = instance["variation_slug"]
        request.framework = self.ModelFramework[instance["framework"]]
        request.version_number = int(version_number)
        self._paging(request, page_size, token)
        api_client = self.client.models.model_api_client
        operation = api_client.list_model_instance_version_files
        return self.requests.call(
            "list_model_instance_version_files",
            lambda: operation(request),
        )


class SnapshotWriter:
    def __init__(self, root, records=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.records = list(records or [])
        self.sequence = 0
        referenced = set()
        for record in self.records:
            relative = Path(record["raw_response_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe raw response path in checkpoint")
            path = root / relative
            if not path.is_file():
                raise ValueError(f"Checkpoint raw response is missing: {path}")
            document = json.loads(path.read_text(encoding="ascii"))
            actual = sha256_text(canonical_json(document))
            if actual != record["response_sha256"]:
                raise ValueError(f"Checkpoint raw response mismatch: {path}")
            referenced.add(relative.as_posix())
            prefix = relative.name.split("-", 1)[0]
            if not prefix.isdigit():
                raise ValueError(f"Invalid raw response name: {relative}")
            self.sequence = max(self.sequence, int(prefix))
        for path in root.glob("*.json"):
            if path.name not in referenced:
                path.unlink()

    def add_pages(self, endpoint, subject, pages):
        projected = []
        safe_subject = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:16]
        for page in pages:
            self.sequence += 1
            relative = Path(
                f"{self.sequence:07d}-{endpoint}-{safe_subject}.json"
            )
            body = canonical_json(page["raw_response"]) + "\n"
            path = self.root / relative
            path.write_text(body, encoding="ascii")
            row = {
                "endpoint": endpoint,
                "subject": subject,
                "request_page_token": page["request_page_token"],
                "next_page_token": page["next_page_token"],
                "item_count": page["item_count"],
                "response_sha256": page["response_sha256"],
                "raw_response_path": relative.as_posix(),
            }
            self.records.append(row)
            projected.append(row)
        return projected


def snapshot(
    backend,
    page_size,
    history_scope,
    model_limit,
    raw_writer,
    model_id_min=None,
    model_id_max_exclusive=None,
    checkpoint_path=None,
    checkpoint_context=None,
    resume_state=None,
    checkpoint_interval=25,
    progress_interval=100,
    embedded_instance_crosscheck_count=100,
):
    if resume_state is None:
        models, model_pages = collect_pages(
            backend.list_models,
            lambda response: response.models,
            page_size,
        )
        raw_writer.add_pages("models", "all", model_pages)
        model_candidates, population_frame = select_model_candidates(
            models,
            model_id_min,
            model_id_max_exclusive,
            model_limit,
        )
        model_rows = []
        variation_rows = []
        subject_failures = []
        next_model_index = 0
    else:
        model_candidates = resume_state["model_candidates"]
        model_rows = resume_state["model_rows"]
        variation_rows = resume_state["variation_rows"]
        subject_failures = resume_state["subject_failures"]
        population_frame = resume_state["population_frame"]
        next_model_index = int(resume_state["next_model_index"])

    def persist(next_index):
        if checkpoint_path is None:
            return
        requests = getattr(backend, "requests", None)
        write_json_atomic(checkpoint_path, {
            "schema": CHECKPOINT_SCHEMA,
            "context": checkpoint_context,
            "next_model_index": next_index,
            "model_candidates": model_candidates,
            "population_frame": population_frame,
            "model_rows": model_rows,
            "variation_rows": variation_rows,
            "subject_failures": subject_failures,
            "raw_response_pages": raw_writer.records,
            "request_count": (
                requests.request_count if requests is not None else None
            ),
            "retry_events": (
                requests.retry_events if requests is not None else []
            ),
        })

    if checkpoint_path is not None and resume_state is None:
        persist(0)

    for model_index in range(next_model_index, len(model_candidates)):
        model = model_candidates[model_index]
        model_ref = model["model_ref"]
        if not model_ref or "/" not in model_ref:
            raise ValueError(f"Model ref is not owner/model: {model_ref!r}")
        owner, model_slug = model_ref.split("/", 1)
        model_row = {
            key: value for key, value in model.items()
            if key != "embedded_instances"
        }
        instances = list(model.get("embedded_instances", []))
        model_row["instance_enumeration_status"] = (
            "ASSESSED_FROM_MODEL_LIST_EMBEDDED_INSTANCES"
        )
        model_row["variation_count"] = len(instances)
        if model_index < embedded_instance_crosscheck_count:
            try:
                endpoint_instances, pages = collect_pages(
                    lambda token, size: backend.list_instances(
                        owner, model_slug, token, size
                    ),
                    lambda response: response.instances,
                    page_size,
                )
                raw_writer.add_pages(
                    "instances_crosscheck", model_ref, pages
                )
                endpoint_facts = [
                    instance_fact(instance) for instance in endpoint_instances
                ]
                expected = sorted(instance_identity(row) for row in instances)
                observed = sorted(
                    instance_identity(row) for row in endpoint_facts
                )
                if expected != observed:
                    raise ValueError(
                        "Embedded model instances differ from the dedicated "
                        f"instance endpoint: {model_ref}"
                    )
                model_row["embedded_instance_crosscheck_status"] = "MATCH"
            except Exception as error:
                failure = public_subject_failure(
                    error, "crosscheck_model_instances", model_ref
                )
                if failure is None:
                    raise
                subject_failures.append(failure)
                model_row["embedded_instance_crosscheck_status"] = failure[
                    "status"
                ]
        else:
            model_row["embedded_instance_crosscheck_status"] = "NOT_SELECTED"
        model_rows.append(model_row)
        for instance in instances:
            framework = instance["framework"]
            tflite_declared = is_tflite_framework(framework)
            variation_ref = "/".join([
                owner,
                model_slug,
                framework,
                instance["variation_slug"],
            ])
            if not tflite_declared and history_scope == "all":
                returned_versions = []
                foreign_versions = []
                version_enumeration_status = (
                    "NOT_REQUESTED_NON_TFLITE_FRAMEWORK"
                )
                versions = []
            elif history_scope == "latest":
                returned_versions = []
                foreign_versions = []
                version_enumeration_status = (
                    "NOT_REQUESTED_CURRENT_FROM_EMBEDDED_INSTANCE"
                )
                current = instance["version_number"]
                versions = [] if current is None else [{
                    "version_number": int(current),
                    "version_id": instance["version_id"],
                    "source": "model_list_embedded_current_version",
                }]
            else:
                try:
                    returned_versions, pages = collect_pages(
                        lambda token, size: backend.list_versions(
                            owner, model_slug, instance, token, size
                        ),
                        lambda response: (
                            response.version_list.versions
                            if response.version_list is not None else []
                        ),
                        page_size,
                    )
                    raw_writer.add_pages("versions", variation_ref, pages)
                    matched, foreign_versions = partition_versions(
                        returned_versions,
                        owner,
                        model_slug,
                        instance,
                    )
                    versions = [{
                        "version_number": int(row.version_number),
                        "version_id": row.id,
                        "source": "version_list",
                    } for row in matched]
                    version_enumeration_status = "ASSESSED"
                except Exception as error:
                    failure = public_subject_failure(
                        error, "list_model_instance_versions", variation_ref
                    )
                    if failure is None:
                        raise
                    subject_failures.append(failure)
                    returned_versions = []
                    foreign_versions = []
                    versions = []
                    version_enumeration_status = failure["status"]
            versions = sorted(
                versions, key=lambda row: row["version_number"]
            )
            version_numbers = [row["version_number"] for row in versions]
            if len(version_numbers) != len(set(version_numbers)):
                raise ValueError(f"Duplicate version number: {variation_ref}")
            version_rows = []
            for version in versions:
                number = version["version_number"]
                version_ref = f"{variation_ref}/{number}"
                if not tflite_declared:
                    file_rows = []
                    file_enumeration_status = (
                        "NOT_REQUESTED_NON_TFLITE_FRAMEWORK"
                    )
                else:
                    try:
                        files, pages = collect_pages(
                            lambda token, size: backend.list_files(
                                owner, model_slug, instance, number, token, size
                            ),
                            lambda response: response.files,
                            page_size,
                        )
                        raw_writer.add_pages("files", version_ref, pages)
                        file_rows = [
                            {
                                "logical_path": row.name,
                                "size_bytes": row.size,
                                "creation_date": (
                                    row.creation_date.isoformat()
                                    if row.creation_date is not None else None
                                ),
                                "is_tflite": is_tflite_path(row.name),
                            }
                            for row in files
                        ]
                        file_enumeration_status = "ASSESSED"
                    except Exception as error:
                        failure = public_subject_failure(
                            error,
                            "list_model_instance_version_files",
                            version_ref,
                        )
                        if failure is None:
                            raise
                        subject_failures.append(failure)
                        file_rows = []
                        file_enumeration_status = failure["status"]
                version_rows.append({
                    "version_number": number,
                    "version_id": version["version_id"],
                    "version_source": version["source"],
                    "file_enumeration_status": file_enumeration_status,
                    "is_latest": bool(
                        instance["version_number"] is not None
                        and number == int(instance["version_number"])
                    ),
                    "file_count": (
                        len(file_rows)
                        if file_enumeration_status == "ASSESSED" else None
                    ),
                    "tflite_file_count": (
                        sum(row["is_tflite"] for row in file_rows)
                        if file_enumeration_status == "ASSESSED" else None
                    ),
                    "files": file_rows,
                })
            latest_has_tflite = (
                bool(
                    version_rows
                    and version_rows[-1]["is_latest"]
                    and version_rows[-1]["file_enumeration_status"]
                    == "ASSESSED"
                    and version_rows[-1]["tflite_file_count"]
                )
                if tflite_declared else None
            )
            variation_rows.append({
                "variation_ref": variation_ref,
                "model_ref": model_ref,
                "model_instance_id": instance["model_instance_id"],
                "framework": framework,
                "framework_value": instance["framework_value"],
                "variation_slug": instance["variation_slug"],
                "declared_tflite_framework": tflite_declared,
                "reported_current_version_number": instance["version_number"],
                "current_version_status": (
                    "PRESENT"
                    if instance["version_number"] is not None
                    else "MISSING"
                ),
                "version_enumeration_status": version_enumeration_status,
                "returned_version_record_count": len(returned_versions),
                "foreign_version_record_count": len(foreign_versions),
                "enumerated_version_count": len(versions),
                "recorded_version_count": len(version_rows),
                "latest_version_contains_tflite": latest_has_tflite,
                "versions": version_rows,
            })
        completed = model_index + 1
        if completed % checkpoint_interval == 0:
            persist(completed)
        if completed % progress_interval == 0:
            print(json.dumps({
                "progress": "kaggle_snapshot",
                "completed_models": completed,
                "total_models": len(model_candidates),
                "variation_count": len(variation_rows),
                "subject_failure_count": len(subject_failures),
                "request_count": getattr(
                    getattr(backend, "requests", None),
                    "request_count",
                    None,
                ),
            }), file=sys.stderr, flush=True)
    persist(len(model_candidates))
    return model_rows, variation_rows, subject_failures, population_frame


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/kaggle-tflite-snapshot.json",
    )
    parser.add_argument(
        "--raw-pages",
        type=Path,
        default=root / "data/kaggle-snapshot-pages",
    )
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--request-interval-seconds", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-cap-seconds", type=float, default=120.0)
    parser.add_argument(
        "--history-scope", choices=("latest", "all"), default="all"
    )
    parser.add_argument("--model-limit", type=int)
    parser.add_argument("--model-id-min", type=int)
    parser.add_argument("--model-id-max-exclusive", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument(
        "--embedded-instance-crosscheck-count", type=int, default=100
    )
    args = parser.parse_args()
    if args.page_size < 1 or args.page_size > 200:
        raise ValueError("page-size must be in 1..200")
    if args.model_limit is not None and args.model_limit < 1:
        raise ValueError("model-limit must be positive")
    if (args.model_id_min is None) != (args.model_id_max_exclusive is None):
        raise ValueError(
            "model-id-min and model-id-max-exclusive must be supplied together"
        )
    if (
        args.model_id_min is not None
        and args.model_id_min >= args.model_id_max_exclusive
    ):
        raise ValueError("model-id-min must be below model-id-max-exclusive")
    if args.request_interval_seconds < 0:
        raise ValueError("request-interval-seconds must be nonnegative")
    if args.max_retries < 0:
        raise ValueError("max-retries must be nonnegative")
    if args.retry_base_seconds <= 0 or args.retry_cap_seconds <= 0:
        raise ValueError("retry delays must be positive")
    if args.checkpoint_interval < 1 or args.progress_interval < 1:
        raise ValueError("checkpoint and progress intervals must be positive")
    if args.embedded_instance_crosscheck_count < 0:
        raise ValueError("embedded-instance-crosscheck-count must be nonnegative")
    if args.checkpoint is None:
        args.checkpoint = (
            root / "cache/kaggle-checkpoints"
            / f"{args.output.name}.checkpoint.json"
        )
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    configuration = {
        "kaggle_cli_version": KAGGLE_VERSION,
        "output": str(args.output.resolve()),
        "raw_pages": str(args.raw_pages.resolve()),
        "page_size": args.page_size,
        "history_scope": args.history_scope,
        "model_limit": args.model_limit,
        "model_id_min": args.model_id_min,
        "model_id_max_exclusive": args.model_id_max_exclusive,
        "request_interval_seconds": args.request_interval_seconds,
        "max_retries": args.max_retries,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_cap_seconds": args.retry_cap_seconds,
        "embedded_instance_crosscheck_count": (
            args.embedded_instance_crosscheck_count
        ),
    }
    if args.resume:
        if not args.checkpoint.is_file():
            raise ValueError(f"checkpoint is missing: {args.checkpoint}")
        resume_state = json.loads(
            args.checkpoint.read_text(encoding="ascii")
        )
        if resume_state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Unexpected checkpoint schema")
        context = resume_state.get("context", {})
        if context.get("configuration") != configuration:
            raise ValueError("Checkpoint configuration does not match command")
        started = context["started_at_utc"]
    else:
        resume_state = None
        if args.checkpoint.exists():
            raise ValueError(
                f"checkpoint already exists; use --resume: {args.checkpoint}"
            )
        if args.raw_pages.exists() and any(args.raw_pages.iterdir()):
            raise ValueError(
                f"raw-pages directory is not empty: {args.raw_pages}"
            )
        started = utc_now()
        context = {
            "started_at_utc": started,
            "configuration": configuration,
        }

    backend = None
    request_controller = RequestController(
        args.request_interval_seconds,
        args.max_retries,
        args.retry_base_seconds,
        args.retry_cap_seconds,
    )
    if resume_state is not None:
        request_controller.request_count = int(
            resume_state.get("request_count", 0)
        )
        request_controller.retry_events = list(
            resume_state.get("retry_events", [])
        )
    try:
        backend = KaggleBackend(request_controller)
        writer = SnapshotWriter(
            args.raw_pages,
            (
                resume_state["raw_response_pages"]
                if resume_state is not None else None
            ),
        )
        models, variations, subject_failures, population_frame = snapshot(
            backend,
            args.page_size,
            args.history_scope,
            args.model_limit,
            writer,
            model_id_min=args.model_id_min,
            model_id_max_exclusive=args.model_id_max_exclusive,
            checkpoint_path=args.checkpoint,
            checkpoint_context=context,
            resume_state=resume_state,
            checkpoint_interval=args.checkpoint_interval,
            progress_interval=args.progress_interval,
            embedded_instance_crosscheck_count=(
                args.embedded_instance_crosscheck_count
            ),
        )
    except Exception as error:
        print(f"snapshot failed: {error}", file=sys.stderr)
        return 2
    finally:
        if backend is not None:
            backend.close()

    latest_tflite = [
        row for row in variations if row["latest_version_contains_tflite"]
    ]
    tflite_files = [
        file
        for variation in latest_tflite
        for version in variation["versions"]
        if version["is_latest"]
        for file in version["files"]
        if file["is_tflite"]
    ]
    failure_stage_counts = Counter(
        row["stage"] for row in subject_failures
    )
    failure_status_counts = Counter(
        row["status"] for row in subject_failures
    )
    document = {
        "schema": SCHEMA,
        "snapshot_status": (
            "INCOMPLETE_LIMITED" if args.model_limit is not None else "COMPLETE"
        ),
        "protocol": "docs/research-population-protocol.md",
        "acquisition": {
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "kaggle_cli_version": KAGGLE_VERSION,
            "sort_order": "createTime",
            "page_size": args.page_size,
            "history_scope": args.history_scope,
            "model_limit": args.model_limit,
            "model_id_min": args.model_id_min,
            "model_id_max_exclusive": args.model_id_max_exclusive,
            "request_interval_seconds": args.request_interval_seconds,
            "max_retries": args.max_retries,
            "retry_base_seconds": args.retry_base_seconds,
            "retry_cap_seconds": args.retry_cap_seconds,
            "embedded_instance_source": "model_list_response",
            "embedded_instance_crosscheck_count": (
                args.embedded_instance_crosscheck_count
            ),
            "request_count": request_controller.request_count,
            "retry_count": len(request_controller.retry_events),
            "retry_events": request_controller.retry_events,
        },
        "summary": {
            "listing_record_count": population_frame["listing_record_count"],
            "listing_unique_model_count": population_frame[
                "listing_unique_model_count"
            ],
            "selected_model_count": population_frame["selected_model_count"],
            "enumerated_model_count": len(models),
            "enumerated_variation_count": len(variations),
            "declared_tflite_variation_count": sum(
                bool(row["declared_tflite_framework"])
                for row in variations
            ),
            "embedded_instance_crosscheck_match_count": sum(
                row["embedded_instance_crosscheck_status"] == "MATCH"
                for row in models
            ),
            "latest_tflite_variation_count": len(latest_tflite),
            "latest_published_tflite_file_count": len(tflite_files),
            "foreign_version_record_count": sum(
                row["foreign_version_record_count"] for row in variations
            ),
            "subject_enumeration_failure_count": len(subject_failures),
            "subject_enumeration_failure_stage_counts": dict(
                sorted(failure_stage_counts.items())
            ),
            "subject_enumeration_failure_status_counts": dict(
                sorted(failure_status_counts.items())
            ),
            "raw_response_page_count": len(writer.records),
        },
        "population_frame": population_frame,
        "models": models,
        "variations": variations,
        "subject_enumeration_failures": subject_failures,
        "raw_response_pages": writer.records,
    }
    write_json_atomic(args.output, document)
    if args.checkpoint.exists():
        args.checkpoint.unlink()
    print(json.dumps(document["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
