import base64
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


ROOT_SCHEMA_ID = "https://cyclonedx.org/schema/2.0/cyclonedx-2.0.schema.json"
PREFIX = "cdx:ai-ml:model:parameter:quantization:"


def fail(message):
    raise ValueError(message)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def stable_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_schema_graph(root, manifest):
    source = manifest["schema_source"]
    archive_path = root / source["archive"]
    if sha256_file(archive_path) != source["archive_sha256"]:
        fail("Pinned CycloneDX schema archive SHA-256 mismatch")
    with gzip.open(archive_path, "rt", encoding="utf-8") as stream:
        archive = json.load(stream)
    if archive["commit"] != source["commit"]:
        fail("Pinned schema commit mismatch")
    if len(archive["files"]) != source["file_count"]:
        fail("Pinned schema member count mismatch")

    schemas = []
    for entry in archive["files"]:
        content = base64.b64decode(entry["content_base64"], validate=True)
        if sha256_bytes(content) != entry["sha256"]:
            fail(f"Pinned schema member SHA-256 mismatch: {entry['path']}")
        if "bundled" not in entry["path"]:
            schemas.append(json.loads(content))

    registry = Registry()
    for schema in schemas:
        registry = registry.with_resource(
            schema["$id"], Resource.from_contents(schema)
        )
    root_schema = next(
        schema for schema in schemas if schema["$id"] == ROOT_SCHEMA_ID
    )
    return Draft202012Validator(root_schema, registry=registry)


def first_parameter(document):
    return (
        document.get("components", [{}])[0]
        .get("modelProperties", {})
        .get("inputs", [None])[0]
    )


def property_map(parameter):
    if not parameter:
        return {}
    rows = parameter.get("properties", [])
    result = {}
    for row in rows:
        name = row.get("name", "")
        if name.startswith(PREFIX):
            key = name.removeprefix(PREFIX)
            if key in result:
                fail(f"Duplicate taxonomy property: {key}")
            result[key] = row["value"]
    return result


def verify_probe_semantics(documents):
    typed_only = first_parameter(documents["typed-only"])
    if set(property_map(typed_only)):
        fail("typed-only unexpectedly contains taxonomy properties")

    extension = first_parameter(documents["taxonomy-extension"])
    if set(property_map(extension)) != {"scale", "zeroPoint"}:
        fail("taxonomy-extension is not a numeric-only extension")

    duplicated = first_parameter(documents["duplicated-scheme"])
    duplicated_properties = property_map(duplicated)
    if duplicated["quantization"]["scheme"] != "affine":
        fail("duplicated-scheme typed source is not affine")
    if duplicated_properties["scheme"] != "affine_asymmetric":
        fail("duplicated-scheme taxonomy source is not affine_asymmetric")

    contradiction = first_parameter(documents["contradiction"])
    contradiction_properties = property_map(contradiction)
    if contradiction["quantization"]["scheme"] != "symmetric":
        fail("contradiction typed scheme changed")
    if contradiction_properties["scheme"] != "affine_asymmetric":
        fail("contradiction taxonomy scheme changed")
    if contradiction["quantization"]["granularity"] != "per-tensor":
        fail("contradiction typed granularity changed")
    if contradiction_properties["granularity"] != "per-axis":
        fail("contradiction taxonomy granularity changed")

    vector = property_map(first_parameter(documents["per-tensor-vector-scale"]))
    scales = json.loads(vector["scale"])
    if vector["granularity"] != "per-tensor" or not isinstance(scales, list):
        fail("per-tensor-vector-scale no longer demonstrates a vector value")

    legacy = documents["legacy-modelcard"]["components"][0]
    if "modelCard" not in legacy or "modelProperties" in legacy:
        fail("legacy-modelcard placement changed")


def verify_result(root, manifest, documents, validity):
    result_path = root / "data/cyclonedx-pr990-validation-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["schema"] != (
        "tensor_quantization_metadata_study.cyclonedx_pr990_validation.v1"
    ):
        fail("Unexpected PR #990 result schema")
    if result["validators"]["independent"]["required_version"] != "4.26.0":
        fail("Independent validator version contract changed")
    jsonschema_version = importlib.metadata.version("jsonschema")
    if jsonschema_version != "4.26.0":
        fail(f"python-jsonschema 4.26.0 required, found {jsonschema_version}")

    expected_rows = {row["id"]: row for row in manifest["probes"]}
    observed_rows = {row["id"]: row for row in result["probes"]}
    if set(observed_rows) != set(expected_rows):
        fail("PR #990 result probe set mismatch")
    for probe_id, expected in expected_rows.items():
        row = observed_rows[probe_id]
        document_path = (
            root
            / "conformance/cyclonedx-2.0/quantization-ownership-probes"
            / expected["path"]
        )
        if row["sha256"] != sha256_file(document_path):
            fail(f"Probe result hash mismatch: {probe_id}")
        if row["schema_valid"] != validity[probe_id]:
            fail(f"Independent schema result mismatch: {probe_id}")
        if row["schema_valid"] != expected["expected_schema_valid"]:
            fail(f"Manifest schema expectation mismatch: {probe_id}")
        if row["taxonomy_status"] != expected["expected_taxonomy_status"]:
            fail(f"Manifest taxonomy expectation mismatch: {probe_id}")
        if row["ownership_status"] != expected["expected_ownership_status"]:
            fail(f"Manifest ownership expectation mismatch: {probe_id}")

    ledger_sha256 = result.pop("ledger_sha256")
    if sha256_bytes(stable_json(result)) != ledger_sha256:
        fail("PR #990 result ledger SHA-256 mismatch")


def verify_public_boundary(root):
    paths = [
        root / "conformance/cyclonedx-2.0/quantization-ownership-probes",
        root / "data/cyclonedx-pr990-validation-result.json",
        root / "docs/quantization-vocabulary-mapping.md",
        root / "reference/cyclonedx-property-taxonomy",
        root / "scripts/probe-cyclonedx-pr990.mjs",
        root / "scripts/verify-cyclonedx-pr990-files.py",
    ]
    for base in paths:
        files = [base] if base.is_file() else list(base.rglob("*"))
        for path in files:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            forbidden = ("deep" + "bom", "depi" + "nity")
            if any(value in text for value in forbidden):
                fail(f"Product identity leaked into standard-facing evidence: {path}")


def main():
    root = Path(__file__).resolve().parents[1]
    probe_root = (
        root / "conformance/cyclonedx-2.0/quantization-ownership-probes"
    )
    manifest = json.loads((probe_root / "manifest.json").read_text())
    if sha256_file(root / manifest["taxonomy_source"]["vendored_path"]) != (
        manifest["taxonomy_source"]["sha256"]
    ):
        fail("Pinned property-taxonomy SHA-256 mismatch")

    validator = load_schema_graph(root, manifest)
    documents = {}
    validity = {}
    for row in manifest["probes"]:
        document = json.loads((probe_root / row["path"]).read_text())
        documents[row["id"]] = document
        validity[row["id"]] = not any(validator.iter_errors(document))

    verify_probe_semantics(documents)
    verify_result(root, manifest, documents, validity)
    verify_public_boundary(root)
    print(json.dumps({
        "status": "pass",
        "independent_validator": (
            f"python-jsonschema {importlib.metadata.version('jsonschema')}"
        ),
        "probe_count": len(documents),
    }, indent=2))


if __name__ == "__main__":
    main()
