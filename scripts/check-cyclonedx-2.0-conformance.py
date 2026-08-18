import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from jsonschema import Draft202012Validator


SPECIFICATION_COMMIT = "49a945618811213e55686a23fa63b287940071c6"
SCHEMA_PATH = "schema/2.0/model/cyclonedx-ai-ml-2.0.schema.json"
SCHEMA_SHA256 = (
    "9b346b22f6115c08a6d63cbb04b0774b93aa769dec08015e303b3956363dfa53"
)
SCHEMA_URL = (
    "https://raw.githubusercontent.com/CycloneDX/specification/"
    f"{SPECIFICATION_COMMIT}/{SCHEMA_PATH}"
)
BUNDLED_SCHEMA_PATH = "schema/2.0/cyclonedx-2.0-bundled.schema.json"
BUNDLED_SCHEMA_SHA256 = (
    "b7e43bc3820052904a9d39617b5db211de261125b66e89446876125ddb90d253"
)
BUNDLED_SCHEMA_URL = (
    "https://raw.githubusercontent.com/CycloneDX/specification/"
    f"{SPECIFICATION_COMMIT}/{BUNDLED_SCHEMA_PATH}"
)


def fail(message):
    raise ValueError(message)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def ensure_pinned_json(path, allow_download, url, expected_sha256, label):
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"Pinned {label} is missing: {path}. Use --download."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "tensor-quantization-metadata-study/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if sha256_bytes(payload) != expected_sha256:
            fail(f"Downloaded {label} SHA-256 mismatch")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected_sha256:
        fail(f"{label} SHA-256 mismatch")
    return json.loads(payload.decode("utf-8"))


def json_pointer(path):
    def escape(value):
        return str(value).replace("~", "~0").replace("/", "~1")

    return "/" + "/".join(escape(value) for value in path)


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
    args = parser.parse_args()

    fixture_root = root / "conformance/cyclonedx-2.0"
    manifest = json.loads(
        (fixture_root / "manifest.json").read_text(encoding="utf-8")
    )
    source = manifest["cyclonedx_source"]
    if source != {
        "repository": "CycloneDX/specification",
        "commit": SPECIFICATION_COMMIT,
        "path": SCHEMA_PATH,
        "sha256": SCHEMA_SHA256,
    }:
        fail("Conformance manifest source pin mismatch")

    schema = ensure_pinned_json(
        args.schema.resolve(),
        args.download,
        SCHEMA_URL,
        SCHEMA_SHA256,
        "CycloneDX AI/ML schema",
    )
    validator = Draft202012Validator({
        "$defs": schema["$defs"],
        "$ref": "#/$defs/modelProperties",
    })
    results = []
    for row in manifest["fixtures"]:
        path = fixture_root / row["path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        model_properties = [
            component["modelProperties"]
            for component in document.get("components", [])
            if "modelProperties" in component
        ]
        if not model_properties:
            fail(f"Fixture has no modelProperties object: {row['path']}")
        errors = []
        for instance in model_properties:
            errors.extend(validator.iter_errors(instance))
        errors.sort(
            key=lambda error: (list(error.absolute_path), error.message),
        )
        valid = not errors
        if valid != row["expected_valid"]:
            first = errors[0].message if errors else "no validation error"
            fail(f"Unexpected result for {row['path']}: {first}")
        if not valid:
            matching = [
                error
                for error in errors
                if json_pointer(error.absolute_path)
                == row["expected_instance_pointer"]
                and error.validator == row["expected_validator"]
            ]
            if not matching:
                observed = [
                    {
                        "pointer": json_pointer(error.absolute_path),
                        "validator": error.validator,
                        "message": error.message,
                    }
                    for error in errors
                ]
                fail(
                    f"Expected rejection not observed for {row['path']}: "
                    f"{json.dumps(observed, ensure_ascii=True)}"
                )
        results.append({"path": row["path"], "valid": valid})

    bundled_path = (
        root
        / "cache/cyclonedx-specification"
        / SPECIFICATION_COMMIT
        / "cyclonedx-2.0-bundled.schema.json"
    )
    bundled_schema = ensure_pinned_json(
        bundled_path,
        args.download,
        BUNDLED_SCHEMA_URL,
        BUNDLED_SCHEMA_SHA256,
        "CycloneDX bundled schema",
    )
    valid_document = json.loads(
        (fixture_root / "valid-ai-ml-quantization-2.0.json")
        .read_text(encoding="utf-8")
    )
    bundled_errors = list(
        Draft202012Validator(bundled_schema).iter_errors(valid_document)
    )
    if not bundled_errors:
        fail("Bundled integration probe unexpectedly passed")

    print(json.dumps({
        "status": "pass",
        "schema_sha256": SCHEMA_SHA256,
        "fixture_count": len(results),
        "valid_fixture_count": sum(row["valid"] for row in results),
        "invalid_fixture_count": sum(not row["valid"] for row in results),
        "bundled_integration_valid": False,
        "bundled_integration_error_count": len(bundled_errors),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
