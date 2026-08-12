import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


FLATC_VERSION = "25.9.23"
FLATC_ARCHIVE_URL = (
    "https://github.com/google/flatbuffers/releases/download/"
    "v25.9.23/Windows.flatc.binary.zip"
)
FLATC_ARCHIVE_SHA256 = (
    "3d6383193ecd274f5de544a6e03464a87f581befb9fc1dda9cf508fa3cce3127"
)
METADATA_SCHEMA_URL = (
    "https://raw.githubusercontent.com/tensorflow/tflite-support/"
    "78d10177b3bc51f81ea78d8209c557233d15df15/"
    "tensorflow_lite_support/metadata/metadata_schema.fbs"
)
METADATA_SCHEMA_SHA256 = (
    "2d3386ba124690ba1195bfc1d51ac814843bd675a7c845afab7c001c7891449e"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def require_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
        )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "cache/flatc-crosscheck",
    )
    args = parser.parse_args()
    archive = args.cache / "downloads/Windows.flatc.binary.zip"
    schema = args.cache / "metadata_schema.fbs"
    binary = args.cache / "bin/flatc.exe"

    if not archive.exists():
        download(FLATC_ARCHIVE_URL, archive)
    require_hash(archive, FLATC_ARCHIVE_SHA256)
    if not schema.exists():
        download(METADATA_SCHEMA_URL, schema)
    require_hash(schema, METADATA_SCHEMA_SHA256)

    binary.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.endswith("flatc.exe")]
        if len(members) != 1:
            raise ValueError(f"Expected one flatc.exe, found {members}")
        binary.write_bytes(bundle.read(members[0]))

    manifest = {
        "flatc_version": FLATC_VERSION,
        "flatc_archive_url": FLATC_ARCHIVE_URL,
        "flatc_archive_sha256": FLATC_ARCHIVE_SHA256,
        "metadata_schema_url": METADATA_SCHEMA_URL,
        "metadata_schema_sha256": METADATA_SCHEMA_SHA256,
    }
    (args.cache / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps({
        "flatc": str(binary),
        "metadata_schema": str(schema),
        "status": "ready",
    }, indent=2))


if __name__ == "__main__":
    main()
