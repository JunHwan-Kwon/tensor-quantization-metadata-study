import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


SCHEMA = "tensor_quantization_metadata_study.kaggle_tflite_materialization.v1"
SNAPSHOT_SCHEMA = "tensor_quantization_metadata_study.kaggle_tflite_snapshot.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_error(error):
    message = str(error)
    return {
        "class": type(error).__name__,
        "message_sha256": hashlib.sha256(
            message.encode("utf-8", errors="replace")
        ).hexdigest(),
    }


def normalized_member_name(value):
    name = value.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive member path: {value!r}")
    return path.as_posix()


def selected_versions(snapshot, scope):
    if snapshot["schema"] != SNAPSHOT_SCHEMA:
        raise ValueError("Unexpected Kaggle snapshot schema")
    if snapshot["snapshot_status"] != "COMPLETE":
        raise ValueError("Materialization requires a complete snapshot")
    rows = []
    for variation in snapshot["variations"]:
        versions = variation["versions"]
        if scope == "latest":
            versions = [row for row in versions if row["is_latest"]]
        for version in versions:
            files = [row for row in version["files"] if row["is_tflite"]]
            if files:
                rows.append((variation, version, files))
    return rows


class KaggleDownloader:
    def __init__(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as error:
            raise RuntimeError(
                "Kaggle CLI 2.2.4 is required; install requirements-kaggle.txt"
            ) from error
        self.api = KaggleApi()
        try:
            self.api.authenticate()
        except BaseException as error:
            raise RuntimeError(
                "Kaggle authentication failed. Run `kaggle auth login` or "
                "set KAGGLE_API_TOKEN."
            ) from error

    def download(self, variation, version, directory):
        directory.mkdir(parents=True, exist_ok=True)
        reference = (
            f"{variation['variation_ref']}/{version['version_number']}"
        )
        return Path(self.api.model_instance_version_download(
            reference,
            path=str(directory),
            force=False,
            quiet=True,
            untar=False,
        ))


def extract_expected(archive, expected_files, model_cache):
    expected = {
        normalized_member_name(row["logical_path"]): row
        for row in expected_files
    }
    results = []
    with tarfile.open(archive, "r:gz") as bundle:
        members = {}
        for member in bundle.getmembers():
            name = normalized_member_name(member.name)
            if name in members:
                raise ValueError(f"Duplicate archive member: {name}")
            members[name] = member
        for name, listed in sorted(expected.items()):
            member = members.get(name)
            if member is None or not member.isfile():
                results.append({
                    "logical_path": listed["logical_path"],
                    "listed_size_bytes": listed["size_bytes"],
                    "status": "DOWNLOAD_MEMBER_MISSING",
                    "artifact_size_bytes": None,
                    "artifact_sha256": None,
                    "cache_path": None,
                })
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archive member: {name}")
            temporary = model_cache / f"pending-{hashlib.sha256(name.encode()).hexdigest()}.tflite"
            model_cache.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as target:
                shutil.copyfileobj(stream, target)
            size = temporary.stat().st_size
            digest = sha256_file(temporary)
            destination = model_cache / f"{digest}.tflite"
            if destination.exists():
                if sha256_file(destination) != digest:
                    raise ValueError(f"Cache collision: {destination}")
                temporary.unlink()
            else:
                temporary.replace(destination)
            status = (
                "ASSESSED_DOWNLOAD"
                if size == listed["size_bytes"]
                else "SIZE_MISMATCH"
            )
            results.append({
                "logical_path": listed["logical_path"],
                "listed_size_bytes": listed["size_bytes"],
                "status": status,
                "artifact_size_bytes": size,
                "artifact_sha256": digest,
                "cache_path": destination.relative_to(
                    model_cache.parents[2]
                ).as_posix(),
            })
    return results


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=root / "data/kaggle-tflite-snapshot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/kaggle-tflite-materialization.json",
    )
    parser.add_argument(
        "--cache", type=Path, default=root / "cache/kaggle-snapshot"
    )
    parser.add_argument(
        "--frame-stability",
        type=Path,
        default=root / "data/kaggle-tflite-frame-stability.json",
    )
    parser.add_argument("--scope", choices=("latest", "all"), default="latest")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    stability = json.loads(args.frame_stability.read_text(encoding="utf-8"))
    if (
        stability.get("status") != "STABLE"
        or stability.get("after_snapshot_sha256") != sha256_file(args.snapshot)
    ):
        raise ValueError(
            "Materialization requires a stable frame bound to this snapshot"
        )
    targets = selected_versions(snapshot, args.scope)
    try:
        downloader = KaggleDownloader()
    except RuntimeError as error:
        print(f"materialization failed: {error}", file=sys.stderr)
        return 2

    results = []
    for variation, version, files in targets:
        version_ref = (
            f"{variation['variation_ref']}/{version['version_number']}"
        )
        archive_directory = (
            args.cache / "archives" / hashlib.sha256(
                version_ref.encode("utf-8")
            ).hexdigest()
        )
        try:
            archive = downloader.download(
                variation, version, archive_directory
            )
            archive_sha256 = sha256_file(archive)
            file_results = extract_expected(
                archive, files, args.cache / "models"
            )
            error = None
        except Exception as exception:
            archive = None
            archive_sha256 = None
            error = public_error(exception)
            file_results = [{
                "logical_path": row["logical_path"],
                "listed_size_bytes": row["size_bytes"],
                "status": "DOWNLOAD_FAILED",
                "artifact_size_bytes": None,
                "artifact_sha256": None,
                "cache_path": None,
            } for row in files]
        results.append({
            "variation_ref": variation["variation_ref"],
            "version_number": version["version_number"],
            "version_ref": version_ref,
            "archive_sha256": archive_sha256,
            "error": error,
            "files": file_results,
        })

    flat_files = [row for result in results for row in result["files"]]
    document = {
        "schema": SCHEMA,
        "source_snapshot_sha256": sha256_file(args.snapshot),
        "source_frame_stability_sha256": sha256_file(args.frame_stability),
        "scope": args.scope,
        "version_count": len(results),
        "published_tflite_file_count": len(flat_files),
        "status_counts": {
            status: sum(row["status"] == status for row in flat_files)
            for status in sorted({row["status"] for row in flat_files})
        },
        "versions": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(document["status_counts"], indent=2))
    return 0 if all(
        row["status"] == "ASSESSED_DOWNLOAD" for row in flat_files
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
