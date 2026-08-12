import hashlib
import json
import urllib.request
from pathlib import Path


COMMIT = "b528ee73c34e68e781cb0e342e294d7a2b69a935"
SOURCES = {
    "artifact_ledger": {
        "repository_path": "data/artifacts.json",
        "sha256": (
            "be9b2b7aca95745478ad9b568f9e8a0ed04fd73919e89e3275b22f4242df9c34"
        ),
    },
    "interface_ledger": {
        "repository_path": "data/interface-contracts.json",
        "sha256": (
            "ea5f85f39e0b3797406998d2ffeb58dc243180cfe454e90235c97bf124c6393a"
        ),
    },
}


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def fetch_source(repository_path):
    url = (
        "https://raw.githubusercontent.com/JunHwan-Kwon/"
        f"tensor-quantization-metadata-study/{COMMIT}/{repository_path}"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tensor-quantization-metadata-study"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), url


def materialize(output, name, source):
    expected = source["sha256"]
    destination = output / f"{expected}.json"
    if destination.is_file():
        payload = destination.read_bytes()
        if sha256_bytes(payload) == expected:
            return destination, "already_present", None

    payload, url = fetch_source(source["repository_path"])
    actual = sha256_bytes(payload)
    if actual != expected:
        raise ValueError(
            f"Pinned {name} hash mismatch: {actual} != {expected}"
        )
    destination.write_bytes(payload)
    return destination, "downloaded", url


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/recorded-run-sources"
    output.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "pass",
        "source_repository_commit": COMMIT,
        "sources": {},
    }
    for name, source in SOURCES.items():
        path, status, url = materialize(output, name, source)
        result["sources"][name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": source["sha256"],
            "status": status,
            "download_url": url,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
