#!/usr/bin/env python3
"""Measure the byte layout and metadata identities of the pinned Granite pair.

This is a bounded GGUF v3 little-endian layout audit, not a general GGUF
validator or a numerical weight decoder. Upstream gguf reads field values;
an independent byte cursor measures the serialized regions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path("experiments/gguf-imatrix")
FIXED_WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class Cursor:
    def __init__(self, stream, size):
        self.stream, self.size = stream, size

    def take(self, length):
        if length < 0 or length > self.size - self.stream.tell():
            raise ValueError("Serialized field exceeds the file boundary")
        value = self.stream.read(length)
        if len(value) != length:
            raise ValueError("Truncated serialized field")
        return value

    def skip(self, length):
        if length < 0 or length > self.size - self.stream.tell():
            raise ValueError("Serialized field exceeds the file boundary")
        self.stream.seek(length, 1)

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self, decode=False):
        length = self.number("Q")
        if decode:
            if length > 65536:
                raise ValueError("Name exceeds this audit's supported length")
            return self.take(length).decode("utf-8")
        self.skip(length)

    def value(self, kind, depth=0):
        if depth > 8:
            raise ValueError("Nested metadata array exceeds this audit's scope")
        if kind in FIXED_WIDTHS:
            self.skip(FIXED_WIDTHS[kind])
        elif kind == 8:
            self.string()
        elif kind == 9:
            item_type, count = self.number("I"), self.number("Q")
            if item_type in FIXED_WIDTHS:
                self.skip(count * FIXED_WIDTHS[item_type])
            else:
                if count > (self.size - self.stream.tell()) // 8:
                    raise ValueError("Array count exceeds the remaining file")
                for _ in range(count):
                    self.value(item_type, depth + 1)
        else:
            raise ValueError(f"Unsupported GGUF metadata type: {kind}")


def measure(path, artifact, inventory_path):
    import gguf

    path = Path(path)
    before = path.stat()
    if before.st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
        raise ValueError(f"Artifact identity mismatch: {artifact['id']}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["source"]["sha256"] != artifact["sha256"]:
        raise ValueError("Inventory belongs to another artifact")
    reader = gguf.GGUFReader(str(path))
    metadata = []
    with path.open("rb") as stream:
        cursor = Cursor(stream, before.st_size)
        if cursor.take(4) != b"GGUF" or cursor.number("I") != 3:
            raise ValueError("This audit requires little-endian GGUF v3")
        tensor_count, kv_count = cursor.number("Q"), cursor.number("Q")
        if tensor_count != len(reader.tensors) or tensor_count != len(inventory["tensors"]):
            raise ValueError("Tensor counts disagree")
        if kv_count > before.st_size // 12:
            raise ValueError("Metadata count exceeds the file boundary")
        header_end = stream.tell()
        names = set()
        for _ in range(kv_count):
            start = stream.tell()
            key = cursor.string(decode=True)
            if key in names:
                raise ValueError("Duplicate metadata key")
            names.add(key)
            cursor.value(cursor.number("I"))
            field = reader.get_field(key)
            if field is None:
                raise ValueError("Upstream reader did not expose a serialized key")
            value = field.contents()
            encoded = canonical(value)
            row = {
                "key": key, "types": [kind.name for kind in field.types],
                "entryStart": start, "entryBytes": stream.tell() - start,
                "canonicalValueSha256": hashlib.sha256(encoded).hexdigest(),
                "canonicalValueBytes": len(encoded),
            }
            # Tokenizer arrays are identified in full, but are not redistributed.
            if len(encoded) <= 256:
                row["value"] = value
            metadata.append(row)
        kv_end = stream.tell()
        for _ in range(tensor_count):
            cursor.string()
            dimensions = cursor.number("I")
            if dimensions > 8:
                raise ValueError("Tensor rank exceeds this audit's supported scope")
            cursor.skip(dimensions * 8 + 4 + 8)
        descriptor_end = stream.tell()

    alignment_field = reader.get_field("general.alignment")
    alignment = alignment_field.contents() if alignment_field else 32
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment < 1:
        raise ValueError("Invalid alignment")
    data_start = (descriptor_end + alignment - 1) // alignment * alignment
    intervals = []
    expected = {row["name"]: row for row in inventory["tensors"]}
    if len(expected) != tensor_count:
        raise ValueError("Duplicate inventory tensor names")
    seen = set()
    for tensor in reader.tensors:
        if tensor.name in seen or tensor.name not in expected:
            raise ValueError("Unexpected or duplicate tensor name")
        seen.add(tensor.name)
        recorded = expected[tensor.name]
        actual = (list(map(int, tensor.shape)), tensor.tensor_type.name,
                  int(tensor.n_bytes), int(tensor.data_offset))
        if actual != (recorded["shape"], recorded["encoding"],
                      recorded["byteLength"], recorded["dataOffset"]):
            raise ValueError("Original bytes disagree with the committed inventory")
        intervals.append((actual[3], actual[3] + actual[2]))
    intervals.sort()
    if not intervals or intervals[0][0] != data_start:
        raise ValueError("First tensor does not start at the calculated data boundary")
    if intervals[-1][1] > before.st_size:
        raise ValueError("Tensor extends beyond the file")
    gaps = [right[0] - left[1] for left, right in zip(intervals, intervals[1:])]
    if any(gap < 0 for gap in gaps):
        raise ValueError("Overlapping tensor intervals are outside this audit's scope")
    payload = sum(end - start for start, end in intervals)
    regions = {
        "headerBytes": header_end,
        "kvBytes": kv_end - header_end,
        "tensorDescriptorBytes": descriptor_end - kv_end,
        "alignmentPaddingBytes": data_start - descriptor_end,
        "tensorPayloadBytes": payload,
        "interTensorPaddingBytes": sum(gaps),
        "trailingBytes": before.st_size - intervals[-1][1],
    }
    if sum(regions.values()) != before.st_size:
        raise ValueError("Regions do not cover the file exactly")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Artifact changed during measurement")
    return {"artifactId": artifact["id"], "artifactSha256": artifact["sha256"],
            "fileBytes": before.st_size, "ggufVersion": 3,
            "tensorCount": tensor_count, "metadataCount": kv_count,
            "regions": regions, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibm-q4km", type=Path, required=True)
    parser.add_argument("--bartowski-q4km", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / EXPERIMENT / "results/file-layout.json")
    args = parser.parse_args()
    manifest_path = ROOT / EXPERIMENT / "artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = [(args.ibm_q4km, "ibm-granite-4.2-3b-q4-k-m", "ibm-q4km"),
              (args.bartowski_q4km, "bartowski-granite-4.2-3b-q4-k-m", "bartowski-q4km")]
    subjects, sources = [], []
    for path, identifier, prefix in inputs:
        artifact = next(row for row in manifest["artifacts"] if row["id"] == identifier)
        inventory = ROOT / EXPERIMENT / "results" / (prefix + "-tensor-inventory.json")
        subjects.append(measure(path, artifact, inventory))
        sources.append({"path": inventory.relative_to(ROOT).as_posix(), "sha256": sha256(inventory)})
    sources.append({"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256(manifest_path)})
    result = {
        "recordType": "tensor-quantization-metadata-study.gguf-file-layout.v1",
        "method": "Hash-checked GGUF v3 LE files; bounded binary cursor and upstream gguf field values",
        "reader": {"package": "gguf", "version": importlib.metadata.version("gguf")},
        "valueDigestDefinition": "SHA-256 of compact key-sorted UTF-8 JSON; ensure_ascii=False; allow_nan=False",
        "sources": sources, "subjects": subjects,
        "boundary": "Metadata and storage layout only; no decoded quantized weights, production-command provenance, or runtime observations",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "measured", "subjects": len(subjects), "metadataCounts": [s["metadataCount"] for s in subjects]}))


if __name__ == "__main__":
    main()
