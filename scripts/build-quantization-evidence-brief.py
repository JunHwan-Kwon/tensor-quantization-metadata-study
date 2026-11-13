#!/usr/bin/env python3
"""Build/check a neutral evidence brief from the existing canonical ledgers.

The check is offline and uses only the Python standard library. It checks
record consistency and source bindings, not the original model bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = "experiments/gguf-imatrix"
QWEN = "experiments/gguf-paired-replication/qwen3-4b"
EFF = "bc2ffe19c1118de0c0c2a9088992da5589722656e0fba81421385300a4a34b16"
YAM = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_inventory(document):
    rows = document["tensors"]
    names = [row["name"] for row in rows]
    require(len(names) == len(set(names)), "Duplicate tensor name")
    require(sorted(row["index"] for row in rows) == list(range(len(rows))), "Tensor index coverage")
    sizes = {"Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "F32": (1, 4)}
    for row in rows:
        require(all(type(d) is int and d > 0 for d in row["shape"]), "Invalid stored shape")
        elements = math.prod(row["shape"])
        require(elements == row["elementCount"], "Shape/element count mismatch")
        block, length = sizes[row["encoding"]]
        require(elements % block == 0 and elements // block * length == row["byteLength"], "Encoding/payload mismatch")
    summary = document["summary"]
    require(len(rows) == summary["tensorCount"], "Tensor count mismatch")
    require(sum(r["elementCount"] for r in rows) == summary["elementCount"], "Element sum mismatch")
    require(sum(r["byteLength"] for r in rows) == summary["serializedTensorBytes"], "Payload sum mismatch")
    counts = Counter(r["encoding"] for r in rows)
    require(counts == {r["encoding"]: r["tensorCount"] for r in summary["encodingInventory"]}, "Encoding count mismatch")
    return {row["name"]: row for row in rows}


def compare_pair(left, right, recorded):
    a, b = validate_inventory(left), validate_inventory(right)
    require(a.keys() == b.keys(), "Tensor-name sets differ")
    transitions, changes = Counter(), []
    for name in sorted(a):
        x, y = a[name], b[name]
        require(x["shape"] == y["shape"], "Paired shape mismatch")
        transitions[x["encoding"] + "->" + y["encoding"]] += 1
        if x["encoding"] != y["encoding"]:
            changes.append({"name": name, "shape": x["shape"],
                            "leftEncoding": x["encoding"], "rightEncoding": y["encoding"],
                            "leftByteLength": x["byteLength"], "rightByteLength": y["byteLength"]})
    require(recorded["transitionCounts"] == dict(transitions), "Transition table differs from the inventories")
    require(recorded["changedTensors"] == changes, "Changed tensor rows differ from the inventories")
    require(recorded["changedEncodingCount"] == len(changes), "Changed count mismatch")
    require([s["sha256"] for s in recorded["subjects"]] == [left["source"]["sha256"], right["source"]["sha256"]], "Pair source binding mismatch")
    return changes


def validate_layout(layout, inventories, root):
    for binding in layout["sources"]:
        require(sha256(root / binding["path"]) == binding["sha256"], "Layout source hash mismatch")
    require(len(layout["subjects"]) == len(inventories), "Layout subject count mismatch")
    metadata = []
    for subject, inventory in zip(layout["subjects"], inventories):
        require(subject["artifactSha256"] == inventory["source"]["sha256"], "Layout artifact binding mismatch")
        require(subject["fileBytes"] == inventory["source"]["bytes"], "Layout file size mismatch")
        regions, entries = subject["regions"], subject["metadata"]
        require(all(type(n) is int and n >= 0 for n in regions.values()), "Negative/noninteger region")
        require(sum(regions.values()) == subject["fileBytes"], "File regions do not sum")
        require(regions["tensorPayloadBytes"] == inventory["summary"]["serializedTensorBytes"], "Layout/inventory payload mismatch")
        require(subject["tensorCount"] == inventory["summary"]["tensorCount"], "Layout/inventory tensor count mismatch")
        require(len(entries) == subject["metadataCount"], "Metadata count mismatch")
        require(len({r["key"] for r in entries}) == len(entries), "Duplicate metadata key")
        position = regions["headerBytes"]
        for row in entries:
            require(row["entryStart"] == position and row["entryBytes"] > 0, "Noncontiguous metadata entries")
            position += row["entryBytes"]
            if "value" in row:
                encoded = canonical(row["value"])
                require(len(encoded) == row["canonicalValueBytes"] and hashlib.sha256(encoded).hexdigest() == row["canonicalValueSha256"], "Metadata inline value digest mismatch")
        require(position == regions["headerBytes"] + regions["kvBytes"], "Metadata byte sum mismatch")
        metadata.append({r["key"]: r for r in entries})
    a, b = metadata
    common = a.keys() & b.keys()
    equal = [key for key in common if (a[key]["types"], a[key]["canonicalValueSha256"]) == (b[key]["types"], b[key]["canonicalValueSha256"])]
    return {"leftCount": len(a), "rightCount": len(b), "commonEqualCount": len(equal),
            "commonDifferentKeys": sorted(common - set(equal)),
            "leftOnlyKeys": sorted(a.keys() - b.keys()), "rightOnlyKeys": sorted(b.keys() - a.keys())}


def build(root):
    sources = []
    def load(relative):
        path = root / relative
        sources.append({"path": relative, "sha256": sha256(path)})
        return json.loads(path.read_text(encoding="utf-8"))

    left = load(GGUF + "/results/ibm-q4km-tensor-inventory.json")
    right = load(GGUF + "/results/bartowski-q4km-tensor-inventory.json")
    comparison = load(GGUF + "/results/tensor-assignment-comparison.json")
    changes = compare_pair(left, right, comparison)
    layout = load(GGUF + "/results/file-layout.json")
    metadata_comparison = validate_layout(layout, [left, right], root)
    qleft = load(QWEN + "/results/left-tensor-inventory.json")
    qright = load(QWEN + "/results/right-tensor-inventory.json")
    qcomparison = load(QWEN + "/results/tensor-assignment-comparison.json")
    qchanges = compare_pair(qleft, qright, qcomparison)
    matrix = load(GGUF + "/results/imatrix-measurement.json")
    matrix_tensors = validate_inventory(matrix)
    scan = matrix["numericalScan"]
    scan_rows = scan["tensors"]
    require(len({r["name"] for r in scan_rows}) == len(scan_rows), "Duplicate matrix scan row")
    require({r["name"] for r in scan_rows} == set(matrix_tensors), "Incomplete matrix scan coverage")
    assessed = [r for r in scan_rows if r["status"] == "assessed"]
    require(len(assessed) == scan["assessedTensorCount"] and len(scan_rows) - len(assessed) == scan["unassessedTensorCount"], "Matrix assessment count mismatch")
    for row in assessed:
        require(row["valueCount"] == matrix_tensors[row["name"]]["elementCount"], "Matrix value count mismatch")
        require(0 <= row["nonfiniteValueCount"] <= row["valueCount"] and 0 <= row["exactZeroValueCount"] <= row["valueCount"], "Invalid matrix numerical count")
        require(row["allZero"] == (row["exactZeroValueCount"] == row["valueCount"]), "Matrix all-zero flag mismatch")
        if row["nonfiniteValueCount"] == 0:
            require(math.isfinite(row["finiteMinimum"]) and math.isfinite(row["finiteMaximum"]) and row["finiteMinimum"] <= row["finiteMaximum"], "Invalid matrix extrema")
            require(row["constant"] == (row["finiteMinimum"] == row["finiteMaximum"]), "Matrix constant flag mismatch")
    for source, target in (("valueCount", "decodedValueCount"), ("nonfiniteValueCount", "nonfiniteValueCount"), ("exactZeroValueCount", "exactZeroValueCount"), ("allZero", "allZeroTensorCount"), ("constant", "constantTensorCount")):
        require(sum(r[source] for r in assessed) == scan[target], "Matrix aggregate mismatch: " + target)
    interfaces = load("data/interface-contracts.json")
    internal = load("data/tflite-granularity-evidence.json")
    require(internal["source_interface_contract_ledger_sha256"] == sha256(root / "data/interface-contracts.json"), "I/O/internal ledger binding mismatch")

    io_rows = []
    for artifact_hash in (EFF, YAM):
        selected = [r for r in interfaces["parameters"] if r["artifact_sha256"] == artifact_hash]
        require(len(selected) == 2 and {r["direction"] for r in selected} == {"input", "output"}, "Expected one named input and one output")
        for row in selected:
            q = row["quantization"]
            io_rows.append({k: row[k] for k in ("qualified_id", "artifact_sha256", "direction", "tensor_index", "tensor_name", "dtype", "shape")} | {
                "quantizationStatus": q["status"], "scales": q.get("scales", []), "zeroPoints": q.get("zero_points", [])})
    roles = []
    for artifact_hash in (EFF, YAM):
        matching = [r for r in internal["artifact_rows"] if r["artifact_sha256"] == artifact_hash]
        require(len(matching) == 1, "Missing/duplicate internal ledger subject")
        row = matching[0]
        weights = row["known_weight_contracts"]
        require(len({(w["subgraph"], w["tensor"]) for w in weights}) == len(weights), "Repeated weight identity")
        require(all(w["dtype"] == "INT8" and w["weight_consumers"] for w in weights), "Unexpected weight contract")
        require(all(c["input_slot"] == 1 and c["op"] in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED") for w in weights for c in w["weight_consumers"]), "Weight role not bound to a supported input slot")
        role_counts = row["constant_role_counts"]
        per_axis = sum(w["semantic_granularity"] == "per_axis" for w in weights)
        require(per_axis == role_counts.get("known_weight:INT8:per_axis", 0), "Per-axis weight count mismatch")
        roles.append({"artifactSha256": artifact_hash, "qualifiedId": row["qualified_id"],
                      "int8WeightTensors": len(weights),
                      "int8IntermediateTensors": row["counts"].get("internal_nonconstant:per_tensor:INT8", 0),
                      "perAxisWeightTensors": per_axis,
                      "perAxisBiasTensors": role_counts.get("known_bias:INT32:per_axis", 0)})
    region_delta = {key: layout["subjects"][1]["regions"][key] - layout["subjects"][0]["regions"][key] for key in layout["subjects"][0]["regions"]}
    payload_delta = right["summary"]["serializedTensorBytes"] - left["summary"]["serializedTensorBytes"]
    require(payload_delta == sum(r["rightByteLength"] - r["leftByteLength"] for r in changes), "Changed rows do not explain the payload difference")
    grouped = Counter(".".join(row["name"].split(".")[2:]) for row in changes)
    qindices = sorted(int(r["name"].split(".")[1]) for r in changes if r["name"].endswith(".attn_q.weight"))
    example_names = ("blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_norm.weight")
    indexed_tensors = {row["name"]: (index, row) for index, row in enumerate(left["tensors"])}
    within_artifact = {
        "subject": left["source"],
        "inventoryPath": GGUF + "/results/ibm-q4km-tensor-inventory.json",
        "tensorCount": len(left["tensors"]),
        "encodingCounts": dict(sorted(Counter(row["encoding"] for row in left["tensors"]).items())),
        "examples": [],
    }
    for name in example_names:
        require(name in indexed_tensors, "Missing within-artifact example")
        index, row = indexed_tensors[name]
        within_artifact["examples"].append({
            "tensorName": name, "storedShape": row["shape"], "encoding": row["encoding"],
            "inventoryPointer": f"/tensors/{index}",
        })
    result = {"recordType": "tensor-quantization-metadata-study.evidence-brief.v1", "sources": sources,
              "granite": {"withinArtifact": within_artifact,
                          "subjects": comparison["subjects"], "tensorCountPerSubject": len(left["tensors"]),
                          "changedEncodingCount": len(changes), "changedTensorKinds": dict(sorted(grouped.items())),
                          "changedQueryBlockIndices": qindices,
                          "payloadDifferenceBytes": payload_delta,
                          "payloadIncreasePercent": 100 * payload_delta / left["summary"]["serializedTensorBytes"],
                          "regionDifferenceBytes": region_delta, "metadataComparison": metadata_comparison},
              "qwen": {"subjects": qcomparison["subjects"], "tensorCountPerSubject": len(qleft["tensors"]), "changedEncodingCount": len(qchanges)},
              "namedInterfaces": io_rows, "internalRoles": roles,
              "readiness": {"artifactExamples": "available", "tensorAssignments": "available",
                            "serializedWeightConsumerBindings": "available_for_selected_tflite_operators",
                            "originalModuleMapping": "not_verified", "standardizedInternalTaggingRoundTrip": "not_performed",
                            "hardwareExecutionObservations": "not_performed"}}
    return result, left, right, matrix, layout


def render(result, left, right, matrix, layout):
    within = result["granite"]["withinArtifact"]
    subject = within["subject"]
    lines = ["# Quantization on identified tensors: artifact evidence", "",
             "A single Granite GGUF artifact contains named tensors with Q4_K, Q6_K and F32",
             "storage encodings. This note records those assignments, identifies the file and",
             "tensors, and links the complete inventory and reproduction method.", "",
             "Additional observations cover a second Granite publication, a Qwen pair, and",
             "named TFLite inputs and outputs. These are selected artifact observations;",
             "they do not estimate ecosystem prevalence.", "",
             "## 1. Different encodings within one artifact", "",
             f"Subject: IBM Granite 4.2 3B, `{subject['filename']}`.", "",
             f"Pinned repository revision: `{subject['revision']}`.", "",
             f"Whole-file SHA-256: `{subject['sha256']}`.", "",
             "The following three tensors occur in that same file:", "",
             "| Exact serialized tensor name | Stored shape | Encoding |",
             "| --- | --- | --- |"]
    for row in within["examples"]:
        lines.append(f"| `{row['tensorName']}` | `{row['storedShape']}` | {row['encoding']} |")
    counts = within["encodingCounts"]
    lines += ["", f"The complete inventory contains {within['tensorCount']} tensors: {counts['Q4_K']} Q4_K,",
              f"{counts['Q6_K']} Q6_K and {counts['F32']} F32. F32 denotes unquantized floating-point storage.",
              "The three rows above illustrate the three encodings; the full inventory is linked below.", "",
              "Each observation identifies its subject by **whole-file SHA-256 + exact tensor name**.",
              "Shapes use GGUF stored dimension order. The `blk.0` prefix is a serialized naming",
              "observation; an original layer or module hierarchy has not been reconstructed.", "",
              "Sources: [all IBM tensor records](../experiments/gguf-imatrix/results/ibm-q4km-tensor-inventory.json),",
              "[download identity](../experiments/gguf-imatrix/artifacts.json),",
              "[measurement protocol](../experiments/gguf-imatrix/protocol.md).",
              "The [derived JSON](../experiments/gguf-imatrix/results/evidence-brief.json) also records",
              "the exact inventory row for each of the three examples.", "",
              "## 2. Across two Granite publications", "",
              "Both publications carry the `Q4_K_M` release label. The files have the same",
              "tensor-name set and name-matched stored shapes; their encoding assignments differ.", "",
              "| Encoding | IBM publication | Bartowski publication |", "| --- | ---: | ---: |"]
    ac, bc = Counter(r["encoding"] for r in left["tensors"]), Counter(r["encoding"] for r in right["tensors"])
    for encoding in ("Q4_K", "Q5_K", "Q6_K", "F32"):
        lines.append(f"| {encoding} | {ac[encoding]} | {bc[encoding]} |")
    lines += [f"| Total | {sum(ac.values())} | {sum(bc.values())} |", "",
              "| Changed tensor-name suffix | Count | IBM → Bartowski |", "| --- | ---: | --- |"]
    for kind, transition in (("attn_k.weight", "Q4_K → Q5_K"), ("attn_output.weight", "Q4_K → Q5_K"), ("attn_q.weight", "Q4_K → Q6_K")):
        lines.append(f"| `{kind}` | {result['granite']['changedTensorKinds'][kind]} | {transition} |")
    lines += ["", "The changed query-weight block indices are `" + ", ".join(map(str, result["granite"]["changedQueryBlockIndices"])) + "`.",
              "Unchanged encodings do not establish unchanged numerical weights.", "",
              "| Artifact | Pinned revision | Whole-file SHA-256 |", "| --- | --- | --- |"]
    for label, subject in zip(("IBM", "Bartowski"), result["granite"]["subjects"]):
        lines.append(f"| {label} | `{subject['revision']}` | `{subject['sha256']}` |")
    lines += ["", "Acquisition URLs and filenames are in the [artifact manifest](../experiments/gguf-imatrix/artifacts.json).",
              "The [complete paired table](../experiments/gguf-imatrix/tensor-comparison.md) includes every tensor;",
              "the original JSON inventories remain the machine-readable evidence.", "",
              "### Storage and metadata", "",
              "| File region | IBM bytes | Bartowski bytes | Difference |", "| --- | ---: | ---: | ---: |"]
    labels = {"headerBytes": "Fixed header", "kvBytes": "KV entries", "tensorDescriptorBytes": "Tensor descriptors", "alignmentPaddingBytes": "Padding before tensor data", "tensorPayloadBytes": "Tensor payload", "interTensorPaddingBytes": "Padding between tensors", "trailingBytes": "Trailing bytes"}
    for key, label in labels.items():
        a, b = [s["regions"][key] for s in layout["subjects"]]
        lines.append(f"| {label} | {a:,} | {b:,} | {b-a:+,} |")
    a, b = [s["fileBytes"] for s in layout["subjects"]]
    g = result["granite"]
    lines += [f"| Whole file | {a:,} | {b:,} | {b-a:+,} |", "",
              f"The payload difference is {g['payloadDifferenceBytes']:,} bytes ({g['payloadIncreasePercent']:.4f}% of IBM's payload).",
              f"The remaining {b-a-g['payloadDifferenceBytes']:,} bytes comprise {g['regionDifferenceBytes']['kvBytes']:,} bytes of KV entries and {g['regionDifferenceBytes']['alignmentPaddingBytes']:,} bytes of alignment padding.",
              "These storage differences do not measure inference RAM, latency, energy, or operating cost.", "",
              f"The files share {g['metadataComparison']['commonEqualCount']} metadata keys with matching types and value digests.",
              f"Bartowski has {len(g['metadataComparison']['rightOnlyKeys'])} additional entries, including {sum(k.startswith('quantize.imatrix.') for k in g['metadataComparison']['rightOnlyKeys'])} `quantize.imatrix.*` entries.",
              "These record producer-supplied metadata. They do not establish the conversion command",
              "or the cause of the encoding differences. Their absence from the IBM file does not",
              "establish whether an importance matrix was used.", "",
              "[Layout and metadata identities](../experiments/gguf-imatrix/results/file-layout.json) record every key,",
              "its serialized span, type and canonical-value digest. Large tokenizer values and producer file",
              "paths are omitted from this record; their original-value digests and lengths are retained.",
              "The [source comparison](../experiments/gguf-imatrix/results/metadata-cross-source-observations.json)",
              "also keeps embedded lineage metadata separate from repository-card declarations.", "",
              "## 3. Across two Qwen publications", "",
              f"Both Qwen3-4B files have the same {result['qwen']['tensorCountPerSubject']} tensor names, stored shapes and encoding assignments.",
              "Their name-to-encoding assignment hashes also match. Numerical weight values were not compared.",
              "The shared release label accompanies different assignments in the Granite pair and matching",
              "assignments in the Qwen pair. These two comparisons do not show how common either outcome is.", "",
              "See the [Qwen protocol, identities and results](../experiments/gguf-paired-replication/qwen3-4b).", "",
              "## 4. Named I/O and internal tensors", "",
              "| Artifact | Direction | Tensor name | Dtype | Shape | Scale | Zero point |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["namedInterfaces"]:
        label = "EfficientNet-Lite0" if row["artifact_sha256"] == EFF else "YAMNet classifier"
        scale = str(row["scales"][0]) if row["scales"] else "Not serialized"
        zero = str(row["zeroPoints"][0]) if row["zeroPoints"] else "Not serialized"
        lines.append(f"| {label} | {row['direction']} | `{row['tensor_name']}` | {row['dtype']} | `{row['shape']}` | {scale} | {zero} |")
    lines += ["", f"EfficientNet artifact SHA-256: `{EFF}`.", f"YAMNet artifact SHA-256: `{YAM}`.", "",
              "| Artifact | INT8 weights | INT8 intermediate tensors | Per-axis weights | Per-axis biases |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["internalRoles"]:
        label = "EfficientNet-Lite0" if row["artifactSha256"] == EFF else "YAMNet classifier"
        lines.append(f"| {label} | {row['int8WeightTensors']} | {row['int8IntermediateTensors']} | {row['perAxisWeightTensors']} | {row['perAxisBiasTensors']} |")
    lines += ["", "Weights are bound to serialized Conv2D, DepthwiseConv2D and FullyConnected input slots,",
              "rather than classified from their names alone. Counts distinguish a tensor from its potentially",
              f"multiple consumers. The {result['internalRoles'][0]['perAxisWeightTensors']+result['internalRoles'][0]['perAxisBiasTensors']} per-axis tensors in EfficientNet comprise {result['internalRoles'][0]['perAxisWeightTensors']} weights and {result['internalRoles'][0]['perAxisBiasTensors']} biases.", "",
              "The [interface ledger](../data/interface-contracts.json) and",
              "[internal tensor contracts](../data/tflite-granularity-evidence.json) contain the source identities",
              "and bindings. Dtype and shape alone do not identify an affine contract; scale and zero point",
              "do not establish RGB order, resizing, or the complete application preprocessing pipeline.", "",
              "## 5. Related importance-matrix evidence", ""]
    scan = matrix["numericalScan"]
    lines += [f"The separately identified importance-matrix artifact contains {scan['assessedTensorCount']:,} F32 tensors",
              f"and {scan['decodedValueCount']:,} decoded values. The recorded complete scan found",
              f"{scan['nonfiniteValueCount']} non-finite values, {scan['exactZeroValueCount']} exact-zero values and",
              f"{scan['constantTensorCount']} constant-valued tensors. This describes the matrix file; it does not",
              "demonstrate that a specific conversion consumed it or that it improved quality.",
              "See the [matrix measurement](../experiments/gguf-imatrix/results/imatrix-measurement.json).", "",
              "## 6. Evidence scope", "",
              "| Observation | Evidence | Boundary |", "| --- | --- | --- |",
              "| Several encodings within one artifact | Exact names, shapes and encodings in the IBM inventory | Names do not establish original layer/module identity |",
              "| Assignments across publications | Complete Granite and Qwen paired inventories | Encoding equality does not establish numerical weight equality |",
              "| External and internal tensor contracts | Named TFLite I/O; selected serialized weight-consumer bindings | No observed runtime kernel placement or device performance |",
              "| Artifact and result identity | Pinned revisions, whole-file hashes and source-record hashes | These do not establish the complete production history |", "",
              "The recorded assignments can serve as test data for model-part descriptions.",
              "Original layer/module mapping and a standardized CycloneDX internal-subject tagging",
              "round trip have not been verified. Model accuracy and device performance were not measured.", "",
              "The existing [CycloneDX probes](../conformance/cyclonedx-2.0/) and",
              "[vocabulary notes](quantization-vocabulary-mapping.md) apply to the draft commits they name.",
              "Passing JSON Schema validation does not establish that a description preserves the observed",
              "tensor assignments. Those probes do not provide a standardized tensor map.", "",
              "## 7. Verification and reproduction", "",
              "Check this brief and the complete paired table using the committed records, without model downloads",
              "or third-party Python packages:", "", "```bash", "python scripts/build-quantization-evidence-brief.py --check", "```", "",
              "This recomputes all selected counts, transitions, payload formulae, metadata-span sums and source",
              "hashes. It verifies the committed observations; it does not reread model bytes. The",
              "[derived record](../experiments/gguf-imatrix/results/evidence-brief.json) names every source and its SHA-256.", "",
              "For original-byte measurements, use the [GGUF protocol](../experiments/gguf-imatrix/protocol.md),",
              "the [paired Qwen protocol](../experiments/gguf-paired-replication/qwen3-4b/protocol.md), and the",
              "[TFLite commands](../README.md#interface-results). Model files remain in the ignored cache.", "",
              "To run the repository's full suite, including earlier experiments, install the main requirements",
              "and run `python scripts/verify-bundle.py`; this has a larger dependency set than the brief check.", "",
              "The GGUF primary reader is upstream `gguf-py`. The author-maintained implementation is disclosed",
              "as a secondary cross-check in its separate directory. Neither the brief nor its offline verifier",
              "requires that implementation. The cases retain their separate acquisition and measurement",
              "protocols; presenting them together is not an independent replication.", ""]
    return "\n".join(lines)


def render_tensors(left, right):
    a, b = {r["name"]: r for r in left["tensors"]}, {r["name"]: r for r in right["tensors"]}
    lines = ["# Complete Granite tensor comparison", "", "Generated from the committed upstream-reader inventories. Shapes use GGUF stored dimension order.",
             "This is an encoding and storage comparison, not a comparison of decoded numerical weight values.", "",
             f"IBM whole-file SHA-256: `{left['source']['sha256']}`.",
             f"Bartowski whole-file SHA-256: `{right['source']['sha256']}`.", "",
             "| # | Tensor | Stored shape | Elements | IBM encoding | Bartowski encoding | IBM payload bytes | Bartowski payload bytes |",
             "| ---: | --- | --- | ---: | --- | --- | ---: | ---: |"]
    for index, name in enumerate(sorted(a), 1):
        x, y = a[name], b[name]
        lines.append(f"| {index} | `{name}` | `{x['shape']}` | {x['elementCount']} | {x['encoding']} | {y['encoding']} | {x['byteLength']} | {y['byteLength']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result, left, right, matrix, layout = build(ROOT)
    outputs = {"docs/quantization-evidence-brief.md": render(result, left, right, matrix, layout),
               GGUF + "/tensor-comparison.md": render_tensors(left, right),
               GGUF + "/results/evidence-brief.json": json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    for relative, expected in outputs.items():
        path = ROOT / relative
        if args.write:
            path.write_text(expected, encoding="utf-8")
        else:
            require(path.read_text(encoding="utf-8") == expected, f"Generated file is stale: {relative}")
    print(json.dumps({"status": "written" if args.write else "pass", "graniteChangedEncodings": result['granite']['changedEncodingCount'],
                      "qwenChangedEncodings": result['qwen']['changedEncodingCount'], "namedInterfaces": len(result['namedInterfaces']),
                      "boundary": "Committed-record verification, not a new model-byte measurement"}, indent=2))


if __name__ == "__main__":
    main()
