#!/usr/bin/env python3
"""Build source-bound, non-normative CycloneDX field observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def observed(metadata: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: metadata[key] for key in keys if key in metadata}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "experiments/gguf-imatrix"
    sources = experiment / "sources"
    results = experiment / "results"
    mapping_dir = experiment / "standards-mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    source_rows = []
    for path in sorted(
        (
            item for item in sources.rglob("*")
            if item.is_file() and item.name != "SOURCE_MANIFEST.json"
        ),
        key=lambda item: item.as_posix().encode("utf-8"),
    ):
        source_rows.append({
            "path": path.relative_to(experiment).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    write(sources / "SOURCE_MANIFEST.json", {
        "recordType": "tensor-quantization-metadata-study.gguf-source-manifest.v1",
        "revisions": {
            "cycloneDxPr990": {
                "repository": "https://github.com/CycloneDX/specification",
                "pullRequest": 990,
                "commit": "38dfe9ca4b9b161510dd98b477dd78e4eefe7ec8"
            },
            "cycloneDxPr1067": {
                "repository": "https://github.com/CycloneDX/specification",
                "pullRequest": 1067,
                "commit": "7a7d2dd599968e528349ca3cf262120da2831ca8"
            },
            "ibmBaseModel": {
                "repository": "https://huggingface.co/ibm-granite/granite-4.2-3b",
                "revision": "e459acceac81e5fe67c07d9cfc72329a332e7eb1"
            },
            "ibmGguf": {
                "repository": "https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF",
                "revision": "c40945d71cd90f249a56985e8155551a9188dc30"
            },
            "bartowskiGguf": {
                "repository": "https://huggingface.co/bartowski/granite-4.2-3b-GGUF",
                "revision": "4093456941a783ab5d8268e00a7725532c1e9af3"
            },
            "ibmGgufPipeline": {
                "repository": "https://github.com/IBM/gguf",
                "commit": "33fdd16c2dd4edbb1df71ac8feef3d98ce3faa77"
            },
            "ibmGranite42Docs": {
                "repository": "https://github.com/ibm-granite/granite-4.2-language-models",
                "commit": "5173b090a381c87e689995bf63db4aeafd12285d"
            },
            "llamaCpp": {
                "repository": "https://github.com/ggml-org/llama.cpp",
                "commit": "c060ca974c773c7c3d17fd1b66dc9d312bc292c0"
            },
            "ggmlDocs": {
                "repository": "https://github.com/ggml-org/ggml",
                "commit": "7840aaba1989c6deeefede1d77d5aaf8f52b947e"
            }
        },
        "files": source_rows,
    })

    schema_path = sources / "pr-990/cyclonedx-ai-ml-2.0.schema.json"
    schema = load(schema_path)
    model_properties = schema["$defs"]["modelProperties"]["properties"]
    quantization = schema["$defs"]["quantization"]
    parameter = schema["$defs"]["modelParameter"]["properties"]
    pr990_dir = mapping_dir / "pr-990"
    pr990_dir.mkdir(parents=True, exist_ok=True)
    write(pr990_dir / "schema-observation.json", {
        "recordType": "tensor-quantization-metadata-study.cyclonedx-pr990-schema-observation.v1",
        "normativeStatus": "observation of a draft schema; no schema change is proposed by this record",
        "source": {
            "pullRequest": 990,
            "head": "38dfe9ca4b9b161510dd98b477dd78e4eefe7ec8",
            "path": "schema/2.0/model/cyclonedx-ai-ml-2.0.schema.json",
            "sha256": sha256(schema_path),
        },
        "observations": [
            {
                "jsonPointer": "/$defs/modelProperties/properties/quantization",
                "description": model_properties["quantization"]["description"],
                "reference": model_properties["quantization"]["$ref"],
            },
            {
                "jsonPointer": "/$defs/modelProperties/properties/inputs",
                "description": model_properties["inputs"]["description"],
                "itemReference": model_properties["inputs"]["items"]["$ref"],
            },
            {
                "jsonPointer": "/$defs/modelProperties/properties/outputs",
                "description": model_properties["outputs"]["description"],
                "itemReference": model_properties["outputs"]["items"]["$ref"],
            },
            {
                "jsonPointer": "/$defs/quantization",
                "description": quantization["description"],
                "propertyNames": sorted(quantization["properties"]),
            },
            {
                "jsonPointer": "/$defs/modelParameter/properties/quantization",
                "description": parameter["quantization"]["description"],
                "reference": parameter["quantization"]["$ref"],
            },
        ],
    })

    perspective_path = sources / "pr-1067/model-card-perspective.json"
    perspective = load(perspective_path)["perspectives"][0]
    required = [item for item in perspective["mappings"] if item.get("relevance") == "required"]
    by_name = {item["nativeName"]: item for item in required}
    ibm_metadata = load(results / "ibm-q4km-tensor-inventory.json")["format"]["selectedMetadata"]
    bart_metadata = load(results / "bartowski-q4km-tensor-inventory.json")["format"]["selectedMetadata"]
    definitions = [
        ("Model Details", ["general.name", "general.version", "general.description"], "partial semantic overlap"),
        ("Developed By", ["general.author", "general.organization"], "potential semantic overlap; not a CycloneDX party-role assertion"),
        ("License", ["general.license", "general.license.name", "general.license.link"], "direct artifact license metadata"),
        ("Supported Tasks", ["general.tags"], "tags may be relevant but are not a standardized task mapping"),
        ("Dataset Identity and Licensing", ["general.datasets"], "dataset-name metadata, if present, does not alone establish all selected identity and license fields"),
        ("Intended Use", [], "no directly corresponding standardized GGUF key identified in the pinned GGUF specification"),
        ("Technical Limitations", ["general.description"], "free-form description is not a structured limitation field"),
        ("Ethical Considerations", [], "no directly corresponding standardized GGUF risk object identified in the pinned GGUF specification"),
    ]
    rows = []
    for name, keys, relationship in definitions:
        mapping = by_name[name]
        rows.append({
            "mapping": name,
            "catalogExpression": mapping["expression"],
            "catalogVia": mapping.get("via", []),
            "ggufStandardizedRelevantKeys": keys,
            "semanticRelationship": relationship,
            "ibmSerializedValues": observed(ibm_metadata, keys),
            "bartowskiSerializedValues": observed(bart_metadata, keys),
            "primaryReader": "gguf-py",
        })
    pr1067_dir = mapping_dir / "pr-1067"
    pr1067_dir.mkdir(parents=True, exist_ok=True)
    write(pr1067_dir / "model-card-artifact-field-observations.json", {
        "recordType": "tensor-quantization-metadata-study.cyclonedx-pr1067-artifact-field-observations.v1",
        "normativeStatus": "field observations only; not a completeness or conformance verdict",
        "source": {
            "pullRequest": 1067,
            "head": "7a7d2dd599968e528349ca3cf262120da2831ca8",
            "path": "perspectives/model-card-perspective.json",
            "sha256": sha256(perspective_path),
            "requiredMappingCount": len(required),
        },
        "subjects": [
            "ibm-granite-4.2-3b-q4-k-m",
            "bartowski-granite-4.2-3b-q4-k-m",
        ],
        "rows": rows,
        "interpretationBoundary": (
            "GGUF permits custom metadata, but this record distinguishes standardized GGUF keys "
            "from arbitrary extension capacity. Path selection, per-subject aggregation, empty-value "
            "handling, and publisher-evidence binding are not decided here."
        ),
    })

    ibm_api = load(sources / "hf-api/ibm-gguf.json")
    bart_api = load(sources / "hf-api/bartowski-gguf.json")
    imatrix_keys = sorted(
        key for key in set(ibm_metadata) | set(bart_metadata)
        if key.startswith("quantize.imatrix.")
    )
    write(results / "metadata-cross-source-observations.json", {
        "recordType": "tensor-quantization-metadata-study.gguf-metadata-cross-source-observations.v1",
        "evidenceClass": "comparison of serialized GGUF metadata and revision-pinned repository metadata",
        "subjects": [
            "ibm-granite-4.2-3b-q4-k-m",
            "bartowski-granite-4.2-3b-q4-k-m",
        ],
        "commonSerializedValues": {
            key: ibm_metadata[key]
            for key in ("general.name", "general.license", "general.file_type")
            if key in ibm_metadata and ibm_metadata[key] == bart_metadata.get(key)
        },
        "importanceMatrixMetadata": {
            "keysConsidered": imatrix_keys,
            "ibmSerializedValues": observed(ibm_metadata, imatrix_keys),
            "bartowskiSerializedValues": observed(bart_metadata, imatrix_keys),
            "boundary": (
                "Presence records serialized producer metadata. Absence does not establish "
                "that an importance matrix was not used."
            ),
        },
        "lineageMetadata": {
            "bartowskiArtifactSerializedBaseModel": observed(
                bart_metadata,
                [
                    "general.base_model.0.name",
                    "general.base_model.0.organization",
                    "general.base_model.0.repo_url",
                ],
            ),
            "bartowskiRepositoryCardBaseModel": bart_api.get("cardData", {}).get("base_model"),
            "ibmRepositoryCardBaseModel": ibm_api.get("cardData", {}).get("base_model"),
            "boundary": (
                "These values may describe different lineage levels or relations. This record "
                "does not collapse them into one canonical source assertion."
            ),
        },
    })


if __name__ == "__main__":
    main()
