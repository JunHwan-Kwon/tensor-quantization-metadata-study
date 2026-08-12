import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


SCHEMA = "tensor_quantization_metadata_study.kaggle_tflite_audit.v1"
MATERIALIZATION_SCHEMA = (
    "tensor_quantization_metadata_study.kaggle_tflite_materialization.v1"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def public_error(error):
    message = str(error)
    return {
        "class": type(error).__name__,
        "message_sha256": hashlib.sha256(
            message.encode("utf-8", errors="replace")
        ).hexdigest(),
    }


def load_script(root, filename, module_name):
    path = root / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def affine_assessment(parameter):
    scales = parameter["scales"]
    zero_points = parameter["zero_points"]
    dimension = parameter["quantized_dimension"]
    shape = parameter["shape"]
    if not scales and not zero_points:
        return {
            "status": "NOT_QUANTIZED",
            "granularity": None,
            "contract_sha256": None,
            "reason": "No scale or zero-point values are serialized.",
        }
    if not scales or not zero_points or len(scales) != len(zero_points):
        return {
            "status": "INVALID_CARDINALITY",
            "granularity": None,
            "contract_sha256": None,
            "reason": "Scale and zero-point arrays are empty or unequal.",
        }
    if any(value <= 0 for value in scales):
        return {
            "status": "INVALID_SCALE",
            "granularity": None,
            "contract_sha256": None,
            "reason": "Every affine scale must be positive.",
        }
    if len(scales) == 1:
        granularity = "PER_TENSOR"
    else:
        axis = dimension
        if axis < 0 or axis >= len(shape):
            return {
                "status": "INVALID_AXIS",
                "granularity": "PER_AXIS",
                "contract_sha256": None,
                "reason": "The quantized dimension is outside the tensor rank.",
            }
        if len(scales) != shape[axis]:
            return {
                "status": "INVALID_AXIS_CARDINALITY",
                "granularity": "PER_AXIS",
                "contract_sha256": None,
                "reason": "Scale count does not equal shape[axis].",
            }
        granularity = "PER_AXIS"
    contract = {
        "dtype": parameter["dtype"],
        "shape": shape,
        "scales": scales,
        "zero_points": zero_points,
        "quantized_dimension": dimension,
    }
    return {
        "status": "COMPLETE",
        "granularity": granularity,
        "contract_sha256": sha256_json(contract),
        "reason": "Affine arrays and cardinality are valid.",
    }


def initializer_fingerprint(model, auditor):
    if model.SubgraphsLength() == 0:
        return {"count": 0, "ledger_sha256": sha256_json([]), "entries": []}
    graph = model.Subgraphs(0)
    input_indices = {
        int(graph.Inputs(index)) for index in range(graph.InputsLength())
    }
    raw_entries = []
    for tensor_index in range(graph.TensorsLength()):
        if tensor_index in input_indices:
            continue
        tensor = graph.Tensors(tensor_index)
        buffer_index = int(tensor.Buffer())
        buffer = model.Buffers(buffer_index)
        values = buffer.DataAsNumpy()
        raw = b"" if isinstance(values, int) else bytes(values)
        if not raw:
            continue
        raw_entries.append({
            "name": auditor.decode_text(tensor.Name()),
            "dtype": auditor.TENSOR_TYPES.get(
                tensor.Type(), f"UNKNOWN_{tensor.Type()}"
            ),
            "shape": auditor.tensor_shape(tensor),
            "size_bytes": len(raw),
            "data_sha256": sha256_bytes(raw),
        })
    raw_entries.sort(key=lambda row: (
        row["name"] or "",
        row["dtype"],
        row["shape"],
        row["size_bytes"],
        row["data_sha256"],
    ))
    entries = []
    occurrences = Counter()
    for row in raw_entries:
        base = sha256_json({
            key: row[key] for key in ("name", "dtype", "shape", "size_bytes")
        })
        ordinal = occurrences[base]
        occurrences[base] += 1
        entries.append({**row, "match_key": f"{base}:{ordinal}"})
    return {
        "count": len(entries),
        "ledger_sha256": sha256_json(entries),
        "entries": entries,
    }


def metadata_assessment(model, path, core_rows, auditor):
    entries = auditor.metadata_entries(model)
    metadata_entries = [
        row for row in entries if row["name"] == "TFLITE_METADATA"
    ]
    parsed = None
    parse_error = None
    if not metadata_entries:
        status = "ABSENT"
    elif len(metadata_entries) > 1:
        status = "MULTIPLE_TFLITE_METADATA_ENTRIES"
    else:
        try:
            parsed = auditor.model_metadata_record(metadata_entries[0]["raw"])
            status = "PRESENT_PARSEABLE"
        except Exception as error:
            status = "PRESENT_PARSE_FAILED"
            parse_error = f"{type(error).__name__}: {error}"
    mappings = auditor.parameter_metadata_mapping(parsed, core_rows)
    return {
        "status": status,
        "parse_error": parse_error,
        "entry_count": len(metadata_entries),
        "model_metadata": None if parsed is None else {
            key: parsed[key]
            for key in (
                "name",
                "description",
                "version",
                "author",
                "license",
                "min_parser_version",
            )
        },
        "mappings": mappings,
        "packaged_file_names": auditor.packaged_file_names(path),
    }


def parse_artifact(path, auditor, litert_reader):
    raw = path.read_bytes()
    model = auditor.tflite.Model.GetRootAsModel(raw, 0)
    core_rows = auditor.core_interface(model)
    litert_rows = litert_reader.interpreter_parameters(path)
    core_projection = [{
        key: row[key]
        for key in ("direction", "ordinal", "tensor_index", "dtype", "shape")
    } for row in core_rows]
    litert_projection = [{
        key: row[key]
        for key in ("direction", "ordinal", "tensor_index", "dtype", "shape")
    } for row in litert_rows]
    if core_projection != litert_projection:
        raise ValueError("Direct FlatBuffer and LiteRT interfaces differ")
    metadata = metadata_assessment(model, path, core_rows, auditor)
    parameters = []
    for core, litert in zip(core_rows, litert_rows):
        mapping_status, tensor_metadata = metadata["mappings"][
            (core["direction"], core["ordinal"])
        ]
        normalization = auditor.normalization_assessment(tensor_metadata)
        content = (
            {"type": "NOT_ASSESSABLE", "color_space": None}
            if tensor_metadata is None else tensor_metadata["content"]
        )
        affine = affine_assessment(litert)
        parameters.append({
            **core,
            "scales": litert["scales"],
            "zero_points": litert["zero_points"],
            "quantized_dimension": litert["quantized_dimension"],
            "affine": affine,
            "metadata_mapping_status": mapping_status,
            "content": content,
            "normalization": normalization,
            "full_preprocessing_pipeline_status": "NOT_ASSESSABLE",
        })
    return {
        "metadata": {
            key: value for key, value in metadata.items() if key != "mappings"
        },
        "initializers": initializer_fingerprint(model, auditor),
        "parameters": parameters,
    }


def summarize(artifacts, parameters):
    assessed = [row for row in artifacts if row["status"] == "ASSESSED"]
    assessed_ids = {row["published_file_id"] for row in assessed}
    assessed_parameters = [
        row for row in parameters if row["published_file_id"] in assessed_ids
    ]
    integer_types = {"INT8", "UINT8", "INT16", "UINT16"}
    integer_files = {
        row["published_file_id"]
        for row in assessed_parameters
        if row["dtype"] in integer_types
        and row["affine"]["status"] == "COMPLETE"
    }
    integer_hashes = {
        row["artifact_sha256"]
        for row in assessed
        if row["published_file_id"] in integer_files
    }
    metadata_files = {
        row["published_file_id"]
        for row in assessed
        if row["metadata"]["status"] == "PRESENT_PARSEABLE"
    }
    metadata_hashes = {
        row["artifact_sha256"]
        for row in assessed
        if row["published_file_id"] in metadata_files
    }
    mapped_images = [
        row for row in assessed_parameters
        if row["direction"] == "input" and row["content"]["type"] == "IMAGE"
    ]
    return {
        "published_file_count": len(artifacts),
        "artifact_status_counts": dict(sorted(Counter(
            row["status"] for row in artifacts
        ).items())),
        "assessed_published_file_count": len(assessed),
        "assessed_unique_artifact_count": len({
            row["artifact_sha256"] for row in assessed
        }),
        "integer_affine_interface_published_file_count": len(integer_files),
        "integer_affine_interface_unique_artifact_count": len(integer_hashes),
        "parseable_tflite_metadata_published_file_count": len(metadata_files),
        "parseable_tflite_metadata_unique_artifact_count": len(metadata_hashes),
        "mapped_image_input_count": len(mapped_images),
        "mapped_image_input_with_valid_normalization_count": sum(
            row["normalization"]["status"] == "PRESENT_VALID"
            for row in mapped_images
        ),
        "external_parameter_count": len(assessed_parameters),
        "affine_status_counts": dict(sorted(Counter(
            row["affine"]["status"] for row in assessed_parameters
        ).items())),
    }


def signature_groups(parameters):
    code_ranges = {
        "INT8": (-128, 127),
        "UINT8": (0, 255),
        "INT16": (-32768, 32767),
        "UINT16": (0, 65535),
        "INT32": (-2147483648, 2147483647),
        "UINT32": (0, 4294967295),
        "INT64": (-9223372036854775808, 9223372036854775807),
        "UINT64": (0, 18446744073709551615),
        "INT4": (-8, 7),
    }
    groups = {}
    for row in parameters:
        if row["affine"]["status"] != "COMPLETE":
            continue
        key = (
            row["direction"],
            row["dtype"],
            tuple(row["shape"]),
        )
        group = groups.setdefault(key, {
            "artifact_sha256": set(),
            "contract_sha256": set(),
            "contracts": {},
        })
        group["artifact_sha256"].add(row["artifact_sha256"])
        contract_hash = row["affine"]["contract_sha256"]
        group["contract_sha256"].add(contract_hash)
        group["contracts"][contract_hash] = {
            "scales": row["scales"],
            "zero_points": row["zero_points"],
            "quantized_dimension": row["quantized_dimension"],
        }
    rows = []
    for key, group in sorted(groups.items()):
        if len(group["artifact_sha256"]) < 2:
            continue
        domains = {}
        qmin, qmax = code_ranges[key[1]]
        for contract_hash, contract in group["contracts"].items():
            endpoints = [
                ((qmin - zero_point) * scale, (qmax - zero_point) * scale)
                for scale, zero_point in zip(
                    contract["scales"], contract["zero_points"]
                )
            ]
            domains[contract_hash] = {
                "real_min": min(row[0] for row in endpoints),
                "real_max": max(row[1] for row in endpoints),
            }
        real_mins = [row["real_min"] for row in domains.values()]
        real_maxs = [row["real_max"] for row in domains.values()]
        rows.append({
            "direction": key[0],
            "dtype": key[1],
            "shape": list(key[2]),
            "unique_artifact_count": len(group["artifact_sha256"]),
            "unique_contract_count": len(group["contract_sha256"]),
            "ambiguous": len(group["contract_sha256"]) > 1,
            "contract_sha256": sorted(group["contract_sha256"]),
            "contracts": dict(sorted(group["contracts"].items())),
            "real_domains": dict(sorted(domains.items())),
            "max_pairwise_real_domain_endpoint_difference": max(
                max(real_mins) - min(real_mins),
                max(real_maxs) - min(real_maxs),
            ),
        })
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--materialization",
        type=Path,
        default=root / "data/kaggle-tflite-materialization.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/kaggle-tflite-audit.json",
    )
    args = parser.parse_args()
    materialization = json.loads(
        args.materialization.read_text(encoding="utf-8")
    )
    if materialization["schema"] != MATERIALIZATION_SCHEMA:
        raise ValueError("Unexpected materialization schema")
    auditor = load_script(
        root, "audit-tflite-metadata.py", "tflite_metadata_auditor"
    )
    litert_reader = load_script(
        root, "verify-litert-interfaces.py", "litert_interface_reader"
    )

    artifacts = []
    parameters = []
    for version in materialization["versions"]:
        for file in version["files"]:
            published_file_id = (
                f"{version['version_ref']}:{file['logical_path']}"
            )
            base = {
                "published_file_id": published_file_id,
                "variation_ref": version["variation_ref"],
                "version_number": version["version_number"],
                "logical_path": file["logical_path"],
                "artifact_sha256": file["artifact_sha256"],
                "artifact_size_bytes": file["artifact_size_bytes"],
            }
            if file["status"] != "ASSESSED_DOWNLOAD":
                artifacts.append({
                    **base,
                    "status": file["status"],
                    "error": version["error"],
                    "metadata": None,
                    "initializers": None,
                })
                continue
            path = root / file["cache_path"]
            if not path.is_file():
                artifacts.append({
                    **base,
                    "status": "CACHE_FILE_MISSING",
                    "error": None,
                    "metadata": None,
                    "initializers": None,
                })
                continue
            if (
                path.stat().st_size != file["artifact_size_bytes"]
                or sha256_file(path) != file["artifact_sha256"]
            ):
                artifacts.append({
                    **base,
                    "status": "CACHE_INTEGRITY_FAILED",
                    "error": None,
                    "metadata": None,
                    "initializers": None,
                })
                continue
            try:
                parsed = parse_artifact(path, auditor, litert_reader)
                artifacts.append({
                    **base,
                    "status": "ASSESSED",
                    "error": None,
                    "metadata": parsed["metadata"],
                    "initializers": parsed["initializers"],
                })
                for row in parsed["parameters"]:
                    parameters.append({
                        "published_file_id": published_file_id,
                        "artifact_sha256": file["artifact_sha256"],
                        **row,
                    })
            except Exception as error:
                artifacts.append({
                    **base,
                    "status": "TFLITE_PARSE_FAILED",
                    "error": public_error(error),
                    "metadata": None,
                    "initializers": None,
                })

    signatures = signature_groups(parameters)
    summary = summarize(artifacts, parameters)
    summary["multi_artifact_affine_signature_count"] = len(signatures)
    summary["ambiguous_affine_signature_count"] = sum(
        row["ambiguous"] for row in signatures
    )
    document = {
        "schema": SCHEMA,
        "source_materialization_sha256": sha256_file(args.materialization),
        "scope": materialization["scope"],
        "summary": summary,
        "signature_groups": signatures,
        "artifacts": artifacts,
        "parameters": parameters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
