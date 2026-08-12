import argparse
import hashlib
import json
import math
import shutil
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import flatbuffers
from ai_edge_litert import schema_py_generated as tflite
from flatbuffers import encode, number_types, packer, table


SCHEMA_VERSION = "tensor_quantization_metadata_study.tflite_metadata_audit.v1"
METADATA_SCHEMA = {
    "semantic_version": "1.5.0",
    "repository": "tensorflow/tflite-support",
    "commit": "78d10177b3bc51f81ea78d8209c557233d15df15",
    "git_blob_sha1": "75e00dfcaa9615ebebe9888fe889802b619d21ce",
    "file_sha256": "2d3386ba124690ba1195bfc1d51ac814843bd675a7c845afab7c001c7891449e",
    "path": "tensorflow_lite_support/metadata/metadata_schema.fbs",
    "flatbuffer_identifier": "M001",
}

CONTENT_TYPES = {
    0: "NONE",
    1: "FEATURE",
    2: "IMAGE",
    3: "BOUNDING_BOX",
    4: "AUDIO",
}
COLOR_SPACES = {0: "UNKNOWN", 1: "RGB", 2: "GRAYSCALE"}
PROCESS_UNIT_TYPES = {
    0: "NONE",
    1: "NORMALIZATION",
    2: "SCORE_CALIBRATION",
    3: "SCORE_THRESHOLDING",
    4: "BERT_TOKENIZER",
    5: "SENTENCEPIECE_TOKENIZER",
    6: "REGEX_TOKENIZER",
}
ASSOCIATED_FILE_TYPES = {
    0: "UNKNOWN",
    1: "DESCRIPTIONS",
    2: "TENSOR_AXIS_LABELS",
    3: "TENSOR_VALUE_LABELS",
    4: "TENSOR_AXIS_SCORE_CALIBRATION",
    5: "VOCABULARY",
    6: "SCANN_INDEX_FILE",
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, expected_size, expected_sha256):
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_sha256
    )


def download_file(url, target, expected_size, expected_sha256):
    if verify_file(target, expected_size, expected_sha256):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tensor-quantization-metadata-study/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    if not verify_file(temporary, expected_size, expected_sha256):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file failed integrity checks: {url}")
    temporary.replace(target)
    return target


def ensure_artifact(cache_root, artifact, allow_download):
    target = cache_root / "models" / f"{artifact['sha256']}.tflite"
    if verify_file(target, artifact["size_bytes"], artifact["sha256"]):
        return target
    if not allow_download:
        raise FileNotFoundError(f"Missing verified artifact cache: {target}")

    source = artifact["download"]
    if source["kind"] == "direct":
        return download_file(
            source["url"],
            target,
            artifact["size_bytes"],
            artifact["sha256"],
        )
    if source["kind"] != "tar_member":
        raise ValueError(f"Unsupported download kind: {source['kind']}")

    archive = cache_root / "archives" / source["archive_filename"]
    download_file(
        source["url"],
        archive,
        source["archive_size_bytes"],
        source["archive_sha256"],
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tflite.part")
    temporary.unlink(missing_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(source["member_path"], "r") as stream:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(stream, output)
    else:
        with tarfile.open(archive, "r:*") as bundle:
            member = bundle.getmember(source["member_path"])
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Archive member has no file data: {member.name}")
            with stream, temporary.open("wb") as output:
                shutil.copyfileobj(stream, output)
    if not verify_file(temporary, artifact["size_bytes"], artifact["sha256"]):
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Extracted member failed integrity checks: {source['member_path']}"
        )
    temporary.replace(target)
    return target


def enum_names(enum_type):
    return {
        value: name
        for name, value in vars(enum_type).items()
        if name.isupper() and isinstance(value, int)
    }


TENSOR_TYPES = enum_names(tflite.TensorType)


def decode_text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def metadata_root(raw):
    offset = encode.Get(packer.uoffset, raw, 0)
    return table.Table(raw, offset)


def field_offset(tab, ordinal):
    return tab.Offset(4 + 2 * ordinal)


def table_field(tab, ordinal):
    offset = field_offset(tab, ordinal)
    if not offset:
        return None
    return table.Table(tab.Bytes, tab.Indirect(offset + tab.Pos))


def table_vector(tab, ordinal):
    offset = field_offset(tab, ordinal)
    if not offset:
        return []
    start = tab.Vector(offset)
    return [
        table.Table(tab.Bytes, tab.Indirect(start + index * 4))
        for index in range(tab.VectorLen(offset))
    ]


def string_field(tab, ordinal):
    offset = field_offset(tab, ordinal)
    if not offset:
        return None
    return decode_text(tab.String(offset + tab.Pos))


def uint8_field(tab, ordinal):
    offset = field_offset(tab, ordinal)
    if not offset:
        return 0
    return tab.Get(number_types.Uint8Flags, offset + tab.Pos)


def float_vector(tab, ordinal):
    offset = field_offset(tab, ordinal)
    if not offset:
        return []
    return [
        float(value)
        for value in tab.GetVectorAsNumpy(number_types.Float32Flags, offset)
    ]


def associated_file_record(tab):
    kind = uint8_field(tab, 2)
    return {
        "name": string_field(tab, 0),
        "description": string_field(tab, 1),
        "type": ASSOCIATED_FILE_TYPES.get(kind, f"UNKNOWN_{kind}"),
        "locale": string_field(tab, 3),
        "version": string_field(tab, 4),
    }


def process_unit_record(tab):
    kind = uint8_field(tab, 0)
    options = table_field(tab, 1)
    row = {
        "type": PROCESS_UNIT_TYPES.get(kind, f"UNKNOWN_{kind}"),
    }
    if kind == 1 and options is not None:
        row["mean"] = float_vector(options, 0)
        row["std"] = float_vector(options, 1)
    return row


def content_record(tab):
    if tab is None:
        return {"type": "NONE", "color_space": None}
    kind = uint8_field(tab, 0)
    properties = table_field(tab, 1)
    row = {
        "type": CONTENT_TYPES.get(kind, f"UNKNOWN_{kind}"),
        "color_space": None,
    }
    if kind == 2 and properties is not None:
        color = uint8_field(properties, 0)
        row["color_space"] = COLOR_SPACES.get(color, f"UNKNOWN_{color}")
    return row


def tensor_metadata_record(tab):
    return {
        "name": string_field(tab, 0),
        "description": string_field(tab, 1),
        "content": content_record(table_field(tab, 3)),
        "process_units": [
            process_unit_record(item) for item in table_vector(tab, 4)
        ],
        "associated_files": [
            associated_file_record(item) for item in table_vector(tab, 6)
        ],
    }


def subgraph_metadata_record(tab):
    return {
        "name": string_field(tab, 0),
        "description": string_field(tab, 1),
        "inputs": [
            tensor_metadata_record(item) for item in table_vector(tab, 2)
        ],
        "outputs": [
            tensor_metadata_record(item) for item in table_vector(tab, 3)
        ],
        "associated_files": [
            associated_file_record(item) for item in table_vector(tab, 4)
        ],
        "input_process_units": [
            process_unit_record(item) for item in table_vector(tab, 5)
        ],
        "output_process_units": [
            process_unit_record(item) for item in table_vector(tab, 6)
        ],
    }


def model_metadata_record(raw):
    if len(raw) < 8 or raw[4:8] != b"M001":
        raise ValueError("TFLITE_METADATA buffer does not have identifier M001")
    root = metadata_root(raw)
    return {
        "name": string_field(root, 0),
        "description": string_field(root, 1),
        "version": string_field(root, 2),
        "subgraphs": [
            subgraph_metadata_record(item) for item in table_vector(root, 3)
        ],
        "author": string_field(root, 4),
        "license": string_field(root, 5),
        "associated_files": [
            associated_file_record(item) for item in table_vector(root, 6)
        ],
        "min_parser_version": string_field(root, 7),
    }


def tensor_shape(tensor):
    values = tensor.ShapeAsNumpy()
    if isinstance(values, int):
        return []
    return [int(value) for value in values]


def core_interface(model):
    if model.SubgraphsLength() == 0:
        return []
    graph = model.Subgraphs(0)
    rows = []
    for direction, length, index_at in (
        ("input", graph.InputsLength(), graph.Inputs),
        ("output", graph.OutputsLength(), graph.Outputs),
    ):
        for ordinal in range(length):
            tensor_index = int(index_at(ordinal))
            tensor = graph.Tensors(tensor_index)
            rows.append({
                "direction": direction,
                "ordinal": ordinal,
                "tensor_index": tensor_index,
                "tensor_name": decode_text(tensor.Name()),
                "dtype": TENSOR_TYPES.get(
                    tensor.Type(), f"UNKNOWN_{tensor.Type()}"
                ),
                "shape": tensor_shape(tensor),
            })
    return rows


def metadata_entries(model):
    rows = []
    for index in range(model.MetadataLength()):
        metadata = model.Metadata(index)
        buffer_index = int(metadata.Buffer())
        buffer = model.Buffers(buffer_index)
        values = buffer.DataAsNumpy()
        raw = b"" if isinstance(values, int) else bytes(values)
        rows.append({
            "ordinal": index,
            "name": decode_text(metadata.Name()),
            "buffer_index": buffer_index,
            "size_bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "identifier": (
                raw[4:8].decode("ascii", errors="replace")
                if len(raw) >= 8
                else None
            ),
            "raw": raw,
        })
    return rows


def packaged_file_names(path):
    if not zipfile.is_zipfile(path):
        return []
    with zipfile.ZipFile(path) as bundle:
        return sorted(bundle.namelist())


def normalization_assessment(tensor_metadata):
    if tensor_metadata is None:
        return {
            "status": "NOT_ASSESSABLE",
            "reason": "No reliably mapped TensorMetadata row is available.",
            "mean": None,
            "std": None,
            "cardinality": None,
        }
    normalizations = [
        row for row in tensor_metadata["process_units"]
        if row["type"] == "NORMALIZATION"
    ]
    if not normalizations:
        return {
            "status": "ABSENT",
            "reason": "No explicit NormalizationOptions process unit is declared.",
            "mean": None,
            "std": None,
            "cardinality": None,
        }
    if len(normalizations) != 1:
        return {
            "status": "PRESENT_INVALID_MULTIPLE",
            "reason": "More than one NormalizationOptions entry applies.",
            "mean": None,
            "std": None,
            "cardinality": None,
        }

    row = normalizations[0]
    mean = row.get("mean", [])
    std = row.get("std", [])
    base = {"mean": mean, "std": std}
    if not mean or not std:
        return {
            **base,
            "status": "PRESENT_INVALID_EMPTY",
            "reason": "Mean and standard-deviation arrays must both be nonempty.",
            "cardinality": None,
        }
    if not all(math.isfinite(value) for value in mean + std):
        return {
            **base,
            "status": "PRESENT_INVALID_NONFINITE",
            "reason": "A normalization value is not finite.",
            "cardinality": None,
        }
    if any(value == 0 for value in std):
        return {
            **base,
            "status": "PRESENT_INVALID_ZERO_STD",
            "reason": "A declared standard deviation is zero.",
            "cardinality": None,
        }

    content = tensor_metadata["content"]
    expected_channels = {
        "RGB": 3,
        "GRAYSCALE": 1,
    }.get(content.get("color_space"))
    if len(mean) == 1 and len(std) == 1:
        return {
            **base,
            "status": "PRESENT_VALID",
            "reason": "Scalar mean and standard deviation broadcast to all channels.",
            "cardinality": "scalar_broadcast",
        }
    if expected_channels is not None:
        if len(mean) in {1, expected_channels} and len(std) in {
            1,
            expected_channels,
        }:
            return {
                **base,
                "status": "PRESENT_VALID",
                "reason": (
                    "Vector cardinality is compatible with the declared "
                    f"{content['color_space']} image content."
                ),
                "cardinality": f"image_channels_{expected_channels}",
            }
        return {
            **base,
            "status": "PRESENT_INVALID_CARDINALITY",
            "reason": (
                "Mean or standard-deviation cardinality is incompatible with "
                f"the declared {content['color_space']} image content."
            ),
            "cardinality": f"image_channels_{expected_channels}",
        }
    return {
        **base,
        "status": "PRESENT_CARDINALITY_UNBOUND",
        "reason": (
            "Normalization values are finite and nonzero, but no declared "
            "channel count is available for semantic cardinality validation."
        ),
        "cardinality": "unbound",
    }


def parameter_metadata_mapping(metadata, core_rows):
    mappings = {}
    if metadata is None:
        for row in core_rows:
            mappings[(row["direction"], row["ordinal"])] = (
                "METADATA_ABSENT",
                None,
            )
        return mappings
    subgraphs = metadata["subgraphs"]
    if not subgraphs:
        for row in core_rows:
            mappings[(row["direction"], row["ordinal"])] = (
                "SUBGRAPH_METADATA_ABSENT",
                None,
            )
        return mappings

    main = subgraphs[0]
    for direction in ("input", "output"):
        core = [row for row in core_rows if row["direction"] == direction]
        declared = main[f"{direction}s"]
        if len(core) != len(declared):
            for row in core:
                mappings[(direction, row["ordinal"])] = (
                    "VECTOR_LENGTH_MISMATCH",
                    None,
                )
            continue
        for row, tensor_metadata in zip(core, declared):
            mappings[(direction, row["ordinal"])] = (
                "MAPPED_BY_DIRECTION_AND_ORDINAL",
                tensor_metadata,
            )
    return mappings


def summarize(artifacts, parameters):
    artifact_status = Counter(row["metadata_status"] for row in artifacts)
    mapping_status = Counter(row["metadata_mapping_status"] for row in parameters)
    normalization_status = Counter(
        row["normalization"]["status"] for row in parameters
        if row["direction"] == "input"
    )
    inputs = [row for row in parameters if row["direction"] == "input"]
    image_inputs = [
        row for row in inputs if row["content"]["type"] == "IMAGE"
    ]
    quantized_inputs = [
        row for row in inputs if row["affine_contract_status"] == "complete"
    ]
    cohorts = {}
    for cohort in sorted({row["cohort"] for row in artifacts}):
        cohort_artifacts = [row for row in artifacts if row["cohort"] == cohort]
        cohort_inputs = [row for row in inputs if row["cohort"] == cohort]
        cohorts[cohort] = {
            "artifact_count": len(cohort_artifacts),
            "parseable_tflite_metadata_count": sum(
                row["metadata_status"] == "PRESENT_PARSEABLE"
                for row in cohort_artifacts
            ),
            "input_parameter_count": len(cohort_inputs),
            "quantized_input_count": sum(
                row["affine_contract_status"] == "complete"
                for row in cohort_inputs
            ),
            "mapped_input_metadata_count": sum(
                row["metadata_mapping_status"]
                == "MAPPED_BY_DIRECTION_AND_ORDINAL"
                for row in cohort_inputs
            ),
            "valid_input_normalization_count": sum(
                row["normalization"]["status"] == "PRESENT_VALID"
                for row in cohort_inputs
            ),
        }
    model_fields = {
        field: sum(
            row["model_metadata"] is not None
            and row["model_metadata"].get(field) is not None
            for row in artifacts
        )
        for field in (
            "name",
            "description",
            "version",
            "author",
            "license",
            "min_parser_version",
        )
    }
    return {
        "artifact_count": len(artifacts),
        "artifact_metadata_status_counts": dict(sorted(artifact_status.items())),
        "artifact_with_parseable_tflite_metadata_count": sum(
            row["metadata_status"] == "PRESENT_PARSEABLE" for row in artifacts
        ),
        "external_parameter_count": len(parameters),
        "input_parameter_count": len(inputs),
        "parameter_mapping_status_counts": dict(sorted(mapping_status.items())),
        "input_normalization_status_counts": dict(
            sorted(normalization_status.items())
        ),
        "mapped_image_input_count": len(image_inputs),
        "mapped_image_input_with_valid_normalization_count": sum(
            row["normalization"]["status"] == "PRESENT_VALID"
            for row in image_inputs
        ),
        "quantized_input_count": len(quantized_inputs),
        "quantized_input_with_valid_normalization_count": sum(
            row["normalization"]["status"] == "PRESENT_VALID"
            for row in quantized_inputs
        ),
        "artifact_with_valid_input_normalization_count": len({
            row["qualified_id"]
            for row in inputs
            if row["normalization"]["status"] == "PRESENT_VALID"
        }),
        "model_metadata_field_presence_counts": model_fields,
        "artifact_with_missing_packaged_associated_file_count": sum(
            bool(row["missing_packaged_associated_file_names"])
            for row in artifacts
        ),
        "cohort_results": cohorts,
        "interpretation_boundary": (
            "NormalizationOptions describes one out-of-graph processing step. "
            "It is separate from core tensor affine metadata and does not by "
            "itself establish a complete application preprocessing pipeline."
        ),
    }


def summary_markdown(result):
    summary = result["summary"]
    statuses = summary["artifact_metadata_status_counts"]
    normalizations = summary["input_normalization_status_counts"]
    cohort_lines = [
        "| Cohort | Artifacts | Metadata | Inputs | Quantized inputs | Valid normalization |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort, row in summary["cohort_results"].items():
        cohort_lines.append(
            f"| {cohort} | {row['artifact_count']} | "
            f"{row['parseable_tflite_metadata_count']} | "
            f"{row['input_parameter_count']} | {row['quantized_input_count']} | "
            f"{row['valid_input_normalization_count']} |"
        )
    lines = [
        "# TFLite metadata audit",
        "",
        "This audit evaluates the existing 50-file curated benchmark corpus.",
        "It is not a prevalence sample of the TFLite ecosystem.",
        "",
        "## Results",
        "",
        f"- Artifacts assessed: {summary['artifact_count']}",
        (
            "- Artifacts with parseable `TFLITE_METADATA`: "
            f"{summary['artifact_with_parseable_tflite_metadata_count']}"
        ),
        f"- Artifact metadata statuses: `{json.dumps(statuses, sort_keys=True)}`",
        f"- External parameters: {summary['external_parameter_count']}",
        f"- Input parameters: {summary['input_parameter_count']}",
        f"- Mapped image inputs: {summary['mapped_image_input_count']}",
        (
            "- Mapped image inputs with valid explicit normalization: "
            f"{summary['mapped_image_input_with_valid_normalization_count']}"
        ),
        f"- Input normalization statuses: `{json.dumps(normalizations, sort_keys=True)}`",
        f"- Quantized inputs: {summary['quantized_input_count']}",
        (
            "- Quantized inputs with valid explicit normalization: "
            f"{summary['quantized_input_with_valid_normalization_count']}"
        ),
        "",
        "## Cohort decomposition",
        "",
        *cohort_lines,
        "",
        "## Interpretation boundary",
        "",
        "Core affine scale and zero-point are converter-serialized tensor facts.",
        "`NormalizationOptions` is a separate publisher-supplied declaration.",
        "Neither metadata presence nor normalization alone establishes resize,",
        "crop, channel ordering, decoder behavior, or the complete operation order.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/audit-tflite-metadata.py --download",
        "python scripts/check-tflite-metadata-audit.py",
        "```",
        "",
        "Downloaded model bytes remain in the ignored `cache` directory.",
        "",
    ]
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=root / "data/artifacts.json",
    )
    parser.add_argument(
        "--interfaces",
        type=Path,
        default=root / "data/interface-contracts.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "cache/tflite-metadata-audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/tflite-metadata-audit.json",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=root / "experiments/tflite-metadata-audit/summary.md",
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    artifact_document = json.loads(args.artifacts.read_text(encoding="utf-8"))
    interface_document = json.loads(args.interfaces.read_text(encoding="utf-8"))
    interfaces = {
        (row["qualified_id"], row["direction"], row["ordinal"]): row
        for row in interface_document["parameters"]
    }

    artifact_rows = []
    parameter_rows = []
    for ordinal, artifact in enumerate(artifact_document["artifacts"], start=1):
        print(
            f"[{ordinal}/{len(artifact_document['artifacts'])}] "
            f"{artifact['qualified_id']}",
            flush=True,
        )
        path = ensure_artifact(args.cache, artifact, args.download)
        raw_model = path.read_bytes()
        model = tflite.Model.GetRootAsModel(raw_model, 0)
        core_rows = core_interface(model)
        entries = metadata_entries(model)
        tflite_entries = [row for row in entries if row["name"] == "TFLITE_METADATA"]
        parsed = None
        parse_error = None
        if not tflite_entries:
            metadata_status = "ABSENT"
        elif len(tflite_entries) > 1:
            metadata_status = "MULTIPLE_TFLITE_METADATA_ENTRIES"
        else:
            try:
                parsed = model_metadata_record(tflite_entries[0]["raw"])
                metadata_status = "PRESENT_PARSEABLE"
            except Exception as error:
                metadata_status = "PRESENT_PARSE_FAILED"
                parse_error = f"{type(error).__name__}: {error}"

        mappings = parameter_metadata_mapping(parsed, core_rows)
        packaged_names = packaged_file_names(path)
        declared_associated = []
        if parsed is not None:
            declared_associated.extend(parsed["associated_files"])
            for subgraph in parsed["subgraphs"]:
                declared_associated.extend(subgraph["associated_files"])
                for tensor_metadata in subgraph["inputs"] + subgraph["outputs"]:
                    declared_associated.extend(tensor_metadata["associated_files"])

        declared_names = sorted({
            row["name"] for row in declared_associated if row["name"]
        })
        missing_packaged = sorted(set(declared_names) - set(packaged_names))
        artifact_rows.append({
            "qualified_id": artifact["qualified_id"],
            "cohort": artifact["cohort"],
            "artifact_sha256": artifact["sha256"],
            "artifact_size_bytes": artifact["size_bytes"],
            "core_metadata_entries": [
                {key: value for key, value in row.items() if key != "raw"}
                for row in entries
            ],
            "tflite_metadata_entry_count": len(tflite_entries),
            "metadata_status": metadata_status,
            "metadata_parse_error": parse_error,
            "model_metadata": (
                None
                if parsed is None
                else {
                    key: parsed[key]
                    for key in (
                        "name",
                        "description",
                        "version",
                        "author",
                        "license",
                        "min_parser_version",
                    )
                }
            ),
            "metadata_subgraph_count": (
                None if parsed is None else len(parsed["subgraphs"])
            ),
            "declared_associated_file_count": (
                None if parsed is None else len(declared_associated)
            ),
            "declared_associated_file_names": (
                None if parsed is None else declared_names
            ),
            "packaged_file_names": packaged_names,
            "missing_packaged_associated_file_names": (
                None if parsed is None else missing_packaged
            ),
        })

        for core in core_rows:
            key = (artifact["qualified_id"], core["direction"], core["ordinal"])
            interface = interfaces.get(key)
            if interface is None:
                raise ValueError(f"Missing interface ledger row: {key}")
            if (
                interface["tensor_index"] != core["tensor_index"]
                or interface["dtype"] != core["dtype"]
                or interface["shape"] != core["shape"]
            ):
                raise ValueError(f"Core/interface ledger mismatch: {key}")
            mapping_status, tensor_metadata = mappings[
                (core["direction"], core["ordinal"])
            ]
            normalization = normalization_assessment(tensor_metadata)
            content = (
                {"type": "NOT_ASSESSABLE", "color_space": None}
                if tensor_metadata is None
                else tensor_metadata["content"]
            )
            affine_status = interface["quantization"]["status"]
            combined_steps = (
                "normalization_and_affine_steps_declared"
                if (
                    core["direction"] == "input"
                    and affine_status == "complete"
                    and normalization["status"] == "PRESENT_VALID"
                )
                else "not_established"
            )
            parameter_rows.append({
                "qualified_id": artifact["qualified_id"],
                "cohort": artifact["cohort"],
                "artifact_sha256": artifact["sha256"],
                "parameter_id": interface["parameter_id"],
                **core,
                "affine_contract_status": affine_status,
                "affine_contract_sha256": interface["quantization"].get(
                    "contract_sha256"
                ),
                "metadata_mapping_status": mapping_status,
                "tensor_metadata_name": (
                    None if tensor_metadata is None else tensor_metadata["name"]
                ),
                "content": content,
                "normalization": normalization,
                "declared_associated_files": (
                    None
                    if tensor_metadata is None
                    else tensor_metadata["associated_files"]
                ),
                "combined_numeric_input_mapping_status": combined_steps,
                "full_preprocessing_pipeline_status": "NOT_ASSESSABLE",
            })

    result = {
        "schema": SCHEMA_VERSION,
        "tool": "scripts/audit-tflite-metadata.py",
        "metadata_schema": METADATA_SCHEMA,
        "source_artifact_manifest_sha256": sha256_file(args.artifacts),
        "source_interface_ledger_sha256": sha256_file(args.interfaces),
        "artifact_count": len(artifact_rows),
        "parameter_count": len(parameter_rows),
        "summary": summarize(artifact_rows, parameter_rows),
        "artifacts": artifact_rows,
        "parameters": parameter_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(summary_markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
