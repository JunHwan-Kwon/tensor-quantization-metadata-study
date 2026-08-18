import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator


def fail(message):
    raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def check_schema_audit(root):
    document = json.loads(
        (root / "data/cyclonedx-2.0-quantization-schema-audit.json")
        .read_text(encoding="utf-8")
    )
    if document["source"]["commit"] != (
        "49a945618811213e55686a23fa63b287940071c6"
    ):
        fail("Unexpected CycloneDX specification commit")
    definition = document["quantization_definition"]
    digest = hashlib.sha256(canonical_json_bytes(definition)).hexdigest()
    if digest != document["quantization_definition_sha256"]:
        fail("Quantization definition digest mismatch")
    validator = Draft202012Validator(definition)
    for case in document["cases"]:
        valid = not list(validator.iter_errors(case["instance"]))
        if valid != case["current_valid"]:
            fail(f"Schema audit result mismatch: {case['id']}")
    if document["case_count"] != len(document["cases"]):
        fail("Schema audit case count mismatch")
    if document["valid_case_count"] != sum(
        row["current_valid"] for row in document["cases"]
    ):
        fail("Schema audit valid count mismatch")
    if (document["valid_case_count"], document["invalid_case_count"]) != (8, 4):
        fail("Unexpected 12-case schema result distribution")
    expected_context_results = {
        "parameter-negative-source-axis-with-static-rank": False,
        "parameter-normalized-effective-axis-with-static-rank": True,
        "parameter-axis-without-shape": True,
        "model-properties-negative-source-axis": False,
        "model-properties-nonnegative-axis-without-weight-rank": True,
    }
    actual_context_results = {
        row["id"]: row["current_valid"]
        for row in document["normalization_probes"]
    }
    if actual_context_results != expected_context_results:
        fail("Unexpected axis-normalization probe results")
    usage_sites = document["quantization_usage_sites"]
    if len(usage_sites) != 2 or {
        row["reference"] for row in usage_sites
    } != {"#/$defs/quantization"}:
        fail("Quantization usage sites do not share one definition")
    usage_by_pointer = {row["json_pointer"]: row for row in usage_sites}
    if usage_by_pointer[
        "/$defs/modelProperties/properties/quantization"
    ]["direct_shape_property_available"]:
        fail("Model-properties scope unexpectedly exposes a weight shape")
    parameter_usage = usage_by_pointer[
        "/$defs/modelParameter/properties/quantization"
    ]
    if not parameter_usage["direct_shape_property_available"]:
        fail("Model-parameter scope is missing its shape association")
    if parameter_usage["shape_required"]:
        fail("Pinned draft unexpectedly requires parameter shape")
    return definition


def check_tflite_evidence(root):
    document = json.loads(
        (root / "data/tflite-granularity-evidence.json")
        .read_text(encoding="utf-8")
    )
    if document["source_artifact_manifest_sha256"] != sha256_file(
        root / "data/artifacts.json"
    ):
        fail("Granularity evidence is not bound to the artifact manifest")
    if document["source_interface_contract_ledger_sha256"] != sha256_file(
        root / "data/interface-contracts.json"
    ):
        fail("Granularity evidence is not bound to the interface ledger")
    rows = document["artifact_rows"]
    if len(rows) != document["verified_artifact_count"]:
        fail("Granularity artifact-row count mismatch")
    if len({row["artifact_sha256"] for row in rows}) != len(rows):
        fail("Granularity artifact SHA-256 values are not unique")
    manifest = json.loads(
        (root / "data/artifacts.json").read_text(encoding="utf-8")
    )
    expected = {
        row["sha256"]: row["qualified_id"] for row in manifest["artifacts"]
    }
    for row in rows:
        if expected.get(row["artifact_sha256"]) != row["qualified_id"]:
            fail("Granularity row does not match the artifact manifest")

    counts = Counter()
    axes = Counter()
    scale_lengths = Counter()
    constant_roles = Counter()
    weight_contracts = Counter()
    weight_tensor_op_axes = Counter()
    weight_consumer_binding_op_axes = Counter()
    projection_statuses = Counter()
    violation_count = 0
    external_parameter_count = 0
    mixed_weight_axis_artifacts = 0
    no_lossless_projection_weight_tensors = 0
    best_candidate_losses = []
    for row in rows:
        counts.update(row["counts"])
        axes.update(row["axis_counts"])
        scale_lengths.update({
            int(key): value
            for key, value in row["scale_vector_length_counts"].items()
        })
        constant_roles.update(row["constant_role_counts"])
        weight_contracts.update(row["known_weight_contract_counts"])
        weight_tensor_op_axes.update(row["known_weight_tensor_op_axis_counts"])
        weight_consumer_binding_op_axes.update(
            row["known_weight_consumer_binding_op_axis_counts"]
        )
        projection = row["model_level_single_object_projection"]
        projection_statuses[projection["status"]] += 1
        mixed_weight_axis_artifacts += (
            len(projection["observed_per_channel_axes"]) > 1
        )
        row_weight_contracts = Counter(
            contract["contract_signature"]
            for contract in row["known_weight_contracts"]
        )
        if dict(sorted(row_weight_contracts.items())) != (
            row["known_weight_contract_counts"]
        ):
            fail("Per-artifact weight-contract conservation failed")
        if projection["known_weight_tensor_count"] != sum(
            row_weight_contracts.values()
        ):
            fail("Projection weight-tensor count mismatch")
        expected_projection_status = (
            "not_assessed_no_known_weight_binding"
            if not row_weight_contracts
            else "lossless_single_object_projection_available"
            if len(row_weight_contracts) == 1
            else "no_lossless_single_object_projection"
        )
        if projection["status"] != expected_projection_status:
            fail("Single-object projection status mismatch")
        if projection["status"] == "no_lossless_single_object_projection":
            no_lossless_projection_weight_tensors += projection[
                "known_weight_tensor_count"
            ]
            best_candidate_losses.append(min(
                candidate["omitted_or_misrepresented_weight_tensors"]
                for candidate in projection["candidate_information_loss"]
            ))
        for candidate in projection["candidate_information_loss"]:
            signature = candidate["selected_contract_signature"]
            covered = row_weight_contracts[signature]
            if candidate["exactly_represented_weight_tensors"] != covered:
                fail("Projection candidate coverage mismatch")
            if candidate["omitted_or_misrepresented_weight_tensors"] != (
                sum(row_weight_contracts.values()) - covered
            ):
                fail("Projection candidate loss mismatch")
        for contract in row["known_weight_contracts"]:
            if contract["scale_count"] != contract["zero_point_count"]:
                fail("Weight scale/zero-point cardinality mismatch")
            if contract["semantic_granularity"] == "per_axis":
                axis = contract["semantic_axis"]
                shape = contract["shape"]
                if axis is None or contract["scale_count"] != shape[axis]:
                    fail("Per-axis weight cardinality mismatch")
            elif contract["semantic_axis"] is not None:
                fail("Per-tensor weight has a semantic axis")
            for binding in contract["weight_consumers"]:
                if binding["role"] != "weight" or binding["input_slot"] != 1:
                    fail("Unexpected known weight binding")
        violation_count += row["violation_count"]
        external_parameter_count += len(row["external_parameters"])
    if dict(sorted(counts.items())) != document["counts"]:
        fail("Granularity count conservation failed")
    if dict(sorted(axes.items())) != document["axis_counts"]:
        fail("Granularity axis-count conservation failed")
    expected_scale_lengths = {
        str(key): value for key, value in sorted(scale_lengths.items())
    }
    if expected_scale_lengths != document["scale_vector_length_counts"]:
        fail("Scale-vector length conservation failed")
    if dict(sorted(constant_roles.items())) != document["constant_role_counts"]:
        fail("Constant-role conservation failed")
    if dict(sorted(weight_contracts.items())) != (
        document["known_weight_contract_counts"]
    ):
        fail("Weight-contract conservation failed")
    if dict(sorted(weight_tensor_op_axes.items())) != (
        document["known_weight_tensor_op_axis_counts"]
    ):
        fail("Weight tensor op-axis conservation failed")
    if dict(sorted(weight_consumer_binding_op_axes.items())) != (
        document["known_weight_consumer_binding_op_axis_counts"]
    ):
        fail("Weight consumer-binding op-axis conservation failed")
    if dict(sorted(projection_statuses.items())) != (
        document["model_level_projection_status_counts"]
    ):
        fail("Projection-status conservation failed")
    if mixed_weight_axis_artifacts != document[
        "mixed_weight_axis_artifact_count"
    ]:
        fail("Mixed weight-axis artifact conservation failed")
    if no_lossless_projection_weight_tensors != document[
        "no_lossless_projection_known_weight_tensor_count"
    ]:
        fail("No-lossless-projection weight count mismatch")
    if sum(best_candidate_losses) != document[
        "best_single_candidate_omitted_weight_tensor_count"
    ]:
        fail("Best-candidate aggregate loss mismatch")
    if [min(best_candidate_losses), max(best_candidate_losses)] != document[
        "best_single_candidate_omitted_weight_tensor_range"
    ]:
        fail("Best-candidate loss range mismatch")
    if violation_count != document["violation_count"] or violation_count != 0:
        fail("Granularity evidence contains a semantic violation")
    if external_parameter_count != document["external_parameter_count"]:
        fail("External parameter count conservation failed")

    expected_counts = {
        "external:per_tensor": 62,
        "external:per_tensor:serialized_axis:0": 62,
        "constant:per_axis": 6926,
        "constant:per_axis:INT8": 3443,
        "constant:per_axis:INT32": 3483,
        "internal_nonconstant:per_tensor": 8661,
    }
    for key, value in expected_counts.items():
        if document["counts"].get(key) != value:
            fail(f"Unexpected granularity evidence count: {key}")
    if document["per_axis_artifact_count"] != 30:
        fail("Unexpected per-axis artifact count")
    if document["constant_per_axis_artifact_count"] != 30:
        fail("Unexpected constant per-axis artifact count")
    if document["multiple_constant_axis_artifact_count"] != 18:
        fail("Unexpected mixed constant-axis artifact count")
    if document["constant_axis_set_artifact_counts"] != {
        "0": 12,
        "0,3": 18,
    }:
        fail("Unexpected constant-axis set distribution")
    if document["minimum_scale_vector_length"] != 2:
        fail("Unexpected minimum scale-vector length")
    if document["maximum_scale_vector_length"] != 2048:
        fail("Unexpected maximum scale-vector length")
    if document["known_weight_contract_counts"] != {
        "per-channel:axis=0": 3116,
        "per-channel:axis=3": 327,
        "per-tensor": 229,
    }:
        fail("Unexpected known weight-contract distribution")
    if document["known_weight_tensor_op_axis_counts"] != {
        "CONV_2D:per_axis:axis=0": 3099,
        "CONV_2D:per_tensor:axis=not-applicable": 180,
        "DEPTHWISE_CONV_2D:per_axis:axis=3": 327,
        "DEPTHWISE_CONV_2D:per_tensor:axis=not-applicable": 43,
        "FULLY_CONNECTED:per_axis:axis=0": 17,
        "FULLY_CONNECTED:per_tensor:axis=not-applicable": 6,
    }:
        fail("Unexpected unique weight tensor op-axis distribution")
    if document["known_weight_tensor_count"] != 3672:
        fail("Unexpected known weight tensor count")
    if document["mixed_weight_axis_artifact_count"] != 18:
        fail("Unexpected mixed weight-axis artifact count")
    if document["no_lossless_projection_known_weight_tensor_count"] != 1033:
        fail("Unexpected no-lossless-projection weight count")
    if document["best_single_candidate_omitted_weight_tensor_count"] != 330:
        fail("Unexpected aggregate best-candidate information loss")
    if document["best_single_candidate_omitted_weight_tensor_range"] != [11, 48]:
        fail("Unexpected best-candidate information-loss range")
    if document["model_level_projection_status_counts"] != {
        "lossless_single_object_projection_available": 17,
        "no_lossless_single_object_projection": 18,
        "not_assessed_no_known_weight_binding": 15,
    }:
        fail("Unexpected model-level projection distribution")


def check_onnx_blocked_evidence(root, quantization_definition):
    path = root / "data/onnx-blocked-quantization-evidence.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        regenerated_path = Path(directory) / "onnx-evidence.json"
        subprocess.run(
            [
                sys.executable,
                str(
                    root
                    / "scripts/audit-onnx-blocked-quantization-evidence.py"
                ),
                "--output",
                str(regenerated_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        regenerated = json.loads(regenerated_path.read_text(encoding="utf-8"))
    if regenerated != committed:
        fail("Committed ONNX blocked evidence is not reproducible")
    if committed["fixture_count"] != 2:
        fail("Unexpected ONNX blocked fixture count")
    if committed["numerically_exact_fixture_count"] != 2:
        fail("ONNX blocked fixture numerical check failed")
    if committed["blocked_shape_conservation_pass_count"] != 2:
        fail("ONNX blocked fixture shape conservation failed")
    validator = Draft202012Validator(quantization_definition)
    for row in committed["fixtures"]:
        if list(
            validator.iter_errors(row["cyclonedx_normalized_projection"])
        ):
            fail("Normalized ONNX projection fails pinned CycloneDX schema")
        if row["equivalent_negative_source_axis"] != -1:
            fail("Unexpected ONNX negative source-axis equivalence")
        if row["effective_axis"] != 1 or row["block_size"] != 2:
            fail("Unexpected normalized ONNX blocked contract")


def check_contribution_cases(root):
    path = root / "data/cyclonedx-2.0-quantization-contribution-cases.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for source in document["source_ledgers"].values():
        if sha256_file(root / source["path"]) != source["sha256"]:
            fail(f"Contribution case source digest mismatch: {source['path']}")

    interface_rows = json.loads(
        (root / "data/interface-contracts.json").read_text(encoding="utf-8")
    )["parameters"]
    statuses = Counter(row["quantization"]["status"] for row in interface_rows)
    expected_status = document["external_parameter_status"]
    if len(interface_rows) != expected_status["parameter_count"]:
        fail("Contribution external-parameter count mismatch")
    if statuses != {"complete": 62, "not_quantized": 52}:
        fail("Unexpected external quantization status distribution")
    if expected_status["complete_affine"] != statuses["complete"]:
        fail("Contribution complete-affine count mismatch")
    if expected_status["not_quantized"] != statuses["not_quantized"]:
        fail("Contribution non-quantized count mismatch")
    if expected_status["incomplete_affine"] != 0:
        fail("The contribution fixture misclassifies non-quantized parameters")

    binding = document["parameter_binding_case"]
    source_parameters = {
        row["parameter_id"]: row
        for row in interface_rows
        if row["qualified_id"] == binding["qualified_id"]
    }
    if binding["artifact_sha256"] != next(iter(source_parameters.values()))[
        "artifact_sha256"
    ]:
        fail("Parameter-binding artifact SHA-256 mismatch")
    for parameter in binding["parameters"]:
        source = source_parameters[parameter["parameter_id"]]
        quantization = source["quantization"]
        expected = {
            "name": source["tensor_name"],
            "shape": source["shape"],
            "scale": quantization["scales"][0],
            "zero_point": quantization["zero_points"][0],
            "contract_sha256": quantization["contract_sha256"],
        }
        observed = {key: parameter[key] for key in expected}
        if observed != expected:
            fail(f"Parameter-binding case mismatch: {parameter['parameter_id']}")
    mappings = {
        (
            row["quantization"]["scales"][0],
            row["quantization"]["zero_points"][0],
        )
        for row in source_parameters.values()
    }
    if len(mappings) != 2:
        fail("Parameter-binding case does not contain two distinct mappings")

    boundary = document["non_quantized_boundary_case"]
    boundary_rows = [
        row for row in interface_rows
        if row["qualified_id"] == boundary["qualified_id"]
    ]
    quantized_inputs = sum(
        row["direction"] == "input"
        and row["quantization"]["status"] == "complete"
        for row in boundary_rows
    )
    non_quantized_outputs = sum(
        row["direction"] == "output"
        and row["quantization"]["status"] == "not_quantized"
        for row in boundary_rows
    )
    if quantized_inputs != boundary["quantized_input_count"]:
        fail("Boundary quantized-input count mismatch")
    if non_quantized_outputs != boundary["non_quantized_output_count"]:
        fail("Boundary non-quantized-output count mismatch")

    granularity = json.loads(
        (root / "data/tflite-granularity-evidence.json")
        .read_text(encoding="utf-8")
    )
    mixed = document["mixed_weight_axis_case"]
    source_mixed = next(
        row for row in granularity["artifact_rows"]
        if row["qualified_id"] == mixed["qualified_id"]
    )
    projection = source_mixed["model_level_single_object_projection"]
    if mixed["artifact_sha256"] != source_mixed["artifact_sha256"]:
        fail("Mixed-axis artifact SHA-256 mismatch")
    if mixed["signatures"] != source_mixed["known_weight_contract_counts"]:
        fail("Mixed-axis signature count mismatch")
    if mixed["operator_semantics"] != (
        source_mixed["known_weight_tensor_op_axis_counts"]
    ):
        fail("Mixed-axis operator binding mismatch")
    if mixed["known_weight_tensor_count"] != projection[
        "known_weight_tensor_count"
    ]:
        fail("Mixed-axis weight count mismatch")
    losses = {
        row["selected_contract_signature"]:
        row["omitted_or_misrepresented_weight_tensors"]
        for row in projection["candidate_information_loss"]
    }
    if losses != {"per-channel:axis=0": 17, "per-channel:axis=3": 36}:
        fail("Mixed-axis projection loss mismatch")

    onnx = json.loads(
        (root / "data/onnx-blocked-quantization-evidence.json")
        .read_text(encoding="utf-8")
    )
    per_group = document["per_group_case"]
    if per_group["fixture_count"] != onnx["fixture_count"]:
        fail("Per-group fixture count mismatch")
    if per_group["shape_and_numerical_checks_passed"] != sum(
        row["scale_replication_conservation"] and row["numerical_output_exact"]
        for row in onnx["fixtures"]
    ):
        fail("Per-group exact-check count mismatch")
    if any(
        row["effective_axis"] != per_group["effective_axis"]
        or row["block_size"] != per_group["group_size"]
        or row["equivalent_negative_source_axis"] != -1
        for row in onnx["fixtures"]
    ):
        fail("Per-group normalized contract mismatch")


def main():
    root = Path(__file__).resolve().parents[1]
    definition = check_schema_audit(root)
    check_tflite_evidence(root)
    check_onnx_blocked_evidence(root, definition)
    check_contribution_cases(root)
    print(json.dumps({
        "status": "pass",
        "cyclonedx_schema_case_count": 12,
        "tflite_artifact_count": 50,
        "external_affine_parameter_count": 62,
        "constant_per_axis_tensor_count": 6926,
        "multiple_constant_axis_artifact_count": 18,
        "known_weight_tensor_count": 3672,
        "onnx_blocked_fixture_count": 2,
    }, indent=2))


if __name__ == "__main__":
    main()
