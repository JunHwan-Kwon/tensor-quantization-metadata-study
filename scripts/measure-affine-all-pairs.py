import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "measure-affine-interface-mismatch.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "affine_interface_measurement", BASE_SCRIPT
)
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)
WORKER_CONTEXT = None


def fail(message):
    raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tensor-quantization-metadata-study/1"},
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if copied and copied % (64 * 1024 * 1024) < len(chunk):
                print(f"downloaded {copied} bytes", flush=True)
    temporary.replace(destination)


def verify_file(path, expected_size, expected_sha256, label):
    if not path.is_file():
        return False
    if path.stat().st_size != expected_size:
        return False
    actual = sha256_file(path)
    if actual != expected_sha256:
        fail(f"{label} SHA-256 mismatch: {actual} != {expected_sha256}")
    return True


def copy_verified(source, destination, artifact):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    shutil.copyfile(source, temporary)
    if not verify_file(
        temporary,
        artifact["size_bytes"],
        artifact["sha256"],
        artifact["qualified_id"],
    ):
        temporary.unlink(missing_ok=True)
        fail(f"Unable to verify copied artifact {artifact['qualified_id']}")
    temporary.replace(destination)


def extract_member(archive, member_name, destination):
    normalized = member_name.removeprefix("./")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            matches = [
                name for name in bundle.namelist()
                if name.removeprefix("./") == normalized
            ]
            if len(matches) != 1:
                fail(f"Archive member match count is {len(matches)}: {member_name}")
            with bundle.open(matches[0]) as source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out)
    else:
        with tarfile.open(archive, "r:*") as bundle:
            matches = [
                member for member in bundle.getmembers()
                if member.name.removeprefix("./") == normalized
                and member.isfile()
            ]
            if len(matches) != 1:
                fail(f"Archive member match count is {len(matches)}: {member_name}")
            source = bundle.extractfile(matches[0])
            if source is None:
                fail(f"Unable to read archive member: {member_name}")
            with source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out)
    temporary.replace(destination)


def resolve_artifact(artifact, cache_root, search_roots):
    destination = cache_root / "models" / f"{artifact['sha256']}.tflite"
    if verify_file(
        destination,
        artifact["size_bytes"],
        artifact["sha256"],
        artifact["qualified_id"],
    ):
        return destination

    filename = artifact["download"]["filename"]
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for candidate in search_root.rglob(filename):
            if candidate.stat().st_size != artifact["size_bytes"]:
                continue
            if sha256_file(candidate) == artifact["sha256"]:
                copy_verified(candidate, destination, artifact)
                return destination

    download = artifact["download"]
    if download["kind"] == "direct":
        download_file(download["url"], destination)
    elif download["kind"] == "tar_member":
        archive = cache_root / "downloads" / download["archive_filename"]
        if not verify_file(
            archive,
            download["archive_size_bytes"],
            download["archive_sha256"],
            download["archive_filename"],
        ):
            download_file(download["url"], archive)
            if not verify_file(
                archive,
                download["archive_size_bytes"],
                download["archive_sha256"],
                download["archive_filename"],
            ):
                fail(f"Unable to verify archive {download['archive_filename']}")
        extract_member(archive, download["member_path"], destination)
    else:
        fail(f"Unsupported download kind: {download['kind']}")

    if not verify_file(
        destination,
        artifact["size_bytes"],
        artifact["sha256"],
        artifact["qualified_id"],
    ):
        destination.unlink(missing_ok=True)
        fail(f"Unable to verify downloaded artifact {artifact['qualified_id']}")
    return destination


def contract_key(contract):
    return (
        contract["dtype"],
        float(contract["scale"]),
        int(contract["zero_point"]),
        int(contract["qmin"]),
        int(contract["qmax"]),
    )


def contract_id(contract):
    fact = {
        "dtype": contract["dtype"],
        "scale": contract["scale"],
        "zero_point": contract["zero_point"],
        "qmin": contract["qmin"],
        "qmax": contract["qmax"],
    }
    return f"affine-{sha256_json(fact)[:12]}"


def open_model(config, filename, num_threads=1):
    if sha256_file(filename) != config["artifact_sha256"]:
        fail(f"Artifact SHA-256 mismatch for {config['qualified_id']}")
    interpreter = Interpreter(model_path=str(filename), num_threads=num_threads)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        fail(f"{config['qualified_id']} must have one input and one output")
    input_detail = inputs[0]
    output_detail = outputs[0]
    if input_detail["shape"].tolist() != config["expected_input_shape"]:
        fail(f"Unexpected input shape for {config['qualified_id']}")
    if output_detail["shape"].tolist() != config["expected_output_shape"]:
        fail(f"Unexpected output shape for {config['qualified_id']}")
    input_contract = BASE.scalar_affine(input_detail)
    output_contract = BASE.scalar_affine(output_detail)
    if contract_key(input_contract) != contract_key(config["input_contract"]):
        fail(f"Input contract differs from public ledger: {config['qualified_id']}")
    return {
        **config,
        "filename": filename,
        "interpreter": interpreter,
        "input_detail": input_detail,
        "output_detail": output_detail,
        "input_contract": input_contract,
        "output_contract": output_contract,
    }


def output_observation(codes, descriptor):
    real = BASE.dequantize(codes, descriptor["output_contract"])
    probabilities = BASE.output_probabilities(
        real, descriptor["output_interpretation"]
    )
    raw_prediction = int(np.argmax(codes))
    prediction = raw_prediction - descriptor["output_label_offset"]
    top5 = set(
        (BASE.top_k(codes, 5) - descriptor["output_label_offset"]).tolist()
    )
    return {
        "prediction": prediction,
        "top5": top5,
        "confidence": float(probabilities[raw_prediction]),
        "probabilities": probabilities,
    }


def measure_image(descriptor, source_contracts, image_row):
    input_detail = descriptor["input_detail"]
    output_detail = descriptor["output_detail"]
    target_contract = descriptor["input_contract"]
    correct_input = BASE.resize_center_crop_rgb(
        image_row["path"],
        resize_short=descriptor["resize_short"],
        crop_size=descriptor["crop_size"],
    )
    identity, identity_low, identity_high = BASE.reencode_with_contract(
        correct_input, target_contract, target_contract
    )
    if identity_low or identity_high or not np.array_equal(identity, correct_input):
        fail(f"Identity re-encoding failed for {descriptor['qualified_id']}")
    correct_codes = BASE.invoke(
        descriptor["interpreter"],
        int(input_detail["index"]),
        int(output_detail["index"]),
        correct_input,
    )
    correct = output_observation(correct_codes, descriptor)
    label = image_row["label"]
    rows = []
    for source in source_contracts:
        wrong_input, low_clip, high_clip = BASE.reencode_with_contract(
            correct_input, target_contract, source["contract"]
        )
        wrong_codes = BASE.invoke(
            descriptor["interpreter"],
            int(input_detail["index"]),
            int(output_detail["index"]),
            wrong_input,
        )
        wrong = output_observation(wrong_codes, descriptor)
        rows.append({
            "target_model_id": descriptor["qualified_id"],
            "source_contract_id": source["contract_id"],
            "relative_path": image_row["relative_path"],
            "wnid": image_row["wnid"],
            "label": label,
            "correct_prediction": correct["prediction"],
            "wrong_prediction": wrong["prediction"],
            "correct_top1_correct": correct["prediction"] == label,
            "wrong_top1_correct": wrong["prediction"] == label,
            "correct_top5_correct": label in correct["top5"],
            "wrong_top5_correct": label in wrong["top5"],
            "correct_confidence": correct["confidence"],
            "wrong_confidence": wrong["confidence"],
            "js_divergence": BASE.js_divergence(
                correct["probabilities"], wrong["probabilities"]
            ),
            "identity_reencode_exact": True,
            "input_element_count": int(correct_input.size),
            "wrong_input_low_clip_count": low_clip,
            "wrong_input_high_clip_count": high_clip,
        })
    return rows


def worker_initializer(config, filename, source_contracts):
    global WORKER_CONTEXT
    WORKER_CONTEXT = (
        open_model(config, Path(filename), 1),
        source_contracts,
    )


def prediction_worker(image_row):
    if WORKER_CONTEXT is None:
        fail("Prediction worker is not initialized")
    return measure_image(WORKER_CONTEXT[0], WORKER_CONTEXT[1], image_row)


def run_target(descriptor, source_contracts, images, workers):
    rows = []
    started = time.perf_counter()
    if workers == 1:
        iterator = (
            measure_image(descriptor, source_contracts, image_row)
            for image_row in images
        )
        executor = None
    else:
        config = {
            key: value for key, value in descriptor.items()
            if key not in {
                "filename", "interpreter", "input_detail", "output_detail",
                "output_contract",
            }
        }
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_initializer,
            initargs=(config, str(descriptor["filename"]), source_contracts),
        )
        iterator = executor.map(prediction_worker, images, chunksize=4)
    try:
        for ordinal, image_rows in enumerate(iterator, start=1):
            rows.extend(image_rows)
            if ordinal % 100 == 0 or ordinal == len(images):
                print(
                    f"{descriptor['qualified_id']}: {ordinal}/{len(images)} images",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows, time.perf_counter() - started


def comparison_metrics(rows, bootstrap_iterations, seed):
    metrics = BASE.paired_metrics(rows, bootstrap_iterations, seed)
    metrics.pop("identity_reencode_mismatch_count", None)
    return metrics


def holm_adjust(comparisons):
    ordered = sorted(
        comparisons,
        key=lambda row: (
            row["metrics"]["mcnemar_exact"]["two_sided_p_value"],
            row["target_model_id"],
            row["source_contract_id"],
        ),
    )
    family_size = len(ordered)
    previous = 0.0
    for ordinal, comparison in enumerate(ordered):
        raw = comparison["metrics"]["mcnemar_exact"]["two_sided_p_value"]
        adjusted = min(1.0, max(previous, (family_size - ordinal) * raw))
        comparison["mcnemar_holm"] = {
            "family_size": family_size,
            "rank": ordinal + 1,
            "adjusted_p_value": adjusted,
            "reject_at_0_05": adjusted < 0.05,
        }
        previous = adjusted


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{100.0 * value:.2f}%"


def write_summary(path, result):
    lines = [
        "# Distinct affine-contract all-pairs experiment",
        "",
        (
            f"{result['dataset']['image_count']} ImageNetV2 images, "
            f"{len(result['models'])} target models, "
            f"{len(result['contract_catalog'])} distinct input contracts, and "
            f"{len(result['comparisons'])} non-identity comparisons."
        ),
        "",
        "| Target | Source contract aliases | Baseline top-1 | Substituted top-1 | Delta | Input clip | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in sorted(
        result["comparisons"],
        key=lambda row: (row["target_model_id"], row["source_contract_id"]),
    ):
        metrics = comparison["metrics"]
        aliases = ", ".join(comparison["source_model_aliases"])
        lines.append(
            f"| {comparison['target_model_id']} | {aliases} | "
            f"{percent(metrics['correct_top1_accuracy'])} | "
            f"{percent(metrics['wrong_top1_accuracy'])} | "
            f"{percent(metrics['paired_top1_accuracy_delta']['point'])} | "
            f"{percent(metrics['wrong_input_clip_fraction'])} | "
            f"{comparison['mcnemar_holm']['adjusted_p_value']:.4g} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        (
            "Model aliases sharing exactly the same dtype, scale, zero-point, "
            "and integer range are collapsed into one source contract. Identity "
            "substitutions are not counted as comparisons."
        ),
        "",
        (
            "Each comparison changes only the affine encoder used for a fixed "
            "target artifact and target-specific image crop. Results establish "
            "effects for these hash-pinned files and selected images; they do "
            "not estimate mismatch prevalence or a universal accuracy loss."
        ),
        "",
        f"Result ledger SHA-256: `{result['ledger_sha256']}`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run all non-identity pairs across distinct affine input contracts."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", default=str(ROOT / "cache" / "all-pairs"))
    parser.add_argument("--dataset-cache-root", default=str(ROOT / "cache"))
    parser.add_argument("--artifact-search-root", action="append", default=[])
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        fail("--workers must be at least 1")
    if args.max_images < 1:
        fail("--max-images must be at least 1")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).resolve()
    search_roots = [Path(value).resolve() for value in args.artifact_search_root]

    artifact_ledger = json.loads(
        (ROOT / "data" / "artifacts.json").read_text(encoding="utf-8")
    )
    artifacts = {
        row["qualified_id"]: row for row in artifact_ledger["artifacts"]
    }
    interface_ledger = json.loads(
        (ROOT / "data" / "interface-contracts.json").read_text(encoding="utf-8")
    )
    inputs = {
        row["qualified_id"]: row
        for row in interface_ledger["parameters"]
        if row["direction"] == "input" and row["ordinal"] == 0
    }
    model_manifest = json.loads(
        (ROOT / "data" / "affine-all-pairs-models.json").read_text(
            encoding="utf-8"
        )
    )

    candidate_configs = []
    for candidate in model_manifest["models"]:
        qualified_id = candidate["qualified_id"]
        artifact = artifacts[qualified_id]
        interface = inputs[qualified_id]
        quantization = interface["quantization"]
        if quantization["status"] != "complete" or quantization["granularity"] != "per_tensor":
            fail(f"Incomplete scalar input contract: {qualified_id}")
        dtype = interface["dtype"]
        if dtype != "UINT8":
            fail(f"All-pairs protocol requires UINT8 input: {qualified_id}")
        contract = {
            "dtype": dtype,
            "scale": quantization["scales"][0],
            "zero_point": quantization["zero_points"][0],
            "qmin": 0,
            "qmax": 255,
            "quantized_dimension": quantization["quantized_dimension"],
        }
        candidate_configs.append({
            **candidate,
            "artifact_sha256": artifact["sha256"],
            "artifact_size_bytes": artifact["size_bytes"],
            "input_contract": contract,
            "input_contract_id": contract_id(contract),
        })

    catalog_by_key = {}
    for config in candidate_configs:
        key = contract_key(config["input_contract"])
        row = catalog_by_key.setdefault(key, {
            "contract_id": config["input_contract_id"],
            "contract": config["input_contract"],
            "model_aliases": [],
        })
        row["model_aliases"].append(config["qualified_id"])
    contract_catalog = sorted(
        catalog_by_key.values(), key=lambda row: row["contract_id"]
    )
    for row in contract_catalog:
        row["model_aliases"].sort()
    if len(contract_catalog) < 2:
        fail("At least two distinct contracts are required")

    model_configs = []
    for config in candidate_configs:
        if not config.get("execute_as_target", False):
            continue
        artifact = artifacts[config["qualified_id"]]
        model_configs.append({
            **config,
            "filename": resolve_artifact(
                artifact, cache_root, search_roots
            ),
        })
    if len(model_configs) != len(contract_catalog):
        fail(
            "The execution set must contain one target representative per "
            "distinct contract"
        )

    dataset = BASE.prepare_dataset(
        Path(args.dataset_cache_root).resolve(), "imagenetv2"
    )
    images = BASE.list_validation_images(
        dataset["root"],
        dataset["validation"],
        None,
        args.max_images,
        args.seed,
    )
    if not images:
        fail("No ImageNetV2 images selected")

    all_rows = []
    comparisons = []
    models = []
    for target_ordinal, config in enumerate(model_configs):
        descriptor = open_model(config, config["filename"], 1)
        sources = [
            row for row in contract_catalog
            if row["contract_id"] != descriptor["input_contract_id"]
        ]
        print(
            f"measuring {descriptor['qualified_id']} against {len(sources)} contracts",
            flush=True,
        )
        rows, elapsed = run_target(descriptor, sources, images, args.workers)
        all_rows.extend(rows)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["source_contract_id"]].append(row)
        for source_ordinal, source in enumerate(sources):
            members = grouped[source["contract_id"]]
            metrics = comparison_metrics(
                members,
                args.bootstrap_iterations,
                args.seed + target_ordinal * 100 + source_ordinal * 2,
            )
            comparisons.append({
                "target_model_id": descriptor["qualified_id"],
                "target_contract_id": descriptor["input_contract_id"],
                "source_contract_id": source["contract_id"],
                "source_model_aliases": source["model_aliases"],
                "metrics": metrics,
            })
        models.append({
            "qualified_id": descriptor["qualified_id"],
            "artifact_sha256": descriptor["artifact_sha256"],
            "artifact_size_bytes": descriptor["artifact_size_bytes"],
            "input_shape": descriptor["expected_input_shape"],
            "output_shape": descriptor["expected_output_shape"],
            "resize_short": descriptor["resize_short"],
            "crop_size": descriptor["crop_size"],
            "input_contract_id": descriptor["input_contract_id"],
            "output_label_offset": descriptor["output_label_offset"],
            "output_interpretation": descriptor["output_interpretation"],
            "elapsed_seconds": elapsed,
        })
    holm_adjust(comparisons)

    predictions_path = output_dir / "predictions.csv"
    write_csv(predictions_path, all_rows)
    image_ledger = [{
        "relative_path": row["relative_path"],
        "wnid": row["wnid"],
        "label": row["label"],
    } for row in images]
    result = {
        "schema": "tensor_quantization_metadata_study.affine_all_pairs.v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "script": Path(__file__).name,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "base_script": BASE_SCRIPT.name,
            "base_script_sha256": sha256_file(BASE_SCRIPT),
            "model_manifest": "data/affine-all-pairs-models.json",
            "model_manifest_sha256": sha256_file(
                ROOT / "data" / "affine-all-pairs-models.json"
            ),
            "artifact_ledger": "data/artifacts.json",
            "artifact_ledger_sha256": sha256_file(
                ROOT / "data" / "artifacts.json"
            ),
            "interface_ledger": "data/interface-contracts.json",
            "interface_ledger_sha256": sha256_file(
                ROOT / "data" / "interface-contracts.json"
            ),
            "python": os.sys.version,
            "ai_edge_litert": version("ai-edge-litert"),
            "numpy": version("numpy"),
            "pillow": version("Pillow"),
            "scipy": version("scipy"),
            "prediction_workers": args.workers,
            "interpreter_threads_per_worker": 1,
        },
        "dataset": {
            "id": "imagenetv2",
            "name": dataset["name"],
            "upstream_url": dataset["upstream_url"],
            "source_revision": dataset["revision"],
            "archive_sha256": dataset["archive_sha256"],
            "split": "matched-frequency",
            "image_count": len(images),
            "class_count": len({row["wnid"] for row in images}),
            "selection_seed": args.seed,
            "image_ledger_sha256": sha256_json(image_ledger),
        },
        "protocol": {
            "paired": True,
            "unit_of_substitution": "distinct affine input contract",
            "duplicate_model_contracts_collapsed": True,
            "identity_pairs_excluded": True,
            "changed_variable": "input affine encoder scale and zero-point",
            "rounding": "IEEE nearest-even via numpy.rint",
            "integer_saturation": True,
            "multiple_testing": "Holm adjustment across all non-identity exact McNemar tests",
            "target_specific_spatial_preprocessing": True,
            "target_selection": (
                "one hash-pinned target representative per distinct input "
                "contract; additional files with an identical contract are "
                "retained as source aliases"
            ),
        },
        "contract_catalog": contract_catalog,
        "models": models,
        "comparisons": comparisons,
        "outputs": {
            "predictions_csv": predictions_path.name,
            "predictions_csv_sha256": sha256_file(predictions_path),
        },
        "interpretation_boundary": [
            "Only hash-pinned UINT8 ImageNet classifiers with a bound raw-RGB application interface are included.",
            "The experiment estimates effects for these artifacts and selected images; it does not estimate mismatch prevalence.",
            "A closest candidate pixel transform is not treated as evidence of an application preprocessing contract.",
            "Model aliases sharing an identical affine contract are collapsed before hypothesis testing.",
        ],
    }
    unsigned = dict(result)
    result["ledger_sha256"] = sha256_json(unsigned)
    measurement_path = output_dir / "measurement.json"
    measurement_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.md", result)
    print(json.dumps({
        "measurement": str(measurement_path),
        "model_count": len(models),
        "distinct_contract_count": len(contract_catalog),
        "comparison_count": len(comparisons),
        "ledger_sha256": result["ledger_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
