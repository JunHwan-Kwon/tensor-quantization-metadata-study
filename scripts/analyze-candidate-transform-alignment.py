import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


CANDIDATES = {
    "raw_storage": lambda pixel: pixel,
    "unit_interval": lambda pixel: pixel / 255.0,
    "minus_one_to_one_255": lambda pixel: 2.0 * pixel / 255.0 - 1.0,
    "center_128_div_128": lambda pixel: (pixel - 128.0) / 128.0,
    "center_127_5_div_127_5": lambda pixel: (
        (pixel - 127.5) / 127.5
    ),
}


def storage_domain(dtype):
    if dtype == "UINT8":
        return 0, 255
    if dtype == "INT8":
        return -128, 127
    raise ValueError(f"Unsupported pixel storage dtype: {dtype}")


def pixel_storage_codes(dtype, pixels):
    if dtype == "UINT8":
        return pixels.copy()
    if dtype == "INT8":
        return np.where(pixels < 128, pixels, pixels - 256)
    raise ValueError(f"Unsupported pixel storage dtype: {dtype}")


def round_ties_away(values):
    return np.where(
        values >= 0,
        np.floor(values + 0.5),
        np.ceil(values - 0.5),
    )


def is_image_input(row, task):
    if row["direction"] != "input":
        return False
    if row["dtype"] not in {"INT8", "UINT8"}:
        return False
    if row["quantization"]["status"] != "complete":
        return False
    if len(row["shape"]) != 4:
        return False
    channel_like = row["shape"][-1] in {1, 3, 4} or row["shape"][1] in {
        1,
        3,
        4,
    }
    task_image_like = any(
        token in task.replace("-", "_")
        for token in ("image", "object_detection", "visual")
    )
    return channel_like and task_image_like


def evaluate_candidate(dtype, scale, zero_point, transform):
    pixels = np.arange(256, dtype=np.float64)
    storage_codes = pixel_storage_codes(dtype, pixels)
    real_values = transform(pixels)
    continuous_codes = real_values / scale + zero_point
    qmin, qmax = storage_domain(dtype)
    rounded_codes = round_ties_away(continuous_codes)
    clipped_codes = np.clip(rounded_codes, qmin, qmax)
    direct_real = (storage_codes - zero_point) * scale
    code_displacement = np.abs(continuous_codes - storage_codes)
    direct_real_error = np.abs(direct_real - real_values)
    exact_count = int(np.count_nonzero(code_displacement <= 1e-9))
    nearest_count = int(np.count_nonzero(clipped_codes == storage_codes))
    clipping_count = int(np.count_nonzero(
        (continuous_codes < qmin) | (continuous_codes > qmax)
    ))
    if exact_count == 256:
        alignment_class = "exact_lattice_alignment"
    elif nearest_count == 256 and clipping_count == 0:
        alignment_class = "rounded_lattice_alignment"
    else:
        alignment_class = "partial_alignment"
    return {
        "alignment_class": alignment_class,
        "exact_storage_code_count": exact_count,
        "nearest_storage_code_match_count": nearest_count,
        "clipping_source_value_count": clipping_count,
        "clipping_fraction": clipping_count / 256.0,
        "reachable_storage_code_count": len(set(clipped_codes.tolist())),
        "max_abs_continuous_code_displacement": float(
            np.max(code_displacement)
        ),
        "mean_abs_continuous_code_displacement": float(
            np.mean(code_displacement)
        ),
        "max_abs_real_error_if_bytes_passed_directly": float(
            np.max(direct_real_error)
        ),
        "rms_real_error_if_bytes_passed_directly": float(
            math.sqrt(np.mean(np.square(direct_real_error)))
        ),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    artifacts = json.loads(
        (root / "data/artifacts.json").read_text(encoding="utf-8")
    )["artifacts"]
    parameters = json.loads(
        (root / "data/interface-contracts.json").read_text(encoding="utf-8")
    )["parameters"]
    artifact_by_id = {row["qualified_id"]: row for row in artifacts}

    records = []
    for row in parameters:
        artifact = artifact_by_id[row["qualified_id"]]
        if not is_image_input(row, artifact["task"]):
            continue
        quantization = row["quantization"]
        if quantization["scale_count"] != 1:
            raise ValueError("Candidate pixel analysis requires per-tensor scale")
        scale = quantization["scales"][0]
        zero_point = quantization["zero_points"][0]
        candidates = {
            name: evaluate_candidate(
                row["dtype"], scale, zero_point, transform
            )
            for name, transform in CANDIDATES.items()
        }
        ranking = sorted(
            candidates,
            key=lambda name: (
                candidates[name]["rms_real_error_if_bytes_passed_directly"],
                candidates[name]["clipping_source_value_count"],
                name,
            ),
        )
        exact = [
            name for name, value in candidates.items()
            if value["alignment_class"] == "exact_lattice_alignment"
        ]
        records.append({
            "qualified_id": row["qualified_id"],
            "artifact_sha256": row["artifact_sha256"],
            "task": artifact["task"],
            "parameter_id": row["parameter_id"],
            "dtype": row["dtype"],
            "shape": row["shape"],
            "scale": scale,
            "zero_point": zero_point,
            "exact_candidate_count": len(exact),
            "exact_candidates": exact,
            "closest_candidate_by_rms_direct_byte_error": ranking[0],
            "candidate_ranking": ranking,
            "candidates": candidates,
            "interpretation": (
                "candidate_alignment_only_not_preprocessing_evidence"
            ),
        })

    exact_counts = Counter(
        candidate
        for row in records
        for candidate in row["exact_candidates"]
    )
    best_counts = Counter(
        row["closest_candidate_by_rms_direct_byte_error"] for row in records
    )
    no_exact = sum(row["exact_candidate_count"] == 0 for row in records)
    result = {
        "schema_version": 1,
        "method": {
            "source_domain": "integer pixel values 0..255",
            "signed_int8_storage": (
                "values 128..255 are interpreted as two's-complement bytes"
            ),
            "encoding_rounding": "nearest_ties_away_from_zero",
            "candidate_transforms": list(CANDIDATES),
            "claim_boundary": (
                "Alignment does not establish the application's preprocessing "
                "pipeline or compliance with a convention."
            ),
        },
        "summary": {
            "assessed_input_count": len(records),
            "input_count_without_exact_candidate": no_exact,
            "exact_alignment_counts": dict(sorted(exact_counts.items())),
            "closest_candidate_counts": dict(sorted(best_counts.items())),
        },
        "inputs": records,
    }
    output = root / "data/candidate-transform-alignment.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
