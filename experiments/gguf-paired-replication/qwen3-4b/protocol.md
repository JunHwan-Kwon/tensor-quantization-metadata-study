# Qwen3-4B GGUF paired-replication protocol

## Purpose

This outcome-neutral second pair was selected before inspecting its tensor
assignments to test whether a shared release label is sufficient to identify
the serialized assignment. The subjects are revision-pinned public `Q4_K_M`
GGUF files associated with the same Qwen3-4B base-model family. Concordant and
divergent results were both valid outcomes.

The primary measurement uses upstream `gguf-py` 0.19.0. A second implementation
maintained by the study author is isolated under `crosschecks/deepbom`; no
conclusion depends solely on that implementation.

## Acquisition and measurement

Artifact bytes are downloaded into the ignored cache and verified against both
the byte length and SHA-256 in `artifacts.json`.

```text
python -m pip install -r requirements-gguf.txt
python scripts/measure-gguf-paired-replication.py \
  --experiment experiments/gguf-paired-replication/qwen3-4b \
  --left cache/gguf-paired-replication/qwen3-4b/official-Qwen3-4B-Q4_K_M.gguf \
  --right cache/gguf-paired-replication/qwen3-4b/bartowski-Qwen3-4B-Q4_K_M.gguf
python scripts/check-gguf-paired-replication.py \
  --experiment experiments/gguf-paired-replication/qwen3-4b
```

The comparison covers serialized tensor names, shapes, encodings, and byte
lengths. It does not infer conversion commands, calibration inputs, numerical
equivalence, runtime behavior, quality, or performance.

GGUF files must never be placed under `experiments/` or committed.
