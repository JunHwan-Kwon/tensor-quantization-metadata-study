# Qwen3-4B GGUF paired-replication protocol

## Purpose

This is a second, outcome-neutral replication of the tensor-assignment
comparison used in the Granite study. The subjects are revision-pinned public
`Q4_K_M` GGUF files associated with the same Qwen3-4B base-model family. A
matching result and a divergent result are both valid observations.

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
