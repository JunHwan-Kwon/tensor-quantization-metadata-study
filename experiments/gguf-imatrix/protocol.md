# GGUF and importance-matrix measurement protocol

## Scope

This experiment measures three revision-pinned public artifacts: two GGUF
files carrying the same `Q4_K_M` release label and one GGUF importance-matrix
file. The two quantized files are derived from the same named IBM Granite 4.2
3B model family but are published by different repositories.

The primary measurement uses the upstream `gguf-py` reader distributed as the
Python `gguf` package. It enumerates serialized tensor descriptors and, for the
importance matrix, decodes every F32 value. Runtime package versions are read
from the executing environment and written into each result.

## Artifact identity

`artifacts.json` records the immutable Hugging Face revision, filename, byte
length, and SHA-256 digest for each artifact. Artifact bytes are deliberately
not committed. A run stops before parsing if a supplied file differs in size
or SHA-256.

## Tensor projections

Two hashes have distinct purposes and names:

- `serializedDescriptorProjectionSha256` hashes compact, key-sorted UTF-8 JSON
  over rows in serialized tensor-index order. Each row contains `index`,
  `name`, `dtype`, `shape`, and `byte_length`.
- `nameEncodingAssignmentSha256` hashes the concatenation of
  `UTF8(name) || 0x00 || UTF8(encoding) || 0x0a`, with rows sorted by the raw
  UTF-8 bytes of the tensor name. Names containing NUL and encodings containing
  line feed are rejected.

No Unicode normalization, layer-name ontology, or cross-format tensor-name
equivalence is asserted. The `blk.<integer>.` grouping is a measured naming
pattern only.

## Importance-matrix scan

Every tensor in the matrix subject must be F32 for the recorded complete scan.
For each tensor the script records value count, non-finite count, exact-zero
count, minimum, maximum, and whether all values are identical. Aggregate counts
are recomputed from those per-tensor rows.

## Evidence boundary

The measurements establish byte identity, serialized metadata, tensor names,
shapes, encodings, lengths, and decoded matrix-value properties. They do not by
themselves establish the source checkpoint, conversion command, calibration
corpus provenance, whether a matrix was consumed by a particular conversion,
publisher authenticity, accuracy, or causal performance effects.

The `crosschecks/deepbom` directory contains raw output from a second
implementation maintained by the study author. It is used only to compare
normalized observations. No primary result depends solely on that output.

## Reproduction

Install the pinned reader, download the three hash-identified subjects into the
ignored cache, and run the measurement:

```bash
python -m pip install -r requirements-gguf.txt
python scripts/acquire-gguf-imatrix-artifacts.py
python scripts/measure-gguf-imatrix.py \
  --ibm-q4km cache/gguf-imatrix/e0406663965846ae22a403456eb826ccce5f450840491f71952f18a7cb78e7d5/granite-4.2-3b-Q4_K_M.gguf \
  --bartowski-q4km cache/gguf-imatrix/d8c0c39e0b775ff4e7ed1ed7f0593d27158acb063b07b132fa4f134b2075567f/granite-4.2-3b-Q4_K_M.gguf \
  --imatrix cache/gguf-imatrix/d4c39ccf163db6adbd620619460428c1c447631ac3dfc32e7ce388fe339f37d2/granite-4.2-3b-imatrix.gguf
python scripts/build-gguf-crosscheck.py
python scripts/build-gguf-standards-observations.py
python scripts/check-gguf-imatrix-results.py
```

## Byte-layout and metadata-identity audit

The additional layout audit rechecks both whole-file SHA-256 digests before
parsing. An independent bounded byte cursor measures GGUF v3 little-endian
header, KV-entry and tensor-descriptor spans. Upstream `gguf-py` provides the
metadata values and tensor offsets, which must match the existing inventory.
Overlapping tensor intervals and files outside this audit's supported scope
are rejected rather than reported as successful measurements.

```bash
python scripts/measure-gguf-layout.py \
  --ibm-q4km cache/gguf-imatrix/e0406663965846ae22a403456eb826ccce5f450840491f71952f18a7cb78e7d5/granite-4.2-3b-Q4_K_M.gguf \
  --bartowski-q4km cache/gguf-imatrix/d8c0c39e0b775ff4e7ed1ed7f0593d27158acb063b07b132fa4f134b2075567f/granite-4.2-3b-Q4_K_M.gguf
python scripts/build-quantization-evidence-brief.py --write
python scripts/build-quantization-evidence-brief.py --check
python scripts/test-quantization-evidence-brief.py
```

`results/file-layout.json` records all metadata keys, types, serialized spans,
and SHA-256 digests of compact key-sorted UTF-8 JSON values. Large values are
identified by their digests; the tokenizer vocabulary and merges are not copied
into this result. Values of at most 256 encoded bytes are also shown inline.
These value digests have a different scope from the whole-artifact digest.

The payload formulae used by the brief check are F32: 4 bytes per element,
Q4_K: 144 bytes per 256 elements, Q5_K: 176 per 256, and Q6_K: 210 per 256.
They correspond to the block definitions in
[`ggml-common.h` at b10603](https://github.com/ggml-org/llama.cpp/blob/c060ca974c773c7c3d17fd1b66dc9d312bc292c0/ggml/src/ggml-common.h).
This byte accounting does not decode all quantized weights or establish an
original training-module mapping. The layout audit is not a general GGUF
validator and does not establish the history of model conversion.
