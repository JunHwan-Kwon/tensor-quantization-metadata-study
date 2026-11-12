# Observed quantization assignments and interface contracts

This study compares serialized tensor encodings and input/output contracts in
hash-identified GGUF and TFLite artifacts. It records which tensor each observation
describes, with complete inventories and reproducible checks.

The two GGUF pairs have different outcomes: Granite has different assignments;
Qwen has matching assignments. Neither pair estimates ecosystem prevalence.

## Evidence at a glance

| Case | Observed result | Boundary |
| --- | --- | --- |
| Granite 4.2 3B, two Q4_K_M publications | 100 of 363 tensor assignments differ | Stored encodings, not decoded weight equality or quality |
| Qwen3-4B, two Q4_K_M publications | 0 of 398 assignments differ | Agreement for this pair, not a universal label guarantee |
| EfficientNet-Lite0 INT8 publication | Named input and output are UINT8 with different affine contracts | External I/O does not describe all internal weights |
| YAMNet audio-classifier publication | FLOAT32 I/O with internal INT8 weights and intermediate tensors | Serialized contracts, not observed integer kernel execution |

## 1. Granite: the same label with different assignments

Both files have the same tensor-name set and name-matched stored shapes.
A tensor is identified here by its artifact SHA-256 and exact serialized name.
The `blk.<index>` grouping is a naming observation, not a recovered training-module hierarchy.

| Encoding | IBM publication | Bartowski publication |
| --- | ---: | ---: |
| Q4_K | 241 | 141 |
| Q5_K | 0 | 80 |
| Q6_K | 41 | 61 |
| F32 | 81 | 81 |
| Total | 363 | 363 |

| Changed tensor-name suffix | Count | IBM → Bartowski |
| --- | ---: | --- |
| `attn_k.weight` | 40 | Q4_K → Q5_K |
| `attn_output.weight` | 40 | Q4_K → Q5_K |
| `attn_q.weight` | 20 | Q4_K → Q6_K |

The changed query-weight block indices are `0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39`.
Unchanged encodings do not establish unchanged numerical weights.

| Artifact | Pinned revision | Whole-file SHA-256 |
| --- | --- | --- |
| IBM | `c40945d71cd90f249a56985e8155551a9188dc30` | `e0406663965846ae22a403456eb826ccce5f450840491f71952f18a7cb78e7d5` |
| Bartowski | `4093456941a783ab5d8268e00a7725532c1e9af3` | `d8c0c39e0b775ff4e7ed1ed7f0593d27158acb063b07b132fa4f134b2075567f` |

Acquisition URLs and filenames are in the [artifact manifest](../experiments/gguf-imatrix/artifacts.json).
The [complete paired table](../experiments/gguf-imatrix/tensor-comparison.md) includes every tensor;
the original JSON inventories remain the machine-readable evidence.

### Storage and metadata

| File region | IBM bytes | Bartowski bytes | Difference |
| --- | ---: | ---: | ---: |
| Fixed header | 24 | 24 | +0 |
| KV entries | 3,549,611 | 3,550,500 | +889 |
| Tensor descriptors | 21,590 | 21,590 | +0 |
| Padding before tensor data | 7 | 14 | +7 |
| Tensor payload | 2,240,440,320 | 2,313,553,920 | +73,113,600 |
| Padding between tensors | 0 | 0 | +0 |
| Trailing bytes | 0 | 0 | +0 |
| Whole file | 2,244,011,552 | 2,317,126,048 | +73,114,496 |

The payload difference is 73,113,600 bytes (3.2634% of IBM's payload).
The remaining 896 bytes comprise 889 bytes of KV entries and 7 bytes of alignment padding.
These storage differences do not measure inference RAM, latency, energy, or operating cost.

The 35 common equal KV entries have matching types and value digests. Bartowski has 10 additional entries,
including 4 `quantize.imatrix.*` entries. Their presence records a declaration; it does not prove
which production command ran or which input caused the assignment differences. Their absence
in IBM's file is not evidence that an importance matrix was not used.

[Layout and metadata identities](../experiments/gguf-imatrix/results/file-layout.json) record every key,
its serialized span, type and canonical-value digest. Large tokenizer values and producer file
paths are omitted from this record; their original-value digests and lengths are retained.
The [source comparison](../experiments/gguf-imatrix/results/metadata-cross-source-observations.json)
also keeps embedded lineage metadata separate from repository-card declarations.

## 2. Qwen: a concordant paired result

The Qwen3-4B pair has matching tensor names, stored shapes, per-tensor encodings and normalized
name-to-encoding assignment hashes. This supports agreement for those identified artifacts.
The Granite result establishes that a shared label is insufficient to guarantee agreement in general;
the Qwen result prevents interpreting that observation as a claim that equal labels always conceal differences.

See the [Qwen protocol, identities and results](../experiments/gguf-paired-replication/qwen3-4b).

## 3. Named I/O and internal tensors

| Artifact | Direction | Tensor name | Dtype | Shape | Scale | Zero point |
| --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-Lite0 | input | `images` | UINT8 | `[1, 224, 224, 3]` | 0.012566016986966133 | 131 |
| EfficientNet-Lite0 | output | `Softmax` | UINT8 | `[1, 1000]` | 0.00390625 | 0 |
| YAMNet classifier | input | `waveform_binary` | FLOAT32 | `[15600]` | Not serialized | Not serialized |
| YAMNet classifier | output | `tower0/network/layer32/final_output` | FLOAT32 | `[1, 521]` | Not serialized | Not serialized |

EfficientNet artifact SHA-256: `bc2ffe19c1118de0c0c2a9088992da5589722656e0fba81421385300a4a34b16`.
YAMNet artifact SHA-256: `4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf`.

| Artifact | INT8 weights | INT8 intermediate tensors | Per-axis weights | Per-axis biases |
| --- | ---: | ---: | ---: | ---: |
| EfficientNet-Lite0 | 50 | 63 | 49 | 49 |
| YAMNet classifier | 28 | 33 | 27 | 27 |

Weights are bound to serialized Conv2D, DepthwiseConv2D and FullyConnected input slots,
rather than classified from their names alone. Counts distinguish a tensor from its potentially
multiple consumers. The 98 per-axis tensors in EfficientNet comprise 49 weights and 49 biases.

The [interface ledger](../data/interface-contracts.json) and
[internal tensor contracts](../data/tflite-granularity-evidence.json) contain the source identities
and bindings. Dtype and shape alone do not identify an affine contract; scale and zero point
do not establish RGB order, resizing, or the complete application preprocessing pipeline.

## 4. Related importance-matrix evidence

The separately identified importance-matrix artifact contains 560 F32 tensors
and 942,360 decoded values. The recorded complete scan found
0 non-finite values, 0 exact-zero values and
280 constant-valued tensors. This describes the matrix file; it does not
demonstrate that a specific conversion consumed it or that it improved quality.
See the [matrix measurement](../experiments/gguf-imatrix/results/imatrix-measurement.json).

## 5. Relevance to model-part descriptions

| Evidence task | Available here | Remaining boundary |
| --- | --- | --- |
| Real model examples | Multiple GGUF publications and named TFLite interfaces | Curated examples, not architecture-wide coverage |
| Quantization attached to an identifiable part | Artifact-bound tensor names; selected serialized weight-consumer bindings | Stored tensors are not automatically original layers or modules |
| Reusable tagging test inputs | Complete inventories, source identities and contrary paired outcomes | A standardized internal-subject tagging round trip has not been performed |
| Relating I/O and weight descriptions | Separate external contracts and internal weight inventories | No vocabulary precedence or new component type is selected |
| Deployment observations | Serialized storage and graph contracts | No device latency, kernel placement or CUDA-level observation in these cases |

These cases are ready as test inputs for a model-part representation. They do not establish that
a layer/module representation, an agreed taxonomy, or consumer interoperability is complete.
Mapping to original modules requires explicit source correspondence in addition to tensor names.

Existing [CycloneDX probes](../conformance/cyclonedx-2.0/) and
[vocabulary notes](quantization-vocabulary-mapping.md) are pinned to their stated draft commits.
They are historical tests, not claims about an unpinned current or future release. JSON Schema
validity, the vocabulary's meaning, and fidelity to an artifact are separate checks. Custom
descriptions and external evidence references should not be mistaken for a standardized tensor map.

## 6. Verification and reproduction

Check this brief and the complete paired table using the committed records, without model downloads
or third-party Python packages:

```bash
python scripts/build-quantization-evidence-brief.py --check
```

This recomputes all selected counts, transitions, payload formulae, metadata-span sums and source
hashes. It verifies the committed observations; it does not reread model bytes. The
[derived record](../experiments/gguf-imatrix/results/evidence-brief.json) names every source and its SHA-256.

For original-byte measurements, use the [GGUF protocol](../experiments/gguf-imatrix/protocol.md),
the [paired Qwen protocol](../experiments/gguf-paired-replication/qwen3-4b/protocol.md), and the
[TFLite commands](../README.md#interface-results). Model files remain in the ignored cache.

To run the repository's full suite, including earlier experiments, install the main requirements
and run `python scripts/verify-bundle.py`; this has a larger dependency set than the brief check.

The GGUF primary reader is upstream `gguf-py`. The author-maintained implementation is disclosed
as a secondary cross-check in its separate directory. Neither the brief nor its offline verifier
requires that implementation. The cases retain their separate acquisition and measurement
protocols; presenting them together is not an independent replication.
