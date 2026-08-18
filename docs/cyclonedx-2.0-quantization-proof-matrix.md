# CycloneDX 2.0 quantization proof matrix

## Decision rule

Four different propositions require four different forms of evidence:

1. **Schema behavior** is proved by validation against a hash-pinned schema.
2. **Artifact behavior** is proved by decoding hash-identified files and
   checking conservation rules with an independent implementation where
   possible.
3. **Framework semantics** is proved by a version-pinned normative
   specification or official conformance fixture.
4. **What CycloneDX ought to require** is a Working Group policy decision. A
   corpus can expose consequences, but cannot manufacture normative intent.

## Coverage matrix

| Question | Evidence unit | Current result | What it proves | Remaining boundary |
| --- | --- | --- | --- | --- |
| What does draft PR #990 accept? | CycloneDX schema commit `49a9456`, schema SHA-256, JSON Schema 2020-12 validator | 12 cases: 8 valid, 4 invalid; all 6 cross-field probes accepted | Exact behavior of that draft | Not WG intent or future head behavior |
| Are basic invalid values rejected? | `bits=0`, `groupSize=0`, unknown predefined scheme | All rejected | These three invalid fixtures can be submitted without changing the draft | Full fixture wrappers still must follow repository conventions |
| Are custom schemes and fractional bits supported? | Custom scheme object and `bits=1.58` | Both accepted | Current extension and numeric domains | Does not validate the semantics of an arbitrary custom scheme |
| Do external TFLite affine contracts occur? | 50 SHA-256-identified artifacts, 114 external parameters | 62 complete affine parameters, all per-tensor; 52 unquantized | Parameter-level affine metadata has observed examples | Curated corpus, not ecosystem prevalence |
| Is the external decoder correct? | Generated FlatBuffer decoder vs LiteRT 2.1.4 | 114/114 parameter rows agree; 0 mismatches | Independent parser agreement for dtype, shape, scale, zero point, and serialized axis | Shared upstream format semantics remain a common dependency |
| Is axis operational for per-channel weights? | Every tensor, constant storage range, operator input binding | 3,443 per-axis INT8 weights; 3,116 axis 0 and 327 axis 3; 0 contract mismatches | Axis is required to bind scale vectors to slices; Conv/FC and Depthwise use different axes as specified | Internal weights are model-level evidence, not external `modelParameter` examples |
| Are bias tensors being confused with weights? | Operator input role | 3,483 per-axis INT32 biases separately classified | Weight-axis claims exclude bias tensors | Complex/custom operators outside known slots need separate role definitions |
| Can one model-level object encode all exact weight contracts? | Per-artifact projection of every known weight `(granularity, axis)` signature | 18 artifacts have no lossless single-object projection; the best candidate still loses 330/1,033 contracts in aggregate; 17 project losslessly; 15 have no known quantized weight binding | A concrete counterexample to using one shared axis as an exhaustive tensor ledger | A deliberately coarse model-level summary remains possible |
| Is negative axis valid anywhere? | ONNX QuantizeLinear normative contract | Valid in `[-rank, rank-1]`; `-1` normalizes to the last axis | Negative axis cannot be called framework-neutral invalid without a normalization rule | CycloneDX may still choose normalized non-negative values if it says so |
| Can parameter-level negative axis be normalized? | Parameter shape and source axis | Yes when rank is known: rank 4, `-1 -> 3` | A deterministic normalization path exists | `modelParameter.shape` is optional, so rank is not always available |
| Can model-level negative axis be normalized? | `modelProperties.quantization` structure | No direct weight-tensor shape exists; heterogeneous tensors may have different ranks and axes | The current model-level object lacks enough context for a general source-axis normalization | A tensor-addressable collection or explicit summary semantics would resolve it |
| Is per-group more than a hypothetical case? | ONNX 1.22.0 official blocked QuantizeLinear fixtures | 2/2 hashes, shape relations, and all 24 output values verify | Source-backed affine per-group example with `axis=1`, `groupSize=2` | Does not generalize to every LLM encoding |
| Does GGUF block width equal CycloneDX `groupSize`? | GGUF type definitions and eight-file local corpus | Not asserted | Avoids a false cross-framework mapping | K-quants may contain superblocks and subblocks; a vocabulary mapping needs encoding-specific rules |
| Does AWQ establish a common group size? | AWQ paper/code and hash-pinned model configs | Evidence acquisition still required for an artifact corpus | The paper establishes grouped weight-only quantization; configs can establish observed values | A commonly used value such as 128 is not a universal schema constraint |
| Should companion fields be mandatory? | Six accepted contradictory/partial combinations plus WG intent | Current behavior is proved; desired semantics are unresolved | Precisely frames the policy question | Must ask whether the object is complete-normalized or partial-best-effort metadata |

## Reproducible evidence files

- [`cyclonedx-2.0-quantization-schema-audit.json`](../data/cyclonedx-2.0-quantization-schema-audit.json)
  pins the draft, extracted definition, 12 base cases, two schema usage sites,
  and five normalization probes.
- [`tflite-granularity-evidence.json`](../data/tflite-granularity-evidence.json)
  binds every result to the artifact and external-contract ledgers, resolves
  inline and file-offset constants, and records per-tensor role and projection
  details.
- [`onnx-blocked-quantization-evidence.json`](../data/onnx-blocked-quantization-evidence.json)
  binds the official ONNX fixtures to exact model hashes and independently
  recalculates their outputs.
- `scripts/check-quantization-fixture-evidence.py` regenerates the ONNX
  evidence and checks all committed conservation rules.

## Remaining evidence acquisition

Two additions would close the framework breadth without weakening the claim:

1. **AWQ/GPTQ corpus.** Select public model repositories at immutable
   revisions, hash the quantization config and weight index, record
   `quant_method`, `bits`, `group_size`, and axis/layout convention, and verify
   the tensor partition cardinality from the stored weight shapes. Report a
   distribution, never a universal default.
2. **GGUF semantic mapping.** Pin a llama.cpp commit and extract each encoding's
   storage block, subblock, scale, and minimum structure from source. Map a
   CycloneDX `groupSize` only when one affine scale/zero-point pair actually
   governs that number of values. Otherwise record a custom granularity object
   rather than flattening a nested K-quant layout.

The AWQ paper is <https://arxiv.org/abs/2306.00978>. The GGUF implementation
source is
<https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.h>.
These sources support an acquisition protocol; they do not replace the
hash-pinned artifact observations proposed above.

## WG questions that evidence cannot answer by itself

The shortest policy request should ask only:

1. Is `$defs.quantization` intended as a complete normalized contract or as
   partial best-effort metadata?
2. If `axis` is normalized, where is source-axis normalization defined when
   parameter shape is absent?
3. Is `modelProperties.quantization` deliberately a coarse summary, or is it
   expected to represent heterogeneous weight-tensor contracts losslessly?

Answers to these questions determine whether `if/then` constraints are
correct, whether negative axes should be accepted, and whether model-level
quantization needs a tensor-addressable collection. Until then, the evidence
supports fixtures and precise problem statements, not an invented policy.
