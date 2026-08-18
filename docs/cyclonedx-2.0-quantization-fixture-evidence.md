# CycloneDX 2.0 quantization fixture evidence

## Scope

This note separates three questions that must not be collapsed:

1. What the pinned CycloneDX 2.0 draft currently accepts.
2. What the hash-identified TFLite benchmark actually serializes.
3. Which cross-field constraints require a Working Group policy decision.

The schema audit is pinned to CycloneDX/specification commit
`49a945618811213e55686a23fa63b287940071c6`, the head of specification PR
#990 when this evidence was prepared. The source path, complete schema
SHA-256, extracted quantization definition, validator version, test instances,
and results are recorded in
[`data/cyclonedx-2.0-quantization-schema-audit.json`](../data/cyclonedx-2.0-quantization-schema-audit.json).

## Reproduced draft behavior

The pinned `$defs.quantization` accepts all six cross-field cases below:

- an empty object;
- `per-tensor` with `axis`;
- `per-tensor` with `groupSize`;
- `per-channel` without `axis`;
- `per-channel` with `groupSize`;
- `per-group` without either `axis` or `groupSize`.

It rejects `bits: 0`, `groupSize: 0`, a negative axis, and an unknown
predefined scheme string. It accepts a custom scheme object and fractional
`bits: 1.58`. The observed pattern is therefore field-local validation with
no granularity-dependent combination validation.

An accepted instance is not automatically a semantically valid contract.
Equally, a rejected instance is not automatically framework-neutral. ONNX
defines negative axis values for per-axis and blocked quantization, so the
current non-negative `axis` constraint needs review before a negative-axis
invalid fixture is proposed.

The same `$defs/quantization` object is referenced from two distinct scopes:

- `modelProperties.quantization`, described as quantization of the model
  weights as distributed; and
- `modelParameter.quantization`, describing one named input or output.

`modelParameter` has an optional `shape`; `modelProperties` has no direct
weight-tensor shape. Five context probes in the schema-audit ledger show that
the draft rejects source axis `-1` in both scopes, accepts its normalized axis
`3` for a rank-4 parameter, and also accepts a parameter axis when `shape` is
absent. Thus the draft neither carries enough information to normalize every
axis nor states whether `axis` is source-faithful or already normalized.

## Model-backed TFLite evidence

The granularity audit re-read all tensors in all subgraphs of the 50
SHA-256-identified TFLite files using the generated LiteRT FlatBuffer schema.
It did not allocate tensors or run inference. The complete per-artifact rows
and source-ledger bindings are in
[`data/tflite-granularity-evidence.json`](../data/tflite-granularity-evidence.json).

Observed totals:

| Analysis unit | Result |
| --- | ---: |
| Hash-verified artifacts | 50 |
| External parameters | 114 |
| Complete affine external parameters | 62 |
| External per-tensor contracts | 62 |
| External per-axis contracts | 0 |
| Artifacts containing a per-axis serialized tensor | 30 |
| Artifacts containing a constant per-axis tensor | 30 |
| Constant per-axis artifacts using both axes 0 and 3 | 18 |
| Constant per-axis tensors | 6,926 |
| Per-axis INT8 weight tensors | 3,443 |
| Per-axis INT32 bias tensors | 3,483 |
| Known quantized weight tensors, all granularities | 3,672 |
| Artifacts with a lossless single-object weight projection | 17 |
| Artifacts with no lossless single-object weight projection | 18 |
| Artifacts without a known quantized weight binding | 15 |
| Axis/cardinality violations | 0 |

The 62 external affine contracts each contain one scale and one zero point.
All also serialize `quantized_dimension=0`, but this value does not select a
slice because a single scale applies to the whole tensor. A normalized BOM
should therefore not infer `axis: 0` merely from this TFLite default field.

The constant detector covers both inline `Buffer.data` and TFLite's file-local
`Buffer.offset/size` representation, with bounds checks against the hashed
artifact. This matters: counting only inline buffers undercounts constants in
modern files. The 6,926 constant per-axis tensors split exactly into 3,443
INT8 weights and 3,483 INT32 biases after binding each tensor to the actual
operator input slot.

Every per-axis weight follows the operator contract: 3,099 unique
`CONV_2D` weights and 17 `FULLY_CONNECTED` weights use axis 0, while 327
`DEPTHWISE_CONV_2D` weights use axis 3. No mismatch was observed. Shared
weights make the consumer-binding counts larger than the unique-tensor
counts. In every per-axis tensor, scale and zero-point vector cardinality
equals `shape[axis]`. This connects the axis distribution to operator
semantics rather than tensor-name heuristics.

Eighteen artifacts use both axes 0 and 3 across known weight tensors. For each
artifact the audit projects every distinct `(granularity, effective axis)`
signature into the current single quantization object and records how many
weight tensors that choice represents exactly and how many it omits or
misrepresents. All 18 mixed-axis artifacts fail a lossless single-object
projection. Choosing axis 0 loses axis-3 contracts; choosing axis 3 loses
axis-0 contracts; omitting axis preserves only a coarse statement that some
weights are per-channel. Across those 18 artifacts, 1,033 known weight tensors
are in scope; even the best single signature per artifact omits or
misrepresents 330, with 11 to 48 affected tensors per artifact. This does not
make a high-level model summary impossible. It shows that the current
model-level object cannot simultaneously serve as an exhaustive weight-tensor
contract for those artifacts.

The result is independently consistent with the public LiteRT 8-bit
quantization specification: per-tensor uses one scale for the whole tensor,
whereas per-axis values apply to slices selected by `quantized_dimension`.
See <https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec>.

## Framework boundary

The 50-file benchmark is curated rather than probability sampled. Its
proportions must not be presented as ecosystem prevalence estimates.

It also does not provide empirical per-group coverage. That evidence is kept
separate in
[`data/onnx-blocked-quantization-evidence.json`](../data/onnx-blocked-quantization-evidence.json).
The audit parses the asymmetric and symmetric blocked `QuantizeLinear`
fixtures distributed with ONNX 1.22.0, verifies their pinned model hashes,
checks `x=[3,4]`, `scale=[3,2]`, `axis=1`, and `block_size=2`, and independently
recomputes all 12 expected output values in each fixture. Both shape and
numerical checks pass. Source axis `-1` normalizes to the same effective axis
for these rank-2 tensors.

GGUF block encodings may use nested or encoding-specific layouts, so a
filename label or one nominal block width is not treated here as proof of
CycloneDX `groupSize` semantics.
See <https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html>.

ONNX also creates two policy questions for a framework-neutral BOM:

- negative axis values are valid and count from the back;
- an omitted axis can inherit an operator default.

CycloneDX must therefore decide whether its quantization object is a
source-faithful partial record or a normalized effective contract. Cross-field
requirements cannot be selected rigorously until that choice is explicit.

## Fixture contribution that is safe now

One grouped valid file, such as `valid-ai-ml-quantization-2.0.json`, can cover:

- observed TFLite per-tensor external input or output metadata;
- a clearly labeled synthetic per-channel external parameter, informed by
  observed TFLite axis and cardinality behavior;
- source-backed ONNX blocked/per-group metadata;
- a custom scheme object;
- fractional bit width.

Independent invalid files can immediately cover constraints the draft already
enforces:

- `invalid-ai-ml-quantization-zero-bits-2.0.json`;
- `invalid-ai-ml-quantization-zero-group-size-2.0.json`;
- `invalid-ai-ml-quantization-unknown-scheme-2.0.json`.

Do not submit negative axis as an unqualified invalid case. It currently fails
the draft, but it is valid in ONNX and exposes a compatibility question.
Do not present an internal TFLite weight tensor as a schema-addressable external
`modelParameter`; the current public corpus contains no such external
per-channel parameter.

## Constraints requiring an explicit policy decision

If the object is defined as a normalized complete contract, the following
rules are coherent:

- `per-tensor`: omit `axis` and `groupSize`;
- `per-channel`: require an effective `axis` and omit `groupSize`;
- `per-group`: require an effective `axis` and positive `groupSize`;
- canonicalize a negative source axis to a non-negative effective axis when
  tensor rank is known, or allow the negative representation explicitly.

If partial disclosure is permitted, missing companion fields cannot simply be
made invalid. The schema should then distinguish `partial` from `complete`, or
document that semantic validation occurs outside JSON Schema. Otherwise the
same object cannot reliably mean both a complete numerical contract and a
best-effort metadata fragment.

## Claim boundary

This evidence supports the existence of a cross-field validation gap and the
operational role of axis in per-axis metadata. It also proves that one shared
model-level axis is not a lossless tensor-contract projection for 18 specific,
hash-identified artifacts, and that ONNX blocked quantization supplies a
source-backed per-group example. It does not show that all six currently
accepted combinations must be rejected, estimate how common any granularity
is across either ecosystem, or transfer ONNX/TFLite semantics to GGUF.
