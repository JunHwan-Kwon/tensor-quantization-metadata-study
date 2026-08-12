# Quantization property value rules

Status: working draft for discussion in CycloneDX property taxonomy PR #175.

These rules describe a self-contained affine contract for a model input or
output parameter. They do not describe model quality, accuracy, or the full
application preprocessing pipeline.

## Scope

The current draft covers `per-tensor` and `per-axis` affine quantization.
Blocked quantization is deliberately deferred because it also needs a block
size and format-specific interpretation rules.

The contract uses separate properties for scheme and granularity:

- `scheme`: `affine_asymmetric` or `affine_symmetric`
- `granularity`: `per-tensor` or `per-axis`
- `scale`: a positive finite scalar or vector
- `zeroPoint`: an integer scalar or vector
- `axis`: the quantized tensor dimension, for `per-axis` only

The scheme records whether the affine mapping is subject to a symmetry
constraint. A zero-valued `zeroPoint` alone does not establish that the scheme
is symmetric.

## Scalar values

`scale` MUST be finite and greater than zero. It is serialized as the shortest
decimal string that round-trips to the source numeric value. JSON non-finite
values such as `NaN` and `Infinity` are not permitted.

`zeroPoint` MUST be an integer in the range of the declared integer data
type. For example, the valid ranges are `0..255` for `uint8` and `-128..127`
for `int8`. The rule is derived from `dataType`; it is not limited to those two
types.

If a source format permits an omitted zero point with a defined default, a
self-contained BOM representation SHOULD record the effective value unless the
declared scheme and integer representation make that value unambiguous.

## Per-tensor contract

`scale` and `zeroPoint` are scalar values. `axis` is omitted. A serialized
source field that has no per-axis meaning, such as TFLite
`quantized_dimension` on a per-tensor interface, is retained in the evidence
ledger but is not promoted to an `axis` property.

## Per-axis contract

`scale` and `zeroPoint` are vectors with equal lengths. `axis` identifies the
dimension indexed by those vectors.

For a format-neutral representation, `axis` SHOULD be normalized to a
non-negative, zero-based index in the range `0..rank-1`. A negative source
axis, where allowed by a source format, remains available in the evidence
ledger.

When the selected dimension is statically known, the vector length MUST equal
`shape[axis]`. When it is dynamic or unknown, cardinality is not fully
assessable and MUST NOT be reported as valid merely because the vector parses.

## Vector serialization

CycloneDX property values are strings. Until a typed representation exists, a
vector is serialized as compact JSON using JSON numbers, for example:

```text
[0.015625,0.03125,0.0625]
```

The representation has no insignificant whitespace and no trailing comma.
Each number follows the scalar round-trip rule above. If byte-for-byte
canonicalization or signing is required, the producer and consumer also need
an agreed JSON canonicalization profile.

## Validation levels

Three results must remain distinct:

1. **Schema-valid**: the property name and string value are accepted by the
   CycloneDX schema.
2. **Lexically valid**: the string parses as the required scalar or compact
   JSON array representation.
3. **Semantically valid**: numeric ranges, scalar/vector pairing, axis bounds,
   and vector cardinality agree with the declared tensor dtype and shape.

The generic CycloneDX name/value property schema cannot enforce lexical or
cross-field semantic rules by itself. Those checks require a profile-specific
validator.
