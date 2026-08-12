# CycloneDX taxonomy questions

Snapshot checked: 2026-08-12 UTC, PR #175 head
`1b380dfae8bf4a83646ae59ea3d3b42d466f3858` and specification PR #990 head
`58a7cc2d04105e7525b0ed369ccf0a4325dc34b2`.

PR #175 now contains draft affine quantization properties and worked examples.
The remaining questions distinguish localized wording corrections from
schema-placement and validation decisions.

## Namespace

The current `cdx:ai-ml:model:parameter` namespace describes learned parameter
count and tuning methods. The draft schema's `modelParameter` object describes
external model inputs and outputs. Would a distinct namespace for interface or
tensor-format properties avoid giving `parameter` two meanings?

## Naming convention

PR #175 currently uses `zeroPoint` and hyphenated granularity values such as
`per-tensor`. Candidate examples use those exact draft spellings so that
interoperability tests detect accidental drift.

## Scheme and symmetry

PR #175 currently combines the mathematical family and symmetry classification
as `affine_asymmetric` and `affine_symmetric`. The descriptions still need to
avoid treating a zero-valued zero point as proof of symmetry or describing
symmetric signed INT8 as halving the integer range.

## Scalar and vector values

Should scalar values use shortest round-trip decimal strings, and should
per-axis vectors use compact JSON arrays? If so, the taxonomy should state
that cardinality and axis validation require a semantic validator beyond the
generic property schema.

## Placement

The superseded #948 branch placed `properties` inside `modelParameter.format`.
The current #990 draft instead exposes sibling `modelParameter.properties` and
does not expose `format.properties`. Candidate examples follow the sibling
location; maintainer confirmation is still required before calling it final.

## Processing stage vocabulary

#948 accepted `pre-processing` and `post-processing`. The superseding #990
branch replaces those values with `raw` and `processed`. A model-ready
quantized tensor would therefore use `processed` if the current vocabulary is
retained.
