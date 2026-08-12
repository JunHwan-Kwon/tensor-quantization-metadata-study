# LiteRT metadata audit protocol

## Purpose

This audit determines whether a hash-identified TFLite artifact contains a
machine-readable application-preprocessing declaration. It is deliberately
separate from the affine interface ledger.

## Evidence layers

| Layer | Serialized location | Producer | Question answered |
| --- | --- | --- | --- |
| Core affine contract | TFLite tensor quantization fields | Converter | How integer storage codes map to real values |
| Model metadata | `TFLITE_METADATA` buffer | Model publisher | What the parameter represents and which out-of-graph processing is declared |
| Normalization | `TensorMetadata.process_units` / `NormalizationOptions` | Model publisher | Which mean and standard deviation are declared before quantization |
| Complete application pipeline | Artifact plus external integration evidence | Publisher or deployer | Resize, crop, color ordering, decoding, normalization, and quantization order |

A valid affine contract does not prove that normalization or the complete
pipeline is declared. A missing `NormalizationOptions` entry does not mean
that tensor scale and zero-point are missing.

## Pinned schema

The parser follows `metadata_schema.fbs` semantic version 1.5.0 from
`tensorflow/tflite-support` commit
`78d10177b3bc51f81ea78d8209c557233d15df15`.

- Git blob: `75e00dfcaa9615ebebe9888fe889802b619d21ce`
- File SHA-256: `2d3386ba124690ba1195bfc1d51ac814843bd675a7c845afab7c001c7891449e`
- Expected FlatBuffer identifier: `M001`

Only fields required by the audit are decoded. Unsupported or malformed data
is reported as not assessable rather than defaulted to absence.

## Artifact-level fields

- artifact identifier and SHA-256;
- number and names of core TFLite metadata entries;
- number of `TFLITE_METADATA` entries;
- metadata buffer SHA-256 and identifier;
- parser status;
- ModelMetadata name, version, author, license, and minimum parser version;
- count of metadata subgraphs and declared associated files.

## External-parameter fields

The main-subgraph metadata vectors are mapped by direction and ordinal, as
required by the metadata schema. Every row records:

- core tensor name, dtype, and shape;
- affine status and contract hash from the independent interface ledger;
- tensor-metadata mapping status;
- content type and image color space when declared;
- normalization status;
- mean and standard-deviation arrays;
- finite-value and nonzero-standard-deviation checks;
- scalar-broadcast or per-channel cardinality assessment;
- associated-file declaration count.

## Normalization validity

`NormalizationOptions` is `PRESENT_VALID` only when:

- exactly one normalization process unit applies to the parameter;
- mean and standard-deviation arrays are both nonempty;
- every value is finite;
- every standard deviation is nonzero;
- each vector is scalar, or both vectors have a cardinality consistent with a
  declared RGB or grayscale image content type.

Multiple normalization entries, malformed vectors, and unknown vector
cardinality receive distinct statuses. They are not collapsed into `ABSENT`.

## Interpretation boundary

Even valid normalization plus affine parameters does not prove a complete
image pipeline. Resize method, crop policy, channel ordering, decoder behavior,
and operation ordering may remain external. The audit therefore reports
`normalization_and_affine_steps_declared`, never `full_preprocessing_known`.
