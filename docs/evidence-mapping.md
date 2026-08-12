# Evidence mapping

This note keeps observed artifact facts separate from derived and synthetic
material used in candidate CycloneDX examples.

## Evidence grades

| Grade | Meaning |
| --- | --- |
| `OBSERVED` | Read directly from a hash-identified serialized artifact. |
| `OBSERVED_CROSS_CHECKED` | Read by the study parser and independently matched by `ai-edge-litert` 2.1.4. |
| `DERIVED_STATIC` | Deterministically calculated from serialized artifact facts without inference. |
| `DERIVED_SERIALIZED_GRAPH` | Reconstructed from static graph nodes and initializers rather than direct interface metadata. |
| `SYNTHETIC` | Constructed to exercise a rule; not attributed to a public artifact. |
| `NOT_ASSESSABLE` | Required evidence was absent, dynamic, ambiguous, or unsupported. |

`NOT_ASSESSABLE` is not equivalent to a zero value or to evidence that a
contract is absent.

## Candidate worked examples

| Example | Artifact | Grade | Source |
| --- | --- | --- | --- |
| EfficientNet-Lite0 float32 | SHA-256 `6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0` | `OBSERVED_CROSS_CHECKED` | `data/artifacts.json`, `data/interface-contracts.json`, `data/litert-interface-crosscheck.json` |
| EfficientNet-Lite0 int8 | SHA-256 `bc2ffe19c1118de0c0c2a9088992da5589722656e0fba81421385300a4a34b16` | `OBSERVED_CROSS_CHECKED` | Same ledgers as above |
| Per-axis external parameter | No public artifact selected | `SYNTHETIC` if needed | Must be labeled synthetic unless a hash-identified external interface is found |

The two EfficientNet files are separate Google MediaPipe objects and separate
hash-identified components. They demonstrate how full-precision and int8
variants from one published model family require different interface records.

The 50-file TFLite corpus contains no per-axis external interface. Internal
per-axis kernels are not used as model input/output examples because the draft
CycloneDX `modelParameter` describes external parameters.

## Interface facts used by the examples

### Float32 variant

- Input: `FLOAT32 [1,224,224,3]`, no affine metadata
- Output: `FLOAT32 [1,1000]`, no affine metadata
- Source object generation: `1682480007106905`

### Int8 variant

- Input: `UINT8 [1,224,224,3]`, scale
  `0.012566016986966133`, zero point `131`
- Output: `UINT8 [1,1000]`, scale `0.00390625`, zero point `0`
- Both external contracts are per-tensor
- Source object generation: `1682480006900522`

Both artifacts contain parseable `TFLITE_METADATA`. Their image inputs declare
`NormalizationOptions` with scalar mean `127.0` and standard deviation
`128.0`, independently matched by `flatc`. This is a publisher-supplied
normalization declaration, not part of the affine tensor contract. It does not
by itself establish resize, crop, image decoding, channel ordering, or the full
application operation sequence.
