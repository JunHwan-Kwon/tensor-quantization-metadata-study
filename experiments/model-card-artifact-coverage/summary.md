# Model-card artifact-contained evidence coverage

This bounded study compares eight required mappings from CycloneDX #1067 head `7a7d2dd` with fields actually contained in fixed public GGUF, TFLite, and ONNX cohorts. It does not define CycloneDX completeness or conformance.

| Format | Artifacts | Strict complete observations | Mapping observations |
| --- | ---: | ---: | ---: |
| GGUF | 2 | 2 | 16 |
| TFLITE | 50 | 24 | 400 |
| ONNX | 15 | 0 | 120 |

Counts are reported per format rather than pooled as an ecosystem prevalence estimate. A strict Model Details observation requires non-empty name, version, and description. Producer-defined extension fields and repository metadata are excluded from standardized artifact capability.

> In the pinned GGUF, TFLite, and ONNX cohorts, standardized artifact-contained fields alone did not establish most model-card mappings under the stated candidate completeness profile.

See `protocol.md` for the interpretation profile and `results/artifact-coverage.json` for every subject/mapping observation.
