# TFLite metadata audit

This audit evaluates the existing 50-file curated benchmark corpus.
It is not a prevalence sample of the TFLite ecosystem.

## Results

- Artifacts assessed: 50
- Artifacts with parseable `TFLITE_METADATA`: 20
- Artifact metadata statuses: `{"ABSENT": 30, "PRESENT_PARSEABLE": 20}`
- External parameters: 114
- Input parameters: 52
- Mapped image inputs: 15
- Mapped image inputs with valid explicit normalization: 15
- Input normalization statuses: `{"ABSENT": 7, "NOT_ASSESSABLE": 30, "PRESENT_VALID": 15}`
- Quantized inputs: 32
- Quantized inputs with valid explicit normalization: 3

## Cohort decomposition

| Cohort | Artifacts | Metadata | Inputs | Quantized inputs | Valid normalization |
| --- | ---: | ---: | ---: | ---: | ---: |
| google-legacy-quantized | 4 | 0 | 4 | 4 | 0 |
| litert-modern-static-int8 | 15 | 0 | 15 | 15 | 0 |
| mcunet-curated-ptq | 11 | 0 | 11 | 10 | 0 |
| mediapipe-public | 20 | 20 | 22 | 3 | 15 |

## Interpretation boundary

Core affine scale and zero-point are converter-serialized tensor facts.
`NormalizationOptions` is a separate publisher-supplied declaration.
Neither metadata presence nor normalization alone establishes resize,
crop, channel ordering, decoder behavior, or the complete operation order.

## Reproduce

```bash
python scripts/audit-tflite-metadata.py --download
python scripts/check-tflite-metadata-audit.py
```

Downloaded model bytes remain in the ignored `cache` directory.
