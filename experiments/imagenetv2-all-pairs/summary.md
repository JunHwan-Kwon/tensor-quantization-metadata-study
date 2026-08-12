# Distinct affine-contract all-pairs experiment

1000 ImageNetV2 images, 3 target models, 3 distinct input contracts, and 6 non-identity comparisons.

| Target | Source contract aliases | Baseline top-1 | Substituted top-1 | Delta | Input clip | Holm p |
|---|---|---:|---:|---:|---:|---:|
| google-legacy-quantized/mobilenet-v2-1.0-224-quant | mediapipe-public/efficientnet-lite2-int8 | 58.10% | 58.00% | -0.10% | 0.00% | 1 |
| google-legacy-quantized/mobilenet-v2-1.0-224-quant | mediapipe-public/efficientnet-lite0-int8 | 58.10% | 56.80% | -1.30% | 0.00% | 0.5462 |
| mediapipe-public/efficientnet-lite0-int8 | google-legacy-quantized/inception-v3-quant, google-legacy-quantized/mobilenet-v1-1.0-224-quant, google-legacy-quantized/mobilenet-v2-1.0-224-quant | 59.30% | 50.10% | -9.20% | 34.11% | 3.314e-13 |
| mediapipe-public/efficientnet-lite0-int8 | mediapipe-public/efficientnet-lite2-int8 | 59.30% | 58.40% | -0.90% | 3.55% | 0.4883 |
| mediapipe-public/efficientnet-lite2-int8 | google-legacy-quantized/inception-v3-quant, google-legacy-quantized/mobilenet-v1-1.0-224-quant, google-legacy-quantized/mobilenet-v2-1.0-224-quant | 62.30% | 56.40% | -5.90% | 31.92% | 2.844e-07 |
| mediapipe-public/efficientnet-lite2-int8 | mediapipe-public/efficientnet-lite0-int8 | 62.30% | 62.00% | -0.30% | 1.72% | 1 |

## Interpretation boundary

Model aliases sharing exactly the same dtype, scale, zero-point, and integer range are collapsed into one source contract. Identity substitutions are not counted as comparisons.

Each comparison changes only the affine encoder used for a fixed target artifact and target-specific image crop. Results establish effects for these hash-pinned files and selected images; they do not estimate mismatch prevalence or a universal accuracy loss.

Result ledger SHA-256: `22f5fc708e0e512133058c409f114d3335616633c374ac3a748b7b1045d47a92`
