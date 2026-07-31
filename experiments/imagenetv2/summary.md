# Affine Interface Mismatch Measurement

Dataset: ImageNetV2 MatchedFrequency, 1000 images, 1000 ImageNet classes.

| Target model | Correct top-1 | Wrong top-1 | Paired delta | 95% CI | Prediction agreement | Input clip | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| google-mobilenet-v2-1.0-224-quant | 58.10% | 56.80% | -1.30 pp | [-3.10, 0.50] pp | 80.80% | 0.00% | 0.1821 |
| mediapipe-efficientnet-lite0-int8 | 59.30% | 50.10% | -9.20 pp | [-11.60, -6.80] pp | 65.70% | 34.11% | 5.523e-14 |

## Interpretation

This is a controlled contract-fault injection. For each target model, the decoded RGB byte tensor is treated as the correct integer input. The same target-real tensor is then re-encoded with the other model's affine contract. Architecture, weights, images, crop, and runtime remain fixed.

Input clipping is counted exactly before invocation. Internal qmin/qmax occupancy is an observed proxy and is not reported as an exact clamp-event count.

## Limits

ImageNetV2 is an independently collected test set, but this run may use a deterministic subset rather than all 10,000 MatchedFrequency images. This experiment estimates the effect of the injected contract mismatch on these two hash-pinned artifacts; it does not estimate mismatch prevalence or establish a universal accuracy effect.

Result ledger SHA-256: `abf0999e288097a82303ea6811888d9a82f717ac734182ad858c174061428270`
