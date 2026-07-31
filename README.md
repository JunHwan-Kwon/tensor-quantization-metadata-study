# Evidence for TFLite affine interface contracts

This repository contains the supporting data for my comment on tensor-level
quantization metadata in the CycloneDX AI/ML schema discussion.

Canonical repository:
<https://github.com/JunHwan-Kwon/tensor-quantization-metadata-study>

I checked 50 public TFLite files:

- 20 Google MediaPipe models
- 11 MCUNet / TinyML models
- 4 Google-hosted legacy quantized models
- 15 revision-pinned modern LiteRT models

The files and their source URLs, revisions, sizes, and SHA-256 values are in
[`data/artifacts.json`](data/artifacts.json). Model files are not included.

## Interface results

The 50 files contain 114 external input and output parameters. Of those,
62 have a complete affine contract. The remaining 52 are unquantized:
47 FLOAT32, 1 INT32, and 4 STRING parameters.

Six direction/dtype/shape signatures occur with more than one affine contract.
For example, 22 models have an `INT8 [1,1000]` output, but all 22 use different
scale and zero-point values.

The complete parameter rows are in:

- [`data/interface-contracts.csv`](data/interface-contracts.csv)
- [`data/interface-contracts.json`](data/interface-contracts.json)

I independently read the same interfaces with `ai-edge-litert` 2.1.4 using
`get_input_details()` and `get_output_details()`. All 114 dtype and shape
records matched. For the 62 quantized parameters, scale, zero-point, and the
serialized quantized-dimension field also matched. The quantized dimension was
0 in every case, consistent with per-tensor quantization. No tensor allocation
or inference was used for that check. The result is in
[`data/litert-interface-crosscheck.json`](data/litert-interface-crosscheck.json).

## Paired input-contract check

I also tested whether using the wrong affine encoder can change predictions
while dtype and shape remain valid.

The test used 1,000 images from ImageNetV2 MatchedFrequency, one per class.
Within each class, the image was selected by a deterministic SHA-256 ordering
of the relative path with seed `20260731`. The exact selected paths are in
[`predictions.csv`](experiments/imagenetv2/predictions.csv), and their ledger
SHA-256 is recorded in
[`measurement.json`](experiments/imagenetv2/measurement.json).
For each row below, the target model, weights, images, resize/crop code, and
runtime were unchanged. Only the affine encoder was replaced by the other
model's encoder.

| Target model | Correct encoder | Other model's encoder | Difference | 95% CI |
|---|---:|---:|---:|---:|
| EfficientNet-Lite0 | 59.3% | 50.1% | -9.2 pp | [-11.6, -6.8] pp |
| MobileNetV2 | 58.1% | 56.8% | -1.3 pp | [-3.1, +0.5] pp |

The EfficientNet mismatch saturated 34.1% of input elements. The MobileNet
mismatch did not clip the input. The EfficientNet result had an exact McNemar
`p = 5.52e-14`; the MobileNet result was not significant (`p = 0.182`).

The recorded measurement and per-image predictions are in
[`experiments/imagenetv2`](experiments/imagenetv2).

This is evidence that dtype and shape do not fully identify the numerical
interface of an affine-quantized tensor. It is not a general model-risk score,
and the two-model experiment is not an estimate of how often such mismatches
occur.

## Check the committed results

The recorded run used Python 3.12.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\verify-bundle.py
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify-bundle.py
```

The verifier recomputes the corpus counts, the numerical examples, the paired
statistics, and all hashes in `SHA256SUMS`.

## Author

Jun-Hwan Kwon, Ph.D.  
Anesthesia and Pain Research Institute  
Yonsei University College of Medicine, Republic of Korea  
[ORCID 0000-0002-6464-3895](https://orcid.org/0000-0002-6464-3895)

This is independently developed research software. The affiliation identifies
the author and does not imply institutional endorsement.
