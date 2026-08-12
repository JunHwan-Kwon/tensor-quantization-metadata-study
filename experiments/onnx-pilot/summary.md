# ONNX external-interface pilot

This pilot covers 15 revision-pinned ONNX files from five public
`onnx-community` model families. Each family contributes its published float,
int8, and uint8 variants. The selection is stratified for parser development;
it is not a prevalence sample.

All downloads were checked against the Hugging Face LFS SHA-256 and recorded
size before parsing with ONNX 1.22.0.

## Result

- 15 artifacts and 39 external parameters were assessed.
- The five float variants contained no recognized quantization operators.
- All ten int8/uint8 variants used a mixture of integer operators and dynamic
  quantization internally.
- All 39 external parameters were float tensors or non-quantized integer
  control/state tensors. None exposed an affine-quantized external interface.

The result does not mean the ten named int8/uint8 artifacts are unquantized.
It shows that their quantization is internal and that the external numerical
interface remains non-affine. File-level precision labels and external
interface contracts answer different questions.

The scanner recursively visits ONNX subgraphs for the model-level pattern
count. This matters for Silero VAD, whose top-level graph delegates computation
to subgraphs.

## Reproduce

```bash
python -m pip install -r requirements-onnx.txt
python scripts/scan-onnx-interface-contracts.py --download
python scripts/test-onnx-interface-contracts.py
python scripts/check-onnx-pilot-results.py
```

Downloaded model files are stored under `cache/onnx-pilot` and are excluded
from the repository. The manifest records the repository revision, path,
expected size, and SHA-256 for every file.

Synthetic minimal graph cases exercise the static-derived, dynamic/unbound,
ambiguous, unsupported, and non-quantized branches. Synthetic tests are kept
separate from the public-artifact result ledger.
