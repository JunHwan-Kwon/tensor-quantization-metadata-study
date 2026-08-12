# Candidate transform alignment

This analysis compares each quantized image input contract against five
explicit pixel-to-real candidate transforms. It evaluates all integer pixel
values from 0 through 255 and records exact lattice alignment, nearest-code
agreement, code displacement, reachable codes, and clipping.

The analysis is not a preprocessing-conformance test. The TFLite artifacts do
not embed the complete application image pipeline, so the closest candidate is
not treated as the declared or intended transform.

Candidate transforms:

- raw storage: `p`
- unit interval: `p / 255`
- minus one to one: `2p / 255 - 1`
- centered 128: `(p - 128) / 128`
- centered 127.5: `(p - 127.5) / 127.5`

For signed int8 inputs, pixel bytes 128 through 255 are compared through their
two's-complement storage codes.

Reproduce with:

```bash
python scripts/analyze-candidate-transform-alignment.py
```

The complete result is in
[`data/candidate-transform-alignment.json`](../../data/candidate-transform-alignment.json).
