# Revision comparison

The comparison uses the same repository and file path at two pinned revisions:

- Repository: `onnx-community/silero-vad`
- Path: `onnx/model_fp16.onnx`
- Before: revision `730bca06348210595fb8cc15f28538707e58abbb`
- After: revision `84f0d86ecb1e2a5cab7369a69880693af6ba4b1d`

The file size and LFS SHA-256 changed. Four external parameters changed from
FLOAT16 to FLOAT32: the `input` and `state` inputs and the `output` and
`stateN` outputs. Their shapes did not change. The `sr` INT64 input did not
change. Neither revision exposes an external affine scale or zero-point, so
this is an external dtype-contract revision rather than an affine-contract
revision.

The graph also changed, but the measured structural delta is narrow:

- both revisions contain 31 recursively nested graphs;
- the recursive node count changed from 160 to 164;
- the complete operator-count delta is four added `Cast` nodes at the model
  boundary;
- all 131 initializer keys, dtypes, shapes, and serialized TensorProto bytes
  are identical between the two revisions.

The observed result is therefore that the same repository path acquired a
different external numerical interface while its serialized initializers
remained byte-identical. The graph was rewired to insert four boundary casts,
so the evidence does not support a claim that only metadata changed or that a
particular converter caused the change. Commit titles are provenance labels
only and are not used for causal attribution.

Reproduce with:

```bash
python -m pip install -r requirements-onnx.txt
python scripts/compare-onnx-revisions.py --download
```

The downloaded files are stored in the ignored `cache` directory.
