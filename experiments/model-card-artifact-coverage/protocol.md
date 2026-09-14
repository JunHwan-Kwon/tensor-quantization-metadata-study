# Model-card artifact-contained evidence protocol

## Question and boundary

This experiment asks which fields relevant to the eight `required` mappings in
CycloneDX specification PR #1067 head
`7a7d2dd599968e528349ca3cf262120da2831ca8` are standardized by each artifact
format and actually present in three fixed public cohorts. It does not decide
the normative meaning of `required`, score model cards, or issue a CycloneDX
conformance verdict.

The cohorts are the two revision-pinned Granite GGUF files already measured in
`experiments/gguf-imatrix`, the curated 50-file TFLite metadata corpus, and the
15-file ONNX pilot. Results are reported separately by format; they are not a
population estimate.

## Primary readers

- GGUF: upstream `gguf-py` 0.19.0 measurements.
- TFLite: the checked-in audit produced from the official LiteRT-generated
  FlatBuffer schema, with its existing `flatc` cross-check.
- ONNX: `onnx` 1.22.0 reads each hash-verified ModelProto directly.

DeepBOM is not a primary reader for this experiment. No conclusion depends on
DeepBOM output.

## Candidate observation profile

For every subject and required mapping the result distinguishes:

- format capability: direct, partial semantic/schema overlap, or absent;
- actual artifact presence;
- the primary-reader values used;
- strict candidate completion;
- partial overlap; and
- evidence that still must be supplied externally.

`Model Details` is strict only when name, version, and description are all
non-empty. A TFLite author or ONNX producer does not establish a CycloneDX party
with the supplier role. GGUF tags do not become a task declaration without a
publisher-bound mapping. Arbitrary custom metadata is not counted as a
standardized format field.

## Reproduction

Place the 15 ONNX files from `data/onnx-pilot-manifest.json` in the ignored
`cache/onnx-pilot` directory. File names do not matter; byte length and SHA-256
must match.

```text
python -m pip install -r requirements-onnx.txt
python scripts/build-model-card-artifact-coverage.py --write
python scripts/build-model-card-artifact-coverage.py --check
```

Artifact bytes are never written under `experiments/` and must not be committed.
