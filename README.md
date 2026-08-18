# Evidence for TFLite affine interface contracts

This repository contains the supporting data for my comment on tensor-level
quantization metadata in the CycloneDX AI/ML schema discussion.

Canonical repository:
<https://github.com/JunHwan-Kwon/tensor-quantization-metadata-study>

The recorded benchmark corpus contains 50 public TFLite files:

- 20 Google MediaPipe models
- 11 MCUNet / TinyML models
- 4 Google-hosted legacy quantized models
- 15 revision-pinned modern LiteRT models

The files and their source URLs, revisions, sizes, and SHA-256 values are in
[`data/artifacts.json`](data/artifacts.json). Model files are not included.

These four subcohorts were selected to exercise different converter
generations, quantization classes, and tasks. They are a curated benchmark,
not a probability sample, and none of the 50-file proportions below are
presented as estimates of the TFLite ecosystem. The prospective population
and estimands are specified in
[`docs/research-population-protocol.md`](docs/research-population-protocol.md).

## CycloneDX draft conformance evidence

The repository also contains standard-facing fixtures and six complete BOM
probes for CycloneDX specification PR #990 and property-taxonomy PR #175. The
suite separates JSON Schema validity, taxonomy wording, and semantic ownership
instead of treating them as one result. It includes an intentionally
contradictory document that is schema-valid at the pinned PR head.

Run the two independent validators with:

```bash
npm ci
npm run check
```

See [`conformance/cyclonedx-2.0/`](conformance/cyclonedx-2.0/), the committed
[`validation result`](data/cyclonedx-pr990-validation-result.json), and the
[`vocabulary mapping note`](docs/quantization-vocabulary-mapping.md).

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
serialized quantized-dimension field also matched. Its serialized value was 0
in all 62 rows; because each contract had one scale and one zero-point, that
field did not carry per-axis semantics. No tensor allocation or inference was
used for that check. The result is in
[`data/litert-interface-crosscheck.json`](data/litert-interface-crosscheck.json).

To download the 50 hash-identified artifacts and rerun the independent LiteRT
interface check rather than only verify the committed result document:

```bash
python scripts/audit-tflite-metadata.py --download
python scripts/verify-litert-interfaces.py \
  --cache-root cache/tflite-metadata-audit --require-all
```

Downloaded model bytes remain under the ignored `cache` directory.

## TFLite model-metadata audit

Core tensor affine metadata and application preprocessing are separate
evidence layers. Scale and zero-point are stored in the core TFLite tensor;
publisher-authored `NormalizationOptions`, when present, are stored in the
`TFLITE_METADATA` buffer.

In the curated corpus, 20 of 50 artifacts contained parseable
`TFLITE_METADATA`. All 20 came from the MediaPipe subcohort; the other three
subcohorts contained none. Of 52 input parameters, 15 mapped image inputs had
a valid explicit normalization declaration. Three of those 15 inputs were
also affine-quantized. This complete cohort separation is why these counts
must not be interpreted as prevalence estimates.

The custom metadata reader was independently compared with FlatBuffers
`flatc` 25.9.23 using the pinned LiteRT Support metadata schema. All 20
metadata-bearing artifacts and all 49 mapped external parameters agreed, with
zero field mismatches. The audit, independent cross-check, and method boundary
are recorded in:

- [`data/tflite-metadata-audit.json`](data/tflite-metadata-audit.json)
- [`data/tflite-metadata-flatc-crosscheck.json`](data/tflite-metadata-flatc-crosscheck.json)
- [`docs/metadata-audit-protocol.md`](docs/metadata-audit-protocol.md)

Neither metadata presence nor `NormalizationOptions` alone establishes resize,
crop, decoding, channel order, or the full application operation sequence.

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

## Expanded distinct-contract experiment

A second run expands the test to three distinct UINT8 affine input contracts.
Five qualified public files contribute contract identities; files with an
identical contract are collapsed, and one target representative is executed
per distinct contract. This produces six non-identity directions over the
same 1,000 ImageNetV2 images.

After Holm correction across all six exact McNemar tests, two directions
remained significant:

- EfficientNet-Lite0 with the Google legacy contract: 59.3% to 50.1%
  (`-9.2` percentage points, Holm `p = 3.31e-13`)
- EfficientNet-Lite2 with the Google legacy contract: 62.3% to 56.4%
  (`-5.9` percentage points, Holm `p = 2.84e-7`)

The other four directions ranged from `-1.3` to `-0.1` percentage points and
were not significant after correction. The two significant substitutions
also clipped 34.1% and 31.9% of input elements, respectively. This supports a
direction-dependent effect; it does not imply that every affine mismatch has
a material accuracy cost.

The complete matrix, method boundary, 6,000 paired prediction rows, and
verification script are in
[`experiments/imagenetv2-all-pairs`](experiments/imagenetv2-all-pairs).

That recorded run is bound to the exact artifact and interface ledgers used at
execution time. Later corrections to the current interface ledger are not
substituted into its provenance. The original hash-matched source documents
are retained under [`data/recorded-run-sources`](data/recorded-run-sources)
for offline verification. They can also be recovered byte-for-byte from the
pinned public repository commit recorded by
`scripts/reconstruct-recorded-run-sources.py`. These historical ledgers are
provenance inputs for the recorded experiment, not the current corpus result.
Historical `deepbom.*` schema identifiers remain unchanged only where exact
bytes are part of that SHA-bound provenance; they are not current schema names.

To reproduce the expanded run (model and dataset bytes are downloaded into
the ignored `cache` directory):

```bash
python scripts/measure-affine-all-pairs.py \
  --output experiments/imagenetv2-all-pairs \
  --max-images 1000 --bootstrap-iterations 5000
```

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

The verifier recomputes the committed ledger counts, numerical examples, and
paired statistics; checks cross-document provenance bindings; runs the
acquisition and parser unit tests; and validates every public file against
`SHA256SUMS`.

To reproduce the independent metadata decoder check on Windows:

```powershell
python scripts\prepare-flatc-crosscheck.py
python scripts\crosscheck-tflite-metadata-flatc.py
python scripts\check-tflite-metadata-flatc-crosscheck.py
```

## Standards preparation

Supporting material prepared for the CycloneDX AI/ML property taxonomy work
is kept separate from the recorded study results:

- [`docs/quantization-property-value-rules.md`](docs/quantization-property-value-rules.md)
- [`docs/evidence-mapping.md`](docs/evidence-mapping.md)
- [`docs/cyclonedx-open-questions.md`](docs/cyclonedx-open-questions.md)
- [`docs/cyclonedx-2.0-quantization-fixture-evidence.md`](docs/cyclonedx-2.0-quantization-fixture-evidence.md)
- [`docs/cyclonedx-2.0-quantization-proof-matrix.md`](docs/cyclonedx-2.0-quantization-proof-matrix.md)
- [`examples/cyclonedx`](examples/cyclonedx)

The example files are candidate fragments, not claims of conformance to the
still-changing CycloneDX 2.0 draft.

The fixture-evidence audit pins specification PR #990 commit
`49a945618811213e55686a23fa63b287940071c6`, records a 12-case JSON Schema
validation matrix, and separately re-reads all serialized tensor
quantization records in the 50 hash-identified TFLite artifacts. Reproduce
the model-backed portion with:

```bash
python scripts/audit-tflite-granularity-evidence.py \
  --cache-root cache/tflite-metadata-audit --require-all
python scripts/audit-onnx-blocked-quantization-evidence.py \
  --output data/onnx-blocked-quantization-evidence.json
python scripts/check-quantization-fixture-evidence.py
```

The TFLite audit resolves both inline and file-offset constant buffers, binds
weights and biases to actual operator input slots, and records whether each
artifact's tensor contracts can be projected losslessly into one model-level
granularity/axis pair. The ONNX audit independently recomputes the two official
blocked `QuantizeLinear` fixtures distributed with ONNX 1.22.0.

## Paper-ready outputs

Deterministically generated manuscript tables, a claim register, and their
source SHA-256 bindings are in [`paper`](paper). The register keeps supported
observations separate from population and deployment claims that have not been
assessed. Regenerate them with `python scripts/build-paper-tables.py --write`
and verify them with `python scripts/build-paper-tables.py --check`.

## ONNX pilot

An additional 15-file ONNX pilot tests whether affine contracts can be
reconstructed at an external graph interface. It records model-level
quantization patterns separately from external parameter contracts:

- [`data/onnx-pilot-manifest.json`](data/onnx-pilot-manifest.json)
- [`data/onnx-pilot-results.json`](data/onnx-pilot-results.json)
- [`experiments/onnx-pilot/summary.md`](experiments/onnx-pilot/summary.md)

The model files are downloaded into the ignored `cache` directory and are not
redistributed. ONNX 1.22.0 is pinned in the main requirements for complete
bundle verification; `requirements-onnx.txt` remains a minimal dependency
file for running only the ONNX acquisition and parser path.

A pinned same-repository, same-path ONNX revision comparison is recorded in
[`data/onnx-revision-comparison.json`](data/onnx-revision-comparison.json).
In the observed Silero VAD pair, four external parameters changed from FLOAT16
to FLOAT32 while their shapes stayed fixed. No external affine scale or
zero-point was present in either revision. All 131 serialized initializers
were byte-identical; the recursive operator-count delta was four boundary
`Cast` nodes. This is evidence of a same-path external dtype-contract revision,
not an affine-contract revision, and it does not infer converter causality from
commit metadata.

Candidate alignment against five explicit pixel transforms is recorded in
[`data/candidate-transform-alignment.json`](data/candidate-transform-alignment.json).
It reports code alignment and clipping, not preprocessing conformance.

## Kaggle identifier-cohort study

The repository contains a separate acquisition path for a protocol-frozen
Kaggle Models identifier cohort (`700000 <= model_id < 725000`). The bounds were
frozen before TFLite-interface outcomes were materialized. The collector
exhaustively paginates the visible model listing, requires it to bracket both
bounds, enumerates selected variations and files, records canonicalized API
response objects and their SHA-256 values, and never converts an enumeration,
download, integrity, or parse failure into a negative finding. Because the
service is not a transactional snapshot, two complete latest-version
enumerations of the bounded cohort must agree before analysis. Results describe
this cohort only; they are not Kaggle-wide prevalence estimates.

TFLite eligibility is registry-defined: a variation must be declared with the
Kaggle TF Lite framework, after which its latest `.tflite` paths are verified.
Variation facts embedded in the model listing are cross-checked against the
dedicated instance endpoint for the first 100 selected models. Variations under
other framework declarations are outside this file-level estimand rather than
counted as negative observations.

The completed stable pair contains 1,966 models and 1,985 embedded variations.
Three variations were declared as TF Lite. Two publicly enumerable variations
yielded nine latest-version `.tflite` files; the third returned HTTP 403 and is
retained as unassessed. All nine downloaded files matched their listed sizes,
parsed successfully, and were unique by SHA-256. One file exposed a complete
integer-affine external interface; the other eight exposed float interfaces.
None contained the `TFLITE_METADATA` FlatBuffer name marker, consistent with
the metadata parser's `ABSENT` result. LiteRT 2.1.4 independently matched all
18 external dtype, shape, scale, zero-point, and quantized-dimension rows.

The corresponding stable all-version frame contained 15 SHA-distinct TFLite
files across five publicly enumerable versions. Five files exposed complete
integer-affine interfaces, giving 10 complete affine rows among 30 external
parameters. LiteRT 2.1.4 independently matched all 30 rows, and both the
metadata parser and an exact byte-marker check found no `TFLITE_METADATA` in
the 15 files. Adjacent-version pairing produced one assessable same-path pair:
the v2 and v3 `model_qad_int8.tflite` files had different artifact hashes and
106 of 107 shared serialized initializers changed, while external dtype,
shape, scale, and zero-point remained unchanged. Eight other path events were
additions or removals. This is not evidence of an affine revision, converter
causality, or a revision-frequency estimate.

These are descriptive counts for the frozen identifier cohort, not estimates
of Kaggle-wide or global TFLite prevalence. Reproducing the acquisition requires
an authenticated Kaggle account and the pinned optional dependency:

```bash
python -m pip install -r requirements-kaggle.txt
kaggle auth login
python scripts/snapshot-kaggle-tflite-population.py --history-scope latest \
  --model-id-min 700000 --model-id-max-exclusive 725000 \
  --embedded-instance-crosscheck-count 100 \
  --output data/kaggle-tflite-snapshot-a.json \
  --raw-pages data/kaggle-snapshot-pages-a
python scripts/snapshot-kaggle-tflite-population.py --history-scope latest \
  --model-id-min 700000 --model-id-max-exclusive 725000 \
  --embedded-instance-crosscheck-count 100 \
  --output data/kaggle-tflite-snapshot-b.json \
  --raw-pages data/kaggle-snapshot-pages-b
python scripts/compare-kaggle-snapshot-frames.py
python scripts/materialize-kaggle-tflite-snapshot.py --scope latest \
  --snapshot data/kaggle-tflite-snapshot-b.json
python scripts/audit-kaggle-tflite-snapshot.py
python scripts/check-kaggle-cohort-results.py
```

The revision analysis uses a separate `--history-scope all` snapshot. Build
its pairs, materialize with `--scope all`, and audit that materialization to an
all-version output before running
`scripts/compare-kaggle-tflite-revisions.py`. The all-version audit is also
cross-checked with LiteRT and an exact metadata-marker scan. Downloaded model
bytes and archives remain under the ignored `cache` directory. A limited pilot
is marked `INCOMPLETE_LIMITED` and is rejected by the population and revision
stages.
The complete command sequence and failure semantics are in
[`docs/kaggle-acquisition-workflow.md`](docs/kaggle-acquisition-workflow.md).

## Author

Jun-Hwan Kwon, Ph.D.  
Anesthesia and Pain Research Institute  
Yonsei University College of Medicine, Republic of Korea  
[ORCID 0000-0002-6464-3895](https://orcid.org/0000-0002-6464-3895)

This is independently developed research software. The affiliation identifies
the author and does not imply institutional endorsement.
