# CycloneDX 2.0 quantization conformance fixtures

These fixtures reproduce the AI/ML model-schema behavior of the CycloneDX 2.0 draft at
specification commit `49a945618811213e55686a23fa63b287940071c6`.
They are standard-facing fixtures, not output fixtures for a particular
analyzer.

`valid-ai-ml-quantization-2.0.json` contains four independent model
components covering:

- a per-tensor external parameter;
- a per-channel parameter with a normalized effective axis;
- source-backed ONNX blocked quantization represented as per-group metadata;
- a custom scheme and fractional nominal bit width.

The three `invalid-*` files cover constraints the pinned draft already
enforces. They intentionally do not encode unresolved policy choices such as
granularity-dependent required fields or negative source axes.

Run:

```shell
python scripts/check-cyclonedx-2.0-conformance.py --download
```

The checker verifies the pinned standalone AI/ML schema SHA-256 before
validating each fixture's `modelProperties` object. Valid fixtures must have
no validation errors. Each invalid fixture
must fail for the expected JSON Pointer and validator keyword recorded in
`manifest.json`.

The checker also records an integration probe against the bundled 2.0 schema.
At the pinned commit, the bundle still exposes legacy `modelCard` and rejects
the PR's new `component.modelProperties` field. Consequently these fixtures
must not be represented as passing full-BOM conformance cases until the bundle
is regenerated and the 2.0 functional test path is active.

The empirical basis and claim boundary are documented in
`docs/cyclonedx-2.0-quantization-fixture-evidence.md`. In particular, the
TFLite corpus contains 62 complete affine external parameters and 52
non-quantized external parameters. The latter are not incomplete affine
contracts.

## Ownership and placement probes

`quantization-ownership-probes/` is a separate six-document suite for the
interaction between the typed quantization object in specification PR #990
and the string-valued entries in property-taxonomy PR #175. It includes an
intentionally contradictory document, so these files are diagnostic probes
rather than recommended examples.

The primary checker validates complete BOM documents with Ajv against a
26-file modular graph drawn from the vendored 28-file source set. The two
bundled schemas are retained for provenance and excluded from that graph. A
second checker uses
`python-jsonschema` and independently recomputes file and ledger hashes:

```shell
python -m pip install -r requirements-cyclonedx-probes.txt
npm ci
npm run check
```

The measured result is committed as
`data/cyclonedx-pr990-validation-result.json`. See
`docs/quantization-vocabulary-mapping.md` for the field crosswalk, observed
outcomes, and design options.
