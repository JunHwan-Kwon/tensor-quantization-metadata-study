# Quantization ownership probes

These six complete BOM documents isolate the interaction between the typed
quantization object in CycloneDX specification PR #990 and the string-valued
quantization properties in property-taxonomy PR #175.

They are diagnostic probes, not recommended worked examples. In particular,
`contradiction.json` is intentionally inconsistent while remaining valid
against the pinned modular JSON Schema graph. `legacy-modelcard.json` records
the rejection of the earlier aggregate placement.

The suite distinguishes four layers:

1. file identity;
2. JSON Schema validity at PR #990 head `49a9456`;
3. the normative property rules at PR #175 head `378f7b4`;
4. semantic ownership between typed fields and taxonomy properties.

From the repository root, install the pinned Python checker and Node
validator, then run both independent implementations:

```shell
python -m pip install -r requirements-cyclonedx-probes.txt
npm ci
npm run check
```

The committed result ledger is
`data/cyclonedx-pr990-validation-result.json`.
