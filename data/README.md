# Data contracts

The JSON and CSV ledgers in this directory are committed measurement results.
`SHA256SUMS` at the repository root binds every public file, and
`scripts/verify-bundle.py` checks the cross-document totals and source hashes.

## Pinned CycloneDX schema set

`cyclonedx-2.0-schema-set.pr990.json.gz` is a deterministic archive of 28 JSON
Schema source files read from CycloneDX/specification commit
`49a945618811213e55686a23fa63b287940071c6`. Every member contains its upstream
path, SHA-256, and base64-encoded Git blob bytes. The archive includes the
two bundled schemas for provenance. Ownership probes validate against the
remaining 26-file modular graph because the bundle had not yet been
regenerated at that commit.

`cyclonedx-pr990-validation-result.json` records the six ownership-probe
outcomes. Its `ledger_sha256` excludes only `/ledger_sha256`, as declared in
the adjacent `hash_contract`. The JavaScript generator and independent Python
checker recompute the same result from the pinned schema and taxonomy files.
