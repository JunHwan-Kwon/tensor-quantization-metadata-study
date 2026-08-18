# CycloneDX examples

`legacy/` preserves two evidence-bound candidate fragments created against
specification PR #990 commit `58a7cc2` and property-taxonomy PR #175 commit
`1b380df`. They are retained for provenance and are not current CycloneDX 2.0
worked examples.

Those files predate the typed `modelProperties` quantization object at PR #990
head `49a9456`. They use an earlier candidate envelope and therefore must not
be copied as current schema guidance. Their observed artifact identities and
external affine values remain independently checked against the study ledger.

Current schema and ownership behavior is recorded under
`conformance/cyclonedx-2.0/`; deliberately invalid or contradictory probes are
kept there rather than presented as examples.
