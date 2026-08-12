# Candidate CycloneDX worked examples

These files are evidence-bound candidate fragments for the AI/ML schema and
property taxonomy discussion. They follow the current draft placement and
names, but are not claimed to validate against CycloneDX 2.0 while the schema
and taxonomy pull requests remain open.

The placement follows CycloneDX/specification PR #990 head
`58a7cc2d04105e7525b0ed369ccf0a4325dc34b2`: quantization properties are
siblings of `format` on the named `modelParameter`. The property names and
values follow CycloneDX/cyclonedx-property-taxonomy PR #175 head
`1b380dfae8bf4a83646ae59ea3d3b42d466f3858`.

Both examples use public Google MediaPipe EfficientNet-Lite0 artifacts. The
float32 and int8 files are separate, hash-identified components. Values under
`candidateModelParameters` are copied from the study ledgers and independently
cross-checked with `ai-edge-litert` 2.1.4.

The corpus contains no per-axis external parameter. An internal per-axis
weight is therefore not promoted into these external-interface examples.
