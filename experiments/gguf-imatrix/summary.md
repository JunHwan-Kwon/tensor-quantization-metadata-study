# GGUF and importance-matrix observations

Two revision-pinned Granite 4.2 3B artifacts use the same `Q4_K_M` release
label and expose the same 363 tensor names and 3,659,737,600 serialized
elements. Their tensor-to-encoding assignments differ for 100 names.

| Observation | IBM publication | Bartowski publication |
| --- | ---: | ---: |
| File bytes | 2,244,011,552 | 2,317,126,048 |
| Serialized tensor bytes | 2,240,440,320 | 2,313,553,920 |
| Effective storage bits per element | 4.897488 | 5.057311 |
| Q4_K tensors | 241 | 141 |
| Q5_K tensors | 0 | 80 |
| Q6_K tensors | 41 | 61 |
| F32 tensors | 81 | 81 |

Observed encoding transitions are 141 `Q4_K → Q4_K`, 80 `Q4_K → Q5_K`,
20 `Q4_K → Q6_K`, 41 `Q6_K → Q6_K`, and 81 `F32 → F32`.

The pinned importance matrix contains 560 F32 tensors and 942,360 values. The
complete scan recorded no non-finite values, no exact-zero values, no all-zero
tensors, and 280 constant-valued tensors.

The Bartowski Q4 file serializes four `quantize.imatrix.*` values, including
573 chunks and 280 entries; the IBM Q4 file serializes none of those keys.
That absence is not used to infer that a matrix was not used. The Bartowski
artifact also names `Granite 4.1 3b Base` in its embedded base-model metadata,
while its revision-pinned repository card describes the published repository
as quantized from `ibm-granite/granite-4.2-3b`. These are retained as separate
lineage observations rather than collapsed into a single source assertion.

These are serialized-artifact observations for three identified files, not an
ecosystem prevalence estimate or a claim about the conversion history. Exact
tensor rows, hash bases, source identities, standards-facing field observations,
and the second-implementation comparison are stored beside this summary.
