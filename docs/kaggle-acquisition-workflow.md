# Kaggle acquisition workflow

This workflow implements the prospective population protocol. It is separate
from the existing 50-file curated benchmark.

## Dependencies and authentication

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-kaggle.txt
kaggle auth login
```

Authentication material is managed by the official Kaggle client. It is never
copied into a study result or raw-response ledger.

The snapshot client spaces API calls by 0.5 seconds by default. HTTP 429 and
transient 5xx responses use bounded exponential backoff, with `Retry-After`
respected when present. The completed snapshot records the request count,
retry count, and retry-event ledger. Authentication failures and response
structure errors are not retried.

Each enumeration uses a new output path and an empty raw-page directory. The
client refuses to overwrite either, preventing pages from separate attempts
from being combined accidentally.

Long enumerations write an atomic checkpoint under the ignored `cache`
directory every 25 completed models. After an interruption, repeat the exact
command with `--resume`. The command configuration, model queue, next model
index, accumulated records, raw-page bindings, request count, and retry ledger
must match before work resumes. Unreferenced raw pages written after the last
checkpoint are removed during checkpoint restoration.

Kaggle's version-list response may include records belonging to another
variation under the same model and framework. The collector retains the raw
response, filters version records by both `modelInstanceId` and
`variationSlug`, and reports the number of excluded foreign records. The
latest-version frame avoids that endpoint and uses the current version number
and identifier already serialized on each model-instance response. Historical
runs still enumerate and filter the full version list.

## Stable latest-version cohort

The prospective cohort is frozen to public model records satisfying
`700000 <= model_id < 725000`. The official listing must visibly bracket both
bounds; the collector rejects a listing that does not. TFLite eligibility is
defined by the platform framework declaration, and `.tflite` paths are then
confirmed from the latest file listing. The first 100 selected models are
cross-checked against the dedicated instance endpoint. Run two consecutive,
complete enumerations with the same bounds and cross-check count:

```bash
python scripts/snapshot-kaggle-tflite-population.py \
  --history-scope latest \
  --model-id-min 700000 \
  --model-id-max-exclusive 725000 \
  --embedded-instance-crosscheck-count 100 \
  --output data/kaggle-tflite-snapshot-a.json \
  --raw-pages data/kaggle-snapshot-pages-a
python scripts/snapshot-kaggle-tflite-population.py \
  --history-scope latest \
  --model-id-min 700000 \
  --model-id-max-exclusive 725000 \
  --embedded-instance-crosscheck-count 100 \
  --output data/kaggle-tflite-snapshot-b.json \
  --raw-pages data/kaggle-snapshot-pages-b
python scripts/compare-kaggle-snapshot-frames.py
```

`STABLE` means that the two crawls use the same identifier bounds and contain
identical selected-model identities and enumeration states, variation
identities, latest-version states, logical file paths, and listed sizes. The
diagnostic minimum and maximum IDs of Kaggle's moving 10,000-record listing
window may change as long as both bounds remain covered and the selected cohort
does not. Stability does not mean that the service supplied an atomic database
snapshot. If the result is `CHANGED_REQUIRES_NEW_ENUMERATION`, retain the
difference ledger and run another complete enumeration before analysis.

Materialize and audit only the second member of a stable pair:

```bash
python scripts/materialize-kaggle-tflite-snapshot.py \
  --snapshot data/kaggle-tflite-snapshot-b.json \
  --frame-stability data/kaggle-tflite-frame-stability.json \
  --scope latest
python scripts/audit-kaggle-tflite-snapshot.py
```

## Historical revision frame

Run the same bounded two-crawl stability procedure with `--history-scope all`,
the same model-ID bounds, and distinct output names. Bind the second all-history
snapshot to its own stable frame ledger. Then run:

```bash
python scripts/build-kaggle-revision-pairs.py \
  --snapshot data/kaggle-tflite-history-b.json \
  --frame-stability data/kaggle-tflite-history-stability.json \
  --output data/kaggle-tflite-revision-pairs.json
python scripts/materialize-kaggle-tflite-snapshot.py \
  --snapshot data/kaggle-tflite-history-b.json \
  --frame-stability data/kaggle-tflite-history-stability.json \
  --scope all \
  --output data/kaggle-tflite-materialization-all.json
python scripts/audit-kaggle-tflite-snapshot.py \
  --materialization data/kaggle-tflite-materialization-all.json \
  --output data/kaggle-tflite-audit-all-versions.json
python scripts/compare-kaggle-tflite-revisions.py \
  --pairs data/kaggle-tflite-revision-pairs.json \
  --audit data/kaggle-tflite-audit-all-versions.json \
  --output data/kaggle-tflite-revision-comparison.json
python scripts/verify-litert-interfaces.py \
  --contracts data/kaggle-tflite-audit-all-versions.json \
  --cache-root cache/kaggle-snapshot \
  --require-all \
  --output data/kaggle-litert-interface-crosscheck-all-versions.json
python scripts/crosscheck-kaggle-metadata-marker.py \
  --materialization data/kaggle-tflite-materialization-all.json \
  --audit data/kaggle-tflite-audit-all-versions.json \
  --output data/kaggle-tflite-metadata-marker-crosscheck-all-versions.json
python scripts/check-kaggle-cohort-results.py
```

Only identical logical paths in adjacent versions are compared. Path additions
and removals are retained as such; rename equivalence is not inferred. In the
completed bounded cohort, nine adjacent-version path events were observed.
Eight were additions or removals. The only same-path pair changed artifact hash
and 106 of 107 shared initializers but retained the same external dtype, shape,
scale, and zero-point contract. No affine revision was observed, and no causal
attribution or revision-frequency estimate is made from that single pair.

## Failure semantics

- Authentication or top-level enumeration failure exits without creating a
  snapshot.
- Exhausted rate-limit or transient-server retries also exit without creating
  a snapshot; a later run starts a new enumeration rather than treating a
  partial page set as a population frame.
- `--model-limit` produces `INCOMPLETE_LIMITED`; downstream population and
  revision stages reject it.
- Supplying only one model-ID bound, reversing the bounds, observing conflicting
  facts for one ID, observing non-descending deduplicated IDs, or failing to
  bracket either boundary aborts the snapshot. Byte-equivalent model facts
  repeated at a moving page boundary are counted and deduplicated.
- A mismatch between embedded variation facts and the dedicated instance
  endpoint aborts the snapshot. A recoverable endpoint access failure remains
  an explicit cross-check nonassessment.
- Non-TFLite framework variations are outside the file-level estimand and use
  `NOT_REQUESTED_NON_TFLITE_FRAMEWORK`, never a numeric zero.
- Download, listed-size, cache-integrity, and parser failures remain explicit
  nonnumeric states.
- Unassessed files are excluded from assessed denominators and are never
  treated as negative findings.
- Kaggle's listing exposes size, not a publisher content hash. SHA-256 values
  in the materialization ledger are calculated by this study after download.

Canonical page files are part of the evidence bundle. Model archives and model
bytes remain under the ignored `cache` directory and are not redistributed.
