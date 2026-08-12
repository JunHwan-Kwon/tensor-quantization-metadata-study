# Research population and estimands

Status: frozen before acquisition and executed for the bounded latest-version
Kaggle cohort on 2026-08-02 UTC. The existing 50-file corpus predates this
protocol and is analyzed as a curated benchmark corpus, not as a probability
sample.

## Research questions

The population study separates four questions that must not share a
denominator:

1. How often does a public TFLite file expose at least one integer external
   parameter with a complete affine contract?
2. How often is application preprocessing described by parseable LiteRT model
   metadata, and specifically by `NormalizationOptions`?
3. How often do identical external signatures (`direction`, `dtype`, and
   `shape`) occur with different affine contracts across hash-identified
   files?
4. How often does an external interface change between two versions of the
   same published model variation and logical file path?

Contract diversity is not deployment-mismatch prevalence. The latter would
require an independently observed application or harness contract.

## Existing 50-file corpus

The committed corpus comprises four frozen, criterion-based public subcohorts:

- Google-hosted legacy quantized models;
- revision-pinned modern LiteRT models;
- curated MCUNet PTQ models;
- public MediaPipe model objects.

It is a criterion-based benchmark corpus selected to exercise converter
generations, quantization classes, and tasks. Results may be reported for these
files and subcohorts. They must not be presented as estimates of the global
TFLite ecosystem.

## Prospective Kaggle identifier cohort

The current Kaggle model-listing response exposes a moving, create-time-sorted
window capped at 10,000 model records. Complete crawls made several hours apart
therefore need not describe the same top-of-list window. Before downloading or
parsing any TFLite file from this study, the following identifier interval was
frozen:

> Public model records returned by the official Kaggle model-listing interface
> with `700000 <= model_id < 725000`, and every variation embedded in those
> records.

The bounds were selected after inspecting only listing coverage and operational
feasibility, before observing TFLite-interface or metadata outcomes for the
cohort. Moving pagination can repeat an identical record at a page boundary.
Such repeats are counted and deduplicated by model ID; conflicting facts for
one ID invalidate the crawl. A crawl is valid only if the deduplicated IDs are
strictly descending and visibly bracket both bounds. The primary eligible
variation set is restricted prospectively to variations whose Kaggle framework field is
`MODEL_FRAMEWORK_TF_LITE` (the legacy spelling
`MODEL_FRAMEWORK_TENSOR_FLOW_LITE` is also accepted). The eligible file set is
the latest public version of each such variation containing at least one path
ending in `.tflite` (case-insensitive). Files placed under a different Kaggle
framework declaration are outside this registry-defined estimand; the study
does not claim to measure their prevalence.

This is an identifier-bounded public registry cohort, not a probability sample
and not a census of all Kaggle models. Quantities derived from it are
descriptive only for this frozen cohort.

Quantization is determined by parsing the downloaded artifact, not by search
labels or file names. Pagination continues until the service returns no next
page token. The snapshot records the CLI/API version, query parameters, page
tokens, retrieval timestamps, and SHA-256 hashes of canonicalized complete
response objects. These are not hashes of the original HTTP response bytes.
Requests are paced, and only HTTP 429 or transient 5xx failures are retried
with bounded exponential backoff. Retry events are retained in the completed
snapshot; an exhausted retry budget invalidates the enumeration attempt.
Long runs use model-boundary checkpoints. A resumed run must restore the same
configuration, ordered model queue, completed-model index, response-page
hashes, and retry telemetry before continuing.

The model-listing response already embeds variation ID, framework, slug, and
current-version facts. The collector uses these serialized fields instead of
re-requesting the same endpoint for every model. As an acquisition-validity
check, the first 100 selected models are also read through the dedicated
model-instance endpoint. Every assessed cross-check must match exactly; any
identity or current-version mismatch invalidates the snapshot. File-listing
requests are made only for the prospectively eligible TFLite framework
variations. Other variations are recorded as
`NOT_REQUESTED_NON_TFLITE_FRAMEWORK`, not as zero-file observations.

The listing service is not assumed to provide a transactional snapshot. Two
complete latest-version enumerations are therefore run consecutively. Both
must bracket the frozen identifier bounds. The frame is considered stable only
when the selected model identities and enumeration states, variation
identities, latest-version states, file paths, and listed sizes are equal. The
moving listing window's diagnostic minimum and maximum IDs may differ without
changing the bounded cohort. A selected-cohort difference is recorded and the
analysis is deferred until a stable pair is obtained.

The primary cross-sectional analysis uses only the latest version of each
variation at `T`. Historical versions are reserved for the revision analysis,
so variations with many releases do not receive extra weight in prevalence
counts.

## Analysis units

The ledger retains every level rather than collapsing them implicitly:

| Unit | Identifier | Use |
| --- | --- | --- |
| Model | owner/model slug | Publisher and family clustering |
| Variation | owner/model/framework/variation | Primary publication cluster |
| Version | variation plus integer version | Revision ordering |
| Published file | version plus logical path | Primary file-level analysis |
| Unique artifact | SHA-256 | Duplicate-sensitive secondary analysis |
| External parameter | artifact, subgraph, direction, ordinal | Contract analysis |

Two files with the same SHA-256 remain two publication records but one unique
artifact. Every result reports both published-file and unique-artifact counts.

## Primary estimands

Let `A` be latest-version `.tflite` files that were downloaded, matched the
size reported by the listing API, were locally identified by SHA-256, and were
parsed successfully. Kaggle does not expose an authoritative publisher hash
in this listing path, so the calculated SHA-256 is not described as
publisher-supplied.

### Integer-interface exposure

`P_integer_file` is the proportion of `A` containing at least one external
INT8, UINT8, INT16, or UINT16 parameter with a complete scalar affine
contract. INT32 control or state tensors are not classified as quantized
interfaces without scale and zero-point evidence.

The same quantity is reported after deduplication by SHA-256 as
`P_integer_unique`.

### Metadata and preprocessing declaration

`P_tflite_metadata` is the proportion of `A` containing exactly one parseable
`TFLITE_METADATA` buffer with the expected FlatBuffer identifier.

`P_normalization_image_input` is the proportion of mapped image input
parameters that contain a parseable and numerically valid
`NormalizationOptions` entry. Its denominator is image input parameters, not
files and not all external parameters.

Core tensor scale and zero-point are converter-serialized affine facts.
`NormalizationOptions` is an author-supplied application-preprocessing
declaration. Absence of the latter must not be described as absence of the
former.

### Signature ambiguity

For each signature group containing parameters from at least two distinct
artifact SHA-256 values, the analysis records:

- number of artifacts;
- number of unique affine-contract hashes;
- whether more than one contract hash occurs;
- maximum pairwise real-domain difference.

This demonstrates whether dtype and shape uniquely identify the numerical
interface. It is not an estimate of how often applications use the wrong
contract.

## Revision estimands

Historical comparison pairs adjacent versions within the same variation and
logical file path. Rename detection is not inferred automatically.

Each pair is assigned exactly one evidence class:

| Class | Definition |
| --- | --- |
| `ARTIFACT_IDENTICAL` | Same logical path and byte-identical artifact SHA-256 in both versions |
| `INTERFACE_UNCHANGED` | File changed but external dtype, shape, and affine contract did not |
| `EXTERNAL_PARAMETER_SET_CHANGED` | External input/output count or ordinal set changed |
| `DTYPE_OR_SHAPE_CHANGED` | External dtype or shape changed |
| `AFFINE_CHANGED_INITIALIZERS_IDENTICAL` | Scale or zero-point changed and all matched serialized initializers remained byte-identical |
| `AFFINE_CHANGED_INITIALIZERS_CHANGED` | Scale or zero-point and at least one initializer changed |
| `PATH_ADDED_OR_REMOVED` | No same-path comparison exists |
| `NOT_ASSESSABLE` | Download, integrity, or parser evidence is incomplete |

Commit titles and release notes are provenance only. They are not used for
converter or causal attribution.

## Missingness and failures

The acquisition ledger uses explicit nonnumeric states:

- `ENUMERATION_FAILED`
- `NOT_PUBLICLY_DOWNLOADABLE`
- `DOWNLOAD_FAILED`
- `SIZE_MISMATCH`
- `SHA256_MISMATCH`
- `TFLITE_PARSE_FAILED`
- `METADATA_ABSENT`
- `METADATA_PARSE_FAILED`
- `ASSESSED`

Unassessed files are never counted as negative findings. Kaggle listings expose
file size but not an authoritative content hash. A downloaded file must match
the listed size and is then locally identified by SHA-256; the calculated hash
is not described as publisher-supplied. The report presents the enumerated
frame, download coverage, size-integrity coverage, and parser coverage before
any prevalence result.

## Inference policy

If the frozen identifier cohort is exhaustively enumerated and all accessible
files are assessed, proportions are descriptive census quantities for that
cohort and do not require sampling confidence intervals. They are not Kaggle-
wide prevalence estimates. If resource limits require a sample within the
cohort, selection must use a recorded probability design at the variation
level; convenience subsampling cannot support population inference.

Multiple files from one variation are clustered. Any resampling analysis uses
the variation, not the external parameter, as the independent resampling unit.

## Claim boundary

The population study can describe public artifact characteristics within its
frozen identifier cohort. It cannot estimate the full Kaggle ecosystem,
deployed harness mismatch, downstream harm, or the prevalence of private
models. Controlled contract-substitution experiments estimate effects for the
tested artifacts and inputs only.
