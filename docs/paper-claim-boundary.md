# Paper claim boundary

This note separates claims supported by committed evidence from claims that
require a broader population design or external deployment evidence.

## Supported by the current bundle

1. In the curated 50-file TFLite benchmark, 62 of 114 external parameters
   carried complete scalar affine contracts.
2. Six direction/dtype/shape signature groups contained more than one affine
   contract. Therefore dtype and shape did not uniquely identify the numerical
   interface in this benchmark.
3. LiteRT 2.1.4 independently matched all 114 interface dtype and shape rows
   and all affine values for the 62 quantized rows without inference.
4. In a deterministic 1,000-image paired experiment, two of six non-identity
   contract substitutions remained significant after Holm correction. Both
   significant directions used the narrower Google legacy contract and
   clipped 31.9% or more of input elements. The other four directions were not
   significant.
5. Twenty curated artifacts contained parseable `TFLITE_METADATA`; all were in
   the MediaPipe subcohort. The custom reader and pinned `flatc` decoder agreed
   for all 20 artifacts and 49 mapped external parameters.
6. One same-repository, same-path ONNX revision pair changed four external
   dtypes from FLOAT16 to FLOAT32 while 131 serialized initializers remained
   byte-identical. This is a dtype-interface revision, not an affine revision.
7. Two independent crawls of the frozen Kaggle model-ID cohort selected the
   same 1,966 models and 1,985 variations. One hundred embedded variation sets
   matched the dedicated instance endpoint exactly.
8. Three cohort variations were declared as TF Lite. Two publicly enumerable
   variations yielded nine assessed, SHA-distinct TFLite files; one variation
   remained unassessed after HTTP 403. One assessed file carried a complete
   integer-affine external interface and none contained parseable
   `TFLITE_METADATA`; an independent byte-marker check also found the metadata
   name absent in all nine files.
9. LiteRT 2.1.4 independently matched all 18 external parameter rows from the
   nine Kaggle files, including the two affine rows.
10. The stable all-version Kaggle frame yielded 15 SHA-distinct TFLite files
    across five enumerable versions. LiteRT 2.1.4 independently matched all 30
    external parameter rows, and the metadata parser and byte-marker check
    agreed that all 15 lacked `TFLITE_METADATA`.
11. Only one adjacent-version pair retained the same logical TFLite path. Its
    artifact hash and 106 of 107 shared serialized initializers changed, while
    its external dtype, shape, scale, and zero-point contract remained
    unchanged. Eight other adjacent-version path events were additions or
    removals.

## Not supported by the current bundle

- Kaggle-wide or global prevalence of integer interfaces, missing metadata, or
  contract ambiguity. The completed Kaggle run describes only its frozen
  identifier cohort.
- The prevalence of application/model contract mismatches in deployed systems.
- A claim that every affine mismatch materially reduces accuracy.
- A claim that `NormalizationOptions` describes the complete preprocessing
  pipeline.
- A claim that an affine revision has been observed between versions of one
  published TFLite file.
- Converter causality inferred from file changes, release notes, or repository
  commits alone.
- General model trust, safety, or fitness inferred from quantization metadata.

## Bounded cohort interpretation

The Kaggle quantities are reportable because the identifier-bounded public
cohort was completely enumerated twice, the selected frame was stable, and
download and parser coverage are explicit. They remain descriptive census
quantities for that cohort and cannot be turned into Kaggle-wide,
private-model, or deployed-harness prevalence estimates.

Historical affine changes may be reported as observations. Causal attribution
requires a controlled conversion experiment or converter instrumentation. No
affine revision was observed in the bounded Kaggle history. The one assessable
same-path pair instead changed nearly all shared initializers while preserving
its external contract, so neither converter causality nor a frequency of
revision behavior is inferred.
