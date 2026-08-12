# Paper-ready tables and claim register

This directory contains deterministic, source-bound tables for manuscript
preparation. It is not a manuscript and does not expand the claim boundary of
the underlying studies.

- `tables/curated-cohorts.csv` keeps cohort denominators explicit.
- `tables/external-interface-summary.csv` records the analysis unit and
  denominator for every count.
- `tables/affine-contract-substitutions.csv` contains all six paired
  directions, confidence intervals, exact McNemar results, Holm adjustments,
  and input clipping fractions.
- `tables/format-pilot-summary.csv` keeps the curated TFLite benchmark and
  targeted ONNX pilot visibly separate.
- `tables/kaggle-identifier-cohort.csv` records the stable, identifier-bounded
  registry cohort and its explicit assessment coverage.
- `tables/kaggle-revision-comparisons.csv` records all nine adjacent-version
  path events in the historical cohort, including the one assessable same-path
  pair and its initializer comparison.
- `claim-register.csv` separates supported claims from quantities that have
  not been assessed.
- `table-provenance.json` binds every generated output to its source files and
  generator by SHA-256.

Regenerate and verify the files with:

```bash
python scripts/build-paper-tables.py --write
python scripts/build-paper-tables.py --check
```

The 50-file TFLite benchmark and 15-file ONNX pilot are not probability
samples. The completed Kaggle run is a descriptive census of its frozen
identifier cohort only; it does not support Kaggle-wide prevalence language.
The historical cohort contains one assessable same-path TFLite revision pair,
so it supports a pair-specific observation, not a revision-frequency estimate.
