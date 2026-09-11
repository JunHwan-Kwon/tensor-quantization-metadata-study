# Secondary implementation cross-check

The files under `raw/` are unmodified JSON outputs from DeepBOM, an evidence
generation implementation maintained by the study author. The primary GGUF
measurements in this experiment are produced with upstream `gguf-py` and do not
depend on these files.

`normalized-comparison.json` compares only common observable fields: artifact
identity, tensor and element counts, serialized tensor bytes, encoding counts,
the explicitly defined serialized-descriptor projection hash, and the complete
importance-matrix numeric totals. Agreement is not treated as independent
publisher attestation or as proof of conversion provenance.
