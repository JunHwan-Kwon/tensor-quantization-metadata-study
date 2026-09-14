# Secondary implementation cross-check

These files are an implementation-reproducibility check produced with DeepBOM,
a measurement implementation maintained by the study author. The primary
inventories under `results/` use upstream `gguf-py` 0.19.0. No conclusion in
this experiment depends solely on DeepBOM output.

The normalized comparison checks artifact identity, tensor cardinality, every
normalized tensor descriptor, and the pairwise assignment-hash relation. Raw
secondary outputs remain in this explicitly named directory; tool branding is
not copied into primary results.
