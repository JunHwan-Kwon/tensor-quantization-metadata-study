# Qwen3-4B paired GGUF replication

The two revision-pinned `Q4_K_M` files each contain 398 tensors with the same
tensor-name set, shapes, per-tensor encodings, and serialized descriptor
projection. Their common name/encoding assignment SHA-256 is
`5351ead5b14fed41548f4ef6855c2679f5cbac89451e53ef58cecfe1e3341aa7`.

| Encoding | Tensor count | Element count | Serialized tensor bytes |
| --- | ---: | ---: | ---: |
| F32 | 145 | 196,096 | 784,384 |
| Q4_K | 216 | 3,137,863,680 | 1,765,048,320 |
| Q6_K | 37 | 884,408,320 | 725,491,200 |

This is a concordant result for this pair. It does not reverse the divergent
Granite observation, and the two pairs together do not estimate ecosystem
prevalence. They show that a shared release label can coincide with either the
same or a different serialized tensor assignment, depending on the pinned
artifacts being compared.

The primary results come from upstream `gguf-py` 0.19.0. The isolated secondary
implementation cross-check reproduces the normalized descriptors and hash
relation but is not required for the conclusion.
