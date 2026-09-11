# Pinned source captures

This directory retains the source documents used to interpret the serialized
measurements. `SOURCE_MANIFEST.json` records a byte length and SHA-256 for every
captured file and binds each source group to an immutable repository revision.

Primary source groups:

- CycloneDX specification PR [#990](https://github.com/CycloneDX/specification/pull/990)
  at `38dfe9ca4b9b161510dd98b477dd78e4eefe7ec8`
- CycloneDX specification PR [#1067](https://github.com/CycloneDX/specification/pull/1067)
  at `7a7d2dd599968e528349ca3cf262120da2831ca8`
- [GGUF format documentation](https://github.com/ggml-org/ggml/blob/7840aaba1989c6deeefede1d77d5aaf8f52b947e/docs/gguf.md)
- [llama.cpp importance-matrix documentation](https://github.com/ggml-org/llama.cpp/blob/c060ca974c773c7c3d17fd1b66dc9d312bc292c0/tools/imatrix/README.md)
- [IBM GGUF pipeline](https://github.com/IBM/gguf/tree/33fdd16c2dd4edbb1df71ac8feef3d98ce3faa77)
- [IBM Granite 4.2 3B](https://huggingface.co/ibm-granite/granite-4.2-3b/tree/e459acceac81e5fe67c07d9cfc72329a332e7eb1)
- [IBM Granite 4.2 3B GGUF](https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF/tree/c40945d71cd90f249a56985e8155551a9188dc30)
- [Bartowski Granite 4.2 3B GGUF](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/tree/4093456941a783ab5d8268e00a7725532c1e9af3)

Mutable API response captures are retained as observations made during package
construction; immutable commit/revision identities and artifact hashes are the
reproducibility anchors.
