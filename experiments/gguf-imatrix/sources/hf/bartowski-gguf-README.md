---
quantized_by: bartowski
pipeline_tag: text-generation
language:
- en
- de
- es
- fr
- ja
- pt
- ar
- cs
- it
- ko
- nl
- zh
license: apache-2.0
base_model: ibm-granite/granite-4.2-3b
tags:
- granite
- granite-4.2
- reasoning
- thinking
- tool-calling
- ibm
base_model_relation: quantized
---

## Llamacpp imatrix Quantizations of granite-4.2-3b by ibm-granite

Using <a href="https://github.com/ggml-org/llama.cpp/">llama.cpp</a> release <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b10603">b10603</a> for quantization.

Original model: https://huggingface.co/ibm-granite/granite-4.2-3b

**Model details:**
- Parameter count: 4B
- Input support: text
- Speculative decoding: no
- imatrix: yes - [details](#imatrix)

[How to run](#how-to-run)

## Prompt format

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
```

**Don't know which to choose?** Grab [Q4_K_M](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q4_K_M.gguf) (2.32GB) - usually a good mix of size and performance. Download instructions available [here](#downloading-using-the-hugging-face-cli)

## Available files:

| Filename | Quant type | File Size | Split | Description |
| -------- | ---------- | --------- | ----- | ----------- |
| [granite-4.2-3b-bf16.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-bf16.gguf) | bf16 | 7.32GB | false | Full BF16 weights. |
| [granite-4.2-3b-Q8_0.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q8_0.gguf) | Q8_0 | 3.89GB | false | Extremely high quality, generally unneeded but max available quant. |
| [granite-4.2-3b-Q6_K_L.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q6_K_L.gguf) | Q6_K_L | 3.25GB | false | Uses Q8_0 for embed and output weights. Very high quality, near perfect, *recommended*. |
| [granite-4.2-3b-Q6_K.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q6_K.gguf) | Q6_K | 3.12GB | false | Very high quality, near perfect, *recommended*. |
| [granite-4.2-3b-Q5_K_L.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q5_K_L.gguf) | Q5_K_L | 2.83GB | false | Uses Q8_0 for embed and output weights. High quality, *recommended*. |
| [granite-4.2-3b-Q5_K_M.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q5_K_M.gguf) | Q5_K_M | 2.67GB | false | High quality, *recommended*. |
| [granite-4.2-3b-Q5_K_S.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q5_K_S.gguf) | Q5_K_S | 2.57GB | false | High quality, *recommended*. |
| [granite-4.2-3b-Q4_K_L.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q4_K_L.gguf) | Q4_K_L | 2.51GB | false | Uses Q8_0 for embed and output weights. Good quality, *recommended*. |
| [granite-4.2-3b-Q4_1.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q4_1.gguf) | Q4_1 | 2.37GB | false | Legacy format, similar performance to Q4_K_S but with improved tokens/watt on Apple silicon. |
| [granite-4.2-3b-Q4_K_M.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q4_K_M.gguf) | Q4_K_M | 2.32GB | false | Good quality, default size for most use cases, *recommended*. |
| [granite-4.2-3b-Q3_K_XL.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q3_K_XL.gguf) | Q3_K_XL | 2.22GB | false | Uses Q8_0 for embed and output weights. Lower quality but usable, good for low RAM availability. |
| [granite-4.2-3b-Q4_K_S.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q4_K_S.gguf) | Q4_K_S | 2.18GB | false | Slightly lower quality with more space savings, *recommended*. |
| [granite-4.2-3b-IQ4_NL.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-IQ4_NL.gguf) | IQ4_NL | 2.18GB | false | Similar to IQ4_XS, but slightly larger. |
| [granite-4.2-3b-Q4_0.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q4_0.gguf) | Q4_0 | 2.17GB | false | Legacy format, kept for compatibility with older tools. |
| [granite-4.2-3b-IQ4_XS.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-IQ4_XS.gguf) | IQ4_XS | 2.08GB | false | Decent quality, smaller than Q4_K_S with similar performance, *recommended*. |
| [granite-4.2-3b-Q3_K_L.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q3_K_L.gguf) | Q3_K_L | 1.99GB | false | Lower quality but usable, good for low RAM availability. |
| [granite-4.2-3b-Q3_K_M.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q3_K_M.gguf) | Q3_K_M | 1.89GB | false | Low quality. |
| [granite-4.2-3b-IQ3_M.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-IQ3_M.gguf) | IQ3_M | 1.79GB | false | Medium-low quality, new method with decent performance comparable to Q3_K_M. |
| [granite-4.2-3b-Q2_K_L.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q2_K_L.gguf) | Q2_K_L | 1.77GB | false | Uses Q8_0 for embed and output weights. Very low quality but surprisingly usable. |
| [granite-4.2-3b-Q3_K_S.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q3_K_S.gguf) | Q3_K_S | 1.73GB | false | Low quality, not recommended. |
| [granite-4.2-3b-IQ3_XS.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-IQ3_XS.gguf) | IQ3_XS | 1.68GB | false | Lower quality, new method with decent performance, slightly better than Q3_K_S. |
| [granite-4.2-3b-IQ3_XXS.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-IQ3_XXS.gguf) | IQ3_XXS | 1.57GB | false | Lower quality, new method with decent performance, comparable to Q3 quants. |
| [granite-4.2-3b-Q2_K.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-Q2_K.gguf) | Q2_K | 1.52GB | false | Very low quality but surprisingly usable. |
| [granite-4.2-3b-IQ2_M.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-IQ2_M.gguf) | IQ2_M | 1.49GB | false | Relatively low quality, uses SOTA techniques to be surprisingly usable. |

Download a specific file:

```
hf download bartowski/granite-4.2-3b-GGUF --include "granite-4.2-3b-Q4_K_M.gguf" --local-dir ./
```

## Downloading using the Hugging Face CLI

<details>
  <summary>Click to view download instructions</summary>

First, make sure you have the Hugging Face CLI installed:

```
pip install -U "huggingface_hub[cli]"
```

Download a specific file:

```
hf download bartowski/granite-4.2-3b-GGUF --include "granite-4.2-3b-Q4_K_M.gguf" --local-dir ./
```

</details>

## How to run

These quants run with [llama.cpp](https://github.com/ggml-org/llama.cpp) - installable in one line via [llama.app](https://llama.app/):

```
curl -LsSf https://llama.app/install.sh | sh
llama-server -hf bartowski/granite-4.2-3b-GGUF:Q4_K_M
```

llama-server includes a built-in chat web UI, served at http://localhost:8080 by default.

These quants were made with llama.cpp release b10603 - if this model's architecture is newly supported, you'll need that release or newer to run them.

They also work in: [LM Studio](https://lmstudio.ai/) · [koboldcpp](https://github.com/LostRuins/koboldcpp) · [ramalama](https://github.com/containers/ramalama) · [Jan AI](https://www.jan.ai/) · [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui) · [LoLLMs](https://github.com/ParisNeo/lollms) · [Atomic Chat](https://atomic.chat/)

## imatrix

All quants made using imatrix option, with a calibration corpus rendered through this model's own chat template. The corpus pairs plain prose with tool-calling and reasoning conversations ([corpus source data](https://gist.github.com/bartowski1182/e26453c0404e24eb317543ec5360f87a)), encoded exactly as this model sees them at inference and processed with `--parse-special`, so chat-format special tokens contribute to the importance matrix. The corpus rendered for this model is included in this repo: [granite-4.2-3b-calibration-v6.txt](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-calibration-v6.txt). The imatrix is available here: [granite-4.2-3b-imatrix.gguf](https://huggingface.co/bartowski/granite-4.2-3b-GGUF/blob/main/granite-4.2-3b-imatrix.gguf).

<details>
<summary>Calibration render details</summary>

```json
{
  "generator": "auto_quant_v2 calibration renderer",
  "recipe": "calibration-v6",
  "model": "granite-4.2-3b",
  "encoder": "chat_template",
  "chunk_size": 512,
  "prose_chunks": 238,
  "tool_chunks": 335,
  "total_chunks": 573,
  "tool_chunk_fraction": 0.585,
  "n_conversations": 137,
  "extension_convs_used": 0,
  "conversation_token_lengths": [
    564,
    1607,
    917,
    1357,
    1139,
    1405,
    2624,
    645,
    1264,
    1198,
    1039,
    2048,
    890,
    992,
    2454,
    1260,
    1095,
    986,
    754,
    686,
    1386,
    1046,
    1427,
    1192,
    1728,
    1524,
    1058,
    750,
    988,
    1454,
    1579,
    812,
    1267,
    1031,
    1046,
    1706,
    1634,
    1169,
    484,
    1909,
    1314,
    1127,
    1401,
    1786,
    1975,
    1334,
    1554,
    775,
    2485,
    1073,
    2687,
    776,
    1014,
    766,
    864,
    705,
    2312,
    756,
    1114,
    1058,
    1257,
    1172,
    958,
    1005,
    911,
    1335,
    915,
    1520,
    1854,
    861,
    331,
    1124,
    3109,
    2900,
    717,
    761,
    1046,
    804,
    1302,
    1097,
    1167,
    824,
    1050,
    1094,
    1274,
    1531,
    1367,
    1436,
    867,
    665,
    2520,
    651,
    1108,
    1625,
    1967,
    1198,
    633,
    1357,
    1077,
    1678,
    1718,
    1760,
    832,
    794,
    1061,
    3021,
    752,
    599,
    788,
    1188,
    1067,
    1290,
    767,
    357,
    320,
    2367,
    759,
    1121,
    1698,
    1742,
    2417,
    2433,
    694,
    975,
    841,
    879,
    1242,
    1031,
    854,
    1387,
    817,
    725,
    1409,
    985,
    909,
    1339,
    1442
  ],
  "warnings": []
}
```

</details>

## Embed/output weights

Some of these quants (Q3_K_XL, Q4_K_L etc) are the standard quantization method with the embeddings and output weights quantized to Q8_0 instead of what they would normally default to.

## ARM/AVX information

llama.cpp automatically "repacks" weights into an interleaved layout at load time for faster inference on ARM and AVX machines - details in [this PR](https://github.com/ggml-org/llama.cpp/pull/9921). This once required downloading special Q4_0_4_4/4_8/8_8 files; those are long gone. Online repacking now covers Q4_0, IQ4_NL, and most K-quants, so no special quant choice is needed for CPU inference.

## Which file should I choose?

<details>
  <summary>Click here for details</summary>

An older (early 2024) but still useful write-up with charts comparing quant performances is provided by Artefact2 [here](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)

The first thing to figure out is how big a model you can run. To do this, you'll need to figure out how much RAM and/or VRAM you have.

If you want your model running as FAST as possible, you'll want to fit the whole thing on your GPU's VRAM. Aim for a quant with a file size 1-2GB smaller than your GPU's total VRAM.

If you want the absolute maximum quality, add both your system RAM and your GPU's VRAM together, then similarly grab a quant with a file size 1-2GB Smaller than that total.

Hugging Face can also do this math for you: add your hardware in your [Local Apps settings](https://huggingface.co/settings/local-apps) and the model page will show which files fit.

Next, you'll need to decide if you want to use an 'I-quant' or a 'K-quant'.

If you don't want to think too much, grab one of the K-quants. These are in format 'QX_K_X', like Q5_K_M.

If you want to get more into the weeds, you can check out this extremely useful feature chart:

[llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)

But basically, if you're aiming for below Q4, and you're running cuBLAS (Nvidia) or rocBLAS (AMD), you should look towards the I-quants. These are in format IQX_X, like IQ3_M. These are newer and offer better performance for their size.

These I-quants can also be used on CPU, but will be slower than their K-quant equivalent, so speed vs performance is a tradeoff you'll have to decide.

</details>

## Credits

Thank you kalomaze and Dampf for assistance in creating the imatrix calibration dataset.

Thank you ZeroWw for the inspiration to experiment with embed/output.

Want to support my work? Visit my ko-fi page here: https://ko-fi.com/bartowski
