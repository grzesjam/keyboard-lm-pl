# FUTO Keyboard — Polish Transformer Model

Scripts to train a Polish-language transformer model for [FUTO Keyboard](https://keyboard.futo.org).
FUTO Keyboard ships an English-only next-word-prediction / autocorrect model; this project builds the Polish equivalent.

**Pretrained models:** [Releases](https://github.com/jblechert/keyboard-lm-de/releases) — drop the `.gguf` into FUTO Keyboard → Settings → Language Models.

---

## Status

This is a Polish adaptation of the original German training pipeline.
All German-specific data sources (podcasts, etc.) have been replaced with Polish equivalents.

---

## Architecture

| Parameter | Value |
|---|---|
| Architecture | Llama (GGUF via llama.cpp) |
| Parameters | **57M** |
| Layers | 10 × 512 hidden dims, 8 attention heads, 2048 FFN |
| Context | 256 tokens |
| Tokenizer | SentencePiece BPE, `treat_whitespace_as_suffix=true` |

Special autocorrect tokens: `<XBU>`, `<CHAR_A>`…`<CHAR_Z>`, `<XBC>`, `<XEC>`

---

## Training data

| Source | Sentences | Weight | License |
|---|---|---|---|---|
| [Tatoeba PL](https://tatoeba.org) | ~770k | 3× | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| [mC4 PL](https://huggingface.co/datasets/allenai/c4) (allenai/c4) | 80M | 1× | [ODC-By](https://opendatacommons.org/licenses/by/) |
| [FineWeb2-HQ PL](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | ~68M | 1× | [ODC-By](https://opendatacommons.org/licenses/by/) |
| Synthetic (Qwen3.6:27b, 27 topics) | ~61k | 3× | generated, non-commercial |

---

## Pipeline

### 1. Download training data

```bash
# Tatoeba
.venv_ml/bin/python 07_download_tatoeba.py

# mC4 (80M sentences, ~12h)
.venv_ml/bin/python 09_download_c4_de.py --target 80000000

# FineWeb2-HQ PL (~68M sentences, ~several hours)
.venv_ml/bin/python 24_download_fineweb2_de.py
```

### 2. Clean training data

```bash
.venv_ml/bin/python 10_clean_training_data.py
```

### 3. Generate synthetic sentences (requires Ollama + qwen3.6:27b)

```bash
.venv_ml/bin/python 12_generate_synthetic_vocab.py --per-topic 2000
```

### 4. Train SentencePiece tokenizer

```bash
.venv_ml/bin/python 04_train_tokenizer.py
```

### 5. Train the model

```bash
.venv_ml/bin/python 05_train_model.py --steps 150000 --version v1.0
# ~30 hours on RX 7900 XTX (ROCm)
```

### 6. Convert to GGUF

```bash
.venv_ml/bin/python 06_convert_to_gguf.py
```

---

## Quality metrics

Results will be reported once training completes.

---

## Requirements

```bash
python -m venv --system-site-packages .venv_ml
.venv_ml/bin/pip install "tokenizers==0.21.0" "transformers>=4.49,<5" datasets gguf sentencepiece
```

- Python 3.10+
- PyTorch with CUDA (NVIDIA) or ROCm (AMD, Linux)
- `bf16=True` requires NVIDIA Ampere (RTX 30xx+) or AMD ROCm — for older cards use `fp16=True`

### Hardware

| GPU | VRAM | 80k steps | 150k steps |
|-----|------|-----------|------------|
| RX 7900 XTX / RTX 4090 | 24 GB | ~16 h | ~30 h |
| RTX 3080 / RTX 4070 | 10–12 GB | ~22 h | ~42 h |
| RTX 3060 12 GB | 12 GB | ~44 h | ~82 h |

---

## License

**Code:** [MIT License](LICENSE)

**Model weights:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for non-commercial use with attribution.

### Training data attribution

| Source | Authors | License |
|---|---|---|
| [Tatoeba PL](https://tatoeba.org) | Tatoeba contributors | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |
| [mC4 PL](https://huggingface.co/datasets/allenai/c4) | Common Crawl / Allen AI | [ODC-By](https://opendatacommons.org/licenses/by/) |
| [FineWeb2-HQ PL](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Common Crawl / HuggingFace FineData | [ODC-By](https://opendatacommons.org/licenses/by/) |
| Synthetic sentences | generated via [Qwen3](https://huggingface.co/Qwen) (Ollama) | non-commercial |

---

## Keyboard Collector App

[`keyboard-collector/`](keyboard-collector/) — Android app to collect correction data for fine-tuning.

Users report misrecognitions (e.g. keyboard predicted "Mahd" instead of "naja") with optional context. Data exports as JSON:

```json
{
  "cases": [
    {
      "recognized": "Mahd",
      "expected": "naja",
      "context": "WhatsApp",
      "ts": "2026-05-31T16:00:00"
    }
  ]
}
```

The exported JSON feeds into a fine-tuning pipeline on top of the trained base model.

**Install:** sideload [`keyboard-collector-debug.apk`](keyboard-collector/keyboard-collector-debug.apk) (debug build, Android 8.0+)

---

## References

- [FUTO Keyboard source](https://github.com/futo-org/android-keyboard)
- [FUTO LM documentation](https://gitlab.futo.org/keyboard/keyboard-wiki/-/wikis/Keyboard-LM-docs)
- [Issue #1212 — Transformer Models for More Languages](https://github.com/futo-org/android-keyboard/issues/1212)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
