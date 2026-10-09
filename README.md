<div align="center">

# ⚡ TokenCraft

**Build, train, compare, and visualize modern subword tokenizers from scratch.**

*Pure Python &bull; Zero External Dependencies &bull; Byte-Level Lossless &bull; Interactive Visualizer*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-19%20Passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

</div>

---

## 📖 Table of Contents
1. [Overview](#-overview)
2. [Architecture & Design](#-architecture--design)
3. [Tokenization Theory & Mathematical Foundations](#-tokenization-theory--mathematical-foundations)
   - [Byte-Pair Encoding (BPE)](#1-byte-pair-encoding-bpe)
   - [WordPiece (BERT-style)](#2-wordpiece-bert-style)
   - [Character & Byte-Level Baselines](#3-character--byte-level-baselines)
4. [Interactive Visualizer & Web Studio](#-interactive-visualizer--web-studio)
5. [CLI Usage](#-cli-usage)
6. [Python API Quickstart](#-python-api-quickstart)
7. [Benchmark Comparison](#-benchmark-comparison)
8. [Automated Commit & Push Scheduler](#-automated-commit--push-scheduler)
9. [Test Suite](#-test-suite)
10. [License](#-license)

---

## 🌟 Overview

**TokenCraft** is a complete, educational, and production-grade library to build, study, compare, and visualize subword and tokenization algorithms from the ground up:

- **Byte-Pair Encoding (BPE)**: Pure Python implementation featuring GPT-2/GPT-4 style byte-level fallback, regex pre-tokenization, priority rank merges, and 100% lossless UTF-8 roundtrip.
- **WordPiece**: BERT-style algorithm utilizing Likelihood Ratio pair scoring ($\text{count}(u, v) / (\text{count}(u) \cdot \text{count}(v))$), continuation markers (`##`), and greedy longest-match prefix tokenization.
- **Character-Level**: Granular character decomposition with tiny vocabulary footprint.
- **Byte-Level (256)** & **Word-Level**: Essential baselines to observe the tension between sequence length and vocabulary size.
- **Side-by-Side Comparison Engine**: Computes Compression Ratio (bytes/token), Fertility (tokens/word), Out-of-Vocabulary (OOV) rate, and token alignment.
- **Dual Visualizer**:
  1. **Terminal Visualizer**: Color-coded ANSI token pills and comparison tables directly in your CLI.
  2. **Interactive Web Dashboard**: Modern browser UI with real-time sentence splitting, token ID inspection, hover byte inspection, and preset playgrounds.

---

## 🏛 Architecture & Design

```
TokenCraft/
├── tokencraft/
│   ├── base.py              # BaseTokenizer, Token, TokenizationResult, TokenizerMetrics
│   ├── pretokenizers.py     # RegexPreTokenizer, Normalizer, bytes_to_unicode bijection
│   ├── bpe.py               # BPETokenizer (Byte-Pair Encoding from scratch)
│   ├── wordpiece.py         # WordPieceTokenizer (Likelihood Ratio subwords)
│   ├── char.py              # CharacterTokenizer
│   ├── baselines.py         # ByteTokenizer (256-byte) & WordTokenizer
│   ├── comparator.py        # TokenizerComparator, span mapping, speed benchmarks
│   ├── cli.py               # Rich CLI with ANSI visualizer & subcommands
│   ├── server.py            # Built-in HTTP server & REST API
│   └── __main__.py          # Module entry point (python -m tokencraft)
├── web/
│   ├── index.html           # Modern interactive UI
│   ├── styles.css           # Responsive dark theme
│   └── app.js               # Visual chips, tooltips & interactive client
├── data/                    # General, code, and multilingual corpora
├── models/                  # Pre-trained BPE, WordPiece, Char, Byte models
├── tests/                   # 19 passing unit tests
├── scripts/
│   ├── train_models.py      # Retrain/export bundled models
│   └── push_sequentially.py # Automated sequential Git commit pusher
└── pyproject.toml & setup.py
```

---

## 🔬 Tokenization Theory & Mathematical Foundations

### 1. Byte-Pair Encoding (BPE)
BPE was originally invented as a data compression technique by Philip Gage (1994) and adapted for NLP by Sennrich et al. (2015). OpenAI extended it to **byte-level BPE** for GPT-2, GPT-3, GPT-4, and Tiktoken.

#### Training:
1. Initialize the vocabulary with the 256 unique byte values mapped to printable Unicode characters via `bytes_to_unicode()`.
2. Pre-tokenize the corpus into word chunks using regex.
3. Count the frequency of all adjacent symbol pairs $(c_i, c_{i+1})$ across the corpus.
4. Select the most frequent pair $(u, v)$ and create a new merge rule:
   $$\text{merge}(u, v) \to uv$$
5. Replace occurrences of $(u, v)$ with $uv$ throughout the vocabulary and word tuples.
6. Repeat until the target vocabulary size is reached.

#### Properties:
- **No Unknown Tokens**: Because every byte $0..255$ is in the base vocabulary, arbitrary Unicode sequences, emojis, and binary payloads are handled losslessly with zero `[UNK]`.

---

### 2. WordPiece (BERT-style)
Developed by Schuster and Nakajima (2012) and popularized in Google BERT (Devlin et al., 2018).

#### Training Scoring:
Rather than choosing the pair with the highest raw co-occurrence frequency, WordPiece optimizes the likelihood of the language model data. It merges the pair $(u, v)$ that maximizes the **Likelihood Ratio**:

$$\text{Score}(u, v) = \frac{\text{count}(u, v)}{\text{count}(u) \times \text{count}(v)}$$

This normalizes by the individual marginal frequencies: common symbols (like single letters `'e'` or `'s'`) don't dominate merges unless their specific joint probability is disproportionately high.

#### Encoding (Greedy Longest Match):
For each pre-tokenized word $W$:
1. Search for the longest prefix of $W$ present in the vocabulary.
2. If matched, the remainder of the word is searched for subwords prefixed with `##` (indicating continuation).
3. If any segment cannot be resolved, the entire word falls back to `[UNK]`.

---

### 3. Character & Byte-Level Baselines

| Property | Byte-Level | Character-Level | BPE / WordPiece | Word-Level |
| :--- | :--- | :--- | :--- | :--- |
| **Vocabulary Size** | 256 (Fixed) | ~100–500 | ~500–100,000 | ~10,000–500,000+ |
| **Sequence Length** | Longest | Long | Balanced (Optimal) | Shortest |
| **OOV Vulnerability** | **0% (Zero)** | Very Low | **0% (Byte BPE)** | Extremely High |
| **Attention Compute** | High ($O(N^2)$) | Moderate | **Lowest practical** | Low (if known) |
| **Morphology Handling** | Raw bytes | Granular | Root + Suffixes | None |

---

## 🎨 Interactive Visualizer & Web Studio

TokenCraft includes a built-in interactive web application powered by Python's standard library `http.server`.

### Launching the Web Visualizer:
```bash
python3 -m tokencraft serve --port 8080
```
Then visit **`http://localhost:8080`** in your browser.

### Key Visualizer Features:
- **Real-Time Playground**: Enter any sentence and instantly see side-by-side tokenization.
- **Color-Coded Token Chips**: Visually distinguished tokens with hover cards displaying token IDs, byte hex values, and character spans.
- **Preset Test Suite**:
  - English General
  - Python Code (`def fibonacci...`)
  - Contractions & Slang (`You'll never believe what's happening!`)
  - Multilingual & Glyphs (`こんにちは世界`, `Bonjour`, `Привет`)
  - German Compounds (`Donaudampfschifffahrtselektrizitätenhauptbetriebswerk`)
- **Metrics Table**: Instant comparison of Compression Ratio, Fertility, and OOV rate.

---

## 💻 CLI Usage

TokenCraft comes with a full-featured CLI:

### 1. Compare Tokenizers on a Sentence
```bash
python3 -m tokencraft compare "The quick brown fox jumps over the lazy dog."
```
Outputs colored ANSI token pills and comparison metrics directly in the terminal!

### 2. Train a Tokenizer
```bash
# Train BPE
python3 -m tokencraft train --type bpe --corpus data/corpus_general.txt --vocab-size 500 --save models/bpe.json

# Train WordPiece
python3 -m tokencraft train --type wordpiece --corpus data/corpus_general.txt --vocab-size 500 --save models/wordpiece.json
```

### 3. Encode and Decode Text
```bash
# Encode
python3 -m tokencraft encode --model models/bpe_general.json "Hello world"

# Decode
python3 -m tokencraft decode --model models/bpe_general.json "87, 283, 35, 108"
```

### 4. Run Speed & Throughput Benchmark
```bash
python3 -m tokencraft benchmark
```

---

## 🐍 Python API Quickstart

```python
from tokencraft import BPETokenizer, WordPieceTokenizer, CharacterTokenizer, TokenizerComparator

# 1. Train BPE from scratch
corpus = [
    "TokenCraft implements byte pair encoding and subword tokenization.",
    "BPE merges frequent adjacent byte pairs iteratively.",
]
bpe = BPETokenizer(name="MyBPE")
bpe.train(corpus, vocab_size=300)

# 2. Encode text losslessly
result = bpe.encode("TokenCraft is fast!")
print("Tokens:", result.token_strings)
print("IDs:", result.token_ids)
print("Compression:", result.compression_ratio, "bytes/token")
print("Fertility:", result.fertility, "tokens/word")

# 3. Decode back to original string
decoded = bpe.decode(result.token_ids)
assert decoded == "TokenCraft is fast!"

# 4. Compare algorithms side-by-side
wp = WordPieceTokenizer(name="MyWordPiece")
wp.train(corpus, vocab_size=300)

char_tok = CharacterTokenizer()
char_tok.train(corpus)

comparator = TokenizerComparator([bpe, wp, char_tok])
report = comparator.compare_sentence("TokenCraft bridges text and embeddings.")
print(report["table"])
```

---

## 📊 Benchmark Comparison

Benchmarked on Apple M-series / Python 3.11 with 50 iterations across diverse sentences:

| Tokenizer | Vocab Size | Encode Latency (ms) | Decode Latency (ms) | Throughput (Tokens/sec) | Throughput (KB/sec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BPE (Byte-Pair)** | 300 | 0.20 ms | 0.01 ms | **351,970 tok/s** | 496 KB/s |
| **WordPiece (BERT)** | 134 | 0.13 ms | 0.01 ms | **425,600 tok/s** | 781 KB/s |
| **Character-Level** | 59 | 0.03 ms | 0.006 ms | **3,478,140 tok/s** | 3,512 KB/s |
| **Byte-Level (256)**| 256 | 0.05 ms | 0.003 ms | **1,833,890 tok/s** | 1,790 KB/s |

---

## ⏱ Automated Commit & Push Scheduler

TokenCraft includes a built-in sequential Git publisher in `scripts/push_sequentially.py` that can push commits to GitHub sequentially one-by-one with randomized 30–45 minute delays:

```bash
# Push commits one at a time with randomized 30-45 minute intervals
python3 scripts/push_sequentially.py --interval-min 30 --interval-max 45

# Push one commit right now (immediate step)
python3 scripts/push_sequentially.py --step-once

# Dry-run inspection
python3 scripts/push_sequentially.py --dry-run
```

---

## 🧪 Test Suite

Run the full automated test suite (19 unit tests covering BPE, WordPiece, Character, Normalizers, and lossless roundtrip):

```bash
python3 -m unittest discover tests
```

Output:
```
...................
----------------------------------------------------------------------
Ran 19 tests in 0.015s

OK
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
