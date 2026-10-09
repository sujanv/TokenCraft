<div align="center">

# ⚡ TokenCraft

**Build, train, compare, and visualize modern subword tokenizers from scratch.**

*Pure Python &bull; Zero External Dependencies &bull; Byte-Level Lossless &bull; Golang Port &bull; Interactive Studio*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Go](https://img.shields.io/badge/Go-1.20%2B-cyan.svg)](https://golang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-27%20Passed-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

</div>

---

## 📖 Table of Contents
1. [Overview](#-overview)
2. [Architecture & Design](#-architecture--design)
3. [Tokenization Theory & Mathematical Foundations](#-tokenization-theory--mathematical-foundations)
   - [Byte-Pair Encoding (BPE)](#1-byte-pair-encoding-bpe)
   - [WordPiece (BERT-style)](#2-wordpiece-bert-style)
   - [Unigram (SentencePiece-style)](#3-unigram-sentencepiece-style)
   - [Character & Byte-Level Baselines](#4-character--byte-level-baselines)
4. [Token Provenance & Merge Tree Visualizer](#-token-provenance--merge-tree-visualizer)
5. [Hugging Face & OpenAI Exporters](#-hugging-face--openai-exporters)
6. [Interactive Web Studio](#-interactive-web-studio)
7. [CLI Usage](#-cli-usage)
8. [Golang High-Performance Engine](#-golang-high-performance-engine)
9. [Python API Quickstart](#-python-api-quickstart)
10. [Benchmark Comparison](#-benchmark-comparison)
11. [Automated Commit & Push Scheduler](#-automated-commit--push-scheduler)
12. [Test Suite](#-test-suite)
13. [License](#-license)

---

## 🌟 Overview

**TokenCraft** is a complete, educational, and production-grade library to build, study, compare, and visualize subword and tokenization algorithms from the ground up:

- **Byte-Pair Encoding (BPE)**: Pure Python implementation featuring GPT-2/GPT-4 style byte-level fallback, regex pre-tokenization, priority rank merges, and 100% lossless UTF-8 roundtrip.
- **WordPiece**: BERT-style algorithm utilizing Likelihood Ratio pair scoring ($\text{count}(u, v) / (\text{count}(u) \cdot \text{count}(v))$), continuation markers (`##`), and greedy longest-match prefix tokenization.
- **Unigram (SentencePiece)**: Probabilistic language modeling algorithm utilizing Viterbi dynamic programming optimal segmentation and expectation-maximization pruning.
- **Character-Level**: Granular character decomposition with tiny vocabulary footprint.
- **Byte-Level (256)** & **Word-Level**: Essential baselines to observe the tension between sequence length and vocabulary size.
- **Merge Derivation Tree**: Traces the exact hierarchy of how BPE tokens were built from elemental bytes.
- **Model Exporters**: Export trained models directly to Hugging Face `tokenizer.json` and OpenAI GPT-2 `vocab.json` + `merges.txt`.
- **Golang Port**: Standalone Go implementation (`golang/`) for ultra-low-latency deployment.
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
│   ├── unigram.py           # UnigramTokenizer (Viterbi DP & EM pruning)
│   ├── char.py              # CharacterTokenizer
│   ├── baselines.py         # ByteTokenizer (256-byte) & WordTokenizer
│   ├── comparator.py        # TokenizerComparator, span mapping, speed benchmarks
│   ├── provenance.py        # BPEProvenanceTracker & ASCII tree visualizer
│   ├── exporter.py          # Hugging Face & OpenAI GPT-2 exporters
│   ├── analytics.py         # Vocabulary statistics & Zipf distributions
│   ├── cli.py               # Rich CLI with ANSI visualizer & subcommands
│   ├── server.py            # Built-in HTTP server & REST API
│   └── __main__.py          # Module entry point (python -m tokencraft)
├── golang/                  # High-performance Go implementation
│   ├── pkg/tokencraft/      # Go BPE package & tests
│   ├── cmd/tokencraft/      # Go CLI binary entrypoint
│   └── go.mod
├── web/
│   ├── index.html           # Modern interactive UI
│   ├── styles.css           # Responsive dark theme
│   └── app.js               # Visual chips, tooltips & interactive client
├── data/                    # General, code, and multilingual corpora
├── models/                  # Pre-trained BPE, WordPiece, Unigram, Char, Byte models
├── tests/                   # 27 passing unit tests
├── scripts/
│   ├── train_models.py      # Retrain/export bundled models
│   └── push_sequentially.py # Automated sequential Git commit pusher
└── pyproject.toml & setup.py
```

---

## 🔬 Tokenization Theory & Mathematical Foundations

### 1. Byte-Pair Encoding (BPE)
BPE was adapted for NLP by Sennrich et al. (2015) and extended to byte-level BPE by OpenAI (GPT-2, GPT-3, GPT-4, Tiktoken).
- **Initialization**: 256 byte-unicode characters via `bytes_to_unicode()`.
- **Merge Criterion**: Selects the pair $(u, v)$ with highest co-occurrence frequency across the corpus.
- **Lossless Guarantee**: Every byte $0..255$ has an entry, preventing out-of-vocabulary `<unk>` tokens on arbitrary Unicode or binary data.

---

### 2. WordPiece (BERT-style)
Popularized by Google BERT (Devlin et al., 2018).
- **Likelihood Ratio Scoring**:
  $$\text{Score}(u, v) = \frac{\text{count}(u, v)}{\text{count}(u) \times \text{count}(v)}$$
  Normalizes by individual marginal frequencies, prioritizing tokens with high mutual information.
- **Segmentation**: Greedy longest-match prefix matching with continuation markers (`##`).

---

### 3. Unigram (SentencePiece-style)
Introduced by Kudo (2018) and utilized in SentencePiece, ALBERT, T5, and LLaMA.
- **Top-Down Pruning**: Starts with a large seed vocabulary and iteratively removes subwords that minimize the increase in corpus negative log-likelihood.
- **Optimal Cuts via Viterbi**: Given word $W$, dynamic programming finds the segmentation maximizing:
  $$P(x_1, \dots, x_m) = \prod_{i=1}^m P(x_i)$$

---

### 4. Character & Byte-Level Baselines

| Property | Byte-Level | Character-Level | BPE / WordPiece / Unigram | Word-Level |
| :--- | :--- | :--- | :--- | :--- |
| **Vocabulary Size** | 256 (Fixed) | ~100–500 | ~500–100,000 | ~10,000–500,000+ |
| **Sequence Length** | Longest | Long | Balanced (Optimal) | Shortest |
| **OOV Vulnerability** | **0% (Zero)** | Very Low | **0% (Byte BPE)** | Extremely High |
| **Attention Compute** | High ($O(N^2)$) | Moderate | **Lowest practical** | Low (if known) |

---

## 🌳 Token Provenance & Merge Tree Visualizer

TokenCraft allows tracing the exact hierarchical derivation of any subword token:

```bash
python3 -m tokencraft trace in --model models/bpe_general.json
```

Output:
```
======================================================================
 TokenCraft Merge Derivation Tree
======================================================================
Derivation Tree for: 'in'
[in] (Merge #0)
├── 'i'
└── 'n'
```

---

## 🚀 Hugging Face & OpenAI Exporters

Export trained TokenCraft tokenizers directly to modern LLM formats:

```bash
# Export to standard Hugging Face tokenizer.json
python3 -m tokencraft export --model models/bpe_general.json --format hf --output exports/tokenizer.json

# Export to OpenAI GPT-2 vocab.json and merges.txt
python3 -m tokencraft export --model models/bpe_general.json --format gpt2 --output exports/gpt2/
```

---

## 🎨 Interactive Web Studio

Launch the browser studio:
```bash
python3 -m tokencraft serve --port 8080
```
Then visit **`http://localhost:8080`**.

Features:
- Side-by-side split visualization across **BPE, WordPiece, Unigram, Character, Byte, and Word**.
- Hover tooltips displaying token IDs, byte representations, and character offsets.
- Preset sentences: English, Python code, Contractions, Multilingual glyphs, and Long compounds.

---

## 💻 CLI Usage

```bash
# Compare all algorithms side-by-side
python3 -m tokencraft compare "The quick brown fox jumps over the lazy dog."

# Analyze vocabulary statistics & distributions
python3 -m tokencraft analyze-vocab --model models/bpe_general.json

# Trace token derivation tree
python3 -m tokencraft trace "token" --model models/bpe_general.json

# Train Unigram model
python3 -m tokencraft train --type unigram --corpus data/corpus_general.txt --vocab-size 400 --save models/unigram.json

# Benchmark encoding/decoding throughput
python3 -m tokencraft benchmark
```

---

## 🐹 Golang High-Performance Engine

TokenCraft includes a native Go implementation in `golang/`:

```bash
# Run Go tokenizer CLI
cd golang
go run cmd/tokencraft/main.go -model ../models/bpe_general.json -encode "Hello world"

# Run Go benchmark
go run cmd/tokencraft/main.go -model ../models/bpe_general.json -benchmark
```

---

## 🧪 Test Suite

Run all 27 unit tests across BPE, WordPiece, Unigram, Provenance, Exporters, and Baselines:

```bash
python3 -m unittest discover tests
```

Output:
```
...........................
----------------------------------------------------------------------
Ran 27 tests in 0.022s

OK
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
