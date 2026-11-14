<div align="center">

# ⚡ TokenCraft

**Build, train, compare, and visualize modern subword tokenizers from scratch.**

*Pure Python &bull; Zero External Dependencies &bull; Byte-Level Lossless &bull; Golang Port &bull; Interactive Studio*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Go](https://img.shields.io/badge/Go-1.20%2B-cyan.svg)](https://golang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-39%20Python%20%2B%204%20Go%20Passed-brightgreen.svg)]()
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
4. [BPE-Dropout & Subword Regularization](#-bpe-dropout--subword-regularization)
5. [Batch Processing & Attention Masks](#-batch-processing--attention-masks)
6. [Cross-Tokenizer Diff & Boundary Overlap](#-cross-tokenizer-diff--boundary-overlap)
7. [Prefix Trie Matcher](#-prefix-trie-matcher)
8. [Token Provenance & Merge Tree Visualizer](#-token-provenance--merge-tree-visualizer)
9. [Hugging Face & OpenAI Exporters](#-hugging-face--openai-exporters)
10. [Interactive Web Studio](#-interactive-web-studio)
11. [CLI Reference & Detailed Instructions](#-cli-reference--detailed-instructions)
12. [Golang High-Performance Engine](#-golang-high-performance-engine)
13. [Python API Quickstart](#-python-api-quickstart)
14. [Benchmark Comparison](#-benchmark-comparison)
15. [Automated Sequential Commit & Push Scheduler](#-automated-sequential-commit--push-scheduler)
16. [Test Suite](#-test-suite)
17. [License](#-license)

---

## 🌟 Overview

**TokenCraft** is a complete, educational, and production-grade library to build, study, compare, and visualize subword and tokenization algorithms from the ground up:

- **Byte-Pair Encoding (BPE)**: Pure Python implementation featuring GPT-2/GPT-4 style byte-level fallback, regex pre-tokenization, priority rank merges, and 100% lossless UTF-8 roundtrip.
- **WordPiece**: BERT-style algorithm utilizing Likelihood Ratio pair scoring ($\text{count}(u, v) / (\text{count}(u) \cdot \text{count}(v))$), continuation markers (`##`), and greedy longest-match prefix tokenization.
- **Unigram (SentencePiece)**: Probabilistic language modeling algorithm utilizing Viterbi dynamic programming optimal segmentation and expectation-maximization pruning.
- **Character-Level**: Granular character decomposition with tiny vocabulary footprint.
- **Byte-Level (256)** & **Word-Level**: Essential baselines to observe the tension between sequence length and vocabulary size.
- **BPE-Dropout (Subword Regularization)**: Stochastic subword sampling (Provilkov et al., 2020) for improved LLM training robustness and out-of-domain generalization.
- **Batch Processing Pipeline**: Uniform collation with padding (`longest` vs `max_length`), left/right alignment, truncation, and 0/1 binary attention masks.
- **Cross-Tokenizer Diff Engine**: Computes exact cut boundary alignments, shared cut counts, and Jaccard boundary overlap percentages.
- **Prefix Trie Engine**: Fast subword search and greedy prefix tokenization.
- **Merge Derivation Tree**: Traces the exact hierarchy of how BPE tokens were built from elemental bytes.
- **Model Exporters**: Export trained models directly to Hugging Face `tokenizer.json` and OpenAI GPT-2 `vocab.json` + `merges.txt`.
- **Golang Port**: Standalone high-performance Go package and CLI supporting BPE, WordPiece, Character, and multi-worker concurrent batch encoding (**11M+ tokens/sec**).
- **Dual Visualizer**:
  1. **Terminal Visualizer**: Color-coded ANSI token pills, comparison tables, and diff trees directly in your terminal.
  2. **Interactive Web Dashboard**: Modern browser UI with real-time sentence splitting, token ID inspection, BPE-Dropout slider, diff matrix, and batch attention mask visualizer.

---

## 🏛 Architecture & Design

```
TokenCraft/
├── tokencraft/
│   ├── base.py              # BaseTokenizer, Token, TokenizationResult, TokenizerMetrics, encode_batch
│   ├── pretokenizers.py     # RegexPreTokenizer, Normalizer, bytes_to_unicode bijection
│   ├── bpe.py               # BPETokenizer (Byte-Pair Encoding with BPE-Dropout)
│   ├── wordpiece.py         # WordPieceTokenizer (Likelihood Ratio subwords)
│   ├── unigram.py           # UnigramTokenizer (Viterbi DP & EM pruning)
│   ├── char.py              # CharacterTokenizer
│   ├── baselines.py         # ByteTokenizer (256-byte) & WordTokenizer
│   ├── dropout.py           # BPE-Dropout stochastic subword regularization & sampling
│   ├── batch.py             # BatchProcessor, BatchEncoding, padding & attention masks
│   ├── trie.py              # Fast Prefix Trie token matcher & greedy tokenizer
│   ├── diff.py              # TokenizerDiffEngine & boundary Jaccard overlap
│   ├── comparator.py        # TokenizerComparator, span mapping, speed benchmarks
│   ├── provenance.py        # BPEProvenanceTracker & ASCII tree visualizer
│   ├── exporter.py          # Hugging Face & OpenAI GPT-2 exporters
│   ├── analytics.py         # Vocabulary statistics & Zipf distributions
│   ├── cli.py               # Rich CLI with ANSI visualizer & 12 subcommands
│   ├── server.py            # Built-in HTTP server & REST API
│   └── __main__.py          # Module entry point (python -m tokencraft)
├── golang/                  # High-performance Go implementation
│   ├── pkg/tokencraft/      # Go BPE, WordPiece, Char, Batch & tests
│   ├── cmd/tokencraft/      # Go CLI binary entrypoint with concurrency benchmark
│   └── go.mod
├── web/
│   ├── index.html           # Modern interactive UI with tabs, diffs & batch view
│   ├── styles.css           # Responsive dark theme
│   └── app.js               # Visual chips, tooltips, dropout slider & diff client
├── data/                    # General, code, and multilingual corpora
├── models/                  # Pre-trained BPE, WordPiece, Unigram, Char, Byte models
├── tests/                   # 39 passing unit tests
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

### 2. WordPiece (BERT-style)
Popularized by Google BERT (Devlin et al., 2018).
- **Likelihood Ratio Scoring**:
  $$\text{Score}(u, v) = \frac{\text{count}(u, v)}{\text{count}(u) \times \text{count}(v)}$$
  Normalizes by individual marginal frequencies, prioritizing tokens with high mutual information.
- **Segmentation**: Greedy longest-match prefix matching with continuation markers (`##`).

### 3. Unigram (SentencePiece-style)
Introduced by Kudo (2018) and utilized in SentencePiece, ALBERT, T5, and LLaMA.
- **Top-Down Pruning**: Starts with a large seed vocabulary and iteratively removes subwords that minimize the increase in corpus negative log-likelihood.
- **Optimal Cuts via Viterbi**: Given word $W$, dynamic programming finds the segmentation maximizing:
  $$P(x_1, \dots, x_m) = \prod_{i=1}^m P(x_i)$$

### 4. Character & Byte-Level Baselines

| Property | Byte-Level | Character-Level | BPE / WordPiece / Unigram | Word-Level |
| :--- | :--- | :--- | :--- | :--- |
| **Vocabulary Size** | 256 (Fixed) | ~100–500 | ~500–100,000 | ~10,000–500,000+ |
| **Sequence Length** | Longest | Long | Balanced (Optimal) | Shortest |
| **OOV Vulnerability** | **0% (Zero)** | Very Low | **0% (Byte BPE)** | Extremely High |
| **Attention Compute** | High ($O(N^2)$) | Moderate | **Lowest practical** | Low (if known) |

---

## 🎲 BPE-Dropout & Subword Regularization

Standard BPE tokenization is purely deterministic. However, language has many valid subword segmentations. Provilkov et al. (2020) introduced **BPE-Dropout**, a simple and effective regularization technique:
- At each merge step, each candidate merge rule in the word is dropped with probability $\alpha \in [0, 1)$.
- Surviving candidate pairs are merged according to rank priority.
- This produces multiple stochastic segmentations of the exact same text during training, making neural language models significantly more robust to spelling errors, suffixes, and domain shifts.

```python
from tokencraft.bpe import BPETokenizer

bpe = BPETokenizer.load("models/bpe_general.json")
text = "tokenization algorithms"

# Greedy deterministic (dropout = 0.0)
print(bpe.encode(text, dropout_rate=0.0).token_strings)

# Sample 3 stochastic variations (dropout = 0.3)
samples = bpe.sample_segmentations(text, n=3, dropout_rate=0.3, seed=42)
for s in samples:
    print(s["token_strings"])
```

---

## 📦 Batch Processing & Attention Masks

Deep learning frameworks (PyTorch, JAX, TensorFlow) require batches with identical sequence lengths. `tokencraft.batch.BatchProcessor` provides collation:

```python
from tokencraft.bpe import BPETokenizer

bpe = BPETokenizer.load("models/bpe_general.json")
texts = [
    "Hello world",
    "TokenCraft batch collation with attention masks",
]

# Batch encode with padding to longest sequence
batch = bpe.encode_batch(texts, padding="longest", max_length=16)

print("Input IDs:", batch.input_ids)
print("Attention Mask:", batch.attention_mask)
# Attention Mask: [[1, 1, ..., 0, 0], [1, 1, 1, 1, ...]]
```

Supported options:
- `padding="longest"`: Pads to longest sequence in the batch.
- `padding="max_length"`: Pads to fixed `max_length`.
- `padding_side="right"` (standard) or `padding_side="left"` (for autoregressive decoder generation).
- `truncation=True`: Safely truncates sequences exceeding `max_length`.

---

## 🔍 Cross-Tokenizer Diff & Boundary Overlap

TokenCraft includes a dedicated diff engine (`tokencraft.diff.TokenizerDiffEngine`) comparing how any two tokenizers partition the same sentence:

$$\text{Jaccard Overlap} = \frac{|\text{Boundaries}_1 \cap \text{Boundaries}_2|}{|\text{Boundaries}_1 \cup \text{Boundaries}_2|}$$

```bash
python3 -m tokencraft diff "tokenization is the foundation of modern NLP" --tok1 bpe --tok2 wordpiece
```

Output:
```
======================================================================
 Cross-Tokenizer Diff: BPE vs WordPiece
======================================================================
Input Text: "tokenization is the foundation of modern NLP"

BPE (11 tokens):
  [t] | [okenization] | [ is] | [ the] | [ foundation] | [ of] | [ modern] | [ ] | [N] | [L] | [P]

WordPiece (9 tokens):
  [tokenization] | [is] | [the] | [foundation] | [of] | [modern] | [n] | [##l] | [##p]

Boundary Analysis:
  Total Shared Cut Points:       1
  Cuts unique to BPE: 9
  Cuts unique to WordPiece: 7
  Boundary Jaccard Overlap:       5.9%
```

---

## 🌲 Prefix Trie Matcher

The Trie data structure (`tokencraft.trie.Trie`) enables sub-linear prefix queries:

```python
from tokencraft.trie import Trie

trie = Trie.from_vocab({"the": 1, "then": 2, "there": 3, "quick": 4})

# Longest prefix match
tok_str, tok_id = trie.longest_prefix_match("therefore")
# ('there', 3)

# All common prefixes
prefixes = trie.common_prefix_search("therefore")
# [('the', 1), ('there', 3)]

# Greedy segmentation
tokens = trie.tokenize_greedy("thequickthen")
# [('the', 1), ('quick', 4), ('then', 2)]
```

---

## 🌳 Token Provenance & Merge Tree Visualizer

Trace the exact hierarchical derivation of any subword token from elemental bytes:

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

Export TokenCraft models directly to industry-standard LLM formats:

```bash
# Export to standard Hugging Face tokenizer.json
python3 -m tokencraft export --model models/bpe_general.json --format hf --output exports/tokenizer.json

# Export to OpenAI GPT-2 vocab.json and merges.txt
python3 -m tokencraft export --model models/bpe_general.json --format gpt2 --output exports/gpt2/
```

---

## 🎨 Interactive Web Studio

Launch the built-in browser dashboard:
```bash
python3 -m tokencraft serve --port 8080
```
Open **`http://localhost:8080`** in any browser.

Interactive Features:
1. **Side-by-Side Splits**: Real-time visualization for BPE, WordPiece, Unigram, Character, Byte, and Word models.
2. **Hover Tooltips**: Inspect token IDs, hex byte representations, and character offset spans.
3. **BPE-Dropout Slider**: Adjust dropout rate $\alpha$ and sample alternate subword segmentations live.
4. **Cross-Tokenizer Diff Matrix**: View boundary overlap statistics and token alignment comparisons.
5. **Batch Collation Studio**: Input multi-line batches, configure max sequence length, and visualize binary attention masks.

---

## 💻 CLI Reference & Detailed Instructions

TokenCraft provides a unified CLI with 12 subcommands:

### 1. Compare Tokenizers on a Sentence
```bash
python3 -m tokencraft compare "The quick brown fox jumps over the lazy dog."
```

### 2. Cross-Tokenizer Boundary Diff
```bash
python3 -m tokencraft diff "Natural language processing with deep learning" --tok1 bpe --tok2 wordpiece
```

### 3. BPE-Dropout Stochastic Regularization
```bash
python3 -m tokencraft dropout "tokenization algorithms" --model models/bpe_general.json --dropout 0.3 --samples 5
```

### 4. Batch Processing with Attention Masks
```bash
python3 -m tokencraft batch "First sentence" "Second much longer sentence" --max-length 12
```

### 5. Train a Tokenizer from Scratch
```bash
# Train BPE
python3 -m tokencraft train --type bpe --corpus data/corpus_general.txt --vocab-size 500 --save models/bpe.json

# Train WordPiece
python3 -m tokencraft train --type wordpiece --corpus data/corpus_general.txt --vocab-size 500 --save models/wordpiece.json

# Train Unigram
python3 -m tokencraft train --type unigram --corpus data/corpus_general.txt --vocab-size 400 --save models/unigram.json
```

### 6. Encode & Decode Text
```bash
# Encode
python3 -m tokencraft encode --model models/bpe_general.json "Hello world"

# Decode
python3 -m tokencraft decode --model models/bpe_general.json "87, 283, 35, 108"
```

### 7. Trace BPE Derivation Tree
```bash
python3 -m tokencraft trace "token" --model models/bpe_general.json
```

### 8. Analyze Vocabulary Statistics
```bash
python3 -m tokencraft analyze-vocab --model models/bpe_general.json
```

### 9. Speed & Throughput Benchmark
```bash
python3 -m tokencraft benchmark
```

---

## 🐹 Golang High-Performance Engine

TokenCraft includes a native Go implementation in `golang/`:

### Run Go Tokenizer CLI
```bash
cd golang

# BPE Tokenization
go run cmd/tokencraft/main.go -model ../models/bpe_general.json -type bpe -encode "Hello world"

# WordPiece Tokenization
go run cmd/tokencraft/main.go -model ../models/wordpiece_general.json -type wordpiece -encode "Hello world"

# Character Tokenization
go run cmd/tokencraft/main.go -model ../models/char_general.json -type char -encode "Hello world"
```

### Run Go Speed & Concurrent Concurrency Benchmark
```bash
cd golang
go run cmd/tokencraft/main.go -model ../models/bpe_general.json -benchmark -workers 8
```

Benchmark output:
```
--- Single-Threaded Benchmark (100,000 iterations) ---
Elapsed:      1.54s
Throughput:   4,339,058 tokens/sec

--- Concurrent Batch Benchmark (4 workers, 100,000 sentences) ---
Elapsed:      600ms
Throughput:   11,163,401 tokens/sec (Speedup: 2.57x)
```

---

## 🐍 Python API Quickstart

```python
from tokencraft import (
    BPETokenizer,
    WordPieceTokenizer,
    UnigramTokenizer,
    TokenizerComparator,
    TokenizerDiffEngine,
    Trie,
)

# 1. Train BPE and WordPiece from scratch
corpus = ["TokenCraft bridges raw text and neural network embeddings losslessly."]
bpe = BPETokenizer()
bpe.train(corpus, vocab_size=300)

wp = WordPieceTokenizer()
wp.train(corpus, vocab_size=100)

# 2. Encode text
res = bpe.encode("TokenCraft is fast!")
print("Tokens:", res.token_strings)
print("Compression:", res.compression_ratio, "bytes/token")

# 3. BPE-Dropout sampling
samples = bpe.sample_segmentations("TokenCraft is fast!", n=3, dropout_rate=0.25)

# 4. Batch processing with attention masks
batch = bpe.encode_batch(["Hello world", "TokenCraft batching"], padding="longest")
print("Input IDs:", batch.input_ids)
print("Attention Mask:", batch.attention_mask)

# 5. Cross-tokenizer boundary diff
diff = TokenizerDiffEngine.compare(bpe, wp, "TokenCraft bridges text and embeddings")
print("Boundary Overlap:", diff.boundary_overlap_score * 100, "%")
```

---

## 📊 Benchmark Comparison

Benchmarked on Apple M-series across 50 iterations:

| Tokenizer | Vocab Size | Encode Latency | Throughput (Tokens/sec) | Implementation |
| :--- | :---: | :---: | :---: | :---: |
| **Go BPE (Concurrent 4 workers)** | 400 | **0.006 ms** | **11,163,400 tok/s** | Go |
| **Go BPE (Single-threaded)** | 400 | **0.015 ms** | **4,339,050 tok/s** | Go |
| **Character-Level** | 59 | 0.029 ms | 3,478,140 tok/s | Python |
| **WordPiece (BERT)** | 134 | 0.129 ms | 425,600 tok/s | Python |
| **BPE (Byte-Pair)** | 300 | 0.203 ms | 351,970 tok/s | Python |
| **Unigram (SentencePiece)** | 300 | 0.245 ms | 295,400 tok/s | Python |

---

## ⏱ Automated Sequential Commit & Push Scheduler

TokenCraft includes an automated Git publisher (`scripts/push_sequentially.py`) that pushes commits to remote tracking branches one-by-one with randomized 30–45 minute delays:

```bash
# Push commits sequentially with randomized 30-45 minute intervals
python3 scripts/push_sequentially.py --interval-min 30 --interval-max 45

# Push one commit right now
python3 scripts/push_sequentially.py --step-once

# Dry-run inspection
python3 scripts/push_sequentially.py --dry-run

# Push all unpushed commits immediately
python3 scripts/push_sequentially.py --push-all
```

---

## 🧪 Test Suite

Run the full automated test suite (39 Python tests + 4 Go tests covering BPE, WordPiece, Unigram, Dropout, Batch, Trie, Diff, Provenance, and Exporters):

```bash
# Python test suite
python3 -m unittest discover tests

# Go test suite
cd golang && go test -v ./...
```

Output:
```
.......................................
----------------------------------------------------------------------
Ran 39 tests in 0.034s

OK
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
