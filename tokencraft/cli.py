"""
Command-Line Interface and Terminal Visualizer for TokenCraft.
Provides ANSI colored token visualization, training, side-by-side comparison,
provenance tracing, model exporting, and interactive server commands.
"""

from __future__ import annotations
import argparse
import sys
import os
import json
from typing import List

from tokencraft.bpe import BPETokenizer
from tokencraft.wordpiece import WordPieceTokenizer
from tokencraft.unigram import UnigramTokenizer
from tokencraft.char import CharacterTokenizer
from tokencraft.baselines import ByteTokenizer, WordTokenizer
from tokencraft.comparator import TokenizerComparator
from tokencraft.provenance import BPEProvenanceTracker
from tokencraft.exporter import HuggingFaceExporter, OpenAIGPT2Exporter
from tokencraft.analytics import VocabularyAnalytics


# ANSI color codes for alternating token pill visualization in terminal
COLORS = [
    "\033[44;37m",  # Blue bg, White text
    "\033[42;30m",  # Green bg, Black text
    "\033[43;30m",  # Yellow bg, Black text
    "\033[45;37m",  # Magenta bg, White text
    "\033[46;30m",  # Cyan bg, Black text
    "\033[41;37m",  # Red bg, White text
    "\033[100;37m", # Dark gray bg, White text
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def format_colored_tokens(tokens: List[str]) -> str:
    """Renders tokens with alternating background colors for distinct visual boundaries."""
    colored_pieces = []
    for i, tok in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        # Escape newlines or control chars for clean terminal display
        safe_tok = tok.replace("\n", "\\n").replace("\t", "\\t")
        colored_pieces.append(f"{color} {safe_tok} {RESET}")
    return " ".join(colored_pieces)


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print an aligned ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(f"{BOLD}{header_line}{RESET}")
    print(sep_line)
    for row in rows:
        print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)))


def cmd_compare(args):
    """Side-by-side comparison of tokenizers on a sentence."""
    sentence = args.sentence

    print(f"\n{BOLD}======================================================================{RESET}")
    print(f"{BOLD} TokenCraft Tokenizer Comparison{RESET}")
    print(f"{BOLD}======================================================================{RESET}")
    print(f"{DIM}Input:{RESET} {BOLD}\"{sentence}\"{RESET}\n")

    corpus = [
        "Tokenization is the foundation of modern Large Language Models and natural language processing.",
        "Byte Pair Encoding merges frequent pairs of bytes or characters iteratively into subwords.",
        "WordPiece uses a likelihood score to maximize training data probability with continuation markers.",
        "Unigram language modeling prunes candidate substrings using Viterbi dynamic programming segmentation.",
        "Character tokenization splits strings into individual letters and glyphs with small vocabulary.",
        "Deep learning architectures like GPT, BERT, LLaMA, and Claude rely on fast tokenization.",
        sentence,
    ]

    bpe = BPETokenizer(name="BPE (GPT-style)")
    bpe.train(corpus, vocab_size=320)

    wp = WordPieceTokenizer(name="WordPiece (BERT-style)")
    wp.train(corpus, vocab_size=320)

    uni = UnigramTokenizer(name="Unigram (SentencePiece)")
    uni.train(corpus, vocab_size=320)

    char_tok = CharacterTokenizer(name="Character-Level")
    char_tok.train(corpus, vocab_size=150)

    byte_tok = ByteTokenizer(name="Byte-Level (256)")
    word_tok = WordTokenizer(name="Word-Level")
    word_tok.train(corpus, vocab_size=200)

    comparator = TokenizerComparator([bpe, wp, uni, char_tok, byte_tok, word_tok])
    res = comparator.compare_sentence(sentence)

    # Print Visual Split Breakdown
    print(f"{BOLD}--- Visual Split Breakdown ---{RESET}")
    for name, tokens in res["splits"].items():
        tok_texts = [t["text"] for t in tokens]
        colored_repr = format_colored_tokens(tok_texts)
        print(f"\n{BOLD}{name}{RESET} ({len(tokens)} tokens):")
        print(f"  {colored_repr}")
        tok_ids_str = ", ".join(str(t["id"]) for t in tokens)
        print(f"  {DIM}IDs: [{tok_ids_str}]{RESET}")

    # Print Metrics Table
    print(f"\n{BOLD}--- Comparative Metrics ---{RESET}")
    headers = ["Tokenizer", "Vocab Size", "Tokens", "Bytes/Token", "Tokens/Word", "OOV %"]
    rows = []
    for r in res["table"]:
        rows.append([
            r["Tokenizer"],
            str(r["Vocab Size"]),
            str(r["Token Count"]),
            r["Compression Ratio"],
            r["Fertility"],
            r["OOV Rate"],
        ])
    print_table(headers, rows)
    print()


def cmd_train(args):
    """Train a tokenizer and save to file."""
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()

    corpus = text.splitlines()
    tok_type = args.type.lower()
    print(f"Training {tok_type.upper()} tokenizer with target vocab size {args.vocab_size}...")

    if tok_type == "bpe":
        tokenizer = BPETokenizer(name=f"BPE-{args.vocab_size}")
        tokenizer.train(corpus, vocab_size=args.vocab_size, show_progress=True)
    elif tok_type == "wordpiece":
        tokenizer = WordPieceTokenizer(name=f"WordPiece-{args.vocab_size}")
        tokenizer.train(corpus, vocab_size=args.vocab_size, show_progress=True)
    elif tok_type == "unigram":
        tokenizer = UnigramTokenizer(name=f"Unigram-{args.vocab_size}")
        tokenizer.train(corpus, vocab_size=args.vocab_size, show_progress=True)
    elif tok_type in ("char", "character"):
        tokenizer = CharacterTokenizer(name=f"Char-{args.vocab_size}")
        tokenizer.train(corpus, vocab_size=args.vocab_size)
    elif tok_type == "word":
        tokenizer = WordTokenizer(name=f"Word-{args.vocab_size}")
        tokenizer.train(corpus, vocab_size=args.vocab_size)
    else:
        print(f"Unknown tokenizer type: {args.type}", file=sys.stderr)
        sys.exit(1)

    tokenizer.save(args.save)
    print(f"Successfully trained and saved model to: {args.save} (Vocab size: {tokenizer.vocab_size})")


def cmd_encode(args):
    """Encode text using a saved model."""
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    with open(args.model, "r", encoding="utf-8") as f:
        data = json.load(f)

    tok_type = data.get("type", "BPETokenizer")
    if tok_type == "BPETokenizer":
        tokenizer = BPETokenizer.from_dict(data)
    elif tok_type == "WordPieceTokenizer":
        tokenizer = WordPieceTokenizer.from_dict(data)
    elif tok_type == "UnigramTokenizer":
        tokenizer = UnigramTokenizer.from_dict(data)
    elif tok_type == "CharacterTokenizer":
        tokenizer = CharacterTokenizer.from_dict(data)
    elif tok_type == "WordTokenizer":
        tokenizer = WordTokenizer.from_dict(data)
    elif tok_type == "ByteTokenizer":
        tokenizer = ByteTokenizer.from_dict(data)
    else:
        raise ValueError(f"Unknown tokenizer type {tok_type}")

    res = tokenizer.encode(args.text)
    print(f"{BOLD}Tokens:{RESET} {format_colored_tokens(res.token_strings)}")
    print(f"{BOLD}Token IDs:{RESET} {res.token_ids}")
    print(f"{BOLD}Stats:{RESET} {res.token_count} tokens | Compression: {res.compression_ratio} B/tok | Fertility: {res.fertility} tok/wd")


def cmd_decode(args):
    """Decode IDs using a saved model."""
    with open(args.model, "r", encoding="utf-8") as f:
        data = json.load(f)

    tok_type = data.get("type", "BPETokenizer")
    if tok_type == "BPETokenizer":
        tokenizer = BPETokenizer.from_dict(data)
    elif tok_type == "WordPieceTokenizer":
        tokenizer = WordPieceTokenizer.from_dict(data)
    elif tok_type == "UnigramTokenizer":
        tokenizer = UnigramTokenizer.from_dict(data)
    elif tok_type == "CharacterTokenizer":
        tokenizer = CharacterTokenizer.from_dict(data)
    else:
        tokenizer = BPETokenizer.from_dict(data)

    ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
    decoded_text = tokenizer.decode(ids)
    print(f"{BOLD}Decoded Text:{RESET} {decoded_text}")


def cmd_trace(args):
    """Trace the merge derivation tree of a BPE token."""
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bpe = BPETokenizer.from_dict(data)
    tracker = BPEProvenanceTracker(bpe.merges)
    tree_str = tracker.trace_token(args.token)
    print(f"\n{BOLD}======================================================================{RESET}")
    print(f"{BOLD} TokenCraft Merge Derivation Tree{RESET}")
    print(f"{BOLD}======================================================================{RESET}")
    print(tree_str)


def cmd_export(args):
    """Export BPE tokenizer to HuggingFace or OpenAI formats."""
    with open(args.model, "r", encoding="utf-8") as f:
        data = json.load(f)
    bpe = BPETokenizer.from_dict(data)

    fmt = args.format.lower()
    if fmt in ("hf", "huggingface"):
        HuggingFaceExporter.export_tokenizer_json(bpe, args.output)
        print(f"Exported Hugging Face tokenizer to: {args.output}")
    elif fmt in ("gpt2", "openai"):
        OpenAIGPT2Exporter.export(bpe, args.output)
        print(f"Exported GPT-2 vocab.json and merges.txt to directory: {args.output}")
    else:
        print(f"Unknown format: {args.format}", file=sys.stderr)
        sys.exit(1)


def cmd_analyze_vocab(args):
    """Analyze tokenizer vocabulary statistics and length distributions."""
    with open(args.model, "r", encoding="utf-8") as f:
        data = json.load(f)

    tok_type = data.get("type", "BPETokenizer")
    if tok_type == "BPETokenizer":
        tok = BPETokenizer.from_dict(data)
    elif tok_type == "WordPieceTokenizer":
        tok = WordPieceTokenizer.from_dict(data)
    elif tok_type == "UnigramTokenizer":
        tok = UnigramTokenizer.from_dict(data)
    else:
        tok = BPETokenizer.from_dict(data)

    stats = VocabularyAnalytics.analyze(tok)
    print(f"\n{BOLD}======================================================================{RESET}")
    print(f"{BOLD} Vocabulary Analysis: {stats['name']}{RESET}")
    print(f"{BOLD}======================================================================{RESET}")
    print(f"Vocab Size:         {stats['vocab_size']}")
    print(f"Avg Token Length:   {stats['avg_token_length']} chars")
    print(f"Longest Token:      \"{stats['longest_token']}\" ({stats['max_token_length']} chars)")
    print(f"Single Characters:  {stats['single_char_tokens']} ({stats['single_char_ratio']}%)")
    print(f"Multi-Char Subwords:{stats['multi_char_tokens']} ({stats['multi_char_ratio']}%)")
    print(f"ASCII vs Unicode:   {stats['ascii_tokens']} ASCII / {stats['unicode_tokens']} Unicode")

    print(f"\n{BOLD}Length Distribution:{RESET}")
    headers = ["Bucket", "Token Count", "Ratio"]
    rows = []
    total = max(1, stats["vocab_size"])
    for k, v in sorted(stats["length_distribution"].items(), key=lambda x: str(x[0])):
        rows.append([f"{k} chars", str(v), f"{(v/total)*100:.1f}%"])
    print_table(headers, rows)
    print()


def cmd_serve(args):
    """Launch the Web Visualizer server."""
    from tokencraft.server import run_server
    run_server(port=args.port)


def cmd_benchmark(args):
    """Run performance benchmarks across tokenizers."""
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning tokenization bridges continuous embeddings and discrete vocabulary units.",
        "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
        "Multilingual tokenization handles different scripts: 日本語, Español, Français, and emoji 🚀🔥✨.",
        "Supercalifragilisticexpialidocious and Donaudampfschifffahrtselektrizitätenhauptbetriebswerkbauunterbeamtengesellschaft.",
    ]
    bpe = BPETokenizer(name="BPE")
    bpe.train(sentences, vocab_size=300)
    wp = WordPieceTokenizer(name="WordPiece")
    wp.train(sentences, vocab_size=300)
    uni = UnigramTokenizer(name="Unigram")
    uni.train(sentences, vocab_size=300)
    ch = CharacterTokenizer(name="Char")
    ch.train(sentences, vocab_size=150)
    by = ByteTokenizer(name="Byte")

    comparator = TokenizerComparator([bpe, wp, uni, ch, by])
    print(f"\n{BOLD}Running Speed & Throughput Benchmark...{RESET}")
    bench = comparator.benchmark_speed(sentences, iterations=50)

    headers = ["Tokenizer", "Vocab Size", "Enc Latency (ms)", "Dec Latency (ms)", "Throughput (tok/s)", "Throughput (KB/s)"]
    rows = []
    for name, stats in bench.items():
        rows.append([
            name,
            str(stats["vocab_size"]),
            str(stats["encode_latency_ms"]),
            str(stats["decode_latency_ms"]),
            str(stats["encode_throughput_tokens_sec"]),
            str(stats["encode_throughput_kb_sec"]),
        ])
    print_table(headers, rows)
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="tokencraft",
        description="TokenCraft - Build, train, compare, and visualize tokenizers from scratch",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare tokenizers on a sentence with visual split")
    p_comp.add_argument("sentence", help="Sentence to analyze and tokenize")
    p_comp.set_defaults(func=cmd_compare)

    # train
    p_train = subparsers.add_parser("train", help="Train a tokenizer on a corpus file")
    p_train.add_argument("--type", choices=["bpe", "wordpiece", "unigram", "char", "word"], default="bpe", help="Tokenizer algorithm")
    p_train.add_argument("--corpus", required=True, help="Path to text corpus file")
    p_train.add_argument("--vocab-size", type=int, default=1000, help="Target vocabulary size")
    p_train.add_argument("--save", required=True, help="Output JSON model file path")
    p_train.set_defaults(func=cmd_train)

    # encode
    p_enc = subparsers.add_parser("encode", help="Encode text using a saved model")
    p_enc.add_argument("--model", required=True, help="Path to saved tokenizer JSON model")
    p_enc.add_argument("text", help="Text to tokenize")
    p_enc.set_defaults(func=cmd_encode)

    # decode
    p_dec = subparsers.add_parser("decode", help="Decode IDs using a saved model")
    p_dec.add_argument("--model", required=True, help="Path to saved tokenizer JSON model")
    p_dec.add_argument("ids", help="Comma-separated token IDs, e.g. '12,45,67'")
    p_dec.set_defaults(func=cmd_decode)

    # trace
    p_trace = subparsers.add_parser("trace", help="Trace the merge derivation tree of a BPE token")
    p_trace.add_argument("token", help="Token to trace, e.g. 'tokenization'")
    p_trace.add_argument("--model", default="models/bpe_general.json", help="Path to BPE model JSON")
    p_trace.set_defaults(func=cmd_trace)

    # export
    p_exp = subparsers.add_parser("export", help="Export BPE tokenizer to HuggingFace or OpenAI formats")
    p_exp.add_argument("--model", required=True, help="Path to BPE model JSON")
    p_exp.add_argument("--format", choices=["hf", "gpt2"], default="hf", help="Target export format")
    p_exp.add_argument("--output", required=True, help="Output path (file for HF, directory for GPT-2)")
    p_exp.set_defaults(func=cmd_export)

    # analyze-vocab
    p_ana = subparsers.add_parser("analyze-vocab", help="Analyze vocabulary length distribution and statistics")
    p_ana.add_argument("--model", required=True, help="Path to tokenizer JSON model")
    p_ana.set_defaults(func=cmd_analyze_vocab)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark speed and throughput across tokenizers")
    p_bench.set_defaults(func=cmd_benchmark)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch interactive Web Visualizer dashboard")
    p_serve.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    p_serve.set_defaults(func=cmd_serve)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
