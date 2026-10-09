"""
Built-in HTTP server and REST API for the TokenCraft Interactive Web Visualizer.
Zero external dependencies, running on Python's standard http.server.
"""

from __future__ import annotations
import http.server
import socketserver
import json
import os
import sys
import urllib.parse
from typing import Dict, Any

from tokencraft.bpe import BPETokenizer
from tokencraft.wordpiece import WordPieceTokenizer
from tokencraft.unigram import UnigramTokenizer
from tokencraft.char import CharacterTokenizer
from tokencraft.baselines import ByteTokenizer, WordTokenizer
from tokencraft.comparator import TokenizerComparator
from tokencraft.provenance import BPEProvenanceTracker


WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))

SAMPLE_TEXTS = {
    "English General": "The quick brown fox jumps over the lazy dog and explores the wonders of tokenization.",
    "Python Code": "def fibonacci(n: int) -> int:\n    if n <= 1: return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "Contractions & Slang": "You'll never believe what's happening! Don't let's argue, it's gotta be done ASAP.",
    "Multilingual & Glyphs": "TokenCraft supports multilingual text: こんにちは世界, Bonjour le monde, and Привет мир!",
    "Compounds & Edge Cases": "Donaudampfschifffahrtselektrizitätenhauptbetriebswerk and supercalifragilisticexpialidocious 🚀🔥",
}

DEFAULT_CORPUS = [
    "Tokenization is the foundational bridge between continuous neural networks and discrete natural language symbols.",
    "Byte Pair Encoding (BPE) starts from byte-level tokens and iteratively merges frequent symbol pairs into subword vocabularies.",
    "WordPiece scores subword combinations by likelihood ratio, dividing co-occurrence counts by independent marginal frequencies.",
    "Unigram subword tokenization models words probabilistically and searches for optimal cuts using Viterbi dynamic programming.",
    "Character-level tokenization operates directly on individual glyphs, keeping vocabulary tiny but fertility very high.",
    "Modern language models such as GPT-4, Claude, BERT, and LLaMA heavily depend on optimized subword compression algorithms.",
    "Computer programs and code involve keywords like def, return, class, import, async, await, and indentation spaces.",
    "Multilingual sentences test byte representations across Unicode codepoints: こんにちは世界, Bonjour, Hola, Danke schön.",
]


class TokenCraftHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving web static files and JSON REST APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/samples":
            self._send_json(SAMPLE_TEXTS)
        elif parsed.path == "/api/health":
            self._send_json({"status": "ok", "version": "1.1.0"})
        elif parsed.path == "/api/trace":
            query = urllib.parse.parse_qs(parsed.query)
            token = query.get("token", [""])[0]
            # Train quick BPE on default corpus
            bpe = BPETokenizer()
            bpe.train(DEFAULT_CORPUS, vocab_size=350)
            tracker = BPEProvenanceTracker(bpe.merges)
            tree_node = tracker.build_tree(token)
            ascii_tree = tracker.trace_token(token)
            self._send_json({
                "token": token,
                "tree": tree_node.to_dict(),
                "ascii": ascii_tree,
            })
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/compare":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            sentence = data.get("sentence", SAMPLE_TEXTS["English General"])
            vocab_size = int(data.get("vocab_size", 400))

            corpus = DEFAULT_CORPUS + [sentence]

            bpe = BPETokenizer(name="BPE (Byte-Pair)")
            bpe.train(corpus, vocab_size=vocab_size)

            wp = WordPieceTokenizer(name="WordPiece (BERT)")
            wp.train(corpus, vocab_size=vocab_size)

            uni = UnigramTokenizer(name="Unigram (SentencePiece)")
            uni.train(corpus, vocab_size=vocab_size)

            char_tok = CharacterTokenizer(name="Character-Level")
            char_tok.train(corpus, vocab_size=150)

            byte_tok = ByteTokenizer(name="Byte-Level (256)")

            word_tok = WordTokenizer(name="Word-Level")
            word_tok.train(corpus, vocab_size=300)

            comparator = TokenizerComparator([bpe, wp, uni, char_tok, byte_tok, word_tok])
            comp_result = comparator.compare_sentence(sentence)

            # Include top learned BPE merges for visualization
            comp_result["bpe_merges_top"] = [
                {"pair": list(p), "rank": i} for i, p in enumerate(bpe.merges[:20])
            ]

            self._send_json(comp_result)

        elif self.path == "/api/encode":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            sentence = data.get("sentence", "")
            tok_type = data.get("type", "bpe")
            vocab_size = int(data.get("vocab_size", 400))

            corpus = DEFAULT_CORPUS + [sentence]
            if tok_type == "bpe":
                tok = BPETokenizer()
                tok.train(corpus, vocab_size=vocab_size)
            elif tok_type == "wordpiece":
                tok = WordPieceTokenizer()
                tok.train(corpus, vocab_size=vocab_size)
            elif tok_type == "unigram":
                tok = UnigramTokenizer()
                tok.train(corpus, vocab_size=vocab_size)
            elif tok_type == "char":
                tok = CharacterTokenizer()
                tok.train(corpus, vocab_size=150)
            elif tok_type == "byte":
                tok = ByteTokenizer()
            else:
                tok = WordTokenizer()
                tok.train(corpus, vocab_size=300)

            res = tok.encode(sentence)
            self._send_json(res.to_dict())

        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, payload: Dict[str, Any], status: int = 200):
        response_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format, *args):
        sys.stderr.write(f"[TokenCraft Server] {args[0]} {args[1]}\n")


def run_server(port: int = 8080):
    """Start the TokenCraft HTTP server."""
    os.makedirs(WEB_DIR, exist_ok=True)
    server_address = ("", port)
    with socketserver.TCPServer(server_address, TokenCraftHandler) as httpd:
        print(f"\n=======================================================")
        print(f" TokenCraft Web Visualizer is live!")
        print(f" Open your browser: http://localhost:{port}")
        print(f" Press Ctrl+C to terminate the server.")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()
