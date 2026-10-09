"""
Script to train and export pre-trained TokenCraft models on bundled datasets.
"""

from __future__ import annotations
import os
import sys

# Ensure repository root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from tokencraft.bpe import BPETokenizer
from tokencraft.wordpiece import WordPieceTokenizer
from tokencraft.unigram import UnigramTokenizer
from tokencraft.char import CharacterTokenizer
from tokencraft.baselines import ByteTokenizer, WordTokenizer


def main():
    corpus_path = os.path.join(ROOT_DIR, "data", "corpus_general.txt")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    corpus = text.splitlines()
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("Training BPE model...")
    bpe = BPETokenizer(name="BPE-General")
    bpe.train(corpus, vocab_size=400, show_progress=False)
    bpe.save(os.path.join(models_dir, "bpe_general.json"))
    print(f"Saved BPE model (vocab: {bpe.vocab_size})")

    print("Training WordPiece model...")
    wp = WordPieceTokenizer(name="WordPiece-General")
    wp.train(corpus, vocab_size=400, show_progress=False)
    wp.save(os.path.join(models_dir, "wordpiece_general.json"))
    print(f"Saved WordPiece model (vocab: {wp.vocab_size})")

    print("Training Unigram model...")
    uni = UnigramTokenizer(name="Unigram-General")
    uni.train(corpus, vocab_size=350, show_progress=False)
    uni.save(os.path.join(models_dir, "unigram_general.json"))
    print(f"Saved Unigram model (vocab: {uni.vocab_size})")

    print("Training Character model...")
    char_tok = CharacterTokenizer(name="Char-General")
    char_tok.train(corpus, vocab_size=200)
    char_tok.save(os.path.join(models_dir, "char_general.json"))
    print(f"Saved Character model (vocab: {char_tok.vocab_size})")

    print("Saving Byte model...")
    byte_tok = ByteTokenizer(name="Byte-General")
    byte_tok.save(os.path.join(models_dir, "byte_general.json"))
    print(f"Saved Byte model (vocab: {byte_tok.vocab_size})")

    print("Training Word model...")
    word_tok = WordTokenizer(name="Word-General")
    word_tok.train(corpus, vocab_size=300)
    word_tok.save(os.path.join(models_dir, "word_general.json"))
    print(f"Saved Word model (vocab: {word_tok.vocab_size})")

    print("\nAll models trained and exported successfully!")


if __name__ == "__main__":
    main()
