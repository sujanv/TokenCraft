"""
Tests for HuggingFace and GPT-2 exporters.
"""

import unittest
import os
import json
import tempfile
from tokencraft.bpe import BPETokenizer
from tokencraft.exporter import HuggingFaceExporter, OpenAIGPT2Exporter


class TestExporters(unittest.TestCase):
    def setUp(self):
        self.corpus = ["hello world this is a test corpus for exporter testing"]
        self.bpe = BPETokenizer()
        self.bpe.train(self.corpus, vocab_size=275)

    def test_huggingface_export(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            HuggingFaceExporter.export_tokenizer_json(self.bpe, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["version"], "1.0")
            self.assertEqual(data["model"]["type"], "BPE")
            self.assertIn("merges", data["model"])
            self.assertIn("vocab", data["model"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_gpt2_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            OpenAIGPT2Exporter.export(self.bpe, tmp_dir)
            vocab_file = os.path.join(tmp_dir, "vocab.json")
            merges_file = os.path.join(tmp_dir, "merges.txt")

            self.assertTrue(os.path.exists(vocab_file))
            self.assertTrue(os.path.exists(merges_file))

            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            self.assertEqual(len(vocab), self.bpe.vocab_size)


if __name__ == "__main__":
    unittest.main()
