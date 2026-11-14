package tokencraft

import (
	"os"
	"testing"
)

func TestWordPieceLoadAndEncode(t *testing.T) {
	modelPath := "../../../models/wordpiece_general.json"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("models/wordpiece_general.json not found, skipping test")
	}

	tok, err := LoadWordPiece(modelPath)
	if err != nil {
		t.Fatalf("Failed to load WordPiece model: %v", err)
	}

	if len(tok.Vocab) == 0 {
		t.Errorf("Expected non-empty vocabulary, got 0")
	}

	res := tok.Encode("tokenization algorithm")
	if len(res.Tokens) == 0 {
		t.Errorf("Expected tokens to be emitted, got 0")
	}
}

func TestCharacterLoadAndEncode(t *testing.T) {
	modelPath := "../../../models/char_general.json"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("models/char_general.json not found, skipping test")
	}

	tok, err := LoadCharacter(modelPath)
	if err != nil {
		t.Fatalf("Failed to load Character model: %v", err)
	}

	res := tok.Encode("hello")
	if len(res.Tokens) != 5 {
		t.Errorf("Expected 5 character tokens for 'hello', got %d", len(res.Tokens))
	}
}

func TestConcurrentBatchEncode(t *testing.T) {
	modelPath := "../../../models/bpe_general.json"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("models/bpe_general.json not found, skipping test")
	}

	tok, err := LoadBPE(modelPath)
	if err != nil {
		t.Fatalf("Failed to load BPE model: %v", err)
	}

	texts := []string{
		"Hello world",
		"TokenCraft is fast",
		"Byte Pair Encoding from scratch",
	}

	batch := BatchEncodeConcurrent(tok, texts, 2)
	if len(batch.Results) != len(texts) {
		t.Fatalf("Expected %d results, got %d", len(texts), len(batch.Results))
	}

	if batch.Results[0].Text != texts[0] {
		t.Errorf("Expected first result text '%s', got '%s'", texts[0], batch.Results[0].Text)
	}
}
