package tokencraft

import (
	"os"
	"testing"
)

func TestBPELoadAndEncode(t *testing.T) {
	// Look for models/bpe_general.json
	modelPath := "../../../models/bpe_general.json"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip("models/bpe_general.json not found, skipping test")
	}

	tok, err := LoadBPE(modelPath)
	if err != nil {
		t.Fatalf("Failed to load BPE model: %v", err)
	}

	if len(tok.Vocab) == 0 {
		t.Errorf("Expected non-empty vocabulary, got %d", len(tok.Vocab))
	}

	res := tok.Encode("Hello world")
	if len(res.Tokens) == 0 {
		t.Errorf("Expected tokens to be emitted, got 0")
	}
}
