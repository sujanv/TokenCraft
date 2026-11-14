package tokencraft

import (
	"encoding/json"
	"os"
	"strings"
)

// CharModelJSON represents serialized Character-level JSON models.
type CharModelJSON struct {
	Name          string         `json:"name"`
	Type          string         `json:"type"`
	Vocab         map[string]int `json:"vocab"`
	SpecialTokens map[string]int `json:"special_tokens"`
}

// CharacterTokenizer implements character-level tokenization in Go.
type CharacterTokenizer struct {
	Name          string
	Vocab         map[string]int
	InverseVocab  map[int]string
	SpecialTokens map[string]int
}

// LoadCharacter loads a Character JSON model into CharacterTokenizer.
func LoadCharacter(path string) (*CharacterTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var m CharModelJSON
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}

	inv := make(map[int]string, len(m.Vocab))
	for k, v := range m.Vocab {
		inv[v] = k
	}

	return &CharacterTokenizer{
		Name:          m.Name,
		Vocab:         m.Vocab,
		InverseVocab:  inv,
		SpecialTokens: m.SpecialTokens,
	}, nil
}

// Encode converts text into individual character tokens.
func (ct *CharacterTokenizer) Encode(text string) *TokenResult {
	runes := []rune(text)
	tokens := make([]Token, len(runes))
	tokenIDs := make([]int, len(runes))
	tokenStrings := make([]string, len(runes))

	unkID := 1
	if id, ok := ct.SpecialTokens["<unk>"]; ok {
		unkID = id
	}

	for i, r := range runes {
		ch := string(r)
		id, ok := ct.Vocab[ch]
		if !ok {
			id = unkID
		}
		tokens[i] = Token{
			ID:    id,
			Text:  ch,
			Start: i,
			End:   i + 1,
		}
		tokenIDs[i] = id
		tokenStrings[i] = ch
	}

	words := strings.Fields(text)
	wordCount := len(words)
	if wordCount == 0 {
		wordCount = 1
	}

	tokCount := len(tokens)
	byteCount := len([]byte(text))

	compRatio := 0.0
	fertility := 0.0
	if tokCount > 0 {
		compRatio = float64(byteCount) / float64(tokCount)
		fertility = float64(tokCount) / float64(wordCount)
	}

	return &TokenResult{
		Text:             text,
		Tokens:           tokens,
		TokenIDs:         tokenIDs,
		TokenStrings:     tokenStrings,
		CompressionRatio: compRatio,
		Fertility:        fertility,
	}
}
