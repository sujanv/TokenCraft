package tokencraft

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
)

// WordPieceModelJSON represents serialized WordPiece JSON models.
type WordPieceModelJSON struct {
	Name          string         `json:"name"`
	Type          string         `json:"type"`
	Vocab         map[string]int `json:"vocab"`
	SpecialTokens map[string]int `json:"special_tokens"`
	Prefix        string         `json:"prefix"`
	UnkToken      string         `json:"unk_token"`
}

// WordPieceTokenizer implements BERT-style WordPiece tokenization in Go.
type WordPieceTokenizer struct {
	Name          string
	Vocab         map[string]int
	InverseVocab  map[int]string
	SpecialTokens map[string]int
	Prefix        string
	UnkToken      string
	Regex         *regexp.Regexp
}

// LoadWordPiece loads a WordPiece JSON model into WordPieceTokenizer.
func LoadWordPiece(path string) (*WordPieceTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var m WordPieceModelJSON
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}

	if m.Prefix == "" {
		m.Prefix = "##"
	}
	if m.UnkToken == "" {
		m.UnkToken = "[UNK]"
	}

	inv := make(map[int]string, len(m.Vocab))
	for k, v := range m.Vocab {
		inv[v] = k
	}

	re := regexp.MustCompile(`\w+|[^\w\s]`)

	return &WordPieceTokenizer{
		Name:          m.Name,
		Vocab:         m.Vocab,
		InverseVocab:  inv,
		SpecialTokens: m.SpecialTokens,
		Prefix:        m.Prefix,
		UnkToken:      m.UnkToken,
		Regex:         re,
	}, nil
}

// Encode tokenizes text into WordPiece subwords.
func (wp *WordPieceTokenizer) Encode(text string) *TokenResult {
	lowerText := strings.ToLower(text)
	matches := wp.Regex.FindAllStringIndex(lowerText, -1)

	var tokens []Token
	var tokenIDs []int
	var tokenStrings []string

	unkID, hasUnk := wp.Vocab[wp.UnkToken]
	if !hasUnk {
		unkID = 1
	}

	for _, match := range matches {
		word := lowerText[match[0]:match[1]]
		subwords := wp.tokenizeWord(word)

		for _, sw := range subwords {
			id, ok := wp.Vocab[sw]
			if !ok {
				id = unkID
			}
			t := Token{
				ID:    id,
				Text:  sw,
				Start: match[0],
				End:   match[1],
			}
			tokens = append(tokens, t)
			tokenIDs = append(tokenIDs, id)
			tokenStrings = append(tokenStrings, sw)
		}
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

func (wp *WordPieceTokenizer) tokenizeWord(word string) []string {
	if len(word) == 0 {
		return nil
	}

	runes := []rune(word)
	var tokens []string
	start := 0
	wordLen := len(runes)

	for start < wordLen {
		end := wordLen
		curSubstr := ""

		for start < end {
			sub := string(runes[start:end])
			if start > 0 {
				sub = wp.Prefix + sub
			}
			if _, ok := wp.Vocab[sub]; ok {
				curSubstr = sub
				break
			}
			end--
		}

		if curSubstr == "" {
			return []string{wp.UnkToken}
		}

		tokens = append(tokens, curSubstr)
		start = end
	}

	return tokens
}
