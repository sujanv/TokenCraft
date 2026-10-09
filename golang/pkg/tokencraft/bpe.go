package tokencraft

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// Token represents a tokenized unit.
type Token struct {
	ID    int    `json:"id"`
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

// TokenResult represents the output of an encode operation.
type TokenResult struct {
	Text             string   `json:"text"`
	Tokens           []Token  `json:"tokens"`
	TokenIDs         []int    `json:"token_ids"`
	TokenStrings     []string `json:"token_strings"`
	CompressionRatio float64  `json:"compression_ratio"`
	Fertility        float64  `json:"fertility"`
}

// ModelJSON represents the serialized TokenCraft JSON model.
type ModelJSON struct {
	Name          string           `json:"name"`
	Type          string           `json:"type"`
	Vocab         map[string]int   `json:"vocab"`
	SpecialTokens map[string]int   `json:"special_tokens"`
	Merges        [][]string       `json:"merges"`
	Pattern       string           `json:"pre_tokenizer_pattern"`
}

// BPETokenizer implements high-performance BPE in Go.
type BPETokenizer struct {
	Name          string
	Vocab         map[string]int
	InverseVocab  map[int]string
	SpecialTokens map[string]int
	Merges        [][2]string
	BPERanks      map[string]int
	Regex         *regexp.Regexp
}

// LoadBPE loads a TokenCraft JSON model into BPETokenizer.
func LoadBPE(path string) (*BPETokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var m ModelJSON
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}

	inv := make(map[int]string, len(m.Vocab))
	for k, v := range m.Vocab {
		inv[v] = k
	}

	merges := make([][2]string, len(m.Merges))
	ranks := make(map[string]int, len(m.Merges))
	for i, pair := range m.Merges {
		if len(pair) == 2 {
			merges[i] = [2]string{pair[0], pair[1]}
			ranks[pair[0]+"\x00"+pair[1]] = i
		}
	}

	// GPT-2 style pretokenization regex
	re := regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]+|\s+`)

	return &BPETokenizer{
		Name:          m.Name,
		Vocab:         m.Vocab,
		InverseVocab:  inv,
		SpecialTokens: m.SpecialTokens,
		Merges:        merges,
		BPERanks:      ranks,
		Regex:         re,
	}, nil
}

// Encode converts text into subword tokens and IDs.
func (b *BPETokenizer) Encode(text string) *TokenResult {
	matches := b.Regex.FindAllStringIndex(text, -1)
	var tokens []Token
	var tokenIDs []int
	var tokenStrings []string

	for _, match := range matches {
		chunk := text[match[0]:match[1]]
		subwords := b.bpeEncodeChunk(chunk)

		for _, sw := range subwords {
			id, ok := b.Vocab[sw]
			if !ok {
				id = b.SpecialTokens["<unk>"]
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

// Decode converts token IDs back into string.
func (b *BPETokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if s, ok := b.InverseVocab[id]; ok {
			sb.WriteString(s)
		}
	}
	return sb.String()
}

func (b *BPETokenizer) bpeEncodeChunk(chunk string) []string {
	runes := []rune(chunk)
	if len(runes) <= 1 {
		return []string{chunk}
	}

	pieces := make([]string, len(runes))
	for i, r := range runes {
		pieces[i] = string(r)
	}

	for len(pieces) > 1 {
		minRank := 100000000
		minIdx := -1

		for i := 0; i < len(pieces)-1; i++ {
			key := pieces[i] + "\x00" + pieces[i+1]
			if rank, ok := b.BPERanks[key]; ok {
				if rank < minRank {
					minRank = rank
					minIdx = i
				}
			}
		}

		if minIdx == -1 {
			break
		}

		// Merge at minIdx
		newPieces := make([]string, 0, len(pieces)-1)
		newPieces = append(newPieces, pieces[:minIdx]...)
		newPieces = append(newPieces, pieces[minIdx]+pieces[minIdx+1])
		newPieces = append(newPieces, pieces[minIdx+2:]...)
		pieces = newPieces
	}

	return pieces
}
