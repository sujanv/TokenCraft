package tokencraft

import (
	"sync"
)

// TokenizerInterface defines the common interface for TokenCraft Go tokenizers.
type TokenizerInterface interface {
	Encode(text string) *TokenResult
}

// BatchResult holds batch tokenization outputs.
type BatchResult struct {
	Results   []*TokenResult
	MaxTokens int
}

// BatchEncodeConcurrent encodes multiple texts concurrently using a worker pool.
func BatchEncodeConcurrent(tok TokenizerInterface, texts []string, numWorkers int) *BatchResult {
	if numWorkers <= 0 {
		numWorkers = 4
	}
	if numWorkers > len(texts) {
		numWorkers = len(texts)
	}
	if numWorkers == 0 {
		numWorkers = 1
	}

	type job struct {
		index int
		text  string
	}
	type jobResult struct {
		index  int
		result *TokenResult
	}

	jobs := make(chan job, len(texts))
	results := make(chan jobResult, len(texts))

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				r := tok.Encode(j.text)
				results <- jobResult{index: j.index, result: r}
			}
		}()
	}

	for i, t := range texts {
		jobs <- job{index: i, text: t}
	}
	close(jobs)

	wg.Wait()
	close(results)

	ordered := make([]*TokenResult, len(texts))
	maxTok := 0
	for res := range results {
		ordered[res.index] = res.result
		if len(res.result.Tokens) > maxTok {
			maxTok = len(res.result.Tokens)
		}
	}

	return &BatchResult{
		Results:   ordered,
		MaxTokens: maxTok,
	}
}
