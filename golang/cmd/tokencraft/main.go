package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sujanv/TokenCraft/pkg/tokencraft"
)

func main() {
	modelPath := flag.String("model", "models/bpe_general.json", "Path to TokenCraft JSON model")
	modelType := flag.String("type", "bpe", "Tokenizer type: bpe, wordpiece, or char")
	encodeText := flag.String("encode", "", "Text to tokenize")
	benchmark := flag.Bool("benchmark", false, "Run speed benchmark")
	workers := flag.Int("workers", 4, "Number of concurrent worker goroutines for batch processing")
	flag.Parse()

	if *encodeText == "" && !*benchmark {
		fmt.Println("TokenCraft (Go Edition) - High Performance Tokenizer Engine")
		fmt.Println("Usage: tokencraft -model models/bpe_general.json -encode \"Hello world\"")
		fmt.Println("       tokencraft -model models/wordpiece_general.json -type wordpiece -encode \"Hello world\"")
		fmt.Println("       tokencraft -model models/bpe_general.json -benchmark -workers 8")
		os.Exit(0)
	}

	var tok tokencraft.TokenizerInterface
	var name string
	var vocabSize int

	switch strings.ToLower(*modelType) {
	case "wordpiece", "wp":
		wpt, err := tokencraft.LoadWordPiece(*modelPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading WordPiece model: %v\n", err)
			os.Exit(1)
		}
		tok = wpt
		name = wpt.Name
		vocabSize = len(wpt.Vocab)
	case "char", "character":
		ct, err := tokencraft.LoadCharacter(*modelPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading Character model: %v\n", err)
			os.Exit(1)
		}
		tok = ct
		name = ct.Name
		vocabSize = len(ct.Vocab)
	default:
		bpe, err := tokencraft.LoadBPE(*modelPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading BPE model: %v\n", err)
			os.Exit(1)
		}
		tok = bpe
		name = bpe.Name
		vocabSize = len(bpe.Vocab)
	}

	if *encodeText != "" {
		res := tok.Encode(*encodeText)
		fmt.Printf("Model:        %s (%s, Vocab: %d)\n", name, *modelType, vocabSize)
		fmt.Printf("Input:        \"%s\"\n", *encodeText)
		fmt.Printf("Tokens (%d):   [%s]\n", len(res.Tokens), strings.Join(res.TokenStrings, " | "))
		fmt.Printf("Token IDs:    %v\n", res.TokenIDs)
		fmt.Printf("Compression:  %.2f bytes/token\n", res.CompressionRatio)
		fmt.Printf("Fertility:    %.2f tokens/word\n", res.Fertility)
	}

	if *benchmark {
		sentence := "The quick brown fox jumps over the lazy dog and explores the foundations of tokenization."
		iterations := 100000

		fmt.Printf("\n--- Single-Threaded Benchmark (%d iterations) ---\n", iterations)
		start := time.Now()
		totalTokens := 0
		for i := 0; i < iterations; i++ {
			r := tok.Encode(sentence)
			totalTokens += len(r.Tokens)
		}
		duration := time.Since(start)
		tokPerSec := float64(totalTokens) / duration.Seconds()
		fmt.Printf("Elapsed:      %v\n", duration)
		fmt.Printf("Throughput:   %.0f tokens/sec\n", tokPerSec)

		// Concurrent batch benchmark
		fmt.Printf("\n--- Concurrent Batch Benchmark (%d workers, %d sentences) ---\n", *workers, iterations)
		texts := make([]string, iterations)
		for i := range texts {
			texts[i] = sentence
		}

		startConcurrent := time.Now()
		batchRes := tokencraft.BatchEncodeConcurrent(tok, texts, *workers)
		durationConcurrent := time.Since(startConcurrent)

		totalConcurrentTokens := 0
		for _, r := range batchRes.Results {
			totalConcurrentTokens += len(r.Tokens)
		}
		concurrentTokPerSec := float64(totalConcurrentTokens) / durationConcurrent.Seconds()
		fmt.Printf("Elapsed:      %v\n", durationConcurrent)
		fmt.Printf("Throughput:   %.0f tokens/sec (Speedup: %.2fx)\n", concurrentTokPerSec, concurrentTokPerSec/tokPerSec)
	}
}
