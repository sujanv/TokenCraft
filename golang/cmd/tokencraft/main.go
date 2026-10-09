package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sujanv/TokenCraft/golang/pkg/tokencraft"
)

func main() {
	modelPath := flag.String("model", "models/bpe_general.json", "Path to TokenCraft JSON model")
	encodeText := flag.String("encode", "", "Text to tokenize")
	benchmark := flag.Bool("benchmark", false, "Run speed benchmark")
	flag.Parse()

	if *encodeText == "" && !*benchmark {
		fmt.Println("TokenCraft (Go Edition) - High Performance Tokenizer Engine")
		fmt.Println("Usage: tokencraft -model models/bpe_general.json -encode \"Hello world\"")
		fmt.Println("       tokencraft -model models/bpe_general.json -benchmark")
		os.Exit(0)
	}

	tok, err := tokencraft.LoadBPE(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}

	if *encodeText != "" {
		res := tok.Encode(*encodeText)
		fmt.Printf("Model:        %s (Vocab: %d)\n", tok.Name, len(tok.Vocab))
		fmt.Printf("Input:        \"%s\"\n", *encodeText)
		fmt.Printf("Tokens (%d):   [%s]\n", len(res.Tokens), strings.Join(res.TokenStrings, " | "))
		fmt.Printf("Token IDs:    %v\n", res.TokenIDs)
		fmt.Printf("Compression:  %.2f bytes/token\n", res.CompressionRatio)
		fmt.Printf("Fertility:    %.2f tokens/word\n", res.Fertility)
	}

	if *benchmark {
		sentence := "The quick brown fox jumps over the lazy dog and explores the foundations of tokenization."
		iterations := 100000

		fmt.Printf("\nBenchmarking Go BPE Encoder across %d iterations...\n", iterations)
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
	}
}
