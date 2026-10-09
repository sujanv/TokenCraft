/**
 * TokenCraft Interactive Visualizer Client Script
 */

const PRESETS = {
  "English General": "The quick brown fox jumps over the lazy dog and explores the wonders of tokenization.",
  "Python Code": "def fibonacci(n: int) -> int:\n    if n <= 1: return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
  "Contractions & Slang": "You'll never believe what's happening! Don't let's argue, it's gotta be done ASAP.",
  "Multilingual & Glyphs": "TokenCraft supports multilingual text: こんにちは世界, Bonjour le monde, and Привет мир!",
  "Compounds & Edge Cases": "Donaudampfschifffahrtselektrizitätenhauptbetriebswerk and supercalifragilisticexpialidocious 🚀🔥",
};

const COLOR_CLASSES = [
  "chip-c0", "chip-c1", "chip-c2", "chip-c3", "chip-c4", "chip-c5", "chip-c6", "chip-c7"
];

const elements = {
  sentenceInput: document.getElementById("sentenceInput"),
  vocabSize: document.getElementById("vocabSize"),
  vocabSizeVal: document.getElementById("vocabSizeVal"),
  compareBtn: document.getElementById("compareBtn"),
  samplePills: document.getElementById("samplePills"),
  metricsTableBody: document.getElementById("metricsTableBody"),
  tooltip: document.getElementById("tooltip"),
  cards: {
    bpe: {
      container: document.getElementById("tokens-bpe"),
      stats: document.getElementById("stats-bpe"),
    },
    wordpiece: {
      container: document.getElementById("tokens-wordpiece"),
      stats: document.getElementById("stats-wordpiece"),
    },
    char: {
      container: document.getElementById("tokens-char"),
      stats: document.getElementById("stats-char"),
    },
    byte: {
      container: document.getElementById("tokens-byte"),
      stats: document.getElementById("stats-byte"),
    },
    word: {
      container: document.getElementById("tokens-word"),
      stats: document.getElementById("stats-word"),
    },
  }
};

// Update slider value display
elements.vocabSize.addEventListener("input", (e) => {
  elements.vocabSizeVal.textContent = e.target.value;
});

// Setup preset buttons
elements.samplePills.querySelectorAll(".sample-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.key;
    if (PRESETS[key]) {
      elements.sentenceInput.value = PRESETS[key];
      runComparison();
    }
  });
});

elements.compareBtn.addEventListener("click", runComparison);

async function runComparison() {
  const sentence = elements.sentenceInput.value.trim();
  const vocabSize = parseInt(elements.vocabSize.value, 10);
  if (!sentence) return;

  elements.compareBtn.disabled = true;
  elements.compareBtn.innerHTML = "<span>Tokenizing...</span>";

  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence, vocab_size: vocabSize }),
    });

    if (!response.ok) throw new Error("Server error");
    const data = await response.json();
    renderResults(data);
  } catch (err) {
    // Fallback: If server is not running (e.g. static preview), run client-side emulation
    console.warn("API request failed, running client-side emulation:", err);
    runClientSideFallback(sentence, vocabSize);
  } finally {
    elements.compareBtn.disabled = false;
    elements.compareBtn.innerHTML = "<span>Compare Tokenizers</span><span class='btn-icon'>➔</span>";
  }
}

function renderResults(data) {
  // 1. Populate Metrics Table
  elements.metricsTableBody.innerHTML = "";
  if (data.table) {
    data.table.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${row.Tokenizer}</strong></td>
        <td>${row["Vocab Size"]}</td>
        <td><strong>${row["Token Count"]}</strong></td>
        <td>${row["Compression Ratio"]}</td>
        <td>${row.Fertility}</td>
        <td>${row["OOV Rate"]}</td>
      `;
      elements.metricsTableBody.appendChild(tr);
    });
  }

  // 2. Map splits to visual cards
  const algoMapping = {
    "BPE (Byte-Pair)": "bpe",
    "BPE (GPT-style)": "bpe",
    "WordPiece (BERT)": "wordpiece",
    "WordPiece (BERT-style)": "wordpiece",
    "Character-Level": "char",
    "Byte-Level (256)": "byte",
    "Word-Level": "word",
  };

  for (const [algoName, tokens] of Object.entries(data.splits)) {
    const key = algoMapping[algoName] || algoName.toLowerCase();
    const cardObj = elements.cards[key];
    if (!cardObj) continue;

    // Update stats header
    const metric = data.metrics[algoName];
    if (metric) {
      cardObj.stats.innerHTML = `
        <span><strong>${metric.token_count}</strong> tokens</span> &bull;
        <span><strong>${metric.compression_ratio}</strong> B/tok</span> &bull;
        <span><strong>${metric.fertility}</strong> tok/wd</span>
      `;
    }

    // Render token chips
    cardObj.container.innerHTML = "";
    tokens.forEach((t, i) => {
      const chip = document.createElement("div");
      const colorClass = COLOR_CLASSES[i % COLOR_CLASSES.length];
      chip.className = `token-chip ${colorClass}`;

      // Escape display text
      const displayText = t.text.replace(/\n/g, "↵").replace(/\t/g, "⇥");
      chip.innerHTML = `<span>${escapeHtml(displayText)}</span><span class="tok-id">#${t.id}</span>`;

      // Tooltip events
      chip.addEventListener("mouseenter", (e) => showTooltip(e, t));
      chip.addEventListener("mousemove", (e) => moveTooltip(e));
      chip.addEventListener("mouseleave", hideTooltip);

      cardObj.container.appendChild(chip);
    });
  }
}

// Tooltip handler
function showTooltip(e, token) {
  const tt = elements.tooltip;
  tt.innerHTML = `
    <div><strong>Token:</strong> "${escapeHtml(token.text)}"</div>
    <div><strong>Token ID:</strong> ${token.id}</div>
    ${token.byte_repr ? `<div><strong>Bytes:</strong> <code>${token.byte_repr}</code></div>` : ""}
    ${token.start >= 0 ? `<div><strong>Span:</strong> [${token.start}:${token.end}]</div>` : ""}
    <div><strong>Special:</strong> ${token.is_special ? "Yes" : "No"}</div>
  `;
  tt.style.display = "block";
  moveTooltip(e);
}

function moveTooltip(e) {
  const tt = elements.tooltip;
  tt.style.left = `${e.pageX + 12}px`;
  tt.style.top = `${e.pageY + 12}px`;
}

function hideTooltip() {
  elements.tooltip.style.display = "none";
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Client-side lightweight fallback emulator when opened as file://
function runClientSideFallback(sentence, vocabSize) {
  // Quick character split
  const charTokens = Array.from(sentence).map((c, i) => ({
    id: c.charCodeAt(0),
    text: c,
    start: i,
    end: i + 1,
    is_special: false
  }));

  // Quick word split
  const words = sentence.match(/\w+|[^\w\s]/g) || [];
  const wordTokens = words.map((w, i) => ({
    id: 100 + i,
    text: w,
    start: sentence.indexOf(w),
    end: sentence.indexOf(w) + w.length,
    is_special: false
  }));

  // Quick byte split
  const encoder = new TextEncoder();
  const rawBytes = Array.from(encoder.encode(sentence));
  const byteTokens = rawBytes.map(b => ({
    id: b,
    text: `0x${b.toString(16).padStart(2, "0")}`,
    byte_repr: `0x${b.toString(16).padStart(2, "0")}`,
    start: -1,
    end: -1,
    is_special: false
  }));

  // Emulated subword BPE & WordPiece
  const bpeTokens = [];
  words.forEach((w, wIdx) => {
    if (w.length > 5) {
      bpeTokens.push({ id: 200 + wIdx * 2, text: w.slice(0, 3), start: -1, end: -1, is_special: false });
      bpeTokens.push({ id: 201 + wIdx * 2, text: w.slice(3), start: -1, end: -1, is_special: false });
    } else {
      bpeTokens.push({ id: 200 + wIdx, text: w, start: -1, end: -1, is_special: false });
    }
  });

  const wpTokens = [];
  words.forEach((w, wIdx) => {
    if (w.length > 4) {
      wpTokens.push({ id: 300 + wIdx * 2, text: w.slice(0, 3), start: -1, end: -1, is_special: false });
      wpTokens.push({ id: 301 + wIdx * 2, text: `##${w.slice(3)}`, start: -1, end: -1, is_special: false });
    } else {
      wpTokens.push({ id: 300 + wIdx, text: w, start: -1, end: -1, is_special: false });
    }
  });

  const byteLen = rawBytes.length;
  const wordCount = Math.max(1, sentence.trim().split(/\s+/).length);

  const mockData = {
    splits: {
      "BPE (GPT-style)": bpeTokens,
      "WordPiece (BERT-style)": wpTokens,
      "Character-Level": charTokens,
      "Byte-Level (256)": byteTokens,
      "Word-Level": wordTokens,
    },
    metrics: {
      "BPE (GPT-style)": { token_count: bpeTokens.length, compression_ratio: (byteLen / bpeTokens.length).toFixed(2), fertility: (bpeTokens.length / wordCount).toFixed(2) },
      "WordPiece (BERT-style)": { token_count: wpTokens.length, compression_ratio: (byteLen / wpTokens.length).toFixed(2), fertility: (wpTokens.length / wordCount).toFixed(2) },
      "Character-Level": { token_count: charTokens.length, compression_ratio: (byteLen / charTokens.length).toFixed(2), fertility: (charTokens.length / wordCount).toFixed(2) },
      "Byte-Level (256)": { token_count: byteTokens.length, compression_ratio: (byteLen / byteTokens.length).toFixed(2), fertility: (byteTokens.length / wordCount).toFixed(2) },
      "Word-Level": { token_count: wordTokens.length, compression_ratio: (byteLen / wordTokens.length).toFixed(2), fertility: (wordTokens.length / wordCount).toFixed(2) },
    },
    table: [
      { Tokenizer: "BPE (GPT-style)", "Vocab Size": vocabSize, "Token Count": bpeTokens.length, "Compression Ratio": `${(byteLen / bpeTokens.length).toFixed(2)} B/tok`, Fertility: `${(bpeTokens.length / wordCount).toFixed(2)} tok/wd`, "OOV Rate": "0.0%" },
      { Tokenizer: "WordPiece (BERT-style)", "Vocab Size": vocabSize, "Token Count": wpTokens.length, "Compression Ratio": `${(byteLen / wpTokens.length).toFixed(2)} B/tok`, Fertility: `${(wpTokens.length / wordCount).toFixed(2)} tok/wd`, "OOV Rate": "0.0%" },
      { Tokenizer: "Character-Level", "Vocab Size": 150, "Token Count": charTokens.length, "Compression Ratio": `${(byteLen / charTokens.length).toFixed(2)} B/tok`, Fertility: `${(charTokens.length / wordCount).toFixed(2)} tok/wd`, "OOV Rate": "0.0%" },
      { Tokenizer: "Byte-Level (256)", "Vocab Size": 256, "Token Count": byteTokens.length, "Compression Ratio": `${(byteLen / byteTokens.length).toFixed(2)} B/tok`, Fertility: `${(byteTokens.length / wordCount).toFixed(2)} tok/wd`, "OOV Rate": "0.0%" },
      { Tokenizer: "Word-Level", "Vocab Size": 300, "Token Count": wordTokens.length, "Compression Ratio": `${(byteLen / wordTokens.length).toFixed(2)} B/tok`, Fertility: `${(wordTokens.length / wordCount).toFixed(2)} tok/wd`, "OOV Rate": "0.0%" },
    ]
  };

  renderResults(mockData);
}

// Initial run
window.addEventListener("DOMContentLoaded", () => {
  runComparison();
});
