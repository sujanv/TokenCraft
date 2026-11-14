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
    unigram: {
      container: document.getElementById("tokens-unigram"),
      stats: document.getElementById("stats-unigram"),
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
    "Unigram (SentencePiece)": "unigram",
    "Unigram (SentencePiece-style)": "unigram",
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

// Advanced Tabs switching
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    const target = document.getElementById(btn.dataset.tab);
    if (target) target.classList.add("active");
  });
});

// Dropout Slider & Action
const dropoutRateInput = document.getElementById("dropoutRate");
const dropoutRateVal = document.getElementById("dropoutRateVal");
if (dropoutRateInput && dropoutRateVal) {
  dropoutRateInput.addEventListener("input", (e) => {
    dropoutRateVal.textContent = e.target.value;
  });
}

const sampleDropoutBtn = document.getElementById("sampleDropoutBtn");
if (sampleDropoutBtn) {
  sampleDropoutBtn.addEventListener("click", async () => {
    const sentence = elements.sentenceInput.value.trim();
    const rate = parseFloat(dropoutRateInput.value);
    const container = document.getElementById("dropoutResults");
    container.innerHTML = "<div>Sampling stochastic subwords...</div>";

    try {
      const resp = await fetch("/api/dropout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence, dropout_rate: rate, samples: 4 })
      });
      const data = await resp.json();
      container.innerHTML = "";

      // Deterministic baseline
      const detRow = document.createElement("div");
      detRow.className = "dropout-sample-row";
      detRow.innerHTML = `<span class="dropout-sample-title">Greedy Deterministic (α = 0.0) &bull; ${data.deterministic.token_count} tokens</span>`;
      const detChips = document.createElement("div");
      detChips.className = "token-container";
      data.deterministic.tokens.forEach((t, idx) => {
        const chip = document.createElement("div");
        chip.className = `token-chip ${COLOR_CLASSES[idx % COLOR_CLASSES.length]}`;
        chip.innerHTML = `<span>${escapeHtml(t.text)}</span>`;
        detChips.appendChild(chip);
      });
      detRow.appendChild(detChips);
      container.appendChild(detRow);

      // Stochastic samples
      data.samples.forEach((sample, i) => {
        const row = document.createElement("div");
        row.className = "dropout-sample-row";
        row.innerHTML = `<span class="dropout-sample-title">Stochastic Sample #${i + 1} (α = ${rate}) &bull; ${sample.token_count} tokens</span>`;
        const chips = document.createElement("div");
        chips.className = "token-container";
        sample.token_strings.forEach((str, idx) => {
          const chip = document.createElement("div");
          chip.className = `token-chip ${COLOR_CLASSES[(idx + i) % COLOR_CLASSES.length]}`;
          chip.innerHTML = `<span>${escapeHtml(str)}</span>`;
          chips.appendChild(chip);
        });
        row.appendChild(chips);
        container.appendChild(row);
      });
    } catch (e) {
      container.innerHTML = `<div style="color:var(--accent-rose)">Dropout request failed: ${e.message}</div>`;
    }
  });
}

// Diff Action
const runDiffBtn = document.getElementById("runDiffBtn");
if (runDiffBtn) {
  runDiffBtn.addEventListener("click", async () => {
    const sentence = elements.sentenceInput.value.trim();
    const tok1 = document.getElementById("diffTok1").value;
    const tok2 = document.getElementById("diffTok2").value;
    const container = document.getElementById("diffResults");
    container.innerHTML = "<div>Computing boundary alignment diff...</div>";

    try {
      const resp = await fetch("/api/diff", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence, tok1, tok2 })
      });
      const data = await resp.json();
      container.innerHTML = `
        <div class="diff-stat-boxes">
          <div class="diff-box">
            <div class="diff-box-val">${(data.boundary_overlap_score * 100).toFixed(1)}%</div>
            <div class="diff-box-lbl">Boundary Jaccard Overlap</div>
          </div>
          <div class="diff-box">
            <div class="diff-box-val">${data.shared_boundaries.length}</div>
            <div class="diff-box-lbl">Shared Cut Points</div>
          </div>
          <div class="diff-box">
            <div class="diff-box-val">${data.exclusive_boundaries1.length}</div>
            <div class="diff-box-lbl">Unique to ${data.tokenizer1_name}</div>
          </div>
          <div class="diff-box">
            <div class="diff-box-val">${data.exclusive_boundaries2.length}</div>
            <div class="diff-box-lbl">Unique to ${data.tokenizer2_name}</div>
          </div>
        </div>
      `;

      // Token comparison lists
      const t1Div = document.createElement("div");
      t1Div.className = "dropout-sample-row";
      t1Div.innerHTML = `<span class="dropout-sample-title">${data.tokenizer1_name} (${data.token_count1} tokens)</span>`;
      const c1 = document.createElement("div");
      c1.className = "token-container";
      data.tokens1.forEach((t, i) => {
        const chip = document.createElement("div");
        chip.className = `token-chip ${COLOR_CLASSES[i % COLOR_CLASSES.length]}`;
        chip.innerHTML = `<span>${escapeHtml(t)}</span>`;
        c1.appendChild(chip);
      });
      t1Div.appendChild(c1);
      container.appendChild(t1Div);

      const t2Div = document.createElement("div");
      t2Div.className = "dropout-sample-row";
      t2Div.innerHTML = `<span class="dropout-sample-title">${data.tokenizer2_name} (${data.token_count2} tokens)</span>`;
      const c2 = document.createElement("div");
      c2.className = "token-container";
      data.tokens2.forEach((t, i) => {
        const chip = document.createElement("div");
        chip.className = `token-chip ${COLOR_CLASSES[(i + 3) % COLOR_CLASSES.length]}`;
        chip.innerHTML = `<span>${escapeHtml(t)}</span>`;
        c2.appendChild(chip);
      });
      t2Div.appendChild(c2);
      container.appendChild(t2Div);
    } catch (e) {
      container.innerHTML = `<div style="color:var(--accent-rose)">Diff request failed: ${e.message}</div>`;
    }
  });
}

// Batch Action
const runBatchBtn = document.getElementById("runBatchBtn");
if (runBatchBtn) {
  runBatchBtn.addEventListener("click", async () => {
    const raw = document.getElementById("batchInput").value;
    const texts = raw.split("\n").map(s => s.trim()).filter(Boolean);
    const maxLen = parseInt(document.getElementById("batchMaxLen").value, 10) || 12;
    const container = document.getElementById("batchResults");
    container.innerHTML = "<div>Processing batch & attention masks...</div>";

    try {
      const resp = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts, max_length: maxLen, padding: "longest" })
      });
      const data = await resp.json();
      let html = `<div class="table-responsive"><table class="batch-matrix-table">`;
      html += `<thead><tr><th>#</th><th>Input Text</th><th>Tokens Matrix</th><th>Attention Mask</th></tr></thead><tbody>`;

      data.input_ids.forEach((ids, idx) => {
        const strList = data.token_strings[idx].map(s => {
          return s === "<pad>" ? `<span class="pad-cell">&lt;pad&gt;</span>` : `<span>[${escapeHtml(s)}]</span>`;
        }).join(" ");
        const maskStr = data.attention_mask[idx].join(", ");

        html += `
          <tr>
            <td><strong>${idx + 1}</strong></td>
            <td>"${escapeHtml(texts[idx] || '')}"</td>
            <td>${strList}</td>
            <td><code>[${maskStr}]</code></td>
          </tr>
        `;
      });
      html += `</tbody></table></div>`;
      container.innerHTML = html;
    } catch (e) {
      container.innerHTML = `<div style="color:var(--accent-rose)">Batch request failed: ${e.message}</div>`;
    }
  });
}

// Initial run
window.addEventListener("DOMContentLoaded", () => {
  runComparison();
});
