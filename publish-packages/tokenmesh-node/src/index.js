/**
 * TokenMesh — Semantic token optimizer for LLM prompts
 * JS port of github.com/ankitsingh36/tokenmesh
 *
 * Uses TF-IDF cosine similarity (same engine as the browser demo).
 * For full neural embedding accuracy, use the Python library.
 *
 * @example
 * const { TokenMesh, TokenMeshLite, TokenMeshAggressive } = require('tokenmesh');
 *
 * const tm = new TokenMesh();
 * const result = tm.optimize(longText, { query: 'stop loss rules' });
 * console.log(result.optimizedText);
 * console.log(result.reductionPercent + '%');
 */

'use strict';

// ── Stopwords ──────────────────────────────────────────────────────────────
const STOPWORDS = new Set([
  'a','an','the','is','it','in','on','of','to','and','or','for','with',
  'are','be','by','at','as','this','that','from','your','you','all','also',
  'any','can','its','has','have','not','but','more','than','was','will',
  'never','always','every','each','use','when','once','after','before',
  'then','do','so','up','per','just','very','really',
]);

// ── Filler patterns (sentence-start only) ────────────────────────────────
const FILLER_RULES = [
  [/^It is (?:very |extremely )?important (?:to note )?that\s+/gim, ''],
  [/^Please note that\s+/gim, ''],
  [/^Note that\s+/gim, ''],
  [/^Furthermore,\s+/gim, ''],
  [/^Additionally,\s+/gim, ''],
  [/^Moreover,\s+/gim, ''],
  [/^That being said,\s+/gim, ''],
  [/^In other words,\s+/gim, ''],
  [/^As (?:previously |earlier )?mentioned,?\s+/gim, ''],
  // Markdown artifacts
  [/^#{1,6}\s+/gm, ''],
  [/\*{1,3}([^*\n]+)\*{1,3}/g, '$1'],
  [/`([^`\n]+)`/g, '$1'],
  [/<[^>]+>/g, ''],
  [/\[([^\]]+)\]\([^)]+\)/g, '$1'],
  [/[ \t]{2,}/g, ' '],
];

// ── Numeric guard — protect sentences with specific rule values ───────────
const NUMERIC_PAT = /\d+:\d|\d+\.?\d*[xX]\b|\b\d+-?EMA\b|\b\d+[Rr]\b|breakeven/i;
function isProtected(sentence) {
  return NUMERIC_PAT.test(sentence);
}

// ── TF-IDF engine ─────────────────────────────────────────────────────────
function tokenize(text) {
  return (text.toLowerCase().match(/[a-z]+/g) || [])
    .filter(w => !STOPWORDS.has(w) && w.length > 3);
}

function buildTFIDF(docs) {
  const N = docs.length;
  if (!N) return [];
  const tokenized = docs.map(tokenize);
  const vocab = {};
  tokenized.forEach(ws => ws.forEach(w => { if (!(w in vocab)) vocab[w] = Object.keys(vocab).length; }));
  const V = Object.keys(vocab).length;
  if (!V) return docs.map(() => new Float32Array(0));

  // TF
  const tf = tokenized.map(ws => {
    const v = new Float32Array(V);
    if (!ws.length) return v;
    const c = {};
    ws.forEach(w => c[w] = (c[w] || 0) + 1);
    Object.entries(c).forEach(([w, n]) => v[vocab[w]] = n / ws.length);
    return v;
  });

  // IDF
  const df = new Float32Array(V);
  tf.forEach(v => v.forEach((x, i) => { if (x > 0) df[i]++; }));
  const idf = Array.from(df).map(d => Math.log((N + 1) / (d + 1)) + 1);

  // TF-IDF + L2 normalize
  return tf.map(v => {
    const tfidf = v.map((x, i) => x * idf[i]);
    const norm = Math.sqrt(tfidf.reduce((s, x) => s + x * x, 0)) || 1e-9;
    return tfidf.map(x => x / norm);
  });
}

function cosineSim(a, b) {
  if (!a.length || !b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ── Stage 0: Normalize ────────────────────────────────────────────────────
function normalize(text) {
  let t = text;
  for (const [pat, rep] of FILLER_RULES) t = t.replace(pat, rep);
  t = t.replace(/\n{3,}/g, '\n\n').replace(/  +/g, ' ').trim();
  return t;
}

// ── Stage 1: Sentence split ───────────────────────────────────────────────
function splitSentences(text) {
  return text.split(/(?<=[.!?])\s+|\n+/)
    .map(s => s.trim())
    .filter(s => s.length > 8);
}

// ── Stage 2: Dedup ────────────────────────────────────────────────────────
function dedup(sentences, threshold) {
  if (sentences.length <= 1) return sentences;
  const matrix = buildTFIDF(sentences);
  const removed = new Set();

  for (let i = 0; i < sentences.length; i++) {
    if (removed.has(i)) continue;
    for (let j = i + 1; j < sentences.length; j++) {
      if (removed.has(j)) continue;
      const sim = cosineSim(matrix[i], matrix[j]);
      if (sim >= threshold) {
        const iProt = isProtected(sentences[i]);
        const jProt = isProtected(sentences[j]);
        let victim;
        if (iProt && jProt)  continue;
        else if (iProt)      victim = j;
        else if (jProt)      victim = i;
        else                 victim = sentences[i].length < sentences[j].length ? i : j;
        removed.add(victim);
      }
    }
  }
  return sentences.filter((_, i) => !removed.has(i));
}

// ── Stage 3: Query scorer ─────────────────────────────────────────────────
function scoreByQuery(sentences, query, minRelevance = 0.12) {
  if (!query || !query.trim()) return sentences;
  const docs = [query, ...sentences];
  const matrix = buildTFIDF(docs);
  if (!matrix.length) return sentences;
  const qVec = matrix[0];
  const scored = sentences.map((s, i) => ({ s, score: cosineSim(qVec, matrix[i + 1]) }));
  const kept = scored.filter(x => x.score >= minRelevance).map(x => x.s);
  return kept.length > 0 ? kept : sentences;
}

// ── Token counter ─────────────────────────────────────────────────────────
function countTokens(text) {
  return Math.max(1, Math.round((text.match(/\S+/g) || []).length * 1.33));
}

// ── OptimizeResult ────────────────────────────────────────────────────────
class OptimizeResult {
  constructor({ optimizedText, originalTokens, optimizedTokens, elapsedMs,
                normSaved = 0, dedupRemoved = 0, scorerDropped = 0 }) {
    this.optimizedText = optimizedText;
    this.originalTokens = originalTokens;
    this.optimizedTokens = optimizedTokens;
    this.elapsedMs = elapsedMs;
    this.normalizationTokensSaved = normSaved;
    this.dedupSentencesRemoved = dedupRemoved;
    this.scorerSentencesDropped = scorerDropped;
  }

  get savedTokens() {
    return Math.max(0, this.originalTokens - this.optimizedTokens);
  }

  get reductionPercent() {
    if (this.originalTokens === 0) return 0;
    return Math.round((1 - this.optimizedTokens / this.originalTokens) * 1000) / 10;
  }

  get estimatedSavingsUsd() {
    // Claude Sonnet pricing: $3 per 1M input tokens
    return (this.savedTokens / 1_000_000) * 3.0;
  }

  summary() {
    return `TokenMesh | ${this.originalTokens} → ${this.optimizedTokens} tokens ` +
      `(${this.reductionPercent}% reduction) | ` +
      `$${this.estimatedSavingsUsd.toFixed(5)} saved | ` +
      `${this.elapsedMs.toFixed(1)}ms`;
  }
}

// ── Main class ────────────────────────────────────────────────────────────
class TokenMesh {
  /**
   * @param {object} options
   * @param {number} [options.dedupThreshold=0.28] - TF-IDF similarity threshold (0–1). Lower = more aggressive.
   * @param {number} [options.minRelevance=0.12]   - Minimum query relevance score to keep a sentence.
   * @param {number|null} [options.topK=null]       - Keep only top-K sentences when query provided.
   * @param {boolean} [options.normalize=true]      - Strip filler phrases (Stage 0).
   */
  constructor({ dedupThreshold = 0.28, minRelevance = 0.12, topK = null, normalize: doNorm = true } = {}) {
    this.dedupThreshold = dedupThreshold;
    this.minRelevance = minRelevance;
    this.topK = topK;
    this.doNorm = doNorm;
  }

  /**
   * Optimize text by removing duplicate and irrelevant content.
   *
   * @param {string} text    - Text to optimize (system prompt, document, etc.)
   * @param {object} options
   * @param {string} [options.query='']  - Optional query for Stage 3 relevance scoring.
   * @returns {OptimizeResult}
   *
   * @example
   * const result = tm.optimize(longText, { query: 'stop loss rules' });
   * console.log(result.optimizedText);  // send to Claude
   * console.log(result.reductionPercent + '%');
   */
  optimize(text, { query = '' } = {}) {
    const t0 = performance.now();
    const originalTokens = countTokens(text);

    // Stage 0: Normalize
    const normalized = this.doNorm ? normalize(text) : text;
    const normSaved = Math.max(0, originalTokens - countTokens(normalized));

    // Stage 1: Split to sentences
    const allSentences = splitSentences(normalized);

    // Stage 2: Dedup
    const afterDedup = dedup(allSentences, this.dedupThreshold);
    const dedupRemoved = allSentences.length - afterDedup.length;

    // Stage 3: Query score (only if query provided)
    let final = afterDedup;
    let scorerDropped = 0;
    if (query && query.trim()) {
      if (this.topK !== null) {
        // top-K mode
        const docs = [query, ...afterDedup];
        const matrix = buildTFIDF(docs);
        const qVec = matrix[0];
        const scored = afterDedup.map((s, i) => ({ s, score: cosineSim(qVec, matrix[i + 1]) }));
        scored.sort((a, b) => b.score - a.score);
        final = scored.slice(0, this.topK).map(x => x.s);
      } else {
        final = scoreByQuery(afterDedup, query, this.minRelevance);
      }
      scorerDropped = afterDedup.length - final.length;
    }

    const optimizedText = final.join('\n');
    const optimizedTokens = countTokens(optimizedText);
    const elapsedMs = performance.now() - t0;

    return new OptimizeResult({
      optimizedText, originalTokens, optimizedTokens, elapsedMs,
      normSaved, dedupRemoved, scorerDropped,
    });
  }
}

// ── Preset factories ──────────────────────────────────────────────────────

/**
 * Lite mode — safe, conservative (20–40% reduction).
 * Best for system prompts, financial/legal instructions.
 */
function TokenMeshLite(options = {}) {
  return new TokenMesh({ dedupThreshold: 0.28, minRelevance: 0.10, ...options });
}

/**
 * Aggressive mode — maximum compression (30–55% reduction in JS, 55–75% in Python).
 * Best for RAG context, pasted articles, conversation history.
 */
function TokenMeshAggressive(options = {}) {
  return new TokenMesh({ dedupThreshold: 0.20, minRelevance: 0.15, ...options });
}

// ── Exports ───────────────────────────────────────────────────────────────
module.exports = {
  TokenMesh,
  TokenMeshLite,
  TokenMeshAggressive,
  OptimizeResult,
  // Internal utilities (for advanced use)
  normalize,
  splitSentences,
  countTokens,
};
