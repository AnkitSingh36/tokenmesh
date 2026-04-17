# 🧵 TokenMesh

**why send 8000 tokens when 3000 do trick**

[![npm](https://img.shields.io/npm/v/tokenmesh.svg)](https://npmjs.com/package/tokenmesh)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Semantic token optimizer for LLM prompts. Removes duplicate, irrelevant, and filler content before your API call — cutting 30–55% of tokens while preserving every unique instruction.

> **JS vs Python:** This package uses TF-IDF similarity (runs anywhere, no install).  
> The [Python library](https://pypi.org/project/tokenmesh) uses neural embeddings and achieves 57% reduction.

---

## Install

```bash
npm install tokenmesh
```

---

## Quick Start

```js
const { TokenMesh } = require('tokenmesh');
// ESM: import { TokenMesh } from 'tokenmesh';

const tm = new TokenMesh();
const result = tm.optimize(yourLongText, { query: 'stop loss rules' });

console.log(result.optimizedText);      // send this to Claude
console.log(result.reductionPercent);   // e.g. 30.3
console.log(result.summary());
// TokenMesh | 412 → 287 tokens (30.3% reduction) | $0.00038 saved | 2.9ms
```

---

## Two Modes

```js
const { TokenMeshLite, TokenMeshAggressive } = require('tokenmesh');

// Safe — 20–40% reduction. Best for system prompts, financial rules.
const result = TokenMeshLite().optimize(text);

// Maximum — 30–55% reduction. Best for RAG context, articles.
const result = TokenMeshAggressive().optimize(text, { query: 'key risks' });
```

---

## Custom Config

```js
const tm = new TokenMesh({
  dedupThreshold: 0.25,   // 0–1. Lower = more aggressive dedup
  minRelevance:   0.12,   // drop chunks below this query-relevance score
  topK:           null,   // keep top-K sentences (null = use minRelevance)
  normalize:      true,   // strip filler phrases (Stage 0)
});
```

---

## Result Object

```js
result.optimizedText          // string  — compressed text
result.originalTokens         // number  — tokens before
result.optimizedTokens        // number  — tokens after
result.reductionPercent       // number  — e.g. 30.3
result.savedTokens            // number  — tokens removed
result.estimatedSavingsUsd    // number  — cost saved (Sonnet $3/1M)
result.elapsedMs              // number  — pipeline time in ms
result.summary()              // string  — one-line report
```

---

## With the Claude SDK

```js
const Anthropic = require('@anthropic-ai/sdk');
const { TokenMeshLite } = require('tokenmesh');

const client = new Anthropic();
const tm = TokenMeshLite();

async function chat(systemPrompt, userMessage) {
  const result = tm.optimize(systemPrompt, { query: userMessage });
  console.log(`Saved ${result.savedTokens} tokens (${result.reductionPercent}%)`);

  return client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 1024,
    system: result.optimizedText,   // compressed
    messages: [{ role: 'user', content: userMessage }],
  });
}
```

---

## TypeScript

```ts
import { TokenMesh, OptimizeResult, TokenMeshOptions } from 'tokenmesh';

const tm = new TokenMesh({ dedupThreshold: 0.25 });
const result: OptimizeResult = tm.optimize(text, { query: 'risks' });
```

---

## Python version (higher accuracy)

```bash
pip install tokenmesh
```

```python
from tokenmesh import TokenMeshAggressive
result = TokenMeshAggressive().optimize(text, query="stop loss rules")
print(result.reduction_percent)   # 57% vs 30% in JS
```

The Python version uses `all-MiniLM-L6-v2` neural embeddings and detects semantic duplicates that TF-IDF misses.

---

## License

MIT
