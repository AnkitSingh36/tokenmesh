# 🧵 TokenMesh

**Why send 8000 tokens when 3000 do the same job?**

[GitHub](https://github.com/AnkitSingh36/tokenmesh) · [Live Demo](https://ankitsingh36.github.io/tokenmesh/demo.html) 

---

## What is this?

TokenMesh is a Python library that **optimizes LLM prompts before sending them to APIs**.

It removes:

* duplicate instructions
* repeated content
* filler text

while **preserving meaning, constraints, and rules**.

---

## Why use it?

LLM prompts are often verbose and redundant, which:

* increases token cost
* slows responses
* hits context limits

TokenMesh helps you:

* reduce **40–75% tokens**
* lower API costs
* keep prompts clean and efficient

Same meaning → fewer tokens → better performance.

---

## Install

```bash
pip install tokenmesh
```

* Python 3.9+
* Works on Windows, Mac, Linux

---

## Quick Start

```python
from tokenmesh import TokenMesh

tm = TokenMesh()

result = tm.optimize(
    text=your_prompt,
    query="optional"
)

print(result.optimized_text)
print(result.reduction_percent)
```

---

## Claude Integration

```python
from tokenmesh.integrations.claude import TokenMeshClaude

client = TokenMeshClaude()

response = client.chat(
    system=long_prompt,
    user="your question",
    model="claude-sonnet-4"
)

print(response.content)
```

---

## Modes

**Lite (safe)**

* 20–40% reduction
* preserves strict rules

```python
from tokenmesh import TokenMeshLite
result = TokenMeshLite().optimize(text)
```

**Aggressive (max compression)**

* 55–75% reduction
* best for long content

```python
from tokenmesh import TokenMeshAggressive
result = TokenMeshAggressive().optimize(text, query="...")
```

---

## Example

Before: ~684 tokens
After: ~290 tokens

Same instructions. ~57% fewer tokens.

---

## License

MIT
