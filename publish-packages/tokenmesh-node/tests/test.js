/**
 * TokenMesh Node.js tests
 * Run: node tests/test.js
 */
const { TokenMesh, TokenMeshLite, TokenMeshAggressive, normalize, countTokens } = require('../src/index.js');

let passed = 0, failed = 0;

function test(label, fn) {
  try {
    fn();
    console.log(`  ✓ ${label}`);
    passed++;
  } catch (e) {
    console.log(`  ✗ FAIL: ${label}`);
    console.log(`    ${e.message}`);
    failed++;
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'Assertion failed');
}

const PROMPT = `You are an expert swing trading assistant for NSE and BSE equity markets.
You are a knowledgeable financial co-pilot helping retail Indian investors trade profitably.
Your role is to assist with trade analysis, scanner interpretation, and risk management.
You help Indian retail investors make smarter decisions in NSE and BSE equity markets.

Always apply strict risk management: every trade must have a defined stop loss.
Never let a losing trade run beyond the defined stop loss under any circumstances.
Risk management is non-negotiable — protecting capital is more important than profit.
Never enter a trade without a pre-defined stop loss and target ratio of at least 1:2.

When analyzing breakouts, always confirm with volume. A breakout without volume is a fake-out.
Look for at least 1.5x average volume on the breakout candle to confirm genuine interest.
Volume confirmation is mandatory before entering any breakout trade in NSE or BSE markets.`;

console.log('\nTokenMesh Node.js test suite');
console.log('='.repeat(50));

// ── Normalizer tests ──────────────────────────────────────────────────────
console.log('\nNormalizer:');
test('strips Furthermore at line start', () => {
  const r = normalize('Furthermore, always use a stop loss.');
  assert(!r.includes('Furthermore'), `Still contains 'Furthermore': ${r}`);
  assert(r.includes('always use a stop loss'), `Lost content: ${r}`);
});
test('preserves non-negotiable mid-sentence', () => {
  const r = normalize('Risk management is non-negotiable.');
  assert(r.includes('non-negotiable'), `Stripped mid-sentence: ${r}`);
});
test('strips markdown headers', () => {
  const r = normalize('### Trading Rules\nAlways use stops.');
  assert(!r.includes('###'), `Still has ###: ${r}`);
  assert(r.includes('Trading Rules'), `Lost content: ${r}`);
});
test('preserves 1:2 ratio', () => {
  const r = normalize('Never enter without a 1:2 R:R ratio.');
  assert(r.includes('1:2'), `Stripped 1:2: ${r}`);
});

// ── TokenMesh core ────────────────────────────────────────────────────────
console.log('\nTokenMesh.optimize():');
test('returns OptimizeResult with correct shape', () => {
  const tm = new TokenMesh();
  const r = tm.optimize(PROMPT);
  assert(typeof r.optimizedText === 'string', 'optimizedText not string');
  assert(typeof r.originalTokens === 'number', 'originalTokens not number');
  assert(typeof r.reductionPercent === 'number', 'reductionPercent not number');
  assert(typeof r.summary === 'function', 'summary not function');
});
test('optimizedTokens <= originalTokens', () => {
  const r = new TokenMesh().optimize(PROMPT);
  assert(r.optimizedTokens <= r.originalTokens,
    `optimized (${r.optimizedTokens}) > original (${r.originalTokens})`);
});
test('reductionPercent >= 0', () => {
  const r = new TokenMesh().optimize(PROMPT);
  assert(r.reductionPercent >= 0, `negative reduction: ${r.reductionPercent}`);
});
test('summary() returns string with %', () => {
  const r = new TokenMesh().optimize(PROMPT);
  const s = r.summary();
  assert(s.includes('%'), `No % in summary: ${s}`);
  assert(s.includes('TokenMesh'), `No 'TokenMesh' in summary: ${s}`);
});
test('1:2 rule preserved after aggressive optimize', () => {
  const r = new TokenMesh({ dedupThreshold: 0.20 }).optimize(PROMPT);
  assert(r.optimizedText.includes('1:2'), `1:2 rule was dropped`);
});
test('1.5x rule preserved after aggressive optimize', () => {
  const r = new TokenMesh({ dedupThreshold: 0.20 }).optimize(PROMPT);
  assert(r.optimizedText.includes('1.5x'), `1.5x rule was dropped`);
});

// ── Presets ───────────────────────────────────────────────────────────────
console.log('\nPresets:');
test('TokenMeshLite() has threshold 0.28', () => {
  const tm = TokenMeshLite();
  assert(tm.dedupThreshold === 0.28, `wrong threshold: ${tm.dedupThreshold}`);
});
test('TokenMeshAggressive() has threshold 0.20', () => {
  const tm = TokenMeshAggressive();
  assert(tm.dedupThreshold === 0.20, `wrong threshold: ${tm.dedupThreshold}`);
});
test('TokenMeshLite reduces tokens', () => {
  const r = TokenMeshLite().optimize(PROMPT);
  assert(r.optimizedTokens <= r.originalTokens, 'Lite mode added tokens');
});

// ── Query scoring ─────────────────────────────────────────────────────────
console.log('\nQuery scoring:');
test('query filters to relevant chunks', () => {
  const r = new TokenMesh({ minRelevance: 0.15 }).optimize(PROMPT, { query: 'volume breakout' });
  assert(r.optimizedText.length > 0, 'empty output with query');
  assert(r.optimizedTokens <= r.originalTokens, 'query mode added tokens');
});
test('empty query does not crash', () => {
  const r = new TokenMesh().optimize(PROMPT, { query: '' });
  assert(r.optimizedText.length > 0, 'empty output');
});

// ── Utilities ─────────────────────────────────────────────────────────────
console.log('\nUtilities:');
test('countTokens returns positive number', () => {
  assert(countTokens('Hello world') > 0, 'countTokens returned 0');
});
test('countTokens empty string', () => {
  assert(countTokens('') >= 0, 'countTokens returned negative');
});

// ── Summary ───────────────────────────────────────────────────────────────
console.log(`\n${'='.repeat(50)}`);
console.log(`${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
