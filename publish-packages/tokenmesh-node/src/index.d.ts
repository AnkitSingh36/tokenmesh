/**
 * TokenMesh TypeScript declarations
 */

export interface TokenMeshOptions {
  /** TF-IDF similarity threshold (0–1). Lower = more aggressive. Default: 0.28 */
  dedupThreshold?: number;
  /** Minimum query relevance score to keep a sentence. Default: 0.12 */
  minRelevance?: number;
  /** Keep only top-K sentences when query provided. Default: null */
  topK?: number | null;
  /** Strip filler phrases (Stage 0). Default: true */
  normalize?: boolean;
}

export interface OptimizeOptions {
  /** Optional query for Stage 3 relevance scoring */
  query?: string;
}

export declare class OptimizeResult {
  /** Compressed text ready to send to Claude */
  optimizedText: string;
  /** Token count before optimization */
  originalTokens: number;
  /** Token count after optimization */
  optimizedTokens: number;
  /** Pipeline latency in milliseconds */
  elapsedMs: number;
  /** Tokens removed by Stage 0 normalizer */
  normalizationTokensSaved: number;
  /** Sentences removed by Stage 2 dedup */
  dedupSentencesRemoved: number;
  /** Sentences dropped by Stage 3 scorer */
  scorerSentencesDropped: number;
  /** Tokens saved (originalTokens - optimizedTokens) */
  readonly savedTokens: number;
  /** Percentage reduction (0–100) */
  readonly reductionPercent: number;
  /** Estimated cost saved in USD (Claude Sonnet pricing) */
  readonly estimatedSavingsUsd: number;
  /** One-line summary string */
  summary(): string;
}

export declare class TokenMesh {
  constructor(options?: TokenMeshOptions);
  /**
   * Optimize text by removing duplicate and irrelevant content.
   * @param text - Text to optimize
   * @param options - Optional query for relevance scoring
   */
  optimize(text: string, options?: OptimizeOptions): OptimizeResult;
}

/** Lite preset — safe, 20–40% reduction. Best for system prompts. */
export declare function TokenMeshLite(options?: TokenMeshOptions): TokenMesh;

/** Aggressive preset — max compression, 30–55% in JS (55–75% in Python). */
export declare function TokenMeshAggressive(options?: TokenMeshOptions): TokenMesh;

/** Strip filler phrases and markdown artifacts from text */
export declare function normalize(text: string): string;

/** Split text into sentence array */
export declare function splitSentences(text: string): string[];

/** Estimate token count (1.33 tokens per word) */
export declare function countTokens(text: string): number;

export default TokenMesh;
