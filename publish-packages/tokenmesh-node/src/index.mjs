/**
 * TokenMesh ESM entry point
 * Supports: import { TokenMesh } from 'tokenmesh'
 */
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const lib = require('./index.js');

export const { TokenMesh, TokenMeshLite, TokenMeshAggressive, OptimizeResult,
               normalize, splitSentences, countTokens } = lib;
export default lib.TokenMesh;
