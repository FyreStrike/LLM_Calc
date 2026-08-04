/** Reads models.json (from gen-models.mjs) and emits src/data/models.ts. */
import { readFileSync } from 'node:fs';

// PowerShell's `>` writes a UTF-8 BOM, which JSON.parse rejects.
const { results } = JSON.parse(readFileSync(process.argv[2], 'utf8').replace(/^﻿/, ''));

/**
 * Published active-parameter counts. Used in preference to the architectural
 * estimate, which runs high on models whose safetensors total includes an MTP
 * head or other auxiliary modules.
 */
const PUBLISHED_ACTIVE = {
  'deepseek-v3': 37e9, 'deepseek-v3.1': 37e9, 'deepseek-v3.2': 37e9, 'deepseek-r1': 37e9,
  'deepseek-v2-lite': 2.4e9, 'deepseek-coder-v2-lite': 2.4e9,
  'kimi-k2': 32e9, 'kimi-k2-thinking': 32e9,
  'gpt-oss-20b': 3.6e9, 'gpt-oss-120b': 5.1e9,
  'glm-4.5': 32e9, 'glm-4.5-air': 12e9, 'glm-4.6': 32e9, 'glm-4.7': 32e9,
  'mixtral-8x7b': 12.9e9, 'mixtral-8x22b': 39e9,
  'minimax-m2': 10e9,
};

/** Models whose sparse-attention behaviour this calculator does not model. */
const SPARSE_ATTENTION = new Set(['deepseek_v4', 'glm_moe_dsa']);

/** Drop: architectures the memory model cannot represent honestly. */
const DROP = new Set(['kimi-linear-48b-a3b']);

const nameHasActive = (n) => /A\d+(?:\.\d+)?\s*B/i.test(n);

const out = [];
for (const r of results) {
  if (DROP.has(r.id)) continue;
  if (!r.maxContext || !Number.isFinite(r.maxContext)) continue;

  const published = PUBLISHED_ACTIVE[r.id];
  const fromName = nameHasActive(r.name);
  const paramsActive = published ?? r.paramsActive;
  // Trustworthy when published or encoded in the model's own name.
  const estimated = r.moe && !published && !fromName;

  const sparse = SPARSE_ATTENTION.has(r.modelType);

  // Sparse-attention models publish a sliding_window that does not describe
  // what is actually retained. Fall back to the full-cache upper bound rather
  // than a wildly optimistic number.
  const slidingWindow = sparse ? null : r.slidingWindow;
  const slidingWindowLayerRatio = sparse ? undefined : r.slidingWindowLayerRatio;

  const note = sparse
    ? 'model.note.sparseAttention'
    : r.kvCacheLayerRatio && r.kvCacheLayerRatio < 1
      ? 'model.note.hybridAttention'
      : r.attention === 'mla'
        ? 'model.note.mla'
        : undefined;

  out.push({
    id: r.id,
    name: r.name,
    family: r.family,
    paramsTotal: r.paramsTotal,
    paramsActive,
    paramsActiveEstimated: estimated || undefined,
    numLayers: r.numLayers,
    hiddenSize: r.hiddenSize,
    numAttentionHeads: r.numAttentionHeads,
    numKeyValueHeads: r.numKeyValueHeads,
    headDim: r.headDim,
    vocabSize: r.vocabSize,
    maxContext: r.maxContext,
    attention: r.attention,
    kvLoraRank: r.kvLoraRank,
    qkRopeHeadDim: r.qkRopeHeadDim,
    moe: r.moe,
    kvCacheLayerRatio: r.kvCacheLayerRatio,
    slidingWindow,
    slidingWindowLayerRatio,
    tieWordEmbeddings: r.tieWordEmbeddings,
    source: 'catalog',
    hfRepo: r.hfRepo,
    note,
  });
}

const FAMILY_ORDER = ['DeepSeek', 'GLM', 'Qwen', 'Moonshot', 'MiniMax', 'OpenAI',
  'Llama', 'Google', 'Mistral', 'NVIDIA', 'Upstage', 'LG AI', 'IBM', 'Microsoft',
  'ByteDance', 'Baidu', 'Tencent', 'ServiceNow', 'Cohere', 'AllenAI', 'Meituan'];
out.sort((a, b) => {
  const fa = FAMILY_ORDER.indexOf(a.family), fb = FAMILY_ORDER.indexOf(b.family);
  return (fa < 0 ? 99 : fa) - (fb < 0 ? 99 : fb);
});

const fmtNum = (n) => {
  if (n === undefined) return undefined;
  if (n >= 1e9) return `${+(n / 1e9).toFixed(3)}e9`;
  if (n >= 1e6) return `${+(n / 1e6).toFixed(3)}e6`;
  return String(n);
};

const lines = [];
let lastFamily = null;
for (const m of out) {
  if (m.family !== lastFamily) {
    lines.push(`\n  // ${'-'.repeat(Math.max(4, 68 - m.family.length))} ${m.family}`);
    lastFamily = m.family;
  }
  const f = [];
  f.push(`    id: '${m.id}'`);
  f.push(`    name: '${m.name.replace(/'/g, "\\'")}'`);
  f.push(`    family: '${m.family}'`);
  f.push(`    paramsTotal: ${fmtNum(m.paramsTotal)}`);
  if (m.paramsActive) f.push(`    paramsActive: ${fmtNum(m.paramsActive)}`);
  if (m.paramsActiveEstimated) f.push(`    paramsActiveEstimated: true`);
  f.push(`    numLayers: ${m.numLayers}`);
  f.push(`    hiddenSize: ${m.hiddenSize}`);
  f.push(`    numAttentionHeads: ${m.numAttentionHeads}`);
  f.push(`    numKeyValueHeads: ${m.numKeyValueHeads}`);
  f.push(`    headDim: ${m.headDim}`);
  f.push(`    vocabSize: ${m.vocabSize}`);
  f.push(`    maxContext: ${m.maxContext}`);
  f.push(`    attention: '${m.attention}'`);
  if (m.kvLoraRank) f.push(`    kvLoraRank: ${m.kvLoraRank}`);
  if (m.qkRopeHeadDim) f.push(`    qkRopeHeadDim: ${m.qkRopeHeadDim}`);
  if (m.moe) {
    const e = [`numExperts: ${m.moe.numExperts}`, `expertsPerToken: ${m.moe.expertsPerToken}`];
    if (m.moe.numSharedExperts) e.push(`numSharedExperts: ${m.moe.numSharedExperts}`);
    if (m.moe.firstKDenseReplace) e.push(`firstKDenseReplace: ${m.moe.firstKDenseReplace}`);
    f.push(`    moe: { ${e.join(', ')} }`);
  }
  if (m.kvCacheLayerRatio) f.push(`    kvCacheLayerRatio: ${+m.kvCacheLayerRatio.toFixed(4)}`);
  if (m.slidingWindow) f.push(`    slidingWindow: ${m.slidingWindow}`);
  if (m.slidingWindowLayerRatio !== undefined && m.slidingWindowLayerRatio < 1) {
    f.push(`    slidingWindowLayerRatio: ${+m.slidingWindowLayerRatio.toFixed(4)}`);
  }
  if (m.tieWordEmbeddings) f.push(`    tieWordEmbeddings: true`);
  f.push(`    source: 'catalog'`);
  f.push(`    hfRepo: '${m.hfRepo}'`);
  if (m.note) f.push(`    note: '${m.note}'`);
  lines.push(`  {\n${f.join(',\n')},\n  },`);
}

console.log(`import type { ModelSpec } from '../core/types';

/**
 * Curated model catalog — ${out.length} models.
 *
 * GENERATED from real HuggingFace \`config.json\` files; do not hand-edit
 * architecture fields. Parameter counts come from the
 * \`?expand=safetensors\` API. Regenerate with the script in the project's
 * scratchpad when new models ship.
 *
 * Three things this file gets right that a name-based catalog cannot:
 *
 * - \`attention\` is derived from config structure. DeepSeek, GLM-5 and Kimi
 *   report \`num_key_value_heads == num_attention_heads\`, which reads as MHA;
 *   they are MLA, and the naive reading overstates the KV cache ~57x.
 * - \`kvCacheLayerRatio\` captures hybrid linear attention. Qwen3.5/3.6 keep one
 *   softmax-attention layer in every four; the rest carry an O(1) recurrent
 *   state and cache nothing that grows with context.
 * - \`slidingWindow\` respects \`use_sliding_window\`. Qwen2.5 ships a window
 *   value with the feature switched off.
 *
 * \`paramsActiveEstimated\` marks MoE models whose active-parameter count was
 * derived from the architecture rather than published or encoded in the name —
 * those run high when the safetensors total includes an MTP head.
 */
export const MODELS: ModelSpec[] = [${lines.join('\n')}
];

export function getModel(id: string): ModelSpec | undefined {
  return MODELS.find((m) => m.id === id);
}

export const MODEL_FAMILIES = Array.from(new Set(MODELS.map((m) => m.family ?? 'Other')));
`);
