import type { AttentionType, ModelSpec } from '../core/types';

/**
 * Import a model definition straight from a HuggingFace `config.json`.
 *
 * Both endpoints used here are public and CORS-open, so this works from a
 * static page with no backend and no token:
 *
 *   https://huggingface.co/{repo}/resolve/main/config.json
 *   https://huggingface.co/api/models/{repo}?expand=safetensors
 *
 * The second one matters: it reports true parameter counts per dtype, which is
 * far more reliable than reconstructing them from layer dimensions.
 */

const CONFIG_URL = (repo: string) =>
  `https://huggingface.co/${repo}/resolve/main/config.json`;
const SAFETENSORS_URL = (repo: string) =>
  `https://huggingface.co/api/models/${repo}?expand=safetensors`;

/** Raw config.json, loosely typed — field names vary wildly between families. */
interface RawConfig {
  model_type?: string;
  architectures?: string[];
  num_hidden_layers?: number;
  hidden_size?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  head_dim?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  intermediate_size?: number;
  tie_word_embeddings?: boolean;
  sliding_window?: number | null;
  sliding_window_pattern?: number;
  /** Qwen2.5 ships a sliding_window value but disables it with this flag. */
  use_sliding_window?: boolean;
  layer_types?: string[];
  /** Hybrid models: full-attention layer every N layers (Qwen3.5/3.6). */
  full_attention_interval?: number;
  /** Kimi Linear nests its layer map here. */
  linear_attn_config?: { full_attn_layers?: number[] };

  // MLA (DeepSeek V2/V3, Kimi K2)
  kv_lora_rank?: number;
  q_lora_rank?: number;
  qk_rope_head_dim?: number;
  qk_nope_head_dim?: number;
  v_head_dim?: number;

  // MoE — every family invented its own vocabulary for the same three concepts
  num_experts?: number;
  num_local_experts?: number;
  n_routed_experts?: number;
  moe_num_experts?: number;
  num_experts_per_tok?: number;
  num_experts_per_token?: number;
  moe_k?: number;
  n_shared_experts?: number;
  moe_num_shared_experts?: number;
  first_k_dense_replace?: number;
  moe_layer_start_index?: number;
  moe_intermediate_size?: number;

  /** Multi-token-prediction head — counted in safetensors totals but idle at decode. */
  num_nextn_predict_layers?: number;

  // Multimodal models nest the language config
  text_config?: RawConfig;
}

export type HfImportErrorKind = 'notFound' | 'gated' | 'network' | 'unsupported';

export class HfImportError extends Error {
  readonly kind: HfImportErrorKind;

  constructor(message: string, kind: HfImportErrorKind) {
    super(message);
    this.name = 'HfImportError';
    this.kind = kind;
  }
}

/**
 * Field-name normalization. Each family invented its own vocabulary for the
 * same three MoE concepts, so a parser that only knows one of them silently
 * mis-reads the other two as dense models.
 */
function readMoe(c: RawConfig) {
  const numExperts =
    c.num_experts ?? c.num_local_experts ?? c.n_routed_experts ?? c.moe_num_experts;
  const expertsPerToken = c.num_experts_per_tok ?? c.num_experts_per_token ?? c.moe_k;
  if (!numExperts || !expertsPerToken) return undefined;

  return {
    numExperts,
    expertsPerToken,
    numSharedExperts: c.n_shared_experts ?? c.moe_num_shared_experts,
    firstKDenseReplace: c.first_k_dense_replace ?? c.moe_layer_start_index,
  };
}

/**
 * Fraction of layers that keep a growing KV cache.
 *
 * Hybrid architectures interleave linear-attention layers (Gated DeltaNet in
 * Qwen3.5/3.6, KDA in Kimi Linear) whose state is constant in sequence length.
 * Only the softmax-attention layers cache, so counting all layers overstates
 * long-context memory by the interleave factor — 4x for Qwen3.5/3.6.
 */
function readKvCacheLayerRatio(c: RawConfig): number | undefined {
  if (Array.isArray(c.layer_types) && c.layer_types.length > 0) {
    const caching = c.layer_types.filter((t) => !t.includes('linear')).length;
    const ratio = caching / c.layer_types.length;
    return ratio < 1 ? ratio : undefined;
  }
  if (c.linear_attn_config?.full_attn_layers && c.num_hidden_layers) {
    return c.linear_attn_config.full_attn_layers.length / c.num_hidden_layers;
  }
  if (c.full_attention_interval && c.full_attention_interval > 1) {
    return 1 / c.full_attention_interval;
  }
  return undefined;
}

/**
 * Classify attention from config *structure*, never from the model name.
 *
 * The order matters. DeepSeek-V3 and Kimi K2 report `num_key_value_heads`
 * equal to `num_attention_heads`, which classifies as MHA under the obvious
 * rule — and produces a KV cache roughly 57x too large. The presence of
 * `kv_lora_rank` is the real signal, so it is checked first.
 */
export function detectAttention(c: RawConfig): AttentionType {
  if (c.kv_lora_rank) return 'mla';

  const heads = c.num_attention_heads ?? 0;
  const kvHeads = c.num_key_value_heads ?? heads;

  if (kvHeads === 1) return 'mqa';
  if (kvHeads < heads) return 'gqa';
  return 'mha';
}

/**
 * Share of layers using sliding-window attention.
 *
 * Gemma 3 gives `sliding_window_pattern: 6`, meaning one global layer in every
 * six. gpt-oss instead publishes an explicit `layer_types` array alternating
 * "sliding_attention" and "full_attention".
 */
function readSlidingWindowRatio(c: RawConfig): number | undefined {
  if (!isSlidingWindowActive(c)) return undefined;

  if (c.layer_types?.length) {
    const sliding = c.layer_types.filter((t) => t.includes('sliding')).length;
    return sliding / c.layer_types.length;
  }
  if (c.sliding_window_pattern && c.sliding_window_pattern > 1) {
    return (c.sliding_window_pattern - 1) / c.sliding_window_pattern;
  }
  return 1;
}

/**
 * Qwen2.5 publishes `sliding_window: 131072` alongside
 * `use_sliding_window: false` — the window is configured but switched off.
 * Honouring the value without the flag understates its KV cache badly.
 */
function isSlidingWindowActive(c: RawConfig): boolean {
  if (!c.sliding_window) return false;
  return c.use_sliding_window !== false;
}

/**
 * Estimate active parameters for an MoE model when the true figure is unknown.
 *
 * Splits the model into a shared backbone (attention + embeddings + dense
 * layers) and the expert FFNs, then counts only top-k of the latter. Expert
 * FFN params per layer are `3 * hidden * moe_intermediate` for a gated MLP.
 */
function estimateActiveParams(c: RawConfig, total: number): number | undefined {
  const moe = readMoe(c);
  if (!moe) return undefined;

  const layers = c.num_hidden_layers ?? 0;
  const hidden = c.hidden_size ?? 0;
  // Qwen3 and DeepSeek publish a separate expert width; Mixtral and gpt-oss
  // reuse `intermediate_size` for it. Missing the fallback silently drops the
  // active-parameter estimate for the whole Mixtral-convention family.
  const moeInter = c.moe_intermediate_size ?? c.intermediate_size;
  if (!layers || !hidden || !moeInter) return undefined;

  const denseLayers = moe.firstKDenseReplace ?? 0;
  const moeLayers = Math.max(0, layers - denseLayers);

  const perExpert = 3 * hidden * moeInter;
  const expertTotal = moeLayers * moe.numExperts * perExpert;
  const backbone = Math.max(0, total - expertTotal);

  const sharedExperts = moe.numSharedExperts ?? 0;
  const activeExperts = moe.expertsPerToken + sharedExperts;

  const active = backbone + moeLayers * activeExperts * perExpert;
  return active < total && active > 0 ? active : undefined;
}

async function fetchJson(url: string): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new HfImportError('Network request failed', 'network');
  }

  if (res.status === 401 || res.status === 403) {
    throw new HfImportError('Repository is gated', 'gated');
  }
  if (res.status === 404) {
    throw new HfImportError('Repository or config.json not found', 'notFound');
  }
  if (!res.ok) {
    throw new HfImportError(`HTTP ${res.status}`, 'network');
  }
  return res.json();
}

/** True parameter count from the safetensors index, if the repo exposes one. */
async function fetchParamCount(repo: string): Promise<number | undefined> {
  try {
    const meta = (await fetchJson(SAFETENSORS_URL(repo))) as {
      safetensors?: { total?: number };
    };
    return meta.safetensors?.total;
  } catch {
    // Non-fatal: fall back to estimating from architecture.
    return undefined;
  }
}

/** Reconstruct parameter count from layer dimensions when metadata is absent. */
function estimateTotalParams(c: RawConfig): number {
  const layers = c.num_hidden_layers ?? 0;
  const hidden = c.hidden_size ?? 0;
  const vocab = c.vocab_size ?? 0;
  const heads = c.num_attention_heads ?? 0;
  const kvHeads = c.num_key_value_heads ?? heads;
  const headDim = c.head_dim ?? (heads ? hidden / heads : 0);
  const inter = c.intermediate_size ?? 4 * hidden;

  const embed = vocab * hidden;
  const lmHead = c.tie_word_embeddings ? 0 : vocab * hidden;

  const attnPerLayer =
    hidden * heads * headDim + // q
    2 * hidden * kvHeads * headDim + // k, v
    heads * headDim * hidden; // o

  const moe = readMoe(c);
  const moeInter = c.moe_intermediate_size ?? inter;
  const denseLayers = moe ? (c.first_k_dense_replace ?? 0) : layers;
  const moeLayers = layers - denseLayers;

  const denseFfn = denseLayers * 3 * hidden * inter;
  const moeFfn = moe
    ? moeLayers * (moe.numExperts + (moe.numSharedExperts ?? 0)) * 3 * hidden * moeInter
    : 0;

  return embed + lmHead + layers * attnPerLayer + denseFfn + moeFfn;
}

export async function importFromHuggingFace(repoInput: string): Promise<ModelSpec> {
  const repo = repoInput
    .trim()
    .replace(/^https?:\/\/huggingface\.co\//, '')
    .replace(/\/(tree|blob)\/.*$/, '')
    .replace(/\/$/, '');

  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    throw new HfImportError(
      'Expected a repo id like "deepseek-ai/DeepSeek-V3"',
      'unsupported',
    );
  }

  const raw = (await fetchJson(CONFIG_URL(repo))) as RawConfig;
  // Multimodal repos nest the language model config one level down.
  const c: RawConfig = { ...raw, ...(raw.text_config ?? {}) };

  const layers = c.num_hidden_layers;
  const hidden = c.hidden_size;
  const heads = c.num_attention_heads;
  if (!layers || !hidden || !heads) {
    throw new HfImportError(
      'config.json is missing the core architecture fields',
      'unsupported',
    );
  }

  const attention = detectAttention(c);
  const kvHeads = c.num_key_value_heads ?? heads;
  // For MLA the per-head dimension is v_head_dim, not hidden/heads — DeepSeek-V3
  // has hidden 7168 over 128 heads, which would imply 56 rather than the real 128.
  const headDim = (attention === 'mla' ? c.v_head_dim : undefined) ?? c.head_dim ?? hidden / heads;

  const paramsTotal = (await fetchParamCount(repo)) ?? estimateTotalParams(c);
  const moe = readMoe(c);
  const paramsActive = moe ? estimateActiveParams(c, paramsTotal) : undefined;

  // The safetensors total counts auxiliary modules that never run during normal
  // decoding — DeepSeek-V3's MTP head (`num_nextn_predict_layers`) is ~14B of
  // its 684.5B. Those land in the backbone term and inflate the active-parameter
  // estimate, so it is flagged rather than presented as authoritative.
  const activeIsEstimate = paramsActive !== undefined;
  const hasAuxModules = (c.num_nextn_predict_layers ?? 0) > 0;

  return {
    id: `hf:${repo}`,
    name: repo.split('/')[1] ?? repo,
    family: repo.split('/')[0],
    paramsTotal,
    paramsActive,
    numLayers: layers,
    hiddenSize: hidden,
    numAttentionHeads: heads,
    numKeyValueHeads: kvHeads,
    headDim,
    vocabSize: c.vocab_size ?? 32000,
    maxContext: c.max_position_embeddings ?? 8192,
    attention,
    kvLoraRank: c.kv_lora_rank,
    qkRopeHeadDim: c.qk_rope_head_dim,
    moe,
    kvCacheLayerRatio: readKvCacheLayerRatio(c),
    slidingWindow: isSlidingWindowActive(c) ? c.sliding_window : null,
    slidingWindowLayerRatio: readSlidingWindowRatio(c),
    tieWordEmbeddings: c.tie_word_embeddings,
    source: 'hf',
    hfRepo: repo,
    note: activeIsEstimate
      ? hasAuxModules
        ? 'model.note.activeEstimatedAux'
        : 'model.note.activeEstimated'
      : undefined,
  };
}
