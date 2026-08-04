/**
 * Generates ModelSpec entries from real HuggingFace config.json files.
 * Never invents architecture numbers — anything it cannot read, it reports.
 *
 * Usage: node gen-models.mjs > models.generated.ts
 */

// [display name, family, repo to read config from, canonical repo for display]
const CANDIDATES = [
  // --- DeepSeek ---
  ['DeepSeek V4 Flash', 'DeepSeek', 'deepseek-ai/DeepSeek-V4-Flash-0731'],
  ['DeepSeek V4 Flash (base)', 'DeepSeek', 'deepseek-ai/DeepSeek-V4-Flash'],
  ['DeepSeek V4 Pro', 'DeepSeek', 'deepseek-ai/DeepSeek-V4-Pro'],
  ['DeepSeek V3.2', 'DeepSeek', 'deepseek-ai/DeepSeek-V3.2-Exp'],
  ['DeepSeek V3.1', 'DeepSeek', 'deepseek-ai/DeepSeek-V3.1'],
  ['DeepSeek V3', 'DeepSeek', 'deepseek-ai/DeepSeek-V3'],
  ['DeepSeek R1', 'DeepSeek', 'deepseek-ai/DeepSeek-R1'],
  ['DeepSeek V2 Lite', 'DeepSeek', 'deepseek-ai/DeepSeek-V2-Lite'],
  ['DeepSeek Coder V2 Lite', 'DeepSeek', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'],

  // --- Z.ai / GLM ---
  ['GLM-5.2', 'GLM', 'zai-org/GLM-5.2'],
  ['GLM-5.1', 'GLM', 'zai-org/GLM-5.1'],
  ['GLM-5', 'GLM', 'zai-org/GLM-5'],
  ['GLM-4.7', 'GLM', 'zai-org/GLM-4.7'],
  ['GLM-4.6', 'GLM', 'zai-org/GLM-4.6'],
  ['GLM-4.5', 'GLM', 'zai-org/GLM-4.5'],
  ['GLM-4.5-Air', 'GLM', 'zai-org/GLM-4.5-Air'],

  // --- Qwen (newest first) ---
  ['Qwen3.6 35B-A3B', 'Qwen', 'Qwen/Qwen3.6-35B-A3B'],
  ['Qwen3.5 9B', 'Qwen', 'Qwen/Qwen3.5-9B'],
  ['Qwen3.5 27B', 'Qwen', 'Qwen/Qwen3.5-27B'],
  ['Qwen3.5 122B-A10B', 'Qwen', 'Qwen/Qwen3.5-122B-A10B'],
  ['Qwen3 0.6B', 'Qwen', 'Qwen/Qwen3-0.6B'],
  ['Qwen3 1.7B', 'Qwen', 'Qwen/Qwen3-1.7B'],
  ['Qwen3 4B', 'Qwen', 'Qwen/Qwen3-4B'],
  ['Qwen3 8B', 'Qwen', 'Qwen/Qwen3-8B'],
  ['Qwen3 14B', 'Qwen', 'Qwen/Qwen3-14B'],
  ['Qwen3 32B', 'Qwen', 'Qwen/Qwen3-32B'],
  ['Qwen3 30B-A3B', 'Qwen', 'Qwen/Qwen3-30B-A3B'],
  ['Qwen3 235B-A22B', 'Qwen', 'Qwen/Qwen3-235B-A22B'],
  ['Qwen3 Coder 30B-A3B', 'Qwen', 'Qwen/Qwen3-Coder-30B-A3B-Instruct'],
  ['Qwen3 Coder 480B-A35B', 'Qwen', 'Qwen/Qwen3-Coder-480B-A35B-Instruct'],
  ['Qwen2.5 0.5B', 'Qwen', 'Qwen/Qwen2.5-0.5B-Instruct'],
  ['Qwen2.5 1.5B', 'Qwen', 'Qwen/Qwen2.5-1.5B-Instruct'],
  ['Qwen2.5 3B', 'Qwen', 'Qwen/Qwen2.5-3B-Instruct'],
  ['Qwen2.5 7B', 'Qwen', 'Qwen/Qwen2.5-7B-Instruct'],
  ['Qwen2.5 14B', 'Qwen', 'Qwen/Qwen2.5-14B-Instruct'],
  ['Qwen2.5 32B', 'Qwen', 'Qwen/Qwen2.5-32B-Instruct'],
  ['Qwen2.5 72B', 'Qwen', 'Qwen/Qwen2.5-72B-Instruct'],
  ['Qwen2.5 Coder 32B', 'Qwen', 'Qwen/Qwen2.5-Coder-32B-Instruct'],
  ['QwQ 32B', 'Qwen', 'Qwen/QwQ-32B'],

  // --- Moonshot / Kimi ---
  ['Kimi K2.6', 'Moonshot', 'moonshotai/Kimi-K2.6'],
  ['Kimi K2.5', 'Moonshot', 'moonshotai/Kimi-K2.5'],
  ['Kimi K2 Thinking', 'Moonshot', 'moonshotai/Kimi-K2-Thinking'],
  ['Kimi K2', 'Moonshot', 'moonshotai/Kimi-K2-Instruct'],
  ['Kimi Linear 48B-A3B', 'Moonshot', 'moonshotai/Kimi-Linear-48B-A3B-Instruct'],

  // --- MiniMax ---
  ['MiniMax M3', 'MiniMax', 'MiniMaxAI/MiniMax-M3'],
  ['MiniMax M2', 'MiniMax', 'MiniMaxAI/MiniMax-M2'],

  // --- OpenAI open weights ---
  ['gpt-oss 20B', 'OpenAI', 'openai/gpt-oss-20b'],
  ['gpt-oss 120B', 'OpenAI', 'openai/gpt-oss-120b'],

  // --- Meta ---
  ['Llama 3.2 1B', 'Llama', 'unsloth/Llama-3.2-1B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct'],
  ['Llama 3.2 3B', 'Llama', 'unsloth/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct'],
  ['Llama 3.1 8B', 'Llama', 'unsloth/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'],
  ['Llama 3.3 70B', 'Llama', 'unsloth/Llama-3.3-70B-Instruct', 'meta-llama/Llama-3.3-70B-Instruct'],
  ['Llama 3.1 405B', 'Llama', 'unsloth/Meta-Llama-3.1-405B-Instruct', 'meta-llama/Llama-3.1-405B-Instruct'],
  ['Llama 4 Scout 109B-A17B', 'Llama', 'unsloth/Llama-4-Scout-17B-16E-Instruct', 'meta-llama/Llama-4-Scout-17B-16E-Instruct'],
  ['Llama 4 Maverick 400B-A17B', 'Llama', 'unsloth/Llama-4-Maverick-17B-128E-Instruct', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'],

  // --- Google ---
  ['Gemma 4 31B', 'Google', 'unsloth/Gemma-4-31B-IT', 'google/gemma-4-31b-it'],
  ['Gemma 3 1B', 'Google', 'unsloth/gemma-3-1b-it', 'google/gemma-3-1b-it'],
  ['Gemma 3 4B', 'Google', 'unsloth/gemma-3-4b-it', 'google/gemma-3-4b-it'],
  ['Gemma 3 12B', 'Google', 'unsloth/gemma-3-12b-it', 'google/gemma-3-12b-it'],
  ['Gemma 3 27B', 'Google', 'unsloth/gemma-3-27b-it', 'google/gemma-3-27b-it'],
  ['Gemma 2 9B', 'Google', 'unsloth/gemma-2-9b-it', 'google/gemma-2-9b-it'],
  ['Gemma 2 27B', 'Google', 'unsloth/gemma-2-27b-it', 'google/gemma-2-27b-it'],

  // --- Mistral ---
  ['Mistral 7B v0.3', 'Mistral', 'mistralai/Mistral-7B-Instruct-v0.3'],
  ['Mistral Nemo 12B', 'Mistral', 'mistralai/Mistral-Nemo-Instruct-2407'],
  ['Mistral Small 24B', 'Mistral', 'mistralai/Mistral-Small-24B-Instruct-2501'],
  ['Mistral Small 3.2 24B', 'Mistral', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'],
  ['Mistral Large 123B', 'Mistral', 'mistralai/Mistral-Large-Instruct-2411'],
  ['Magistral Small 24B', 'Mistral', 'mistralai/Magistral-Small-2509'],
  ['Devstral Small 24B', 'Mistral', 'mistralai/Devstral-Small-2507'],
  ['Mixtral 8x7B', 'Mistral', 'mistralai/Mixtral-8x7B-Instruct-v0.1'],
  ['Mixtral 8x22B', 'Mistral', 'mistralai/Mixtral-8x22B-Instruct-v0.1'],

  // --- NVIDIA ---
  ['Nemotron 3 Super 120B-A12B', 'NVIDIA', 'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B'],
  ['Llama 3.3 Nemotron Super 49B', 'NVIDIA', 'nvidia/Llama-3_3-Nemotron-Super-49B-v1_5'],
  ['Nemotron Nano 9B v2', 'NVIDIA', 'nvidia/NVIDIA-Nemotron-Nano-9B-v2'],

  // --- Others ---
  ['Solar Open2 250B', 'Upstage', 'upstage/Solar-Open2-250B'],
  ['K-EXAONE 2.0 750B-A37B', 'LG AI', 'LGAI-EXAONE/K-EXAONE-2.0-750B-A37B'],
  ['LongCat Flash Lite', 'Meituan', 'meituan-longcat/LongCat-Flash-Lite-Sparse'],
  ['Granite 4.1 8B', 'IBM', 'ibm-granite/granite-4.1-8b'],
  ['Phi-4 14B', 'Microsoft', 'microsoft/phi-4'],
  ['Phi-4 mini 3.8B', 'Microsoft', 'microsoft/Phi-4-mini-instruct'],
  ['Command A 111B', 'Cohere', 'CohereLabs/c4ai-command-a-03-2025'],
  ['Seed OSS 36B', 'ByteDance', 'ByteDance-Seed/Seed-OSS-36B-Instruct'],
  ['ERNIE 4.5 300B-A47B', 'Baidu', 'baidu/ERNIE-4.5-300B-A47B-PT'],
  ['Hunyuan A13B', 'Tencent', 'tencent/Hunyuan-A13B-Instruct'],
  ['Apriel 1.5 15B', 'ServiceNow', 'ServiceNow-AI/Apriel-1.5-15b-Thinker'],
  ['OLMo 3 32B', 'AllenAI', 'allenai/Olmo-3-32B-Instruct'],
];

const slug = (s) =>
  s.toLowerCase().replace(/[^a-z0-9.]+/g, '-').replace(/^-|-$/g, '');

async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function readMoe(c) {
  const numExperts =
    c.num_experts ?? c.num_local_experts ?? c.n_routed_experts ?? c.moe_num_experts;
  const expertsPerToken =
    c.num_experts_per_tok ?? c.num_experts_per_token ?? c.moe_topk ?? c.moe_k;
  if (!numExperts || !expertsPerToken) return undefined;
  return {
    numExperts,
    expertsPerToken: Array.isArray(expertsPerToken) ? expertsPerToken[0] : expertsPerToken,
    numSharedExperts: c.n_shared_experts ?? c.moe_num_shared_experts ?? undefined,
    firstKDenseReplace: c.first_k_dense_replace ?? c.moe_layer_start_index ?? undefined,
  };
}

/** Qwen2.5 sets sliding_window but disables it via use_sliding_window. */
function swActive(c) {
  return Boolean(c.sliding_window) && c.use_sliding_window !== false;
}

/** Fraction of layers keeping a KV cache that grows with sequence length. */
function kvLayerRatio(c) {
  if (Array.isArray(c.layer_types) && c.layer_types.length) {
    const caching = c.layer_types.filter((t) => !String(t).includes('linear')).length;
    const r = caching / c.layer_types.length;
    return r < 1 ? r : undefined;
  }
  if (c.linear_attn_config?.full_attn_layers && c.num_hidden_layers) {
    return c.linear_attn_config.full_attn_layers.length / c.num_hidden_layers;
  }
  if (c.full_attention_interval > 1) return 1 / c.full_attention_interval;
  return undefined;
}

function detectAttention(c) {
  if (c.kv_lora_rank) return 'mla';
  const h = c.num_attention_heads ?? 0;
  const kv = c.num_key_value_heads ?? h;
  if (kv === 1) return 'mqa';
  if (kv < h) return 'gqa';
  return 'mha';
}

function slidingRatio(c) {
  if (!swActive(c)) return undefined;
  if (Array.isArray(c.layer_types) && c.layer_types.length) {
    const s = c.layer_types.filter((t) => String(t).includes('sliding')).length;
    return s / c.layer_types.length;
  }
  if (c.sliding_window_pattern > 1) {
    return (c.sliding_window_pattern - 1) / c.sliding_window_pattern;
  }
  return 1;
}

/** Active params from the naming convention "…-A22B" / "…A3B", which is exact. */
function activeFromName(name) {
  const m = name.match(/A(\d+(?:\.\d+)?)\s*B/i);
  return m ? parseFloat(m[1]) * 1e9 : undefined;
}

function estimateActive(c, total) {
  const moe = readMoe(c);
  if (!moe) return undefined;
  const L = c.num_hidden_layers ?? 0;
  const h = c.hidden_size ?? 0;
  const mi = c.moe_intermediate_size ?? c.intermediate_size;
  if (!L || !h || !mi) return undefined;
  const dense = moe.firstKDenseReplace ?? 0;
  const moeL = Math.max(0, L - dense);
  const perExpert = 3 * h * mi;
  const backbone = Math.max(0, total - moeL * moe.numExperts * perExpert);
  const act = backbone + moeL * (moe.expertsPerToken + (moe.numSharedExperts ?? 0)) * perExpert;
  return act > 0 && act < total ? act : undefined;
}

const results = [];
const failures = [];

for (const [name, family, repo, canonical] of CANDIDATES) {
  try {
    const raw = await getJson(`https://huggingface.co/${repo}/resolve/main/config.json`);
    const c = { ...raw, ...(raw.text_config ?? {}) };

    const L = c.num_hidden_layers;
    const h = c.hidden_size;
    const H = c.num_attention_heads;
    if (!L || !h || !H) throw new Error('missing core fields');

    let total;
    try {
      const meta = await getJson(`https://huggingface.co/api/models/${repo}?expand=safetensors`);
      total = meta?.safetensors?.total;
    } catch { /* fall through */ }
    if (!total) throw new Error('no safetensors param count');

    const moe = readMoe(c);
    const active = moe ? (activeFromName(name) ?? estimateActive(c, total)) : undefined;
    const attention = detectAttention(c);

    results.push({
      id: slug(name),
      name,
      family,
      paramsTotal: total,
      paramsActive: active,
      numLayers: L,
      hiddenSize: h,
      numAttentionHeads: H,
      numKeyValueHeads: c.num_key_value_heads ?? H,
      // MLA's per-head width is v_head_dim; hidden/heads gives 56 for
      // DeepSeek-V3 where the real value is 128.
      headDim:
        (attention === 'mla' ? c.v_head_dim : undefined) ?? c.head_dim ?? Math.round(h / H),
      vocabSize: c.vocab_size,
      maxContext: c.max_position_embeddings ?? c.model_max_length,
      attention,
      kvLoraRank: c.kv_lora_rank,
      qkRopeHeadDim: c.qk_rope_head_dim,
      moe,
      kvCacheLayerRatio: kvLayerRatio(c),
      slidingWindow: swActive(c) ? c.sliding_window : null,
      slidingWindowLayerRatio: slidingRatio(c),
      tieWordEmbeddings: c.tie_word_embeddings,
      hfRepo: canonical ?? repo,
      modelType: c.model_type,
    });
    process.stderr.write(`  ok   ${name}\n`);
  } catch (e) {
    failures.push([name, repo, e.message]);
    process.stderr.write(`  FAIL ${name} (${repo}): ${e.message}\n`);
  }
}

process.stderr.write(`\n${results.length} ok, ${failures.length} failed\n`);
console.log(JSON.stringify({ results, failures }, null, 1));
