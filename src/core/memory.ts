import { bytesPerParam } from './quant';
import type {
  KvQuantSpec,
  MemoryBreakdown,
  ModelSpec,
  QuantSpec,
  Runtime,
  Workload,
} from './types';

export const GB = 1024 ** 3;
export const MB = 1024 ** 2;

/** Fixed CUDA context cost per GPU. Does not scale with model size. */
export const CUDA_CONTEXT_BYTES = 400 * MB;

/**
 * Per-layer transient activation tensors held live during a forward pass:
 * Q, K, V, attention output, and the MLP gate/up/down intermediates. With
 * FlashAttention the O(S^2) score matrix is never materialized, so this stays
 * linear in sequence length.
 */
const ACTIVATION_TENSORS_PER_LAYER = 8;

/**
 * Weight storage.
 *
 * For MoE this is deliberately `paramsTotal`: every expert must be resident in
 * memory even though only `paramsActive` are read per token. Memory is driven
 * by total params, bandwidth and FLOPs by active params — conflating the two is
 * the most common error in MoE sizing.
 */
export function weightBytes(model: ModelSpec, quant: QuantSpec): number {
  return model.paramsTotal * bytesPerParam(quant);
}

/**
 * KV cache bytes per token, per sequence — the quantity that decides whether
 * long context is affordable.
 *
 * MHA / GQA / MQA:
 *   2 (K and V) * n_kv_heads * head_dim * n_layers * bytes_per_element
 *
 * MLA (DeepSeek V2/V3/R1):
 *   (kv_lora_rank + qk_rope_head_dim) * n_layers * bytes_per_element
 *
 * MLA caches a single compressed latent plus the decoupled RoPE key, *not*
 * per-head K and V — so there is no factor of 2 and no head count. DeepSeek-V3
 * reports `num_key_value_heads: 128` in its config, which would suggest full
 * MHA; applying the GQA formula there overstates the cache by roughly 57x
 * (~4 MB/token instead of the correct ~70 KB/token).
 */
export function kvBytesPerToken(model: ModelSpec, kvQuant: KvQuantSpec): number {
  const bytes = kvQuant.bytesPerElement;

  // Hybrid models only grow a cache on their softmax-attention layers; the
  // linear-attention layers carry a fixed-size recurrent state instead.
  const ratio = Math.min(1, Math.max(0, model.kvCacheLayerRatio ?? 1));
  const cachingLayers = model.numLayers * ratio;

  if (model.attention === 'mla') {
    const latent = model.kvLoraRank ?? 0;
    const rope = model.qkRopeHeadDim ?? 0;
    if (latent <= 0) {
      throw new Error(
        `Model ${model.id} is marked MLA but has no kvLoraRank; cannot size the KV cache.`,
      );
    }
    return (latent + rope) * cachingLayers * bytes;
  }

  return 2 * model.numKeyValueHeads * model.headDim * cachingLayers * bytes;
}

/**
 * Total KV cache across the batch.
 *
 * Sliding-window layers stop growing once the context passes the window, but
 * models rarely apply the window everywhere: Gemma 2 alternates local and
 * global layers, Gemma 3 keeps one global layer in every six, and gpt-oss
 * alternates. The global layers still hold the full context, and at 128k that
 * remainder dominates — so the layer ratio is not a detail that can be
 * rounded away.
 */
export function kvCacheBytes(
  model: ModelSpec,
  kvQuant: KvQuantSpec,
  contextLength: number,
  batchSize: number,
): number {
  const perTokenAllLayers = kvBytesPerToken(model, kvQuant);

  if (!model.slidingWindow) {
    return perTokenAllLayers * contextLength * batchSize;
  }

  // Distribute across the layers that actually cache, not all layers.
  const cachingLayers =
    model.numLayers * Math.min(1, Math.max(0, model.kvCacheLayerRatio ?? 1));
  const perTokenPerLayer = cachingLayers > 0 ? perTokenAllLayers / cachingLayers : 0;

  const ratio = Math.min(1, Math.max(0, model.slidingWindowLayerRatio ?? 1));
  const windowedLayers = cachingLayers * ratio;
  const globalLayers = cachingLayers - windowedLayers;
  const windowedContext = Math.min(contextLength, model.slidingWindow);

  return (
    (windowedLayers * windowedContext + globalLayers * contextLength) *
    perTokenPerLayer *
    batchSize
  );
}

/**
 * Activation working set at peak.
 *
 * The dominant term during prefill is usually the logits tensor, materialized
 * in FP32 as `batch * positions * vocab * 4`.
 *
 * How many positions get projected to logits depends on the runtime, and the
 * difference is large. vLLM and llama.cpp only project the position they are
 * about to sample from, so `positions = 1` per sequence. HF Transformers
 * projects every position in the forward pass — at a 4096-token prefill with a
 * 128k vocab that is 2.1 GB of logits alone, which is the classic and
 * frequently misdiagnosed Transformers OOM.
 */
export function activationBytes(model: ModelSpec, workload: Workload): number {
  const positions = Math.max(
    1,
    Math.min(workload.promptTokens, workload.prefillChunkTokens),
  );

  const logitPositions = workload.runtime === 'transformers' ? positions : 1;
  const logitsBytes = workload.batchSize * logitPositions * model.vocabSize * 4;

  const perLayerBytes =
    workload.batchSize *
    positions *
    model.hiddenSize *
    ACTIVATION_TENSORS_PER_LAYER *
    2; // transient tensors stay in bf16

  return logitsBytes + perLayerBytes;
}

/**
 * Runtime-specific overhead beyond weights, KV cache and activations.
 *
 * - llama.cpp is lean: a few hundred MB of compute buffer, less with `-fa`.
 * - vLLM captures CUDA graphs (1.5-2 GB) and preallocates to
 *   `gpu_memory_utilization` (default 0.90).
 * - transformers is the loosest: allocator fragmentation plus cuBLAS
 *   workspaces run 10-15% of the working set.
 */
export function frameworkOverheadBytes(
  runtime: Runtime,
  weightsAndKvBytes: number,
): number {
  switch (runtime) {
    case 'llamacpp':
      return 500 * MB;
    case 'vllm':
      return 2 * GB;
    case 'transformers':
      return 0.15 * weightsAndKvBytes;
  }
}

export function computeMemory(
  model: ModelSpec,
  quant: QuantSpec,
  kvQuant: KvQuantSpec,
  workload: Workload,
  numGpus: number,
): MemoryBreakdown {
  const weights = weightBytes(model, quant);
  const kv = kvCacheBytes(model, kvQuant, workload.contextLength, workload.batchSize);
  const activations = activationBytes(model, workload);
  const cudaContext = CUDA_CONTEXT_BYTES * numGpus;
  const framework = frameworkOverheadBytes(workload.runtime, weights + kv);

  return {
    weightsBytes: weights,
    kvCacheBytes: kv,
    activationsBytes: activations,
    cudaContextBytes: cudaContext,
    frameworkOverheadBytes: framework,
    totalBytes: weights + kv + activations + cudaContext + framework,
  };
}

/**
 * Memory the runtime can actually address.
 *
 * Apple's unified memory is shared with the OS: the Metal default caps GPU
 * allocation near 75% of total, raisable to ~92% via `iogpu.wired_limit_mb`.
 * Discrete GPUs expose essentially all of their VRAM.
 */
export function usableVramBytes(
  vramGb: number,
  numGpus: number,
  unified: boolean | undefined,
): number {
  const raw = vramGb * GB * numGpus;
  return unified ? raw * 0.75 : raw;
}
