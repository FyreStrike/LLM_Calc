import { describe, expect, it } from 'vitest';
import { activationBytes, GB, kvBytesPerToken, kvCacheBytes, weightBytes } from '../memory';
import { getKvQuant, getQuant } from '../quant';
import type { ModelSpec, Workload } from '../types';

const fp16 = getQuant('fp16');
const kvFp16 = getKvQuant('fp16');

// --- fixtures mirroring real config.json files -----------------------------

/** unsloth/Meta-Llama-3.1-8B-Instruct */
const llama8b: ModelSpec = {
  id: 'llama-3.1-8b',
  name: 'Llama 3.1 8B',
  paramsTotal: 8.03e9,
  numLayers: 32,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  headDim: 128,
  vocabSize: 128256,
  maxContext: 131072,
  attention: 'gqa',
};

/** Qwen/Qwen3-235B-A22B — GQA + MoE */
const qwen235b: ModelSpec = {
  id: 'qwen3-235b-a22b',
  name: 'Qwen3 235B A22B',
  paramsTotal: 235e9,
  paramsActive: 22e9,
  numLayers: 94,
  hiddenSize: 4096,
  numAttentionHeads: 64,
  numKeyValueHeads: 4,
  headDim: 128,
  vocabSize: 151936,
  maxContext: 40960,
  attention: 'gqa',
  moe: { numExperts: 128, expertsPerToken: 8 },
};

/** deepseek-ai/DeepSeek-V3 — MLA + MoE */
const deepseekV3: ModelSpec = {
  id: 'deepseek-v3',
  name: 'DeepSeek V3',
  paramsTotal: 671e9,
  paramsActive: 37e9,
  numLayers: 61,
  hiddenSize: 7168,
  numAttentionHeads: 128,
  // The config really does say 128 here — the MLA branch must ignore it.
  numKeyValueHeads: 128,
  headDim: 128,
  vocabSize: 129280,
  maxContext: 163840,
  attention: 'mla',
  kvLoraRank: 512,
  qkRopeHeadDim: 64,
  moe: { numExperts: 256, expertsPerToken: 8, numSharedExperts: 1, firstKDenseReplace: 3 },
};

describe('weight memory', () => {
  it('sizes Llama-3.1-8B at FP16 to about 16 GB', () => {
    const bytes = weightBytes(llama8b, fp16);
    expect(bytes / GB).toBeCloseTo(14.96, 1); // 8.03e9 * 2 bytes
  });

  it('uses total params for MoE, not active params', () => {
    // A 235B-A22B model needs 235B worth of weights resident, not 22B.
    const bytes = weightBytes(qwen235b, fp16);
    expect(bytes).toBeCloseTo(235e9 * 2, -9);
    expect(bytes).toBeGreaterThan(weightBytes({ ...qwen235b, paramsTotal: 22e9 }, fp16));
  });

  it('scales with bits per weight', () => {
    const q4 = getQuant('q4_k_m');
    const ratio = weightBytes(llama8b, q4) / weightBytes(llama8b, fp16);
    expect(ratio).toBeCloseTo(4.8944 / 16, 4);
  });
});

describe('KV cache — GQA', () => {
  it('gives Llama-3.1-8B 128 KiB per token at FP16', () => {
    // 2 * 8 kv heads * 128 head_dim * 32 layers * 2 bytes = 131072
    expect(kvBytesPerToken(llama8b, kvFp16)).toBe(131072);
  });

  it('gives Qwen3-235B-A22B about 6.3 GB at 32k context', () => {
    // 2 * 4 * 128 * 94 * 2 = 192512 bytes/token
    expect(kvBytesPerToken(qwen235b, kvFp16)).toBe(192512);
    const bytes = kvCacheBytes(qwen235b, kvFp16, 32768, 1);
    expect(bytes / GB).toBeCloseTo(5.875, 2);
  });

  it('halves when the KV cache is quantized to FP8', () => {
    const fp8 = getKvQuant('fp8');
    expect(kvBytesPerToken(llama8b, fp8)).toBe(kvBytesPerToken(llama8b, kvFp16) / 2);
  });

  it('scales linearly with batch size', () => {
    const one = kvCacheBytes(llama8b, kvFp16, 4096, 1);
    const eight = kvCacheBytes(llama8b, kvFp16, 4096, 8);
    expect(eight).toBe(one * 8);
  });

  it('stops growing past a sliding window', () => {
    const windowed: ModelSpec = { ...llama8b, slidingWindow: 4096 };
    const short = kvCacheBytes(windowed, kvFp16, 4096, 1);
    const long = kvCacheBytes(windowed, kvFp16, 131072, 1);
    expect(long).toBe(short);
  });
});

describe('KV cache — MLA', () => {
  it('gives DeepSeek-V3 70.3 KB per token at FP16', () => {
    // (512 + 64) * 61 * 2 = 70272 bytes/token — reproduces the widely cited
    // "70 KB/token" figure from the DeepSeek-V3 technical report.
    expect(kvBytesPerToken(deepseekV3, kvFp16)).toBe(70272);
  });

  it('needs about 2.15 GB at 32k context', () => {
    const bytes = kvCacheBytes(deepseekV3, kvFp16, 32768, 1);
    expect(bytes / GB).toBeCloseTo(2.145, 2);
  });

  it('REGRESSION: must not apply the GQA formula to an MLA model', () => {
    // DeepSeek-V3's config reports num_key_value_heads: 128. Reading that as
    // GQA yields ~4 MB/token instead of ~70 KB/token — a 57x overestimate that
    // would make the model look unrunnable at any context length.
    const naiveGqaBytes =
      2 * deepseekV3.numKeyValueHeads * deepseekV3.headDim * deepseekV3.numLayers * 2;
    expect(naiveGqaBytes).toBe(3997696);

    const actual = kvBytesPerToken(deepseekV3, kvFp16);
    expect(actual).toBeLessThan(naiveGqaBytes / 50);
    expect(naiveGqaBytes / actual).toBeCloseTo(56.9, 0);
  });

  it('refuses to size an MLA model with no kvLoraRank', () => {
    const broken: ModelSpec = { ...deepseekV3, kvLoraRank: undefined };
    expect(() => kvBytesPerToken(broken, kvFp16)).toThrow(/kvLoraRank/);
  });

  it('beats an equivalent GQA model per token despite being far larger', () => {
    // 671B MLA caches less per token than a 235B GQA model.
    expect(kvBytesPerToken(deepseekV3, kvFp16)).toBeLessThan(
      kvBytesPerToken(qwen235b, kvFp16),
    );
  });
});

describe('activations', () => {
  const workload: Workload = {
    contextLength: 8192,
    promptTokens: 4096,
    outputTokens: 512,
    batchSize: 1,
    runtime: 'vllm',
    prefillChunkTokens: 4096,
    allowOffload: false,
    hostRamBandwidthGBs: 90,
    mbu: 0.7,
    mfu: 0.4,
  };

  it('projects only the sampled position to logits under vLLM and llama.cpp', () => {
    const vllm = activationBytes(llama8b, workload);
    const llamacpp = activationBytes(llama8b, { ...workload, runtime: 'llamacpp' });
    expect(vllm).toBe(llamacpp);
  });

  it('projects every prefill position under HF Transformers', () => {
    // 4096 positions x 128256 vocab x 4 bytes = 2.1 GB of logits alone — the
    // classic Transformers prefill OOM.
    const hf = activationBytes(llama8b, { ...workload, runtime: 'transformers' });
    const vllm = activationBytes(llama8b, workload);
    expect(hf).toBeGreaterThan(vllm);
    expect(hf / GB).toBeGreaterThan(2);
  });

  it('is capped by the prefill chunk size', () => {
    const unchunked = activationBytes(llama8b, {
      ...workload,
      runtime: 'transformers',
      promptTokens: 131072,
      prefillChunkTokens: 131072,
    });
    const chunked = activationBytes(llama8b, {
      ...workload,
      runtime: 'transformers',
      promptTokens: 131072,
      prefillChunkTokens: 2048,
    });
    expect(chunked).toBeLessThan(unchunked / 10);
  });
});

export { deepseekV3, llama8b, qwen235b };
