import { describe, expect, it } from 'vitest';
import { API_PRICES, MODEL_TO_API_PRICE } from '../../data/apiPrices';
import { GPUS } from '../../data/gpus';
import { getModel, MODELS } from '../../data/models';
import { REGIONS } from '../../data/regions';
import {
  GB,
  kvBytesPerToken,
  kvCacheBytes,
  linearAttentionStateBytes,
  usableVramBytes,
} from '../memory';
import { QUANTIZATIONS, getKvQuant } from '../quant';

const kvFp16 = getKvQuant('fp16');

describe('model catalog integrity', () => {
  it('has unique ids', () => {
    expect(new Set(MODELS.map((m) => m.id)).size).toBe(MODELS.length);
  });

  it('never lets MoE active params exceed total params', () => {
    for (const m of MODELS) {
      if (m.paramsActive !== undefined) {
        expect(m.paramsActive, m.id).toBeLessThan(m.paramsTotal);
        expect(m.paramsActive, m.id).toBeGreaterThan(0);
      }
    }
  });

  it('gives every MoE model an active param count and vice versa', () => {
    for (const m of MODELS) {
      if (m.moe) expect(m.paramsActive, `${m.id} is MoE but has no paramsActive`).toBeDefined();
      if (m.paramsActive) expect(m.moe, `${m.id} has paramsActive but no moe spec`).toBeDefined();
    }
  });

  it('gives every MLA model the latent dimensions its cache formula needs', () => {
    for (const m of MODELS.filter((x) => x.attention === 'mla')) {
      expect(m.kvLoraRank, m.id).toBeGreaterThan(0);
      expect(m.qkRopeHeadDim, m.id).toBeGreaterThan(0);
      // Must not throw.
      expect(kvBytesPerToken(m, kvFp16)).toBeGreaterThan(0);
    }
  });

  it('labels attention consistently with the head counts', () => {
    for (const m of MODELS) {
      if (m.attention === 'gqa') {
        expect(m.numKeyValueHeads, m.id).toBeLessThan(m.numAttentionHeads);
      }
      if (m.attention === 'mha') {
        expect(m.numKeyValueHeads, m.id).toBe(m.numAttentionHeads);
      }
    }
  });

  it('keeps top-k below the expert count', () => {
    for (const m of MODELS.filter((x) => x.moe)) {
      expect(m.moe!.expertsPerToken, m.id).toBeLessThan(m.moe!.numExperts);
    }
  });

  it('has plausible architecture dimensions everywhere', () => {
    for (const m of MODELS) {
      expect(m.numLayers, m.id).toBeGreaterThan(0);
      expect(m.hiddenSize, m.id).toBeGreaterThan(0);
      expect(m.headDim, m.id).toBeGreaterThan(0);
      expect(m.vocabSize, m.id).toBeGreaterThan(0);
      expect(m.maxContext, m.id).toBeGreaterThan(0);
    }
  });
});

describe('sliding window attention', () => {
  it('applies the window to only half of gpt-oss layers', () => {
    const m = getModel('gpt-oss-120b')!;
    expect(m.slidingWindowLayerRatio).toBe(0.5);

    // At 128k context the 18 global layers dominate; the 18 windowed layers
    // are pinned at 128 tokens.
    const full = kvCacheBytes({ ...m, slidingWindow: null }, kvFp16, 131072, 1);
    const windowed = kvCacheBytes(m, kvFp16, 131072, 1);
    expect(windowed).toBeLessThan(full);
    expect(windowed / full).toBeCloseTo(0.5, 1);
  });

  it('keeps Gemma 3 cheaper at long context thanks to its 5-in-6 window', () => {
    const m = getModel('gemma-3-27b')!;
    const windowed = kvCacheBytes(m, kvFp16, 131072, 1);
    const full = kvCacheBytes({ ...m, slidingWindow: null }, kvFp16, 131072, 1);
    expect(windowed / full).toBeLessThan(0.25);
  });

  it('makes no difference below the window size', () => {
    const m = getModel('gemma-2-9b')!;
    const short = kvCacheBytes(m, kvFp16, 1024, 1);
    const full = kvCacheBytes({ ...m, slidingWindow: null }, kvFp16, 1024, 1);
    expect(short).toBeCloseTo(full, 0);
  });
});

describe('hybrid linear attention', () => {
  it('caches on only one layer in four for Qwen3.6 35B-A3B', () => {
    const m = getModel('qwen3.6-35b-a3b')!;
    expect(m.kvCacheLayerRatio).toBeCloseTo(0.25, 3);
  });

  it('makes a hybrid model far cheaper at long context than its layer count implies', () => {
    const m = getModel('qwen3.6-35b-a3b')!;
    const hybrid = kvCacheBytes(m, kvFp16, 262144, 1);
    const asIfAllLayersCached = kvCacheBytes(
      { ...m, kvCacheLayerRatio: undefined },
      kvFp16,
      262144,
      1,
    );
    expect(hybrid / asIfAllLayersCached).toBeCloseTo(0.25, 2);
  });

  it('charges the recurrent state that linear layers still carry', () => {
    // Linear attention grows no cache, but its matrix-valued state is not
    // free. Reducing the growing term without adding this one understated
    // hybrid models, badly so at large batch.
    const m = getModel('qwen3.6-35b-a3b')!;
    const state = linearAttentionStateBytes(m, kvFp16, 1);
    expect(state).toBeGreaterThan(0);

    // Constant in sequence length...
    expect(linearAttentionStateBytes(m, kvFp16, 1)).toBe(state);
    // ...but linear in batch.
    expect(linearAttentionStateBytes(m, kvFp16, 32)).toBeCloseTo(state * 32, 0);
  });

  it('adds no recurrent state to a purely softmax model', () => {
    expect(linearAttentionStateBytes(getModel('llama-3.1-8b')!, kvFp16, 8)).toBe(0);
  });

  it('makes the recurrent state visible in the total at large batch', () => {
    const m = getModel('qwen3.6-35b-a3b')!;
    const withState = kvCacheBytes(m, kvFp16, 1024, 64);
    const withoutState = kvCacheBytes({ ...m, kvCacheLayerRatio: undefined }, kvFp16, 1024, 64);
    // Even though the hybrid caches on a quarter of the layers, the state
    // keeps it from being a clean quarter of the dense figure.
    expect(withState).toBeGreaterThan(withoutState * 0.25);
  });

  it('applies the same treatment to every hybrid model in the catalog', () => {
    const hybrids = MODELS.filter((m) => m.kvCacheLayerRatio !== undefined);
    expect(hybrids.length).toBeGreaterThan(0);
    for (const m of hybrids) {
      expect(m.kvCacheLayerRatio, m.id).toBeGreaterThan(0);
      expect(m.kvCacheLayerRatio, m.id).toBeLessThan(1);
    }
  });
});

describe('sliding window flags', () => {
  it('does not treat Qwen2.5 as windowed — use_sliding_window is false', () => {
    // Qwen2.5 configs carry `sliding_window: 131072` with the feature disabled.
    // Honouring the value would understate the KV cache several-fold.
    for (const m of MODELS.filter((x) => x.family === 'Qwen' && x.name.includes('2.5'))) {
      expect(m.slidingWindow ?? null, m.id).toBeNull();
    }
  });

  it('keeps the window on models that genuinely use one', () => {
    expect(getModel('gemma-2-9b')!.slidingWindow).toBe(4096);
    expect(getModel('gpt-oss-120b')!.slidingWindow).toBe(128);
  });
});

describe('MLA head dimensions', () => {
  it('uses v_head_dim rather than hidden/heads for DeepSeek V3', () => {
    // hidden 7168 / 128 heads = 56, but the real per-head width is 128.
    const m = getModel('deepseek-v3')!;
    expect(m.headDim).toBe(128);
    expect(m.hiddenSize / m.numAttentionHeads).toBe(56);
  });
});

describe('active parameter provenance', () => {
  it('uses the published 37B for DeepSeek V3 rather than the inflated estimate', () => {
    // The safetensors total includes a ~14B MTP head, which pushes the
    // architectural estimate to ~54B.
    const m = getModel('deepseek-v3')!;
    expect(m.paramsActive).toBeCloseTo(37e9, -9);
    expect(m.paramsActiveEstimated).toBeUndefined();
  });

  it('trusts the count encoded in the model name', () => {
    expect(getModel('qwen3-235b-a22b')!.paramsActive).toBeCloseTo(22e9, -9);
    expect(getModel('qwen3-235b-a22b')!.paramsActiveEstimated).toBeUndefined();
  });

  it('flags every remaining MoE estimate so it is never mistaken for published', () => {
    const estimated = MODELS.filter((m) => m.paramsActiveEstimated);
    expect(estimated.length).toBeGreaterThan(0);
    for (const m of estimated) expect(m.moe, m.id).toBeDefined();
  });
});

describe('catalog coverage', () => {
  it('covers the major open-weight families', () => {
    const families = new Set(MODELS.map((m) => m.family));
    for (const f of ['DeepSeek', 'GLM', 'Qwen', 'Moonshot', 'MiniMax', 'OpenAI', 'Llama',
      'Google', 'Mistral']) {
      expect(families.has(f), `missing family ${f}`).toBe(true);
    }
    expect(MODELS.length).toBeGreaterThanOrEqual(70);
  });

  it('includes the current frontier open-weight models', () => {
    for (const id of ['deepseek-v4-flash', 'glm-5.2', 'kimi-k2.6', 'minimax-m3',
      'qwen3.6-35b-a3b', 'gpt-oss-120b']) {
      expect(getModel(id), `missing ${id}`).toBeDefined();
    }
  });
});

describe('headline sanity checks', () => {
  it('makes DeepSeek V3 cheaper per token of context than Llama 3.3 70B', () => {
    // 671B MLA vs 70B GQA: the big model caches less. This is the single most
    // counter-intuitive result the calculator produces, so it is pinned here.
    const ds = kvBytesPerToken(getModel('deepseek-v3')!, kvFp16);
    const llama = kvBytesPerToken(getModel('llama-3.3-70b')!, kvFp16);
    expect(ds).toBeLessThan(llama);
  });

  it('needs about 2.1 GB of KV cache for Kimi K2 at 32k context', () => {
    const bytes = kvCacheBytes(getModel('kimi-k2')!, kvFp16, 32768, 1);
    expect(bytes / GB).toBeCloseTo(2.145, 1);
  });
});

describe('gpu catalog integrity', () => {
  it('has unique ids and positive specs', () => {
    expect(new Set(GPUS.map((g) => g.id)).size).toBe(GPUS.length);
    for (const g of GPUS) {
      expect(g.vramGb, g.id).toBeGreaterThan(0);
      expect(g.bandwidthGBs, g.id).toBeGreaterThan(0);
      expect(g.fp16TFlops, g.id).toBeGreaterThan(0);
      expect(g.tdpW, g.id).toBeGreaterThan(g.idleW);
    }
  });

  it('orders precision throughput fp16 <= fp8 <= fp4', () => {
    for (const g of GPUS) {
      if (g.fp8TFlops) expect(g.fp8TFlops, g.id).toBeGreaterThanOrEqual(g.fp16TFlops);
      if (g.fp4TFlops && g.fp8TFlops) {
        expect(g.fp4TFlops, g.id).toBeGreaterThanOrEqual(g.fp8TFlops);
      }
    }
  });

  it('uses dense H100 figures, not the sparse marketing numbers', () => {
    const h100 = GPUS.find((g) => g.id === 'h100-sxm')!;
    // NVIDIA headlines 1979 TFLOPS FP16 with sparsity; dense is 989.
    expect(h100.fp16TFlops).toBe(989);
    expect(h100.fp8TFlops).toBe(1979);
    // Reproduces the ~295 FLOP/byte ridge point in the thesis design.
    expect((h100.fp16TFlops * 1e12) / (h100.bandwidthGBs * 1e9)).toBeCloseTo(295.2, 0);
  });

  it('marks Apple Silicon as unified memory', () => {
    for (const g of GPUS.filter((x) => x.vendor === 'apple')) {
      expect(g.unified, g.id).toBe(true);
    }
  });

  it('marks every integrated part as an SoC', () => {
    // The CPU shares the die and the power budget, so the host model must not
    // add a discrete CPU on top of the GPU's TDP.
    for (const g of GPUS.filter((x) => x.vendor === 'apple')) {
      expect(g.soc, g.id).toBe(true);
    }
    expect(GPUS.find((g) => g.id === 'dgx-spark-gb10')!.soc).toBe(true);
  });

  it('does not impose Apple’s Metal cap on Grace-Blackwell', () => {
    // Metal defaults to ~75% of unified memory; the coherent memory on DGX
    // Spark hands over nearly the whole pool under Linux. Treating both the
    // same understated the Spark by about 25 GB.
    const apple = GPUS.find((g) => g.id === 'm3-ultra')!;
    const spark = GPUS.find((g) => g.id === 'dgx-spark-gb10')!;

    expect(apple.usableMemoryFraction).toBeCloseTo(0.75, 2);
    expect(spark.usableMemoryFraction).toBeGreaterThan(0.9);

    expect(usableVramBytes(spark.vramGb, 1, spark.usableMemoryFraction) / GB).toBeGreaterThan(
      120,
    );
  });

  it('gives discrete cards their full VRAM', () => {
    for (const g of GPUS.filter((x) => !x.unified)) {
      expect(usableVramBytes(g.vramGb, 1, g.usableMemoryFraction) / GB).toBeCloseTo(
        g.vramGb,
        3,
      );
    }
  });
});

describe('quantization table', () => {
  it('has unique ids and descending bpw within GGUF', () => {
    expect(new Set(QUANTIZATIONS.map((q) => q.id)).size).toBe(QUANTIZATIONS.length);
    for (const q of QUANTIZATIONS) {
      expect(q.bpw, q.id).toBeGreaterThan(0);
      expect(q.bpw, q.id).toBeLessThanOrEqual(32);
    }
  });

  it('uses the measured file-level bpw for mixed GGUF types, not the block size', () => {
    const q4km = QUANTIZATIONS.find((q) => q.id === 'q4_k_m')!;
    // The Q4_K block structure implies 4.5 bpw, but _M promotes some tensors
    // to Q6_K/Q8_0, landing the file at 4.89.
    expect(q4km.bpw).toBeCloseTo(4.8944, 4);
    expect(q4km.bpw).toBeGreaterThan(4.5);
  });
});

describe('pricing and regions', () => {
  it('has unique region and price ids', () => {
    expect(new Set(REGIONS.map((r) => r.id)).size).toBe(REGIONS.length);
    expect(new Set(API_PRICES.map((p) => p.id)).size).toBe(API_PRICES.length);
  });

  it('prices output at least as high as input everywhere', () => {
    for (const p of API_PRICES) {
      expect(p.outputPerMTokUsd, p.id).toBeGreaterThanOrEqual(p.inputPerMTokUsd);
    }
  });

  it('resolves every model->API mapping to a real model and a real price', () => {
    for (const [modelId, priceId] of Object.entries(MODEL_TO_API_PRICE)) {
      expect(getModel(modelId), `no such model: ${modelId}`).toBeDefined();
      expect(
        API_PRICES.find((p) => p.id === priceId),
        `no such price: ${priceId}`,
      ).toBeDefined();
    }
  });

  it('uses the German household price as documented by Eurostat', () => {
    const de = REGIONS.find((r) => r.id === 'de-household')!;
    expect(de.pricePerKWh).toBeCloseTo(0.3869, 4);
    expect(de.currency).toBe('EUR');
  });
});
