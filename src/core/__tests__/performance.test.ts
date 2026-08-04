import { describe, expect, it } from 'vitest';
import { decodeParamsRead, offloadFraction } from '../performance';
import { getKvQuant, getQuant } from '../quant';
import { runCalculation } from '../index';
import { parallelEfficiency, ridgePoint } from '../roofline';
import type { CalcInput, GpuSpec, ModelSpec, Workload } from '../types';
import { deepseekV3, llama8b, qwen235b } from './memory.test';

const h100: GpuSpec = {
  id: 'h100-sxm',
  name: 'H100 SXM',
  vendor: 'nvidia',
  vramGb: 80,
  bandwidthGBs: 3350,
  fp16TFlops: 1979,
  fp8TFlops: 3958,
  tdpW: 700,
  idleW: 75,
  architecture: 'hopper',
  nvlink: true,
};

const rtx3060: GpuSpec = {
  id: 'rtx-3060-12gb',
  name: 'RTX 3060 12GB',
  vendor: 'nvidia',
  vramGb: 12,
  bandwidthGBs: 360,
  fp16TFlops: 51,
  tdpW: 170,
  idleW: 12,
  architecture: 'ampere',
};

const workload: Workload = {
  contextLength: 4096,
  promptTokens: 1024,
  outputTokens: 512,
  batchSize: 1,
  runtime: 'vllm',
  prefillChunkTokens: 2048,
  allowOffload: false,
  hostRamBandwidthGBs: 90,
  mbu: 0.7,
  mfu: 0.4,
};

function makeInput(model: ModelSpec, gpu: GpuSpec, over: Partial<CalcInput> = {}): CalcInput {
  return {
    model,
    hardware: { gpu, numGpus: 1, parallelism: 'tp' },
    quant: getQuant('fp16'),
    kvQuant: getKvQuant('fp16'),
    workload,
    energy: {
      mode: 'simple',
      psuEfficiency: 0.9,
      pue: 1.0,
      hostOverheadW: 80,
    },
    cost: {
      region: {
        id: 'de',
        label: 'Germany',
        pricePerKWh: 0.3869,
        currency: 'EUR',
        gridIntensityGCO2PerKWh: 380,
      },
      inputRatio: 0.7,
      usdToCurrency: 0.92,
    },
    ...over,
  };
}

describe('roofline', () => {
  it('computes the H100 FP16 ridge point at ~295 FLOP/byte', () => {
    // From the thesis research design: 989e12 / 3.35e12 ~ 295.2 FLOP/byte
    // (989 TFLOPS being the FP16-with-FP32-accumulate figure).
    expect(ridgePoint(989e12, 3.35e12)).toBeCloseTo(295.2, 1);
  });

  it('computes the B200 FP4 ridge point at 1125 FLOP/byte', () => {
    expect(ridgePoint(9000e12, 8.0e12)).toBeCloseTo(1125, 0);
  });

  it('penalizes PCIe tensor parallelism more than NVLink', () => {
    expect(parallelEfficiency(4, 'tp', true)).toBeGreaterThan(
      parallelEfficiency(4, 'tp', false),
    );
    expect(parallelEfficiency(1, 'tp', false)).toBe(1);
  });
});

describe('MoE parameter reads', () => {
  it('reads only the active params at batch 1', () => {
    const read = decodeParamsRead(qwen235b, 1);
    expect(read).toBeCloseTo(22e9, -9);
  });

  it('reads more as the batch activates a wider union of experts', () => {
    const b1 = decodeParamsRead(qwen235b, 1);
    const b32 = decodeParamsRead(qwen235b, 32);
    const b256 = decodeParamsRead(qwen235b, 256);
    expect(b32).toBeGreaterThan(b1);
    expect(b256).toBeGreaterThan(b32);
    expect(b256).toBeLessThanOrEqual(qwen235b.paramsTotal);
  });

  it('saturates at total params, never exceeding them', () => {
    expect(decodeParamsRead(qwen235b, 100000)).toBeLessThanOrEqual(qwen235b.paramsTotal);
  });

  it('reads the whole model for a dense model at any batch size', () => {
    expect(decodeParamsRead(llama8b, 1)).toBe(llama8b.paramsTotal);
    expect(decodeParamsRead(llama8b, 64)).toBe(llama8b.paramsTotal);
  });
});

describe('decode speed', () => {
  it('is memory-bound at batch 1', () => {
    const r = runCalculation(makeInput(llama8b, h100));
    expect(r.performance.decodeBound).toBe('memory');
    expect(r.performance.decodeIntensity).toBeLessThan(r.performance.ridgePoint);
  });

  it('puts decode intensity near the batch size', () => {
    // The research design derives I ~ B_batch in the bandwidth-bound region,
    // since the weight transfer term D*F dominates the denominator.
    const r = runCalculation(
      makeInput(llama8b, h100, { workload: { ...workload, batchSize: 8 } }),
    );
    expect(r.performance.decodeIntensity).toBeGreaterThan(4);
    expect(r.performance.decodeIntensity).toBeLessThan(16);
  });

  it('gives Llama-3.1-8B FP16 on an H100 a plausible batch-1 rate', () => {
    const r = runCalculation(makeInput(llama8b, h100));
    // ~16 GB of weights over 3.35 TB/s at 70% MBU -> roughly 145 tok/s.
    expect(r.performance.decodeTokensPerSecPerSequence).toBeGreaterThan(90);
    expect(r.performance.decodeTokensPerSecPerSequence).toBeLessThan(220);
  });

  it('makes a sparse MoE far faster than a dense model of the same size', () => {
    const dense235b: ModelSpec = { ...qwen235b, paramsActive: undefined, moe: undefined };
    const sparse = runCalculation(
      makeInput(qwen235b, h100, {
        hardware: { gpu: h100, numGpus: 8, parallelism: 'tp' },
      }),
    );
    const dense = runCalculation(
      makeInput(dense235b, h100, {
        hardware: { gpu: h100, numGpus: 8, parallelism: 'tp' },
      }),
    );
    expect(sparse.performance.decodeTokensPerSecPerSequence).toBeGreaterThan(
      dense.performance.decodeTokensPerSecPerSequence * 3,
    );
    // ...while needing exactly the same weight memory.
    expect(sparse.memory.weightsBytes).toBe(dense.memory.weightsBytes);
  });

  it('slows down as context grows, because the KV cache is re-read each step', () => {
    const short = runCalculation(
      makeInput(llama8b, h100, { workload: { ...workload, contextLength: 1024 } }),
    );
    const long = runCalculation(
      makeInput(llama8b, h100, { workload: { ...workload, contextLength: 131072 } }),
    );
    expect(long.performance.decodeTokensPerSecPerSequence).toBeLessThan(
      short.performance.decodeTokensPerSecPerSequence,
    );
  });

  it('raises aggregate throughput with batching while per-sequence rate falls', () => {
    const b1 = runCalculation(makeInput(llama8b, h100));
    const b32 = runCalculation(
      makeInput(llama8b, h100, { workload: { ...workload, batchSize: 32 } }),
    );
    expect(b32.performance.decodeTokensPerSecTotal).toBeGreaterThan(
      b1.performance.decodeTokensPerSecTotal,
    );
    expect(b32.performance.decodeTokensPerSecPerSequence).toBeLessThanOrEqual(
      b1.performance.decodeTokensPerSecPerSequence,
    );
  });
});

describe('prefill', () => {
  it('is compute-bound', () => {
    const r = runCalculation(makeInput(llama8b, h100));
    expect(r.performance.prefillBound).toBe('compute');
    expect(r.performance.prefillIntensity).toBeGreaterThan(r.performance.ridgePoint);
  });

  it('grows superlinearly with prompt length via the quadratic attention term', () => {
    const short = runCalculation(
      makeInput(llama8b, h100, { workload: { ...workload, promptTokens: 8192 } }),
    );
    const long = runCalculation(
      makeInput(llama8b, h100, { workload: { ...workload, promptTokens: 65536 } }),
    );
    const ratio = long.performance.ttftMs / short.performance.ttftMs;
    expect(ratio).toBeGreaterThan(8); // more than the 8x linear scaling
  });
});

describe('fit and offload', () => {
  it('reports that a 671B model does not fit on one RTX 3060', () => {
    const r = runCalculation(makeInput(deepseekV3, rtx3060));
    expect(r.fits).toBe(false);
    expect(r.warnings.some((w) => w.key === 'warn.doesNotFit')).toBe(true);
  });

  it('fits Llama-3.1-8B at Q4_K_M on an RTX 3060', () => {
    const r = runCalculation(
      makeInput(llama8b, rtx3060, {
        quant: getQuant('q4_k_m'),
        workload: { ...workload, runtime: 'llamacpp' },
      }),
    );
    expect(r.fits).toBe(true);
    expect(r.utilizationPct).toBeLessThan(100);
  });

  it('computes an offload fraction only when offloading is enabled', () => {
    expect(offloadFraction(20e9, 12e9, 16e9, false)).toBe(0);
    expect(offloadFraction(20e9, 12e9, 16e9, true)).toBeCloseTo(0.5, 2);
    expect(offloadFraction(10e9, 12e9, 8e9, true)).toBe(0);
  });

  it('makes offloading disproportionately slow, since time adds rather than speed', () => {
    const onGpu = runCalculation(
      makeInput(llama8b, rtx3060, {
        quant: getQuant('q4_k_m'),
        workload: { ...workload, runtime: 'llamacpp' },
      }),
    );
    const offloaded = runCalculation(
      makeInput(llama8b, rtx3060, {
        quant: getQuant('fp16'),
        workload: { ...workload, runtime: 'llamacpp', allowOffload: true },
      }),
    );
    expect(offloaded.performance.offloadFraction).toBeGreaterThan(0);
    expect(offloaded.performance.decodeTokensPerSecPerSequence).toBeLessThan(
      onGpu.performance.decodeTokensPerSecPerSequence,
    );
  });
});

describe('multi-GPU', () => {
  it('scales throughput sublinearly', () => {
    const one = runCalculation(makeInput(llama8b, h100));
    const four = runCalculation(
      makeInput(llama8b, h100, {
        hardware: { gpu: h100, numGpus: 4, parallelism: 'tp', nvlink: true },
      }),
    );
    const speedup =
      four.performance.decodeTokensPerSecPerSequence /
      one.performance.decodeTokensPerSecPerSequence;
    expect(speedup).toBeGreaterThan(1);
    expect(speedup).toBeLessThan(4);
  });
});
