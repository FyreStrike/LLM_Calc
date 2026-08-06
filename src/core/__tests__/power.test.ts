import { describe, expect, it } from 'vitest';
import { getGpu } from '../../data/gpus';
import { DECODE_TDP_FRACTION, decodeTdpFraction } from '../energy';
import { runCalculation } from '../index';
import { getKvQuant, getQuant } from '../quant';
import type { CalcInput, GpuSpec, Workload } from '../types';
import { llama8b } from './memory.test';

/**
 * Power draw is the number users sanity-check against their own wall meter,
 * so the segment bands are pinned to published measurements rather than left
 * to drift.
 */

const workload: Workload = {
  contextLength: 4096, promptTokens: 1024, outputTokens: 512, batchSize: 1,
  runtime: 'vllm', prefillChunkTokens: 2048, allowOffload: false,
  hostRamBandwidthGBs: 90, mbu: 0.7, mfu: 0.4,
};

function gpuPowerW(gpu: GpuSpec, batchSize: number): number {
  const input: CalcInput = {
    model: llama8b,
    hardware: { gpu, numGpus: 1, parallelism: 'tp' },
    quant: getQuant('fp16'),
    kvQuant: getKvQuant('fp16'),
    workload: { ...workload, batchSize },
    energy: { mode: 'simple', psuEfficiency: 0.9, pue: 1, hostOverheadW: 60 },
    cost: {
      region: { id: 'de', label: 'DE', pricePerKWh: 0.3869, currency: 'EUR', gridIntensityGCO2PerKWh: 380 },
      inputRatio: 0.7, usdToCurrency: 0.92,
    },
  };
  return runCalculation(input).energy.decodePowerW;
}

describe('measured anchor points', () => {
  const rtx4090 = getGpu('rtx-4090')!;
  const h100 = getGpu('h100-sxm')!;

  it('puts an RTX 4090 at batch 1 in the measured 250-320 W band', () => {
    // Reported sustained draw during LLM generation. The old flat 20% band,
    // taken from datacenter measurements, predicted ~110 W here.
    const w = gpuPowerW(rtx4090, 1);
    expect(w).toBeGreaterThan(240);
    expect(w).toBeLessThan(330);
  });

  it('puts an RTX 4090 at large batch near the 360-410 W observation', () => {
    const w = gpuPowerW(rtx4090, 256);
    expect(w).toBeGreaterThan(350);
    expect(w).toBeLessThan(440);
  });

  it('keeps an H100 at batch 1 inside the ML.ENERGY 20-40% band', () => {
    const fraction = gpuPowerW(h100, 1) / h100.tdpW;
    expect(fraction).toBeGreaterThan(0.2);
    expect(fraction).toBeLessThan(0.45);
  });

  it('drives an H100 near its rating under production batching', () => {
    // A loaded node running continuous batching really does sit at 600-700 W;
    // capping datacenter decode at 55% of TDP understated that badly.
    const w = gpuPowerW(h100, 256);
    expect(w).toBeGreaterThan(600);
    expect(w).toBeLessThanOrEqual(h100.tdpW);
  });
});

describe('segment bands', () => {
  it('separates the segments at low batch, where headroom differs', () => {
    // A 700 W board has far more idle headroom to fall into than a 450 W one.
    expect(decodeTdpFraction(1, 'datacenter')).toBeLessThan(
      decodeTdpFraction(1, 'consumer'),
    );
  });

  it('converges every segment toward TDP at large batch', () => {
    for (const segment of ['consumer', 'workstation', 'datacenter', 'soc'] as const) {
      expect(decodeTdpFraction(256, segment)).toBeGreaterThan(0.8);
    }
  });

  it('rises monotonically with batch size', () => {
    const f = (b: number) => decodeTdpFraction(b, 'datacenter');
    expect(f(1)).toBeLessThan(f(8));
    expect(f(8)).toBeLessThan(f(64));
  });

  it('never exceeds the rating', () => {
    for (const [, [min, max]] of Object.entries(DECODE_TDP_FRACTION)) {
      expect(min).toBeGreaterThan(0);
      expect(max).toBeLessThanOrEqual(1);
      expect(max).toBeGreaterThan(min);
    }
  });

  it('distinguishes the H200 PCIe part from the SXM one', () => {
    const sxm = getGpu('h200-sxm')!;
    const nvl = getGpu('h200-nvl')!;

    // Same memory subsystem...
    expect(nvl.vramGb).toBe(sxm.vramGb);
    expect(nvl.bandwidthGBs).toBe(sxm.bandwidthGBs);
    // ...but a tighter power envelope and lower clocks.
    expect(nvl.tdpW).toBeLessThan(sxm.tdpW);
    expect(nvl.fp16TFlops).toBeLessThan(sxm.fp16TFlops);
    expect(nvl.segment).toBe('datacenter');
  });

  it('classifies the catalog into segments', () => {
    expect(getGpu('rtx-4090')!.segment).toBe('consumer');
    expect(getGpu('h100-sxm')!.segment).toBe('datacenter');
    expect(getGpu('rtx-6000-ada')!.segment).toBe('workstation');
    expect(getGpu('m3-ultra')!.segment).toBe('soc');
    expect(getGpu('dgx-spark-gb10')!.segment).toBe('soc');
    expect(getGpu('mi300x')!.segment).toBe('datacenter');
  });
});

describe('energy units', () => {
  it('reports watt-hours consistently with joules', () => {
    const input: CalcInput = {
      model: llama8b,
      hardware: { gpu: getGpu('rtx-4090')!, numGpus: 1, parallelism: 'tp' },
      quant: getQuant('fp16'),
      kvQuant: getKvQuant('fp16'),
      workload,
      energy: { mode: 'simple', psuEfficiency: 0.9, pue: 1, hostOverheadW: 60 },
      cost: {
        region: { id: 'de', label: 'DE', pricePerKWh: 0.3869, currency: 'EUR', gridIntensityGCO2PerKWh: 380 },
        inputRatio: 0.7, usdToCurrency: 0.92,
      },
    };
    const e = runCalculation(input).energy;

    // 1 Wh = 3600 J, so the two views must agree exactly.
    expect(e.wattHoursPerKTokens).toBeCloseTo((e.wallJoulesPerToken * 1000) / 3600, 9);
    expect(e.kWhPerMTokens).toBeCloseTo(e.wattHoursPerKTokens, 6);
  });

  it('lands watt-hours per 1000 tokens on a readable scale', () => {
    // Per single token would be ~0.002 Wh, which reads as noise.
    const input: CalcInput = {
      model: llama8b,
      hardware: { gpu: getGpu('rtx-4090')!, numGpus: 1, parallelism: 'tp' },
      quant: getQuant('q4_k_m'),
      kvQuant: getKvQuant('fp16'),
      workload,
      energy: { mode: 'simple', psuEfficiency: 0.9, pue: 1, hostOverheadW: 60 },
      cost: {
        region: { id: 'de', label: 'DE', pricePerKWh: 0.3869, currency: 'EUR', gridIntensityGCO2PerKWh: 380 },
        inputRatio: 0.7, usdToCurrency: 0.92,
      },
    };
    const e = runCalculation(input).energy;
    expect(e.wattHoursPerKTokens).toBeGreaterThan(0.05);
    expect(e.wattHoursPerKTokens).toBeLessThan(500);
  });
});
