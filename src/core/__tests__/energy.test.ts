import { describe, expect, it } from 'vitest';
import { blendedApiPricePerMTokens } from '../cost';
import { defaultEpsFlopPJ, defaultEpsMopPJ } from '../energy';
import { runCalculation } from '../index';
import { getKvQuant, getQuant } from '../quant';
import type { ApiPrice, CalcInput, GpuSpec, ModelSpec, Region, Workload } from '../types';
import { llama8b, qwen235b } from './memory.test';

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
  // Decode power bands differ by segment; without this the fixture would be
  // costed as a consumer card.
  segment: 'datacenter',
};

/** Eurostat H2 2025, household price including all taxes and levies. */
const germany: Region = {
  id: 'de',
  label: 'Germany',
  pricePerKWh: 0.3869,
  currency: 'EUR',
  gridIntensityGCO2PerKWh: 380,
};

const workload: Workload = {
  contextLength: 4096,
  promptTokens: 1024,
  outputTokens: 512,
  batchSize: 32,
  runtime: 'vllm',
  prefillChunkTokens: 2048,
  allowOffload: false,
  hostRamBandwidthGBs: 90,
  mbu: 0.7,
  mfu: 0.4,
};

function makeInput(model: ModelSpec, over: Partial<CalcInput> = {}): CalcInput {
  return {
    model,
    hardware: { gpu: h100, numGpus: 1, parallelism: 'tp' },
    quant: getQuant('fp16'),
    kvQuant: getKvQuant('fp16'),
    workload,
    energy: { mode: 'simple', psuEfficiency: 0.9, pue: 1.0, hostOverheadW: 80 },
    cost: { region: germany, inputRatio: 0.7, usdToCurrency: 0.92 },
    ...over,
  };
}

describe('power draw', () => {
  it('keeps decode well below TDP, as the ML.ENERGY benchmark reports', () => {
    const r = runCalculation(makeInput(llama8b, { workload: { ...workload, batchSize: 1 } }));
    // 20-40% of TDP at low batch.
    expect(r.energy.decodePowerW).toBeLessThan(h100.tdpW * 0.45);
    expect(r.energy.decodePowerW).toBeGreaterThan(h100.tdpW * 0.15);
  });

  it('draws more at higher batch, and prefill more than decode', () => {
    const b1 = runCalculation(makeInput(llama8b, { workload: { ...workload, batchSize: 1 } }));
    const b64 = runCalculation(makeInput(llama8b, { workload: { ...workload, batchSize: 64 } }));
    expect(b64.energy.decodePowerW).toBeGreaterThan(b1.energy.decodePowerW);
    expect(b1.energy.prefillPowerW).toBeGreaterThan(b1.energy.decodePowerW);
  });

  it('never exceeds the physical TDP ceiling', () => {
    const r = runCalculation(
      makeInput(llama8b, {
        energy: {
          mode: 'roofline',
          epsFlopPJ: 1e6, // absurd calibration
          epsMopPJ: 1e6,
          psuEfficiency: 0.9,
          pue: 1,
          hostOverheadW: 80,
        },
      }),
    );
    expect(r.energy.decodePowerW).toBeLessThanOrEqual(h100.tdpW);
  });
});

describe('joules per token', () => {
  it('lands in the range the thesis measurements anticipate', () => {
    // statistical_analysis.py anchors H100 FP16 at batch 32 near 0.11 J/token.
    const r = runCalculation(makeInput(llama8b));
    expect(r.energy.joulesPerToken).toBeGreaterThan(0.02);
    expect(r.energy.joulesPerToken).toBeLessThan(1.0);
  });

  it('improves with batching — the dominant efficiency lever', () => {
    const b1 = runCalculation(makeInput(llama8b, { workload: { ...workload, batchSize: 1 } }));
    const b64 = runCalculation(makeInput(llama8b, { workload: { ...workload, batchSize: 64 } }));
    expect(b64.energy.joulesPerToken).toBeLessThan(b1.energy.joulesPerToken / 5);
  });

  it('is lower for a sparse MoE than a dense model of equal total size', () => {
    const dense: ModelSpec = { ...qwen235b, paramsActive: undefined, moe: undefined };
    const hw = { gpu: h100, numGpus: 8, parallelism: 'tp' as const, nvlink: true };
    const sparse = runCalculation(makeInput(qwen235b, { hardware: hw }));
    const denseR = runCalculation(makeInput(dense, { hardware: hw }));
    expect(sparse.energy.joulesPerToken).toBeLessThan(denseR.energy.joulesPerToken);
  });
});

describe('roofline energy decomposition', () => {
  it('splits energy into compute, memory and static terms', () => {
    const r = runCalculation(
      makeInput(llama8b, {
        energy: { mode: 'roofline', psuEfficiency: 0.9, pue: 1, hostOverheadW: 80 },
      }),
    );
    const d = r.energy.decomposition;
    expect(d).toBeDefined();
    expect(d!.computeJoules).toBeGreaterThan(0);
    expect(d!.memoryJoules).toBeGreaterThan(0);
    expect(d!.staticJoules).toBeGreaterThan(0);
  });

  it('shows data movement dominating compute during memory-bound decode', () => {
    // E = eps_flop*W + eps_mop*Q + pi_0*T. At batch 1 the GEMV reads every
    // weight for two FLOPs each, so the eps_mop*Q term should dominate.
    const r = runCalculation(
      makeInput(llama8b, {
        workload: { ...workload, batchSize: 1 },
        energy: { mode: 'roofline', psuEfficiency: 0.9, pue: 1, hostOverheadW: 80 },
      }),
    );
    const d = r.energy.decomposition!;
    expect(d.memoryJoules).toBeGreaterThan(d.computeJoules);
  });

  it('derives sane default eps coefficients from the GPU spec', () => {
    const epsFlop = defaultEpsFlopPJ(h100, 1979e12);
    const epsMop = defaultEpsMopPJ(h100);
    expect(epsFlop).toBeGreaterThan(0);
    expect(epsMop).toBeGreaterThan(0);
    // Moving a byte costs far more energy than a single FLOP — the physical
    // fact the whole energy roofline rests on.
    expect(epsMop).toBeGreaterThan(epsFlop * 100);
  });
});

describe('wall-plug scaling', () => {
  it('adds host overhead, PSU losses and PUE on top of GPU energy', () => {
    const r = runCalculation(makeInput(llama8b));
    expect(r.energy.wallJoulesPerToken).toBeGreaterThan(r.energy.joulesPerToken);
  });

  it('scales linearly with PUE', () => {
    const home = runCalculation(makeInput(llama8b));
    const datacenter = runCalculation(
      makeInput(llama8b, {
        energy: { mode: 'simple', psuEfficiency: 0.9, pue: 1.54, hostOverheadW: 80 },
      }),
    );
    expect(datacenter.energy.wallJoulesPerToken / home.energy.wallJoulesPerToken).toBeCloseTo(
      1.54,
      2,
    );
  });
});

describe('cost comparison', () => {
  const glm52: ApiPrice = {
    id: 'z-ai/glm-5.2',
    label: 'GLM 5.2',
    inputPerMTokUsd: 0.76,
    outputPerMTokUsd: 2.42,
  };

  it('blends input and output prices at the stated mix', () => {
    // 70% input, 30% output: 0.7*0.76 + 0.3*2.42 = 1.258 USD -> EUR at 0.92
    expect(blendedApiPricePerMTokens(glm52, 0.7, 0.92)).toBeCloseTo(1.157, 3);
    // An all-output workload is materially more expensive.
    expect(blendedApiPricePerMTokens(glm52, 0, 1)).toBeCloseTo(2.42, 3);
    expect(blendedApiPricePerMTokens(glm52, 1, 1)).toBeCloseTo(0.76, 3);
  });

  it('reports local electricity cost per million tokens', () => {
    const r = runCalculation(makeInput(llama8b));
    expect(r.cost.localElectricityPerMTokens).toBeGreaterThan(0);
    expect(r.cost.currency).toBe('EUR');
  });

  it('computes savings and a break-even volume against an API price', () => {
    const r = runCalculation(
      makeInput(llama8b, {
        apiPrice: glm52,
        cost: {
          region: germany,
          inputRatio: 0.7,
          usdToCurrency: 0.92,
          hardwareCapex: 30000,
          dailyTokens: 5e6,
        },
      }),
    );
    expect(r.cost.apiPerMTokens).toBeCloseTo(1.157, 2);
    expect(r.cost.savingsPerMTokens).toBeDefined();
    expect(r.cost.breakEvenTokens).toBeGreaterThan(0);
    expect(r.cost.breakEvenDays).toBeGreaterThan(0);
  });

  it('reports no break-even when local energy costs more than the API', () => {
    const cheapApi: ApiPrice = {
      id: 'cheap',
      label: 'Very cheap API',
      inputPerMTokUsd: 0.0001,
      outputPerMTokUsd: 0.0001,
    };
    const r = runCalculation(
      makeInput(llama8b, {
        apiPrice: cheapApi,
        workload: { ...workload, batchSize: 1 },
        cost: {
          region: germany,
          inputRatio: 0.7,
          usdToCurrency: 0.92,
          hardwareCapex: 30000,
          dailyTokens: 1e6,
        },
      }),
    );
    expect(r.cost.savingsPerMTokens!).toBeLessThan(0);
    expect(r.cost.breakEvenTokens).toBeUndefined();
  });

  it('tracks CO2 alongside cost', () => {
    const r = runCalculation(makeInput(llama8b));
    expect(r.cost.co2GramsPerMTokens).toBeCloseTo(
      r.energy.kWhPerMTokens * germany.gridIntensityGCO2PerKWh,
      6,
    );
  });

  it('makes cheap electricity change the verdict', () => {
    const norway: Region = { ...germany, id: 'no', pricePerKWh: 0.1, gridIntensityGCO2PerKWh: 30 };
    const de = runCalculation(makeInput(llama8b));
    const no = runCalculation(
      makeInput(llama8b, { cost: { region: norway, inputRatio: 0.7, usdToCurrency: 0.92 } }),
    );
    expect(no.cost.localElectricityPerMTokens).toBeLessThan(
      de.cost.localElectricityPerMTokens,
    );
    expect(no.cost.co2GramsPerMTokens).toBeLessThan(de.cost.co2GramsPerMTokens);
  });
});
