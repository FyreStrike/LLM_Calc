import { describe, expect, it } from 'vitest';
import { getRamType, HOST_RAM_USABLE_FRACTION } from '../../data/ram';
import {
  activeWattsPerModule,
  hostAvailableBytes,
  ramBandwidthGBs,
  ramPower,
} from '../host';
import { runCalculation } from '../index';
import { planOffload } from '../performance';
import { getKvQuant, getQuant } from '../quant';
import { GB } from '../memory';
import type { CalcInput, GpuSpec, HostSpec, RamSpec, Workload } from '../types';
import { llama8b, qwen235b } from './memory.test';

const ddr4 = getRamType('ddr4');
const ddr5 = getRamType('ddr5');

const ram = (over: Partial<RamSpec> = {}): RamSpec => ({
  typeId: 'ddr5',
  speedMTps: 5600,
  channels: 2,
  totalCapacityGb: 32,
  modules: 2,
  ...over,
});

describe('memory bandwidth', () => {
  // GB/s = MT/s x 8 bytes x channels / 1000. Pure arithmetic, so these are
  // exact rather than approximate.
  it('computes DDR4-3200 dual channel as 51.2 GB/s', () => {
    expect(ramBandwidthGBs(ram({ typeId: 'ddr4', speedMTps: 3200, channels: 2 }))).toBeCloseTo(
      51.2,
      6,
    );
  });

  it('computes DDR5-5600 dual channel as 89.6 GB/s', () => {
    expect(ramBandwidthGBs(ram({ speedMTps: 5600, channels: 2 }))).toBeCloseTo(89.6, 6);
  });

  it('computes 8-channel DDR5-4800 as 307.2 GB/s', () => {
    expect(ramBandwidthGBs(ram({ speedMTps: 4800, channels: 8 }))).toBeCloseTo(307.2, 6);
  });

  it('scales linearly with channels', () => {
    const dual = ramBandwidthGBs(ram({ channels: 2 }));
    const octa = ramBandwidthGBs(ram({ channels: 8 }));
    expect(octa / dual).toBeCloseTo(4, 6);
  });
});

describe('memory power', () => {
  it('scales idle draw with installed modules, not capacity', () => {
    // The point of the whole feature: a server pays for the DIMMs it has
    // populated, not for the memory the model happens to need.
    const few = ramPower(ram({ modules: 2, totalCapacityGb: 512 }), ddr5, 0);
    const many = ramPower(ram({ modules: 16, totalCapacityGb: 512 }), ddr5, 0);

    expect(few.idleW).toBeCloseTo(2 * ddr5.idleWattsPerModule, 6);
    expect(many.idleW).toBeCloseTo(16 * ddr5.idleWattsPerModule, 6);
    expect(many.idleW).toBeGreaterThan(few.idleW);
  });

  it('charges a 512 GB server about 30 W of standing memory draw', () => {
    const server = ramPower(ram({ modules: 16, totalCapacityGb: 512 }), ddr5, 0);
    expect(server.idleW).toBeGreaterThan(25);
    expect(server.idleW).toBeLessThan(40);
    // Paid even with the bus completely idle.
    expect(server.activeW).toBe(0);
  });

  it('adds no active draw when nothing is streaming from host memory', () => {
    const p = ramPower(ram(), ddr5, 0);
    expect(p.activeW).toBe(0);
    expect(p.totalW).toBe(p.idleW);
  });

  it('adds the full active delta at a saturated bus', () => {
    const spec = ram({ modules: 4 });
    const idle = ramPower(spec, ddr5, 0);
    const full = ramPower(spec, ddr5, 1);
    const perModule = activeWattsPerModule(spec, ddr5);

    expect(full.totalW).toBeCloseTo(4 * perModule, 5);
    expect(full.totalW).toBeGreaterThan(idle.totalW);
  });

  it('interpolates the active delta with the duty cycle', () => {
    const spec = ram({ modules: 4 });
    const half = ramPower(spec, ddr5, 0.5);
    const full = ramPower(spec, ddr5, 1);
    expect(half.activeW).toBeCloseTo(full.activeW / 2, 6);
  });

  it('draws more per module for DDR5 than DDR4 at their reference clocks', () => {
    const d4 = activeWattsPerModule(ram({ typeId: 'ddr4', speedMTps: 2666 }), ddr4);
    const d5 = activeWattsPerModule(ram({ speedMTps: 4800 }), ddr5);
    expect(d4).toBeCloseTo(3.3, 3);
    expect(d5).toBeCloseTo(5.8, 3);
  });

  it('never lets active draw fall below idle', () => {
    // A very low clock would otherwise scale the reference figure under idle.
    const slow = activeWattsPerModule(ram({ speedMTps: 100 }), ddr5);
    expect(slow).toBeGreaterThanOrEqual(ddr5.idleWattsPerModule);
  });
});

describe('offload bounded by host memory', () => {
  const weights = 400 * GB;

  it('holds back a share of RAM for the operating system', () => {
    const avail = hostAvailableBytes(ram({ totalCapacityGb: 64 }), HOST_RAM_USABLE_FRACTION);
    expect(avail).toBeCloseTo(64 * GB * 0.85, -6);
    expect(avail).toBeLessThan(64 * GB);
  });

  it('caps the offload at what actually fits and flags it', () => {
    // 400 GB of weights spilling into a 32 GB machine: previously this
    // produced a throughput figure for an impossible configuration.
    const hostBytes = hostAvailableBytes(ram({ totalCapacityGb: 32 }), HOST_RAM_USABLE_FRACTION);
    const plan = planOffload(440 * GB, 24 * GB, weights, true, hostBytes);

    expect(plan.exceedsHostRam).toBe(true);
    expect(plan.fraction).toBeLessThan(0.1);
    expect(plan.fraction * weights).toBeLessThanOrEqual(hostBytes + 1);
  });

  it('leaves the plan untouched when host memory is ample', () => {
    const hostBytes = hostAvailableBytes(ram({ totalCapacityGb: 768 }), HOST_RAM_USABLE_FRACTION);
    const plan = planOffload(440 * GB, 24 * GB, weights, true, hostBytes);
    expect(plan.exceedsHostRam).toBe(false);
    expect(plan.fraction).toBeGreaterThan(0);
  });

  it('does nothing when offloading is switched off', () => {
    const plan = planOffload(440 * GB, 24 * GB, weights, false, 1e15);
    expect(plan.fraction).toBe(0);
    expect(plan.exceedsHostRam).toBe(false);
  });
});

describe('end to end with a host specification', () => {
  const rtx4090: GpuSpec = {
    id: 'rtx-4090', name: 'RTX 4090', vendor: 'nvidia', vramGb: 24,
    bandwidthGBs: 1008, fp16TFlops: 165, tdpW: 450, idleW: 20,
  };

  const workload: Workload = {
    contextLength: 8192, promptTokens: 2048, outputTokens: 512, batchSize: 1,
    runtime: 'llamacpp', prefillChunkTokens: 2048, allowOffload: false,
    hostRamBandwidthGBs: 89.6, mbu: 0.7, mfu: 0.4,
  };

  const host = (over: Partial<RamSpec> = {}, baseW = 55): HostSpec => ({
    ram: ram(over),
    baseOverheadW: baseW,
  });

  function input(over: Partial<CalcInput> = {}): CalcInput {
    return {
      model: llama8b,
      hardware: { gpu: rtx4090, numGpus: 1, parallelism: 'tp' },
      quant: getQuant('q4_k_m'),
      kvQuant: getKvQuant('fp16'),
      workload,
      energy: { mode: 'simple', psuEfficiency: 0.9, pue: 1, hostOverheadW: 80 },
      cost: {
        region: { id: 'de', label: 'DE', pricePerKWh: 0.3869, currency: 'EUR', gridIntensityGCO2PerKWh: 380 },
        inputRatio: 0.7, usdToCurrency: 0.92,
      },
      host: host(),
      ...over,
    };
  }

  it('reports the host draw split into base, idle RAM and active RAM', () => {
    const r = runCalculation(input());
    expect(r.energy.host.baseW).toBe(55);
    expect(r.energy.host.ramIdleW).toBeCloseTo(2 * ddr5.idleWattsPerModule, 5);
    expect(r.energy.host.ramActiveW).toBe(0); // nothing offloaded
    expect(r.energy.host.totalW).toBeCloseTo(55 + r.energy.host.ramIdleW, 5);
  });

  it('makes an over-populated server cost more per token on a small model', () => {
    // Same model, same GPU, same work — only the installed DIMMs differ.
    const desktop = runCalculation(input({ host: host({ modules: 2, totalCapacityGb: 32 }) }));
    const server = runCalculation(input({ host: host({ modules: 16, totalCapacityGb: 512 }) }));

    expect(server.energy.host.ramIdleW).toBeGreaterThan(
      desktop.energy.host.ramIdleW + 20,
    );
    expect(server.cost.localElectricityPerMTokens).toBeGreaterThan(
      desktop.cost.localElectricityPerMTokens,
    );
    expect(server.warnings.some((w) => w.key === 'warn.ramIdleDominates')).toBe(true);
  });

  it('warns and caps when the spill exceeds installed memory', () => {
    const r = runCalculation(
      input({
        model: qwen235b,
        quant: getQuant('fp16'),
        workload: { ...workload, allowOffload: true },
        host: host({ totalCapacityGb: 32, modules: 2 }),
      }),
    );

    expect(r.performance.offloadExceedsHostRam).toBe(true);
    expect(r.warnings.some((w) => w.key === 'warn.offloadExceedsHostRam')).toBe(true);
  });

  it('raises the RAM duty cycle once weights stream from host memory', () => {
    // Llama-8B at FP16 is ~15 GB of weights, so it sits entirely in a 24 GB
    // card and offloads nothing. A 12 GB card forces the spill.
    const small: GpuSpec = { ...rtx4090, id: 'rtx-3060', vramGb: 12, bandwidthGBs: 360 };

    const resident = runCalculation(input());
    const offloaded = runCalculation(
      input({
        hardware: { gpu: small, numGpus: 1, parallelism: 'tp' },
        quant: getQuant('fp16'),
        workload: { ...workload, allowOffload: true },
        host: host({ totalCapacityGb: 128, modules: 4 }),
      }),
    );

    expect(resident.performance.ramDutyCycle).toBe(0);
    expect(offloaded.performance.offloadFraction).toBeGreaterThan(0);
    expect(offloaded.performance.offloadExceedsHostRam).toBe(false);
    expect(offloaded.performance.ramDutyCycle).toBeGreaterThan(0);
    expect(offloaded.energy.host.ramActiveW).toBeGreaterThan(0);

    // Streaming over a 90 GB/s bus instead of 360 GB/s of VRAM dominates the
    // step, so the memory bus is busy for most of it.
    expect(offloaded.performance.ramDutyCycle).toBeGreaterThan(0.3);
  });

  it('still works without a host specification', () => {
    const r = runCalculation(input({ host: undefined }));
    expect(r.energy.host.baseW).toBe(80); // falls back to hostOverheadW
    expect(r.energy.host.ramIdleW).toBe(0);
    expect(r.cost.localElectricityPerMTokens).toBeGreaterThan(0);
  });
});
