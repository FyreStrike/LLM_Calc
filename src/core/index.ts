import { HOST_RAM_USABLE_FRACTION, getRamType } from '../data/ram';
import { computeCost } from './cost';
import { computeEnergy } from './energy';
import { hostAvailableBytes, ramPower } from './host';
import { computeMemory, GB, usableVramBytes } from './memory';
import { computePerformance } from './performance';
import { peakFlopsFor } from './roofline';
import type { CalcInput, CalcResult, Warning } from './types';

export * from './types';
export * from './quant';
export * from './memory';
export * from './host';
export * from './roofline';
export * from './performance';
export * from './energy';
export * from './cost';

/**
 * Run the full pipeline: memory -> fit -> performance -> energy -> cost.
 */
export function runCalculation(input: CalcInput): CalcResult {
  const { model, hardware, quant, kvQuant, workload, energy, cost, apiPrice, host } =
    input;

  const memory = computeMemory(
    model,
    quant,
    kvQuant,
    workload,
    hardware.numGpus,
  );

  const totalVramBytes = hardware.gpu.vramGb * GB * hardware.numGpus;
  const usable = usableVramBytes(
    hardware.gpu.vramGb,
    hardware.numGpus,
    hardware.gpu.unified,
  );

  const hostRamType = host ? getRamType(host.ram.typeId) : undefined;
  const hostAvailable = host
    ? hostAvailableBytes(host.ram, HOST_RAM_USABLE_FRACTION)
    : undefined;

  const performance = computePerformance(
    model,
    hardware,
    quant,
    kvQuant,
    workload,
    memory.totalBytes,
    usable,
    hostAvailable,
  );

  const peakFlops = peakFlopsFor(hardware.gpu, quant.computePrecision);

  // RAM power follows what the memory bus is actually doing: idle draw is paid
  // for every installed module regardless, the active delta only while weights
  // stream from host memory.
  const hostPower =
    host && hostRamType
      ? (() => {
          const p = ramPower(host.ram, hostRamType, performance.ramDutyCycle);
          return { baseW: host.baseOverheadW, ramIdleW: p.idleW, ramActiveW: p.activeW };
        })()
      : undefined;

  const energyResult = computeEnergy(
    hardware.gpu,
    hardware.numGpus,
    performance,
    workload,
    energy,
    peakFlops,
    hostPower,
  );

  const costResult = computeCost(energyResult, cost, apiPrice);

  const fits = memory.totalBytes <= usable;

  return {
    memory,
    totalVramBytes,
    usableVramBytes: usable,
    fits,
    utilizationPct: (memory.totalBytes / usable) * 100,
    performance,
    energy: energyResult,
    cost: costResult,
    warnings: collectWarnings(input, memory.totalBytes, usable, fits, performance, hostAvailable),
  };
}

function collectWarnings(
  input: CalcInput,
  totalBytes: number,
  usableBytes: number,
  fits: boolean,
  performance: { offloadFraction: number; offloadExceedsHostRam: boolean },
  hostAvailable: number | undefined,
): Warning[] {
  const { model, hardware, quant, workload, host } = input;
  const offload = performance.offloadFraction;
  const warnings: Warning[] = [];

  // The spill is larger than host memory can hold. Reporting a throughput
  // figure here would be meaningless — the OS would swap to disk.
  if (performance.offloadExceedsHostRam && host && hostAvailable !== undefined) {
    warnings.push({
      level: 'error',
      key: 'warn.offloadExceedsHostRam',
      values: {
        installed: host.ram.totalCapacityGb,
        usable: (hostAvailable / GB).toFixed(0),
      },
    });
  }

  if (!fits && offload === 0) {
    warnings.push({
      level: 'error',
      key: 'warn.doesNotFit',
      values: {
        needed: (totalBytes / GB).toFixed(1),
        available: (usableBytes / GB).toFixed(1),
      },
    });
  }

  if (offload > 0) {
    warnings.push({
      level: 'warn',
      key: 'warn.offloading',
      values: { percent: (offload * 100).toFixed(0) },
    });
  }

  if (hardware.gpu.unified) {
    // Metal caps GPU allocation near 75% of unified memory by default.
    warnings.push({ level: 'info', key: 'warn.unifiedMemory' });
  }

  if (workload.contextLength > model.maxContext) {
    warnings.push({
      level: 'warn',
      key: 'warn.contextExceedsMax',
      values: { max: model.maxContext },
    });
  }

  if (model.attention === 'mla') {
    warnings.push({ level: 'info', key: 'warn.mlaCache' });
  }

  if (model.moe && workload.batchSize > 1) {
    // The union of routed experts grows with batch, so bandwidth per token
    // rises above the headline "active params" figure.
    warnings.push({ level: 'info', key: 'warn.moeBatchUnion' });
  }

  // Quantizing a small model to 4 bit can *raise* energy use: the dequant
  // overhead in the SMs outweighs the HBM savings below roughly 3B params.
  if (quant.bpw < 5 && model.paramsTotal < 3e9) {
    warnings.push({ level: 'warn', key: 'warn.smallModelQuantEnergy' });
  }

  if (hardware.numGpus > 1 && !(hardware.nvlink ?? hardware.gpu.nvlink)) {
    warnings.push({ level: 'warn', key: 'warn.noNvlink' });
  }

  if (workload.runtime === 'vllm') {
    warnings.push({ level: 'info', key: 'warn.vllmPreallocation' });
  }

  // Idle RAM draw scales with installed modules, not with what the model
  // needs. A heavily populated server pays this around the clock, and on a
  // small model it can rival the GPU's own decode draw.
  if (host) {
    const ramType = getRamType(host.ram.typeId);
    const idleW = host.ram.modules * ramType.idleWattsPerModule;
    if (idleW > 20) {
      warnings.push({
        level: 'info',
        key: 'warn.ramIdleDominates',
        values: {
          watts: idleW.toFixed(0),
          modules: host.ram.modules,
          capacity: host.ram.totalCapacityGb,
        },
      });
    }
  }

  return warnings;
}
