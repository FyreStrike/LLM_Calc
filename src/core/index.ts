import { computeCost } from './cost';
import { computeEnergy } from './energy';
import { computeMemory, GB, usableVramBytes } from './memory';
import { computePerformance } from './performance';
import { peakFlopsFor } from './roofline';
import type { CalcInput, CalcResult, Warning } from './types';

export * from './types';
export * from './quant';
export * from './memory';
export * from './roofline';
export * from './performance';
export * from './energy';
export * from './cost';

/**
 * Run the full pipeline: memory -> fit -> performance -> energy -> cost.
 */
export function runCalculation(input: CalcInput): CalcResult {
  const { model, hardware, quant, kvQuant, workload, energy, cost, apiPrice } = input;

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

  const performance = computePerformance(
    model,
    hardware,
    quant,
    kvQuant,
    workload,
    memory.totalBytes,
    usable,
  );

  const peakFlops = peakFlopsFor(hardware.gpu, quant.computePrecision);
  const energyResult = computeEnergy(
    hardware.gpu,
    hardware.numGpus,
    performance,
    workload,
    energy,
    peakFlops,
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
    warnings: collectWarnings(input, memory.totalBytes, usable, fits, performance.offloadFraction),
  };
}

function collectWarnings(
  input: CalcInput,
  totalBytes: number,
  usableBytes: number,
  fits: boolean,
  offload: number,
): Warning[] {
  const { model, hardware, quant, workload } = input;
  const warnings: Warning[] = [];

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

  return warnings;
}
