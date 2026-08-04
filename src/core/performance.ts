import { kvCacheBytes, weightBytes } from './memory';
import { bytesPerParam } from './quant';
import {
  arithmeticIntensity,
  boundOf,
  effectiveHardware,
  peakFlopsFor,
  ridgePoint,
} from './roofline';
import type {
  HardwareConfig,
  KvQuantSpec,
  ModelSpec,
  PerformanceResult,
  QuantSpec,
  Workload,
} from './types';

/**
 * Parameters actually read from memory per decode step.
 *
 * For a dense model this is simply the whole model. For MoE it is more subtle:
 * a single token routes to `k` of `N` experts, but a *batch* of B tokens
 * activates the union of their choices, which saturates toward all experts as
 * B grows. Assuming uniform routing:
 *
 *   experts_read = N * (1 - (1 - k/N)^B)
 *
 * Real routing is skewed rather than uniform, so this is an upper bound on the
 * union and therefore a conservative (slower) speed estimate.
 *
 * The per-expert and shared-backbone parameter counts are recovered from the
 * two numbers we always have:
 *
 *   params_active = base + k * pe
 *   params_total  = base + N * pe
 *   => pe = (total - active) / (N - k)
 */
export function decodeParamsRead(model: ModelSpec, batchSize: number): number {
  const active = model.paramsActive ?? model.paramsTotal;
  if (!model.moe || active >= model.paramsTotal) return model.paramsTotal;

  const { numExperts: n, expertsPerToken: k } = model.moe;
  if (n <= k) return model.paramsTotal;

  const perExpert = (model.paramsTotal - active) / (n - k);
  const base = active - k * perExpert;

  const expertsRead = n * (1 - Math.pow(1 - k / n, batchSize));
  const paramsRead = base + expertsRead * perExpert;

  return Math.min(model.paramsTotal, Math.max(active, paramsRead));
}

/**
 * Fraction of weights that must live in host RAM because they do not fit in
 * VRAM. Only weights are offloaded — pushing the KV cache across PCIe every
 * step is catastrophic, and runtimes avoid it.
 */
export function offloadFraction(
  totalBytes: number,
  usableBytes: number,
  weightsBytes: number,
  allowOffload: boolean,
): number {
  if (totalBytes <= usableBytes) return 0;
  if (!allowOffload) return 0;
  const overflow = totalBytes - usableBytes;
  return Math.min(1, overflow / weightsBytes);
}

export interface OffloadPlan {
  fraction: number;
  /** Set when the spill is larger than host memory can hold. */
  exceedsHostRam: boolean;
}

/**
 * Offload plan bounded by what host memory can actually hold.
 *
 * Without this bound the model happily reports a throughput figure for
 * spilling 400 GB of weights into a 32 GB machine. That number is not merely
 * optimistic, it is meaningless: the OS would swap to disk and throughput
 * would collapse by orders of magnitude.
 */
export function planOffload(
  totalBytes: number,
  usableVramBytes: number,
  weightsBytes: number,
  allowOffload: boolean,
  hostAvailableBytes: number | undefined,
): OffloadPlan {
  const wanted = offloadFraction(totalBytes, usableVramBytes, weightsBytes, allowOffload);
  if (wanted === 0 || hostAvailableBytes === undefined) {
    return { fraction: wanted, exceedsHostRam: false };
  }

  const wantedBytes = weightsBytes * wanted;
  if (wantedBytes <= hostAvailableBytes) {
    return { fraction: wanted, exceedsHostRam: false };
  }

  return {
    fraction: Math.min(wanted, hostAvailableBytes / weightsBytes),
    exceedsHostRam: true,
  };
}

export function computePerformance(
  model: ModelSpec,
  hardware: HardwareConfig,
  quant: QuantSpec,
  kvQuant: KvQuantSpec,
  workload: Workload,
  memoryTotalBytes: number,
  usableBytes: number,
  hostAvailableBytes?: number,
): PerformanceResult {
  const peakFlops = peakFlopsFor(hardware.gpu, quant.computePrecision);
  const eff = effectiveHardware(
    hardware,
    peakFlops,
    workload.mbu,
    workload.mfu,
    workload.batchSize,
  );

  const weights = weightBytes(model, quant);
  const plan = planOffload(
    memoryTotalBytes,
    usableBytes,
    weights,
    workload.allowOffload,
    hostAvailableBytes,
  );
  const offload = plan.fraction;

  const hostBytesPerSec = workload.hostRamBandwidthGBs * 1e9 * workload.mbu;

  // --- decode -------------------------------------------------------------
  const paramsRead = decodeParamsRead(model, workload.batchSize);
  const weightBytesRead = paramsRead * bytesPerParam(quant);
  const kvBytesRead = kvCacheBytes(
    model,
    kvQuant,
    workload.contextLength,
    workload.batchSize,
  );
  const bytesPerDecodeStep = weightBytesRead + kvBytesRead;

  const activeParams = model.paramsActive ?? model.paramsTotal;
  const flopsPerDecodeStep = 2 * activeParams * workload.batchSize;

  // Offloaded weights stream over the far slower host path. Time adds, speed
  // does not: this is why moving half the layers to CPU costs far more than
  // half the throughput.
  const weightBytesOnHost = weightBytesRead * offload;
  const weightBytesOnGpu = weightBytesRead - weightBytesOnHost;
  const hostSeconds = weightBytesOnHost / Math.max(1, hostBytesPerSec);
  const decodeMemorySeconds =
    (weightBytesOnGpu + kvBytesRead) / eff.bytesPerSec + hostSeconds;

  const decodeComputeSeconds = flopsPerDecodeStep / eff.flopsPerSec;
  const decodeStepSeconds = Math.max(decodeMemorySeconds, decodeComputeSeconds);

  // Share of the step during which the host memory bus is saturated. This is
  // what decides whether the DIMMs sit at idle draw or near their peak.
  const ramDutyCycle = decodeStepSeconds > 0 ? hostSeconds / decodeStepSeconds : 0;

  const decodeTokensPerSecTotal =
    decodeStepSeconds > 0 ? workload.batchSize / decodeStepSeconds : 0;
  const decodeTokensPerSecPerSequence =
    decodeStepSeconds > 0 ? 1 / decodeStepSeconds : 0;

  // --- prefill ------------------------------------------------------------
  // 2*N*T for the linear projections, plus the quadratic attention-score term
  // which starts to matter above roughly 8-16k tokens.
  const t = workload.promptTokens;
  const prefillFlops =
    (2 * activeParams * t + 4 * model.numLayers * t * t * model.hiddenSize) *
    workload.batchSize;

  // Prefill touches every expert, so the full weight set streams in.
  const prefillWeightBytes = weights;
  // Plus the KV cache written for the whole prompt.
  const prefillKvBytes = kvCacheBytes(model, kvQuant, t, workload.batchSize);
  const prefillBytesMoved = prefillWeightBytes + prefillKvBytes;
  const prefillMemorySeconds =
    (prefillWeightBytes * (1 - offload)) / eff.bytesPerSec +
    (prefillWeightBytes * offload) / Math.max(1, hostBytesPerSec);
  const prefillComputeSeconds = prefillFlops / eff.flopsPerSec;
  const ttftSeconds = Math.max(prefillComputeSeconds, prefillMemorySeconds);

  // --- roofline diagnostics ----------------------------------------------
  const decodeIntensity = arithmeticIntensity(flopsPerDecodeStep, bytesPerDecodeStep);
  const prefillIntensity = arithmeticIntensity(prefillFlops, prefillWeightBytes);
  const ridge = ridgePoint(
    peakFlops * hardware.numGpus,
    hardware.gpu.bandwidthGBs * 1e9 * hardware.numGpus,
  );

  return {
    decodeTokensPerSecPerSequence,
    decodeTokensPerSecTotal,
    msPerToken: decodeStepSeconds * 1000,
    ttftMs: ttftSeconds * 1000,
    bytesPerDecodeStep,
    flopsPerDecodeStep,
    prefillFlops,
    prefillBytesMoved,
    decodeIntensity,
    prefillIntensity,
    ridgePoint: ridge,
    decodeBound: boundOf(decodeIntensity, ridge),
    prefillBound: boundOf(prefillIntensity, ridge),
    effectiveBandwidthBytesPerSec: eff.bytesPerSec,
    effectiveFlops: eff.flopsPerSec,
    offloadFraction: offload,
    ramDutyCycle,
    offloadExceedsHostRam: plan.exceedsHostRam,
  };
}
