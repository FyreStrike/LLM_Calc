import { coolingPowerW } from './host';
import type {
  CoolingSpec,
  EnergyOptions,
  EnergyResult,
  GpuSpec,
  PerformanceResult,
  Workload,
} from './types';

/**
 * Energy modelling.
 *
 * Two interchangeable models sit behind one interface:
 *
 * 1. `simple` — power as a fraction of TDP, differentiated by phase. This is
 *    what most calculators do, but done honestly: decode is memory-bound and
 *    draws far less than TDP.
 *
 * 2. `roofline` — the extended energy roofline from the thesis research design:
 *
 *      E = eps_flop * W + eps_mop * Q + pi_0 * T
 *
 *    where W is the FLOP count, Q the bytes moved, pi_0 the static idle draw
 *    and T the wall time. This decomposes energy into compute, data movement
 *    and leakage, which is what makes DVFS reasoning possible: in the
 *    bandwidth-bound decode region, lowering the clock cuts dynamic power
 *    without extending T much, so "race to halt" is energetically suboptimal.
 */

/**
 * The ML.ENERGY benchmark measures LLM decoding at roughly 20-40% of GPU TDP,
 * because low arithmetic intensity leaves most of the compute fabric idle.
 * Prefill saturates the tensor cores and approaches TDP.
 *
 * Source: The ML.ENERGY Benchmark, https://arxiv.org/html/2505.06371v1
 */
export const DECODE_TDP_FRACTION_MIN = 0.2;
export const DECODE_TDP_FRACTION_MAX = 0.55;
export const PREFILL_TDP_FRACTION = 0.85;

/**
 * Share of dynamic power attributable to compute vs. data movement when the
 * respective unit is saturated. Used only to derive default eps coefficients;
 * both are editable in the UI so the model can be recalibrated against real
 * NVML measurements.
 */
const COMPUTE_POWER_SHARE = 0.7;
const MEMORY_POWER_SHARE = 0.35;

/**
 * Default eps_flop in picojoules per FLOP: the dynamic energy the GPU would
 * spend per operation if the tensor cores ran at peak.
 */
export function defaultEpsFlopPJ(gpu: GpuSpec, peakFlops: number): number {
  if (peakFlops <= 0) return 0;
  const dynamicW = Math.max(0, gpu.tdpW - gpu.idleW);
  return ((dynamicW * COMPUTE_POWER_SHARE) / peakFlops) * 1e12;
}

/** Default eps_mop in picojoules per byte moved from device memory. */
export function defaultEpsMopPJ(gpu: GpuSpec): number {
  const bytesPerSec = gpu.bandwidthGBs * 1e9;
  if (bytesPerSec <= 0) return 0;
  const dynamicW = Math.max(0, gpu.tdpW - gpu.idleW);
  return ((dynamicW * MEMORY_POWER_SHARE) / bytesPerSec) * 1e12;
}

/**
 * Decode power rises with batch size: more of the compute fabric is busy per
 * byte fetched. Interpolates between the ML.ENERGY bounds on a log scale,
 * saturating around batch 64.
 */
function decodeTdpFraction(batchSize: number): number {
  const span = Math.min(1, Math.log2(1 + batchSize) / Math.log2(1 + 64));
  return (
    DECODE_TDP_FRACTION_MIN +
    (DECODE_TDP_FRACTION_MAX - DECODE_TDP_FRACTION_MIN) * span
  );
}

export function computeEnergy(
  gpu: GpuSpec,
  numGpus: number,
  performance: PerformanceResult,
  workload: Workload,
  options: EnergyOptions,
  peakFlops: number,
  /**
   * Host draw split into its parts. When absent, `options.hostOverheadW` is
   * used as a single undifferentiated figure.
   */
  hostPower?: {
    baseW: number;
    ramIdleW: number;
    ramActiveW: number;
    /** Chassis cooling, applied to the heat the system actually dissipates. */
    cooling?: CoolingSpec;
  },
): EnergyResult {
  const idleW = options.idleW ?? gpu.idleW;
  const epsFlopPJ = options.epsFlopPJ ?? defaultEpsFlopPJ(gpu, peakFlops);
  const epsMopPJ = options.epsMopPJ ?? defaultEpsMopPJ(gpu);

  const decodeStepSeconds = performance.msPerToken / 1000;
  const ttftSeconds = performance.ttftMs / 1000;

  let decodePowerW: number;
  let prefillPowerW: number;
  let decomposition: EnergyResult['decomposition'];

  if (options.mode === 'roofline') {
    // E = eps_flop * W + eps_mop * Q + pi_0 * T, evaluated per phase and
    // scaled across the rig. eps values are per-GPU, so the FLOPs and bytes
    // are divided across GPUs while pi_0 is paid by each of them.
    const computeJoules = (epsFlopPJ * 1e-12) * performance.flopsPerDecodeStep;
    const memoryJoules = (epsMopPJ * 1e-12) * performance.bytesPerDecodeStep;
    const staticJoules = idleW * numGpus * decodeStepSeconds;

    const decodeStepJoules = computeJoules + memoryJoules + staticJoules;
    decodePowerW = decodeStepSeconds > 0 ? decodeStepJoules / decodeStepSeconds : idleW;

    const prefillJoules =
      (epsFlopPJ * 1e-12) * performance.prefillFlops +
      // Prefill moves its own byte volume — weights read once plus the KV
      // written for the prompt — not the decode step's.
      (epsMopPJ * 1e-12) * performance.prefillBytesMoved +
      idleW * numGpus * ttftSeconds;
    prefillPowerW = ttftSeconds > 0 ? prefillJoules / ttftSeconds : idleW;

    decomposition = { computeJoules, memoryJoules, staticJoules };
  } else {
    const dynamicW = Math.max(0, gpu.tdpW - gpu.idleW);
    decodePowerW =
      (idleW + dynamicW * decodeTdpFraction(workload.batchSize)) * numGpus;
    prefillPowerW = (idleW + dynamicW * PREFILL_TDP_FRACTION) * numGpus;
  }

  // Cap at the physical limit — the roofline decomposition can overshoot if
  // the eps coefficients have been calibrated aggressively.
  const powerCeiling = gpu.tdpW * numGpus;
  decodePowerW = Math.min(decodePowerW, powerCeiling);
  prefillPowerW = Math.min(prefillPowerW, powerCeiling);

  const tokensPerSec = performance.decodeTokensPerSecTotal;
  const joulesPerToken = tokensPerSec > 0 ? decodePowerW / tokensPerSec : 0;

  // Wall-plug energy: the GPU is only part of what the meter sees. The host
  // draws power too, the PSU wastes some, and a datacenter multiplies the
  // whole thing by its PUE.
  let host: EnergyResult['host'];
  if (hostPower) {
    // Everything dissipating heat inside the chassis. Fan power is excluded
    // from its own input to avoid a circular definition; it is exhausted
    // rather than reheating the intake, so the omission is second order.
    const heatLoadW =
      decodePowerW + hostPower.baseW + hostPower.ramIdleW + hostPower.ramActiveW;
    const coolingW = hostPower.cooling
      ? coolingPowerW(hostPower.cooling, heatLoadW)
      : 0;

    host = {
      baseW: hostPower.baseW,
      ramIdleW: hostPower.ramIdleW,
      ramActiveW: hostPower.ramActiveW,
      coolingW,
      heatLoadW,
      totalW: hostPower.baseW + hostPower.ramIdleW + hostPower.ramActiveW + coolingW,
    };
  } else {
    host = {
      baseW: options.hostOverheadW,
      ramIdleW: 0,
      ramActiveW: 0,
      coolingW: 0,
      heatLoadW: decodePowerW + options.hostOverheadW,
      totalW: options.hostOverheadW,
    };
  }

  const systemPowerW = (decodePowerW + host.totalW) / Math.max(0.5, options.psuEfficiency);
  const wallPowerW = systemPowerW * options.pue;
  const wallJoulesPerToken = tokensPerSec > 0 ? wallPowerW / tokensPerSec : 0;

  const kWhPerMTokens = (wallJoulesPerToken * 1e6) / 3.6e6;

  return {
    decodePowerW,
    prefillPowerW,
    joulesPerToken,
    wallJoulesPerToken,
    kWhPerMTokens,
    wallPowerW,
    decomposition,
    epsFlopPJ,
    epsMopPJ,
    host,
  };
}
