import type { Bound, HardwareConfig, Parallelism } from './types';

/**
 * The roofline model, following the formulation in the thesis research design.
 *
 *   P_attain = min(F, B * I)
 *   I_ridge  = F / B
 *
 * where F is peak compute (FLOP/s), B is peak memory bandwidth (byte/s) and
 * I is arithmetic intensity (FLOP/byte). Work below the ridge point is
 * bandwidth-bound; above it, compute-bound.
 *
 * The two inference phases sit on opposite sides of that ridge:
 *
 * - Prefill is a matrix-matrix product (GEMM). Intensity is high — on the
 *   order of 2048 FLOP/byte at N=4096 in FP16 — placing it firmly on the
 *   compute plateau.
 * - Decode is a matrix-vector product (GEMV): every weight is read once to
 *   produce one token, so intensity collapses to roughly the batch size and
 *   the phase is deep in the bandwidth-bound region.
 *
 * That second result is why `I ~ B_batch` and why batching is the single most
 * effective lever on decode efficiency.
 */

export function arithmeticIntensity(flops: number, bytes: number): number {
  if (bytes <= 0) return 0;
  return flops / bytes;
}

/** I_ridge = F / B — the intensity at which a kernel stops being memory-bound. */
export function ridgePoint(flopsPerSec: number, bytesPerSec: number): number {
  if (bytesPerSec <= 0) return Infinity;
  return flopsPerSec / bytesPerSec;
}

/** P_attain = min(F, B * I). */
export function attainablePerformance(
  flopsPerSec: number,
  bytesPerSec: number,
  intensity: number,
): number {
  return Math.min(flopsPerSec, bytesPerSec * intensity);
}

export function boundOf(intensity: number, ridge: number): Bound {
  return intensity >= ridge ? 'compute' : 'memory';
}

/**
 * Multi-GPU scaling efficiency.
 *
 * Tensor parallelism issues two all-reduces per layer, so it is sensitive to
 * interconnect: NVLink/NVSwitch holds up well (a 2-GPU NVLink pair gains ~50%
 * throughput, tapering to ~10% by 4 GPUs), while PCIe all-reduce can eat
 * 30-40% of step time on 8-GPU nodes. Pipeline parallelism only moves
 * activations at stage boundaries, so it survives PCIe far better — but at
 * batch 1 it does not improve latency at all, since only one stage is active
 * at a time.
 *
 * No authoritative closed form for this exists in the literature; the
 * coefficients below are a fit to published measurements and are exposed as a
 * heuristic rather than presented as ground truth.
 */
export function parallelEfficiency(
  numGpus: number,
  parallelism: Parallelism,
  nvlink: boolean,
  batchSize = 1,
): number {
  if (numGpus <= 1) return 1;
  const steps = Math.log2(numGpus);

  if (parallelism === 'pp') {
    // Pipeline parallelism does not aggregate bandwidth for one sequence: the
    // token walks the stages in order, so only one GPU is working at a time
    // and the others sit in the bubble. Utilisation is the classic GPipe
    // fraction, microbatches over microbatches-plus-bubble:
    //
    //     eta = B / (B + N - 1)
    //
    // At B=1 over 8 stages that is 1/8, which cancels the 8x aggregation in
    // `effectiveHardware` and correctly leaves a single GPU's bandwidth. The
    // previous flat ~0.95 claimed an 8x speedup the hardware cannot deliver,
    // contradicting this function's own documentation.
    return Math.max(0.02, batchSize / (batchSize + numGpus - 1));
  }

  // Tensor parallelism issues two all-reduces per layer, so it is bound by
  // the interconnect rather than by scheduling.
  const penaltyPerDoubling = nvlink ? 0.05 : 0.2;
  return Math.max(0.1, 1 - penaltyPerDoubling * steps);
}

export interface EffectiveHardware {
  bytesPerSec: number;
  flopsPerSec: number;
  efficiency: number;
}

/**
 * Aggregate bandwidth and compute across the rig, after the interconnect
 * penalty. `peakFlops` is the dense figure for the compute precision actually
 * in use — never the 2:4-sparse marketing number.
 */
export function effectiveHardware(
  hardware: HardwareConfig,
  peakFlops: number,
  mbu: number,
  mfu: number,
  batchSize = 1,
): EffectiveHardware {
  const nvlink = hardware.nvlink ?? hardware.gpu.nvlink ?? false;
  const efficiency = parallelEfficiency(
    hardware.numGpus,
    hardware.parallelism,
    nvlink,
    batchSize,
  );

  const bytesPerSec =
    hardware.gpu.bandwidthGBs * 1e9 * hardware.numGpus * efficiency * mbu;
  const flopsPerSec = peakFlops * hardware.numGpus * efficiency * mfu;

  return { bytesPerSec, flopsPerSec, efficiency };
}

/**
 * Dense peak FLOPS for the compute precision the quantization implies.
 * Falls back to the FP16 rate when the hardware has no native support.
 */
export function peakFlopsFor(
  gpu: { fp16TFlops: number; fp8TFlops?: number; fp4TFlops?: number },
  precision: 'fp16' | 'fp8' | 'fp4' | undefined,
): number {
  const tflops =
    precision === 'fp4'
      ? gpu.fp4TFlops ?? gpu.fp8TFlops ?? gpu.fp16TFlops
      : precision === 'fp8'
        ? gpu.fp8TFlops ?? gpu.fp16TFlops
        : gpu.fp16TFlops;
  return tflops * 1e12;
}
