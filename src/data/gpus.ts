import type { GpuSegment, GpuSpec } from '../core/types';

/**
 * Hardware reference database.
 *
 * IMPORTANT — all TFLOPS figures here are **dense**, i.e. without 2:4
 * structured sparsity. Vendor marketing routinely quotes the sparse number,
 * which is exactly 2x the dense one and irrelevant for LLM inference. NVIDIA's
 * H100 datasheet, for instance, headlines "1,979 TFLOPS FP16" with an asterisk;
 * the dense figure is 989.5, and that is what appears below and what the
 * thesis research design uses when computing the H100 ridge point of
 * ~295 FLOP/byte.
 *
 * For GeForce cards the figure is FP16 with FP32 accumulate, which runs at half
 * rate on consumer silicon — again, the number that matters in practice.
 *
 * Idle power is the headless desktop-idle figure where measured; datacenter
 * idle values are estimates, as no reliable published figures were found.
 */
export const GPUS: GpuSpec[] = [
  // -------------------------------------------------------------- NVIDIA consumer
  {
    id: 'rtx-3060-12gb',
    name: 'RTX 3060 12GB',
    vendor: 'nvidia',
    architecture: 'Ampere',
    vramGb: 12,
    bandwidthGBs: 360,
    fp16TFlops: 51,
    tdpW: 170,
    idleW: 12,
    priceUsd: 329,
  },
  {
    id: 'rtx-3090',
    name: 'RTX 3090',
    vendor: 'nvidia',
    architecture: 'Ampere',
    vramGb: 24,
    bandwidthGBs: 936,
    fp16TFlops: 71,
    tdpW: 350,
    idleW: 20,
    priceUsd: 1499,
    nvlink: true,
  },
  {
    id: 'rtx-4060-ti-16gb',
    name: 'RTX 4060 Ti 16GB',
    vendor: 'nvidia',
    architecture: 'Ada Lovelace',
    vramGb: 16,
    bandwidthGBs: 288,
    fp16TFlops: 82,
    fp8TFlops: 165,
    tdpW: 165,
    idleW: 10,
    priceUsd: 499,
    note: 'gpu.note.narrowBus',
  },
  {
    id: 'rtx-4070',
    name: 'RTX 4070',
    vendor: 'nvidia',
    architecture: 'Ada Lovelace',
    vramGb: 12,
    bandwidthGBs: 504,
    fp16TFlops: 58,
    fp8TFlops: 117,
    tdpW: 200,
    idleW: 12,
    priceUsd: 599,
  },
  {
    id: 'rtx-4080',
    name: 'RTX 4080',
    vendor: 'nvidia',
    architecture: 'Ada Lovelace',
    vramGb: 16,
    bandwidthGBs: 717,
    fp16TFlops: 97,
    fp8TFlops: 195,
    tdpW: 320,
    idleW: 13,
    priceUsd: 1199,
  },
  {
    id: 'rtx-4090',
    name: 'RTX 4090',
    vendor: 'nvidia',
    architecture: 'Ada Lovelace',
    vramGb: 24,
    bandwidthGBs: 1008,
    fp16TFlops: 165,
    fp8TFlops: 330,
    tdpW: 450,
    idleW: 20,
    priceUsd: 1599,
  },
  {
    id: 'rtx-5070',
    name: 'RTX 5070',
    vendor: 'nvidia',
    architecture: 'Blackwell',
    vramGb: 12,
    bandwidthGBs: 672,
    fp16TFlops: 123.5,
    fp8TFlops: 247,
    fp4TFlops: 494,
    tdpW: 250,
    idleW: 15,
    priceUsd: 549,
    note: 'gpu.note.blackwellEstimate',
  },
  {
    id: 'rtx-5070-ti',
    name: 'RTX 5070 Ti',
    vendor: 'nvidia',
    architecture: 'Blackwell',
    vramGb: 16,
    bandwidthGBs: 896,
    fp16TFlops: 177.4,
    fp8TFlops: 355,
    fp4TFlops: 710,
    tdpW: 300,
    idleW: 17,
    priceUsd: 749,
    note: 'gpu.note.blackwellEstimate',
  },
  {
    id: 'rtx-5080',
    name: 'RTX 5080',
    vendor: 'nvidia',
    architecture: 'Blackwell',
    vramGb: 16,
    bandwidthGBs: 960,
    fp16TFlops: 225.1,
    fp8TFlops: 450,
    fp4TFlops: 900,
    tdpW: 360,
    idleW: 20,
    priceUsd: 999,
    note: 'gpu.note.blackwellEstimate',
  },
  {
    id: 'rtx-5090',
    name: 'RTX 5090',
    vendor: 'nvidia',
    architecture: 'Blackwell',
    vramGb: 32,
    bandwidthGBs: 1792,
    fp16TFlops: 419.2,
    fp8TFlops: 838,
    fp4TFlops: 1676,
    tdpW: 575,
    idleW: 30,
    priceUsd: 1999,
    note: 'gpu.note.blackwellEstimate',
  },

  // ---------------------------------------------------------- NVIDIA workstation
  {
    id: 'rtx-a6000',
    name: 'RTX A6000 (Ampere)',
    vendor: 'nvidia',
    architecture: 'Ampere',
    vramGb: 48,
    bandwidthGBs: 768,
    fp16TFlops: 155,
    tdpW: 300,
    idleW: 20,
    priceUsd: 4650,
    nvlink: true,
  },
  {
    id: 'rtx-6000-ada',
    name: 'RTX 6000 Ada',
    vendor: 'nvidia',
    architecture: 'Ada Lovelace',
    vramGb: 48,
    bandwidthGBs: 960,
    fp16TFlops: 182,
    fp8TFlops: 364,
    tdpW: 300,
    idleW: 20,
    priceUsd: 6800,
  },
  {
    id: 'rtx-pro-6000-blackwell',
    name: 'RTX PRO 6000 Blackwell',
    vendor: 'nvidia',
    architecture: 'Blackwell',
    vramGb: 96,
    bandwidthGBs: 1792,
    fp16TFlops: 503,
    fp8TFlops: 1007,
    fp4TFlops: 2014,
    tdpW: 600,
    idleW: 30,
    priceUsd: 13250,
    note: 'gpu.note.proBlackwellEstimate',
  },

  // ----------------------------------------------------------- NVIDIA datacenter
  {
    id: 'a100-40gb-sxm',
    name: 'A100 40GB SXM',
    vendor: 'nvidia',
    architecture: 'Ampere',
    vramGb: 40,
    bandwidthGBs: 1555,
    fp16TFlops: 312,
    tdpW: 400,
    idleW: 50,
    nvlink: true,
  },
  {
    id: 'a100-80gb-sxm',
    name: 'A100 80GB SXM',
    vendor: 'nvidia',
    architecture: 'Ampere',
    vramGb: 80,
    bandwidthGBs: 2039,
    fp16TFlops: 312,
    tdpW: 400,
    idleW: 50,
    nvlink: true,
  },
  {
    id: 'a100-80gb-pcie',
    name: 'A100 80GB PCIe',
    vendor: 'nvidia',
    architecture: 'Ampere',
    vramGb: 80,
    bandwidthGBs: 1935,
    fp16TFlops: 312,
    tdpW: 300,
    idleW: 50,
  },
  {
    id: 'h100-sxm',
    name: 'H100 SXM5',
    vendor: 'nvidia',
    architecture: 'Hopper (SM 9.0)',
    vramGb: 80,
    bandwidthGBs: 3350,
    fp16TFlops: 989,
    fp8TFlops: 1979,
    tdpW: 700,
    idleW: 75,
    nvlink: true,
  },
  {
    id: 'h100-pcie',
    name: 'H100 PCIe',
    vendor: 'nvidia',
    architecture: 'Hopper (SM 9.0)',
    vramGb: 80,
    bandwidthGBs: 2000,
    fp16TFlops: 756,
    fp8TFlops: 1513,
    tdpW: 350,
    idleW: 60,
  },
  {
    id: 'h200-sxm',
    name: 'H200 SXM',
    vendor: 'nvidia',
    architecture: 'Hopper (SM 9.0)',
    vramGb: 141,
    bandwidthGBs: 4800,
    fp16TFlops: 989,
    fp8TFlops: 1979,
    tdpW: 700,
    idleW: 75,
    nvlink: true,
  },
  {
    id: 'h200-nvl',
    name: 'H200 NVL (PCIe)',
    vendor: 'nvidia',
    architecture: 'Hopper (SM 9.0)',
    vramGb: 141,
    bandwidthGBs: 4800,
    // Same memory as the SXM part, but lower clocks in the PCIe envelope:
    // 1,671 TFLOPS FP16 and 3,341 FP8 with sparsity, so half those dense.
    fp16TFlops: 835,
    fp8TFlops: 1671,
    // Configurable 450-600 W; the ceiling is used here.
    tdpW: 600,
    idleW: 65,
    // NVLink bridge on the card, 2-way or 4-way at 900 GB/s per GPU.
    nvlink: true,
    note: 'gpu.note.h200nvl',
  },
  {
    id: 'b200-sxm',
    name: 'B200 SXM',
    vendor: 'nvidia',
    architecture: 'Blackwell (SM 10.0)',
    vramGb: 192,
    bandwidthGBs: 8000,
    fp16TFlops: 2250,
    fp8TFlops: 4500,
    fp4TFlops: 9000,
    tdpW: 1000,
    idleW: 100,
    nvlink: true,
    note: 'gpu.note.b200Capacity',
  },
  {
    id: 'b300-sxm',
    name: 'B300 SXM (Blackwell Ultra)',
    vendor: 'nvidia',
    architecture: 'Blackwell Ultra',
    vramGb: 288,
    bandwidthGBs: 8000,
    fp16TFlops: 3500,
    fp8TFlops: 7000,
    fp4TFlops: 15000,
    tdpW: 1400,
    idleW: 120,
    nvlink: true,
  },
  {
    id: 'l40s',
    name: 'L40S',
    vendor: 'nvidia',
    architecture: 'Ada Lovelace',
    vramGb: 48,
    bandwidthGBs: 864,
    fp16TFlops: 181,
    fp8TFlops: 362,
    tdpW: 350,
    idleW: 30,
    priceUsd: 8000,
  },
  {
    id: 'dgx-spark-gb10',
    name: 'DGX Spark (GB10)',
    vendor: 'nvidia',
    architecture: 'Grace Blackwell',
    vramGb: 128,
    bandwidthGBs: 273,
    fp16TFlops: 125,
    fp8TFlops: 250,
    fp4TFlops: 500,
    tdpW: 140,
    idleW: 15,
    priceUsd: 3999,
    unified: true,
    soc: true,
    // Grace-Blackwell coherent memory under Linux is not subject to Metal's
    // 75% cap: nearly the whole 128 GB pool is allocatable, leaving only an
    // OS reserve. Confirmed by hands-on testing rather than assumed from the
    // Apple behaviour.
    usableMemoryFraction: 0.95,
    note: 'gpu.note.gb10',
  },

  // ------------------------------------------------------------------------ AMD
  {
    id: 'rx-7900-xtx',
    name: 'Radeon RX 7900 XTX',
    vendor: 'amd',
    architecture: 'RDNA 3',
    vramGb: 24,
    bandwidthGBs: 960,
    fp16TFlops: 123,
    tdpW: 355,
    idleW: 25,
    priceUsd: 999,
  },
  {
    id: 'mi300x',
    name: 'Instinct MI300X',
    vendor: 'amd',
    architecture: 'CDNA 3',
    vramGb: 192,
    bandwidthGBs: 5325,
    fp16TFlops: 1307,
    fp8TFlops: 2615,
    tdpW: 750,
    idleW: 90,
  },
  {
    id: 'mi325x',
    name: 'Instinct MI325X',
    vendor: 'amd',
    architecture: 'CDNA 3',
    vramGb: 256,
    bandwidthGBs: 6000,
    fp16TFlops: 1307,
    fp8TFlops: 2615,
    tdpW: 1000,
    idleW: 100,
  },
  {
    id: 'mi355x',
    name: 'Instinct MI355X',
    vendor: 'amd',
    architecture: 'CDNA 4',
    vramGb: 288,
    bandwidthGBs: 8000,
    fp16TFlops: 2300,
    fp8TFlops: 4600,
    fp4TFlops: 9200,
    tdpW: 1400,
    idleW: 120,
    note: 'gpu.note.unverified',
  },

  // ---------------------------------------------------------------- Apple Silicon
  // Unified memory: Metal caps GPU allocation near 75% by default, raisable
  // via iogpu.wired_limit_mb. These are SoCs, so `tdpW` is package power and
  // already includes the CPU — no separate CPU may be added on top.
  // FP16 figures are approximate; decode is bandwidth-bound on these parts,
  // so bandwidth is the number that actually matters.
  { id: 'm1', name: 'Apple M1', vendor: 'apple', vramGb: 16, bandwidthGBs: 68.25, fp16TFlops: 5.2, tdpW: 20, idleW: 3, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm1-pro', name: 'Apple M1 Pro', vendor: 'apple', vramGb: 32, bandwidthGBs: 200, fp16TFlops: 10.4, tdpW: 30, idleW: 4, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm1-max', name: 'Apple M1 Max', vendor: 'apple', vramGb: 64, bandwidthGBs: 400, fp16TFlops: 20.8, tdpW: 60, idleW: 5, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm1-ultra', name: 'Apple M1 Ultra', vendor: 'apple', vramGb: 128, bandwidthGBs: 800, fp16TFlops: 41.6, tdpW: 120, idleW: 8, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm2', name: 'Apple M2', vendor: 'apple', vramGb: 24, bandwidthGBs: 100, fp16TFlops: 7.1, tdpW: 20, idleW: 3, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm2-pro', name: 'Apple M2 Pro', vendor: 'apple', vramGb: 32, bandwidthGBs: 200, fp16TFlops: 13.6, tdpW: 30, idleW: 4, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm2-max', name: 'Apple M2 Max', vendor: 'apple', vramGb: 96, bandwidthGBs: 400, fp16TFlops: 27.2, tdpW: 60, idleW: 5, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm2-ultra', name: 'Apple M2 Ultra', vendor: 'apple', vramGb: 192, bandwidthGBs: 800, fp16TFlops: 54.4, tdpW: 120, idleW: 8, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm3', name: 'Apple M3', vendor: 'apple', vramGb: 24, bandwidthGBs: 100, fp16TFlops: 7.1, tdpW: 20, idleW: 3, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm3-pro', name: 'Apple M3 Pro', vendor: 'apple', vramGb: 36, bandwidthGBs: 150, fp16TFlops: 12.8, tdpW: 30, idleW: 4, unified: true, soc: true, usableMemoryFraction: 0.75, note: 'gpu.note.m3ProRegression' },
  { id: 'm3-max', name: 'Apple M3 Max', vendor: 'apple', vramGb: 128, bandwidthGBs: 400, fp16TFlops: 28.4, tdpW: 65, idleW: 5, unified: true, soc: true, usableMemoryFraction: 0.75, note: 'gpu.note.appleBinning' },
  { id: 'm3-ultra', name: 'Apple M3 Ultra', vendor: 'apple', vramGb: 512, bandwidthGBs: 819, fp16TFlops: 56.8, tdpW: 140, idleW: 8, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm4', name: 'Apple M4', vendor: 'apple', vramGb: 32, bandwidthGBs: 120, fp16TFlops: 8.5, tdpW: 22, idleW: 3, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm4-pro', name: 'Apple M4 Pro', vendor: 'apple', vramGb: 64, bandwidthGBs: 273, fp16TFlops: 17, tdpW: 35, idleW: 4, unified: true, soc: true, usableMemoryFraction: 0.75 },
  { id: 'm4-max', name: 'Apple M4 Max', vendor: 'apple', vramGb: 128, bandwidthGBs: 546, fp16TFlops: 34, tdpW: 70, idleW: 5, unified: true, soc: true, usableMemoryFraction: 0.75, note: 'gpu.note.appleBinning' },

  // ---------------------------------------------------------------------- Intel
  {
    id: 'arc-b580',
    name: 'Intel Arc B580',
    vendor: 'intel',
    architecture: 'Battlemage',
    vramGb: 12,
    bandwidthGBs: 456,
    fp16TFlops: 46,
    tdpW: 190,
    idleW: 10,
    priceUsd: 249,
    note: 'gpu.note.unverified',
  },
  {
    id: 'arc-pro-b60',
    name: 'Intel Arc Pro B60 24GB',
    vendor: 'intel',
    architecture: 'Battlemage',
    vramGb: 24,
    bandwidthGBs: 456,
    fp16TFlops: 46,
    tdpW: 200,
    idleW: 10,
    priceUsd: 599,
    note: 'gpu.note.unverified',
  },
];

/**
 * Market segment per GPU, kept in one place rather than repeated on every
 * entry. This drives decode power: measured draw as a fraction of TDP differs
 * sharply between a consumer card and a datacenter part, because the memory
 * subsystem costs roughly the same absolute power in both while the thermal
 * rating does not.
 */
const DATACENTER = new Set([
  'a100-40gb-sxm', 'a100-80gb-sxm', 'a100-80gb-pcie', 'h100-sxm', 'h100-pcie',
  'h200-sxm', 'h200-nvl', 'b200-sxm', 'b300-sxm', 'l40s', 'mi300x', 'mi325x',
  'mi355x',
]);
const WORKSTATION = new Set([
  'rtx-a6000', 'rtx-6000-ada', 'rtx-pro-6000-blackwell', 'arc-pro-b60',
]);

function inferSegment(gpu: GpuSpec): GpuSegment {
  if (gpu.soc) return 'soc';
  if (DATACENTER.has(gpu.id)) return 'datacenter';
  if (WORKSTATION.has(gpu.id)) return 'workstation';
  return 'consumer';
}

// Applied once at module load so every consumer sees a populated field.
for (const gpu of GPUS) {
  gpu.segment ??= inferSegment(gpu);
}

export function getGpu(id: string): GpuSpec | undefined {
  return GPUS.find((g) => g.id === id);
}

export const GPU_GROUPS: { vendor: string; labelKey: string }[] = [
  { vendor: 'nvidia', labelKey: 'gpu.group.nvidia' },
  { vendor: 'amd', labelKey: 'gpu.group.amd' },
  { vendor: 'apple', labelKey: 'gpu.group.apple' },
  { vendor: 'intel', labelKey: 'gpu.group.intel' },
];
