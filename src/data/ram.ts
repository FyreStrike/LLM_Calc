import type { RamTypeSpec } from '../core/types';

/**
 * System memory reference data.
 *
 * Bandwidth is arithmetic, not an estimate:
 *
 *   GB/s = MT/s x 8 bytes x channels / 1000
 *
 * A DDR5 DIMM splits into two 32-bit sub-channels, but the module is still
 * 64 bits wide, so the same formula holds for both generations.
 *
 * Power figures are per module. Idle is what the DIMMs draw while the GPU does
 * the work; active is what they draw while weights are being streamed from
 * host RAM during offloading. The gap is large enough to matter — four
 * DDR5-5200 modules go from ~7.6 W idle to ~29 W saturated.
 *
 * Sources: Micron TN-40-07 (DDR4 calculated),
 * https://www.mouser.com/pdfDocs/tn4007_ddr4_power_calculation.pdf
 * and der8auer's measurements of DDR5-4800/5200/5400 at 1.25/1.35/1.5 V,
 * https://x.com/aschilling/status/1455556756357226500
 *
 * The measured DDR5 points rise steeply partly because voltage rises with
 * them, so `activeWattsPerModule` scales the reference point linearly with
 * clock rather than claiming a curve those three points cannot support.
 */
export const RAM_TYPES: RamTypeSpec[] = [
  {
    id: 'ddr4',
    label: 'DDR4',
    speeds: [2133, 2400, 2666, 2933, 3200, 3600, 4000],
    defaultSpeed: 3200,
    idleWattsPerModule: 1.4,
    referenceSpeedMTps: 2666,
    referenceActiveWatts: 3.3,
  },
  {
    id: 'ddr5',
    label: 'DDR5',
    speeds: [4400, 4800, 5200, 5600, 6000, 6400, 7200, 8000],
    defaultSpeed: 5600,
    idleWattsPerModule: 1.9,
    referenceSpeedMTps: 4800,
    referenceActiveWatts: 5.8,
  },
  {
    id: 'lpddr5',
    label: 'LPDDR5 / LPDDR5X',
    speeds: [5500, 6400, 7500, 8533, 9600],
    defaultSpeed: 6400,
    // Soldered, low-voltage, and typically on a wide bus — far more efficient
    // per bit than a socketed DIMM.
    idleWattsPerModule: 0.6,
    referenceSpeedMTps: 6400,
    referenceActiveWatts: 2.5,
  },
];

export function getRamType(id: string): RamTypeSpec {
  return RAM_TYPES.find((r) => r.id === id) ?? RAM_TYPES[1];
}

/**
 * Everything drawing power in the host besides the GPU and the RAM: CPU at
 * idle, board and VRM losses, drives, fans. RAM is costed separately because
 * it is the one component whose draw depends on what the model is doing.
 *
 * Rough composition of the desktop figure: board and VRM ~20 W, CPU idle
 * ~20 W, one NVMe ~2 W, three fans ~5 W, plus headroom for the CPU-side work
 * of tokenizing and scheduling during inference.
 */
export const HOST_PRESETS = [
  { id: 'laptop', labelKey: 'host.laptop', baseOverheadW: 12, ramModules: 2, ramCapacityGb: 16 },
  { id: 'desktop', labelKey: 'host.desktop', baseOverheadW: 55, ramModules: 2, ramCapacityGb: 32 },
  { id: 'workstation', labelKey: 'host.workstation', baseOverheadW: 95, ramModules: 4, ramCapacityGb: 128 },
  { id: 'server', labelKey: 'host.server', baseOverheadW: 250, ramModules: 16, ramCapacityGb: 768 },
] as const;

export type HostPresetId = (typeof HOST_PRESETS)[number]['id'];

export function getHostPreset(id: string) {
  return HOST_PRESETS.find((h) => h.id === id) ?? HOST_PRESETS[1];
}

/** Common channel counts, so the field is a pick rather than a guess. */
export const CHANNEL_OPTIONS = [1, 2, 4, 6, 8, 12];

/**
 * Share of installed RAM the OS and other processes leave available for
 * offloaded weights. Filling host memory to the brim triggers swapping, which
 * is far worse than not offloading at all.
 */
export const HOST_RAM_USABLE_FRACTION = 0.85;
