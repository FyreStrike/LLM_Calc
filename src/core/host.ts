import { GB } from './memory';
import type { RamSpec, RamTypeSpec } from './types';

/**
 * Host system model: memory bandwidth, capacity and power.
 *
 * Two facts drive everything here:
 *
 * 1. Bandwidth is arithmetic. `MT/s x 8 bytes x channels` is exact, which is
 *    why asking for the RAM specification beats asking the user to estimate
 *    a GB/s figure they have no way of knowing.
 *
 * 2. Idle RAM power scales with what is *installed*, not with what is *used*.
 *    A 512 GB server pays for 512 GB of refresh and PMIC losses around the
 *    clock even while running an 8B model. That cost lands in every token,
 *    and it is invisible if RAM is folded into a single host-overhead number.
 */

/** Peak theoretical bandwidth of the memory subsystem, in bytes per second. */
export function ramBandwidthBytesPerSec(ram: RamSpec): number {
  return ram.speedMTps * 8 * ram.channels * 1e6;
}

/** Convenience for display, in GB/s. */
export function ramBandwidthGBs(ram: RamSpec): number {
  return ramBandwidthBytesPerSec(ram) / 1e9;
}

/**
 * Active draw per module at this clock. The reference point is scaled
 * linearly with frequency — the published measurements span only a few
 * speeds, and their voltages differ, so a fitted curve would imply precision
 * the data does not support.
 */
export function activeWattsPerModule(ram: RamSpec, type: RamTypeSpec): number {
  const scaled =
    type.referenceActiveWatts * (ram.speedMTps / type.referenceSpeedMTps);
  // Active can never sit below idle.
  return Math.max(type.idleWattsPerModule, scaled);
}

export interface RamPower {
  /** Paid continuously for every installed module, used or not. */
  idleW: number;
  /** Additional draw while weights are streaming from host memory. */
  activeW: number;
  totalW: number;
}

/**
 * RAM power at a given duty cycle.
 *
 * `activeFraction` is the share of wall-clock time the memory bus is actually
 * saturated — near zero when the model sits entirely in VRAM, and approaching
 * one when a large fraction of the weights is streamed from host RAM every
 * single token.
 */
export function ramPower(
  ram: RamSpec,
  type: RamTypeSpec,
  activeFraction: number,
): RamPower {
  const duty = Math.min(1, Math.max(0, activeFraction));
  const idleW = ram.modules * type.idleWattsPerModule;
  const perModuleDelta = activeWattsPerModule(ram, type) - type.idleWattsPerModule;
  const activeW = ram.modules * perModuleDelta * duty;

  return { idleW, activeW, totalW: idleW + activeW };
}

/**
 * Bytes of host memory available for offloaded weights. Filling RAM to the
 * brim pushes the OS into swapping, which is dramatically worse than not
 * offloading at all, so a fixed share is held back.
 */
export function hostAvailableBytes(ram: RamSpec, usableFraction: number): number {
  return ram.totalCapacityGb * GB * usableFraction;
}
