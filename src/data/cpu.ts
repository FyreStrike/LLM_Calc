import type { CoolingSpec, CpuSpec, DriveSpec } from '../core/types';

/**
 * CPU reference data for the host power model.
 *
 * A flat "55 W of host overhead" is meaningless across the range this tool
 * covers: two Xeon Platinum sockets idle at more than triple that before the
 * board, drives or fans are counted.
 *
 * `tdpW` is the published figure. `idleW` is package power at idle and is an
 * **estimate** — vendors do not publish it, and measurements in the wild are
 * whole-system numbers that fold in board, RAM and PSU losses. The values
 * below sit at roughly 15-30% of TDP, higher for server parts because a large
 * IO die draws substantial power regardless of core activity. Reported
 * system-level idle for single-socket EPYC platforms lands near 100 W, which
 * these figures reproduce once board, RAM and cooling are added on top.
 *
 * Sources for the system-level anchors:
 * https://forums.servethehome.com/index.php?threads/xeon-scalable-vs-epyc-idle-power-consumption.39037/
 * https://www.anandtech.com/show/11544/intel-skylake-ep-vs-amd-epyc-7000-cpu-battle-of-the-decade/22
 */
export const CPUS: CpuSpec[] = [
  // ------------------------------------------------------------------ mobile
  { id: 'mobile-u', label: 'Laptop U-Serie / SoC', vendor: 'intel', segment: 'mobile', cores: 10, tdpW: 15, idleW: 3 },
  { id: 'mobile-h', label: 'Laptop H-Serie', vendor: 'intel', segment: 'mobile', cores: 14, tdpW: 45, idleW: 6 },
  { id: 'apple-m-soc', label: 'Apple Silicon (SoC)', vendor: 'apple', segment: 'mobile', cores: 12, tdpW: 30, idleW: 3 },

  // ----------------------------------------------------------------- desktop
  { id: 'ryzen5-7600', label: 'Ryzen 5 7600', vendor: 'amd', segment: 'desktop', cores: 6, tdpW: 65, idleW: 12 },
  { id: 'ryzen7-9700x', label: 'Ryzen 7 9700X', vendor: 'amd', segment: 'desktop', cores: 8, tdpW: 65, idleW: 15 },
  { id: 'ryzen9-9950x', label: 'Ryzen 9 9950X', vendor: 'amd', segment: 'desktop', cores: 16, tdpW: 170, idleW: 25 },
  { id: 'core-i5-14600k', label: 'Core i5-14600K', vendor: 'intel', segment: 'desktop', cores: 14, tdpW: 125, idleW: 14 },
  { id: 'core-i7-14700k', label: 'Core i7-14700K', vendor: 'intel', segment: 'desktop', cores: 20, tdpW: 125, idleW: 18 },
  { id: 'core-i9-14900k', label: 'Core i9-14900K', vendor: 'intel', segment: 'desktop', cores: 24, tdpW: 125, idleW: 24 },

  // ------------------------------------------------------------- workstation
  { id: 'threadripper-7960x', label: 'Threadripper 7960X', vendor: 'amd', segment: 'workstation', cores: 24, tdpW: 350, idleW: 60 },
  { id: 'threadripper-7995wx', label: 'Threadripper PRO 7995WX', vendor: 'amd', segment: 'workstation', cores: 96, tdpW: 350, idleW: 85 },
  { id: 'xeon-w5-3435x', label: 'Xeon w5-3435X', vendor: 'intel', segment: 'workstation', cores: 16, tdpW: 270, idleW: 45 },

  // ------------------------------------------------------------------ server
  { id: 'xeon-silver-4410y', label: 'Xeon Silver 4410Y', vendor: 'intel', segment: 'server', cores: 12, tdpW: 150, idleW: 40 },
  { id: 'xeon-gold-6430', label: 'Xeon Gold 6430', vendor: 'intel', segment: 'server', cores: 32, tdpW: 270, idleW: 60 },
  { id: 'xeon-platinum-8480', label: 'Xeon Platinum 8480+', vendor: 'intel', segment: 'server', cores: 56, tdpW: 350, idleW: 85 },
  { id: 'epyc-9354', label: 'EPYC 9354 (Genoa)', vendor: 'amd', segment: 'server', cores: 32, tdpW: 280, idleW: 75 },
  { id: 'epyc-9654', label: 'EPYC 9654 (Genoa)', vendor: 'amd', segment: 'server', cores: 96, tdpW: 360, idleW: 95 },
  { id: 'epyc-7763', label: 'EPYC 7763 (Milan)', vendor: 'amd', segment: 'server', cores: 64, tdpW: 280, idleW: 85 },
  { id: 'grace-cpu', label: 'NVIDIA Grace (72-core Arm)', vendor: 'other', segment: 'server', cores: 72, tdpW: 250, idleW: 55 },
];

export function getCpu(id: string): CpuSpec {
  return CPUS.find((c) => c.id === id) ?? CPUS[4];
}

/**
 * Cooling and chassis airflow, as a fraction of the heat being removed.
 *
 * Two things drive these numbers:
 *
 * 1. **Chassis height caps fan diameter.** 1U (44.45 mm) admits only 40 mm
 *    fans, 2U 60 mm, 4U 80-120 mm. For geometrically similar fans, airflow
 *    goes as `RPM x d^3` while power goes as `RPM^3 x d^5`, so a small fan
 *    must spin far faster for the same airflow and pays cubically for it. A
 *    40 mm server fan delivering 27 CFM draws 12 W; a 120 mm fan moves more
 *    air for 1-3 W.
 * 2. **Restriction.** A 1U air path is narrow and packed, so the fans work
 *    against high back-pressure — an inefficient operating point.
 *
 * Scaling with load rather than fixing a constant matters: eight H100s in a
 * 4U dissipate roughly ten times an empty chassis, and the fans follow.
 *
 * Sources: https://coolingfans.blog/guide-to-selecting-fans-for-1u-2u-3u-server-racks/
 * https://www.servethehome.com/testing-conventional-wisdom-1u-v-2u-power-consumption/
 */
export const COOLING: CoolingSpec[] = [
  { id: 'passive', labelKey: 'cooling.passive', heatFractionOfLoad: 0, idleFloorW: 0 },
  { id: 'laptop', labelKey: 'cooling.laptop', heatFractionOfLoad: 0.04, idleFloorW: 1 },
  { id: 'desktop-air', labelKey: 'cooling.desktopAir', heatFractionOfLoad: 0.02, idleFloorW: 4 },
  { id: 'desktop-highflow', labelKey: 'cooling.desktopHighflow', heatFractionOfLoad: 0.03, idleFloorW: 8 },
  { id: 'aio', labelKey: 'cooling.aio', heatFractionOfLoad: 0.025, idleFloorW: 8 },
  { id: 'server-4u', labelKey: 'cooling.server4u', heatFractionOfLoad: 0.03, idleFloorW: 20 },
  { id: 'server-2u', labelKey: 'cooling.server2u', heatFractionOfLoad: 0.06, idleFloorW: 30 },
  { id: 'server-1u', labelKey: 'cooling.server1u', heatFractionOfLoad: 0.1, idleFloorW: 40 },
];

export function getCooling(id: string): CoolingSpec {
  return COOLING.find((c) => c.id === id) ?? COOLING[2];
}

/** Per-drive idle draw. Active spikes are brief and not modelled. */
export const DRIVES: DriveSpec[] = [
  { id: 'nvme', labelKey: 'drive.nvme', idleW: 2 },
  { id: 'sata-ssd', labelKey: 'drive.sataSsd', idleW: 1 },
  { id: 'hdd', labelKey: 'drive.hdd', idleW: 6 },
];

export function getDrive(id: string): DriveSpec {
  return DRIVES.find((d) => d.id === id) ?? DRIVES[0];
}

/**
 * Mainboard, VRM losses and platform silicon. Server boards carry a BMC,
 * more PCIe lanes and far heavier VRMs, so they idle well above a desktop.
 */
export const BOARDS = [
  { id: 'laptop', labelKey: 'board.laptop', watts: 5 },
  { id: 'desktop', labelKey: 'board.desktop', watts: 20 },
  { id: 'workstation', labelKey: 'board.workstation', watts: 35 },
  { id: 'server', labelKey: 'board.server', watts: 55 },
] as const;

export function getBoard(id: string) {
  return BOARDS.find((b) => b.id === id) ?? BOARDS[1];
}
