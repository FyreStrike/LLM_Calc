import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  ApiPrice,
  CalcInput,
  EnergyMode,
  HostComponents,
  HostSpec,
  ModelSpec,
  Parallelism,
  Region,
  Runtime,
} from '../core/types';
import { hostBaseOverheadW, ramBandwidthGBs } from '../core/host';
import { getKvQuant, getQuant } from '../core/quant';
import { getBoard, getCooling, getCpu, getDrive } from '../data/cpu';
import { API_PRICES, MODEL_TO_API_PRICE } from '../data/apiPrices';
import { GPUS, getGpu } from '../data/gpus';
import { MODELS, getModel } from '../data/models';
import { DEFAULT_REGION_ID, DEFAULT_USD_TO_EUR, REGIONS, getRegion } from '../data/regions';

export interface AppState {
  // selection
  modelId: string;
  customModels: ModelSpec[];
  gpuId: string;
  numGpus: number;
  parallelism: Parallelism;
  nvlinkOverride: boolean | null;
  quantId: string;
  kvQuantId: string;

  // workload
  contextLength: number;
  promptTokens: number;
  outputTokens: number;
  batchSize: number;
  runtime: Runtime;
  prefillChunkTokens: number;
  allowOffload: boolean;
  mbu: number;
  mfu: number;

  // host system — RAM bandwidth and power are derived from these
  hostPresetId: string;
  ramTypeId: string;
  ramSpeedMTps: number;
  ramChannels: number;
  ramCapacityGb: number;
  ramModules: number;
  cpuId: string;
  cpuSockets: number;
  boardId: string;
  coolingId: string;
  driveId: string;
  driveCount: number;
  /** Manual override of the computed component sum; null means "derive it". */
  hostBaseOverheadOverrideW: number | null;

  // energy
  energyMode: EnergyMode;
  epsFlopPJ: number | null;
  epsMopPJ: number | null;
  psuEfficiency: number;
  pue: number;
  hostOverheadW: number;

  // cost
  regionId: string;
  /**
   * Overrides the selected region's tariff when set. Presets are sourced
   * averages; an actual contract rarely matches one, and the price feeds
   * straight into the local-vs-API verdict.
   */
  customPricePerKWh: number | null;
  /** Overrides the region's grid carbon intensity (own PV, green tariff). */
  customGridIntensity: number | null;
  inputRatio: number;
  hardwareCapex: number | null;
  dailyTokens: number;
  usdToCurrency: number;
  /** 'auto' | 'none' | 'custom' | an id from the price list. */
  apiPriceId: string | null;
  /** Used when apiPriceId is 'custom'. USD per 1M tokens, as providers quote. */
  customApiInputPerMTokUsd: number | null;
  customApiOutputPerMTokUsd: number | null;
  livePrices: ApiPrice[] | null;
  priceOrigin: 'live' | 'cache' | 'snapshot';
  priceAsOf: string;

  // ui
  advanced: boolean;
}

export interface AppActions {
  set: <K extends keyof AppState>(key: K, value: AppState[K]) => void;
  patch: (values: Partial<AppState>) => void;
  addCustomModel: (model: ModelSpec) => void;
  removeCustomModel: (id: string) => void;
  setPrices: (
    prices: ApiPrice[],
    origin: 'live' | 'cache' | 'snapshot',
    asOf: string,
  ) => void;
  reset: () => void;
}

/**
 * Explicit selection sentinels for the API comparison.
 *
 * These exist because a single `null` previously meant both "no comparison"
 * and "pick the listing that matches the model", so selecting "no comparison"
 * silently kept comparing against the mapped model.
 */
export const API_PRICE_AUTO = 'auto';
export const API_PRICE_NONE = 'none';
export const API_PRICE_CUSTOM = 'custom';

const DEFAULTS: AppState = {
  modelId: 'llama-3.1-8b',
  customModels: [],
  gpuId: 'rtx-4090',
  numGpus: 1,
  parallelism: 'tp',
  nvlinkOverride: null,
  quantId: 'q4_k_m',
  kvQuantId: 'fp16',

  contextLength: 8192,
  promptTokens: 2048,
  outputTokens: 512,
  batchSize: 1,
  runtime: 'llamacpp',
  prefillChunkTokens: 2048,
  allowOffload: false,

  hostPresetId: 'desktop',
  ramTypeId: 'ddr5',
  ramSpeedMTps: 5600,
  ramChannels: 2,
  ramCapacityGb: 32,
  ramModules: 2,
  cpuId: 'ryzen7-9700x',
  cpuSockets: 1,
  boardId: 'desktop',
  coolingId: 'desktop-air',
  driveId: 'nvme',
  driveCount: 1,
  hostBaseOverheadOverrideW: null,
  // Databricks measured ~60% MBU for batch-1 single-GPU decode; well-tuned
  // setups reach 70-90%. 0.70 is a defensible middle.
  mbu: 0.7,
  // Prefill MFU is typically 30-50%.
  mfu: 0.4,

  energyMode: 'simple',
  epsFlopPJ: null,
  epsMopPJ: null,
  psuEfficiency: 0.9,
  pue: 1.0,
  hostOverheadW: 80,

  regionId: DEFAULT_REGION_ID,
  customPricePerKWh: null,
  customGridIntensity: null,
  inputRatio: 0.7,
  hardwareCapex: null,
  dailyTokens: 1_000_000,
  usdToCurrency: DEFAULT_USD_TO_EUR,
  apiPriceId: API_PRICE_AUTO,
  customApiInputPerMTokUsd: 0.5,
  customApiOutputPerMTokUsd: 1.5,
  livePrices: null,
  priceOrigin: 'snapshot',
  priceAsOf: '',

  advanced: false,
};

export const useStore = create<AppState & AppActions>()(
  persist(
    (set) => ({
      ...DEFAULTS,
      set: (key, value) => set({ [key]: value } as Partial<AppState>),
      patch: (values) => set(values),
      addCustomModel: (model) =>
        set((s) => ({
          customModels: [...s.customModels.filter((m) => m.id !== model.id), model],
          modelId: model.id,
        })),
      removeCustomModel: (id) =>
        set((s) => ({
          customModels: s.customModels.filter((m) => m.id !== id),
          modelId: s.modelId === id ? DEFAULTS.modelId : s.modelId,
        })),
      setPrices: (prices, origin, asOf) =>
        set({ livePrices: prices, priceOrigin: origin, priceAsOf: asOf }),
      reset: () => set(DEFAULTS),
    }),
    {
      name: 'llmcalc.state.v1',
      // Live prices are cached separately and should not bloat the state blob.
      partialize: (s) =>
        Object.fromEntries(
          Object.entries(s).filter(([k]) => k !== 'livePrices' && typeof (s as never)[k as never] !== 'function'),
        ) as AppState,
    },
  ),
);

// ---------------------------------------------------------------------------
// Derived selectors
// ---------------------------------------------------------------------------

export function allModels(state: AppState): ModelSpec[] {
  return [...state.customModels, ...MODELS];
}

export function selectedModel(state: AppState): ModelSpec {
  return (
    state.customModels.find((m) => m.id === state.modelId) ??
    getModel(state.modelId) ??
    MODELS[0]
  );
}

export function availablePrices(state: AppState): ApiPrice[] {
  return state.livePrices ?? API_PRICES;
}

/**
 * Default the API comparison to the listing that serves the same weights, so
 * the headline number compares like with like rather than an arbitrary model.
 */
export function selectedApiPrice(state: AppState): ApiPrice | undefined {
  // `null` used to mean both "no comparison" and "pick one matching the
  // model", so choosing "no comparison" silently kept comparing.
  if (state.apiPriceId === API_PRICE_NONE) return undefined;

  if (state.apiPriceId === API_PRICE_CUSTOM) {
    return {
      id: API_PRICE_CUSTOM,
      label: 'Custom',
      inputPerMTokUsd: state.customApiInputPerMTokUsd ?? 0,
      outputPerMTokUsd: state.customApiOutputPerMTokUsd ?? 0,
    };
  }

  const prices = availablePrices(state);
  if (state.apiPriceId && state.apiPriceId !== API_PRICE_AUTO) {
    const explicit = prices.find((p) => p.id === state.apiPriceId);
    if (explicit) return explicit;
  }

  // Auto: the listing that serves the same weights, so the headline compares
  // like with like.
  const mapped = MODEL_TO_API_PRICE[state.modelId];
  return mapped ? prices.find((p) => p.id === mapped) : undefined;
}

/**
 * The region actually used for costing: the selected preset, with the user's
 * own tariff and grid intensity substituted where they supplied them.
 */
export function effectiveRegion(state: AppState): Region {
  const base = getRegion(state.regionId) ?? REGIONS[0];
  const custom = state.customPricePerKWh !== null || state.customGridIntensity !== null;
  if (!custom) return base;

  return {
    ...base,
    id: 'custom',
    label: 'region.custom',
    pricePerKWh: state.customPricePerKWh ?? base.pricePerKWh,
    gridIntensityGCO2PerKWh: state.customGridIntensity ?? base.gridIntensityGCO2PerKWh,
    source: undefined,
  };
}

export function hostComponents(state: AppState): HostComponents {
  return {
    cpuIdleW: getCpu(state.cpuId).idleW,
    sockets: state.cpuSockets,
    boardW: getBoard(state.boardId).watts,
    drivesW: getDrive(state.driveId).idleW * state.driveCount,
  };
}

export function hostSpec(state: AppState): HostSpec {
  const components = hostComponents(state);
  return {
    ram: {
      typeId: state.ramTypeId,
      speedMTps: state.ramSpeedMTps,
      channels: state.ramChannels,
      totalCapacityGb: state.ramCapacityGb,
      modules: state.ramModules,
    },
    // Derived from the itemised components unless explicitly overridden.
    baseOverheadW: state.hostBaseOverheadOverrideW ?? hostBaseOverheadW(components),
    cooling: getCooling(state.coolingId),
    components,
  };
}

export function buildCalcInput(state: AppState): CalcInput {
  const model = selectedModel(state);
  const gpu = getGpu(state.gpuId) ?? GPUS[0];
  const region = effectiveRegion(state);
  const host = hostSpec(state);

  return {
    model,
    hardware: {
      gpu,
      numGpus: state.numGpus,
      parallelism: state.parallelism,
      nvlink: state.nvlinkOverride ?? undefined,
    },
    quant: getQuant(state.quantId),
    kvQuant: getKvQuant(state.kvQuantId),
    workload: {
      contextLength: state.contextLength,
      promptTokens: state.promptTokens,
      outputTokens: state.outputTokens,
      batchSize: state.batchSize,
      runtime: state.runtime,
      prefillChunkTokens: state.prefillChunkTokens,
      allowOffload: state.allowOffload,
      // Derived from the RAM specification rather than guessed at a slider.
      hostRamBandwidthGBs: ramBandwidthGBs(host.ram),
      mbu: state.mbu,
      mfu: state.mfu,
    },
    energy: {
      mode: state.energyMode,
      epsFlopPJ: state.epsFlopPJ ?? undefined,
      epsMopPJ: state.epsMopPJ ?? undefined,
      psuEfficiency: state.psuEfficiency,
      pue: state.pue,
      hostOverheadW: state.hostOverheadW,
    },
    cost: {
      region,
      inputRatio: state.inputRatio,
      hardwareCapex: state.hardwareCapex ?? undefined,
      dailyTokens: state.dailyTokens,
      usdToCurrency: region.currency === 'USD' ? 1 : state.usdToCurrency,
    },
    apiPrice: selectedApiPrice(state),
    host,
  };
}

// ---------------------------------------------------------------------------
// URL sharing
// ---------------------------------------------------------------------------

/** Keys worth round-tripping through a shareable link. */
const SHARE_KEYS = [
  'modelId', 'gpuId', 'numGpus', 'parallelism', 'quantId', 'kvQuantId',
  'contextLength', 'promptTokens', 'outputTokens', 'batchSize', 'runtime',
  'allowOffload', 'mbu', 'mfu', 'energyMode', 'psuEfficiency', 'pue',
  'hostPresetId', 'ramTypeId', 'ramSpeedMTps', 'ramChannels', 'ramCapacityGb',
  'ramModules', 'cpuId', 'cpuSockets', 'boardId', 'coolingId', 'driveId',
  'driveCount', 'hostBaseOverheadOverrideW',
  'hostOverheadW', 'regionId', 'customPricePerKWh', 'customGridIntensity',
  'inputRatio', 'hardwareCapex', 'dailyTokens', 'apiPriceId',
  'customApiInputPerMTokUsd', 'customApiOutputPerMTokUsd', 'advanced',
] as const satisfies readonly (keyof AppState)[];

export function stateToQuery(state: AppState): string {
  const params = new URLSearchParams();
  for (const key of SHARE_KEYS) {
    const value = state[key];
    if (value === null || value === undefined) continue;
    if (value === DEFAULTS[key]) continue; // keep links short
    params.set(key, String(value));
  }
  return params.toString();
}

/**
 * Keys whose default is `null` — their type cannot be inferred from the
 * default, so they are declared explicitly rather than parsed as strings.
 */
const NULLABLE_NUMBER_KEYS = new Set<keyof AppState>([
  'hardwareCapex',
  'customPricePerKWh',
  'customGridIntensity',
  'hostBaseOverheadOverrideW',
  'epsFlopPJ',
  'epsMopPJ',
]);

export function queryToState(query: string): Partial<AppState> {
  const params = new URLSearchParams(query);
  const out: Record<string, unknown> = {};

  for (const key of SHARE_KEYS) {
    const raw = params.get(key);
    if (raw === null) continue;

    const fallback = DEFAULTS[key];
    if (typeof fallback === 'number' || NULLABLE_NUMBER_KEYS.has(key)) {
      const n = Number(raw);
      if (Number.isFinite(n)) out[key] = n;
    } else if (typeof fallback === 'boolean') {
      out[key] = raw === 'true';
    } else {
      out[key] = raw;
    }
  }

  return out as Partial<AppState>;
}

export { DEFAULTS };
