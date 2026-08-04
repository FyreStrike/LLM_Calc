import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  ApiPrice,
  CalcInput,
  EnergyMode,
  ModelSpec,
  Parallelism,
  Runtime,
} from '../core/types';
import { getKvQuant, getQuant } from '../core/quant';
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
  hostRamBandwidthGBs: number;
  mbu: number;
  mfu: number;

  // energy
  energyMode: EnergyMode;
  epsFlopPJ: number | null;
  epsMopPJ: number | null;
  psuEfficiency: number;
  pue: number;
  hostOverheadW: number;

  // cost
  regionId: string;
  inputRatio: number;
  hardwareCapex: number | null;
  dailyTokens: number;
  usdToCurrency: number;
  apiPriceId: string | null;
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
  hostRamBandwidthGBs: 90,
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
  inputRatio: 0.7,
  hardwareCapex: null,
  dailyTokens: 1_000_000,
  usdToCurrency: DEFAULT_USD_TO_EUR,
  apiPriceId: null,
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
  const prices = availablePrices(state);
  if (state.apiPriceId) {
    const explicit = prices.find((p) => p.id === state.apiPriceId);
    if (explicit) return explicit;
  }
  const mapped = MODEL_TO_API_PRICE[state.modelId];
  return mapped ? prices.find((p) => p.id === mapped) : undefined;
}

export function buildCalcInput(state: AppState): CalcInput {
  const model = selectedModel(state);
  const gpu = getGpu(state.gpuId) ?? GPUS[0];
  const region = getRegion(state.regionId) ?? REGIONS[0];

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
      hostRamBandwidthGBs: state.hostRamBandwidthGBs,
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
  'hostOverheadW', 'regionId', 'inputRatio', 'hardwareCapex', 'dailyTokens',
  'apiPriceId', 'advanced',
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

export function queryToState(query: string): Partial<AppState> {
  const params = new URLSearchParams(query);
  const out: Record<string, unknown> = {};

  for (const key of SHARE_KEYS) {
    const raw = params.get(key);
    if (raw === null) continue;

    const fallback = DEFAULTS[key];
    if (typeof fallback === 'number') {
      const n = Number(raw);
      if (Number.isFinite(n)) out[key] = n;
    } else if (typeof fallback === 'boolean') {
      out[key] = raw === 'true';
    } else {
      out[key] = raw;
    }
  }

  // hardwareCapex defaults to null, so it needs handling outside the switch.
  const capex = params.get('hardwareCapex');
  if (capex !== null && Number.isFinite(Number(capex))) {
    out.hardwareCapex = Number(capex);
  }

  return out as Partial<AppState>;
}

export { DEFAULTS };
