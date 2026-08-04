/**
 * Core domain types for the LLM VRAM / speed / energy calculator.
 *
 * Everything in `src/core` is pure TypeScript with no UI dependencies, so the
 * whole model can be unit-tested and cited independently of the React app.
 */

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/**
 * Attention structure. This drives the KV-cache formula and must be derived
 * from the model config rather than assumed — see `services/hfConfig.ts`.
 *
 * - `mha`  full multi-head attention (n_kv_heads == n_heads)
 * - `gqa`  grouped-query attention  (1 < n_kv_heads < n_heads)
 * - `mqa`  multi-query attention    (n_kv_heads == 1)
 * - `mla`  multi-head latent attention (DeepSeek V2/V3/R1) — caches a single
 *          compressed latent instead of per-head K/V, roughly 57x smaller
 */
export type AttentionType = 'mha' | 'gqa' | 'mqa' | 'mla';

export interface MoeSpec {
  /** Total routed experts (config: num_experts | num_local_experts | n_routed_experts). */
  numExperts: number;
  /** Top-k routed experts per token (config: num_experts_per_tok). */
  expertsPerToken: number;
  /** Always-on shared experts, if any (config: n_shared_experts). */
  numSharedExperts?: number;
  /** Leading layers that stay dense (config: first_k_dense_replace). */
  firstKDenseReplace?: number;
}

export interface ModelSpec {
  id: string;
  name: string;
  family?: string;

  /**
   * Absolute parameter count, e.g. 8.03e9. For MoE this is the count that must
   * be resident in memory — every expert has to be loaded even if unused.
   */
  paramsTotal: number;
  /**
   * Parameters actually touched per token. Defaults to `paramsTotal` for dense
   * models. For MoE this is what drives bandwidth and FLOPs, not memory.
   */
  paramsActive?: number;
  /**
   * True when `paramsActive` was derived from the architecture rather than
   * published or encoded in the model name. Such estimates run high whenever
   * the safetensors total includes auxiliary modules that stay idle at decode
   * time — a multi-token-prediction head, most commonly.
   */
  paramsActiveEstimated?: boolean;

  numLayers: number;
  hiddenSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  vocabSize: number;
  maxContext: number;

  attention: AttentionType;

  /** MLA only — the compressed KV latent dimension (config: kv_lora_rank). */
  kvLoraRank?: number;
  /** MLA only — decoupled RoPE key dimension (config: qk_rope_head_dim). */
  qkRopeHeadDim?: number;

  moe?: MoeSpec;

  /**
   * Fraction of layers that maintain a KV cache growing with sequence length,
   * 0..1. Defaults to 1.
   *
   * Hybrid models interleave linear-attention (Gated DeltaNet, KDA, Mamba)
   * layers whose recurrent state is O(1) in sequence length with a minority of
   * real softmax-attention layers. Qwen3.5/3.6 keep one full-attention layer in
   * every four; Kimi Linear keeps 7 of 27. Treating those as ordinary attention
   * overstates long-context memory several-fold.
   */
  kvCacheLayerRatio?: number;

  /** Sliding-window attention span; null/undefined means global attention. */
  slidingWindow?: number | null;
  /**
   * Share of layers that actually use the sliding window, 0..1. Modern models
   * interleave: Gemma 2 alternates every other layer (0.5), Gemma 3 uses a
   * 6-layer pattern with one global layer (5/6), gpt-oss alternates (0.5).
   * The remaining layers keep a full-context cache, so this materially
   * changes long-context memory. Defaults to 1 when a window is set.
   */
  slidingWindowLayerRatio?: number;
  tieWordEmbeddings?: boolean;

  source?: 'catalog' | 'custom' | 'hf';
  hfRepo?: string;
  /** Free-text provenance note rendered in the UI. */
  note?: string;
}

// ---------------------------------------------------------------------------
// Hardware
// ---------------------------------------------------------------------------

export type Vendor = 'nvidia' | 'amd' | 'apple' | 'intel';

export interface GpuSpec {
  id: string;
  name: string;
  vendor: Vendor;
  /** For Apple Silicon this is total unified memory, not a dedicated pool. */
  vramGb: number;
  bandwidthGBs: number;
  /**
   * Dense FP16 tensor throughput in TFLOPS — NOT the 2:4-sparse marketing
   * number, and for GeForce cards the FP32-accumulate (half-rate) figure.
   */
  fp16TFlops: number;
  fp8TFlops?: number;
  fp4TFlops?: number;
  tdpW: number;
  idleW: number;
  priceUsd?: number;
  /** Unified memory (Apple) — usable fraction is lower than total. */
  unified?: boolean;
  architecture?: string;
  /** Whether NVLink is available for multi-GPU tensor parallelism. */
  nvlink?: boolean;
  note?: string;
}

export type Parallelism = 'tp' | 'pp';

export interface HardwareConfig {
  gpu: GpuSpec;
  numGpus: number;
  parallelism: Parallelism;
  /** Overrides `gpu.nvlink` when the user knows their rig. */
  nvlink?: boolean;
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

export type QuantGroup = 'float' | 'gguf' | 'gptq-awq' | 'mx';

export interface QuantSpec {
  id: string;
  label: string;
  /**
   * Effective bits per weight, averaged over the whole file. GGUF `_M`/`_S`
   * variants mix precisions (embeddings and some projections are kept higher),
   * so e.g. Q4_K_M is 4.89 bpw, not 4.0.
   */
  bpw: number;
  group: QuantGroup;
  /** Compute precision used for the roofline, if the format is compute-native. */
  computePrecision?: 'fp16' | 'fp8' | 'fp4';
  note?: string;
}

export interface KvQuantSpec {
  id: string;
  label: string;
  bytesPerElement: number;
  note?: string;
}

// ---------------------------------------------------------------------------
// Workload
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Host system
// ---------------------------------------------------------------------------

export interface RamTypeSpec {
  id: string;
  label: string;
  /** Selectable transfer rates in MT/s. */
  speeds: number[];
  defaultSpeed: number;
  /** Per-module draw while the bus is idle — paid whenever the machine is on. */
  idleWattsPerModule: number;
  /** Clock the measured active figure was taken at. */
  referenceSpeedMTps: number;
  /** Per-module draw at `referenceSpeedMTps` with the bus saturated. */
  referenceActiveWatts: number;
}

export interface RamSpec {
  typeId: string;
  speedMTps: number;
  channels: number;
  /**
   * Installed capacity. Drives both the offload feasibility check and, via
   * `modules`, the idle power the machine pays continuously.
   */
  totalCapacityGb: number;
  /** Number of physical modules — what idle power actually scales with. */
  modules: number;
}

export interface CpuSpec {
  id: string;
  label: string;
  vendor: 'intel' | 'amd' | 'apple' | 'other';
  segment: 'mobile' | 'desktop' | 'workstation' | 'server';
  cores: number;
  tdpW: number;
  /** Package power at idle. Estimated — vendors do not publish this. */
  idleW: number;
}

export interface CoolingSpec {
  id: string;
  labelKey: string;
  watts: number;
}

export interface DriveSpec {
  id: string;
  labelKey: string;
  idleW: number;
}

/**
 * The non-RAM, non-GPU parts of the host, itemised.
 *
 * A single overhead figure cannot span this range: a laptop and a dual-socket
 * 1U server differ by an order of magnitude, and most of the gap is CPU idle
 * and chassis airflow rather than anything the model does.
 */
export interface HostComponents {
  cpuIdleW: number;
  sockets: number;
  boardW: number;
  coolingW: number;
  drivesW: number;
}

export interface HostSpec {
  ram: RamSpec;
  /**
   * Everything else in the host: CPU at idle, board and VRM losses, drives,
   * fans. RAM is excluded because it is costed separately.
   */
  baseOverheadW: number;
  /** Itemisation behind `baseOverheadW`, for display. */
  components?: HostComponents;
}

export type Runtime = 'vllm' | 'llamacpp' | 'transformers';

export interface Workload {
  /** Sequence length the KV cache must hold (prompt + generated). */
  contextLength: number;
  /** Prompt length used for the TTFT / prefill calculation. */
  promptTokens: number;
  /** Tokens generated per request, used for the per-request energy split. */
  outputTokens: number;
  /** Sequences processed together in one forward pass. */
  batchSize: number;
  runtime: Runtime;
  /** Prefill chunk size — vLLM chunked prefill defaults to 2048. */
  prefillChunkTokens: number;
  allowOffload: boolean;
  /** Host RAM bandwidth for the offloaded fraction (DDR5 dual channel ~90). */
  hostRamBandwidthGBs: number;
  /** Memory bandwidth utilization achieved in practice, 0..1. */
  mbu: number;
  /** Model FLOPs utilization achieved during prefill, 0..1. */
  mfu: number;
}

// ---------------------------------------------------------------------------
// Energy
// ---------------------------------------------------------------------------

export type EnergyMode = 'simple' | 'roofline';

export interface EnergyOptions {
  mode: EnergyMode;
  /**
   * Dynamic energy per FLOP, in picojoules. Roofline mode only.
   * `undefined` derives a default from the GPU's TDP and peak FLOPS.
   */
  epsFlopPJ?: number;
  /** Dynamic energy per byte moved from memory, in picojoules. */
  epsMopPJ?: number;
  /** Static idle draw pi_0 in watts; defaults to `gpu.idleW`. */
  idleW?: number;

  /** PSU efficiency, 0..1 (80+ Gold ~0.90, Titanium ~0.94). */
  psuEfficiency: number;
  /** Power usage effectiveness. 1.0 for a home PC, ~1.54 datacenter average. */
  pue: number;
  /** Non-GPU system draw in watts (CPU, RAM, fans). */
  hostOverheadW: number;
}

// ---------------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------------

export interface Region {
  id: string;
  label: string;
  /** Electricity price in the region's currency per kWh. */
  pricePerKWh: number;
  currency: 'EUR' | 'USD';
  /** Grid carbon intensity in grams CO2-equivalent per kWh. */
  gridIntensityGCO2PerKWh: number;
  source?: string;
}

export interface ApiPrice {
  id: string;
  label: string;
  provider?: string;
  /** USD per 1M input tokens. */
  inputPerMTokUsd: number;
  /** USD per 1M output tokens. */
  outputPerMTokUsd: number;
  contextLength?: number;
  /** ISO date of the price snapshot, so the UI can show staleness. */
  asOf?: string;
}

export interface CostOptions {
  region: Region;
  /** Share of billed tokens that are input, 0..1. Materially changes the answer. */
  inputRatio: number;
  /** Hardware purchase price for break-even analysis, in region currency. */
  hardwareCapex?: number;
  /** Expected tokens generated per day, for the break-even horizon. */
  dailyTokens?: number;
  /** USD -> region currency rate. */
  usdToCurrency: number;
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

export interface MemoryBreakdown {
  weightsBytes: number;
  kvCacheBytes: number;
  activationsBytes: number;
  cudaContextBytes: number;
  frameworkOverheadBytes: number;
  totalBytes: number;
}

export type Bound = 'memory' | 'compute';

export interface PerformanceResult {
  /** Tokens per second for a single sequence (what a single user perceives). */
  decodeTokensPerSecPerSequence: number;
  /** Aggregate tokens per second across the whole batch. */
  decodeTokensPerSecTotal: number;
  msPerToken: number;
  ttftMs: number;
  /** Bytes that must cross the memory bus per decode step (whole batch). */
  bytesPerDecodeStep: number;
  /** FLOPs per decode step (whole batch). */
  flopsPerDecodeStep: number;
  prefillFlops: number;
  /** Arithmetic intensity of the decode phase, FLOP/byte. */
  decodeIntensity: number;
  /** Arithmetic intensity of the prefill phase, FLOP/byte. */
  prefillIntensity: number;
  /** Ridge point F/B — the intensity above which the kernel is compute-bound. */
  ridgePoint: number;
  decodeBound: Bound;
  prefillBound: Bound;
  /** Effective aggregate bandwidth after multi-GPU and offload penalties. */
  effectiveBandwidthBytesPerSec: number;
  effectiveFlops: number;
  /** Fraction of weights served from host RAM rather than VRAM, 0..1. */
  offloadFraction: number;
  /**
   * Share of wall-clock time the host memory bus is saturated. Drives RAM
   * power; zero when nothing is offloaded.
   */
  ramDutyCycle: number;
  /** True when the weights that must spill exceed available host memory. */
  offloadExceedsHostRam: boolean;
}

export interface EnergyResult {
  /** Average GPU-only power during decode. */
  decodePowerW: number;
  prefillPowerW: number;
  /** GPU energy per generated token. */
  joulesPerToken: number;
  /** Wall-plug energy per token, including PSU, host overhead and PUE. */
  wallJoulesPerToken: number;
  kWhPerMTokens: number;
  /** Sustained wall-plug draw while generating. */
  wallPowerW: number;
  /** Roofline decomposition — populated in roofline mode. */
  decomposition?: {
    computeJoules: number;
    memoryJoules: number;
    staticJoules: number;
  };
  epsFlopPJ: number;
  epsMopPJ: number;
  /**
   * Host draw split into its parts, so the standing cost of installed memory
   * is visible rather than buried in a single overhead figure.
   */
  host: {
    /** CPU, board, drives, fans. */
    baseW: number;
    /** Paid for every installed module whether or not it is used. */
    ramIdleW: number;
    /** Extra draw while weights stream from host memory. */
    ramActiveW: number;
    totalW: number;
  };
}

export interface CostResult {
  /** Electricity cost of generating 1M tokens locally. */
  localElectricityPerMTokens: number;
  /** Electricity plus amortized hardware, if capex was supplied. */
  localTotalPerMTokens?: number;
  apiPerMTokens?: number;
  /** Positive means running locally is cheaper. */
  savingsPerMTokens?: number;
  /** Tokens that must be generated before hardware pays for itself. */
  breakEvenTokens?: number;
  breakEvenDays?: number;
  co2GramsPerMTokens: number;
  currency: 'EUR' | 'USD';
}

export interface CalcInput {
  model: ModelSpec;
  hardware: HardwareConfig;
  quant: QuantSpec;
  kvQuant: KvQuantSpec;
  workload: Workload;
  energy: EnergyOptions;
  cost: CostOptions;
  apiPrice?: ApiPrice;
  /**
   * Optional: when supplied, host RAM bandwidth, offload feasibility and
   * memory power are derived from the actual specification rather than from
   * `workload.hostRamBandwidthGBs` and `energy.hostOverheadW`.
   */
  host?: HostSpec;
}

export interface CalcResult {
  memory: MemoryBreakdown;
  totalVramBytes: number;
  /** Memory actually addressable — Apple unified memory reserves some for the OS. */
  usableVramBytes: number;
  fits: boolean;
  utilizationPct: number;
  performance: PerformanceResult;
  energy: EnergyResult;
  cost: CostResult;
  warnings: Warning[];
}

export interface Warning {
  level: 'info' | 'warn' | 'error';
  /** i18n key, so warnings translate. */
  key: string;
  values?: Record<string, string | number>;
}
