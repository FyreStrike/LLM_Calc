import type { KvQuantSpec, QuantSpec } from './types';

/**
 * Effective bits-per-weight tables.
 *
 * The GGUF numbers are *file-level averages* measured by llama.cpp on
 * Llama-3.1-8B, not the pure per-tensor block sizes. The `_S`/`_M`/`_L`
 * variants deliberately mix precisions — attention.wv, feed_forward.w2 and the
 * embedding/output tensors get promoted to Q6_K or Q8_0 — so Q4_K_M lands at
 * 4.89 bpw rather than the 4.5 bpw its block structure implies.
 *
 * Source: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
 *
 * Caveat worth surfacing in the UI: because bpw is averaged over *all*
 * parameters including embeddings, small models with large vocabularies (a
 * 0.6B model with a 151k vocab, say) deviate from this reference by more than
 * the usual few percent.
 */
export const QUANTIZATIONS: QuantSpec[] = [
  // --- native float / integer formats -------------------------------------
  { id: 'fp32', label: 'FP32', bpw: 32, group: 'float', computePrecision: 'fp16' },
  { id: 'fp16', label: 'FP16 / BF16', bpw: 16, group: 'float', computePrecision: 'fp16' },
  {
    id: 'fp8',
    label: 'FP8 (E4M3)',
    bpw: 8,
    group: 'float',
    computePrecision: 'fp8',
    note: 'quant.note.fp8',
  },
  { id: 'int8', label: 'INT8', bpw: 8.06, group: 'float', computePrecision: 'fp8' },
  {
    id: 'fp4',
    label: 'FP4 / NVFP4',
    bpw: 4.5,
    group: 'float',
    computePrecision: 'fp4',
    note: 'quant.note.fp4',
  },

  // --- GPTQ / AWQ ----------------------------------------------------------
  {
    id: 'gptq-awq-4bit-g128',
    label: 'GPTQ / AWQ 4-bit (g128)',
    bpw: 4.25,
    group: 'gptq-awq',
    note: 'quant.note.groupwise',
  },
  {
    id: 'gptq-awq-4bit-g32',
    label: 'GPTQ / AWQ 4-bit (g32)',
    bpw: 4.6,
    group: 'gptq-awq',
    note: 'quant.note.groupwise',
  },

  // --- MX ------------------------------------------------------------------
  {
    id: 'mxfp4',
    label: 'MXFP4',
    bpw: 4.25,
    group: 'mx',
    computePrecision: 'fp4',
    note: 'quant.note.mxfp4',
  },

  // --- GGUF (llama.cpp measured file-level averages) -----------------------
  { id: 'q8_0', label: 'Q8_0', bpw: 8.5008, group: 'gguf' },
  { id: 'q6_k', label: 'Q6_K', bpw: 6.5633, group: 'gguf' },
  { id: 'q5_k_m', label: 'Q5_K_M', bpw: 5.7036, group: 'gguf' },
  { id: 'q5_k_s', label: 'Q5_K_S', bpw: 5.5704, group: 'gguf' },
  { id: 'q4_k_m', label: 'Q4_K_M', bpw: 4.8944, group: 'gguf' },
  { id: 'q4_k_s', label: 'Q4_K_S', bpw: 4.6672, group: 'gguf' },
  { id: 'iq4_nl', label: 'IQ4_NL', bpw: 4.6818, group: 'gguf' },
  { id: 'iq4_xs', label: 'IQ4_XS', bpw: 4.4597, group: 'gguf' },
  // Q4_0 is absent from llama.cpp's table (deprecated / repacked); derived
  // from its block struct: 2-byte fp16 scale + 16 bytes of nibbles per 32
  // weights = 18 * 8 / 32 = 4.5 bpw exactly.
  { id: 'q4_0', label: 'Q4_0', bpw: 4.5, group: 'gguf', note: 'quant.note.q4_0' },
  { id: 'q3_k_l', label: 'Q3_K_L', bpw: 4.2979, group: 'gguf' },
  { id: 'q3_k_m', label: 'Q3_K_M', bpw: 3.996, group: 'gguf' },
  { id: 'q3_k_s', label: 'Q3_K_S', bpw: 3.6429, group: 'gguf' },
  { id: 'iq3_m', label: 'IQ3_M', bpw: 3.7628, group: 'gguf' },
  { id: 'iq3_s', label: 'IQ3_S', bpw: 3.6606, group: 'gguf' },
  { id: 'iq3_xxs', label: 'IQ3_XXS', bpw: 3.2548, group: 'gguf' },
  { id: 'q2_k', label: 'Q2_K', bpw: 3.1593, group: 'gguf' },
  { id: 'q2_k_s', label: 'Q2_K_S', bpw: 2.9697, group: 'gguf' },
  { id: 'iq2_m', label: 'IQ2_M', bpw: 2.9294, group: 'gguf' },
  { id: 'iq2_s', label: 'IQ2_S', bpw: 2.7403, group: 'gguf' },
  { id: 'iq2_xs', label: 'IQ2_XS', bpw: 2.5882, group: 'gguf' },
  { id: 'iq2_xxs', label: 'IQ2_XXS', bpw: 2.3824, group: 'gguf' },
  { id: 'iq1_m', label: 'IQ1_M', bpw: 2.146, group: 'gguf' },
  { id: 'iq1_s', label: 'IQ1_S', bpw: 2.0042, group: 'gguf' },
];

/**
 * KV cache precisions.
 *
 * llama.cpp's `-ctk`/`-ctv q8_0` carries per-block scales, so it is really
 * 8.5 bits per element rather than 8; vLLM's FP8 KV cache uses a per-tensor
 * scale and is a clean byte.
 */
export const KV_QUANTIZATIONS: KvQuantSpec[] = [
  { id: 'fp32', label: 'FP32', bytesPerElement: 4 },
  { id: 'fp16', label: 'FP16 / BF16', bytesPerElement: 2 },
  { id: 'fp8', label: 'FP8', bytesPerElement: 1 },
  { id: 'q8_0', label: 'Q8_0 (GGUF)', bytesPerElement: 8.5 / 8, note: 'quant.note.kvBlockScale' },
  { id: 'q5_1', label: 'Q5_1 (GGUF)', bytesPerElement: 6 / 8, note: 'quant.note.kvBlockScale' },
  { id: 'q4_0', label: 'Q4_0 (GGUF)', bytesPerElement: 4.5 / 8, note: 'quant.note.kvBlockScale' },
];

export function getQuant(id: string): QuantSpec {
  const q = QUANTIZATIONS.find((x) => x.id === id);
  if (!q) throw new Error(`Unknown quantization: ${id}`);
  return q;
}

export function getKvQuant(id: string): KvQuantSpec {
  const q = KV_QUANTIZATIONS.find((x) => x.id === id);
  if (!q) throw new Error(`Unknown KV quantization: ${id}`);
  return q;
}

/** Bytes of storage each parameter occupies at the given quantization. */
export function bytesPerParam(quant: QuantSpec): number {
  return quant.bpw / 8;
}
