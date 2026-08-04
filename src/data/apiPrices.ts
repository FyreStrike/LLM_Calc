import type { ApiPrice } from '../core/types';

/**
 * Bundled API price snapshot — the offline fallback.
 *
 * Fetched live from OpenRouter on the date below. The app refreshes these at
 * runtime from https://openrouter.ai/api/v1/models (public, no auth, CORS
 * open); this list only exists so the tool still works with no network.
 *
 * Prices are USD per 1 million tokens. Note that OpenRouter's API returns
 * price per *single* token as a decimal string, so the service layer
 * multiplies by 1e6 — forgetting that is a six-orders-of-magnitude error.
 */
export const API_PRICE_SNAPSHOT_DATE = '2026-08-04';

export const API_PRICES: ApiPrice[] = [
  // --- open weight, i.e. models you could actually self-host ---------------
  {
    id: 'deepseek/deepseek-v4-flash-0731',
    label: 'DeepSeek V4 Flash',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.09,
    outputPerMTokUsd: 0.18,
    contextLength: 1048576,
  },
  {
    id: 'deepseek/deepseek-v3.2',
    label: 'DeepSeek V3.2',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.269,
    outputPerMTokUsd: 0.4,
    contextLength: 163840,
  },
  {
    id: 'deepseek/deepseek-r1',
    label: 'DeepSeek R1',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.7,
    outputPerMTokUsd: 2.5,
    contextLength: 163840,
  },
  {
    id: 'z-ai/glm-5.2',
    label: 'GLM 5.2',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.76,
    outputPerMTokUsd: 2.42,
    contextLength: 1048576,
  },
  {
    id: 'z-ai/glm-4.6',
    label: 'GLM 4.6',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.5,
    outputPerMTokUsd: 2.0,
    contextLength: 204800,
  },
  {
    id: 'z-ai/glm-4.5-air',
    label: 'GLM 4.5 Air',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.13,
    outputPerMTokUsd: 0.85,
    contextLength: 131072,
  },
  {
    id: 'qwen/qwen3-235b-a22b',
    label: 'Qwen3 235B A22B',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.455,
    outputPerMTokUsd: 1.82,
    contextLength: 131072,
  },
  {
    id: 'qwen/qwen3-30b-a3b',
    label: 'Qwen3 30B A3B',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.12,
    outputPerMTokUsd: 0.5,
    contextLength: 131072,
  },
  {
    id: 'moonshotai/kimi-k2',
    label: 'Kimi K2',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.57,
    outputPerMTokUsd: 2.3,
    contextLength: 131072,
  },
  {
    id: 'openai/gpt-oss-120b',
    label: 'gpt-oss 120B',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.037,
    outputPerMTokUsd: 0.17,
    contextLength: 131072,
  },
  {
    id: 'openai/gpt-oss-20b',
    label: 'gpt-oss 20B',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.03,
    outputPerMTokUsd: 0.13,
    contextLength: 131072,
  },
  {
    id: 'meta-llama/llama-3.3-70b-instruct',
    label: 'Llama 3.3 70B',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.1,
    outputPerMTokUsd: 0.32,
    contextLength: 131072,
  },
  {
    id: 'meta-llama/llama-3.1-8b-instruct',
    label: 'Llama 3.1 8B',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.05,
    outputPerMTokUsd: 0.08,
    contextLength: 131072,
  },
  {
    id: 'mistralai/mistral-small-24b-instruct-2501',
    label: 'Mistral Small 24B',
    provider: 'OpenRouter',
    inputPerMTokUsd: 0.05,
    outputPerMTokUsd: 0.08,
    contextLength: 32768,
  },

  // --- closed frontier models, for reference only -------------------------
  {
    id: 'anthropic/claude-haiku-4.5',
    label: 'Claude Haiku 4.5',
    provider: 'OpenRouter',
    inputPerMTokUsd: 1,
    outputPerMTokUsd: 5,
    contextLength: 200000,
  },
  {
    id: 'anthropic/claude-sonnet-5',
    label: 'Claude Sonnet 5',
    provider: 'OpenRouter',
    inputPerMTokUsd: 2,
    outputPerMTokUsd: 10,
    contextLength: 1000000,
  },
  {
    id: 'anthropic/claude-opus-5',
    label: 'Claude Opus 5',
    provider: 'OpenRouter',
    inputPerMTokUsd: 5,
    outputPerMTokUsd: 25,
    contextLength: 1000000,
  },
  {
    id: 'openai/gpt-5.2',
    label: 'GPT-5.2',
    provider: 'OpenRouter',
    inputPerMTokUsd: 1.75,
    outputPerMTokUsd: 14,
    contextLength: 400000,
  },
  {
    id: 'google/gemini-3.1-pro-preview-customtools',
    label: 'Gemini 3.1 Pro',
    provider: 'OpenRouter',
    inputPerMTokUsd: 2,
    outputPerMTokUsd: 12,
    contextLength: 1048576,
  },
].map((p) => ({ ...p, asOf: API_PRICE_SNAPSHOT_DATE }));

/**
 * Maps a catalog model to the API listing that serves the same weights, so the
 * comparison defaults to something meaningful rather than an arbitrary model.
 */
export const MODEL_TO_API_PRICE: Record<string, string> = {
  'deepseek-v4-flash': 'deepseek/deepseek-v4-flash-0731',
  'deepseek-v4-flash-base': 'deepseek/deepseek-v4-flash-0731',
  'deepseek-v3': 'deepseek/deepseek-v3.2',
  'deepseek-v3.1': 'deepseek/deepseek-v3.2',
  'deepseek-v3.2': 'deepseek/deepseek-v3.2',
  'deepseek-r1': 'deepseek/deepseek-r1',
  'glm-5.2': 'z-ai/glm-5.2',
  'glm-4.6': 'z-ai/glm-4.6',
  'glm-4.5': 'z-ai/glm-4.5-air',
  'glm-4.5-air': 'z-ai/glm-4.5-air',
  'kimi-k2': 'moonshotai/kimi-k2',
  'kimi-k2-thinking': 'moonshotai/kimi-k2',
  'qwen3-235b-a22b': 'qwen/qwen3-235b-a22b',
  'qwen3-30b-a3b': 'qwen/qwen3-30b-a3b',
  'qwen3-coder-30b-a3b': 'qwen/qwen3-30b-a3b',
  'gpt-oss-120b': 'openai/gpt-oss-120b',
  'gpt-oss-20b': 'openai/gpt-oss-20b',
  'llama-3.3-70b': 'meta-llama/llama-3.3-70b-instruct',
  'llama-3.1-8b': 'meta-llama/llama-3.1-8b-instruct',
  'mistral-small-24b': 'mistralai/mistral-small-24b-instruct-2501',
  'mistral-small-3.2-24b': 'mistralai/mistral-small-24b-instruct-2501',
};
