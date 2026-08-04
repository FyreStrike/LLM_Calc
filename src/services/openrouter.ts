import { API_PRICES, API_PRICE_SNAPSHOT_DATE } from '../data/apiPrices';
import type { ApiPrice } from '../core/types';

/**
 * Live API pricing from OpenRouter.
 *
 * `https://openrouter.ai/api/v1/models` is public, needs no auth, and sends
 * permissive CORS headers, so a static page can call it directly.
 *
 * The one thing to get right: `pricing.prompt` and `pricing.completion` are
 * USD per *single token*, encoded as decimal strings ("0.00000076"). Everything
 * downstream works in dollars per million tokens, so they are scaled by 1e6 on
 * the way in. Skipping that is a factor-of-a-million error that looks
 * superficially plausible in a UI.
 */

const ENDPOINT = 'https://openrouter.ai/api/v1/models';
const CACHE_KEY = 'llmcalc.openrouter.v1';
const CACHE_TTL_MS = 6 * 60 * 60 * 1000; // 6 hours

interface OpenRouterModel {
  id: string;
  name: string;
  context_length?: number;
  pricing?: {
    prompt?: string;
    completion?: string;
  };
}

export interface PriceFetchResult {
  prices: ApiPrice[];
  /** Where the data came from, so the UI can be honest about staleness. */
  origin: 'live' | 'cache' | 'snapshot';
  asOf: string;
  error?: string;
}

function parsePrices(models: OpenRouterModel[]): ApiPrice[] {
  const asOf = new Date().toISOString().slice(0, 10);

  return models
    .map((m): ApiPrice | null => {
      const prompt = Number(m.pricing?.prompt);
      const completion = Number(m.pricing?.completion);
      if (!Number.isFinite(prompt) || !Number.isFinite(completion)) return null;
      // Skip free and unpriced listings — they distort the comparison.
      if (prompt <= 0 && completion <= 0) return null;

      return {
        id: m.id,
        label: m.name ?? m.id,
        provider: 'OpenRouter',
        inputPerMTokUsd: prompt * 1e6,
        outputPerMTokUsd: completion * 1e6,
        contextLength: m.context_length,
        asOf,
      };
    })
    .filter((p): p is ApiPrice => p !== null)
    .sort((a, b) => a.label.localeCompare(b.label));
}

function readCache(): PriceFetchResult | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { at: number; prices: ApiPrice[] };
    if (Date.now() - parsed.at > CACHE_TTL_MS) return null;
    if (!Array.isArray(parsed.prices) || parsed.prices.length === 0) return null;
    return {
      prices: parsed.prices,
      origin: 'cache',
      asOf: new Date(parsed.at).toISOString().slice(0, 10),
    };
  } catch {
    return null;
  }
}

function writeCache(prices: ApiPrice[]): void {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify({ at: Date.now(), prices }));
  } catch {
    // Storage may be full or blocked; caching is best-effort.
  }
}

const snapshot = (error?: string): PriceFetchResult => ({
  prices: API_PRICES,
  origin: 'snapshot',
  asOf: API_PRICE_SNAPSHOT_DATE,
  error,
});

/**
 * Resolve API prices, preferring fresh data but always returning something
 * usable. The bundled snapshot means the calculator still works offline.
 */
export async function fetchApiPrices(force = false): Promise<PriceFetchResult> {
  if (!force) {
    const cached = readCache();
    if (cached) return cached;
  }

  try {
    const res = await fetch(ENDPOINT);
    if (!res.ok) return snapshot(`HTTP ${res.status}`);

    const json = (await res.json()) as { data?: OpenRouterModel[] };
    const prices = parsePrices(json.data ?? []);
    if (prices.length === 0) return snapshot('No usable prices returned');

    writeCache(prices);
    return { prices, origin: 'live', asOf: new Date().toISOString().slice(0, 10) };
  } catch (e) {
    return snapshot(e instanceof Error ? e.message : 'Request failed');
  }
}

export { parsePrices as __parsePricesForTest };
