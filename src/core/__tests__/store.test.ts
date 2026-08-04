import { describe, expect, it } from 'vitest';
import { REGIONS } from '../../data/regions';
import {
  API_PRICE_AUTO,
  API_PRICE_CUSTOM,
  API_PRICE_NONE,
  buildCalcInput,
  DEFAULTS,
  effectiveRegion,
  hostComponents,
  isSocPlatform,
  queryToState,
  selectedApiPrice,
  stateToQuery,
  type AppState,
} from '../../state/store';
import { runCalculation } from '../index';

const base = (over: Partial<AppState> = {}): AppState => ({ ...DEFAULTS, ...over });

describe('effective region', () => {
  it('uses the preset when nothing is overridden', () => {
    const r = effectiveRegion(base({ regionId: 'de-household' }));
    const preset = REGIONS.find((x) => x.id === 'de-household')!;
    expect(r.id).toBe('de-household');
    expect(r.pricePerKWh).toBeCloseTo(preset.pricePerKWh, 6);
  });

  it('substitutes a custom tariff and marks the region as custom', () => {
    const r = effectiveRegion(base({ regionId: 'de-household', customPricePerKWh: 0.15 }));
    expect(r.id).toBe('custom');
    expect(r.pricePerKWh).toBeCloseTo(0.15, 6);
    // Currency and CO2 still come from the preset it was derived from.
    expect(r.currency).toBe('EUR');
    expect(r.gridIntensityGCO2PerKWh).toBe(380);
  });

  it('substitutes a custom grid intensity independently of price', () => {
    const r = effectiveRegion(base({ regionId: 'de-household', customGridIntensity: 25 }));
    expect(r.gridIntensityGCO2PerKWh).toBe(25);
    // Price untouched.
    expect(r.pricePerKWh).toBeCloseTo(0.3869, 4);
  });

  it('drops the preset source once values are overridden', () => {
    // A hand-entered tariff must not keep claiming an Eurostat citation.
    expect(effectiveRegion(base({ customPricePerKWh: 0.2 })).source).toBeUndefined();
    expect(effectiveRegion(base({})).source).toBeDefined();
  });

  it('treats zero as a real price, not as absent', () => {
    // Someone on fully self-generated power should get 0, not the preset.
    const r = effectiveRegion(base({ customPricePerKWh: 0 }));
    expect(r.pricePerKWh).toBe(0);
  });
});

describe('SoC platforms', () => {
  it('adds no separate CPU draw on an SoC', () => {
    // Apple Silicon and DGX Spark put the CPU on the GPU's die and inside its
    // power budget; charging a discrete CPU on top double-counts it.
    const desktop = hostComponents(base({ gpuId: 'rtx-4090', cpuId: 'ryzen9-9950x' }));
    const apple = hostComponents(base({ gpuId: 'm3-ultra', cpuId: 'ryzen9-9950x' }));
    const spark = hostComponents(base({ gpuId: 'dgx-spark-gb10', cpuId: 'xeon-gold-6430' }));

    expect(desktop.cpuIdleW).toBeGreaterThan(0);
    expect(apple.cpuIdleW).toBe(0);
    expect(spark.cpuIdleW).toBe(0);
  });

  it('ignores a stale socket count on an SoC', () => {
    // Switching from a dual-socket server to a Spark must not keep 2x.
    const spark = hostComponents(base({ gpuId: 'dgx-spark-gb10', cpuSockets: 2 }));
    expect(spark.sockets).toBe(1);
    expect(spark.cpuIdleW * spark.sockets).toBe(0);
  });

  it('detects SoC platforms and only those', () => {
    expect(isSocPlatform(base({ gpuId: 'm4-max' }))).toBe(true);
    expect(isSocPlatform(base({ gpuId: 'dgx-spark-gb10' }))).toBe(true);
    expect(isSocPlatform(base({ gpuId: 'h100-sxm' }))).toBe(false);
    expect(isSocPlatform(base({ gpuId: 'mi300x' }))).toBe(false);
  });

  it('lets a DGX Spark hold a model an Apple-style cap would reject', () => {
    // ~121 GB usable rather than 96 GB: the difference decides whether a
    // 100 GB quantised model runs at all.
    const spark = runCalculation(
      buildCalcInput(
        base({
          gpuId: 'dgx-spark-gb10',
          numGpus: 1,
          modelId: 'llama-3.3-70b',
          quantId: 'q8_0',
          contextLength: 8192,
          runtime: 'llamacpp',
        }),
      ),
    );
    expect(spark.usableVramBytes / 1024 ** 3).toBeGreaterThan(120);
    expect(spark.fits).toBe(true);
  });
});

describe('API price selection', () => {
  it('auto-selects the listing that serves the same weights', () => {
    const p = selectedApiPrice(base({ modelId: 'llama-3.1-8b', apiPriceId: API_PRICE_AUTO }));
    expect(p?.id).toBe('meta-llama/llama-3.1-8b-instruct');
  });

  it('REGRESSION: "no comparison" actually removes the comparison', () => {
    // A single null used to mean both "none" and "auto", so picking "no
    // comparison" fell through to the model mapping and kept comparing.
    const model = base({ modelId: 'llama-3.1-8b', apiPriceId: API_PRICE_NONE });
    expect(selectedApiPrice(model)).toBeUndefined();

    // ...while auto on the same model still resolves.
    expect(selectedApiPrice({ ...model, apiPriceId: API_PRICE_AUTO })).toBeDefined();
  });

  it('honours an explicitly chosen listing over the model mapping', () => {
    const p = selectedApiPrice(
      base({ modelId: 'llama-3.1-8b', apiPriceId: 'anthropic/claude-opus-5' }),
    );
    expect(p?.id).toBe('anthropic/claude-opus-5');
  });

  it('returns the user’s own prices in custom mode', () => {
    const p = selectedApiPrice(
      base({
        apiPriceId: API_PRICE_CUSTOM,
        customApiInputPerMTokUsd: 0.42,
        customApiOutputPerMTokUsd: 1.7,
      }),
    );
    expect(p?.id).toBe('custom');
    expect(p?.inputPerMTokUsd).toBe(0.42);
    expect(p?.outputPerMTokUsd).toBe(1.7);
  });

  it('treats a cleared custom field as zero rather than falling back', () => {
    const p = selectedApiPrice(
      base({
        apiPriceId: API_PRICE_CUSTOM,
        customApiInputPerMTokUsd: null,
        customApiOutputPerMTokUsd: null,
      }),
    );
    expect(p?.inputPerMTokUsd).toBe(0);
    expect(p?.outputPerMTokUsd).toBe(0);
  });

  it('feeds custom prices through the blended comparison', () => {
    const r = runCalculation(
      buildCalcInput(
        base({
          apiPriceId: API_PRICE_CUSTOM,
          customApiInputPerMTokUsd: 1,
          customApiOutputPerMTokUsd: 3,
          inputRatio: 0.5,
          regionId: 'us-commercial', // USD, so no conversion to reason about
        }),
      ),
    );
    // 0.5 x 1 + 0.5 x 3 = 2.00 USD per 1M
    expect(r.cost.apiPerMTokens).toBeCloseTo(2, 6);
  });

  it('produces no comparison figures when set to none', () => {
    const r = runCalculation(buildCalcInput(base({ apiPriceId: API_PRICE_NONE })));
    expect(r.cost.apiPerMTokens).toBeUndefined();
    expect(r.cost.savingsPerMTokens).toBeUndefined();
  });
});

describe('share link round-trip', () => {
  it('restores a custom tariff as a number, not a string', () => {
    const state = base({ customPricePerKWh: 0.2137, customGridIntensity: 42 });
    const restored = queryToState(stateToQuery(state));

    expect(restored.customPricePerKWh).toBe(0.2137);
    expect(typeof restored.customPricePerKWh).toBe('number');
    expect(restored.customGridIntensity).toBe(42);
    expect(typeof restored.customGridIntensity).toBe('number');
  });

  it('handles every nullable-number key the same way', () => {
    const state = base({ hardwareCapex: 2500, customPricePerKWh: 0.31 });
    const restored = queryToState(stateToQuery(state));
    expect(typeof restored.hardwareCapex).toBe('number');
    expect(typeof restored.customPricePerKWh).toBe('number');
  });

  it('omits values left at their default to keep links short', () => {
    const q = stateToQuery(base());
    expect(q).not.toContain('customPricePerKWh');
    expect(q).not.toContain('regionId');
  });

  it('survives a full round-trip of a realistic configuration', () => {
    const state = base({
      modelId: 'deepseek-v3',
      gpuId: 'h100-sxm',
      numGpus: 8,
      quantId: 'fp8',
      contextLength: 32768,
      batchSize: 32,
      customPricePerKWh: 0.1899,
      advanced: true,
    });
    const restored = queryToState(stateToQuery(state));

    expect(restored.modelId).toBe('deepseek-v3');
    expect(restored.numGpus).toBe(8);
    expect(restored.contextLength).toBe(32768);
    expect(restored.customPricePerKWh).toBe(0.1899);
    expect(restored.advanced).toBe(true);
  });

  it('ignores junk rather than writing NaN into the state', () => {
    const restored = queryToState('customPricePerKWh=abc&numGpus=xyz');
    expect(restored.customPricePerKWh).toBeUndefined();
    expect(restored.numGpus).toBeUndefined();
  });
});
