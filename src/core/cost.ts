import type { ApiPrice, CostOptions, CostResult, EnergyResult } from './types';

/**
 * Local electricity cost vs. API list price.
 *
 * The comparison most people get wrong in two ways:
 *
 * 1. They compare against the *output* token price alone. Real traffic is
 *    mostly input tokens, which are usually 2-5x cheaper, so `inputRatio` is a
 *    first-class input here rather than a hidden constant.
 *
 * 2. They ignore hardware capital cost, which makes local look free. The
 *    break-even calculation below puts the purchase price back in.
 */

/** Blended API price per 1M tokens at the user's input/output mix. */
export function blendedApiPricePerMTokens(
  price: ApiPrice,
  inputRatio: number,
  usdToCurrency: number,
): number {
  const r = Math.min(1, Math.max(0, inputRatio));
  const usd = r * price.inputPerMTokUsd + (1 - r) * price.outputPerMTokUsd;
  return usd * usdToCurrency;
}

export function computeCost(
  energy: EnergyResult,
  options: CostOptions,
  apiPrice?: ApiPrice,
): CostResult {
  const { region } = options;

  const localElectricityPerMTokens = energy.kWhPerMTokens * region.pricePerKWh;
  const co2GramsPerMTokens = energy.kWhPerMTokens * region.gridIntensityGCO2PerKWh;

  const result: CostResult = {
    localElectricityPerMTokens,
    co2GramsPerMTokens,
    currency: region.currency,
  };

  if (apiPrice) {
    const apiPerMTokens = blendedApiPricePerMTokens(
      apiPrice,
      options.inputRatio,
      options.usdToCurrency,
    );
    result.apiPerMTokens = apiPerMTokens;
    result.savingsPerMTokens = apiPerMTokens - localElectricityPerMTokens;

    // Break-even: the hardware pays for itself once the accumulated API bill
    // you avoided exceeds its purchase price. Only meaningful while local
    // marginal cost is genuinely below the API's.
    if (options.hardwareCapex && options.hardwareCapex > 0) {
      const marginPerMTokens = apiPerMTokens - localElectricityPerMTokens;
      if (marginPerMTokens > 0) {
        const breakEvenMTokens = options.hardwareCapex / marginPerMTokens;
        result.breakEvenTokens = breakEvenMTokens * 1e6;

        if (options.dailyTokens && options.dailyTokens > 0) {
          result.breakEvenDays = result.breakEvenTokens / options.dailyTokens;
        }

        // Amortize the capex over the break-even volume so the "total cost"
        // line is comparable to the API price at that operating point.
        result.localTotalPerMTokens =
          localElectricityPerMTokens + options.hardwareCapex / breakEvenMTokens;
      }
    }
  }

  if (!result.localTotalPerMTokens && options.hardwareCapex && options.dailyTokens) {
    // No API price selected, but we can still amortize over a 3-year life.
    const lifetimeMTokens = (options.dailyTokens * 365 * 3) / 1e6;
    if (lifetimeMTokens > 0) {
      result.localTotalPerMTokens =
        localElectricityPerMTokens + options.hardwareCapex / lifetimeMTokens;
    }
  }

  return result;
}
