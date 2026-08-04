import type { Region } from '../core/types';

/**
 * Electricity prices and grid carbon intensity.
 *
 * EU prices: Eurostat, second half of 2025, EUR/kWh including all taxes and
 * levies. Household and non-household are listed separately because the gap is
 * large — a German household pays 0.3869 EUR/kWh while a business pays 0.2264,
 * which is nearly enough on its own to flip a local-vs-API verdict.
 * https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Electricity_price_statistics
 *
 * US prices: EIA 2026 year-to-date averages by sector.
 * https://www.eia.gov/electricity/sales_revenue_price/pdf/table_4.pdf
 *
 * Carbon intensities are annual grid averages and vary enormously by hour and
 * season; treat them as an order-of-magnitude guide, not a measurement.
 */
export const REGIONS: Region[] = [
  {
    id: 'de-household',
    label: 'region.deHousehold',
    pricePerKWh: 0.3869,
    currency: 'EUR',
    gridIntensityGCO2PerKWh: 380,
    source: 'Eurostat H2 2025',
  },
  {
    id: 'de-business',
    label: 'region.deBusiness',
    pricePerKWh: 0.2264,
    currency: 'EUR',
    gridIntensityGCO2PerKWh: 380,
    source: 'Eurostat H2 2025',
  },
  {
    id: 'de-industrial',
    label: 'region.deIndustrial',
    pricePerKWh: 0.05,
    currency: 'EUR',
    gridIntensityGCO2PerKWh: 380,
    source: 'Industriestrompreis (subsidised)',
  },
  {
    id: 'eu-household',
    label: 'region.euHousehold',
    pricePerKWh: 0.2896,
    currency: 'EUR',
    gridIntensityGCO2PerKWh: 250,
    source: 'Eurostat H2 2025',
  },
  {
    id: 'eu-business',
    label: 'region.euBusiness',
    pricePerKWh: 0.1837,
    currency: 'EUR',
    gridIntensityGCO2PerKWh: 250,
    source: 'Eurostat H2 2025',
  },
  {
    id: 'fr-household',
    label: 'region.frHousehold',
    pricePerKWh: 0.2316,
    currency: 'EUR',
    gridIntensityGCO2PerKWh: 56,
    source: 'Eurostat H2 2025',
  },
  {
    id: 'no-household',
    label: 'region.noHousehold',
    pricePerKWh: 0.115,
    currency: 'EUR',
    gridIntensityGCO2PerKWh: 30,
    source: 'Eurostat H2 2025',
  },
  {
    id: 'us-residential',
    label: 'region.usResidential',
    pricePerKWh: 0.183,
    currency: 'USD',
    gridIntensityGCO2PerKWh: 370,
    source: 'EIA 2026 YTD',
  },
  {
    id: 'us-commercial',
    label: 'region.usCommercial',
    pricePerKWh: 0.135,
    currency: 'USD',
    gridIntensityGCO2PerKWh: 370,
    source: 'EIA 2026 YTD',
  },
  {
    id: 'us-industrial',
    label: 'region.usIndustrial',
    pricePerKWh: 0.085,
    currency: 'USD',
    gridIntensityGCO2PerKWh: 370,
    source: 'EIA 2026 YTD',
  },
];

export function getRegion(id: string): Region | undefined {
  return REGIONS.find((r) => r.id === id);
}

export const DEFAULT_REGION_ID = 'de-household';

/**
 * Power usage effectiveness presets.
 * - Home PC: no datacenter overhead at all.
 * - Industry average 1.54 (Uptime Institute 2025 Global Data Center Survey).
 * - Hyperscaler 1.09 (Google fleet-wide, 2025).
 */
export const PUE_PRESETS = [
  { id: 'home', labelKey: 'pue.home', value: 1.0 },
  { id: 'office', labelKey: 'pue.office', value: 1.15 },
  { id: 'datacenter', labelKey: 'pue.datacenter', value: 1.54 },
  { id: 'hyperscaler', labelKey: 'pue.hyperscaler', value: 1.09 },
];

/** Approximate USD -> EUR rate; editable in the UI. */
export const DEFAULT_USD_TO_EUR = 0.92;
