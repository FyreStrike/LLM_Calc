import type { Language } from '../i18n';

const GB = 1024 ** 3;

export function locale(language: Language): string {
  return language === 'de' ? 'de-DE' : 'en-US';
}

export function num(
  value: number,
  language: Language,
  digits = 1,
): string {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString(locale(language), {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function gb(bytes: number, language: Language, digits = 2): string {
  return num(bytes / GB, language, digits);
}

/** Compact byte formatting that picks a sensible unit. */
export function bytes(value: number, language: Language): string {
  if (value >= GB) return `${num(value / GB, language, 2)} GB`;
  if (value >= 1024 ** 2) return `${num(value / 1024 ** 2, language, 1)} MB`;
  if (value >= 1024) return `${num(value / 1024, language, 1)} KB`;
  return `${num(value, language, 0)} B`;
}

/** Parameter counts read better as 8B / 671B than 8,030,000,000. */
export function params(value: number, language: Language): string {
  if (value >= 1e12) return `${num(value / 1e12, language, 2)}T`;
  if (value >= 1e9) return `${num(value / 1e9, language, value >= 1e11 ? 0 : 1)}B`;
  if (value >= 1e6) return `${num(value / 1e6, language, 0)}M`;
  return num(value, language, 0);
}

export function tokens(value: number, language: Language): string {
  if (value >= 1e12) {
    return `${num(value / 1e12, language, 1)} ${language === 'de' ? 'Bio.' : 'T'}`;
  }
  if (value >= 1e9) {
    return `${num(value / 1e9, language, 1)} ${language === 'de' ? 'Mrd.' : 'B'}`;
  }
  if (value >= 1e6) return `${num(value / 1e6, language, 1)} M`;
  if (value >= 1e3) return `${num(value / 1e3, language, 0)} k`;
  return num(value, language, 0);
}

export function money(
  value: number,
  currency: 'EUR' | 'USD',
  language: Language,
  digits = 3,
): string {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString(locale(language), {
    style: 'currency',
    currency,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function contextLabel(value: number, language: Language): string {
  if (value >= 1024 * 1024) return `${num(value / (1024 * 1024), language, 0)}M`;
  if (value >= 1024) return `${num(value / 1024, language, 0)}k`;
  return num(value, language, 0);
}

/** Scientific notation for the roofline coefficients, which span many decades. */
export function sci(value: number, language: Language, digits = 3): string {
  if (!Number.isFinite(value) || value === 0) return '0';
  const exp = Math.floor(Math.log10(Math.abs(value)));
  const mantissa = value / 10 ** exp;
  return `${num(mantissa, language, digits)}e${exp}`;
}

export function duration(days: number, language: Language): string {
  if (!Number.isFinite(days)) return '—';
  if (days >= 365) return `${num(days / 365, language, 1)} ${language === 'de' ? 'Jahre' : 'years'}`;
  if (days >= 30) return `${num(days / 30, language, 1)} ${language === 'de' ? 'Monate' : 'months'}`;
  return `${num(days, language, 0)} ${language === 'de' ? 'Tage' : 'days'}`;
}
