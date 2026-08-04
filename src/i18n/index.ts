import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { de } from './de';
import { en } from './en';

export type Language = 'de' | 'en';

const BUNDLES = { de, en } as const;

interface LanguageState {
  language: Language;
  setLanguage: (language: Language) => void;
}

export const useLanguage = create<LanguageState>()(
  persist(
    (set) => ({
      // Default to German: the primary audience is the thesis and its reviewers.
      language: typeof navigator !== 'undefined' && !navigator.language.startsWith('de')
        ? 'en'
        : 'de',
      setLanguage: (language) => set({ language }),
    }),
    { name: 'llmcalc.lang.v1' },
  ),
);

/** Resolve a dotted key such as `results.vramUsage` against a bundle. */
function resolve(bundle: unknown, key: string): string | undefined {
  const value = key
    .split('.')
    .reduce<unknown>((acc, part) => (acc as Record<string, unknown>)?.[part], bundle);
  return typeof value === 'string' ? value : undefined;
}

export type TranslateFn = (key: string, values?: Record<string, string | number>) => string;

/**
 * Minimal translator. Falls back to English, then to the key itself, so a
 * missing string is visible in the UI rather than silently blank.
 */
export function translate(
  language: Language,
  key: string,
  values?: Record<string, string | number>,
): string {
  const text = resolve(BUNDLES[language], key) ?? resolve(BUNDLES.en, key) ?? key;
  if (!values) return text;

  return text.replace(/\{\{(\w+)\}\}/g, (match, name: string) =>
    name in values ? String(values[name]) : match,
  );
}

export function useT(): TranslateFn {
  const language = useLanguage((s) => s.language);
  return (key, values) => translate(language, key, values);
}

export { de, en };
