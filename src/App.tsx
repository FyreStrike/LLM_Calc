import { useEffect, useMemo, useState } from 'react';
import { EnergyCostPanel } from './components/EnergyCostPanel';
import { HardwarePanel } from './components/HardwarePanel';
import { ModelPanel } from './components/ModelPanel';
import { Results } from './components/Results';
import { WorkloadPanel } from './components/WorkloadPanel';
import { runCalculation } from './core';
import { useLanguage, useT, type Language } from './i18n';
import { buildCalcInput, queryToState, stateToQuery, useStore } from './state/store';
import { Button, SegmentedControl, Toggle } from './ui/primitives';

export default function App() {
  const t = useT();
  const { language, setLanguage } = useLanguage();
  const state = useStore();
  const patch = useStore((s) => s.patch);
  const [copied, setCopied] = useState(false);

  // Hydrate from a shared link once on mount. Query params win over persisted
  // state, so a link always shows the sender's configuration.
  useEffect(() => {
    const query = window.location.hash.replace(/^#\??/, '') || window.location.search.slice(1);
    if (query) patch(queryToState(query));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const result = useMemo(() => {
    try {
      return runCalculation(buildCalcInput(state));
    } catch (e) {
      console.error('Calculation failed', e);
      return null;
    }
  }, [state]);

  function share() {
    const url = `${window.location.origin}${window.location.pathname}#?${stateToQuery(state)}`;
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50">{t('app.title')}</h1>
          <p className="text-sm text-slate-600 dark:text-slate-400">{t('app.subtitle')}</p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <SegmentedControl<Language>
            value={language}
            onChange={setLanguage}
            options={[
              { value: 'de', label: 'DE' },
              { value: 'en', label: 'EN' },
            ]}
          />
          <Button onClick={share}>{copied ? t('app.shareCopied') : t('app.share')}</Button>
          <Button onClick={() => state.reset()}>{t('app.reset')}</Button>
        </div>
      </header>

      <div className="mb-4">
        <Toggle
          checked={state.advanced}
          onChange={(v) => state.set('advanced', v)}
          label={t('app.advanced')}
          help={t('app.advancedHint')}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <div className="space-y-4">
          <ModelPanel />
          <HardwarePanel />
          <WorkloadPanel />
          <EnergyCostPanel />
        </div>

        <div>
          {result ? (
            <Results result={result} />
          ) : (
            <p className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-200">
              Calculation failed — check the model configuration.
            </p>
          )}
        </div>
      </div>

      <footer className="mt-8 border-t border-slate-200 pt-4 text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400">
        <p>
          {language === 'de'
            ? 'Alle Formeln und Quellen sind in METHODIK.md dokumentiert. Energiemodell nach dem erweiterten Energie-Roofline-Modell:'
            : 'All formulas and sources are documented in METHODIK.md. Energy model follows the extended energy roofline:'}{' '}
          <code>E = ε_flop·W + ε_mop·Q + π₀·T</code>
        </p>
      </footer>
    </div>
  );
}
