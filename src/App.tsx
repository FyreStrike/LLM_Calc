import { useEffect, useMemo, useState } from 'react';
import { EnergyCostPanel } from './components/EnergyCostPanel';
import { HardwarePanel } from './components/HardwarePanel';
import { HostPanel } from './components/HostPanel';
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
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-[var(--border)] bg-[var(--surface)]/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-4 gap-y-2 px-5 py-3">
          <div className="mr-auto min-w-0">
            <h1 className="truncate text-[15px] font-semibold tracking-tight text-[var(--text)]">
              {t('app.title')}
            </h1>
            <p className="truncate text-[12px] text-[var(--text-3)]">{t('app.subtitle')}</p>
          </div>

          <Toggle
            checked={state.advanced}
            onChange={(v) => state.set('advanced', v)}
            label={t('app.advanced')}
          />

          <SegmentedControl<Language>
            value={language}
            onChange={setLanguage}
            options={[
              { value: 'de', label: 'DE' },
              { value: 'en', label: 'EN' },
            ]}
          />

          <Button variant={copied ? 'primary' : 'secondary'} onClick={share}>
            {copied ? t('app.shareCopied') : t('app.share')}
          </Button>
          <Button variant="ghost" onClick={() => state.reset()}>
            {t('app.reset')}
          </Button>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] px-5 py-5">
        <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
          {/* Inputs stay in view while the results update beside them. */}
          <div className="space-y-4 lg:sticky lg:top-[72px] lg:max-h-[calc(100vh-92px)] lg:overflow-y-auto lg:pr-1 lg:pb-4">
            <ModelPanel />
            <HardwarePanel />
            <HostPanel />
            <WorkloadPanel />
            <EnergyCostPanel />
          </div>

          <div>
            {result ? (
              <Results result={result} />
            ) : (
              <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bad-soft)] p-4 text-[13px] text-[var(--bad)]">
                {language === 'de'
                  ? 'Berechnung fehlgeschlagen — bitte die Modellkonfiguration prüfen.'
                  : 'Calculation failed — check the model configuration.'}
              </div>
            )}
          </div>
        </div>

        <footer className="mt-8 border-t border-[var(--border)] pt-4 pb-2 text-[11px] leading-relaxed text-[var(--text-3)]">
          <p>
            {language === 'de'
              ? 'Alle Formeln und Quellen sind in METHODIK.md dokumentiert. Energiemodell nach dem erweiterten Energie-Roofline-Modell:'
              : 'All formulas and sources are documented in METHODIK.md. Energy model follows the extended energy roofline:'}{' '}
            <code className="rounded bg-[var(--surface-2)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--text-2)]">
              E = ε_flop·W + ε_mop·Q + π₀·T
            </code>
          </p>
        </footer>
      </main>
    </div>
  );
}
