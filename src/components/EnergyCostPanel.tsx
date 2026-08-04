import { useEffect } from 'react';
import { defaultEpsFlopPJ, defaultEpsMopPJ } from '../core/energy';
import { getQuant } from '../core/quant';
import { peakFlopsFor } from '../core/roofline';
import type { EnergyMode } from '../core/types';
import { GPUS, getGpu } from '../data/gpus';
import { PUE_PRESETS, REGIONS } from '../data/regions';
import { useLanguage, useT } from '../i18n';
import { fetchApiPrices } from '../services/openrouter';
import { availablePrices, useStore } from '../state/store';
import { num, sci } from '../ui/format';
import { Button, Card, Field, NumberInput, Select, Slider } from '../ui/primitives';

export function EnergyCostPanel() {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const state = useStore();
  const gpu = getGpu(state.gpuId) ?? GPUS[0];
  const prices = availablePrices(state);

  // Refresh prices once on mount; the service falls back to cache, then to the
  // bundled snapshot, so this can fail silently without breaking the page.
  useEffect(() => {
    let cancelled = false;
    fetchApiPrices().then((res) => {
      if (!cancelled) state.setPrices(res.prices, res.origin, res.asOf);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const peakFlops = peakFlopsFor(gpu, getQuant(state.quantId).computePrecision);

  return (
    <Card title={`${t('section.energy')} & ${t('section.cost')}`}>
      <div className="space-y-3">
        <Field label={t('energy.mode')} help={t(`energy.${state.energyMode}Help`)}>
          <Select
            value={state.energyMode}
            onChange={(v) => state.set('energyMode', v as EnergyMode)}
          >
            <option value="simple">{t('energy.simple')}</option>
            <option value="roofline">{t('energy.roofline')}</option>
          </Select>
        </Field>

        {state.energyMode === 'roofline' && state.advanced && (
          <div className="space-y-3 rounded-lg bg-slate-50 p-3 dark:bg-slate-800/60">
            <Field label={`${t('energy.epsFlop')} (pJ/FLOP)`}>
              <NumberInput
                value={state.epsFlopPJ ?? Number(defaultEpsFlopPJ(gpu, peakFlops).toPrecision(4))}
                onChange={(v) => state.set('epsFlopPJ', v)}
                step={0.00001}
                min={0}
              />
            </Field>
            <Field label={`${t('energy.epsMop')} (pJ/byte)`}>
              <NumberInput
                value={state.epsMopPJ ?? Number(defaultEpsMopPJ(gpu).toPrecision(4))}
                onChange={(v) => state.set('epsMopPJ', v)}
                step={0.001}
                min={0}
              />
            </Field>
            <p className="text-xs tabular-nums text-slate-500 dark:text-slate-400">
              {t('energy.idle')}: {gpu.idleW} W · defaults ε_flop={' '}
              {sci(defaultEpsFlopPJ(gpu, peakFlops), language)} · ε_mop={' '}
              {sci(defaultEpsMopPJ(gpu), language)}
            </p>
            <Button
              onClick={() => {
                state.set('epsFlopPJ', null);
                state.set('epsMopPJ', null);
              }}
            >
              {t('energy.calibrate')}
            </Button>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <Field label={t('energy.pue')}>
            <Select
              value={String(state.pue)}
              onChange={(v) => state.set('pue', Number(v))}
            >
              {PUE_PRESETS.map((p) => (
                <option key={p.id} value={p.value}>
                  {t(p.labelKey)}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={`${t('energy.hostOverhead')} (W)`}>
            <NumberInput
              value={state.hostOverheadW}
              onChange={(v) => state.set('hostOverheadW', v ?? 0)}
              min={0}
              step={10}
            />
          </Field>
        </div>

        {state.advanced && (
          <Field
            label={t('energy.psu')}
            hint={`${num(state.psuEfficiency * 100, language, 0)}%`}
          >
            <Slider
              value={state.psuEfficiency}
              onChange={(v) => state.set('psuEfficiency', v)}
              min={0.7}
              max={0.98}
              step={0.01}
            />
          </Field>
        )}

        <div className="border-t border-slate-200 pt-3 dark:border-slate-700">
          <Field label={t('cost.region')}>
            <Select value={state.regionId} onChange={(v) => state.set('regionId', v)}>
              {REGIONS.map((r) => (
                <option key={r.id} value={r.id}>
                  {t(r.label)} — {num(r.pricePerKWh, language, 3)} {r.currency}/kWh
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <Field
          label={t('cost.apiModel')}
          help={`${t(`cost.priceOrigin.${state.priceOrigin}`)}${
            state.priceAsOf ? ` — ${t('cost.asOf', { date: state.priceAsOf })}` : ''
          }`}
        >
          <Select
            value={state.apiPriceId ?? ''}
            onChange={(v) => state.set('apiPriceId', v === '' ? null : v)}
          >
            <option value="">{t('cost.apiNone')}</option>
            {prices.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label} — ${num(p.inputPerMTokUsd, language, 2)}/$
                {num(p.outputPerMTokUsd, language, 2)}
              </option>
            ))}
          </Select>
        </Field>

        <Field
          label={t('cost.inputRatio')}
          hint={`${num(state.inputRatio * 100, language, 0)}% / ${num(
            (1 - state.inputRatio) * 100,
            language,
            0,
          )}%`}
          help={t('cost.inputRatioHelp')}
        >
          <Slider
            value={state.inputRatio}
            onChange={(v) => state.set('inputRatio', v)}
            min={0}
            max={1}
            step={0.05}
          />
        </Field>

        <div className="grid grid-cols-2 gap-3">
          <Field label={t('cost.hardwareCapex')} help={t('cost.hardwareCapexHelp')}>
            <NumberInput
              value={state.hardwareCapex}
              onChange={(v) => state.set('hardwareCapex', v)}
              min={0}
              step={100}
              placeholder={gpu.priceUsd ? String(gpu.priceUsd * state.numGpus) : '—'}
            />
          </Field>
          <Field label={t('cost.dailyTokens')}>
            <NumberInput
              value={state.dailyTokens}
              onChange={(v) => state.set('dailyTokens', v ?? 0)}
              min={0}
              step={100000}
            />
          </Field>
        </div>

        <Button
          onClick={() =>
            fetchApiPrices(true).then((res) =>
              state.setPrices(res.prices, res.origin, res.asOf),
            )
          }
        >
          {t('cost.refresh')}
        </Button>
      </div>
    </Card>
  );
}
