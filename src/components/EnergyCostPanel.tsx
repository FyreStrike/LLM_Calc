import { useEffect } from 'react';
import { defaultEpsFlopPJ, defaultEpsMopPJ } from '../core/energy';
import { getQuant } from '../core/quant';
import { peakFlopsFor } from '../core/roofline';
import type { EnergyMode } from '../core/types';
import { GPUS, getGpu } from '../data/gpus';
import { getRegion, PUE_PRESETS, REGIONS } from '../data/regions';
import { useLanguage, useT } from '../i18n';
import { fetchApiPrices } from '../services/openrouter';
import {
  API_PRICE_AUTO,
  API_PRICE_CUSTOM,
  API_PRICE_NONE,
  availablePrices,
  effectiveRegion,
  useStore,
} from '../state/store';
import { num, sci } from '../ui/format';
import { Button, Card, Field, NumberInput, Select, Slider } from '../ui/primitives';

export function EnergyCostPanel() {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const state = useStore();
  const gpu = getGpu(state.gpuId) ?? GPUS[0];
  const prices = availablePrices(state);
  const baseRegion = getRegion(state.regionId) ?? REGIONS[0];
  const region = effectiveRegion(state);
  const isCustomPrice = state.customPricePerKWh !== null;
  const matchedPue = PUE_PRESETS.find((p) => Math.abs(p.value - state.pue) < 1e-9);

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
          <div className="space-y-3 rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-3">
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
            <p className="text-[11px] tabular-nums text-[var(--text-3)]">
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

        <div className="grid grid-cols-[1fr_auto] gap-3">
          <Field label={t('energy.pue')} help={t('energy.pueHelp')}>
            <Select
              // A custom value belongs to no preset, so the select falls back
              // to a placeholder rather than mislabelling it.
              value={matchedPue ? String(matchedPue.value) : ''}
              onChange={(v) => state.set('pue', Number(v))}
            >
              {!matchedPue && <option value="">{t('energy.pueCustom')}</option>}
              {PUE_PRESETS.map((p) => (
                <option key={p.id} value={p.value}>
                  {t(p.labelKey)}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="&nbsp;">
            <NumberInput
              value={state.pue}
              onChange={(v) => state.set('pue', Math.max(1, v ?? 1))}
              min={1}
              max={3}
              step={0.01}
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

        <div className="space-y-3 border-t border-[var(--border)] pt-3">
          <Field label={t('cost.region')}>
            <Select
              value={state.regionId}
              onChange={(v) => {
                // Picking a preset means wanting the preset, so any manual
                // override is cleared rather than silently masking it.
                state.patch({
                  regionId: v,
                  customPricePerKWh: null,
                  customGridIntensity: null,
                });
              }}
            >
              {REGIONS.map((r) => (
                <option key={r.id} value={r.id}>
                  {t(r.label)} — {num(r.pricePerKWh, language, 3)} {r.currency}/kWh
                </option>
              ))}
            </Select>
          </Field>

          <Field
            label={t('cost.pricePerKWh')}
            help={isCustomPrice ? undefined : baseRegion.source}
            hint={
              isCustomPrice ? (
                <button
                  type="button"
                  className="cursor-pointer text-[11px] font-medium text-[var(--accent)] hover:underline"
                  onClick={() => state.set('customPricePerKWh', null)}
                >
                  {t('cost.resetPreset')}
                </button>
              ) : undefined
            }
          >
            <NumberInput
              value={region.pricePerKWh}
              onChange={(v) => state.set('customPricePerKWh', v)}
              min={0}
              step={0.01}
              suffix={`${region.currency}/kWh`}
            />
          </Field>

          {state.advanced && (
            <Field
              label={t('cost.gridIntensity')}
              hint={
                state.customGridIntensity !== null ? (
                  <button
                    type="button"
                    className="cursor-pointer text-[11px] font-medium text-[var(--accent)] hover:underline"
                    onClick={() => state.set('customGridIntensity', null)}
                  >
                    {t('cost.resetPreset')}
                  </button>
                ) : undefined
              }
            >
              <NumberInput
                value={region.gridIntensityGCO2PerKWh}
                onChange={(v) => state.set('customGridIntensity', v)}
                min={0}
                step={10}
                suffix="g/kWh"
              />
            </Field>
          )}
        </div>

        <Field
          label={t('cost.apiModel')}
          help={`${t(`cost.priceOrigin.${state.priceOrigin}`)}${
            state.priceAsOf ? ` — ${t('cost.asOf', { date: state.priceAsOf })}` : ''
          }`}
        >
          <Select
            value={state.apiPriceId ?? API_PRICE_AUTO}
            onChange={(v) => state.set('apiPriceId', v)}
          >
            <option value={API_PRICE_AUTO}>{t('cost.apiAuto')}</option>
            <option value={API_PRICE_NONE}>{t('cost.apiNone')}</option>
            <option value={API_PRICE_CUSTOM}>{t('cost.apiCustom')}</option>
            {prices.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label} — ${num(p.inputPerMTokUsd, language, 2)}/$
                {num(p.outputPerMTokUsd, language, 2)}
              </option>
            ))}
          </Select>
        </Field>

        {state.apiPriceId === API_PRICE_CUSTOM && (
          <div className="grid grid-cols-2 gap-3 rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2.5">
            <Field label={t('cost.customInput')}>
              <NumberInput
                value={state.customApiInputPerMTokUsd}
                onChange={(v) => state.set('customApiInputPerMTokUsd', v)}
                min={0}
                step={0.05}
                suffix="$"
              />
            </Field>
            <Field label={t('cost.customOutput')}>
              <NumberInput
                value={state.customApiOutputPerMTokUsd}
                onChange={(v) => state.set('customApiOutputPerMTokUsd', v)}
                min={0}
                step={0.05}
                suffix="$"
              />
            </Field>
            <p className="col-span-2 text-[11px] leading-relaxed text-[var(--text-3)]">
              {t('cost.customPriceHelp', { currency: region.currency })}
            </p>
          </div>
        )}

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
