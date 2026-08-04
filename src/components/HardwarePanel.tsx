import { QUANTIZATIONS, KV_QUANTIZATIONS } from '../core/quant';
import type { QuantGroup } from '../core/types';
import { GPUS, GPU_GROUPS, getGpu } from '../data/gpus';
import { useLanguage, useT } from '../i18n';
import { useStore } from '../state/store';
import { num } from '../ui/format';
import { Badge, Card, Field, Note, Select, Toggle } from '../ui/primitives';

const QUANT_GROUPS: QuantGroup[] = ['float', 'gguf', 'gptq-awq', 'mx'];

export function HardwarePanel() {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const state = useStore();
  const gpu = getGpu(state.gpuId) ?? GPUS[0];

  return (
    <Card title={t('section.hardware')}>
      <div className="space-y-3">
        <Field label={t('hardware.gpu')}>
          <Select value={state.gpuId} onChange={(v) => state.set('gpuId', v)}>
            {GPU_GROUPS.map((g) => (
              <optgroup key={g.vendor} label={t(g.labelKey)}>
                {GPUS.filter((x) => x.vendor === g.vendor).map((x) => (
                  <option key={x.id} value={x.id}>
                    {x.name} — {x.vramGb} GB
                  </option>
                ))}
              </optgroup>
            ))}
          </Select>
        </Field>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
          <SpecRow label={t('hardware.vram')} value={`${gpu.vramGb} GB`} />
          <SpecRow
            label={t('hardware.bandwidth')}
            value={`${num(gpu.bandwidthGBs, language, 0)} GB/s`}
          />
          <SpecRow
            label={t('hardware.compute')}
            value={`${num(gpu.fp16TFlops, language, 0)} TFLOPS`}
          />
          <SpecRow label={t('hardware.tdp')} value={`${gpu.tdpW} W`} />
        </dl>

        {gpu.unified && <Badge tone="info">{t('hardware.unifiedMemory')}</Badge>}
        {gpu.note && <Note tone="warn">{t(gpu.note)}</Note>}

        <div className="grid grid-cols-2 gap-3">
          <Field label={t('hardware.numGpus')}>
            <Select
              value={String(state.numGpus)}
              onChange={(v) => state.set('numGpus', Number(v))}
            >
              {[1, 2, 4, 8, 16, 32, 64].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t('hardware.parallelism')}>
            <Select
              value={state.parallelism}
              onChange={(v) => state.set('parallelism', v as 'tp' | 'pp')}
            >
              <option value="tp">{t('hardware.tp')}</option>
              <option value="pp">{t('hardware.pp')}</option>
            </Select>
          </Field>
        </div>

        {state.numGpus > 1 && (
          <Toggle
            checked={state.nvlinkOverride ?? gpu.nvlink ?? false}
            onChange={(v) => state.set('nvlinkOverride', v)}
            label={t('hardware.nvlink')}
            help={t('hardware.nvlinkHelp')}
          />
        )}

        <div className="grid grid-cols-2 gap-3 border-t border-[var(--border)] pt-3">
          <Field label={t('quant.weights')}>
            <Select value={state.quantId} onChange={(v) => state.set('quantId', v)}>
              {QUANT_GROUPS.map((group) => (
                <optgroup key={group} label={t(`quant.group.${group}`)}>
                  {QUANTIZATIONS.filter((q) => q.group === group).map((q) => (
                    <option key={q.id} value={q.id}>
                      {q.label} — {num(q.bpw, language, 2)} bpw
                    </option>
                  ))}
                </optgroup>
              ))}
            </Select>
          </Field>
          <Field label={t('quant.kvCache')}>
            <Select value={state.kvQuantId} onChange={(v) => state.set('kvQuantId', v)}>
              {KV_QUANTIZATIONS.map((q) => (
                <option key={q.id} value={q.id}>
                  {q.label}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <p className="text-[11px] leading-relaxed text-[var(--text-3)]">
          {t('quant.ggufHint')}
        </p>
      </div>
    </Card>
  );
}

function SpecRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-[var(--text-3)]">{label}</dt>
      <dd className="font-semibold tabular-nums text-[var(--text-2)]">{value}</dd>
    </div>
  );
}
