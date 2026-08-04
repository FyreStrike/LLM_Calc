import { ramBandwidthGBs, ramPower } from '../core/host';
import { CHANNEL_OPTIONS, getRamType, HOST_PRESETS, RAM_TYPES } from '../data/ram';
import { useLanguage, useT } from '../i18n';
import { hostSpec, useStore } from '../state/store';
import { num } from '../ui/format';
import { Card, Field, NumberInput, Note, Select } from '../ui/primitives';

/**
 * Host system: memory specification and non-GPU draw.
 *
 * The RAM fields replace a bandwidth slider nobody could estimate. They also
 * make two effects visible that a single "host overhead" number hides — that
 * offloading is bounded by installed capacity, and that idle memory draw is
 * paid for every module whether the model needs it or not.
 */
export function HostPanel() {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const state = useStore();

  const host = hostSpec(state);
  const ramType = getRamType(state.ramTypeId);
  const bandwidth = ramBandwidthGBs(host.ram);
  const idlePower = ramPower(host.ram, ramType, 0);
  const fullPower = ramPower(host.ram, ramType, 1);

  function applyPreset(id: string) {
    const preset = HOST_PRESETS.find((p) => p.id === id);
    if (!preset) return;
    state.patch({
      hostPresetId: id,
      hostBaseOverheadW: preset.baseOverheadW,
      ramModules: preset.ramModules,
      ramCapacityGb: preset.ramCapacityGb,
      // Servers run many channels; desktops two.
      ramChannels: id === 'server' ? 8 : id === 'workstation' ? 4 : 2,
    });
  }

  return (
    <Card title={t('section.host')}>
      <div className="space-y-3">
        <Field label={t('host.preset')}>
          <Select value={state.hostPresetId} onChange={applyPreset}>
            {HOST_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {t(p.labelKey)}
              </option>
            ))}
          </Select>
        </Field>

        <div className="grid grid-cols-2 gap-3">
          <Field label={t('host.ramType')}>
            <Select
              value={state.ramTypeId}
              onChange={(v) => {
                const type = getRamType(v);
                state.patch({ ramTypeId: v, ramSpeedMTps: type.defaultSpeed });
              }}
            >
              {RAM_TYPES.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t('host.ramSpeed')}>
            <Select
              value={String(state.ramSpeedMTps)}
              onChange={(v) => state.set('ramSpeedMTps', Number(v))}
            >
              {ramType.speeds.map((s) => (
                <option key={s} value={s}>
                  {ramType.label}-{s}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <Field label={t('host.channels')}>
            <Select
              value={String(state.ramChannels)}
              onChange={(v) => state.set('ramChannels', Number(v))}
            >
              {CHANNEL_OPTIONS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t('host.capacity')}>
            <NumberInput
              value={state.ramCapacityGb}
              onChange={(v) => state.set('ramCapacityGb', v ?? 0)}
              min={1}
              step={8}
              suffix="GB"
            />
          </Field>
          <Field label={t('host.modules')} help={t('host.modulesHelp')}>
            <NumberInput
              value={state.ramModules}
              onChange={(v) => state.set('ramModules', Math.max(1, v ?? 1))}
              min={1}
              step={1}
            />
          </Field>
        </div>

        {/* The two numbers the specification actually buys you. */}
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2.5 text-[11px]">
          <div className="flex justify-between gap-2">
            <dt className="text-[var(--text-3)]">{t('host.bandwidth')}</dt>
            <dd className="font-semibold tabular-nums text-[var(--text-2)]">
              {num(bandwidth, language, 1)} GB/s
            </dd>
          </div>
          <div className="flex justify-between gap-2">
            <dt className="text-[var(--text-3)]">{t('host.ramIdlePower')}</dt>
            <dd className="font-semibold tabular-nums text-[var(--text-2)]">
              {num(idlePower.idleW, language, 1)} W
            </dd>
          </div>
          <div className="flex justify-between gap-2">
            <dt className="text-[var(--text-3)]">{t('host.ramFullPower')}</dt>
            <dd className="font-semibold tabular-nums text-[var(--text-2)]">
              {num(fullPower.totalW, language, 1)} W
            </dd>
          </div>
          {/* Shown here only when the editable field below is hidden, so the
              same figure never appears twice. */}
          {!state.advanced && (
            <div className="flex justify-between gap-2">
              <dt className="text-[var(--text-3)]">{t('host.baseOverhead')}</dt>
              <dd className="font-semibold tabular-nums text-[var(--text-2)]">
                {num(state.hostBaseOverheadW, language, 0)} W
              </dd>
            </div>
          )}
        </dl>

        {state.advanced && (
          <Field label={t('host.baseOverhead')} help={t('host.baseOverheadHelp')}>
            <NumberInput
              value={state.hostBaseOverheadW}
              onChange={(v) => state.set('hostBaseOverheadW', v ?? 0)}
              min={0}
              step={5}
              suffix="W"
            />
          </Field>
        )}

        {idlePower.idleW > 20 && (
          <Note tone="warn">
            {t('host.idleDominatesNote', {
              watts: num(idlePower.idleW, language, 0),
              modules: state.ramModules,
            })}
          </Note>
        )}
      </div>
    </Card>
  );
}
