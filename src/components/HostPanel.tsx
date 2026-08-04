import { hostBaseOverheadW, ramBandwidthGBs, ramPower } from '../core/host';
import { BOARDS, COOLING, CPUS, DRIVES, getCpu } from '../data/cpu';
import { CHANNEL_OPTIONS, getRamType, HOST_PRESETS, RAM_TYPES } from '../data/ram';
import { useLanguage, useT } from '../i18n';
import { hostComponents, hostSpec, useStore } from '../state/store';
import { num } from '../ui/format';
import { Button, Card, Field, NumberInput, Note, Select } from '../ui/primitives';

const CPU_SEGMENTS = ['mobile', 'desktop', 'workstation', 'server'] as const;

/**
 * Host system: CPU, memory, board, cooling and drives.
 *
 * Every field here exists because a single overhead number cannot span a
 * laptop and a dual-socket 1U server. Two Xeon sockets idle above 170 W before
 * anything else is counted, and rack cooling adds nearly as much again — none
 * of which a flat figure can express.
 */
export function HostPanel() {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const state = useStore();

  const host = hostSpec(state);
  const parts = hostComponents(state);
  const ramType = getRamType(state.ramTypeId);
  const bandwidth = ramBandwidthGBs(host.ram);
  const idlePower = ramPower(host.ram, ramType, 0);
  const fullPower = ramPower(host.ram, ramType, 1);
  const computed = hostBaseOverheadW(parts);
  const overridden = state.hostBaseOverheadOverrideW !== null;

  function applyPreset(id: string) {
    const preset = HOST_PRESETS.find((p) => p.id === id);
    if (!preset) return;
    const byPreset: Record<string, Partial<typeof state>> = {
      laptop: { cpuId: 'mobile-u', cpuSockets: 1, boardId: 'laptop', coolingId: 'laptop', driveCount: 1, ramChannels: 2 },
      desktop: { cpuId: 'ryzen7-9700x', cpuSockets: 1, boardId: 'desktop', coolingId: 'desktop-air', driveCount: 1, ramChannels: 2 },
      workstation: { cpuId: 'threadripper-7960x', cpuSockets: 1, boardId: 'workstation', coolingId: 'desktop-highflow', driveCount: 2, ramChannels: 4 },
      server: { cpuId: 'epyc-9354', cpuSockets: 2, boardId: 'server', coolingId: 'server-2u', driveCount: 4, ramChannels: 8 },
    };
    state.patch({
      hostPresetId: id,
      ramModules: preset.ramModules,
      ramCapacityGb: preset.ramCapacityGb,
      // A preset means wanting the preset — drop any manual override.
      hostBaseOverheadOverrideW: null,
      ...byPreset[id],
    });
  }

  const row = (label: string, value: string) => (
    <div className="flex justify-between gap-2">
      <dt className="text-[var(--text-3)]">{label}</dt>
      <dd className="font-semibold tabular-nums text-[var(--text-2)]">{value}</dd>
    </div>
  );

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

        {/* ---------------------------------------------------------- CPU */}
        <div className="grid grid-cols-[1fr_auto] gap-3">
          <Field label={t('host.cpu')}>
            <Select value={state.cpuId} onChange={(v) => state.set('cpuId', v)}>
              {CPU_SEGMENTS.map((seg) => (
                <optgroup key={seg} label={t(`host.segment.${seg}`)}>
                  {CPUS.filter((c) => c.segment === seg).map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.label} — {c.cores}C, {c.idleW} W idle
                    </option>
                  ))}
                </optgroup>
              ))}
            </Select>
          </Field>
          <Field label={t('host.sockets')}>
            <Select
              value={String(state.cpuSockets)}
              onChange={(v) => state.set('cpuSockets', Number(v))}
            >
              {[1, 2, 4, 8].map((n) => (
                <option key={n} value={n}>
                  {n}×
                </option>
              ))}
            </Select>
          </Field>
        </div>

        {/* --------------------------------------------------------- RAM */}
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

        {/* --------------------------------------------- chassis & drives */}
        <div className="grid grid-cols-2 gap-3">
          <Field label={t('host.cooling')} help={t('host.coolingHelp')}>
            <Select value={state.coolingId} onChange={(v) => state.set('coolingId', v)}>
              {COOLING.map((c) => (
                <option key={c.id} value={c.id}>
                  {t(c.labelKey)} — {num(c.heatFractionOfLoad * 100, language, 1)} %
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t('host.board')}>
            <Select value={state.boardId} onChange={(v) => state.set('boardId', v)}>
              {BOARDS.map((b) => (
                <option key={b.id} value={b.id}>
                  {t(b.labelKey)} — {b.watts} W
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <div className="grid grid-cols-[1fr_auto] gap-3">
          <Field label={t('host.drives')}>
            <Select value={state.driveId} onChange={(v) => state.set('driveId', v)}>
              {DRIVES.map((d) => (
                <option key={d.id} value={d.id}>
                  {t(d.labelKey)} — {d.idleW} W
                </option>
              ))}
            </Select>
          </Field>
          <Field label={t('host.driveCount')}>
            <NumberInput
              value={state.driveCount}
              onChange={(v) => state.set('driveCount', Math.max(0, v ?? 0))}
              min={0}
              step={1}
            />
          </Field>
        </div>

        {/* ------------------------------------------------------ summary */}
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 rounded-[var(--radius-sm)] bg-[var(--surface-2)] p-2.5 text-[11px]">
          {row(t('host.bandwidth'), `${num(bandwidth, language, 1)} GB/s`)}
          {row(
            `${t('host.cpu')} ${state.cpuSockets > 1 ? `${state.cpuSockets}×` : ''}`,
            `${num(parts.cpuIdleW * parts.sockets, language, 0)} W`,
          )}
          {row(t('host.ramIdlePower'), `${num(idlePower.idleW, language, 1)} W`)}
          {row(t('host.ramFullPower'), `${num(fullPower.totalW, language, 1)} W`)}
          {row(t('host.drives'), `${num(parts.drivesW, language, 0)} W`)}
          {row(t('host.board'), `${num(parts.boardW, language, 0)} W`)}
        </dl>

        <div className="flex items-baseline justify-between gap-2 text-[12px]">
          <span className="font-medium text-[var(--text-2)]">{t('host.baseOverhead')}</span>
          <span className="font-semibold tabular-nums text-[var(--text)]">
            {num(host.baseOverheadW, language, 0)} W
          </span>
        </div>

        {state.advanced && (
          <Field
            label={t('host.overrideBase')}
            help={t('host.overrideBaseHelp')}
            hint={
              overridden ? (
                <button
                  type="button"
                  className="cursor-pointer text-[11px] font-medium text-[var(--accent)] hover:underline"
                  onClick={() => state.set('hostBaseOverheadOverrideW', null)}
                >
                  {t('cost.resetPreset')}
                </button>
              ) : undefined
            }
          >
            <NumberInput
              value={host.baseOverheadW}
              onChange={(v) =>
                state.set('hostBaseOverheadOverrideW', v === computed ? null : v)
              }
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

        {getCpu(state.cpuId).segment === 'server' && state.cpuSockets > 1 && (
          <Note tone="warn">
            {t('host.serverIdleNote', {
              watts: num(host.baseOverheadW + idlePower.idleW, language, 0),
            })}
          </Note>
        )}
      </div>
    </Card>
  );
}

export { Button };
