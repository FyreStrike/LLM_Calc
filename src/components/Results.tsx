import { GB } from '../core/memory';
import type { CalcResult, ModelSpec, Warning } from '../core/types';
import { getGpu, GPUS } from '../data/gpus';
import { useLanguage, useT } from '../i18n';
import { selectedModel, useStore } from '../state/store';
import { bytes, duration, money, num, tokens } from '../ui/format';
import { Badge, Card, Divider, Note, Stat, Swatch } from '../ui/primitives';

// Fixed categorical order — never cycled, never reassigned by rank.
const MEMORY_SERIES = [
  { key: 'weightsBytes', color: 'var(--series-1)', label: 'results.weights' },
  { key: 'kvCacheBytes', color: 'var(--series-2)', label: 'results.kvCache' },
  { key: 'activationsBytes', color: 'var(--series-3)', label: 'results.activations' },
  { key: 'cudaContextBytes', color: 'var(--series-4)', label: 'results.cudaContext' },
  { key: 'frameworkOverheadBytes', color: 'var(--series-5)', label: 'results.framework' },
] as const;

export function Results({ result }: { result: CalcResult }) {
  const advanced = useStore((s) => s.advanced);

  return (
    <div className="space-y-4">
      <Verdict result={result} />
      <div className="grid gap-4 xl:grid-cols-2">
        <MemoryCard result={result} />
        <CostCard result={result} />
      </div>
      {advanced && (
        <div className="grid gap-4 xl:grid-cols-2">
          <EnergyCard result={result} />
          <RooflineCard result={result} />
        </div>
      )}
      <Warnings warnings={result.warnings} />
    </div>
  );
}

/* ------------------------------------------------------------------- verdict */

/**
 * The headline panel. Three questions get answered before any scrolling:
 * does it fit, how fast is it, and is it cheaper than the API.
 */
function Verdict({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const state = useStore();
  const model = selectedModel(state);
  const gpu = getGpu(state.gpuId) ?? GPUS[0];

  const { memory, usableVramBytes, fits, utilizationPct, performance, cost } = result;
  const pct = Math.min(100, utilizationPct);
  const tone = !fits ? 'bad' : utilizationPct > 88 ? 'warn' : 'good';
  const toneVar = { good: 'var(--good)', warn: 'var(--warn)', bad: 'var(--bad)' }[tone];

  const hasApi = cost.apiPerMTokens !== undefined;
  const localCheaper = (cost.savingsPerMTokens ?? 0) > 0;

  return (
    <section className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-md)]">
      {/* Status strip carries the colour, so the cards below stay calm. */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-5 pt-4 pb-3">
        <span
          className="inline-flex items-center gap-2 text-[15px] font-semibold tracking-tight"
          style={{ color: toneVar }}
        >
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: toneVar }}
          />
          {fits ? t('results.fits') : t('results.doesNotFit')}
        </span>
        <span className="text-[12px] text-[var(--text-3)]">
          {model.name} · {state.quantId.toUpperCase()} ·{' '}
          {state.numGpus > 1 ? `${state.numGpus}× ` : ''}
          {gpu.name}
        </span>
      </div>

      <Divider />

      <div className="grid gap-px bg-[var(--border)] sm:grid-cols-3">
        {/* memory */}
        <div className="bg-[var(--surface)] px-5 py-4">
          <Stat
            label={t('results.vramUsage')}
            value={`${num(memory.totalBytes / GB, language, 1)} GB`}
            sub={`${t('results.of')} ${num(usableVramBytes / GB, language, 0)} GB · ${num(
              utilizationPct,
              language,
              0,
            )} %`}
            tone={tone}
            size="lg"
          />
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-[var(--surface-3)]">
            <div
              className="h-full rounded-full transition-[width] duration-300"
              style={{ width: `${pct}%`, background: toneVar }}
            />
          </div>
        </div>

        {/* speed */}
        <div className="bg-[var(--surface)] px-5 py-4">
          <Stat
            label={t('results.generationSpeed')}
            value={num(performance.decodeTokensPerSecPerSequence, language, 1)}
            unit={t('units.tokensPerSec')}
            // Composed from units rather than a lowercased sentence: German
            // capitalises nouns, so `t('results.latency').toLowerCase()` would
            // render "latenz pro token".
            sub={`${num(performance.msPerToken, language, 1)} ms/Token · TTFT ${num(
              performance.ttftMs,
              language,
              0,
            )} ms`}
            size="lg"
          />
          {state.batchSize > 1 && (
            <div className="mt-2 text-[11px] text-[var(--text-3)]">
              {t('results.totalThroughput')}:{' '}
              <span className="font-semibold tabular-nums text-[var(--text-2)]">
                {num(performance.decodeTokensPerSecTotal, language, 0)}{' '}
                {t('units.tokensPerSec')}
              </span>
            </div>
          )}
        </div>

        {/* cost */}
        <div className="bg-[var(--surface)] px-5 py-4">
          <Stat
            label={t('results.localCost')}
            value={money(cost.localElectricityPerMTokens, cost.currency, language, 3)}
            sub={t('results.perMTokens')}
            size="lg"
          />
          {hasApi && (
            <div className="mt-2 flex items-center gap-1.5 text-[11px]">
              <Badge tone={localCheaper ? 'good' : 'warn'}>
                {localCheaper ? '↓' : '↑'}{' '}
                {money(Math.abs(cost.savingsPerMTokens!), cost.currency, language, 3)}
              </Badge>
              <span className="text-[var(--text-3)]">
                {t('results.apiCost')}{' '}
                {money(cost.apiPerMTokens!, cost.currency, language, 3)}
              </span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------- memory */

function MemoryCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const { memory, usableVramBytes } = result;

  const scale = Math.max(memory.totalBytes, usableVramBytes);
  const freeBytes = Math.max(0, usableVramBytes - memory.totalBytes);

  return (
    <Card title={t('results.memoryBreakdown')}>
      <div
        className="flex h-7 w-full gap-[2px] overflow-hidden rounded-[6px]"
        role="img"
        aria-label={t('results.memoryBreakdown')}
      >
        {MEMORY_SERIES.map((s) => {
          const pct = (memory[s.key] / scale) * 100;
          if (pct < 0.4) return null;
          return (
            <div
              key={s.key}
              style={{ width: `${pct}%`, background: s.color }}
              className="transition-[width] duration-300"
              title={t(s.label)}
            />
          );
        })}
        {freeBytes > 0 && (
          <div
            style={{ width: `${(freeBytes / scale) * 100}%` }}
            className="bg-[var(--surface-3)]"
          />
        )}
      </div>

      {/* Table view — light-mode contrast on slots 3-5 sits below 3:1, so
          identity must never rest on colour alone. */}
      <table className="mt-3 w-full text-[12px]">
        <tbody>
          {MEMORY_SERIES.map((s) => (
            <tr key={s.key} className="border-t border-[var(--border)] first:border-0">
              <td className="py-1.5">
                <Swatch color={s.color} />
                <span className="text-[var(--text-2)]">{t(s.label)}</span>
              </td>
              <td className="py-1.5 text-right font-semibold tabular-nums text-[var(--text)]">
                {bytes(memory[s.key], language)}
              </td>
              <td className="w-14 py-1.5 text-right tabular-nums text-[var(--text-3)]">
                {num((memory[s.key] / memory.totalBytes) * 100, language, 1)} %
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

/* ---------------------------------------------------------------------- cost */

function CostCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const dailyTokens = useStore((s) => s.dailyTokens);
  const { cost, energy } = result;

  const hasApi = cost.apiPerMTokens !== undefined;
  const localCheaper = (cost.savingsPerMTokens ?? 0) > 0;
  const max = Math.max(cost.localElectricityPerMTokens, cost.apiPerMTokens ?? 0, 1e-9);
  const dailyCost = (cost.localElectricityPerMTokens * dailyTokens) / 1e6;

  return (
    <Card title={t('results.energyAndCost')}>
      <div className="space-y-2.5">
        <CostBar
          label={t('results.localCost')}
          value={cost.localElectricityPerMTokens}
          max={max}
          color="var(--series-1)"
          currency={cost.currency}
          language={language}
        />
        {hasApi && (
          <CostBar
            label={t('results.apiCost')}
            value={cost.apiPerMTokens!}
            max={max}
            color="var(--series-2)"
            currency={cost.currency}
            language={language}
          />
        )}
      </div>

      {hasApi && (
        <div className="mt-3">
          <Note tone={localCheaper ? 'info' : 'warn'}>
            {t(localCheaper ? 'results.cheaperLocal' : 'results.cheaperApi', {
              amount: money(Math.abs(cost.savingsPerMTokens!), cost.currency, language, 3),
            })}
          </Note>
        </div>
      )}

      <div className="mt-4 grid grid-cols-2 gap-4 border-t border-[var(--border)] pt-3.5 sm:grid-cols-4">
        <Stat
          label={t('energy.powerDraw')}
          value={num(energy.wallPowerW, language, 0)}
          unit={t('units.watt')}
        />
        <Stat
          label={t('energy.joulesPerToken')}
          value={num(energy.wallJoulesPerToken, language, 2)}
          unit="J"
        />
        <Stat
          label={t('results.perDay')}
          value={money(dailyCost, cost.currency, language, 2)}
          sub={`${tokens(dailyTokens, language)} tok`}
        />
        <Stat
          label={t('results.co2')}
          value={num(cost.co2GramsPerMTokens, language, 0)}
          unit="g"
          sub={t('results.perMTokens')}
        />
      </div>

      {cost.breakEvenTokens !== undefined ? (
        <div className="mt-3 rounded-[var(--radius-sm)] bg-[var(--surface-2)] px-2.5 py-2 text-[11px] leading-relaxed">
          <span className="font-semibold text-[var(--text-2)]">
            {t('results.breakEven')}:{' '}
          </span>
          <span className="text-[var(--text-3)]">
            {t('results.breakEvenAfter', { tokens: tokens(cost.breakEvenTokens, language) })}
            {cost.breakEvenDays !== undefined &&
              ` — ${t('results.breakEvenDays', {
                days: duration(cost.breakEvenDays, language),
                daily: tokens(dailyTokens, language),
              })}`}
          </span>
        </div>
      ) : (
        hasApi &&
        !localCheaper && (
          <p className="mt-3 text-[11px] text-[var(--text-3)]">{t('results.breakEvenNever')}</p>
        )
      )}
    </Card>
  );
}

function CostBar({
  label,
  value,
  max,
  color,
  currency,
  language,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  currency: 'EUR' | 'USD';
  language: 'de' | 'en';
}) {
  const t = useT();
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-[12px]">
        <span className="flex items-center text-[var(--text-2)]">
          <Swatch color={color} />
          {label}
        </span>
        <span className="font-semibold tabular-nums text-[var(--text)]">
          {money(value, currency, language, 3)}
          <span className="ml-1 font-normal text-[var(--text-3)]">
            {t('results.perMTokens')}
          </span>
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-[var(--surface-3)]">
        <div
          className="h-full rounded-full transition-[width] duration-300"
          style={{ width: `${Math.max(1.5, (value / max) * 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------- energy */

function EnergyCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const { energy } = result;
  const ramDutyPct = result.performance.ramDutyCycle * 100;
  const d = energy.decomposition;
  const totalJ = d ? d.computeJoules + d.memoryJoules + d.staticJoules : 0;

  const parts = d
    ? [
        { j: d.computeJoules, color: 'var(--series-1)', label: t('energy.compute') },
        { j: d.memoryJoules, color: 'var(--series-2)', label: t('energy.memory') },
        { j: d.staticJoules, color: 'var(--series-3)', label: t('energy.static') },
      ]
    : [];

  return (
    <Card title={t('section.energy')}>
      <div className="grid grid-cols-3 gap-4">
        <Stat
          label={`GPU · ${t('results.decode')}`}
          value={num(energy.decodePowerW, language, 0)}
          unit="W"
        />
        <Stat
          label={`GPU · ${t('results.prefill')}`}
          value={num(energy.prefillPowerW, language, 0)}
          unit="W"
        />
        <Stat
          label={t('energy.wallPower')}
          value={num(energy.wallPowerW, language, 0)}
          unit="W"
        />
      </div>

      {/* Host draw broken out, so the standing cost of installed memory is
          visible rather than folded into one overhead number. */}
      <div className="mt-4 border-t border-[var(--border)] pt-3.5">
        <h3 className="mb-2 text-[11px] font-medium tracking-wide text-[var(--text-3)] uppercase">
          {t('section.host')}
        </h3>
        <table className="w-full text-[12px]">
          <tbody>
            <tr>
              <td className="py-1 text-[var(--text-2)]">{t('host.baseOverhead')}</td>
              <td className="py-1 text-right font-semibold tabular-nums text-[var(--text)]">
                {num(energy.host.baseW, language, 0)} W
              </td>
            </tr>
            <tr>
              <td className="py-1 text-[var(--text-2)]">{t('host.ramIdlePower')}</td>
              <td className="py-1 text-right font-semibold tabular-nums text-[var(--text)]">
                {num(energy.host.ramIdleW, language, 1)} W
              </td>
            </tr>
            {energy.host.ramActiveW > 0.05 && (
              <tr>
                <td className="py-1 text-[var(--text-2)]">
                  {t('host.ramFullPower')} ({num(ramDutyPct, language, 0)} %)
                </td>
                <td className="py-1 text-right font-semibold tabular-nums text-[var(--text)]">
                  +{num(energy.host.ramActiveW, language, 1)} W
                </td>
              </tr>
            )}
            <tr className="border-t border-[var(--border)]">
              <td className="py-1 font-medium text-[var(--text-2)]">
                {t('energy.hostOverhead')}
              </td>
              <td className="py-1 text-right font-semibold tabular-nums text-[var(--text)]">
                {num(energy.host.totalW, language, 0)} W
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {d && totalJ > 0 && (
        <div className="mt-4 border-t border-[var(--border)] pt-3.5">
          <h3 className="mb-2 font-mono text-[11px] text-[var(--text-3)]">
            E = ε_flop·W + ε_mop·Q + π₀·T
          </h3>
          <div className="flex h-6 w-full gap-[2px] overflow-hidden rounded-[6px]">
            {parts.map((x) => (
              <div
                key={x.label}
                style={{ width: `${(x.j / totalJ) * 100}%`, background: x.color }}
                title={x.label}
              />
            ))}
          </div>
          <table className="mt-2 w-full text-[12px]">
            <tbody>
              {parts.map((x) => (
                <tr key={x.label}>
                  <td className="py-1">
                    <Swatch color={x.color} />
                    <span className="text-[var(--text-2)]">{x.label}</span>
                  </td>
                  <td className="py-1 text-right font-semibold tabular-nums text-[var(--text)]">
                    {num((x.j / totalJ) * 100, language, 1)} %
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------ roofline */

function RooflineCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const p = result.performance;

  const rows: { label: string; color?: string; intensity: string; bound?: 'memory' | 'compute' }[] =
    [
      {
        label: t('results.prefill'),
        color: 'var(--series-1)',
        intensity: num(p.prefillIntensity, language, 0),
        bound: p.prefillBound,
      },
      {
        label: t('results.decode'),
        color: 'var(--series-2)',
        intensity: num(p.decodeIntensity, language, 2),
        bound: p.decodeBound,
      },
    ];

  return (
    <Card title={t('results.roofline')}>
      <RooflinePlot result={result} />

      <table className="mt-2 w-full text-[12px]">
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-t border-[var(--border)] first:border-0">
              <td className="py-1.5">
                {r.color && <Swatch color={r.color} />}
                <span className="text-[var(--text-2)]">{r.label}</span>
              </td>
              <td className="py-1.5 text-right font-semibold tabular-nums text-[var(--text)]">
                {r.intensity}
              </td>
              <td className="py-1.5 pl-2 text-right">
                <Badge tone={r.bound === 'compute' ? 'info' : 'warn'}>
                  {t(r.bound === 'compute' ? 'results.computeBound' : 'results.memoryBound')}
                </Badge>
              </td>
            </tr>
          ))}
          <tr className="border-t border-[var(--border)]">
            <td className="py-1.5 text-[var(--text-3)]">{t('results.ridgePoint')} (F/B)</td>
            <td className="py-1.5 text-right tabular-nums text-[var(--text-2)]">
              {num(p.ridgePoint, language, 0)}
            </td>
            <td className="py-1.5 pl-2 text-right text-[11px] text-[var(--text-3)]">
              {t('units.flopPerByte')}
            </td>
          </tr>
          <tr>
            <td className="py-1.5 text-[var(--text-3)]">{t('results.effectiveBandwidth')}</td>
            <td className="py-1.5 text-right tabular-nums text-[var(--text-2)]">
              {num(p.effectiveBandwidthBytesPerSec / 1e9, language, 0)}
            </td>
            <td className="py-1.5 pl-2 text-right text-[11px] text-[var(--text-3)]">GB/s</td>
          </tr>
          <tr>
            <td className="py-1.5 text-[var(--text-3)]">{t('results.bytesPerToken')}</td>
            <td
              className="py-1.5 text-right tabular-nums text-[var(--text-2)]"
              colSpan={2}
            >
              {bytes(p.bytesPerDecodeStep, language)}
            </td>
          </tr>
        </tbody>
      </table>
    </Card>
  );
}

/**
 * Log-log roofline: a bandwidth-limited diagonal meeting a compute ceiling at
 * the ridge point, with both inference phases plotted on it. Raw SVG because
 * both axes are logarithmic and the roof segments must meet exactly at F/B.
 */
function RooflinePlot({ result }: { result: CalcResult }) {
  const t = useT();
  const p = result.performance;

  const W = 520;
  const H = 190;
  const PAD = { l: 34, r: 12, t: 14, b: 26 };

  const peakFlops = p.effectiveFlops;
  const bw = p.effectiveBandwidthBytesPerSec;

  const xMin = 0.05;
  const xMax = Math.max(p.prefillIntensity * 4, p.ridgePoint * 8, 1e4);
  const yMax = peakFlops * 2;
  const yMin = Math.max(peakFlops / 1e6, bw * xMin * 0.5);

  const lx = (v: number) =>
    PAD.l +
    ((Math.log10(Math.max(v, xMin)) - Math.log10(xMin)) /
      (Math.log10(xMax) - Math.log10(xMin))) *
      (W - PAD.l - PAD.r);
  const ly = (v: number) =>
    H -
    PAD.b -
    ((Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) /
      (Math.log10(yMax) - Math.log10(yMin))) *
      (H - PAD.t - PAD.b);

  const attain = (i: number) => Math.min(peakFlops, bw * i);
  const ridge = p.ridgePoint;

  const decades: number[] = [];
  for (let e = Math.ceil(Math.log10(xMin)); e <= Math.log10(xMax); e++) decades.push(10 ** e);

  const points = [
    { pt: { x: p.prefillIntensity, y: attain(p.prefillIntensity) }, color: 'var(--series-1)', label: t('results.prefill') },
    { pt: { x: p.decodeIntensity, y: attain(p.decodeIntensity) }, color: 'var(--series-2)', label: t('results.decode') },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={t('results.roofline')}>
      {decades.map((d) => (
        <line key={d} x1={lx(d)} x2={lx(d)} y1={PAD.t} y2={H - PAD.b} stroke="var(--grid)" strokeWidth={1} />
      ))}
      <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} stroke="var(--axis)" strokeWidth={1} />

      {/* Shade the bandwidth-bound region so the two regimes read at a glance. */}
      <rect
        x={PAD.l}
        y={PAD.t}
        width={Math.max(0, lx(ridge) - PAD.l)}
        height={H - PAD.b - PAD.t}
        fill="var(--accent)"
        opacity={0.04}
      />

      <polyline
        points={`${lx(xMin)},${ly(bw * xMin)} ${lx(ridge)},${ly(peakFlops)} ${lx(xMax)},${ly(peakFlops)}`}
        fill="none"
        stroke="var(--text-3)"
        strokeWidth={2}
        strokeLinejoin="round"
      />
      <line
        x1={lx(ridge)}
        x2={lx(ridge)}
        y1={PAD.t}
        y2={H - PAD.b}
        stroke="var(--axis)"
        strokeWidth={1}
        strokeDasharray="3 4"
      />

      {points.map((s) => {
        const cx = lx(s.pt.x);
        const cy = ly(s.pt.y);
        const above = cy - 10 > PAD.t + 8;
        const anchor = cx > W - PAD.r - 34 ? 'end' : cx < PAD.l + 34 ? 'start' : 'middle';
        return (
          <g key={s.label}>
            <circle cx={cx} cy={cy} r={5} fill={s.color} stroke="var(--surface-ring)" strokeWidth={2} />
            <text
              x={cx}
              y={above ? cy - 11 : cy + 18}
              textAnchor={anchor}
              fontSize={10}
              fontWeight={600}
              fill="var(--text-2)"
            >
              {s.label}
            </text>
          </g>
        );
      })}

      {decades.map((d) => (
        <text key={`t${d}`} x={lx(d)} y={H - PAD.b + 13} textAnchor="middle" fontSize={9} fill="var(--muted)">
          {d >= 1 ? d.toLocaleString('en-US', { notation: 'compact' }) : d}
        </text>
      ))}
      <text x={W - PAD.r} y={H - 3} textAnchor="end" fontSize={9} fill="var(--muted)">
        {t('units.flopPerByte')}
      </text>
      <text x={2} y={PAD.t - 4} fontSize={9} fill="var(--muted)">
        FLOP/s
      </text>
    </svg>
  );
}

/* ------------------------------------------------------------------ warnings */

function Warnings({ warnings }: { warnings: Warning[] }) {
  const t = useT();
  if (warnings.length === 0) return null;

  const order = { error: 0, warn: 1, info: 2 };
  const sorted = [...warnings].sort((a, b) => order[a.level] - order[b.level]);

  return (
    <div className="space-y-1.5">
      {sorted.map((w, i) => (
        <Note
          key={`${w.key}-${i}`}
          tone={w.level === 'error' ? 'bad' : w.level === 'warn' ? 'warn' : 'neutral'}
        >
          {t(w.key, w.values)}
        </Note>
      ))}
    </div>
  );
}

export type { ModelSpec };
