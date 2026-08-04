import type { CalcResult, Warning } from '../core/types';
import { GB } from '../core/memory';
import { useLanguage, useT } from '../i18n';
import { useStore } from '../state/store';
import { bytes, contextLabel, money, num, tokens, duration } from '../ui/format';
import { Badge, Card, Stat } from '../ui/primitives';

// Fixed categorical order — never cycled, never reassigned by rank.
const MEMORY_SERIES = [
  { key: 'weightsBytes', color: 'var(--series-1)', label: 'results.weights' },
  { key: 'kvCacheBytes', color: 'var(--series-2)', label: 'results.kvCache' },
  { key: 'activationsBytes', color: 'var(--series-3)', label: 'results.activations' },
  { key: 'cudaContextBytes', color: 'var(--series-4)', label: 'results.cudaContext' },
  { key: 'frameworkOverheadBytes', color: 'var(--series-5)', label: 'results.framework' },
] as const;

export function Results({ result }: { result: CalcResult }) {
  return (
    <div className="space-y-4">
      <VramCard result={result} />
      <PerformanceCard result={result} />
      <CostCompareCard result={result} />
      <EnergyCard result={result} />
      <RooflineCard result={result} />
      <Warnings warnings={result.warnings} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// VRAM
// ---------------------------------------------------------------------------

function VramCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const { memory, usableVramBytes, fits, utilizationPct } = result;

  const scale = Math.max(memory.totalBytes, usableVramBytes);
  const freeBytes = Math.max(0, usableVramBytes - memory.totalBytes);

  const tone = !fits ? 'bad' : utilizationPct > 85 ? 'warn' : 'good';

  return (
    <Card
      title={t('results.vramUsage')}
      right={
        <Badge tone={fits ? 'good' : 'bad'}>
          {fits ? t('results.fits') : t('results.doesNotFit')}
        </Badge>
      }
    >
      <div className="mb-3 flex items-baseline gap-3">
        <Stat
          label=""
          value={`${num(memory.totalBytes / GB, language, 1)} GB`}
          sub={`${t('results.of')} ${num(usableVramBytes / GB, language, 1)} GB — ${num(
            utilizationPct,
            language,
            0,
          )}%`}
          tone={tone}
        />
      </div>

      {/* Stacked bar. 2px surface gaps between segments; rounded data-ends. */}
      <div
        className="flex h-6 w-full gap-[2px] overflow-hidden rounded"
        role="img"
        aria-label={t('results.memoryBreakdown')}
      >
        {MEMORY_SERIES.map((s) => {
          const value = memory[s.key];
          const pct = (value / scale) * 100;
          if (pct < 0.15) return null;
          return (
            <div
              key={s.key}
              style={{ width: `${pct}%`, background: s.color }}
              className="first:rounded-l last:rounded-r"
            />
          );
        })}
        {freeBytes > 0 && (
          <div
            style={{ width: `${(freeBytes / scale) * 100}%` }}
            className="rounded-r bg-slate-200 dark:bg-slate-700"
          />
        )}
      </div>

      {/* The table view — light-mode contrast on slots 3-5 requires that
          identity never rest on color alone. */}
      <table className="mt-3 w-full text-xs">
        <tbody>
          {MEMORY_SERIES.map((s) => {
            const value = memory[s.key];
            return (
              <tr key={s.key}>
                <td className="py-0.5">
                  <span
                    className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                    style={{ background: s.color }}
                  />
                  <span className="text-slate-600 dark:text-slate-300">{t(s.label)}</span>
                </td>
                <td className="py-0.5 text-right font-medium tabular-nums text-slate-800 dark:text-slate-100">
                  {bytes(value, language)}
                </td>
                <td className="w-12 py-0.5 text-right tabular-nums text-slate-500 dark:text-slate-400">
                  {num((value / memory.totalBytes) * 100, language, 1)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

function PerformanceCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const batchSize = useStore((s) => s.batchSize);
  const p = result.performance;

  return (
    <Card title={t('results.performance')}>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat
          label={t('results.generationSpeed')}
          value={num(p.decodeTokensPerSecPerSequence, language, 1)}
          unit={t('units.tokensPerSec')}
          sub={t('results.perUser')}
        />
        {batchSize > 1 && (
          <Stat
            label={t('results.totalThroughput')}
            value={num(p.decodeTokensPerSecTotal, language, 0)}
            unit={t('units.tokensPerSec')}
          />
        )}
        <Stat
          label={t('results.ttft')}
          value={num(p.ttftMs, language, p.ttftMs < 100 ? 1 : 0)}
          unit={t('units.ms')}
        />
        <Stat
          label={t('results.latency')}
          value={num(p.msPerToken, language, 1)}
          unit={t('units.ms')}
        />
      </div>

      {p.offloadFraction > 0 && (
        <p className="mt-3 text-xs text-amber-700 dark:text-amber-400">
          {t('warn.offloading', { percent: num(p.offloadFraction * 100, language, 0) })}
        </p>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Cost comparison — the headline feature
// ---------------------------------------------------------------------------

function CostCompareCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const dailyTokens = useStore((s) => s.dailyTokens);
  const { cost, energy } = result;
  const currency = cost.currency;

  const hasApi = cost.apiPerMTokens !== undefined;
  const localCheaper = (cost.savingsPerMTokens ?? 0) > 0;
  const max = Math.max(cost.localElectricityPerMTokens, cost.apiPerMTokens ?? 0, 1e-9);

  const dailyCost = (cost.localElectricityPerMTokens * dailyTokens) / 1e6;

  return (
    <Card title={t('results.energyAndCost')}>
      <div className="space-y-3">
        {/* Two bars, categorical slots 1 and 2, each directly labelled. */}
        <div className="space-y-2">
          <CostBar
            label={t('results.localCost')}
            value={cost.localElectricityPerMTokens}
            max={max}
            color="var(--series-1)"
            currency={currency}
            language={language}
          />
          {hasApi && (
            <CostBar
              label={t('results.apiCost')}
              value={cost.apiPerMTokens!}
              max={max}
              color="var(--series-2)"
              currency={currency}
              language={language}
            />
          )}
        </div>

        {hasApi && (
          <p
            className={`rounded-lg p-2 text-sm font-medium ${
              localCheaper
                ? 'bg-emerald-50 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300'
                : 'bg-amber-50 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300'
            }`}
          >
            {t(localCheaper ? 'results.cheaperLocal' : 'results.cheaperApi', {
              amount: money(Math.abs(cost.savingsPerMTokens!), currency, language, 3),
            })}
          </p>
        )}

        <div className="grid grid-cols-2 gap-4 border-t border-slate-200 pt-3 sm:grid-cols-4 dark:border-slate-700">
          <Stat
            label={t('energy.powerDraw')}
            value={num(energy.wallPowerW, language, 0)}
            unit={t('units.watt')}
          />
          <Stat
            label={t('energy.joulesPerToken')}
            value={num(energy.wallJoulesPerToken, language, 3)}
            unit={t('units.joulePerToken')}
          />
          <Stat
            label={t('results.perDay')}
            value={money(dailyCost, currency, language, 2)}
            sub={`${tokens(dailyTokens, language)} tok`}
          />
          <Stat
            label={t('results.co2PerMTokens')}
            value={num(cost.co2GramsPerMTokens, language, 1)}
            unit={t('units.gramCo2')}
          />
        </div>

        {cost.breakEvenTokens !== undefined ? (
          <div className="rounded-lg bg-slate-50 p-2 text-xs dark:bg-slate-800/60">
            <span className="font-medium text-slate-700 dark:text-slate-200">
              {t('results.breakEven')}:{' '}
            </span>
            <span className="text-slate-600 dark:text-slate-300">
              {t('results.breakEvenAfter', {
                tokens: tokens(cost.breakEvenTokens, language),
              })}
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
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {t('results.breakEvenNever')}
            </p>
          )
        )}
      </div>
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
      <div className="mb-0.5 flex items-baseline justify-between text-xs">
        <span className="text-slate-600 dark:text-slate-300">{label}</span>
        <span className="font-semibold tabular-nums text-slate-900 dark:text-slate-50">
          {money(value, currency, language, 3)}{' '}
          <span className="font-normal text-slate-500">{t('results.perMTokens')}</span>
        </span>
      </div>
      <div className="h-3 w-full overflow-hidden rounded bg-slate-100 dark:bg-slate-800">
        <div
          className="h-full rounded"
          style={{ width: `${Math.max(1, (value / max) * 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Energy detail
// ---------------------------------------------------------------------------

function EnergyCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const advanced = useStore((s) => s.advanced);
  const { energy } = result;

  if (!advanced) return null;

  const d = energy.decomposition;
  const totalJ = d ? d.computeJoules + d.memoryJoules + d.staticJoules : 0;

  return (
    <Card title={t('section.energy')}>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <Stat
          label={`${t('energy.powerDraw')} (${t('results.decode')})`}
          value={num(energy.decodePowerW, language, 0)}
          unit={t('units.watt')}
        />
        <Stat
          label={`${t('energy.powerDraw')} (${t('results.prefill')})`}
          value={num(energy.prefillPowerW, language, 0)}
          unit={t('units.watt')}
        />
        <Stat
          label={t('energy.wallPower')}
          value={num(energy.wallPowerW, language, 0)}
          unit={t('units.watt')}
        />
      </div>

      {d && totalJ > 0 && (
        <div className="mt-4">
          <h3 className="mb-2 text-xs font-medium text-slate-600 dark:text-slate-300">
            {t('energy.decomposition')} — E = ε_flop·W + ε_mop·Q + π₀·T
          </h3>
          <div className="flex h-5 w-full gap-[2px] overflow-hidden rounded">
            {[
              { j: d.computeJoules, color: 'var(--series-1)', label: t('energy.compute') },
              { j: d.memoryJoules, color: 'var(--series-2)', label: t('energy.memory') },
              { j: d.staticJoules, color: 'var(--series-3)', label: t('energy.static') },
            ].map((x) => (
              <div
                key={x.label}
                style={{ width: `${(x.j / totalJ) * 100}%`, background: x.color }}
                className="first:rounded-l last:rounded-r"
              />
            ))}
          </div>
          <table className="mt-2 w-full text-xs">
            <tbody>
              {[
                { j: d.computeJoules, color: 'var(--series-1)', label: t('energy.compute') },
                { j: d.memoryJoules, color: 'var(--series-2)', label: t('energy.memory') },
                { j: d.staticJoules, color: 'var(--series-3)', label: t('energy.static') },
              ].map((x) => (
                <tr key={x.label}>
                  <td className="py-0.5">
                    <span
                      className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                      style={{ background: x.color }}
                    />
                    <span className="text-slate-600 dark:text-slate-300">{x.label}</span>
                  </td>
                  <td className="py-0.5 text-right tabular-nums text-slate-500 dark:text-slate-400">
                    {num((x.j / totalJ) * 100, language, 1)}%
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

// ---------------------------------------------------------------------------
// Roofline
// ---------------------------------------------------------------------------

function RooflineCard({ result }: { result: CalcResult }) {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const advanced = useStore((s) => s.advanced);
  const p = result.performance;

  if (!advanced) return null;

  return (
    <Card title={t('results.roofline')}>
      <RooflinePlot result={result} />

      <table className="mt-3 w-full text-xs">
        <thead>
          <tr className="text-slate-500 dark:text-slate-400">
            <th className="text-left font-medium"> </th>
            <th className="text-right font-medium">{t('results.arithmeticIntensity')}</th>
            <th className="text-right font-medium">{t('results.performance')}</th>
          </tr>
        </thead>
        <tbody className="tabular-nums">
          <tr>
            <td className="py-0.5">
              <span
                className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                style={{ background: 'var(--series-1)' }}
              />
              {t('results.prefill')}
            </td>
            <td className="text-right">
              {num(p.prefillIntensity, language, 0)} {t('units.flopPerByte')}
            </td>
            <td className="text-right">
              <Badge tone={p.prefillBound === 'compute' ? 'info' : 'warn'}>
                {t(p.prefillBound === 'compute' ? 'results.computeBound' : 'results.memoryBound')}
              </Badge>
            </td>
          </tr>
          <tr>
            <td className="py-0.5">
              <span
                className="mr-2 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                style={{ background: 'var(--series-2)' }}
              />
              {t('results.decode')}
            </td>
            <td className="text-right">
              {num(p.decodeIntensity, language, 2)} {t('units.flopPerByte')}
            </td>
            <td className="text-right">
              <Badge tone={p.decodeBound === 'compute' ? 'info' : 'warn'}>
                {t(p.decodeBound === 'compute' ? 'results.computeBound' : 'results.memoryBound')}
              </Badge>
            </td>
          </tr>
          <tr className="border-t border-slate-200 dark:border-slate-700">
            <td className="py-0.5 text-slate-500 dark:text-slate-400">
              {t('results.ridgePoint')} (F/B)
            </td>
            <td className="text-right">
              {num(p.ridgePoint, language, 0)} {t('units.flopPerByte')}
            </td>
            <td />
          </tr>
          <tr>
            <td className="py-0.5 text-slate-500 dark:text-slate-400">
              {t('results.effectiveBandwidth')}
            </td>
            <td className="text-right">
              {num(p.effectiveBandwidthBytesPerSec / 1e9, language, 0)} GB/s
            </td>
            <td />
          </tr>
          <tr>
            <td className="py-0.5 text-slate-500 dark:text-slate-400">
              {t('results.bytesPerToken')}
            </td>
            <td className="text-right">{bytes(p.bytesPerDecodeStep, language)}</td>
            <td />
          </tr>
        </tbody>
      </table>
    </Card>
  );
}

/**
 * Log-log roofline: a bandwidth-limited diagonal that meets a compute ceiling
 * at the ridge point, with the two inference phases plotted on it. Drawn as
 * raw SVG because the axes are logarithmic in both dimensions and the two
 * roof segments must meet exactly at F/B.
 */
function RooflinePlot({ result }: { result: CalcResult }) {
  const t = useT();
  const p = result.performance;

  const W = 520;
  const H = 220;
  const PAD = { l: 46, r: 12, t: 12, b: 28 };

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

  const decodePoint = { x: p.decodeIntensity, y: attain(p.decodeIntensity) };
  const prefillPoint = { x: p.prefillIntensity, y: attain(p.prefillIntensity) };

  const decades = [];
  for (let e = Math.ceil(Math.log10(xMin)); e <= Math.log10(xMax); e++) {
    decades.push(10 ** e);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={t('results.roofline')}>
      {decades.map((d) => (
        <line
          key={d}
          x1={lx(d)}
          x2={lx(d)}
          y1={PAD.t}
          y2={H - PAD.b}
          stroke="var(--grid)"
          strokeWidth={1}
        />
      ))}
      <line
        x1={PAD.l}
        x2={W - PAD.r}
        y1={H - PAD.b}
        y2={H - PAD.b}
        stroke="var(--axis)"
        strokeWidth={1}
      />

      {/* The roof: bandwidth-limited diagonal, then the compute ceiling. */}
      <polyline
        points={`${lx(xMin)},${ly(bw * xMin)} ${lx(ridge)},${ly(peakFlops)} ${lx(xMax)},${ly(peakFlops)}`}
        fill="none"
        stroke="var(--muted)"
        strokeWidth={2}
      />

      {/* Ridge point marker */}
      <line
        x1={lx(ridge)}
        x2={lx(ridge)}
        y1={PAD.t}
        y2={H - PAD.b}
        stroke="var(--axis)"
        strokeWidth={1}
        strokeDasharray="3 3"
      />

      {[
        { pt: prefillPoint, color: 'var(--series-1)', label: t('results.prefill') },
        { pt: decodePoint, color: 'var(--series-2)', label: t('results.decode') },
      ].map((s) => {
        const cx = lx(s.pt.x);
        const cy = ly(s.pt.y);
        // Flip the label below the marker when it would collide with the top
        // edge, and pull it inboard near the right edge.
        const above = cy - 10 > PAD.t + 8;
        const anchor = cx > W - PAD.r - 30 ? 'end' : cx < PAD.l + 30 ? 'start' : 'middle';
        return (
          <g key={s.label}>
            <circle
              cx={cx}
              cy={cy}
              r={5}
              fill={s.color}
              stroke="var(--surface-ring, #fcfcfb)"
              strokeWidth={2}
            />
            <text
              x={cx}
              y={above ? cy - 10 : cy + 18}
              textAnchor={anchor}
              fontSize={10}
              fill="currentColor"
            >
              {s.label}
            </text>
          </g>
        );
      })}

      {decades.map((d) => (
        <text
          key={`t${d}`}
          x={lx(d)}
          y={H - PAD.b + 12}
          textAnchor="middle"
          fontSize={9}
          fill="var(--muted)"
        >
          {d >= 1 ? d.toLocaleString('en-US', { notation: 'compact' }) : d}
        </text>
      ))}
      <text x={PAD.l} y={H - 4} fontSize={9} fill="var(--muted)">
        {t('units.flopPerByte')}
      </text>
      <text
        x={4}
        y={PAD.t + 8}
        fontSize={9}
        fill="var(--muted)"
      >
        FLOP/s
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Warnings
// ---------------------------------------------------------------------------

function Warnings({ warnings }: { warnings: Warning[] }) {
  const t = useT();
  if (warnings.length === 0) return null;

  const tone = {
    error: 'bg-rose-50 text-rose-900 dark:bg-rose-950/40 dark:text-rose-200',
    warn: 'bg-amber-50 text-amber-900 dark:bg-amber-950/40 dark:text-amber-200',
    info: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
  };

  return (
    <div className="space-y-1.5">
      {warnings.map((w, i) => (
        <p key={`${w.key}-${i}`} className={`rounded-lg p-2 text-xs leading-snug ${tone[w.level]}`}>
          {t(w.key, w.values)}
        </p>
      ))}
    </div>
  );
}

export { contextLabel };
