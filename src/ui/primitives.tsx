import type { ReactNode } from 'react';

export function Card({
  title,
  children,
  right,
  className = '',
}: {
  title?: string;
  children: ReactNode;
  right?: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900 ${className}`}
    >
      {title && (
        <header className="mb-3 flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
            {title}
          </h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Field({
  label,
  help,
  children,
  hint,
}: {
  label: string;
  help?: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{label}</span>
        {hint && <span className="text-xs tabular-nums text-slate-500 dark:text-slate-400">{hint}</span>}
      </div>
      {children}
      {help && <p className="mt-1 text-xs leading-snug text-slate-500 dark:text-slate-400">{help}</p>}
    </label>
  );
}

const inputClass =
  'w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-500 focus:ring-2 focus:ring-sky-500/20 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100';

export function Select<T extends string>({
  value,
  onChange,
  children,
}: {
  value: T;
  onChange: (value: T) => void;
  children: ReactNode;
}) {
  return (
    <select
      className={inputClass}
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
    >
      {children}
    </select>
  );
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  placeholder,
}: {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
}) {
  return (
    <input
      type="number"
      className={inputClass}
      value={value ?? ''}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === '') return onChange(null);
        const n = Number(raw);
        onChange(Number.isFinite(n) ? n : null);
      }}
    />
  );
}

export function TextInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      className={inputClass}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

/**
 * Log-scale slider. Context length and batch size both span several orders of
 * magnitude, so a linear slider wastes almost all of its travel on the top
 * decade.
 */
export function LogSlider({
  value,
  onChange,
  min,
  max,
  steps,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  steps?: number[];
}) {
  const ladder = steps ?? defaultLadder(min, max);
  const index = nearestIndex(ladder, value);

  return (
    <div>
      <input
        type="range"
        className="w-full accent-sky-600"
        min={0}
        max={ladder.length - 1}
        step={1}
        value={index}
        onChange={(e) => onChange(ladder[Number(e.target.value)])}
      />
      <div className="mt-1 flex justify-between text-[10px] tabular-nums text-slate-400">
        {ladder
          .filter((_, i) => i % Math.ceil(ladder.length / 6) === 0 || i === ladder.length - 1)
          .map((s) => (
            <span key={s}>{s >= 1024 ? `${s / 1024}k` : s}</span>
          ))}
      </div>
    </div>
  );
}

function defaultLadder(min: number, max: number): number[] {
  const out: number[] = [];
  for (let v = min; v <= max; v *= 2) out.push(v);
  if (out[out.length - 1] !== max) out.push(max);
  return out;
}

function nearestIndex(ladder: number[], value: number): number {
  let best = 0;
  let bestDelta = Infinity;
  ladder.forEach((v, i) => {
    const d = Math.abs(Math.log(v) - Math.log(value));
    if (d < bestDelta) {
      bestDelta = d;
      best = i;
    }
  });
  return best;
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <input
      type="range"
      className="w-full accent-sky-600"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  help,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  help?: string;
}) {
  return (
    <div>
      <label className="flex cursor-pointer items-center gap-2">
        <input
          type="checkbox"
          className="h-4 w-4 accent-sky-600"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="text-sm font-medium text-slate-700 dark:text-slate-200">{label}</span>
      </label>
      {help && <p className="mt-1 ml-6 text-xs leading-snug text-slate-500 dark:text-slate-400">{help}</p>}
    </div>
  );
}

export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-slate-300 bg-slate-100 p-0.5 dark:border-slate-600 dark:bg-slate-800">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1 text-xs font-medium transition ${
            value === o.value
              ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-600 dark:text-white'
              : 'text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({
  label,
  value,
  unit,
  sub,
  tone = 'default',
}: {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  tone?: 'default' | 'good' | 'bad' | 'warn';
}) {
  const toneClass = {
    default: 'text-slate-900 dark:text-slate-50',
    good: 'text-emerald-600 dark:text-emerald-400',
    bad: 'text-rose-600 dark:text-rose-400',
    warn: 'text-amber-600 dark:text-amber-400',
  }[tone];

  return (
    <div>
      <div className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</div>
      <div className={`mt-0.5 text-xl font-semibold tabular-nums ${toneClass}`}>
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-slate-500 dark:text-slate-400">{unit}</span>}
      </div>
      {sub && <div className="text-xs text-slate-500 dark:text-slate-400">{sub}</div>}
    </div>
  );
}

export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: 'neutral' | 'good' | 'bad' | 'warn' | 'info';
}) {
  const toneClass = {
    neutral: 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200',
    good: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300',
    bad: 'bg-rose-100 text-rose-800 dark:bg-rose-900/50 dark:text-rose-300',
    warn: 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300',
    info: 'bg-sky-100 text-sky-800 dark:bg-sky-900/50 dark:text-sky-300',
  }[tone];

  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-[11px] font-medium ${toneClass}`}>
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  variant = 'secondary',
  disabled,
  type = 'button',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  disabled?: boolean;
  type?: 'button' | 'submit';
}) {
  const variantClass = {
    primary: 'bg-sky-600 text-white hover:bg-sky-500 disabled:bg-slate-300',
    secondary:
      'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700',
    ghost: 'text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white',
  }[variant];

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-60 ${variantClass}`}
    >
      {children}
    </button>
  );
}
