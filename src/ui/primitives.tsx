import type { ReactNode } from 'react';

/* ------------------------------------------------------------------ surfaces */

export function Card({
  title,
  children,
  right,
  subtle = false,
  className = '',
}: {
  title?: string;
  children: ReactNode;
  right?: ReactNode;
  /** Nested panel styling — flat fill, no elevation. */
  subtle?: boolean;
  className?: string;
}) {
  return (
    <section
      className={`rounded-[var(--radius)] border border-[var(--border)] ${
        subtle
          ? 'bg-[var(--surface-2)]'
          : 'bg-[var(--surface)] shadow-[var(--shadow-sm)]'
      } ${className}`}
    >
      {title && (
        <header className="flex items-center justify-between gap-3 px-4 pt-3.5 pb-2">
          <h2 className="text-[13px] font-semibold tracking-tight text-[var(--text)]">
            {title}
          </h2>
          {right}
        </header>
      )}
      <div className={title ? 'px-4 pb-4' : 'p-4'}>{children}</div>
    </section>
  );
}

/** Thin divider that matches the card border exactly. */
export function Divider({ className = '' }: { className?: string }) {
  return <div className={`h-px bg-[var(--border)] ${className}`} />;
}

/* -------------------------------------------------------------------- fields */

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
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-[12px] font-medium text-[var(--text-2)]">{label}</span>
        {hint && (
          <span className="text-[12px] font-semibold tabular-nums text-[var(--text)]">
            {hint}
          </span>
        )}
      </div>
      {children}
      {help && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-[var(--text-3)]">{help}</p>
      )}
    </label>
  );
}

const controlClass =
  'w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-[13px] text-[var(--text)] transition-colors hover:border-[var(--border-strong)] focus:border-[var(--accent)] focus:outline-none';

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
      className={`${controlClass} cursor-pointer`}
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
  suffix,
}: {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  suffix?: string;
}) {
  return (
    <div className="relative">
      <input
        type="number"
        className={`${controlClass} tabular-nums ${suffix ? 'pr-10' : ''}`}
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
      {suffix && (
        <span className="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2 text-[11px] text-[var(--text-3)]">
          {suffix}
        </span>
      )}
    </div>
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
      className={controlClass}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

/* ------------------------------------------------------------------- sliders */

/**
 * Log-scale slider. Context length and batch size span several orders of
 * magnitude; a linear track spends almost all its travel in the top decade.
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
  const ticks = ladder.filter(
    (_, i) => i % Math.ceil(ladder.length / 5) === 0 || i === ladder.length - 1,
  );

  return (
    <div>
      <input
        type="range"
        min={0}
        max={ladder.length - 1}
        step={1}
        value={index}
        onChange={(e) => onChange(ladder[Number(e.target.value)])}
      />
      <div className="mt-0.5 flex justify-between text-[10px] tabular-nums text-[var(--text-3)]">
        {ticks.map((s) => (
          <span key={s}>{formatTick(s)}</span>
        ))}
      </div>
    </div>
  );
}

function formatTick(v: number): string {
  if (v >= 1024 * 1024) return `${v / (1024 * 1024)}M`;
  if (v >= 1024) return `${v / 1024}k`;
  return String(v);
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
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

/* ------------------------------------------------------------------ controls */

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
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className="flex w-full cursor-pointer items-center gap-2.5 text-left"
      >
        <span
          className={`relative inline-flex h-[18px] w-[32px] shrink-0 items-center rounded-full transition-colors ${
            checked ? 'bg-[var(--accent)]' : 'bg-[var(--surface-3)]'
          }`}
        >
          <span
            className={`inline-block h-[14px] w-[14px] rounded-full bg-white shadow-sm transition-transform ${
              checked ? 'translate-x-[16px]' : 'translate-x-[2px]'
            }`}
          />
        </span>
        <span className="text-[12px] font-medium text-[var(--text)]">{label}</span>
      </button>
      {help && (
        <p className="mt-1 ml-[42px] text-[11px] leading-relaxed text-[var(--text-3)]">
          {help}
        </p>
      )}
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
    <div className="inline-flex rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--surface-2)] p-[3px]">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`cursor-pointer rounded-[6px] px-2.5 py-1 text-[12px] font-medium transition-all ${
            value === o.value
              ? 'bg-[var(--surface)] text-[var(--text)] shadow-[var(--shadow-sm)]'
              : 'text-[var(--text-3)] hover:text-[var(--text)]'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = 'secondary',
  disabled,
  type = 'button',
  size = 'md',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  disabled?: boolean;
  type?: 'button' | 'submit';
  size?: 'sm' | 'md';
}) {
  const variantClass = {
    primary:
      'bg-[var(--accent)] text-[var(--accent-ink)] hover:bg-[var(--accent-hover)] shadow-[var(--shadow-sm)]',
    secondary:
      'border border-[var(--border)] bg-[var(--surface)] text-[var(--text-2)] hover:border-[var(--border-strong)] hover:text-[var(--text)]',
    ghost: 'text-[var(--text-3)] hover:text-[var(--text)] hover:bg-[var(--surface-2)]',
  }[variant];

  const sizeClass = size === 'sm' ? 'px-2.5 py-1 text-[12px]' : 'px-3 py-1.5 text-[12px]';

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`cursor-pointer rounded-[var(--radius-sm)] font-medium transition-all disabled:cursor-not-allowed disabled:opacity-50 ${variantClass} ${sizeClass}`}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------- display */

const TONE_INK = {
  default: 'text-[var(--text)]',
  good: 'text-[var(--good)]',
  bad: 'text-[var(--bad)]',
  warn: 'text-[var(--warn)]',
  accent: 'text-[var(--accent)]',
} as const;

export type Tone = keyof typeof TONE_INK;

export function Stat({
  label,
  value,
  unit,
  sub,
  tone = 'default',
  size = 'md',
}: {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  tone?: Tone;
  size?: 'md' | 'lg';
}) {
  return (
    <div>
      <div className="text-[11px] font-medium tracking-wide text-[var(--text-3)] uppercase">
        {label}
      </div>
      <div
        className={`mt-1 font-semibold tracking-tight tabular-nums ${TONE_INK[tone]} ${
          size === 'lg' ? 'text-[28px] leading-none' : 'text-[19px] leading-tight'
        }`}
      >
        {value}
        {unit && (
          <span className="ml-1 text-[12px] font-normal text-[var(--text-3)]">{unit}</span>
        )}
      </div>
      {sub && <div className="mt-1 text-[11px] text-[var(--text-3)]">{sub}</div>}
    </div>
  );
}

const BADGE_TONE = {
  neutral: 'bg-[var(--surface-3)] text-[var(--text-2)]',
  good: 'bg-[var(--good-soft)] text-[var(--good)]',
  bad: 'bg-[var(--bad-soft)] text-[var(--bad)]',
  warn: 'bg-[var(--warn-soft)] text-[var(--warn)]',
  info: 'bg-[var(--accent-soft)] text-[var(--accent)]',
} as const;

export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode;
  tone?: keyof typeof BADGE_TONE;
}) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-[3px] text-[11px] font-medium whitespace-nowrap ${BADGE_TONE[tone]}`}
    >
      {children}
    </span>
  );
}

/** Small colour chip that ties a table row to its mark in the chart above it. */
export function Swatch({ color }: { color: string }) {
  return (
    <span
      className="mr-2 inline-block h-2.5 w-2.5 shrink-0 rounded-[3px] align-middle"
      style={{ background: color }}
    />
  );
}

export function Note({
  children,
  tone = 'info',
}: {
  children: ReactNode;
  tone?: 'info' | 'warn' | 'bad' | 'neutral';
}) {
  const cls = {
    info: 'bg-[var(--accent-soft)] text-[var(--accent)]',
    warn: 'bg-[var(--warn-soft)] text-[var(--warn)]',
    bad: 'bg-[var(--bad-soft)] text-[var(--bad)]',
    neutral: 'bg-[var(--surface-2)] text-[var(--text-2)]',
  }[tone];
  return (
    <p className={`rounded-[var(--radius-sm)] px-2.5 py-2 text-[11px] leading-relaxed ${cls}`}>
      {children}
    </p>
  );
}
