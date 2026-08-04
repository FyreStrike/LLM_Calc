import { useState } from 'react';
import type { ModelSpec } from '../core/types';
import { MODELS } from '../data/models';
import { useLanguage, useT } from '../i18n';
import { HfImportError, importFromHuggingFace } from '../services/hfConfig';
import { allModels, selectedModel, useStore } from '../state/store';
import { params } from '../ui/format';
import {
  Badge,
  Button,
  Card,
  Field,
  NumberInput,
  Select,
  TextInput,
  Toggle,
} from '../ui/primitives';

export function ModelPanel() {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const state = useStore();
  const model = selectedModel(state);
  const [showCustom, setShowCustom] = useState(false);

  const customs = state.customModels;
  const families = Array.from(new Set(MODELS.map((m) => m.family ?? 'Other')));

  return (
    <Card
      title={t('section.model')}
      right={
        <Button variant="ghost" onClick={() => setShowCustom((v) => !v)}>
          {showCustom ? t('model.cancel') : t('model.addCustom')}
        </Button>
      }
    >
      <div className="space-y-3">
        <Field label={t('model.select')}>
          <Select value={state.modelId} onChange={(v) => state.set('modelId', v)}>
            {customs.length > 0 && (
              <optgroup label={t('model.custom')}>
                {customs.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </optgroup>
            )}
            {families.map((family) => (
              <optgroup key={family} label={family}>
                {MODELS.filter((m) => (m.family ?? 'Other') === family).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </optgroup>
            ))}
          </Select>
        </Field>

        <ModelSummary model={model} />

        {model.source !== 'catalog' && (
          <Button variant="ghost" onClick={() => state.removeCustomModel(model.id)}>
            {t('model.remove')}
          </Button>
        )}

        {showCustom && (
          <div className="space-y-4 border-t border-slate-200 pt-4 dark:border-slate-700">
            <HfImport onDone={() => setShowCustom(false)} />
            <CustomModelForm onDone={() => setShowCustom(false)} />
          </div>
        )}
      </div>

      {model.note && (
        <p className="mt-3 rounded-lg bg-sky-50 p-2 text-xs leading-snug text-sky-900 dark:bg-sky-950/50 dark:text-sky-200">
          {t(model.note)}
        </p>
      )}

      <p className="sr-only">{params(model.paramsTotal, language)}</p>
    </Card>
  );
}

function ModelSummary({ model }: { model: ModelSpec }) {
  const t = useT();
  const language = useLanguage((s) => s.language);

  const rows: [string, string][] = [
    [t('model.paramsTotal'), params(model.paramsTotal, language)],
    ...(model.paramsActive
      ? ([
          [
            t('model.paramsActive'),
            params(model.paramsActive, language) +
              (model.paramsActiveEstimated ? ` (${t('model.activeEstimatedBadge')})` : ''),
          ],
        ] as [string, string][])
      : []),
    [t('model.layers'), String(model.numLayers)],
    [t('model.heads'), `${model.numAttentionHeads} / ${model.numKeyValueHeads} KV`],
    [t('model.headDim'), String(model.headDim)],
    [t('model.maxContext'), params(model.maxContext, language)],
  ];

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-1.5">
        <Badge tone={model.attention === 'mla' ? 'info' : 'neutral'}>
          {t(`attention.${model.attention}`)}
        </Badge>
        <Badge tone={model.moe ? 'info' : 'neutral'}>
          {model.moe
            ? `${t('model.moe')} ${model.moe.numExperts}×top-${model.moe.expertsPerToken}`
            : t('model.dense')}
        </Badge>
        {model.slidingWindow && (
          <Badge tone="warn">
            {t('model.slidingWindow')} {model.slidingWindow}
            {model.slidingWindowLayerRatio && model.slidingWindowLayerRatio < 1
              ? ` (${Math.round(model.slidingWindowLayerRatio * 100)}%)`
              : ''}
          </Badge>
        )}
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-2">
            <dt className="text-slate-500 dark:text-slate-400">{label}</dt>
            <dd className="font-medium tabular-nums text-slate-800 dark:text-slate-100">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function HfImport({ onDone }: { onDone: () => void }) {
  const t = useT();
  const addCustomModel = useStore((s) => s.addCustomModel);
  const [repo, setRepo] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const model = await importFromHuggingFace(repo);
      addCustomModel(model);
      onDone();
    } catch (e) {
      if (e instanceof HfImportError) {
        setError(
          t(
            {
              notFound: 'error.hfNotFound',
              gated: 'error.hfGated',
              network: 'error.hfNetwork',
              unsupported: 'error.hfUnsupported',
            }[e.kind],
          ),
        );
      } else {
        setError(t('error.hfNetwork'));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-2">
      <Field label={t('model.importHf')} help={t('model.importHfHelp')}>
        <TextInput value={repo} onChange={setRepo} placeholder={t('model.importHfPlaceholder')} />
      </Field>
      <Button variant="primary" onClick={run} disabled={busy || repo.trim() === ''}>
        {busy ? t('model.importing') : t('model.import')}
      </Button>
      {error && <p className="text-xs text-rose-600 dark:text-rose-400">{error}</p>}
    </div>
  );
}

const BLANK: ModelSpec = {
  id: '',
  name: '',
  paramsTotal: 8e9,
  numLayers: 32,
  hiddenSize: 4096,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  headDim: 128,
  vocabSize: 128256,
  maxContext: 32768,
  attention: 'gqa',
  source: 'custom',
};

function CustomModelForm({ onDone }: { onDone: () => void }) {
  const t = useT();
  const addCustomModel = useStore((s) => s.addCustomModel);
  const [draft, setDraft] = useState<ModelSpec>(BLANK);
  const [isMoe, setIsMoe] = useState(false);

  function field<K extends keyof ModelSpec>(key: K, value: ModelSpec[K]) {
    setDraft((d) => ({ ...d, [key]: value }));
  }

  function save() {
    const id = `custom:${draft.name.trim() || 'model'}-${Date.now().toString(36)}`;
    const spec: ModelSpec = {
      ...draft,
      id,
      name: draft.name.trim() || 'Custom model',
      // Attention type decides the KV formula, so keep the MLA fields
      // consistent with it rather than trusting whatever is in the form.
      kvLoraRank: draft.attention === 'mla' ? (draft.kvLoraRank ?? 512) : undefined,
      qkRopeHeadDim: draft.attention === 'mla' ? (draft.qkRopeHeadDim ?? 64) : undefined,
      moe: isMoe ? (draft.moe ?? { numExperts: 128, expertsPerToken: 8 }) : undefined,
      paramsActive: isMoe ? (draft.paramsActive ?? draft.paramsTotal / 8) : undefined,
      source: 'custom',
    };
    addCustomModel(spec);
    onDone();
  }

  const numField = (
    label: string,
    key: keyof ModelSpec,
    help?: string,
    scale = 1,
  ) => (
    <Field label={label} help={help}>
      <NumberInput
        value={((draft[key] as number | undefined) ?? 0) / scale}
        onChange={(v) => field(key, ((v ?? 0) * scale) as never)}
        min={0}
      />
    </Field>
  );

  return (
    <div className="space-y-3">
      <Field label={t('model.name')}>
        <TextInput value={draft.name} onChange={(v) => field('name', v)} />
      </Field>

      <div className="grid grid-cols-2 gap-3">
        {numField(`${t('model.paramsTotal')} (B)`, 'paramsTotal', t('model.paramsTotalHelp'), 1e9)}
        <Field label={t('model.attention')}>
          <Select
            value={draft.attention}
            onChange={(v) => field('attention', v as ModelSpec['attention'])}
          >
            <option value="mha">MHA</option>
            <option value="gqa">GQA</option>
            <option value="mqa">MQA</option>
            <option value="mla">MLA</option>
          </Select>
        </Field>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {numField(t('model.layers'), 'numLayers')}
        {numField(t('model.hiddenSize'), 'hiddenSize')}
        {numField(t('model.heads'), 'numAttentionHeads')}
        {numField(t('model.kvHeads'), 'numKeyValueHeads')}
        {numField(t('model.headDim'), 'headDim')}
        {numField(t('model.vocabSize'), 'vocabSize')}
        {numField(t('model.maxContext'), 'maxContext')}
      </div>

      {draft.attention === 'mla' && (
        <div className="grid grid-cols-2 gap-3 rounded-lg bg-sky-50 p-2 dark:bg-sky-950/40">
          {numField(t('model.kvLoraRank'), 'kvLoraRank')}
          {numField(t('model.qkRopeHeadDim'), 'qkRopeHeadDim')}
        </div>
      )}

      <Toggle checked={isMoe} onChange={setIsMoe} label={t('model.isMoe')} />

      {isMoe && (
        <div className="grid grid-cols-2 gap-3 rounded-lg bg-slate-50 p-2 dark:bg-slate-800/60">
          {numField(`${t('model.paramsActive')} (B)`, 'paramsActive', t('model.paramsActiveHelp'), 1e9)}
          <Field label={t('model.numExperts')}>
            <NumberInput
              value={draft.moe?.numExperts ?? 128}
              onChange={(v) =>
                field('moe', {
                  numExperts: v ?? 128,
                  expertsPerToken: draft.moe?.expertsPerToken ?? 8,
                })
              }
              min={2}
            />
          </Field>
          <Field label={t('model.expertsPerToken')}>
            <NumberInput
              value={draft.moe?.expertsPerToken ?? 8}
              onChange={(v) =>
                field('moe', {
                  numExperts: draft.moe?.numExperts ?? 128,
                  expertsPerToken: v ?? 8,
                })
              }
              min={1}
            />
          </Field>
        </div>
      )}

      <div className="flex gap-2">
        <Button variant="primary" onClick={save}>
          {t('model.save')}
        </Button>
        <Button onClick={onDone}>{t('model.cancel')}</Button>
      </div>
    </div>
  );
}

export { allModels };
