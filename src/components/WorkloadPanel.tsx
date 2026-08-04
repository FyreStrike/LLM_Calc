import type { Runtime } from '../core/types';
import { useLanguage, useT } from '../i18n';
import { useStore } from '../state/store';
import { contextLabel, num } from '../ui/format';
import { Card, Field, LogSlider, Select, Slider, Toggle } from '../ui/primitives';

const CONTEXT_STEPS = [
  512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];
const BATCH_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256];

export function WorkloadPanel() {
  const t = useT();
  const language = useLanguage((s) => s.language);
  const state = useStore();

  return (
    <Card title={t('section.workload')}>
      <div className="space-y-3">
        <Field
          label={t('workload.contextLength')}
          hint={`${contextLabel(state.contextLength, language)} tok`}
        >
          <LogSlider
            value={state.contextLength}
            onChange={(v) => state.set('contextLength', v)}
            min={512}
            max={1048576}
            steps={CONTEXT_STEPS}
          />
        </Field>

        <Field
          label={t('workload.promptTokens')}
          hint={`${contextLabel(state.promptTokens, language)} tok`}
        >
          <LogSlider
            value={state.promptTokens}
            onChange={(v) => state.set('promptTokens', v)}
            min={128}
            max={524288}
            steps={[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]}
          />
        </Field>

        <Field
          label={t('workload.batchSize')}
          hint={String(state.batchSize)}
          help={t('workload.batchHelp')}
        >
          <LogSlider
            value={state.batchSize}
            onChange={(v) => state.set('batchSize', v)}
            min={1}
            max={256}
            steps={BATCH_STEPS}
          />
        </Field>

        <Field label={t('workload.runtime')}>
          <Select
            value={state.runtime}
            onChange={(v) => state.set('runtime', v as Runtime)}
          >
            <option value="llamacpp">{t('runtime.llamacpp')}</option>
            <option value="vllm">{t('runtime.vllm')}</option>
            <option value="transformers">{t('runtime.transformers')}</option>
          </Select>
        </Field>

        <Toggle
          checked={state.allowOffload}
          onChange={(v) => state.set('allowOffload', v)}
          label={t('workload.offload')}
          help={t('workload.offloadHelp')}
        />

        {state.advanced && (
          <div className="space-y-3 border-t border-[var(--border)] pt-3">
            <Field
              label={t('workload.mbu')}
              hint={`${num(state.mbu * 100, language, 0)}%`}
              help={t('workload.mbuHelp')}
            >
              <Slider
                value={state.mbu}
                onChange={(v) => state.set('mbu', v)}
                min={0.2}
                max={1}
                step={0.01}
              />
            </Field>
            <Field
              label={t('workload.mfu')}
              hint={`${num(state.mfu * 100, language, 0)}%`}
              help={t('workload.mfuHelp')}
            >
              <Slider
                value={state.mfu}
                onChange={(v) => state.set('mfu', v)}
                min={0.05}
                max={0.9}
                step={0.01}
              />
            </Field>
          </div>
        )}
      </div>
    </Card>
  );
}
