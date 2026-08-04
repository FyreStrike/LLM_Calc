import { describe, expect, it } from 'vitest';
import { __parsePricesForTest as parsePrices } from '../../services/openrouter';
import { detectAttention } from '../../services/hfConfig';

describe('attention detection from config.json', () => {
  it('detects MLA from kv_lora_rank even when head counts say MHA', () => {
    // This is DeepSeek-V3's real config: 128 attention heads AND 128 KV heads,
    // which reads as full MHA. The kv_lora_rank field is the only reliable
    // signal, and misreading it inflates the KV cache by ~57x.
    expect(
      detectAttention({
        num_attention_heads: 128,
        num_key_value_heads: 128,
        kv_lora_rank: 512,
        qk_rope_head_dim: 64,
      }),
    ).toBe('mla');
  });

  it('detects MLA for Kimi K2, whose head counts also look like MHA', () => {
    expect(
      detectAttention({
        num_attention_heads: 64,
        num_key_value_heads: 64,
        kv_lora_rank: 512,
      }),
    ).toBe('mla');
  });

  it('detects GQA when KV heads are fewer than attention heads', () => {
    expect(detectAttention({ num_attention_heads: 32, num_key_value_heads: 8 })).toBe('gqa');
    expect(detectAttention({ num_attention_heads: 64, num_key_value_heads: 4 })).toBe('gqa');
  });

  it('detects MQA at a single KV head', () => {
    expect(detectAttention({ num_attention_heads: 32, num_key_value_heads: 1 })).toBe('mqa');
  });

  it('detects genuine MHA', () => {
    expect(detectAttention({ num_attention_heads: 32, num_key_value_heads: 32 })).toBe('mha');
  });

  it('treats a missing num_key_value_heads as MHA', () => {
    expect(detectAttention({ num_attention_heads: 32 })).toBe('mha');
  });
});

describe('OpenRouter price parsing', () => {
  it('converts per-token strings to dollars per million tokens', () => {
    // GLM 5.2's real listing: 0.00000076 USD/token -> 0.76 USD per 1M.
    const [p] = parsePrices([
      {
        id: 'z-ai/glm-5.2',
        name: 'Z.ai: GLM 5.2',
        context_length: 1048576,
        pricing: { prompt: '0.00000076', completion: '0.00000242' },
      },
    ]);
    expect(p.inputPerMTokUsd).toBeCloseTo(0.76, 6);
    expect(p.outputPerMTokUsd).toBeCloseTo(2.42, 6);
    expect(p.contextLength).toBe(1048576);
  });

  it('drops free and unpriced listings', () => {
    const parsed = parsePrices([
      { id: 'free/model', name: 'Free', pricing: { prompt: '0', completion: '0' } },
      { id: 'broken/model', name: 'Broken', pricing: {} },
      { id: 'none/model', name: 'None' },
    ]);
    expect(parsed).toHaveLength(0);
  });

  it('keeps a model priced on output only', () => {
    const parsed = parsePrices([
      { id: 'x/y', name: 'X', pricing: { prompt: '0', completion: '0.000001' } },
    ]);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].outputPerMTokUsd).toBeCloseTo(1, 6);
  });

  it('sorts by label so the picker is navigable', () => {
    const parsed = parsePrices([
      { id: 'b', name: 'Zeta', pricing: { prompt: '0.000001', completion: '0.000002' } },
      { id: 'a', name: 'Alpha', pricing: { prompt: '0.000001', completion: '0.000002' } },
    ]);
    expect(parsed.map((p) => p.label)).toEqual(['Alpha', 'Zeta']);
  });
});
