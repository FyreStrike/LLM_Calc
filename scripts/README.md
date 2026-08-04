# Catalog generation

`src/data/models.ts` is **generated**, not hand-written. Every architecture field
comes from the model's real `config.json` on HuggingFace, and parameter counts
come from the `?expand=safetensors` API — so the catalog can be regenerated when
new models ship without anyone re-typing head counts.

## Regenerating

```bash
node scripts/fetch-model-configs.mjs > models.json
```

```bash
node scripts/emit-models.mjs models.json > src/data/models.ts
```

The fetch step prints progress and failures to stderr. Gated repos (`meta-llama`,
`CohereLabs`, some `nvidia` and `allenai` repos) return 401 without a token; the
candidate list works around that by reading from ungated mirrors such as
`unsloth/*` while keeping the canonical repo id for display.

## Adding a model

Append to `CANDIDATES` in `fetch-model-configs.mjs`:

```js
['Display Name', 'Family', 'org/repo-to-read'],
['Display Name', 'Family', 'unsloth/mirror', 'org/canonical-repo'],  // gated
```

If the model is MoE and its active-parameter count is published, add it to
`PUBLISHED_ACTIVE` in `emit-models.mjs`. If the name already encodes it
(`…-A22B`), that is picked up automatically and needs no entry. Anything else is
estimated from the architecture and flagged `paramsActiveEstimated: true` — the
estimate runs high whenever the safetensors total includes a multi-token
prediction head.

## What the emitter handles that a naive parser would not

- **MLA detection** keys off `kv_lora_rank`, never head counts. DeepSeek, GLM-5
  and Kimi all report `num_key_value_heads == num_attention_heads`.
- **`use_sliding_window: false`** — Qwen2.5 ships a window value with the feature
  switched off.
- **Hybrid linear attention** — `layer_types`, `full_attention_interval` and
  `linear_attn_config.full_attn_layers` all describe how many layers actually
  keep a growing KV cache.
- **MoE field-name variants** — `num_experts` / `num_local_experts` /
  `n_routed_experts` / `moe_num_experts`, and three spellings of top-k.
- **Sparse attention (DSA)** — DeepSeek V4 and GLM-5's `sliding_window` is not
  applied, because it cannot be interpreted unambiguously without `layer_types`;
  the full-cache upper bound is reported instead.

Architectures the memory model cannot represent honestly are listed in `DROP` and
excluded rather than approximated.
