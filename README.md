# LLM Inference Calculator

[![CI](https://github.com/FyreStrike/LLM_Calc/actions/workflows/ci.yml/badge.svg)](https://github.com/FyreStrike/LLM_Calc/actions/workflows/ci.yml)
[![Deploy](https://github.com/FyreStrike/LLM_Calc/actions/workflows/deploy.yml/badge.svg)](https://github.com/FyreStrike/LLM_Calc/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Live: https://fyrestrike.github.io/LLM_Calc/**

VRAM, speed, energy and cost for local LLM inference — with an instant
comparison against what the same tokens would cost via an API.

Like [apxml's VRAM calculator](https://apxml.com/tools/vram-calculator), plus the
part it leaves out: **is running this locally actually cheaper than paying for
the API?**

## What it does

- **Memory** — weights, KV cache, activations and framework overhead, broken down
  per component. Handles MHA, GQA, MQA and **MLA** (DeepSeek, GLM-5, Kimi), dense
  and MoE, interleaved sliding-window attention, and **hybrid linear attention**
  (Qwen3.5/3.6), where only a quarter of layers keep a growing cache.
- **Speed** — decode tokens/s and time-to-first-token from the roofline model,
  with multi-GPU scaling and CPU offload.
- **Energy** — power draw and joules per token, either as a simple TDP fraction
  or via the extended energy roofline `E = ε_flop·W + ε_mop·Q + π₀·T`.
- **Cost** — local electricity vs. live API prices per million tokens, plus
  hardware break-even and CO₂.

The catalog carries **80 models** across 18 families — DeepSeek V3/V4, GLM-4.x/5.x,
the Qwen2.5/3/3.5/3.6 line, Kimi K2.x, MiniMax, gpt-oss, Llama 3.x/4, Gemma 2/3/4,
Mistral, Nemotron, Solar, EXAONE, ERNIE, Granite, Phi and more. Every
architecture field is generated from the model's real HuggingFace `config.json`,
so nothing is inferred from a name. Models can also be defined by hand or
imported from any HuggingFace repo. API prices are fetched live from OpenRouter
with a bundled snapshot as fallback.

Fully static — no backend, no API keys. Works offline on the bundled data.

## Running it

```bash
npm install
```

```bash
npm run dev
```

```bash
npm run test
```

```bash
npm run build
```

The build output in `dist/` is a static site; it can be served from anywhere,
including GitHub Pages (`base` is already relative).

## Layout

```
src/
  core/         calculation engine — pure TypeScript, no UI imports
    memory.ts       weights · KV cache (MHA/GQA/MLA) · activations · overhead
    roofline.ts     P_attain = min(F, B·I), ridge point, parallel efficiency
    performance.ts  decode · prefill · MoE expert union · offload
    energy.ts       simple heuristic + E = ε_flop·W + ε_mop·Q + π₀·T
    cost.ts         local vs API, break-even, CO₂
    __tests__/      unit tests against hand-computed reference values
  data/         model catalog (generated), GPU specs, quantization table, prices
  services/     HuggingFace config import, OpenRouter price fetch
  components/   input panels and result cards
  i18n/         German and English
```

`src/core/` deliberately has zero UI dependencies: it is unit-testable in
isolation and portable to Python for cross-validation against real measurements.

## Accuracy

Every formula and reference number is documented with its source in
[METHODIK.md](METHODIK.md), including the known limitations. Two things worth
knowing up front:

- **Quantization sizes are measured, not nominal.** GGUF `Q4_K_M` is 4.89 bits
  per weight, not 4.0 — the `_M` variants keep some tensors at higher precision.
- **MLA is detected structurally.** DeepSeek-V3's config reports 128 KV heads,
  which reads as full MHA; applying the GQA formula there overstates the KV cache
  by 57×. Classification keys off `kv_lora_rank`, never the head counts.

### What the tests do and do not establish

The suite checks formulas against hand-computed values, pins invariants, and
guards regressions — Llama-3.1-8B's KV cache is exactly 131 072 bytes/token,
DeepSeek-V3's exactly 70 272, the H100 FP16 ridge point 295.2 FLOP/byte.

**It does not validate against real measurements.** No figure here has been
compared with llama.cpp or vLLM running on actual hardware across several GPUs.
The memory formulas are arithmetic and should hold; the speed, power and cooling
models rest on utilisation factors (MBU, MFU), estimated CPU idle draw and
fitted chassis coefficients that are plausible and sourced but unverified. Treat
absolute numbers as estimates and relative comparisons as the more trustworthy
output.

Closing that gap — profiling a handful of configurations and fitting the
coefficients to the results — is the obvious next step and the one that would
most improve the tool.

## License

[MIT](LICENSE).

## Context

Built alongside a bachelor thesis on hardware-level analysis of LLM inference.
The energy model implements the thesis's extended energy roofline rather than a
TDP heuristic, which is what makes the numbers defensible rather than merely
plausible.
