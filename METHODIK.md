# Methodik

Dieses Dokument belegt jede Formel und jeden Referenzwert des Rechners. Es ist so
geschrieben, dass es als Anhang der Bachelorarbeit verwendbar ist.

Die Berechnungslogik liegt vollständig in `src/core/` und ist frei von
UI-Abhängigkeiten. Jede hier dokumentierte Formel besitzt einen zugehörigen
Test in `src/core/__tests__/`.

---

## 1. Theoretische Grundlage

Das Modell folgt dem Forschungsdesign der Bachelorarbeit
(`Bachelorarbeit/old/Forschungsdesign_Tiefenanalyse.md`).

### 1.1 Performance-Roofline

$$P_{\text{attain}} = \min(F, B \cdot I), \qquad I_{\text{ridge}} = \frac{F}{B}$$

* $F$ — Spitzenrechenleistung (FLOP/s), **dense**, ohne 2:4-Sparsity
* $B$ — Speicherbandbreite (Byte/s)
* $I$ — arithmetische Intensität (FLOP/Byte)

Implementierung: `src/core/roofline.ts`.

**Verifikation** (`performance.test.ts`): Der Ridge Point der H100 in FP16 ergibt
$989 \cdot 10^{12} / 3{,}35 \cdot 10^{12} \approx 295{,}2$ FLOP/Byte, in FP8
$\approx 590{,}7$; die B200 in FP4 liegt bei $9000/8{,}0 = 1125$ FLOP/Byte. Alle
drei reproduzieren die Werte des Forschungsdesigns exakt.

### 1.2 Phasenabhängige Intensität

| Phase | Operation | Intensität | Regime |
| --- | --- | --- | --- |
| Prefill | GEMM (Matrix-Matrix) | hoch (~2048 FLOP/Byte bei N=4096, FP16) | rechenleistungsgebunden |
| Decode | GEMV (Matrix-Vektor) | $I \approx B_{\text{batch}}$ | speicherbandbreitenlimitiert |

Im speicherbegrenzten Bereich dominiert der Gewichtstransfer $D \cdot F$ den
Nenner der Intensitätsgleichung, sodass $I_{\text{matmul}} \approx B_{\text{batch}}$.
Deshalb ist die Batch-Größe der stärkste Hebel auf die Energie pro Token — der
Test `puts decode intensity near the batch size` fixiert dieses Verhalten.

### 1.3 Erweitertes Energie-Roofline-Modell

$$E = \varepsilon_{\text{flop}} \cdot W + \varepsilon_{\text{mop}} \cdot Q + \pi_0 \cdot T$$

* $\varepsilon_{\text{flop}}$ — dynamische Energie pro Rechenoperation (J/FLOP)
* $W$ — Anzahl ausgeführter FLOPs
* $\varepsilon_{\text{mop}}$ — dynamische Energie pro übertragenem Byte (J/Byte)
* $Q$ — übertragenes Datenvolumen (Byte)
* $\pi_0$ — statische Verlustleistung im Idle (W)
* $T$ — Laufzeit (s)

Implementierung: `src/core/energy.ts`. Die Zerlegung ist in der UI sichtbar; im
speicherbandbreitenlimitierten Decode dominiert erwartungsgemäß der Term
$\varepsilon_{\text{mop}} \cdot Q$ (gemessen ~63 % gegenüber ~1 % Rechenanteil bei
DeepSeek V3 auf 8×H100).

**Standardkoeffizienten** werden aus den Hardwaredaten abgeleitet:

$$\varepsilon_{\text{flop}} = \frac{(\text{TDP} - \pi_0) \cdot 0{,}70}{F}, \qquad
\varepsilon_{\text{mop}} = \frac{(\text{TDP} - \pi_0) \cdot 0{,}35}{B}$$

Die Anteile 0,70 / 0,35 sind Kalibrierungsparameter und in der UI im
Erweitert-Modus editierbar — sie sind der Ansatzpunkt für eine Kalibrierung gegen
reale NVML-Messungen aus `power_monitor.py`.

---

## 2. Speicherbedarf

Implementierung: `src/core/memory.ts`.

### 2.1 Gewichte

$$M_{\text{weights}} = P_{\text{total}} \cdot \frac{\text{bpw}}{8}$$

Bei MoE ist **bewusst $P_{\text{total}}$** eingesetzt: alle Experten müssen
resident sein. Nur Bandbreite und FLOPs richten sich nach $P_{\text{active}}$.

**Bits pro Gewicht** stammen aus der von llama.cpp gemessenen Tabelle für
Llama-3.1-8B ([Quelle](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)).
Es sind Durchschnitte auf **Dateiebene**, nicht die reinen Blockgrößen: die
Varianten `_S`/`_M`/`_L` heben `attention.wv`, `feed_forward.w2` sowie
Embedding-/Output-Tensoren auf Q6_K bzw. Q8_0 an.

| Typ | bpw | Typ | bpw |
| --- | --- | --- | --- |
| F16 | 16,00 | Q4_K_M | **4,8944** |
| Q8_0 | 8,5008 | Q4_K_S | 4,6672 |
| Q6_K | 6,5633 | IQ4_XS | 4,4597 |
| Q5_K_M | 5,7036 | Q3_K_M | 3,9960 |
| Q5_K_S | 5,5704 | Q2_K | 3,1593 |

`Q4_0` fehlt in der llama.cpp-Tabelle (deprecated) und wurde aus der
Blockstruktur abgeleitet: 2-Byte-FP16-Skalierung + 16 Byte Nibbles je 32 Gewichte
$= 18 \cdot 8 / 32 = 4{,}5$ bpw exakt.

> **Einschränkung.** bpw mittelt über *alle* Parameter inklusive Embeddings. Bei
> kleinen Modellen mit großem Vokabular (z. B. 0,6B Parameter bei 151k Vokabular)
> weicht der reale Dateiwert spürbar ab.

### 2.2 KV-Cache

**MHA / GQA / MQA:**

$$M_{\text{KV}} = 2 \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot L \cdot S \cdot B \cdot b$$

**MLA (DeepSeek V2/V3/R1, Kimi K2):**

$$M_{\text{KV}} = (r_{\text{kv\_lora}} + d_{\text{qk\_rope}}) \cdot L \cdot S \cdot B \cdot b$$

MLA speichert einen einzelnen komprimierten Latent plus den entkoppelten
RoPE-Key — kein Faktor 2, keine Head-Anzahl.

> **Die zentrale Falle.** DeepSeek-V3 meldet in `config.json`
> `num_attention_heads: 128` **und** `num_key_value_heads: 128`. Nach der
> naheliegenden Regel ist das volles MHA. Tatsächlich ist es MLA: Der Cache
> beträgt $(512 + 64) \cdot 61 \cdot 2 = 70\,272$ Byte/Token statt
> $2 \cdot 128 \cdot 128 \cdot 61 \cdot 2 = 3\,997\,696$ Byte/Token — ein Faktor
> **56,9**. Die Klassifikation erfolgt daher über die *Struktur*
> (`kv_lora_rank` vorhanden ⇒ MLA), nie über die Head-Zahlen.
> Der Test `REGRESSION: must not apply the GQA formula to an MLA model` fixiert das.

**Verifizierte Referenzwerte:**

| Modell | Byte/Token (FP16) | @ 32k Kontext |
| --- | --- | --- |
| Llama 3.1 8B (GQA) | 131 072 (128 KiB) | 4,0 GiB |
| Qwen3 235B-A22B (GQA) | 192 512 | 5,9 GiB |
| DeepSeek V3 (MLA) | **70 272** | **2,1 GiB** |

Der 70-KB-Wert reproduziert die im DeepSeek-V3-Report genannte Größenordnung.
Bemerkenswert: das 671B-MLA-Modell cached **weniger pro Token** als ein 70B-GQA-Modell.

**Sliding Window.** Moderne Modelle mischen lokale und globale Layer. Der
Rechner berücksichtigt den Anteil:

$$M_{\text{KV}} = \big(L_{\text{window}} \cdot \min(S, S_{\text{window}}) + L_{\text{global}} \cdot S\big) \cdot m_{\text{layer}} \cdot B$$

| Modell | Fenster | Anteil gefensterter Layer |
| --- | --- | --- |
| Gemma 2 9B | 4096 | 1/2 (alternierend) |
| Gemma 3 27B / Gemma 4 31B | 1024 | 5/6 (`sliding_window_pattern: 6`) |
| gpt-oss 20B/120B | 128 | 1/2 (`layer_types` alternierend) |

Bei 128k Kontext dominieren die verbleibenden globalen Layer — der Anteil ist
kein vernachlässigbares Detail.

> **Zweite Falle: `use_sliding_window`.** Qwen2.5 gibt `sliding_window: 131072`
> an **und** `use_sliding_window: false` — das Fenster ist konfiguriert, aber
> abgeschaltet. Wer nur das erste Feld liest, unterschätzt den KV-Cache um ein
> Vielfaches. Der Parser prüft daher beide Felder.

### 2.2.1 Hybride lineare Attention

Modelle der Generation 2026 mischen Softmax-Attention mit linearer Attention
(Gated DeltaNet in Qwen3.5/3.6, KDA in Kimi Linear). Lineare Layer halten einen
rekurrenten Zustand konstanter Größe und cachen **nichts**, was mit der
Sequenzlänge wächst. Nur die Softmax-Layer tragen einen KV-Cache:

$$M_{\text{KV}} = m_{\text{layer}} \cdot L \cdot \rho_{\text{kv}} \cdot S \cdot B, \qquad
\rho_{\text{kv}} = \frac{L_{\text{softmax}}}{L}$$

| Modell | Erkennung | $\rho_{\text{kv}}$ |
| --- | --- | --- |
| Qwen3.5 / Qwen3.6 | `full_attention_interval: 4`, `layer_types` | 1/4 |
| Kimi Linear 48B | `linear_attn_config.full_attn_layers` (7 von 27) | 0,26 |

Wirkung: Qwen3.6 35B-A3B benötigt bei 128k Kontext **2,5 GiB** statt der
10 GiB, die eine Berechnung über alle Layer ergäbe.

### 2.2.2 Sparse Attention (DSA)

DeepSeek V4 und GLM-5.x verwenden DeepSeek Sparse Attention: Jede Query greift
über einen Indexer auf eine ausgewählte Teilmenge der Keys zu
(`index_topk`: 512 bzw. 2048). Das reduziert den **Rechenaufwand**, nicht
zwangsläufig den **Speicher** — der vollständige Cache muss vorgehalten werden,
damit überhaupt ausgewählt werden kann. Der Rechner gibt daher die obere
Schranke (voller Cache) aus und weist in der UI darauf hin. Der in diesen
Konfigurationen zusätzlich gesetzte `sliding_window`-Wert wird bewusst
**nicht** angewendet, da er ohne `layer_types` nicht eindeutig interpretierbar
ist und zu unrealistisch niedrigen Werten führen würde.

### 2.3 Aktivierungen und Overhead

Der dominante Term im Prefill ist der Logits-Tensor in FP32. Entscheidend ist,
**wie viele Positionen** die Runtime auf Logits projiziert:

| Runtime | Logit-Positionen | Framework-Overhead |
| --- | --- | --- |
| vLLM | 1 je Sequenz | 2,0 GB (CUDA-Graphen) |
| llama.cpp | 1 je Sequenz | 500 MB Compute-Buffer |
| HF Transformers | alle Prefill-Positionen | 15 % von (Gewichte + KV) |

Bei 4096 Prefill-Token und 128k Vokabular sind das unter Transformers allein
2,1 GB Logits — die klassische, oft fehldiagnostizierte OOM-Ursache.
Hinzu kommt ein CUDA-Kontext von 400 MB je GPU.

> Die verbreitete Faustformel $M = P \cdot (Q/8) \cdot 1{,}2$ wird **nicht**
> verwendet. Sie faltet KV-Cache und Overhead in pauschale 20 %, was bei langem
> Kontext grob falsch ist: Ein 70B-Modell bei 128k Kontext hat allein 43 GB
> KV-Cache — bei 4-Bit-Gewichten (35 GB) über 100 % statt 20 %.

---

## 3. Geschwindigkeit

Implementierung: `src/core/performance.ts`.

### 3.1 Decode

$$Q_{\text{step}} = P_{\text{read}} \cdot \frac{\text{bpw}}{8} + M_{\text{KV}},
\qquad W_{\text{step}} = 2 \cdot P_{\text{active}} \cdot B_{\text{batch}}$$

$$t_{\text{step}} = \max\left(\frac{Q_{\text{step}}}{B \cdot \text{MBU}}, \frac{W_{\text{step}}}{F \cdot \text{MFU}}\right)$$

Die `max`-Form ist äquivalent zu $W / P_{\text{attain}}$ und behandelt beide
Regime einheitlich.

**MoE-Expertenvereinigung.** Ein einzelnes Token routet zu $k$ von $N$ Experten,
ein Batch von $B$ Token aktiviert jedoch deren Vereinigungsmenge:

$$N_{\text{read}} = N \cdot \left(1 - \left(1 - \tfrac{k}{N}\right)^{B}\right)$$

Bei angenommener Gleichverteilung ist das eine obere Schranke (reales Routing ist
schief), also eine konservative Geschwindigkeitsschätzung. Die Aufteilung in
Backbone und Experten folgt aus den beiden bekannten Größen:
$p_e = (P_{\text{total}} - P_{\text{active}}) / (N - k)$.

**MBU / MFU.** Databricks misst ~60 % MBU bei Batch 1 auf einer H100; gut
abgestimmte Setups erreichen 70–90 %. Prefill-MFU liegt typisch bei 30–50 %.
Voreinstellung: MBU 0,70 / MFU 0,40, im Erweitert-Modus editierbar.
([Quelle](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices))

### 3.2 Prefill

$$W_{\text{prefill}} = \big(2 \cdot P_{\text{active}} \cdot T + 4 \cdot L \cdot T^2 \cdot d_{\text{model}}\big) \cdot B_{\text{batch}}$$

Der quadratische Attention-Term wird oberhalb von ~8–16k Token relevant. Der Test
`grows superlinearly with prompt length` prüft, dass eine Verachtfachung der
Promptlänge die TTFT um mehr als das Achtfache erhöht.

### 3.3 Mehr-GPU-Skalierung

$$\eta_{\text{TP}} = 1 - c \cdot \log_2(N), \qquad c = \begin{cases} 0{,}05 & \text{NVLink} \\ 0{,}20 & \text{PCIe} \end{cases}$$

> **Einschränkung.** Für diesen Zusammenhang existiert keine publizierte
> geschlossene Form. Die Koeffizienten sind eine Anpassung an veröffentlichte
> Messungen (NVLink-Paar ~+50 % Durchsatz bei 2 GPUs, ~+10 % bei 4; PCIe-All-Reduce
> 30–40 % der Schrittzeit auf 8-GPU-Knoten) und sind ausdrücklich heuristisch.

### 3.4 CPU-Offloading

$$t_{\text{step}} = \frac{Q_{\text{GPU}}}{B_{\text{GPU}}} + \frac{Q_{\text{Host}}}{B_{\text{Host}}}$$

**Zeiten addieren sich, Geschwindigkeiten nicht.** Die Hälfte der Layer
auszulagern kostet daher weit mehr als die Hälfte des Durchsatzes, weil der
langsame Summand dominiert. Nur Gewichte werden ausgelagert — den KV-Cache je
Schritt über PCIe zu schieben wäre ruinös.

---

## 4. Energie und Kosten

Implementierung: `src/core/energy.ts`, `src/core/cost.ts`.

### 4.1 Leistungsaufnahme

Der ML.ENERGY-Benchmark misst LLM-Decoding bei etwa **20–40 % der TDP**, da die
geringe Rechenintensität große Teile der Recheneinheiten ungenutzt lässt; Prefill
nähert sich der TDP. ([Quelle](https://arxiv.org/html/2505.06371v1))

$$P_{\text{decode}} = \pi_0 + (\text{TDP} - \pi_0) \cdot u(B_{\text{batch}}), \qquad u \in [0{,}20; 0{,}55]$$

$u$ interpoliert logarithmisch mit der Batch-Größe und sättigt um Batch 64.
Prefill verwendet $u = 0{,}85$.

### 4.2 Energie pro Token

$$E_{\text{Token}} = \frac{P_{\text{GPU}}}{\text{tok/s}}, \qquad
E_{\text{Steckdose}} = \frac{(P_{\text{GPU}} + P_{\text{Host}})}{\eta_{\text{PSU}}} \cdot \text{PUE} \cdot \frac{1}{\text{tok/s}}$$

| Parameter | Vorgabe | Quelle |
| --- | --- | --- |
| $\eta_{\text{PSU}}$ | 0,90 (80+ Gold) | — |
| $P_{\text{Host}}$ | 80 W Desktop | — |
| PUE | 1,0 Heim-PC / 1,54 RZ-Mittel / 1,09 Hyperscaler | Uptime Institute 2025, Google 2025 |

### 4.3 Strompreise und CO₂

Eurostat, 2. Halbjahr 2025, EUR/kWh inklusive aller Steuern und Abgaben:

| Region | Haushalt | Gewerbe |
| --- | --- | --- |
| **Deutschland** | **0,3869** | **0,2264** |
| EU-Durchschnitt | 0,2896 | 0,1837 |

USA (EIA 2026 YTD): Haushalt 0,183 USD/kWh, Gewerbe 0,135, Industrie 0,085.

CO₂-Intensitäten sind Jahresmittel des Netzes und schwanken stark nach Stunde und
Jahreszeit — sie sind als Größenordnung zu lesen, nicht als Messwert.

### 4.4 API-Vergleich

$$c_{\text{API}} = r \cdot c_{\text{in}} + (1 - r) \cdot c_{\text{out}}$$

$r$ ist der Anteil der Input-Token und eine **Eingabegröße**, keine versteckte
Konstante: Input-Token sind meist 2–5× günstiger als Output-Token, sodass der
Mix das Ergebnis maßgeblich verschiebt.

$$n_{\text{break-even}} = \frac{\text{Anschaffungskosten}}{c_{\text{API}} - c_{\text{lokal}}}$$

Preise werden zur Laufzeit von `https://openrouter.ai/api/v1/models` geladen
(öffentlich, ohne Authentifizierung, CORS-offen). **Achtung:** Die API liefert
Preise pro *einzelnem* Token als Dezimalstring; die Umrechnung auf 1 Mio. Token
erfolgt in `src/services/openrouter.ts`. Ein mitgelieferter Snapshot dient als
Offline-Fallback.

---

## 5. Datenquellen der Modellkatalogeinträge

Der Katalog umfasst **80 Modelle** aus 18 Familien. Alle Architekturfelder wurden
per Skript aus der realen `config.json` des jeweiligen Repos gelesen, nicht aus
dem Modellnamen abgeleitet. Parameterzahlen stammen aus
`https://huggingface.co/api/models/{repo}?expand=safetensors`.
`src/data/models.ts` ist damit eine **generierte** Datei; Architekturfelder
sollten nicht von Hand geändert, sondern neu erzeugt werden.

Die Auswahl orientiert sich an den meistgenutzten offenen Modellen (HuggingFace
Downloads und Trending) sowie an den Ranglisten von apxml.com/models.

### 5.1 Herkunft der aktiven Parameter (MoE)

Drei Fälle, absteigend nach Verlässlichkeit:

1. **Publiziert** — feste Tabelle im Generator für Modelle mit offizieller
   Angabe (DeepSeek V3/R1: 37B, Kimi K2: 32B, gpt-oss 120B: 5,1B, GLM-4.5: 32B,
   Mixtral 8x7B: 12,9B …).
2. **Im Namen kodiert** — die Konvention `…-A22B` ist eindeutig und wird direkt
   übernommen (Qwen3 235B-A22B, K-EXAONE 750B-A37B, ERNIE 300B-A47B …).
3. **Geschätzt** — aus Backbone- und Expertenparametern rekonstruiert. Diese
   Einträge tragen `paramsActiveEstimated: true` und werden in der UI mit
   „geschätzt" gekennzeichnet.

Fall 3 fällt systematisch zu hoch aus, wenn die safetensors-Summe
Zusatzmodule enthält, die beim Decoding ruhen — DeepSeek V3 hat einen
MTP-Kopf von rund 14B, wodurch die Schätzung bei 54B statt 37B landet. Genau
deshalb existiert die Tabelle aus Fall 1.

**Feldnamen-Normalisierung** (`src/services/hfConfig.ts`) — jede Familie hat eine
eigene Bezeichnung für dieselben Konzepte:

| Konzept | Feldnamen |
| --- | --- |
| Anzahl Experten | `num_experts` (Qwen3), `num_local_experts` (Mixtral), `n_routed_experts` (DeepSeek) |
| Top-k | `num_experts_per_tok` (alle) |
| Experten-FFN-Breite | `moe_intermediate_size` (Qwen3, DeepSeek), `intermediate_size` (Mixtral) |
| head_dim | `head_dim`, sonst `hidden_size / num_attention_heads` |

Ein Parser, der nur eine dieser Konventionen kennt, liest die anderen Modelle
stillschweigend als dense.

---

## 6. Bekannte Einschränkungen

1. **B200-Speicher.** Quellen widersprechen sich (180 GB / 7,7 TB/s gegenüber
   192 GB / 8,0 TB/s). Der Katalog folgt den Werten des Forschungsdesigns
   (192 GB / 8,0 TB/s).
2. **RTX PRO 6000 Blackwell.** FP16-Durchsatz aus der SM-Anzahl extrapoliert,
   nicht verifiziert.
3. **Consumer-Blackwell (RTX 50).** Unklar, ob die publizierten FP16-Werte mit
   FP16- oder FP32-Akkumulation gelten.
4. **Apple M3/M4.** Bandbreiten aus Sekundärquellen; M3 Max und M4 Max sind
   binning-abhängig.
5. **Idle-Leistung von Rechenzentrums-GPUs.** Keine belastbaren publizierten
   Werte gefunden; die Angaben sind Schätzungen.
6. **Mehr-GPU-Effizienz.** Heuristik ohne publizierte geschlossene Form (§3.3).
7. **API-Preise.** Momentaufnahme; ändern sich wöchentlich. Der Live-Abruf ist
   der belastbare Pfad, der Snapshot nur Fallback.
8. **ε-Koeffizienten.** Aus TDP und Spitzenleistung abgeleitet, **nicht** gegen
   reale Messungen kalibriert. Das ist der offensichtliche nächste Schritt: die
   Werte lassen sich direkt aus NVML-Messreihen von `power_monitor.py` bestimmen.

---

## 7. Verifikation

```bash
npm run test
```

128 Tests in `src/core/__tests__/` prüfen jede Formel gegen von Hand berechnete
Referenzwerte. Die inhaltlich wichtigsten:

* Llama-3.1-8B Q4_K_M ⇒ 4,58 GB Gewichte (deckungsgleich mit llama.cpp)
* Llama-3.1-8B KV ⇒ exakt 131 072 Byte/Token
* DeepSeek-V3 MLA ⇒ exakt 70 272 Byte/Token; Regressionstest gegen die
  GQA-Fehlinterpretation
* DeepSeek-V3 `headDim` = 128 (aus `v_head_dim`), nicht 56 aus `hidden/heads`
* H100-Ridge-Point ⇒ 295,2 FLOP/Byte (FP16), 590,7 (FP8)
* MoE: Speicher folgt $P_{\text{total}}$, Geschwindigkeit $P_{\text{active}}$
* Qwen3.6 ⇒ $\rho_{\text{kv}} = 0{,}25$; KV bei 128k ist ein Viertel der
  naiven Rechnung
* Qwen2.5 ⇒ kein Sliding Window trotz gesetztem `sliding_window`
* DeepSeek V3 ⇒ publizierte 37B aktiv statt der geschätzten 54B
* Katalog-Integritätsprüfungen über alle 80 Modelle (MLA-Modelle besitzen
  `kvLoraRank`, Top-k < N, GQA hat weniger KV- als Attention-Heads,
  Output-Preis ≥ Input-Preis, jede MoE hat aktive Parameter, …)
