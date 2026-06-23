# Surya-on-M4 speedup — findings & learnings

Investigation log for making Surya OCR (`surya-ocr-2`, 650M Qwen3-VL-style VLM) fast on
Apple Silicon, driven by parsing scanned Manipur electoral-roll PDFs. **Mission: speed.**
Field accuracy (e.g. EPIC ids) is out of scope as a *goal*, but every speed variant is
scored against unmodified f16 Surya output (the "gold standard") so we know the cost.

Environment: **M4 / 16 GB**, macOS, `llama.cpp` backend (auto-selected on Apple Silicon),
`surya-ocr` 0.20, model + mmproj GGUF cached from `datalab-to/surya-ocr-2-gguf`.
Data: `electoral_rolls/manipur/eroll2025/pdfs/english/` — **1,756 PDFs × ~14 pp ≈ 24k pages**.
Harnesses: `bench/bench_surya.py`, `bench/compare_quality.py`, `bench/prompt_tokens.py`.

---

## Measured baseline

| Metric | Value | Source |
|---|---|---|
| Warm voter page, real Surya pipeline (f16) | **177.8 s/page** | `compare_quality.py`, AC01 pp3-5 |
| Vision encode (image → tokens) | **~5 s/page** | `llama-mtmd-cli` |
| ⇒ Decode share | **~95% of wall time** | derived |
| Gold output size | ~7,000–9,200 HTML chars/page | block dumps |
| Rendered page @192 dpi | 1587 × 2246 px | render |
| Model: f16 GGUF | 1,206 MiB (16.0 BPW) | `llama-quantize` |
| Model: Q4_K_M GGUF | 384 MiB (5.1 BPW) | `llama-quantize` |

A page's structure (full-page mode): page 1 = cover, page 2 = **maps/photos** (correctly
`Picture`/skipped), pages 3+ = one `<table>` block of voters + headers/footers.

---

## Decode throughput measurements (the cost driver)

**Pure-text decode scales with concurrency** (`llama-batched-bench`, machine cool):

| Batch | tok/s | vs B=1 |
|---|---|---|
| 1 | 63.7 | 1.0× |
| 2 | 105.8 | 1.7× |
| 4 | 167.6 | 2.6× |
| 8 | 198.7 | **3.1×** |

**Real multimodal decode is much slower** and **thermally sensitive** (`llama-mtmd-cli`,
one real page, fixed 600 tokens, `--ignore-eos`):

| State | f16 tok/s | Q4 tok/s | Q4 / f16 |
|---|---|---|---|
| Cool (early) | 48.1 | 67.8 | **1.41×** |
| Hot (after long run) | 14.4 / 17.4 | 19.0 / 28.4 | ~1.3–1.6× |

The image injects a large token context (Qwen-VL ≥1024 image tokens), so real-page decode
runs ~3–4× slower than pure-text decode even when cool.

---

## MLX backend — the big win ✅

mlx-vlm 0.6.3 supports Surya's `qwen3_5` architecture directly (no port needed). Converted
`datalab-to/surya-ocr-2` → MLX 4-bit (496 MB, 6.25 bpw, `models/surya-mlx-4bit/`). Real page,
`bench/bench_mlx.py`:

| Backend | decode tok/s | vs llama.cpp f16 |
|---|---|---|
| llama.cpp f16 (cool) | 48 | 1.0× |
| llama.cpp Q4 (cool) | 68 | 1.4× |
| **MLX 4-bit** | **188.6** | **~3.9×** |

- prefill 666 tok/s, **peak memory only 1.96 GB** (huge headroom on 16 GB).
- Output correct: clean voter `<table>`, well-spaced names, EPICs captured.
- **End-to-end confirmed** (`bench/mlx_quality.py`, full pages): page 3 = 7,723 tok in
  **49.5 s** (175 tok/s); page 4 = 4,531 tok in **30.6 s** (181 tok/s). vs **177.8 s/page gold
  → ~3.6× faster**, and **sustains ~175–181 tok/s across pages** (not just burst).
- **Accuracy vs f16 gold:** name 100%, rel/house/age/sex ~97% on the cleanly-parsed page
  (page 4: 30/30 name, 29/30 rel, 30/30 house/age/sex). Page-3 low match is a *harness*
  artifact (MLX emitted a different valid table layout; the `extract_voters` regex mis-keyed
  it), not a Surya error.
- ⇒ **MLX 4-bit is the single biggest lever found** (~3.6× end-to-end, <2 GB), before token
  reduction or distillation.
- **Full-PDF end-to-end** (`savitr/mlx_ocr.py`, all 14 pages of AC01_part001):
  **475 s = 33.9 s/page → ~5.2× faster than gold** (light cover/maps pages pull the average
  down further). Sustained ~175 tok/s with no thermal collapse over the full PDF.
- A **layout-robust voter parser** (`parse_voters`, anchors on `Name :`, requires age/gender,
  excludes header/relation matches) extracts ~28–37 voters/page with correct
  name/relation/house/age/sex. (MLX emits >1 valid table layout; cell-structure parsers break,
  this one doesn't.) Open: full-state thermal behavior; tighten count vs MLX output + serial dedup.

## Releasable Surya MLX backend (#3) ✅

`savitr/mlx_backend.py` is a `Backend` subclass that adds `method="mlx"` to
`SuryaInferenceManager` — it spawns mlx-vlm's OpenAI server (its own subprocess/venv, no dep
conflict) and reuses Surya's `chat_completions_batch`, so the **full Surya pipeline runs
unchanged on MLX**. Validated: `SuryaInferenceManager(method="mlx")` + `RecognitionPredictor`
→ full-page OCR returned 8 parsed layout blocks with the correct voter table. `PR.md` has the
exact upstream diff (new `backends/mlx.py` + a 3-line `_build_backend` branch + settings).
Bonus: mlx-vlm's server also exposes vision-feature caching and `--draft-model` (speculative)
for future speedups.

## Pipeline integration (#2) — MLX → canonical voter CSV ✅

`savitr/parse_manipur_mlx.py` OCRs a PDF with the MLX engine, parses voters
layout-robustly, and reuses the repo's `fields.py` (cover metadata) + column schema, so the
output is drop-in compatible with `parse_manipur_2025.py`. Full PDF (AC01_part001):

| metric | value |
|---|---|
| time | 429 s (~31 s/page) — **~5× the RapidOCR/llama.cpp path** |
| voters | **268** (declared net ≈ 289 → ~93% recall) |
| EPIC ids | 265/268, **100% unique, 0 duplicate-EPIC rows** |
| serials | 213/268 |
| fields | name/relation/house/age/sex correct |

`parse_voters` handles the two table layouts the model emits + de-loops decode repeats;
`dedupe_voters` keys by EPIC (unique) then identity. **Known limits** (all root-caused to the
VLM's inconsistent free-form output, which distillation #1 fixes): ~7% recall gap and partial
serials on looped layout-B pages; cover-metadata regexes (net electors, pin, polling station)
need tuning for the MLX cover HTML.

## Q4 vs f16 quality (gold = f16)

`compare_quality.py`, AC01 pp3-5, per-field exact match of Q4 vs f16 gold:

| Field | match |
|---|---|
| name | 95.5% |
| father/husband | 94.3% |
| house | 96.6% |
| age | 96.6% |
| sex | 96.6% |
| text similarity | 81.3% |

⇒ **Q4 keeps ~95% field fidelity** — an acceptable accuracy cost for the speedup.

---

## Token-reduction lever (compact vs bbox prompt)

The gold pipeline uses `HIGH_ACCURACY_BBOX_PROMPT`, which makes the model emit a
`data-bbox` coordinate set for every div — tokens we pay for but never use (we parse the
voter table, not bboxes). `prompt_tokens.py` compares tokens generated by the gold prompt
vs a compact `"OCR this image to HTML."` prompt, same model/session.

**RESULT (AC01 pp3-4, f16, sequential): the compact prompt does NOT reliably save tokens.**

| page | bbox tok | compact tok | reduction |
|---|---|---|---|
| 3 | 6,047 | 8,744 | 0.69× (compact *worse*) |
| 4 | 8,553 | 3,769 | 2.27× |
| **total** | 14,600 | 12,513 | **1.17× (inconsistent)** |

Compact-vs-gold field match stayed high (name 98%, rel 96%, house/age/sex 98%) — so the
compact prompt is **accurate but not faster**. The hypothesis was wrong: `data-bbox` overhead
is only ~10 tokens × ~7 blocks/page ≈ negligible. **The real token cost is the verbose voter
HTML content** — `<td><p><b>N</b></p><p>Name : … <br/>Father's Name: … <br/>House Number : …`
repeated 30×/page (~6–8k tokens). The labels/tags are the overhead, not the prompt.

### Guided JSON schema — also fails (the model only does its trained tasks)

Forcing a compact per-voter JSON array via Surya `guided_json` (`bench/guided_schema.py`):

| variant | output | tokens | voters |
|---|---|---|---|
| guided JSON (strict schema) | `[ ]` (empty array) | 26 | 0 |
| same prompt, **unguided** | reverts to trained **layout** format: `[{"label":"Table","bbox":"…","count":6100}, …]` | 128 | n/a |

The model is an **OCR engine trained on fixed prompts** (OCR→HTML, layout JSON). Constrained
to an unfamiliar schema it emits the trivial empty array; given the same instruction unguided
it ignores it and falls back to its trained layout output. (Aside: that layout `count:6100`
is the model's *own* token estimate for the voter table — confirming the ~6–8k tokens/page is
real content, not waste.)

⇒ **You cannot prompt- or schema-engineer this model into terser output. Reducing output
tokens requires fine-tuning / distillation** (train a model that emits compact records).
Off-the-shelf, the only working speed lever is **Q4 quant (~1.4×)**; everything else (compact
prompt, guided schema, parallelism) was tested and failed.

---

## Learnings

**Methodology**
1. **Thermal throttling is a ~3× confound.** The identical fixed-length decode ran at 48
   tok/s cool vs 14 tok/s hot. Any sequential A/B where one variant runs second (after a
   long first run) is biased — the Q4 "2× slower" pipeline result was *entirely* this
   artifact, not a real regression. **Compare by `token_count` (thermal-independent), or
   interleave variants, or cooldown between runs.**
2. **Per-token speed ≠ end-to-end speed.** What dominates wall-clock is *tokens generated*
   (and whether Surya's slow layout+block fallback triggers), not raw tok/s.
3. **Trust on-disk reality, not summaries.** The repo ships RapidOCR only (no `SuryaEngine`);
   `fields.py` is geometry-based and engine-agnostic; full-page Surya emits one HTML
   `<table>`/page, not per-line bboxes. Two earlier assumptions were wrong until measured.
4. **pdf2image images are lazy** — force `.convert("RGB")` before threaded use or they
   truncate.

**Engineering / strategy**
5. **Warming the server is NOT a 13× win.** The 126 s/page was ~108 s one-time cold start +
   page; warm steady-state is still ~110–178 s/page. The bottleneck is decode, not startup.
6. **The dominant lever is fewer decoded tokens — but the model can't be talked into it.**
   The cost is the verbose labeled voter HTML (~6–8k tokens/page), not the prompt or bbox
   overhead. A compact prompt didn't help (1.17×, inconsistent). A guided JSON schema emitted
   an empty array (the model only does its trained OCR→HTML / layout tasks). **So cutting
   output tokens requires training the model (distillation/fine-tune), not prompting.**
7. **Q4 quant is a real ~1.3–1.5× win**, stackable, ~95% field fidelity. Keep it.
8. **Parallelism does not help here and can hurt.** Decode *throughput* scales with
   concurrency only for light pure-text loads; on dense real pages the M4's thermal/compute
   ceiling means concurrent requests split throughput, and at high concurrency they thrash
   and exceed the 600 s/request timeout (then retry, compounding). Keep concurrency low.
9. **MLX changes the scale math.** At gold 178 s/page, Manipur's ~24k pages ≈ ~49 days/M4.
   At MLX's ~40 s/page that's **~11 days/M4** — and MLX uses <2 GB and far less power (likely
   less throttling). Stack distillation + multiple machines for all-India.
10. **(Out of scope, noted)** The gold `HIGH_ACCURACY_BBOX` prompt drops/mangles ~90% of
    EPIC ids on this layout (treats the top-right EPIC as a phantom voter name); the plain
    prompt actually captured EPICs better. Relevant if accuracy re-enters scope.

---

## Lever ranking (what to pursue)

1. **MLX 4-bit backend — CONFIRMED ~3.6× end-to-end** (49.5 s vs 177.8 s/page), sustains
   ~175–181 tok/s, ~97–100% field accuracy, <2 GB. **The single biggest win.** Next:
   productionize as the engine + a layout-robust voter parser; releasable as a Surya MLX PR.
2. **Distillation / fine-tune (mlx-tune, local)** — the ONLY way to cut output tokens
   (the dominant cost). Compact records + fewer params → faster AND cooler. Stacks on MLX.
   Highest effort, highest ceiling; the durable lever for local all-India.
3. **Q4_K_M GGUF** — ~1.3–1.5×; now superseded by MLX, but a fallback if staying on llama.cpp.
4. ~~Compact prompt~~ — tested, does not help (1.17×, inconsistent).
5. ~~Guided JSON schema~~ — tested, emits empty array (model only does its trained tasks).
6. ~~Parallelism~~ — tested, hurts on dense pages on this hardware.

## Reproduce
```bash
cd ~/Documents/GitHub/savitr && source .venv/bin/activate
PDF=…/electoral_rolls/manipur/eroll2025/pdfs/english/AC01_part001_final_ENG.pdf
python bench/bench_surya.py "$PDF" --pages 3-5            # timing + structure
python bench/compare_quality.py run "$PDF" --pages 3-5 --out gold.json
python bench/compare_quality.py run "$PDF" --pages 3-5 \
    --model models/surya-2-Q4_K_M.gguf --mmproj <mmproj.gguf> --out q4.json
python bench/compare_quality.py cmp gold.json q4.json     # speed × accuracy vs gold
python bench/prompt_tokens.py "$PDF" --pages 3-4          # token-reduction lever
```
