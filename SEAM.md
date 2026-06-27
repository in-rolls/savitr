# SEAM — where savitr sits in the electoral-roll OCR pipeline

Three sibling repos, one job each. Keep the boundaries; don't duplicate the shared core.

```
electoral_rolls          scraping only: CEO sites -> scanned PDFs
  manipur/<year>/manipur.py                     (e.g. manipur/2025/)
        |
        v   PDFs
parse_unsearchable_rolls   the SHARED roll-parsing core + non-Surya engines + benchmark
  scripts/manipur/fields.py        <-- the one shared core: COLUMNS schema,
                                       cover-page metadata, geometry voter parser
  scripts/manipur/ocr_engines.py   <-- RapidOCR / PaddleOCR engines
  scripts/manipur/benchmark/       <-- cross-engine field-level comparison (score.py)
        ^
        |   imports fields.py (COLUMNS + cover metadata)
savitr (this repo)        make Surya FAST on Apple Silicon + Surya-specific parsing
  savitr/mlx_backend.py     releasable Surya MLX Backend (upstream candidate)
  savitr/mlx_ocr.py         MLX Surya engine + its OWN voter parser/dedup (VLM-tuned)
  savitr/parse_manipur_mlx.py  PDF -> voter CSV via MLX (byte-compatible with the parse repo)
  bench/                    Surya speed + accuracy-vs-f16 benchmarks
```

## The rule
`parse_unsearchable_rolls/scripts/manipur/fields.py` is the **single** source of the
roll-generic core: the `COLUMNS` schema, cover-page metadata regexes, and the canonical
geometry voter parser. **Never copy it.** savitr imports it (see below).

**Engine-specific glue is NOT the shared core** and may live with each engine:
- savitr's `mlx_ocr.parse_voters` + `dedupe_voters` are tuned to the VLM's HTML output (key by
  EPIC; collapse looped rows) — different from `fields.py`'s serial-keyed dedup for the geometry
  path. Both are correct for their engine; keeping them separate is intended, not duplication.

## How savitr depends on the parse repo
`parse_manipur_mlx.py` adds the parse repo to `sys.path` and imports `fields` + `COLUMNS`:
```python
MANIPUR_DIR = os.environ.get(
    "MANIPUR_DIR",
    "<abs path>/parse_unsearchable_rolls/scripts/manipur")  # override via env if relocated
```
Dependency flows **savitr -> parse_unsearchable_rolls** (for the shared schema/metadata) and
**parse_unsearchable_rolls -> savitr** (the benchmark consumes savitr's MLX engine as its
fast Surya reference). No code is copied in either direction.

## Data flow
scrape (electoral_rolls) → PDFs → parse → voter CSV (shared `COLUMNS`), where "parse" is
RapidOCR/PaddleOCR in the parse repo **or** Surya-MLX here — all on the one `fields.py` schema.
