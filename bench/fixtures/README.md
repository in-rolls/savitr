# fixtures

Reference OCR outputs used by `bench/compare_quality.py cmp` and friends.

These are **gitignored** (`*.json`): the real gold/Q4 fixtures contain actual voter records
(names, EPIC ids) — PII — so they stay local and are never committed. Regenerate your own
from any roll PDF:

```bash
python bench/compare_quality.py run roll.pdf --pages 3-5 --out bench/fixtures/gold.json
```
