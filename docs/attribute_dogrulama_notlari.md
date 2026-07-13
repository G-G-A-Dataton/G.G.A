# Attribute Parser Validation

**Validated:** 13 July 2026
**Implementation:** `src/attributes.py`

## Catalog Format

The production catalog uses a flat comma-separated format rather than JSON:

```text
materyal: tekstil, renk: gri, materyal bileşeni: astar: 100% polyester
```

The parser now supports this format as well as legacy Python-dict and JSON
strings. It preserves colons inside values and normalizes keys and values with
Unicode-aware case folding.

## Measured Validation

The complete `items.csv` catalog was scanned:

| Check | Result |
|---|---:|
| Catalog rows | 962,873 |
| Rows with at least one parsed attribute | 943,848 |
| Parse coverage | 98.02% |
| Full scan throughput | ~46,500 rows/second |

On the first 100,000 catalog rows, the normalized extractors returned:

| Extractor | Non-empty rows |
|---|---:|
| Color | 77,710 |
| Material | 58,323 |
| Size | 814 |

These are parser coverage measurements, not model importance or accuracy
claims. Query-side matches still depend on whether the query explicitly names
the corresponding value.

## Regression Contract

Automated tests cover:

- real flat catalog text;
- embedded colons in values;
- Python-dict and JSON compatibility;
- color/material query matches;
- complete-word matching to prevent substring false positives;
- item-text flattening for TF-IDF and embedding inputs.

Run them with:

```bash
python -m unittest discover -s tests -v
```
