# Model Validation Status

**Current status (13 July 2026): no production model is validated yet.**

Earlier experiment scores were produced before three contract fixes:

1. validation rows from the same `term_id` could appear in train and validation;
2. sampled negative generation could label a known positive outside the sample as negative;
3. the catalog's flat `attributes` format was not parsed, so attribute features were inactive.

Those scores and thresholds remain historical records only. They must not be
used for model selection, leaderboard projections, or production inference.

The current acceptance contract is:

- negatives exclude every pair in the full positive reference set;
- every positive pair receives the configured per-term negative quota;
- validation uses `StratifiedGroupKFold(group=term_id)`;
- the threshold is optimized from the new grouped OOF predictions;
- production artifacts include a full-training manifest and verified SHA-256 hashes;
- the final submission passes `src.validate_submission` against the sample file.

Re-establish a validated baseline with:

```bash
python -m unittest discover -s tests -v
python scripts/training/run_train_full_v2.py
python scripts/submission/run_pipeline.py --mode predict
```

After the full training run, record its grouped fold scores, threshold, manifest
hashes, and any real Kaggle score in `docs/experiment_log.md` as a new experiment.
