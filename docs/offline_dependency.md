# Offline Reproducibility

The exact runtime is Python 3.13.5 with every package pinned in `requirements.txt`. Verify it without importing large ML frameworks:

```bash
python scripts/verify_environment.py
```

Build an offline wheelhouse on a connected machine:

```bash
python -m pip download --only-binary=:all: \
  --requirement requirements.txt --dest offline_packages
python -m pip install --no-index --find-links offline_packages \
  --requirement requirements.txt
```

Canonical LightGBM/XGBoost prediction requires no network service. All fitted models and TF-IDF artifacts are local and manifest-hashed.

The optional sentence-transformer experiment additionally requires this local directory:

```text
models/paraphrase-multilingual-MiniLM-L12-v2/
```

Verify that optional asset with:

```bash
python scripts/verify_environment.py --require-embedding-model
```

Embedding production must use `--offline --model <local-path>`. Each term/item matrix has a manifest that binds the model identity, source hash, dimensions, IDs, normalization contract, and artifact hashes. Online API predictions and silent zero/synthetic fallbacks are prohibited.

Data availability is separately verified against `configs/final_v1.json`:

```bash
python scripts/data/verify_data_freeze.py
```
