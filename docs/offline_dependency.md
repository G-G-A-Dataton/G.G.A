# Offline Reproducibility

The accepted runtime is Python 3.13.5. `requirements.txt` pins the 17 direct
dependencies; `requirements.lock` additionally pins and hashes the complete
158-package transitive environment. Install the lock, then verify every
installed distribution without importing the large ML frameworks:

```bash
python -m pip install --require-hashes --requirement requirements.lock
PYTHONNOUSERSITE=1 python scripts/verify_environment.py \
  --lock requirements.lock
```

Build an offline wheelhouse on a connected machine:

```bash
python -m pip download --require-hashes --only-binary=:all: \
  --requirement requirements.lock --dest offline_packages
```

Create the target environment and install only from that wheelhouse:

```bash
python3.13 -m venv venv
venv/bin/python -m pip install --no-index --require-hashes \
  --find-links offline_packages --requirement requirements.lock
PYTHONNOUSERSITE=1 venv/bin/python scripts/verify_environment.py \
  --lock requirements.lock
```

The lock file is regenerated only after an intentional direct-dependency
change. It must be reviewed and the clean-environment dry-run must pass before
the update is accepted.

Canonical LightGBM/XGBoost prediction requires no network service. All fitted
models and TF-IDF artifacts are local and manifest-hashed. Verify the accepted
delivery from a clean detached clone with user-site and online model access
disabled:

```bash
PYTHONNOUSERSITE=1 venv/bin/python \
  scripts/run_reproducibility_dry_run.py --python venv/bin/python
```

The accepted 16 July result is recorded in
[`reproducibility_dry_run.md`](reproducibility_dry_run.md).

The optional sentence-transformer experiment additionally requires this local directory:

```text
models/paraphrase-multilingual-MiniLM-L12-v2/
```

Verify that optional asset with:

```bash
python scripts/verify_environment.py --lock requirements.lock \
  --require-embedding-model
```

Embedding production must use `--offline --model <local-path>`. Each term/item matrix has a manifest that binds the model identity, source hash, dimensions, IDs, normalization contract, and artifact hashes. Online API predictions and silent zero/synthetic fallbacks are prohibited.

Data availability is separately verified against `configs/final_v1.json`:

```bash
python scripts/data/verify_data_freeze.py
```
