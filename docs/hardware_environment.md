# Hardware And Environment

## Development Machine

| Component | Observed configuration |
|---|---|
| CPU | AMD Ryzen 7 6800H, 8 physical cores / 16 threads |
| RAM | 27 GiB visible to the operating environment |
| GPU | No working discrete GPU required or used by the accepted pipeline |
| Architecture | x86_64 |
| Operating system | LMDE 7, Linux 6.12.95+deb13-amd64 |
| Storage | Local NVMe filesystem; at least 5 GiB free workspace recommended |

The accepted full training run used eight model threads, completed in
48 minutes 28 seconds, and recorded peak RSS of 3,765,812 KiB. Inference is
CPU-only and uses disk-backed feature arrays so the full 3,359,679-row test set
is not materialized as one pandas frame.

## Software Contract

- Python `3.13.5`
- 158 transitive Python distributions pinned with hashes in
  `requirements.lock`
- LightGBM `4.6.0`
- XGBoost `3.0.2`
- scikit-learn `1.7.0`
- pandas `2.3.0`
- NumPy `2.3.1`

`step1.sh` creates and verifies this environment. `step3.sh` explicitly sets
offline flags and does not download models, tokenizers, or data.
