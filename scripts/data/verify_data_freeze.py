"""CLI for the versioned final dataset freeze."""

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data_freeze import verify_data_freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify final source data hashes")
    parser.add_argument(
        "--config", default=os.path.join(PROJECT_ROOT, "configs", "final_v1.json")
    )
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "datasets"))
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args(argv)
    result = verify_data_freeze(args.config, args.data_dir, args.chunk_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
