"""Create the deploy decision used by the official offline inference step."""

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.solution_contract import write_deploy_decision


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a verified deploy decision")
    parser.add_argument("--model-dump-path", required=True)
    parser.add_argument("--competition-data-path", required=True)
    parser.add_argument("--allow-sample", action="store_true")
    args = parser.parse_args(argv)
    output_path, decision = write_deploy_decision(
        os.path.abspath(args.model_dump_path),
        os.path.abspath(args.competition_data_path),
        require_full=not args.allow_sample,
    )
    print(
        f"decision={output_path} selected={decision['deploy']['selected_model']} "
        f"threshold={decision['deploy']['threshold']:.8f}"
    )
    return output_path


if __name__ == "__main__":
    main()
