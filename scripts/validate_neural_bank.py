from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.bank import validate_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a neural physics bank checkpoint.")
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_bank(args.bank, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

