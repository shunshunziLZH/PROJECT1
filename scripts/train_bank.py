from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.config import load_config
from stage2.trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 neural physics bank encoders.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.resume is not None:
        config["training"]["resume"] = str(args.resume)
    if args.device is not None:
        config["device"] = args.device
    train(config)


if __name__ == "__main__":
    main()

