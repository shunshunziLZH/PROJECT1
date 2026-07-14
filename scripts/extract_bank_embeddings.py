from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.config import load_config
from stage2.extraction import extract_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Stage 2 neural key-value training samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    manifest = extract_embeddings(
        config,
        args.checkpoint,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"Embedding manifest: count={manifest['count']}, shards={len(manifest['shards'])}")


if __name__ == "__main__":
    main()

