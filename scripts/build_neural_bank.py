from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.bank import EmbeddingReader, build_and_save_bank
from stage2.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster trained keys and build the offline neural bank.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Stage 2 checkpoint. Defaults to the path recorded during extraction.",
    )
    parser.add_argument("--backend", choices=("auto", "sklearn", "torch"), default="auto")
    return parser.parse_args()


def _resolve_checkpoint(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    extraction_config = EmbeddingReader(args.embeddings).load_config()
    recorded = extraction_config.get("stage2_checkpoint")
    if recorded:
        candidate = Path(str(recorded))
        if candidate.is_file():
            return candidate
    fallback = Path(config["training"]["output_dir"]) / "best.pt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        "Stage 2 checkpoint was not found. Pass --checkpoint, re-extract embeddings, "
        f"or create {fallback}."
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = _resolve_checkpoint(args, config)
    bank_config = config["bank"]
    output = args.output or Path(bank_config["output"])
    payload = build_and_save_bank(
        args.embeddings,
        checkpoint,
        output,
        configuration=config,
        num_prototypes=int(bank_config["num_prototypes"]),
        trim_fraction=float(bank_config["trim_fraction"]),
        batch_size=int(bank_config["batch_size"]),
        max_iterations=int(bank_config["max_iterations"]),
        seed=int(bank_config["seed"]),
        backend=args.backend,
    )
    print(
        f"Saved {payload['keys'].shape[0]} prototypes to {output} "
        f"(key_dim={payload['keys'].shape[1]}, value_dim={payload['values'].shape[1]})"
    )


if __name__ == "__main__":
    main()

