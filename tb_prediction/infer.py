from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_prediction.data import IMAGE_EXTENSIONS, load_rgb, save_rgb
from tb_prediction.checkpoint import load_checkpoint_file
from tb_prediction.model import TBResUNet


def find_inputs(path: Path, recursive: bool = False) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    iterator = path.rglob("*") if recursive else path.iterdir()
    files = [item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def load_model(checkpoint_path: Path, device: torch.device, base_channels: Optional[int] = None) -> TBResUNet:
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    model_config: Dict[str, object] = {}
    if isinstance(checkpoint, dict):
        model_config = dict(checkpoint.get("model_config", {}))
    if base_channels is not None:
        model_config["base_channels"] = base_channels
    model = TBResUNet(**model_config).to(device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_from_image(path: Path, device: torch.device) -> torch.Tensor:
    array = load_rgb(path)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer T and B maps with a trained TBResUNet checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory of images.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tb_prediction"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-channels", type=int, default=None, help="Override base channels for plain state dict checkpoints.")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device, base_channels=args.base_channels)
    inputs = find_inputs(args.input, recursive=args.recursive)
    if not inputs:
        raise RuntimeError(f"No input images found under {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for input_path in inputs:
            image = tensor_from_image(input_path, device)
            outputs = model(image)
            save_rgb(outputs["T"][0], args.output_dir / f"{input_path.stem}_T.png")
            save_rgb(outputs["B"][0], args.output_dir / f"{input_path.stem}_B.png")
            print(f"Saved {input_path.stem}_T.png and {input_path.stem}_B.png")


if __name__ == "__main__":
    main()
