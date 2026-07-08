from __future__ import annotations

import os
import pathlib
from pathlib import Path
from typing import Any

import torch


def load_checkpoint_file(path: Path, map_location: Any) -> Any:
    """Load TB checkpoints saved with full training metadata.

    PyTorch 2.6 defaults torch.load to weights_only=True. These checkpoints
    store argparse/path metadata in addition to tensors, so they need full
    unpickling. Some existing checkpoints were created on Windows and contain
    WindowsPath objects, which need a small compatibility alias on Linux.
    """

    original_windows_path = pathlib.WindowsPath
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
    finally:
        pathlib.WindowsPath = original_windows_path
