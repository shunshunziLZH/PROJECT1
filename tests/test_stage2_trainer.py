from __future__ import annotations

import tempfile
import unittest
import warnings
from copy import deepcopy
from pathlib import Path

import torch

from stage2.config import DEFAULT_CONFIG
from stage2.trainer import train
from stage2.utils import load_checkpoint
from test_stage2_data_extraction import _make_dataset


class Stage2TrainerTests(unittest.TestCase):
    def test_warmup_joint_best_checkpoint_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, output_dir = root / "data", root / "output"
            _make_dataset(data_root, count=3)
            config = deepcopy(DEFAULT_CONFIG)
            config["device"] = "cpu"
            config["data"].update(
                root=str(data_root),
                val_ratio=1 / 3,
                patch_size=64,
                patches_per_image=1,
                train_limit=2,
                val_limit=1,
                physical_error_warn=None,
                use_hflip=False,
                use_vflip=False,
                use_rot90=False,
            )
            config["model"]["base_channels"] = 4
            config["training"].update(
                output_dir=str(output_dir),
                batch_size=2,
                epochs=2,
                warmup_epochs=1,
                num_workers=0,
                use_amp=False,
                val_frequency=1,
                diagnostic_frequency=0,
                log_frequency=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                metrics = train(config)
            self.assertTrue((output_dir / "last.pt").is_file())
            self.assertTrue((output_dir / "best_warmup.pt").is_file())
            self.assertTrue((output_dir / "best.pt").is_file())
            warmup = load_checkpoint(output_dir / "best_warmup.pt")
            joint = load_checkpoint(output_dir / "best.pt")
            self.assertEqual(warmup["epoch"], 1)
            self.assertEqual(joint["epoch"], 2)
            self.assertIn("data_manifest", joint)
            self.assertIn("rng_state", joint)
            self.assertFalse(set(joint["data_manifest"]["train_ids"]) & set(joint["data_manifest"]["val_ids"]))
            changed = any(
                not torch.equal(warmup["key_projector"][name], joint["key_projector"][name])
                for name in warmup["key_projector"]
            )
            self.assertTrue(changed, "joint alignment should update the key projector")
            self.assertEqual(metrics["best_joint_validation_total"], joint["metrics"]["best_joint_validation_total"])

            resumed = deepcopy(config)
            resumed["training"]["resume"] = str(output_dir / "last.pt")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                resumed_metrics = train(resumed)
            self.assertEqual(
                resumed_metrics["best_joint_validation_total"], metrics["best_joint_validation_total"]
            )

            incompatible = deepcopy(resumed)
            incompatible["training"]["epochs"] = 3
            with self.assertRaisesRegex(ValueError, "training.epochs"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    train(incompatible)


if __name__ == "__main__":
    unittest.main()

