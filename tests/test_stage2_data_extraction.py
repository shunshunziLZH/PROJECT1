from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from stage2.bank import build_and_save_bank, load_embeddings, validate_bank
from stage2.config import DEFAULT_CONFIG
from stage2.data import (
    AlignedBankDataset,
    build_data_manifest,
    physical_reconstruction_error,
    validate_aligned_tensors,
)
from stage2.extraction import extract_embeddings
from stage2.runtime import build_stage2_model, module_state_dicts


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _make_dataset(root: Path, count: int = 2) -> None:
    generator = np.random.default_rng(17)
    for index in range(count):
        clean = generator.integers(0, 140, size=(64, 64, 3), dtype=np.uint8)
        transmission = np.full_like(clean, 110 + index * 60)
        backscatter = np.full_like(clean, 10 + index * 30)
        degraded = np.rint(
            clean.astype(np.float32) * (transmission.astype(np.float32) / 255.0)
            + backscatter.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        name = f"{index:05d}.png"
        for field, array in (("I", degraded), ("J", clean), ("T", transmission), ("B", backscatter)):
            _save_rgb(root / field / name, array)


class Stage2DataAndExtractionTests(unittest.TestCase):
    def test_physical_consistency_checker(self) -> None:
        torch.manual_seed(2)
        J = torch.rand(3, 32, 32) * 0.6
        T = torch.rand(3, 32, 32) * 0.5 + 0.4
        B = torch.rand(3, 32, 32) * 0.1
        I = J * T + B
        error = validate_aligned_tensors(
            {"I": I, "J": J, "T": T, "B": B},
            physical_error_warn=None,
            sample_id="synthetic",
        )
        self.assertLess(error, 1e-7)
        self.assertLess(float(physical_reconstruction_error(I, J, T, B)), 1e-7)

    def test_synchronized_patch_and_tiny_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            _make_dataset(root, count=3)
            dataset = AlignedBankDataset(
                root,
                split="train",
                val_ratio=1 / 3,
                patch_size=64,
                patches_per_image=2,
                deterministic=True,
                physical_error_warn=None,
            )
            self.assertEqual(len(dataset), 4)
            sample = dataset[0]
            self.assertEqual(tuple(sample["I"].shape), (3, 64, 64))
            self.assertLess(sample["physical_error"], 0.01)

            config = deepcopy(DEFAULT_CONFIG)
            config["device"] = "cpu"
            config["data"].update(
                root=str(root),
                val_ratio=1 / 3,
                patch_size=64,
                patches_per_image=2,
                use_hflip=False,
                use_vflip=False,
                use_rot90=False,
                physical_error_warn=None,
                cache_data="none",
            )
            config["model"]["base_channels"] = 4
            config["extraction"].update(batch_size=2, num_workers=0, chunk_size=2, patches_per_image=2)
            config["bank"].update(num_prototypes=2, max_iterations=10, batch_size=4)
            config["retrieval"]["top_k"] = 2
            model = build_stage2_model(config)
            checkpoint_path = Path(directory) / "best.pt"
            torch.save(
                {
                    **module_state_dicts(model),
                    "config": config,
                    "data_manifest": build_data_manifest(config),
                },
                checkpoint_path,
            )
            output_dir = Path(directory) / "embeddings"
            manifest = extract_embeddings(config, checkpoint_path, output_dir=output_dir)
            self.assertEqual(manifest["count"], 4)
            keys, values, metadata = load_embeddings(output_dir)
            self.assertEqual(keys.shape, (4, 64))
            self.assertEqual(values.shape, (4, 128))
            self.assertEqual(len(metadata), 4)
            np.testing.assert_allclose(np.linalg.norm(keys, axis=1), 1.0, atol=1e-5)
            self.assertTrue(all(row["split"] == "train" for row in metadata))

            bank_path = Path(directory) / "neural_physics_bank_v0.pt"
            payload = build_and_save_bank(
                output_dir,
                checkpoint_path,
                bank_path,
                configuration=config,
                num_prototypes=2,
                backend="torch",
                max_iterations=10,
            )
            self.assertEqual(tuple(payload["keys"].shape), (2, 64))
            self.assertTrue(validate_bank(bank_path)["valid"])

            wrong_checkpoint = Path(directory) / "wrong.pt"
            wrong_model = build_stage2_model(config)
            torch.save(
                {
                    **module_state_dicts(wrong_model),
                    "config": config,
                    "data_manifest": build_data_manifest(config),
                },
                wrong_checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_and_save_bank(
                    output_dir,
                    wrong_checkpoint,
                    Path(directory) / "wrong_bank.pt",
                    configuration=config,
                    num_prototypes=2,
                    backend="torch",
                    max_iterations=5,
                )


if __name__ == "__main__":
    unittest.main()
