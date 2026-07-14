from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from stage2.bank import build_bank, save_bank_checkpoint, validate_bank
from stage2.config import DEFAULT_CONFIG
from stage2.retrieval import BankRetriever, retrieve
from stage2.runtime import build_stage2_model, module_state_dicts


class BankBuildAndRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(11)
        centers = generator.normal(size=(8, 64)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        self.keys = np.concatenate(
            [center[None] + generator.normal(0, 0.01, size=(5, 64)) for center in centers], axis=0
        ).astype(np.float32)
        self.values = generator.normal(size=(40, 128)).astype(np.float32)

    def test_eight_prototype_build_and_validation(self) -> None:
        bank = build_bank(
            self.keys,
            self.values,
            num_prototypes=8,
            backend="torch",
            max_iterations=20,
            seed=3,
        )
        self.assertEqual(tuple(bank["keys"].shape), (8, 64))
        self.assertEqual(tuple(bank["values"].shape), (8, 128))
        self.assertTrue(torch.all(bank["cluster_count"] > 0))
        self.assertEqual(int(bank["cluster_count"].sum()), 40)
        self.assertTrue(torch.isfinite(bank["keys"]).all())
        self.assertTrue(torch.isfinite(bank["values"]).all())
        torch.testing.assert_close(bank["keys"].norm(dim=1), torch.ones(8), atol=1e-5, rtol=1e-5)
        report = validate_bank(bank)
        self.assertTrue(report["valid"])
        self.assertEqual(report["empty_clusters"], 0)

    def test_final_checkpoint_and_global_token_retrieval(self) -> None:
        bank = build_bank(self.keys, self.values, num_prototypes=8, backend="torch", max_iterations=10)
        config = deepcopy(DEFAULT_CONFIG)
        config["model"]["base_channels"] = 4
        config["bank"]["num_prototypes"] = 8
        config["retrieval"]["top_k"] = 4
        config["split"] = "train"
        config["training_split_only"] = True
        model = build_stage2_model(config)
        checkpoint = {**module_state_dicts(model), "config": config}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neural_physics_bank_v0.pt"
            payload = save_bank_checkpoint(path, bank, checkpoint, config)
            report = validate_bank(path)
            self.assertTrue(report["valid"])
            self.assertEqual(report["number_of_prototypes"], 8)
            self.assertTrue(payload["configuration"]["training_split_only"])

            query = payload["keys"][:2]
            result = retrieve(query, payload["keys"], payload["values"], top_k=4, temperature=0.1)
            repeated = retrieve(query, payload["keys"], payload["values"], top_k=4, temperature=0.1)
            retrieved, indices, weights, similarities = result
            self.assertEqual(tuple(retrieved.shape), (2, 128))
            self.assertEqual(tuple(indices.shape), (2, 4))
            self.assertEqual(tuple(similarities.shape), (2, 4))
            torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
            for first, second in zip(result, repeated):
                torch.testing.assert_close(first, second)

            token_query = query[:, None, :].expand(2, 3, 64)
            token_result = BankRetriever.from_checkpoint(path)(token_query)
            self.assertEqual(tuple(token_result[0].shape), (2, 3, 128))
            self.assertEqual(tuple(token_result[1].shape), (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
