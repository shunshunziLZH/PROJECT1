from __future__ import annotations

import unittest

import torch
from torch import nn

from stage2.losses import (
    BankLossWeights,
    BankPretrainingLoss,
    build_value_invariance_pair,
    shuffle_values,
)
from stage2.models import BankPretrainingModel


class BankModelShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = BankPretrainingModel(
            patch_size=64,
            key_dim=64,
            value_dim=128,
            projection_dim=64,
            encoder_base_channels=8,
            restorer_base_channels=8,
        )
        self.J = torch.rand(2, 3, 64, 64) * 0.6
        self.T = torch.rand(2, 3, 64, 64) * 0.4 + 0.5
        self.B = torch.rand(2, 3, 64, 64) * 0.1
        self.I = self.J * self.T + self.B

    def test_required_shapes_and_no_batch_norm(self) -> None:
        J_aug, I_aug = build_value_invariance_pair(self.J, self.T, self.B)
        outputs = self.model(
            self.I,
            self.J,
            self.T,
            self.B,
            self.T * 0.95,
            self.B * 1.02,
            J_aug,
            I_aug,
        )
        self.assertEqual(outputs["q"].shape, (2, 64))
        self.assertEqual(outputs["q_raw"].shape, (2, 64))
        self.assertEqual(outputs["q_pred"].shape, (2, 64))
        self.assertEqual(outputs["v"].shape, (2, 128))
        self.assertEqual(outputs["v_aug"].shape, (2, 128))
        self.assertEqual(outputs["P_reconstructed"].shape, (2, 9, 64, 64))
        self.assertEqual(outputs["J_temp"].shape, (2, 3, 64, 64))
        self.assertFalse(any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in self.model.modules()))

    def test_full_backward_step_is_finite(self) -> None:
        J_aug, I_aug = build_value_invariance_pair(self.J, self.T, self.B)
        outputs = self.model(
            self.I,
            self.J,
            self.T,
            self.B,
            self.T * 0.97,
            self.B * 1.01,
            J_aug,
            I_aug,
        )
        wrong = self.model.temporary_restorer(self.I, shuffle_values(outputs["v"]))
        criterion = BankPretrainingLoss(
            BankLossWeights(ranking=0.1),
            physical_relation_size=4,
        )
        components = criterion(outputs, self.J, wrong_restoration=wrong, enable_alignment=True)
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        components["total"].backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()

