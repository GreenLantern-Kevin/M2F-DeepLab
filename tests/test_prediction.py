"""Verify that evaluation uses the model's per-pixel class decisions."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from utils.engine import predict_probabilities, probabilities_to_masks


class FixedLogits(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.logits


class PredictionTests(unittest.TestCase):
    def test_single_forward_and_pixelwise_class_decisions(self):
        logits = torch.zeros(1, 2, 5, 5)
        logits[:, 0] = 1
        logits[0, 1, 2, 2] = 2
        model = FixedLogits(logits)
        rgb = torch.zeros(1, 3, 5, 5)
        output, probs = predict_probabilities(
            SimpleNamespace(model='single_deeplabv3plus_xception16'),
            model, rgb, rgb, torch.tensor([False]))
        self.assertEqual(model.calls, 1)
        torch.testing.assert_close(output, logits)
        torch.testing.assert_close(probs, logits.softmax(1))
        mask = probabilities_to_masks(probs)
        expected = np.zeros((1, 5, 5), dtype=np.uint8)
        expected[0, 2, 2] = 1
        np.testing.assert_array_equal(mask, expected)

    def test_equal_probabilities_choose_first_class(self):
        probs = torch.full((1, 2, 3, 3), 0.5)
        self.assertFalse(probabilities_to_masks(probs).any())


if __name__ == '__main__':
    unittest.main()
