import unittest

import torch

torch.set_num_threads(2)

from utils.loss import BoundaryDiceLoss, FocalLoss, LandslideDiceLoss, SegmentationCriterion


class LossTests(unittest.TestCase):
    def test_all_ignored_has_zero_gradient(self):
        logits = torch.randn(2, 2, 3, 3, requires_grad=True)
        labels = torch.full((2, 3, 3), 255, dtype=torch.long)
        for name in SegmentationCriterion.LOSS_TYPES:
            value = SegmentationCriterion(name)(logits, labels)
            self.assertEqual(value.item(), 0)
            value.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_padding_does_not_change_segmentation_loss(self):
        logits = torch.tensor([[[[1., -1.]], [[-1., 1.]]]], requires_grad=True)
        labels = torch.tensor([[[0, 1]]])
        padded = torch.cat([logits, torch.full_like(logits, 100)], dim=-1)
        padded_labels = torch.tensor([[[0, 1, 255, 255]]])
        for loss in [FocalLoss(), LandslideDiceLoss(), SegmentationCriterion()]:
            torch.testing.assert_close(loss(logits, labels), loss(padded, padded_labels))

    def test_boundary_padding_is_ignored(self):
        logits = torch.tensor([[[[0., 1., 100.]]]], requires_grad=True)
        labels = torch.tensor([[[[0., 1., 255.]]]])
        criterion = BoundaryDiceLoss()
        value = criterion(logits, labels)
        torch.testing.assert_close(value, criterion(logits[..., :2], labels[..., :2]))
        value.backward()
        self.assertEqual(logits.grad[..., 2].item(), 0)
        self.assertNotEqual(logits.grad[..., 1].item(), 0)

    def test_combined_criterion_does_not_add_an_edge_term(self):
        logits = torch.randn(2, 2, 3, 3)
        labels = torch.randint(0, 2, (2, 3, 3))
        expected = .3 * FocalLoss()(logits, labels) + .8 * LandslideDiceLoss()(logits, labels)
        torch.testing.assert_close(SegmentationCriterion()(logits, labels), expected)


if __name__ == "__main__":
    unittest.main()
