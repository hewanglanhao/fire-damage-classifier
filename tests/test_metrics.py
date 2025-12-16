import unittest
import torch
import numpy as np
from src.utils.metrics import ClassificationMetrics


class TestClassificationMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        metrics = ClassificationMetrics()
        preds = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 0, 1])

        metrics.update(preds, targets)
        results = metrics.compute()

        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["macro_f1"], 1.0)

    def test_imperfect_prediction(self):
        metrics = ClassificationMetrics()
        # 0, 1, 2
        # Preds:   0, 1, 0, 0, 2
        # Targets: 0, 1, 2, 0, 2
        # Correct: 1, 1, 0, 1, 1 -> 4/5 = 0.8
        preds = torch.tensor([0, 1, 0, 0, 2])
        targets = torch.tensor([0, 1, 2, 0, 2])

        metrics.update(preds, targets)
        results = metrics.compute()

        self.assertEqual(results["accuracy"], 0.8)

    def test_logits_input(self):
        metrics = ClassificationMetrics()
        # Logits for 3 classes
        preds = torch.tensor(
            [[10.0, 1.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]  # 0  # 1  # 2
        )
        targets = torch.tensor([0, 1, 2])

        metrics.update(preds, targets)
        results = metrics.compute()

        self.assertEqual(results["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
