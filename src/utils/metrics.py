import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class ClassificationMetrics:
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        """
        Args:
            preds: (batch_size, num_classes) logits or (batch_size,) class indices
            targets: (batch_size,) class indices
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()
            if preds.dim() > 1:
                preds = torch.argmax(preds, dim=1)
            preds = preds.numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = np.concatenate(self.preds)
        targets = np.concatenate(self.targets)

        labels = None
        if self.num_classes is not None:
            labels = list(range(self.num_classes))

        acc = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0, labels=labels
        )

        # Weighted metrics are also useful for imbalanced datasets
        w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
            targets, preds, average="weighted", zero_division=0, labels=labels
        )

        cm = confusion_matrix(targets, preds, labels=labels)

        # Per-class accuracy
        class_acc = np.divide(
            cm.diagonal(),
            cm.sum(axis=1),
            out=np.zeros_like(cm.diagonal(), dtype=float),
            where=cm.sum(axis=1) != 0
        )

        return {
            "accuracy": acc,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            "weighted_f1": w_f1,
            "confusion_matrix": cm.tolist(),
            "per_class_accuracy": class_acc.tolist(),
        }
