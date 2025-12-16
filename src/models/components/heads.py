import torch
import torch.nn as nn


class HierarchicalHead(nn.Module):
    """Coarse-to-Fine Classification Head"""

    def __init__(self, input_dim, num_classes, dropout=0.0):
        super().__init__()
        # Level 1: Binary (Damage vs No Damage) - Assuming Class 0 is No Damage
        self.binary_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, 2))

        # Level 2: Severity (All classes)
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        logits_binary = self.binary_head(x)
        logits_severity = self.severity_head(x)
        return logits_binary, logits_severity
