import torch
import torch.nn as nn
import timm
from .components.heads import HierarchicalHead


class BaselineViT(nn.Module):
    def __init__(
        self, num_classes, model_name="vit_base_patch16_224", pretrained=False, drop_rate=0.0, drop_path_rate=0.0, align_params=True
    ):
        super().__init__()
        # Use num_classes=0 to get features (CLS token usually for ViT)
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        
        self.align_params = align_params
        if self.align_params:
            # Add MLP to match parameter count of the Method model (~110M vs ~86M)
            # We need ~24M extra parameters.
            # 768 -> 4096 (3.1M) -> 4096 (16.7M) -> 768 (3.1M) ~= 23M
            self.param_adapter = nn.Sequential(
                nn.Linear(self.model.num_features, 4096),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(4096, self.model.num_features),
                nn.LayerNorm(self.model.num_features)
            )

        self.head = HierarchicalHead(self.model.num_features, num_classes, dropout=drop_rate)

    def forward(self, x):
        features = self.model(x)
        
        if self.align_params:
            features = features + self.param_adapter(features)

        logits_binary, logits_severity = self.head(features)

        return {
            "logits": logits_severity,
            "logits_binary": logits_binary,
            "logits_severity": logits_severity,
        }


class BaselineCNN(nn.Module):
    def __init__(
        self, num_classes, model_name="resnet101", pretrained=False, drop_rate=0.0, align_params=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            drop_rate=drop_rate
        )
        
        self.align_params = align_params
        if self.align_params:
            # Adjust hidden dim to match parameter count of BaselineViT (~110M)
            # ResNet101 is ~44M. We need ~66M more.
            # With hidden_dim=6144:
            # 2048*6144 + 6144*6144 + 6144*2048 = 12.5M + 37.7M + 12.5M = 62.7M
            # Total = 44 + 62.7 = 106.7M (Close to 110M)
            hidden_dim = 6144
            self.param_adapter = nn.Sequential(
                nn.Linear(self.model.num_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(hidden_dim, self.model.num_features),
                nn.LayerNorm(self.model.num_features)
            )

        self.head = HierarchicalHead(self.model.num_features, num_classes, dropout=drop_rate)

    def forward(self, x):
        features = self.model(x)
        
        if self.align_params:
            features = features + self.param_adapter(features)

        logits_binary, logits_severity = self.head(features)

        return {
            "logits": logits_severity,
            "logits_binary": logits_binary,
            "logits_severity": logits_severity,
        }


if __name__ == "__main__":
    # Test instantiation
    try:
        model = BaselineViT(num_classes=5)
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        print(f"Baseline Output shape: {y.shape}")
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure 'timm' is installed: pip install timm")
