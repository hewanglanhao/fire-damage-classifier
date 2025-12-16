import torch
import torch.nn as nn
import timm


class CNNEncoder(nn.Module):
    def __init__(
        self, model_name="resnet18", pretrained=False, output_dim=512, drop_rate=0.0
    ):
        super().__init__()
        # remove classifier
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate
        )
        self.output_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)


class ViTEncoder(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.output_dim = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)
