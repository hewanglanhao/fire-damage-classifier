import torch
import torch.nn as nn
import timm
from .components.encoders import CNNEncoder, ViTEncoder
from .components.decoders import TextDecoder, ImageDecoder
from .components.heads import HierarchicalHead


class TextVAE(nn.Module):
    def __init__(
        self,
        encoder_type="cnn",
        latent_dim=512,
        vocab_size=1000,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        if encoder_type == "cnn":
            self.encoder = CNNEncoder(model_name="resnet18", drop_rate=dropout)
        else:
            self.encoder = ViTEncoder(
                model_name="vit_base_patch16_224",
                drop_rate=dropout,
                drop_path_rate=drop_path_rate,
            )

        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(self.encoder.output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.output_dim, latent_dim)

        # Decoder for reconstruction (training only)
        self.decoder = TextDecoder(
            vocab_size=vocab_size, embed_dim=256, latent_dim=latent_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, target_text=None):
        features = self.encoder(x)
        features = self.dropout(features)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        z = self.reparameterize(mu, logvar)

        recon_text = None
        if target_text is not None:
            recon_text = self.decoder(z, target_text)

        return z, mu, logvar, recon_text


class ImageVAE(nn.Module):
    """Method A: VAE for Image"""

    def __init__(self, latent_dim=512, dropout=0.0, drop_path_rate=0.0):
        super().__init__()
        self.encoder = ViTEncoder(
            model_name="vit_base_patch16_224",
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(self.encoder.output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.output_dim, latent_dim)
        self.decoder = ImageDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder(x)
        features = self.dropout(features)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        recon_img = self.decoder(z)
        return z, mu, logvar, recon_img


class ImageAlignment(nn.Module):
    """Method B: CLIP-like Alignment"""

    def __init__(self, latent_dim=512, dropout=0.0, drop_path_rate=0.0):
        super().__init__()
        self.encoder = ViTEncoder(
            model_name="vit_base_patch16_224",
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.encoder.output_dim, latent_dim)

    def forward(self, x):
        features = self.encoder(x)
        features = self.dropout(features)
        z = self.proj(features)
        return z, None, None, None


class GatedFusion(nn.Module):
    """Dynamic Gating Fusion for Image and Text Latents"""

    def __init__(self, latent_dim=512):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_img, z_text):
        # z_img: (B, D), z_text: (B, D)
        combined = torch.cat([z_img, z_text], dim=1)
        alpha = self.gate_net(combined)  # (B, 1)

        # Weighted sum
        fused = alpha * z_img + (1 - alpha) * z_text
        return fused, alpha


class FireDamageClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["model"]["latent_dim"]
        self.num_classes = config["model"]["num_classes"]
        self.vocab_size = config["data"]["vocab_size"]
        self.method_option = config["model"]["method_option"]

        # Configurable parts
        self.use_coarse = config["model"].get("use_coarse", True)
        self.use_fine = config["model"].get("use_fine", True)
        self.use_image = config["model"].get("use_image", True)

        # New Features
        self.use_gated_fusion = config["model"].get("use_gated_fusion", False)
        # Hierarchical Head is now standard

        # Fusion control
        self.fusion_include_coarse = config["model"].get("fusion_include_coarse", True)
        self.fusion_include_fine = config["model"].get("fusion_include_fine", True)
        self.fusion_include_image = config["model"].get("fusion_include_image", True)

        self.coarse_encoder_type = config["model"].get("coarse_encoder_type", "cnn")
        self.fine_encoder_type = config["model"].get("fine_encoder_type", "vit")

        # Regularization params
        self.dropout_rate = config["training"].get("dropout", 0.0)
        self.drop_path_rate = config["training"].get("drop_path_rate", 0.0)

        # Model 1 - Coarse
        if self.use_coarse:
            self.text_vae_coarse = TextVAE(
                encoder_type=self.coarse_encoder_type,
                latent_dim=self.latent_dim,
                vocab_size=self.vocab_size,
                dropout=self.dropout_rate,
                drop_path_rate=self.drop_path_rate,
            )

        # Model 1 - Fine
        if self.use_fine:
            self.text_vae_fine = TextVAE(
                encoder_type=self.fine_encoder_type,
                latent_dim=self.latent_dim,
                vocab_size=self.vocab_size,
                dropout=self.dropout_rate,
                drop_path_rate=self.drop_path_rate,
            )

        # Model 2 - Image
        if self.method_option == "vae":
            self.image_model = ImageVAE(
                latent_dim=self.latent_dim,
                dropout=self.dropout_rate,
                drop_path_rate=self.drop_path_rate,
            )
        elif self.method_option == "alignment":
            self.image_model = ImageAlignment(
                latent_dim=self.latent_dim,
                dropout=self.dropout_rate,
                drop_path_rate=self.drop_path_rate,
            )
        else:
            raise ValueError(f"Unknown method option: {self.method_option}")

        # Gated Fusion
        if self.use_gated_fusion:
            self.gated_fusion = GatedFusion(latent_dim=self.latent_dim)

        # Classification Head (ViT Decoder style)
        self.head_dim = self.latent_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.head_dim, nhead=8, batch_first=True, dropout=self.dropout_rate
        )
        self.classifier_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.head_dim))

        # Always use Hierarchical Head
        self.head = HierarchicalHead(
            self.head_dim, self.num_classes, dropout=self.dropout_rate
        )

    def forward(self, img, coarse_text=None, fine_text=None):
        outputs = {}
        latents = []

        # CLS Token
        B = img.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        latents.append(cls_tokens)

        # Text VAE Coarse
        z_coarse_val = None
        if self.use_coarse:
            z_coarse, mu_c, logvar_c, recon_c = self.text_vae_coarse(img, coarse_text)
            outputs.update(
                {
                    "z_coarse": z_coarse,
                    "mu_c": mu_c,
                    "logvar_c": logvar_c,
                    "recon_c": recon_c,
                }
            )
            z_coarse_val = z_coarse
            if self.fusion_include_coarse and not self.use_gated_fusion:
                latents.append(z_coarse.unsqueeze(1))
        else:
            outputs.update(
                {"z_coarse": None, "mu_c": None, "logvar_c": None, "recon_c": None}
            )

        # Text VAE Fine
        z_fine_val = None
        if self.use_fine:
            z_fine, mu_f, logvar_f, recon_f = self.text_vae_fine(img, fine_text)
            outputs.update(
                {
                    "z_fine": z_fine,
                    "mu_f": mu_f,
                    "logvar_f": logvar_f,
                    "recon_f": recon_f,
                }
            )
            z_fine_val = z_fine
            if self.fusion_include_fine and not self.use_gated_fusion:
                latents.append(z_fine.unsqueeze(1))
        else:
            outputs.update(
                {"z_fine": None, "mu_f": None, "logvar_f": None, "recon_f": None}
            )

        # Image Model
        z_img, mu_i, logvar_i, recon_i = self.image_model(img)
        outputs.update(
            {"z_img": z_img, "mu_i": mu_i, "logvar_i": logvar_i, "recon_i": recon_i}
        )

        if self.use_image and self.fusion_include_image and not self.use_gated_fusion:
            latents.append(z_img.unsqueeze(1))

        # Gated Fusion Logic
        if self.use_gated_fusion:
            # Aggregate text latents (average if both exist)
            z_text_agg = None
            if z_coarse_val is not None and z_fine_val is not None:
                z_text_agg = (z_coarse_val + z_fine_val) / 2
            elif z_coarse_val is not None:
                z_text_agg = z_coarse_val
            elif z_fine_val is not None:
                z_text_agg = z_fine_val
            else:
                # No text available, fallback to just image or zeros?
                # If no text, gate should probably be 1.0 for image
                z_text_agg = torch.zeros_like(z_img)

            fused_latent, alpha = self.gated_fusion(z_img, z_text_agg)
            outputs["gate_alpha"] = alpha
            latents.append(fused_latent.unsqueeze(1))

        # Fusion
        seq = torch.cat(latents, dim=1)

        # Transformer
        out_seq = self.classifier_transformer(seq)

        # Take CLS token output
        cls_out = out_seq[:, 0, :]

        logits_binary, logits_severity = self.head(cls_out)
        outputs["logits"] = logits_severity  # Main logits for metrics
        outputs["logits_binary"] = logits_binary
        outputs["logits_severity"] = logits_severity

        return outputs
