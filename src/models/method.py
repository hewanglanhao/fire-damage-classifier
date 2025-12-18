import os
import json
import torch
import torch.nn as nn
import timm
from .components.encoders import CNNEncoder, ViTEncoder
from .components.decoders import TextDecoder, ImageDecoder
from .components.heads import HierarchicalHead
from train_sentence_encoder.model import MiniLMEncoder


class TextVAE(nn.Module):
    def __init__(
        self,
        encoder_type="cnn",
        latent_dim=512,
        vocab_size=1000,
        vit_model_name="vit_base_patch16_224",
        cnn_model_name="resnet18",
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        if encoder_type == "cnn":
            self.encoder = CNNEncoder(model_name=cnn_model_name, drop_rate=dropout)
        else:
            self.encoder = ViTEncoder(
                model_name=vit_model_name,
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

    def __init__(
        self,
        latent_dim=512,
        model_name="vit_base_patch16_224",
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            model_name=model_name,
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

    def __init__(
        self,
        latent_dim=512,
        model_name="vit_base_patch16_224",
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            model_name=model_name,
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
        self.use_sentence_encoder = config["model"].get("use_sentence_encoder", False)
        self.sentence_encoder_name = config["model"].get(
            "sentence_encoder_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.align_text_source = config["model"].get("align_text_source", "fine")
        self.sentence_pretrained = config["model"].get("sentence_pretrained", True)
        self.freeze_sentence_encoder = config["model"].get(
            "freeze_sentence_encoder", self.sentence_pretrained
        )
        # Where to run sentence encoder (helps avoid CUDA OOM). Typical: "cpu" when frozen.
        self.sentence_encoder_device = config["model"].get(
            "sentence_encoder_device", "cpu" if self.freeze_sentence_encoder else None
        )
        self.sentence_encoder_path = config["model"].get("sentence_encoder_name", None)

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
        self.backbone = config["model"].get("backbone", "vit_base_patch16_224")

        # Optional sentence-level encoder (e.g., SentenceTransformer)
        # NOTE: When `sentence_encoder_device` is "cpu", we intentionally keep sentence encoders
        # out of the registered module tree (store in __dict__) so `model.to("cuda")` will not
        # try to move them onto GPU and trigger CUDA OOM.
        self.sentence_encoder = None
        self.sentence_fallback = None
        self.sentence_tokenizer = None
        self.sentence_proj = None
        if self.use_sentence_encoder:
            if self.sentence_pretrained:
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore
                except Exception as e:  # noqa: BLE001
                    raise ImportError(
                        "sentence-transformers is required for use_sentence_encoder=True "
                        "but could not be imported. Please install/upgrade it (and its "
                        "dependencies such as pyarrow/datasets) or set use_sentence_encoder=False.\n"
                        f"Original error: {e}"
                    ) from e

                st_device = self.sentence_encoder_device
                # SentenceTransformer by default moves itself to CUDA when available; force device if provided.
                sentence_encoder = (
                    SentenceTransformer(self.sentence_encoder_name, device=st_device)
                    if st_device is not None
                    else SentenceTransformer(self.sentence_encoder_name)
                )
                try:
                    sent_dim = int(sentence_encoder.get_sentence_embedding_dimension())
                except Exception:
                    sent_dim = self.latent_dim
                self.sentence_proj = nn.Identity() if sent_dim == self.latent_dim else nn.Linear(sent_dim, self.latent_dim)

                if self.freeze_sentence_encoder:
                    for p in sentence_encoder.parameters():
                        p.requires_grad = False
                    sentence_encoder.eval()

                if st_device == "cpu":
                    self.__dict__["sentence_encoder"] = sentence_encoder
                else:
                    self.sentence_encoder = sentence_encoder
            else:
                # Load custom encoder trained by train_sentence_encoder (MiniLMEncoder)
                if self.sentence_encoder_path is None:
                    raise ValueError("sentence_encoder_name/path must be provided when sentence_pretrained=False")
                ckpt_path = os.path.join(self.sentence_encoder_path, "best_model.pt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(self.sentence_encoder_path, "last_model.pt")
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Cannot find sentence encoder checkpoint at {self.sentence_encoder_path}")

                checkpoint = torch.load(ckpt_path, map_location="cpu")
                cfg = checkpoint.get("config", {})
                vocab_path = os.path.join(self.sentence_encoder_path, "tokenizer.json")
                if not os.path.exists(vocab_path):
                    raise FileNotFoundError(f"tokenizer.json not found in {self.sentence_encoder_path}")
                with open(vocab_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)

                vocab_size = len(tok_data["token_to_id"])
                max_len = tok_data["max_length"]
                hidden_size = cfg.get("hidden_size", self.latent_dim)
                self.sentence_hidden_size = hidden_size
                self.sentence_tokenizer = tok_data
                sentence_fallback = MiniLMEncoder(
                    vocab_size=vocab_size,
                    max_length=max_len,
                    hidden_size=hidden_size,
                    num_layers=cfg.get("num_layers", 6),
                    num_heads=cfg.get("num_heads", 8),
                    dim_feedforward=cfg.get("dim_feedforward", 1024),
                    dropout=cfg.get("dropout", 0.1),
                    pool_method=cfg.get("pool_method", "mean"),
                )
                sentence_fallback.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
                if hidden_size != self.latent_dim:
                    self.sentence_proj = nn.Linear(hidden_size, self.latent_dim)
                else:
                    self.sentence_proj = nn.Identity()

                if self.freeze_sentence_encoder:
                    for p in sentence_fallback.parameters():
                        p.requires_grad = False
                    sentence_fallback.eval()

                # Put the (possibly frozen) sentence encoder on the requested device to avoid CUDA OOM.
                if self.sentence_encoder_device is not None:
                    sentence_fallback.to(self.sentence_encoder_device)

                if self.sentence_encoder_device == "cpu":
                    self.__dict__["sentence_fallback"] = sentence_fallback
                else:
                    self.sentence_fallback = sentence_fallback

        # Model 1 - Coarse
        if self.use_coarse:
            self.text_vae_coarse = TextVAE(
                encoder_type=self.coarse_encoder_type,
                latent_dim=self.latent_dim,
                vocab_size=self.vocab_size,
                vit_model_name=self.backbone,
                dropout=self.dropout_rate,
                drop_path_rate=self.drop_path_rate,
            )

        # Model 1 - Fine
        if self.use_fine:
            self.text_vae_fine = TextVAE(
                encoder_type=self.fine_encoder_type,
                latent_dim=self.latent_dim,
                vocab_size=self.vocab_size,
                vit_model_name=self.backbone,
                dropout=self.dropout_rate,
                drop_path_rate=self.drop_path_rate,
            )

        # Model 2 - Image
        if self.method_option == "vae":
            self.image_model = ImageVAE(
                latent_dim=self.latent_dim,
                model_name=self.backbone,
                dropout=self.dropout_rate,
                drop_path_rate=self.drop_path_rate,
            )
        elif self.method_option == "alignment":
            self.image_model = ImageAlignment(
                latent_dim=self.latent_dim,
                model_name=self.backbone,
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

    def _encode_sentence_fallback(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute sentence embedding for MiniLMEncoder without building large MLM logits.
        This mirrors `train_sentence_encoder/model.py` but skips `mlm_head`.
        """
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.sentence_fallback.token_embeddings(input_ids) + self.sentence_fallback.position_embeddings(positions)
        x = self.sentence_fallback.embed_dropout(self.sentence_fallback.embed_layer_norm(x))

        padding_mask = attention_mask == 0
        sequence_output = self.sentence_fallback.encoder(x, src_key_padding_mask=padding_mask)

        if self.sentence_fallback.pool_method == "cls":
            sentence_embedding = sequence_output[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1)
            summed = (sequence_output * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            sentence_embedding = summed / denom

        return sentence_embedding

    def forward(
        self,
        img,
        coarse_text=None,
        fine_text=None,
        coarse_text_raw=None,
        fine_text_raw=None,
    ):
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

        # Optional sentence-transformer text embedding (coarse or fine raw text)
        z_sentence = None
        if self.use_sentence_encoder:
            text_list = None
            if self.align_text_source == "fine" and fine_text_raw is not None:
                text_list = fine_text_raw
            if text_list is None and coarse_text_raw is not None:
                text_list = coarse_text_raw

            if self.sentence_encoder is not None and text_list is not None:
                if self.freeze_sentence_encoder:
                    self.sentence_encoder.eval()

                try:
                    if self.freeze_sentence_encoder:
                        with torch.no_grad():
                            z_sentence = self.sentence_encoder.encode(
                                list(text_list),
                                convert_to_tensor=True,
                                device=self.sentence_encoder_device or img.device,
                                show_progress_bar=False,
                            )
                    else:
                        z_sentence = self.sentence_encoder.encode(
                            list(text_list),
                            convert_to_tensor=True,
                            device=self.sentence_encoder_device or img.device,
                            show_progress_bar=False,
                        )
                except Exception as e:
                    print(f"Sentence encoder failed: {e}")
                    z_sentence = None
                if z_sentence is not None and self.sentence_proj is not None:
                    # Some SentenceTransformer backends return "inference tensors" (created under
                    # torch.inference_mode). Autograd can't save them for backward through our
                    # trainable projection, so we materialize a normal tensor here.
                    if self.freeze_sentence_encoder:
                        z_sentence = z_sentence.detach().clone()
                    # Move sentence embeddings to the same device as the rest of the model
                    z_sentence = z_sentence.to(img.device)
                    z_sentence = self.sentence_proj(z_sentence)
            elif self.sentence_fallback is not None and self.sentence_tokenizer is not None and text_list is not None:
                # Tokenize raw text using saved tokenizer.json
                token_to_id = self.sentence_tokenizer["token_to_id"]
                pad_id = token_to_id["[PAD]"]
                cls_id = token_to_id["[CLS]"]
                sep_id = token_to_id["[SEP]"]
                max_len = self.sentence_tokenizer["max_length"]

                def encode_one(txt: str):
                    toks = txt.strip().lower().split() if txt else []
                    ids = [cls_id] + [token_to_id.get(t, token_to_id["[UNK]"]) for t in toks][: max_len - 2] + [sep_id]
                    if len(ids) < max_len:
                        ids += [pad_id] * (max_len - len(ids))
                    else:
                        ids = ids[:max_len]
                    attn = [1 if i != pad_id else 0 for i in ids]
                    return ids, attn

                ids_list, attn_list = zip(*(encode_one(t) for t in text_list))
                enc_device = self.sentence_encoder_device or img.device
                input_ids = torch.tensor(ids_list, dtype=torch.long, device=enc_device)
                attention_mask = torch.tensor(attn_list, dtype=torch.long, device=enc_device)
                if self.freeze_sentence_encoder:
                    with torch.no_grad():
                        z_sentence = self._encode_sentence_fallback(input_ids, attention_mask)
                else:
                    z_sentence = self._encode_sentence_fallback(input_ids, attention_mask)
                z_sentence = z_sentence.to(img.device)
                if self.sentence_proj is not None:
                    z_sentence = self.sentence_proj(z_sentence)

            outputs["z_sentence"] = z_sentence

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
