import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.method_option = config["model"]["method_option"]
        self.lambda_cls = config["training"]["lambda_cls"]
        self.lambda_vae = config["training"]["lambda_vae"]
        self.lambda_align = config["training"]["lambda_align"]

        self.cls_loss = nn.CrossEntropyLoss()
        self.text_recon_loss = nn.CrossEntropyLoss(
            ignore_index=0
        )  # Assuming 0 is padding
        self.img_recon_loss = nn.MSELoss()

    def vae_loss(self, recon, target, mu, logvar, mode="text"):
        B = mu.size(0)
        if mode == "text":
            # recon: (B, Seq, Vocab), target: (B, Seq)
            _, S, V = recon.shape
            recon = recon.view(-1, V)
            target = target.view(-1)
            r_loss = self.text_recon_loss(recon, target)
        else:
            # Image
            r_loss = self.img_recon_loss(recon, target)

        # KL Divergence
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss /= B  # Normalize by batch size

        return r_loss + 0.001 * kld_loss  # Weight KLD small usually

    def forward(self, outputs, targets, images, coarse_text, fine_text):
        # Classification Loss
        loss_cls = self.cls_loss(outputs["logits"], targets)

        # Model 1 Coarse VAE Loss
        loss_m1_c = 0
        if outputs.get("recon_c") is not None:
            loss_m1_c = self.vae_loss(
                outputs["recon_c"],
                coarse_text,
                outputs["mu_c"],
                outputs["logvar_c"],
                mode="text",
            )

        # Model 1 Fine VAE Loss
        loss_m1_f = 0
        if outputs.get("recon_f") is not None:
            loss_m1_f = self.vae_loss(
                outputs["recon_f"],
                fine_text,
                outputs["mu_f"],
                outputs["logvar_f"],
                mode="text",
            )

        loss_m2 = 0
        if self.method_option == "vae":
            # Model 2 Image VAE Loss
            if outputs.get("recon_i") is not None:
                loss_m2 = self.vae_loss(
                    outputs["recon_i"],
                    images,
                    outputs["mu_i"],
                    outputs["logvar_i"],
                    mode="image",
                )
        else:
            # Method: Alignment Loss
            # Align z_img with z_fine (Thinking Text Latent)
            # If z_fine is None (e.g. only coarse used), we can't align with it.
            # Fallback to coarse or skip?
            # Assuming if alignment is chosen, we must have at least one text source.

            z_text = None
            if outputs.get("z_fine") is not None:
                z_text = outputs["z_fine"]
            elif outputs.get("z_coarse") is not None:
                z_text = outputs["z_coarse"]

            if z_text is not None and outputs.get("z_img") is not None:
                # Cosine Embedding Loss or MSE
                # Normalize vectors
                z_img = F.normalize(outputs["z_img"], dim=1)
                z_text = F.normalize(z_text, dim=1)

                loss_m2 = 1 - F.cosine_similarity(z_img, z_text).mean()

        total_loss = (
            self.lambda_cls * loss_cls
            + self.lambda_vae * (loss_m1_c + loss_m1_f)
            + self.lambda_align * loss_m2
        )

        # Hierarchical Loss
        if "logits_binary" in outputs:
            # Binary Target: 0 if class 0, 1 otherwise
            binary_targets = (targets > 0).long()
            loss_binary = F.cross_entropy(outputs["logits_binary"], binary_targets, label_smoothing=0.1)

            # Severity Target: The original targets
            loss_severity = F.cross_entropy(outputs["logits_severity"], targets, label_smoothing=0.1)

            # Combine (can weight them)
            loss_cls = 0.5 * loss_binary + 0.5 * loss_severity

            # Recompute total loss with new cls loss
            total_loss = (
                self.lambda_cls * loss_cls
                + self.lambda_vae * (loss_m1_c + loss_m1_f)
                + self.lambda_align * loss_m2
            )

        return total_loss, {
            "cls": loss_cls.item(),
            "m1_c": loss_m1_c.item() if isinstance(loss_m1_c, torch.Tensor) else loss_m1_c,
            "m1_f": loss_m1_f.item() if isinstance(loss_m1_f, torch.Tensor) else loss_m1_f,
            "m2": loss_m2.item() if isinstance(loss_m2, torch.Tensor) else loss_m2,
        }
