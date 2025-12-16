import unittest
import sys
from unittest.mock import MagicMock
import torch

# Mock timm
sys.modules["timm"] = MagicMock()

from src.models.method import FireDamageClassifier, TextVAE, ImageVAE, ImageAlignment


class TestModelStructure(unittest.TestCase):
    def test_method_vae_structure(self):
        config = {
            "model": {"latent_dim": 512, "num_classes": 4, "method_option": "vae"},
            "data": {"vocab_size": 1000},
        }

        model = FireDamageClassifier(config)

        self.assertIsInstance(model.text_vae_coarse, TextVAE)
        self.assertIsInstance(model.text_vae_fine, TextVAE)
        self.assertIsInstance(model.image_model, ImageVAE)
        self.assertNotIsInstance(model.image_model, ImageAlignment)

    def test_method_alignment_structure(self):
        config = {
            "model": {
                "latent_dim": 512,
                "num_classes": 4,
                "method_option": "alignment",
            },
            "data": {"vocab_size": 1000},
        }

        model = FireDamageClassifier(config)

        self.assertIsInstance(model.image_model, ImageAlignment)
        self.assertNotIsInstance(model.image_model, ImageVAE)


if __name__ == "__main__":
    unittest.main()
