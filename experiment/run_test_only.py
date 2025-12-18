import argparse
import ast
import json
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

import torch

# Allow running as a script without installing the package:
# `python3 experiment/run_test_only.py ...`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.baseline import BaselineCNN, BaselineViT
from src.models.method import FireDamageClassifier
from src.training.losses import CombinedLoss
from src.utils.metrics import ClassificationMetrics


_THOP_STAT_NAMES = ("total_ops", "total_params")


def _safe_torch_load(path: str, map_location: Any) -> Any:
    """
    Compatible `torch.load` across PyTorch versions.
    Prefer `weights_only=True` when available to avoid loading arbitrary objects.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _filter_thop_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in state_dict.items() if k.split(".")[-1] not in _THOP_STAT_NAMES}


def _count_total_parameters(model: torch.nn.Module) -> int:
    """
    Count *structural* parameters.

    Note: sentence encoders may be stored outside the registered module tree (to keep them on CPU
    and avoid `model.to("cuda")` moving them). We include them if present.
    """

    total = sum(p.numel() for p in model.parameters())

    extra_modules = []
    if hasattr(model, "sentence_encoder") and isinstance(getattr(model, "sentence_encoder"), torch.nn.Module):
        if "sentence_encoder" not in model._modules:
            extra_modules.append(getattr(model, "sentence_encoder"))
    if hasattr(model, "sentence_fallback") and isinstance(getattr(model, "sentence_fallback"), torch.nn.Module):
        if "sentence_fallback" not in model._modules:
            extra_modules.append(getattr(model, "sentence_fallback"))

    for m in extra_modules:
        total += sum(p.numel() for p in m.parameters())

    return int(total)


def _read_flops_from_model_info(exp_dir: str) -> Optional[str]:
    """
    Best-effort read of FLOPs string from `results/<exp>/model_info.txt`.
    Returns `None` if missing/unparseable.
    """

    path = os.path.join(exp_dir, "model_info.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("FLOPs:"):
                return line[len("FLOPs:") :].strip()
    return None


def _load_config_from_model_info(exp_dir: str) -> Dict[str, Any]:
    """
    Reads `results/<exp>/model_info.txt` and parses the `Config: {...}` line.
    This avoids needing the (temporary) YAML used during the experiment run.
    """

    path = os.path.join(exp_dir, "model_info.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find `{path}`. Provide `--config` or ensure the experiment has `model_info.txt`."
        )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Config:"):
                config_str = line[len("Config:") :].strip()
                try:
                    cfg = ast.literal_eval(config_str)
                except Exception as e:  # noqa: BLE001
                    raise ValueError(
                        f"Failed to parse Config from `{path}`. "
                        f"Line was: {line.strip()}\nOriginal error: {e}"
                    ) from e
                if not isinstance(cfg, dict):
                    raise ValueError(f"Parsed config is not a dict in `{path}`.")
                return cfg

    raise ValueError(f"`Config:` line not found in `{path}`.")


def _load_config_from_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError("PyYAML is required to load YAML configs. Install `pyyaml`.") from e

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config at `{path}`")
    return cfg


def _resolve_checkpoint(exp_dir: str, ckpt: str) -> str:
    if ckpt in {"best", "best_model"}:
        path = os.path.join(exp_dir, "best_model.pth")
    elif ckpt in {"last", "last_model"}:
        path = os.path.join(exp_dir, "last_model.pth")
    else:
        path = ckpt
        if not os.path.isabs(path):
            path = os.path.join(exp_dir, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: `{path}`")
    return path


def _build_model_and_criterion(config: Dict[str, Any]) -> Tuple[torch.nn.Module, CombinedLoss, str, int]:
    num_classes = int(config["model"]["num_classes"])
    model_type = config["model"].get("type", "method")

    if model_type == "baseline":
        model = BaselineViT(
            num_classes=num_classes,
            model_name=config["model"]["backbone"],
            drop_rate=config["training"].get("dropout", 0.0),
            drop_path_rate=config["training"].get("drop_path_rate", 0.0),
        )
    elif model_type == "baseline_cnn":
        model = BaselineCNN(
            num_classes=num_classes,
            model_name=config["model"]["backbone"],
            drop_rate=config["training"].get("dropout", 0.0),
        )
    else:
        model = FireDamageClassifier(config)

    criterion = CombinedLoss(config)
    return model, criterion, model_type, num_classes


def _evaluate(model, loader, criterion, device: str, model_type: str, num_classes: int):
    model.eval()
    metrics = ClassificationMetrics(num_classes=num_classes)
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                imgs, targets, coarse_txt, fine_txt = batch
                coarse_text_raw, fine_text_raw = None, None
            else:
                imgs, targets, coarse_txt, fine_txt, coarse_text_raw, fine_text_raw = batch

            imgs, targets = imgs.to(device), targets.to(device)
            coarse_txt, fine_txt = coarse_txt.to(device), fine_txt.to(device)

            if model_type.startswith("baseline"):
                outputs = model(imgs)
                loss, _ = criterion(outputs, targets, imgs, coarse_txt, fine_txt)
            else:
                outputs = model(
                    imgs,
                    coarse_txt,
                    fine_txt,
                    coarse_text_raw=coarse_text_raw,
                    fine_text_raw=fine_text_raw,
                )
                loss, _ = criterion(outputs, targets, imgs, coarse_txt, fine_txt)

            total_loss += float(loss.item())
            metrics.update(outputs["logits"], targets)

    return total_loss / max(1, len(loader)), metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test-only evaluation for an existing experiment.")
    parser.add_argument("--exp_name", required=True, help="Experiment name under `results/` (e.g. sentence_align_fine_pretrained)")
    parser.add_argument("--config", default=None, help="Optional config YAML path. If omitted, parse from results/<exp>/model_info.txt.")
    parser.add_argument("--checkpoint", default="best", help="best | last | relative/absolute path")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Override device (default: auto)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size for test loader only.")
    parser.add_argument(
        "--sentence_device",
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Override `model.sentence_encoder_device` (helps control memory/speed).",
    )
    args = parser.parse_args()

    exp_dir = os.path.join("results", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    if args.config:
        config = _load_config_from_yaml(args.config)
    else:
        config = _load_config_from_model_info(exp_dir)

    if args.batch_size is not None:
        config.setdefault("data", {})
        config["data"]["batch_size"] = int(args.batch_size)

    if args.sentence_device is not None:
        config.setdefault("model", {})
        config["model"]["sentence_encoder_device"] = args.sentence_device

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ckpt_path = _resolve_checkpoint(exp_dir, args.checkpoint)
    print(f"Experiment: {args.exp_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    model, criterion, model_type, num_classes = _build_model_and_criterion(config)

    # Load checkpoint safely:
    # - filter out THOP stats keys like *.total_ops / *.total_params
    # - use strict=False so extra keys won't crash test-only runs
    state = _safe_torch_load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint `{ckpt_path}` is not a state_dict dict.")
    state = _filter_thop_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys ({len(missing)}). Example: {missing[:3]}")
    if unexpected:
        print(f"Warning: unexpected keys ({len(unexpected)}). Example: {unexpected[:3]}")

    total_params = _count_total_parameters(model)
    flops = _read_flops_from_model_info(exp_dir)

    model.to(device)

    from src.data.dataset import create_dataloaders

    _train_loader, _val_loader, test_loader, _tokenizer = create_dataloaders(config)
    test_loss, test_metrics = _evaluate(
        model, test_loader, criterion, device, model_type=model_type, num_classes=num_classes
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_metrics['accuracy']:.4f}")
    print(f"Total Params: {total_params:,}")
    if flops is not None:
        print(f"FLOPs (from model_info.txt): {flops}")

    # Write standard artifact names to match training script output.
    results_path = os.path.join(exp_dir, "results.json")
    metrics_path = os.path.join(exp_dir, "test_metrics.json")

    # If old files exist, back them up (so we don't destroy previous runs).
    if os.path.exists(results_path):
        os.replace(results_path, results_path + ".bak")
    if os.path.exists(metrics_path):
        os.replace(metrics_path, metrics_path + ".bak")

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "parameters": int(total_params),
        "flops": flops,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Saved: {results_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
