import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from train_sentence_encoder.data import SentenceDataset, SentenceTokenizer, load_texts
from train_sentence_encoder.model import MiniLMEncoder

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # noqa: BLE001
    plt = None


@dataclass
class TrainingConfig:
    annotations_path: str = "src/data/annotations.jsonl"  # 标注文件路径
    text_field: str = "fine"  # 训练文本来源：fine（细粒度）或 coarse（粗粒度）
    output_dir: str = "train_sentence_encoder/output"  # 模型与指标输出目录
    vocab_size: int = 20000  # 词表大小
    max_length: int = 96  # 最大序列长度（含 CLS/SEP）
    hidden_size: int = 512  # 编码器隐藏维度/输出维度
    num_layers: int = 6  # Transformer 编码层数
    num_heads: int = 8  # 多头注意力头数
    dim_feedforward: int = 1024  # 前馈层维度
    dropout: float = 0.1  # Dropout 比例
    batch_size: int = 128  # 训练批大小
    lr: float = 3e-4  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    epochs: int = 70  # 训练轮数
    mlm_prob: float = 0.15  # MLM 掩码概率
    val_ratio: float = 0.1  # 验证集占比
    test_ratio: float = 0.1  # 测试集占比
    num_workers: int = 8  # DataLoader 工作线程数
    seed: int = 42  # 随机种子
    pool_method: str = "mean"  # 句向量池化方式：mean 或 cls


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a MiniLM-style sentence encoder from scratch.")
    parser.add_argument("--annotations_path", type=str, default=TrainingConfig.annotations_path)
    parser.add_argument("--text_field", type=str, choices=["fine", "coarse"], default=TrainingConfig.text_field)
    parser.add_argument("--output_dir", type=str, default=TrainingConfig.output_dir)
    parser.add_argument("--vocab_size", type=int, default=TrainingConfig.vocab_size)
    parser.add_argument("--max_length", type=int, default=TrainingConfig.max_length)
    parser.add_argument("--hidden_size", type=int, default=TrainingConfig.hidden_size)
    parser.add_argument("--num_layers", type=int, default=TrainingConfig.num_layers)
    parser.add_argument("--num_heads", type=int, default=TrainingConfig.num_heads)
    parser.add_argument("--dim_feedforward", type=int, default=TrainingConfig.dim_feedforward)
    parser.add_argument("--dropout", type=float, default=TrainingConfig.dropout)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--mlm_prob", type=float, default=TrainingConfig.mlm_prob)
    parser.add_argument("--val_ratio", type=float, default=TrainingConfig.val_ratio)
    parser.add_argument("--test_ratio", type=float, default=TrainingConfig.test_ratio)
    parser.add_argument("--num_workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--pool_method", type=str, choices=["mean", "cls"], default=TrainingConfig.pool_method)
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_texts(texts: List[str], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    rng.shuffle(texts)

    total = len(texts)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_size = total - val_size - test_size

    # Ensure we always have at least 1 sample in each split when possible
    if total >= 3 and train_size == 0:
        train_size = 1
        val_size = max(1, val_size)
        test_size = total - train_size - val_size

    train_texts = texts[:train_size]
    val_texts = texts[train_size : train_size + val_size]
    test_texts = texts[train_size + val_size :]
    return train_texts, val_texts, test_texts


def create_dataloaders(
    cfg: TrainingConfig, tokenizer: SentenceTokenizer, texts: List[str]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_texts, val_texts, test_texts = split_texts(texts, cfg.val_ratio, cfg.test_ratio, cfg.seed)

    train_dataset = SentenceDataset(train_texts, tokenizer, mlm_prob=cfg.mlm_prob)
    val_dataset = SentenceDataset(val_texts, tokenizer, mlm_prob=cfg.mlm_prob)
    test_dataset = SentenceDataset(test_texts, tokenizer, mlm_prob=cfg.mlm_prob)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader, test_loader


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)


def mlm_loss_and_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, int, int]:
    """
    计算 MLM 损失和基于被 mask token 的准确率统计。
    返回：loss, 正确预测的 token 数, 有效 token 总数
    """
    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
    )

    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # (B, L)
        mask = labels != -100
        total = mask.sum().item()
        correct = ((preds == labels) & mask).sum().item()

    return loss, correct, total


def run_epoch(
    model: MiniLMEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss, correct, total = mlm_loss_and_accuracy(outputs["mlm_logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_correct += correct
        total_tokens += total

    avg_loss = total_loss / max(1, len(dataloader))
    avg_acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model: MiniLMEncoder, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss, correct, total = mlm_loss_and_accuracy(outputs["mlm_logits"], labels)
        total_loss += loss.item()
        total_correct += correct
        total_tokens += total

    avg_loss = total_loss / max(1, len(dataloader))
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    accuracy = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    return {"loss": avg_loss, "perplexity": perplexity, "accuracy": accuracy}


def save_checkpoint(
    output_dir: Path,
    model: MiniLMEncoder,
    tokenizer: SentenceTokenizer,
    cfg: TrainingConfig,
    filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
    }
    torch.save(checkpoint, output_dir / filename)
    tokenizer.save(output_dir / "tokenizer.json")


def save_curves(history: List[Dict[str, float]], output_dir: Path) -> None:
    """
    根据每个 epoch 的训练/验证 loss 与 accuracy 画曲线。
    """
    if plt is None or not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLM Loss")
    plt.legend()

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLM Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "loss_accuracy_curves.png")
    plt.close()


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


def run_training(cfg: TrainingConfig) -> Dict[str, Any]:
    """
    按给定配置完成一次训练 + 验证 + 测试，保存模型和曲线，返回主要指标。
    """
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    texts = load_texts(cfg.annotations_path, cfg.text_field)
    if not texts:
        raise RuntimeError(f"No texts found in {cfg.annotations_path} for field {cfg.text_field}")

    print(f"Loaded {len(texts)} {cfg.text_field} texts from annotations.")

    tokenizer = SentenceTokenizer(vocab_size=cfg.vocab_size, max_length=cfg.max_length)
    tokenizer.build(texts)
    print(f"Tokenizer built with vocab size {len(tokenizer)}")

    train_loader, val_loader, test_loader = create_dataloaders(cfg, tokenizer, texts)
    model = MiniLMEncoder(
        vocab_size=len(tokenizer),
        max_length=cfg.max_length,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        pool_method=cfg.pool_method,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val PPL: {val_metrics['perplexity']:.2f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_perplexity": float(val_metrics["perplexity"]),
            }
        )

        torch.save(model.state_dict(), output_dir / "last_model.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(output_dir, model, tokenizer, cfg, "best_model.pt")
            print("Saved new best model.")

    # Final evaluation
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Test PPL: {test_metrics['perplexity']:.2f}"
    )

    # 保存曲线图
    save_curves(history, output_dir)

    summary: Dict[str, Any] = {
        "val_loss": float(best_val_loss),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_perplexity": float(test_metrics["perplexity"]),
        "history": history,
        "config": asdict(cfg),
    }

    metrics_path = Path(cfg.output_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    main()
