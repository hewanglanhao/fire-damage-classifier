import torch
import yaml
import os
import argparse
import json
import csv
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from src.models.method import FireDamageClassifier
from src.models.baseline import BaselineViT, BaselineCNN
from src.training.losses import CombinedLoss
from src.utils.metrics import ClassificationMetrics

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from thop import profile
except ImportError:
    profile = None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, device, model_type="method"):
    if profile is None:
        return "thop not installed"

    # Create dummy inputs
    img = torch.randn(1, 3, 224, 224).to(device)

    if model_type.startswith("baseline"):
        flops, params = profile(model, inputs=(img,), verbose=False)
    else:
        # Method needs text inputs
        coarse = torch.randint(0, 1000, (1, 50)).to(device)
        fine = torch.randint(0, 1000, (1, 50)).to(device)
        flops, params = profile(model, inputs=(img, coarse, fine), verbose=False)

    return flops


def save_plots(log_file, output_dir):
    if plt is None:
        return

    try:
        epochs = []
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []

        with open(log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                train_acc.append(float(row["train_acc"]))
                val_acc.append(float(row["val_acc"]))

        plt.figure(figsize=(12, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curve")

        plt.savefig(os.path.join(output_dir, "curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting: {e}")


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate(model, loader, criterion, device, model_type="method", num_classes=None):
    model.eval()
    metrics = ClassificationMetrics(num_classes=num_classes)
    total_loss = 0

    with torch.no_grad():
        for imgs, targets, coarse_txt, fine_txt in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            coarse_txt, fine_txt = coarse_txt.to(device), fine_txt.to(device)

            if model_type.startswith("baseline"):
                outputs = model(imgs)
                # Pass dummy text/images to criterion if needed, or just outputs
                loss, _ = criterion(outputs, targets, imgs, coarse_txt, fine_txt)
            else:
                outputs = model(imgs, coarse_txt, fine_txt)
                loss, _ = criterion(outputs, targets, imgs, coarse_txt, fine_txt)

            total_loss += loss.item()
            metrics.update(outputs["logits"], targets)

    return total_loss / len(loader), metrics.compute()


def train_one_epoch(
    model, loader, optimizer, criterion, scheduler, device, model_type="method", num_classes=None
):
    model.train()
    metrics = ClassificationMetrics(num_classes=num_classes)
    total_loss = 0

    for batch_idx, (imgs, targets, coarse_txt, fine_txt) in enumerate(loader):
        imgs, targets = imgs.to(device), targets.to(device)
        coarse_txt, fine_txt = coarse_txt.to(device), fine_txt.to(device)

        optimizer.zero_grad()

        if model_type.startswith("baseline"):
            outputs = model(imgs)
            loss, loss_dict = criterion(outputs, targets, imgs, coarse_txt, fine_txt)
        else:
            outputs = model(imgs, coarse_txt, fine_txt)
            loss, loss_dict = criterion(outputs, targets, imgs, coarse_txt, fine_txt)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        metrics.update(outputs["logits"], targets)

        if batch_idx % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Batch {batch_idx}: Loss {loss.item():.4f} (Cls: {loss_dict['cls']:.4f}) LR: {lr:.6f}"
            )

    return total_loss / len(loader), metrics.compute()


def main(config_path="configs/config.yaml", exp_name=None):
    # Load Config
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return

    config = load_config(config_path)
    print(f"Loaded config: {config}")

    # Setup Output Directory
    if exp_name is None:
        exp_name = f"exp_{int(time.time())}"
    output_dir = os.path.join("results", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Configuration
    BATCH_SIZE = config["data"]["batch_size"]
    NUM_CLASSES = config["model"]["num_classes"]
    VOCAB_SIZE = config["data"]["vocab_size"]
    SEQ_LEN = config["data"]["seq_len"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_TYPE = config["model"].get("type", "method")

    print(f"Using device: {DEVICE}")
    print(f"Model Type: {MODEL_TYPE}")

    # Data Loading
    from src.data.dataset import create_dataloaders

    try:
        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(config)
        print(
            f"Data loaded: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}"
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Falling back to dummy data for testing...")
        # Dummy Data
        imgs = torch.randn(20, 3, 224, 224)
        targets = torch.randint(0, NUM_CLASSES, (20,))
        coarse_txt = torch.randint(0, VOCAB_SIZE, (20, SEQ_LEN))
        fine_txt = torch.randint(0, VOCAB_SIZE, (20, SEQ_LEN))
        dataset = TensorDataset(imgs, targets, coarse_txt, fine_txt)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False
        )  # Use same for test

    # Model
    if MODEL_TYPE == "baseline":
        model = BaselineViT(
            num_classes=NUM_CLASSES, 
            model_name=config["model"]["backbone"],
            drop_rate=config["training"].get("dropout", 0.0),
            drop_path_rate=config["training"].get("drop_path_rate", 0.0)
        )
        # Use CombinedLoss for baseline too to support hierarchical loss
        criterion = CombinedLoss(config)
    elif MODEL_TYPE == "baseline_cnn":
        model = BaselineCNN(
            num_classes=NUM_CLASSES,
            model_name=config["model"]["backbone"],
            drop_rate=config["training"].get("dropout", 0.0)
        )
        criterion = CombinedLoss(config)
    else:
        model = FireDamageClassifier(config)
        criterion = CombinedLoss(config)

    model.to(DEVICE)

    # Model Info
    params_count = count_parameters(model)
    flops_count = calculate_flops(model, DEVICE, MODEL_TYPE)
    print(f"Parameters: {params_count:,}")
    print(f"FLOPs: {flops_count}")

    with open(os.path.join(output_dir, "model_info.txt"), "w") as f:
        f.write(f"Model Type: {MODEL_TYPE}\n")
        f.write(f"Parameters: {params_count}\n")
        f.write(f"FLOPs: {flops_count}\n")
        f.write(f"Config: {config}\n")

    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    # Train Loop
    print("Starting training loop...")
    epochs = config["training"].get("epochs", 10)

    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(config["training"]["lr"]),
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    # Logging Setup
    log_file = os.path.join(output_dir, "log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )

    best_val_acc = 0.0

    for epoch in range(epochs):
        loss, metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scheduler,
            DEVICE,
            model_type=MODEL_TYPE,
            num_classes=NUM_CLASSES,
        )
        print(f"Epoch {epoch+1}: Loss {loss:.4f}, Acc {metrics['accuracy']:.4f}")

        # Validation
        val_loss = 0.0
        val_acc = 0.0
        if val_loader:
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, DEVICE, model_type=MODEL_TYPE, num_classes=NUM_CLASSES
            )
            val_acc = val_metrics["accuracy"]
            print(f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, loss, metrics["accuracy"], val_loss, val_acc, current_lr]
            )

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"New best model saved with Val Acc: {val_acc:.4f}")

        # Save last
        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pth"))

    # Plotting
    save_plots(log_file, output_dir)

    # Test Evaluation
    print("Running Test Evaluation...")
    if test_loader:
        # Load best model
        best_model_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for testing.")

        test_loss, test_metrics = evaluate(
            model, test_loader, criterion, DEVICE, model_type=MODEL_TYPE, num_classes=NUM_CLASSES
        )
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")

        # Save Results
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "best_val_accuracy": best_val_acc,
            "parameters": params_count,
            "flops": str(flops_count),
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        # Save detailed test metrics
        with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment Name")
    args = parser.parse_args()
    main(args.config, args.exp_name)
