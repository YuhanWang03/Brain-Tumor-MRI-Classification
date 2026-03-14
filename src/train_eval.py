from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .model import freeze_backbone, unfreeze_last_n_blocks


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Get available device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy_from_logits(logits, y):
    """
    Compute batch accuracy from logits.
    """
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a dataloader.
    """
    model.eval()
    losses, accs = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, yb))

    return float(np.mean(losses)), float(np.mean(accs))


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """
    Train model for one epoch.
    """
    model.train()
    losses, accs = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, yb))

    return float(np.mean(losses)), float(np.mean(accs))


def fit_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs_stage1: int = 30,
    epochs_stage2: int = 15,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    weight_decay: float = 1e-4,
    fine_tune: bool = True,
    unfreeze_last_n: int = 30,
    models_dir: str = "models",
    class_weights=None,
):
    """
    Two-stage training:
    1. Freeze backbone, train classifier head
    2. Unfreeze last N parameter tensors and fine-tune
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print("Using class weights:", class_weights.tolist())
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Stage 1
    freeze_backbone(model)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1,
        weight_decay=weight_decay,
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    best_path = models_dir / "best_pytorch.pt"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs_stage1 + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler if use_amp else None
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(
            f"[Stage-1][{epoch:02d}/{epochs_stage1}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_path)

    # Stage 2
    if fine_tune:
        unfreeze_last_n_blocks(model, n=unfreeze_last_n)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_stage2,
            weight_decay=weight_decay,
        )

        for epoch in range(1, epochs_stage2 + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                scaler=scaler if use_amp else None
            )
            va_loss, va_acc = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            print(
                f"[Stage-2][{epoch:02d}/{epochs_stage2}] "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
            )

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                torch.save(model.state_dict(), best_path)

    print("Best model saved to:", best_path)

    return history, best_path


def plot_history(
    history,
    method_name: str = "default",
    seed: int | None = None,
    save_plots: bool = True,
    show_plots: bool = True,
):
    """
    Plot and optionally save training curves.
    """
    if seed is None:
        output_dir = Path("outputs") / method_name
    else:
        output_dir = Path("outputs") / method_name / f"seed_{seed}"

    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title(f"Accuracy Curve - {method_name}" + (f" - seed {seed}" if seed is not None else ""))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    if save_plots:
        acc_path = output_dir / "accuracy_curve.png"
        plt.savefig(acc_path, dpi=300, bbox_inches="tight")
        print("Accuracy curve saved to:", acc_path)

    if show_plots:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title(f"Loss Curve - {method_name}" + (f" - seed {seed}" if seed is not None else ""))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_plots:
        loss_path = output_dir / "loss_curve.png"
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")
        print("Loss curve saved to:", loss_path)

    if show_plots:
        plt.show()
    else:
        plt.close()


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def test_and_report(
    model,
    test_loader,
    class_names,
    device,
    show_confusion_matrix: bool = True,
    method_name: str = "default",
    seed: int | None = None,
    save_confusion_matrix: bool = True,
):
    """
    Evaluate on test set and print classification report.
    """
    model.eval()

    y_true, y_pred = [], []

    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()

        y_pred.extend(preds)
        y_true.extend(yb.numpy().tolist())

    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    print("Test accuracy:", test_acc)
    print("Classification report:")
    print(report)

    if show_confusion_matrix:
        plt.figure(figsize=(7, 5))
        title = f"Confusion Matrix (Test) - {method_name}"
        if seed is not None:
            title += f" - seed {seed}"
        plt.title(title)

        plt.imshow(cm, cmap="Blues")
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
        plt.colorbar()

        threshold = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        if save_confusion_matrix:
            if seed is None:
                output_dir = Path("outputs") / method_name
            else:
                output_dir = Path("outputs") / method_name / f"seed_{seed}"

            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "confusion_matrix_test.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print("Confusion matrix saved to:", save_path)

        plt.show()

    return {
        "accuracy": test_acc,
        "report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_final_model(model, save_path: str = "models/final_pytorch.pt"):
    """
    Save final model state dict.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved:", save_path)
