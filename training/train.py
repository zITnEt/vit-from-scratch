import time

import torch
import torch.nn as nn

from config import config, CHECKPOINT_DIR
from training.dataset import build_loaders
from model.transformer import Transformer


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """One pass over `loader`. Returns (avg_loss, accuracy)."""
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)               # [batch, num_classes]
        loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    device = config.device
    print(f"device: {device}")

    train_loader, test_loader = build_loaders()

    model = Transformer(
        classes=config.num_classes,
        d_model=config.d_model,
        heads=config.num_heads,
        d_ff=config.d_ff,
        num_encoder_layers=config.num_encoder_layers,
        patches=config.num_patches,
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        dropout=config.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.1f}M parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    # Linear warmup, then cosine decay over the remaining epochs. Warmup gives
    # the attention layers a few gentle epochs before the full learning rate,
    # which stabilizes ViT training on small datasets.
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=config.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs - config.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs])

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        mins = (time.time() - t0) / 60

        print(f"epoch {epoch:2d} | train {train_loss:.3f}/{train_acc:.3f} | "
              f"val {val_loss:.3f}/{val_acc:.3f} | {mins:.1f} min")

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}
        torch.save(ckpt, CHECKPOINT_DIR / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, CHECKPOINT_DIR / "best.pt")
            print(f"  -> new best (val acc {val_acc:.3f}), saved {CHECKPOINT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
