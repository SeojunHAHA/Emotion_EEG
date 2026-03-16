import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import set_seed, load_config, get_logger, save_checkpoint
from data.dataset import load_seed_file_paths, get_cross_subject_splits
from models.reve_classifier import REVEClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for eeg, labels in loader:
        eeg, labels = eeg.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(eeg)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for eeg, labels in loader:
        eeg, labels = eeg.to(device), labels.to(device)
        logits = model(eeg)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def run_stage(stage_name, model, train_loader, val_loader,
              optimizer, scheduler, criterion, cfg, device, logger, save_dir):
    stage_cfg = cfg.train.linear_probe if stage_name == "linear_probe" else cfg.train.fine_tuning
    best_acc = 0.0

    for epoch in range(stage_cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"[{stage_name}] Epoch {epoch+1}/{stage_cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch,
                            {"val_acc": val_acc},
                            os.path.join(save_dir, f"best_{stage_name}.pt"))

    return best_acc


def main():
    cfg = load_config()
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger = get_logger("train", log_dir=cfg.logging.log_dir)
    logger.info(f"Device: {device}")

    logger.info("Loading SEED file paths...")
    subject_data = load_seed_file_paths(cfg.data.root)
    logger.info(f"Subjects: {sorted(subject_data.keys())}")

    all_results = []

    for train_dataset, test_dataset, test_subject in get_cross_subject_splits(subject_data):
        logger.info(f"\n{'='*50}")
        logger.info(f"Test Subject: {test_subject}")
        logger.info(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                                  shuffle=True, num_workers=cfg.train.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size,
                                 shuffle=False, num_workers=cfg.train.num_workers)

        save_dir = os.path.join(cfg.logging.save_dir, f"subject_{test_subject}")
        model = REVEClassifier(cfg, num_classes=cfg.data.num_classes).to(device)
        model.set_channel_info(train_dataset.ch_names)
        criterion = nn.CrossEntropyLoss()

        # Stage 1: Linear Probing (encoder frozen)
        logger.info("Stage 1: Linear Probing")
        model.freeze_encoder()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg.train.linear_probe.lr,
                          weight_decay=cfg.train.linear_probe.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.linear_probe.epochs)
        run_stage("linear_probe", model, train_loader, test_loader,
                  optimizer, scheduler, criterion, cfg, device, logger, save_dir)

        # Stage 2: Full Fine-tuning with LoRA
        logger.info("Stage 2: Fine-tuning with LoRA")
        model.unfreeze_encoder()
        model.apply_lora(cfg.train.fine_tuning)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg.train.fine_tuning.lr,
                          weight_decay=cfg.train.fine_tuning.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.fine_tuning.epochs)
        best_acc = run_stage("fine_tuning", model, train_loader, test_loader,
                             optimizer, scheduler, criterion, cfg, device, logger, save_dir)

        all_results.append({"subject": test_subject, "acc": best_acc})
        logger.info(f"Subject {test_subject} best acc: {best_acc:.4f}")

    mean_acc = sum(r["acc"] for r in all_results) / len(all_results)
    logger.info(f"\n{'='*50}")
    logger.info(f"Cross-subject mean accuracy: {mean_acc:.4f}")
    for r in all_results:
        logger.info(f"  Subject {r['subject']}: {r['acc']:.4f}")


if __name__ == "__main__":
    main()
