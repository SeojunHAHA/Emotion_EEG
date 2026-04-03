import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from utils import set_seed, load_config, get_logger, save_checkpoint
from data.dataset import load_seed_file_paths, get_cross_subject_splits, get_intra_split
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
    all_preds, all_labels = [], []

    for eeg, labels in loader:
        eeg, labels = eeg.to(device), labels.to(device)
        logits = model(eeg)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / total, acc, f1


def run_stage(stage_name, model, train_loader, test_loader,
              optimizer, scheduler, criterion, stage_cfg, device, logger):
    best_metrics = {}

    for epoch in range(stage_cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        is_best = not best_metrics or test_acc > best_metrics["test_acc"]
        if is_best:
            best_metrics = {"test_acc": test_acc, "test_f1": test_f1, "epoch": epoch + 1}

        logger.info(
            f"[{stage_name}] Epoch {epoch+1}/{stage_cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f}"
            + (" *" if is_best else "")
        )

    return best_metrics


def main():
    cfg = load_config()
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger = get_logger("train", log_dir=cfg.logging.log_dir)
    logger.info(f"Device: {device}")

    logger.info("Loading SEED file paths...")
    subject_data = load_seed_file_paths(cfg.data.root)
    logger.info(f"Subjects: {sorted(subject_data.keys())}")

    # Load existing results to resume from where we left off
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    results_path = os.path.join(cfg.logging.log_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        all_results = saved.get("subjects", [])
        done_subjects = {r["subject"] for r in all_results}
        logger.info(f"Resuming: {len(done_subjects)} subjects already done {sorted(done_subjects)}")
    else:
        all_results = []
        done_subjects = set()

    for train_dataset, test_dataset, test_subject in get_cross_subject_splits(subject_data):
        if test_subject in done_subjects:
            logger.info(f"Skipping Subject {test_subject} (already done)")
            continue

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
        lp_metrics = run_stage("linear_probe", model, train_loader, test_loader,
                               optimizer, scheduler, criterion, cfg.train.linear_probe, device, logger)

        # Stage 2: Full Fine-tuning with LoRA
        logger.info("Stage 2: Fine-tuning with LoRA")
        model.unfreeze_encoder()
        model.apply_lora(cfg.train.fine_tuning)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg.train.fine_tuning.lr,
                          weight_decay=cfg.train.fine_tuning.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.fine_tuning.epochs)
        ft_metrics = run_stage("fine_tuning", model, train_loader, test_loader,
                               optimizer, scheduler, criterion, cfg.train.fine_tuning, device, logger)

        # Stage 3: Intra-subject adaptation (5% adapt / 95% eval, classifier head only)
        logger.info("Stage 3: Intra-subject adaptation")
        intra_cfg = cfg.train.intra_adaptation
        adapt_dataset, intra_eval_dataset = get_intra_split(
            test_dataset.file_paths, adapt_ratio=intra_cfg.adapt_ratio, seed=cfg.train.seed
        )
        logger.info(f"Intra adapt: {len(adapt_dataset)} | Intra eval: {len(intra_eval_dataset)}")
        adapt_loader = DataLoader(adapt_dataset, batch_size=cfg.train.batch_size,
                                  shuffle=True, num_workers=cfg.train.num_workers)
        intra_eval_loader = DataLoader(intra_eval_dataset, batch_size=cfg.train.batch_size,
                                       shuffle=False, num_workers=cfg.train.num_workers)
        model.freeze_encoder()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=intra_cfg.lr,
                          weight_decay=intra_cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=intra_cfg.epochs)
        intra_metrics = run_stage("intra_adaptation", model, adapt_loader, intra_eval_loader,
                                  optimizer, scheduler, criterion, intra_cfg, device, logger)

        # Save model checkpoint for this subject
        ckpt_path = os.path.join(save_dir, "final.pt")
        save_checkpoint(model, optimizer, intra_metrics["epoch"], intra_metrics, ckpt_path)

        subject_result = {
            "subject": test_subject,
            "linear_probe": lp_metrics,
            "fine_tuning": ft_metrics,
            "intra_adaptation": intra_metrics,
        }
        all_results.append(subject_result)
        logger.info(
            f"Subject {test_subject} | "
            f"LP acc: {lp_metrics['test_acc']:.4f} f1: {lp_metrics['test_f1']:.4f} | "
            f"FT acc: {ft_metrics['test_acc']:.4f} f1: {ft_metrics['test_f1']:.4f} | "
            f"Intra acc: {intra_metrics['test_acc']:.4f} f1: {intra_metrics['test_f1']:.4f}"
        )

        # Save incremental results after each subject
        ft_accs     = [r["fine_tuning"]["test_acc"]       for r in all_results]
        ft_f1s      = [r["fine_tuning"]["test_f1"]        for r in all_results]
        intra_accs  = [r["intra_adaptation"]["test_acc"]  for r in all_results]
        intra_f1s   = [r["intra_adaptation"]["test_f1"]   for r in all_results]
        summary = {
            "cross_subject_mean_acc":  float(np.mean(ft_accs)),
            "cross_subject_std_acc":   float(np.std(ft_accs)),
            "cross_subject_mean_f1":   float(np.mean(ft_f1s)),
            "cross_subject_std_f1":    float(np.std(ft_f1s)),
            "intra_mean_acc":          float(np.mean(intra_accs)),
            "intra_std_acc":           float(np.std(intra_accs)),
            "intra_mean_f1":           float(np.mean(intra_f1s)),
            "intra_std_f1":            float(np.std(intra_f1s)),
            "subjects": all_results,
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Checkpoint saved: {ckpt_path} | Results updated: {results_path}")

    # Final summary
    ft_accs     = [r["fine_tuning"]["test_acc"]       for r in all_results]
    ft_f1s      = [r["fine_tuning"]["test_f1"]        for r in all_results]
    intra_accs  = [r["intra_adaptation"]["test_acc"]  for r in all_results]
    intra_f1s   = [r["intra_adaptation"]["test_f1"]   for r in all_results]
    summary = {
        "cross_subject_mean_acc":  float(np.mean(ft_accs)),
        "cross_subject_std_acc":   float(np.std(ft_accs)),
        "cross_subject_mean_f1":   float(np.mean(ft_f1s)),
        "cross_subject_std_f1":    float(np.std(ft_f1s)),
        "intra_mean_acc":          float(np.mean(intra_accs)),
        "intra_std_acc":           float(np.std(intra_accs)),
        "intra_mean_f1":           float(np.mean(intra_f1s)),
        "intra_std_f1":            float(np.std(intra_f1s)),
        "subjects": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Cross-subject mean acc: {summary['cross_subject_mean_acc']:.4f} ± {summary['cross_subject_std_acc']:.4f}")
    logger.info(f"Cross-subject mean F1:  {summary['cross_subject_mean_f1']:.4f} ± {summary['cross_subject_std_f1']:.4f}")
    logger.info(f"Intra-subject mean acc: {summary['intra_mean_acc']:.4f} ± {summary['intra_std_acc']:.4f}")
    logger.info(f"Intra-subject mean F1:  {summary['intra_mean_f1']:.4f} ± {summary['intra_std_f1']:.4f}")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
