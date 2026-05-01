"""
Subject-dependent evaluation (upper bound check).

Split: Session 1, 2 → train | Session 3 → test
Each subject trained independently — no cross-subject transfer.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import set_seed, load_config, get_logger, save_checkpoint
from data.dataset import SEEDDataset, load_seed_file_paths
from models.reve_classifier import REVEClassifier
from train import mixup_batch, train_one_epoch, evaluate, run_stage


def get_subject_dependent_splits(subject_data):
    """
    For each subject, split by session:
      train: sessions 1, 2
      test:  session 3

    File name format: SEED_{sub}_{ses}_{trial}_{idx}.pkl
    Session is extracted from the filename (index 2 after split by '_').

    Yields:
        (train_dataset, test_dataset, subject_id)
    """
    for subject_id, file_paths in sorted(subject_data.items()):
        train_paths, test_paths = [], []
        for path in file_paths:
            fname = os.path.basename(path)           # SEED_1_2_11_0.pkl
            parts = fname.split('_')                 # ['SEED', '1', '2', '11', '0.pkl']
            session = int(parts[2])
            if session == 3:
                test_paths.append(path)
            else:
                train_paths.append(path)

        if not train_paths or not test_paths:
            continue

        yield SEEDDataset(train_paths), SEEDDataset(test_paths), subject_id


def main():
    cfg = load_config()
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join(cfg.logging.log_dir, "subject_dependent")
    save_dir = os.path.join(cfg.logging.save_dir, "subject_dependent")
    os.makedirs(log_dir, exist_ok=True)

    logger = get_logger("train_sd", log_dir=log_dir)
    logger.info(f"Device: {device}")
    logger.info("Subject-dependent evaluation | Train: ses 1+2 | Test: ses 3")

    logger.info("Loading SEED file paths...")
    subject_data = load_seed_file_paths(cfg.data.root)
    logger.info(f"Subjects: {sorted(subject_data.keys())}")

    results_path = os.path.join(log_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            saved = json.load(f)
        all_results = saved.get("subjects", [])
        done_subjects = {r["subject"] for r in all_results}
        logger.info(f"Resuming: {len(done_subjects)} subjects already done {sorted(done_subjects)}")
    else:
        all_results = []
        done_subjects = set()

    for train_dataset, test_dataset, subject_id in get_subject_dependent_splits(subject_data):
        if subject_id in done_subjects:
            logger.info(f"Skipping Subject {subject_id} (already done)")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Subject: {subject_id} | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                                  shuffle=True, num_workers=cfg.train.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size,
                                 shuffle=False, num_workers=cfg.train.num_workers)

        model = REVEClassifier(cfg, num_classes=cfg.data.num_classes).to(device)
        model.set_channel_info(train_dataset.ch_names)
        criterion = nn.CrossEntropyLoss()

        # Stage 1: Linear Probing
        logger.info("Stage 1: Linear Probing")
        model.freeze_encoder()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg.train.linear_probe.lr,
                          weight_decay=cfg.train.linear_probe.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        lp_metrics = run_stage("linear_probe", model, train_loader, test_loader,
                               optimizer, scheduler, criterion, cfg.train.linear_probe, device, logger,
                               num_classes=cfg.data.num_classes)

        # Stage 2: Fine-tuning with LoRA
        logger.info("Stage 2: Fine-tuning with LoRA")
        model.unfreeze_encoder()
        model.apply_lora(cfg.train.fine_tuning)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg.train.fine_tuning.lr,
                          weight_decay=cfg.train.fine_tuning.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        ft_metrics = run_stage("fine_tuning", model, train_loader, test_loader,
                               optimizer, scheduler, criterion, cfg.train.fine_tuning, device, logger,
                               num_classes=cfg.data.num_classes)

        subject_result = {
            "subject": subject_id,
            "linear_probe": lp_metrics,
            "fine_tuning": ft_metrics,
        }
        all_results.append(subject_result)
        logger.info(
            f"Subject {subject_id} | "
            f"LP acc: {lp_metrics['test_acc']:.4f} f1: {lp_metrics['test_f1']:.4f} | "
            f"FT acc: {ft_metrics['test_acc']:.4f} f1: {ft_metrics['test_f1']:.4f}"
        )

        # Save incremental results
        lp_accs = [r["linear_probe"]["test_acc"] for r in all_results]
        ft_accs  = [r["fine_tuning"]["test_acc"]  for r in all_results]
        ft_f1s   = [r["fine_tuning"]["test_f1"]   for r in all_results]
        summary = {
            "lp_mean_acc":  float(np.mean(lp_accs)),
            "ft_mean_acc":  float(np.mean(ft_accs)),
            "ft_std_acc":   float(np.std(ft_accs)),
            "ft_mean_f1":   float(np.mean(ft_f1s)),
            "ft_std_f1":    float(np.std(ft_f1s)),
            "subjects": all_results,
        }
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results updated: {results_path}")

    # Final summary
    lp_accs = [r["linear_probe"]["test_acc"] for r in all_results]
    ft_accs  = [r["fine_tuning"]["test_acc"]  for r in all_results]
    ft_f1s   = [r["fine_tuning"]["test_f1"]   for r in all_results]
    summary = {
        "lp_mean_acc":  float(np.mean(lp_accs)),
        "ft_mean_acc":  float(np.mean(ft_accs)),
        "ft_std_acc":   float(np.std(ft_accs)),
        "ft_mean_f1":   float(np.mean(ft_f1s)),
        "ft_std_f1":    float(np.std(ft_f1s)),
        "subjects": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"LP   mean acc: {summary['lp_mean_acc']:.4f}")
    logger.info(f"FT   mean acc: {summary['ft_mean_acc']:.4f} ± {summary['ft_std_acc']:.4f}")
    logger.info(f"FT   mean F1:  {summary['ft_mean_f1']:.4f} ± {summary['ft_std_f1']:.4f}")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
