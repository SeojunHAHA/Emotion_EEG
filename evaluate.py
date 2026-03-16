import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from utils import load_config, get_logger, load_checkpoint
from data.dataset import load_seed_file_paths, get_cross_subject_splits
from models.reve_classifier import REVEClassifier

LABEL_NAMES = ["negative", "neutral", "positive"]


@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for eeg, labels in loader:
        eeg = eeg.to(device)
        logits = model(eeg)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    cfg = load_config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger = get_logger("evaluate")

    logger.info("Loading SEED file paths...")
    subject_data = load_seed_file_paths(cfg.data.root)

    all_accs = []

    for _, test_dataset, test_subject in get_cross_subject_splits(subject_data):
        ckpt_path = f"{cfg.logging.save_dir}/subject_{test_subject}/best_fine_tuning.pt"
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size,
                                 shuffle=False, num_workers=cfg.train.num_workers)

        model = REVEClassifier(cfg, num_classes=cfg.data.num_classes).to(device)
        model.set_channel_info(test_dataset.ch_names)
        model.apply_lora(cfg.train.fine_tuning)
        load_checkpoint(model, None, ckpt_path)

        preds, labels = evaluate_full(model, test_loader, device)
        acc = (preds == labels).mean()
        all_accs.append(acc)

        logger.info(f"\nSubject {test_subject} | Accuracy: {acc:.4f}")
        logger.info("\n" + classification_report(labels, preds, target_names=LABEL_NAMES))
        logger.info("Confusion Matrix:")
        logger.info(str(confusion_matrix(labels, preds)))

    logger.info(f"\nCross-subject mean accuracy: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f}")


if __name__ == "__main__":
    main()
