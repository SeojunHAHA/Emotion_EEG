import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class SEEDDataset(Dataset):
    """
    SEED pkl 데이터셋

    각 샘플: dict with keys X (62, 800), Y, ch_names, ch_locations
    구조: sorted_scaled/{sub}_{ses}/{label}/SEED_{sub}_{ses}_{trial}_{idx}.pkl
    """

    def __init__(self, file_paths):
        self.file_paths = file_paths

        # 채널 정보는 모든 파일에서 동일 → 첫 파일에서 한 번만 로드
        with open(file_paths[0], 'rb') as f:
            sample = pickle.load(f)
        self.ch_names = sample['ch_names']

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'rb') as f:
            data = pickle.load(f)
        eeg = torch.FloatTensor(data['X'])       # (62, 800)
        label = torch.tensor(data['Y'], dtype=torch.long)
        return eeg, label


def load_seed_file_paths(root):
    """
    sorted_scaled 폴더에서 pkl 파일 경로를 subject_id 기준으로 수집

    Returns:
        subject_data: dict {subject_id (int): [file_path, ...]}
    """
    scaled_dir = os.path.join(root, "sorted_scaled")
    subject_data = {}

    for folder in sorted(os.listdir(scaled_dir)):
        folder_path = os.path.join(scaled_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        subject_id = int(folder.split('_')[0])  # "1_2" → 1

        for label in ['0', '1', '2']:
            label_dir = os.path.join(folder_path, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if fname.endswith('.pkl'):
                    path = os.path.join(label_dir, fname)
                    subject_data.setdefault(subject_id, []).append(path)

    return subject_data


def get_intra_split(file_paths, adapt_ratio=0.80, seed=42):
    """
    test subject 데이터를 adapt / eval 로 stratified split

    Returns:
        (adapt_dataset, eval_dataset)
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for p in file_paths:
        label = os.path.basename(os.path.dirname(p))  # '0', '1', '2'
        by_label[label].append(p)

    adapt_paths, eval_paths = [], []
    for paths in by_label.values():
        rng.shuffle(paths)
        n_adapt = max(1, int(len(paths) * adapt_ratio))
        adapt_paths.extend(paths[:n_adapt])
        eval_paths.extend(paths[n_adapt:])

    return SEEDDataset(adapt_paths), SEEDDataset(eval_paths)


def get_cross_subject_splits(subject_data):
    """
    Leave-one-subject-out cross-subject split

    Yields:
        (train_dataset, test_dataset, test_subject_id)
    """
    subject_ids = sorted(subject_data.keys())

    for test_id in subject_ids:
        train_paths, test_paths = [], []

        for sid in subject_ids:
            if sid == test_id:
                test_paths.extend(subject_data[sid])
            else:
                train_paths.extend(subject_data[sid])

        yield SEEDDataset(train_paths), SEEDDataset(test_paths), test_id
