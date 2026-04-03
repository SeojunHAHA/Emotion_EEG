"""
sorted/ -> sorted_scaled/ 전처리 스크립트

REVE 논문 기준:
- 0.5-99.5 Hz bandpass filter
- Z-score normalization (session 전체 통계 기준)
- ±15 std clipping
"""

import os
import pickle
import numpy as np
from scipy.signal import butter, sosfilt
from collections import defaultdict


SFREQ = 200
LOWCUT = 0.5
HIGHCUT = 99.5
CLIP_STD = 15


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """data: (n_channels, n_times)"""
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfilt(sos, data, axis=-1)


def load_session_paths(scaled_dir, subject_session):
    """session 내 모든 pkl 경로 수집"""
    paths = []
    folder = os.path.join(scaled_dir, subject_session)
    for label in ['0', '1', '2']:
        label_dir = os.path.join(folder, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if fname.endswith('.pkl'):
                paths.append(os.path.join(label_dir, fname))
    return paths


def compute_session_stats(paths):
    """session 전체 데이터로 channel-wise mean/std 계산"""
    all_data = []
    for p in paths:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        all_data.append(d['X'])  # (62, 800)
    all_data = np.concatenate(all_data, axis=-1)  # (62, N)
    mean = all_data.mean(axis=-1, keepdims=True)   # (62, 1)
    std = all_data.std(axis=-1, keepdims=True)     # (62, 1)
    std = np.where(std < 1e-6, 1.0, std)           # zero-std 방지
    return mean, std


def preprocess_session(src_root, dst_root, subject_session):
    paths = load_session_paths(src_root, subject_session)
    if not paths:
        return 0

    # session 전체 통계 계산
    mean, std = compute_session_stats(paths)

    count = 0
    for src_path in paths:
        # 상대 경로 유지
        rel = os.path.relpath(src_path, src_root)
        dst_path = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        with open(src_path, 'rb') as f:
            d = pickle.load(f)

        x = d['X'].astype(np.float32)              # (62, 800)
        x = bandpass_filter(x, LOWCUT, HIGHCUT, SFREQ)
        x = (x - mean) / std
        x = np.clip(x, -CLIP_STD, CLIP_STD)
        x = x.astype(np.float32)

        d_out = {
            'X': x,
            'Y': d['Y'],
            'ch_names': d['ch_names'],
            'ch_locations': d['ch_locations'],
        }

        with open(dst_path, 'wb') as f:
            pickle.dump(d_out, f)
        count += 1

    return count


def main():
    src_root = '/home/seojun/Datasets/SEED/sorted'
    dst_root = '/home/seojun/Datasets/SEED/sorted_scaled'

    sessions = sorted(os.listdir(src_root))
    total = 0

    for session in sessions:
        if not os.path.isdir(os.path.join(src_root, session)):
            continue
        count = preprocess_session(src_root, dst_root, session)
        print(f"{session}: {count} files")
        total += count

    print(f"\nDone. Total: {total} files -> {dst_root}")


if __name__ == '__main__':
    main()
