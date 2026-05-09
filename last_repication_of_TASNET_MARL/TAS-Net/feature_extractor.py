from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import scipy.io as scio
import torch

from Model_architecture import EEGfuseNet_Channel_62


SEED_LABEL_ORDER = np.array([2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0], dtype=np.int64)
TRIAL_KEY_RE = re.compile(r".*eeg(\d+)$", re.IGNORECASE)


def norminy(x: np.ndarray) -> np.ndarray:
    """
    Normalize each channel independently to [-1, 1].
    x shape: (channels, time)
    """
    x = x.astype(np.float32)
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)
    return (x - mn) / (mx - mn + 1e-8) * 2.0 - 1.0


def to_channels_time(trial: np.ndarray, n_channels: int = 62) -> np.ndarray:
    """
    Convert raw trial to shape (channels, time).
    Accepts either (62, T) or (T, 62).
    """
    trial = np.asarray(trial).squeeze().astype(np.float32)

    if trial.ndim != 2:
        raise ValueError(f"Expected 2D trial array, got shape {trial.shape}")

    if trial.shape[0] == n_channels:
        pass
    elif trial.shape[1] == n_channels:
        trial = trial.T
    else:
        raise ValueError(
            f"Cannot find channel dimension {n_channels} in shape {trial.shape}. "
            f"Expected either ({n_channels}, T) or (T, {n_channels})."
        )

    if trial.shape[0] != n_channels:
        raise ValueError(f"Channel dimension mismatch after transpose: {trial.shape}")

    return trial


def segment_trial(
    trial_ct: np.ndarray,
    window_size: int = 384,
    stride: int = 384,
    pad_tail: bool = True,
) -> list[np.ndarray]:
    """
    Split one trial into fixed-length windows along time.

    Input:
      trial_ct: (channels, time)

    Output:
      list of windows, each with shape (channels, window_size)
    """
    c, t = trial_ct.shape
    windows: list[np.ndarray] = []

    if t <= 0:
        return windows

    start = 0
    while start < t:
        end = start + window_size
        window = trial_ct[:, start:end]

        if window.shape[1] < window_size:
            if not pad_tail:
                break
            pad = np.zeros((c, window_size - window.shape[1]), dtype=np.float32)
            window = np.concatenate([window, pad], axis=1)

        windows.append(window)

        if end >= t:
            break
        start += stride

    return windows


def load_subject_session_files(root: Path) -> dict[str, list[Path]]:
    """
    Group files by subject prefix.
    Example:
      10_20131130.mat -> subject '10'
    """
    subject_map: dict[str, list[Path]] = defaultdict(list)

    for p in sorted(root.glob("*.mat")):
        if p.name.lower() == "label.mat":
            continue
        subject_id = p.stem.split("_", 1)[0]
        subject_map[subject_id].append(p)

    for subj in subject_map:
        subject_map[subj].sort()
    return subject_map


def load_trials_from_mat(mat_path: Path) -> list[tuple[int, str, np.ndarray]]:
    """
    Extract all top-level trial variables from one .mat file.
    Returns list of:
      (trial_index, variable_name, raw_array)
    """
    data = scio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    trials = []
    for key, value in data.items():
        if key.startswith("__"):
            continue

        m = TRIAL_KEY_RE.match(key)
        if m:
            trial_idx = int(m.group(1))
            arr = np.asarray(value)

            if arr.dtype == object and arr.size == 1:
                arr = np.asarray(arr.item())

            trials.append((trial_idx, key, arr))

    trials.sort(key=lambda x: x[0])
    return trials


def extract_features_for_trial(
    trial_raw: np.ndarray,
    model,
    device,
    window_size: int = 384,
    stride: int = 384,
    batch_size: int = 32,
    pad_tail: bool = True,
) -> np.ndarray:
    """
    Convert one raw trial into many feature vectors.
    Returns shape: (num_windows, 192)
    """
    trial_ct = to_channels_time(trial_raw, n_channels=62)
    windows = segment_trial(trial_ct, window_size=window_size, stride=stride, pad_tail=pad_tail)

    if not windows:
        return np.empty((0, 192), dtype=np.float32)

    window_batch = []
    for w in windows:
        w = norminy(w)
        window_batch.append(w[np.newaxis, :, :])  # (1, 62, 384)

    feats = []
    with torch.no_grad():
        for start in range(0, len(window_batch), batch_size):
            end = min(start + batch_size, len(window_batch))
            batch_np = np.stack(window_batch[start:end], axis=0).astype(np.float32)  # (B, 1, 62, 384)
            batch = torch.from_numpy(batch_np).to(device)
            _, features = model(batch)
            feats.append(features.detach().cpu().numpy())

    return np.vstack(feats).astype(np.float32)


def build_feature_h5(
    data_root: Path,
    output_h5: Path,
    pretrained_model_path: Path,
    session_id: int = 1,
    batch_size: int = 32,
    window_size: int = 384,
    stride: int = 384,
):
    """
    For each subject, load the chosen session file, split each trial into windows,
    extract EEGFuseNet features per window, and save them as:
      source_01_video_01/features
      source_01_video_01/labels
      ...
    """
    if session_id not in (1, 2, 3):
        raise ValueError("session_id must be 1, 2, or 3")

    session_index = session_id - 1
    subject_files = load_subject_session_files(data_root)

    def subject_sort_key(x: str):
        return int(x) if x.isdigit() else x

    subjects = sorted(subject_files.keys(), key=subject_sort_key)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EEGfuseNet_Channel_62(16, 1, 1, window_size).to(device)
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    # Store features per subject/trial group
    group_features: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    total_windows_per_subject: dict[int, int] = defaultdict(int)

    for subj in subjects:
        files = subject_files[subj]
        if len(files) <= session_index:
            print(f"[WARN] Subject {subj} has only {len(files)} file(s), skipping session {session_id}")
            continue

        mat_file = files[session_index]
        trials = load_trials_from_mat(mat_file)

        if len(trials) != 15:
            print(f"[WARN] {mat_file.name} has {len(trials)} trials, expected 15")

        print(f"[INFO] Loading {mat_file.name} -> {len(trials)} trials")

        for trial_idx, var_name, raw_trial in trials:
            feat = extract_features_for_trial(
                raw_trial,
                model=model,
                device=device,
                window_size=window_size,
                stride=stride,
                batch_size=batch_size,
                pad_tail=True,
            )

            if feat.shape[0] == 0:
                print(f"[WARN] Empty feature result for {mat_file.name} / {var_name}")
                continue

            group_features[(int(subj), trial_idx)].append(feat)
            total_windows_per_subject[int(subj)] += feat.shape[0]

            print(
                f"[INFO] {mat_file.name} / {var_name} -> windows: {feat.shape[0]}, feat shape: {feat.shape}"
            )

    # Save HDF5 in TAS-Net-friendly structure
    with h5py.File(output_h5, "w") as f:
        saved_groups = 0

        for subj in sorted(total_windows_per_subject.keys()):
            print(f"[INFO] Subject {subj} total windows: {total_windows_per_subject[subj]}")

        for subj in sorted(subject_files.keys(), key=subject_sort_key):
            for trial_no in range(1, 16):
                key = (int(subj), trial_no)
                if key not in group_features:
                    continue

                feat = np.vstack(group_features[key]).astype(np.float32)
                group_name = f"source_{int(subj):02d}_video_{trial_no:02d}"

                g = f.create_group(group_name)
                g.create_dataset("features", data=feat)
                g.create_dataset("labels", data=np.array(SEED_LABEL_ORDER[trial_no - 1], dtype=np.int64))

                saved_groups += 1

        print(f"[INFO] Saved {saved_groups} groups to: {output_h5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEED feature extractor adapted for local .mat files")
    parser.add_argument(
        "--data_root",
        type=str,
        default=r"D:\PycharmProjects\pythonProject\RL_reserach\Preprocessed_EEG\Preprocessed_EEG",
        help="Folder containing SEED .mat files",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=r"D:\PycharmProjects\pythonProject\RL_reserach\last_repication_of_TASNET\features",
        help="Where to save extracted feature .h5 file",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=r"D:\PycharmProjects\pythonProject\RL_reserach\last_repication_of_TASNET\Pretrained_model_SEED.pkl",
        help="Path to EEGFuseNet pretrained weights",
    )
    parser.add_argument(
        "--session_id",
        type=int,
        default=1,
        help="Session to process (1, 2, or 3).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=384,
        help="Window length for segmenting raw EEG.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=384,
        help="Stride for segmenting raw EEG.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    pretrained_model = Path(args.pretrained_model)

    out_dir = output_root / f"session_{args.session_id}"
    out_h5 = out_dir / "source_h5_file.h5"

    build_feature_h5(
        data_root=data_root,
        output_h5=out_h5,
        pretrained_model_path=pretrained_model,
        session_id=args.session_id,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
    )