from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List


def build_keyframe_pairs(max_kf: int, key_interval: int) -> list[int]:
    pairs = list(range(0, max_kf, key_interval))
    return pairs


def discover_sequences(data_root: Path, split: str) -> List[dict]:
    split_dir = data_root / split
    if not split_dir.is_dir():
        return []

    seqs = []
    for seq_dir in sorted(split_dir.iterdir()):
        if not seq_dir.is_dir() or seq_dir.name.startswith("."):
            continue
        events_path = seq_dir / "decompress-ing" / "packets"
        npz_files = list(events_path.rglob("events.npz"))
        if not npz_files:
            continue
        packet_dir = npz_files[0].parent
        seqs.append({
            "seq_id": seq_dir.name,
            "packet_dir": packet_dir,
            "gt_dir_name": seq_dir.name,
        })
    return seqs


def preload_events(seqs: List[dict]) -> Dict[str, dict]:
    events_cache = {}
    for seq in seqs:
        events = dict(np.load(seq["packet_dir"] / "events.npz", allow_pickle=True))
        events_cache[seq["seq_id"]] = events
    return events_cache


def count_gt_frames(gt_root: Path, seqs: List[dict]) -> Dict[str, int]:
    frame_counts = {}
    for seq in seqs:
        gt_dir = gt_root / seq["gt_dir_name"]
        n = len(list(gt_dir.glob("*.png")))
        frame_counts[seq["seq_id"]] = n
    return frame_counts


def load_gt_frame(gt_dir: Path, fid: int, target_h: int, target_w: int) -> np.ndarray:
    p = gt_dir / f"{fid:08d}.png"
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(f"missing GT frame: {p}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    if img_rgb.shape[:2] != (target_h, target_w):
        img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img_rgb


def extract_temporal_slices(
    events: dict,
    start_fid: int,
    end_fid: int,
    target_h: int,
    target_w: int,
    num_slices: int = 8,
    event_downsample: int = 2,
):
    frame_offsets = events.get("frame_offsets", [])
    x_arr = events.get("x", [])
    y_arr = events.get("y", [])
    polarity_arr = events.get("polarity", [])
    delta_arr = events.get("delta", [])

    if len(frame_offsets) <= 1:
        return np.zeros((num_slices, 2, target_h, target_w), dtype=np.float32)

    total_frames = len(frame_offsets) - 1
    if end_fid > total_frames:
        end_fid = total_frames
    if start_fid >= end_fid:
        return np.zeros((num_slices, 2, target_h, target_w), dtype=np.float32)

    slices = np.zeros((num_slices, 2, target_h, target_w), dtype=np.float32)
    num_pairs = end_fid - start_fid

    has_delta = len(delta_arr) > 0

    for pair_idx in range(num_pairs):
        fid = start_fid + pair_idx
        s_off = int(frame_offsets[fid])
        e_off = int(frame_offsets[fid + 1])

        if e_off <= s_off:
            continue

        slice_idx = pair_idx * num_slices // num_pairs
        slice_idx = min(slice_idx, num_slices - 1)

        xs = x_arr[s_off:e_off].astype(np.int32)
        ys = y_arr[s_off:e_off].astype(np.int32)
        ps = polarity_arr[s_off:e_off]

        xs_full = (xs * event_downsample + event_downsample // 2).astype(np.int32)
        ys_full = (ys * event_downsample + event_downsample // 2).astype(np.int32)

        valid = (xs_full >= 0) & (xs_full < target_w) & (ys_full >= 0) & (ys_full < target_h)
        xs_full = xs_full[valid]
        ys_full = ys_full[valid]
        ps = ps[valid]

        pos_mask = (ps > 0)
        neg_mask = (ps < 0)

        if has_delta:
            ds = delta_arr[s_off:e_off][valid]
            np.add.at(slices[slice_idx, 0], (ys_full[pos_mask], xs_full[pos_mask]), ds[pos_mask])
            np.add.at(slices[slice_idx, 1], (ys_full[neg_mask], xs_full[neg_mask]), ds[neg_mask])
        else:
            np.add.at(slices[slice_idx, 0], (ys_full[pos_mask], xs_full[pos_mask]), ps[pos_mask])
            np.add.at(slices[slice_idx, 1], (ys_full[neg_mask], xs_full[neg_mask]), -ps[neg_mask])

    return slices