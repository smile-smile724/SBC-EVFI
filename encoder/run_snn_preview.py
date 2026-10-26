from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# SNN: events → per-timestep grayscale reconstruction
# ==============================================================================

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire with persistent membrane potential."""
    def __init__(self, thresh: float = 0.5, decay: float = 0.5):
        super().__init__()
        self.thresh = thresh
        self.decay = decay

    def forward(self, current: torch.Tensor, membrane: torch.Tensor | None = None):
        if membrane is None:
            membrane = torch.zeros_like(current)

        membrane = membrane * self.decay
        membrane = membrane + current
        spike = (membrane >= self.thresh).float()
        membrane = membrane - spike * self.thresh

        return spike, membrane


class SNNFromEvents(nn.Module):
    """
    SNN: event slices → per-timestep grayscale reconstruction.

    Outputs:
      per_t:     (B, T, 1, H, W) — per-timestep grayscale
      mean_gray: (B, 1, H, W)     — time-averaged grayscale
    """
    def __init__(self, T: int = 4, hidden: int = 64):
        super().__init__()
        self.T = T

        self.encoder = nn.Sequential(
            nn.Conv2d(2, hidden, 3, 1, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.lif = LIFNeuron(thresh=0.5, decay=0.5)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, 1, 1),
            nn.GroupNorm(min(8, hidden // 2), hidden // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward_serial(self, event_slices: torch.Tensor, membrane: torch.Tensor | None = None):
        B, T, _, H, W = event_slices.shape
        T_use = min(T, self.T)

        decoded = []
        for t in range(T_use):
            feat = self.encoder(event_slices[:, t])
            spike, membrane = self.lif(feat, membrane)
            gray_t = self.decoder(spike)
            decoded.append(gray_t)

        per_t = torch.stack(decoded, dim=1)                  # (B, T, 1, H, W)
        mean_gray = per_t.mean(dim=1)                        # (B, 1, H, W)

        return per_t, mean_gray, membrane

    def forward(self, event_slices: torch.Tensor):
        per_t, mean_gray, _ = self.forward_serial(event_slices, None)
        return per_t, mean_gray


# ==============================================================================
# Data loading utilities
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SNN preview: visualize raw SNN output")
    parser.add_argument("--packet-dir", type=Path, required=True,
                        help="Path to decompress-ing packet directory")
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Path to ground truth frames directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path (default: output/snn_preview_YYYYMMDD_HHMMSS/panel.avi)")
    parser.add_argument("--target-size", type=int, nargs=2, default=None,
                        help="Target resolution H W (default: auto-detect from keyframes)")
    parser.add_argument("--num-slices", type=int, default=8,
                        help="Number of temporal event slices")
    parser.add_argument("--num-pairs", type=int, default=3,
                        help="Number of keyframe pairs to render")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    return parser.parse_args()


def load_keyframes(keyframe_dir: Path) -> dict[int, np.ndarray]:
    kf = {}
    for p in sorted(keyframe_dir.glob("*.png")):
        fid = int(p.stem)
        kf[fid] = cv2.imread(str(p))
    return kf


def load_events(npz_path: Path) -> dict:
    return dict(np.load(npz_path, allow_pickle=True))


def load_gt_frames(gt_dir: Path) -> dict[int, np.ndarray]:
    gt = {}
    for p in sorted(gt_dir.glob("*.png")):
        fid = int(p.stem)
        img = cv2.imread(str(p))
        if img is not None:
            gt[fid] = img
    return gt


def detect_event_downsample(events: dict, target_w: int, target_h: int) -> int:
    x_arr = events.get("x", [])
    y_arr = events.get("y", [])
    if len(x_arr) == 0 or len(y_arr) == 0:
        return 1
    max_x = int(np.max(x_arr))
    max_y = int(np.max(y_arr))
    ds_x = max(1, int(round(target_w / max(1, max_x + 1))))
    ds_y = max(1, int(round(target_h / max(1, max_y + 1))))
    return max(ds_x, ds_y)


def extract_temporal_slices(events, start_fid, end_fid, target_h, target_w, num_slices=4, event_downsample=2):
    frame_offsets = events.get("frame_offsets", [])
    x_arr = events.get("x", [])
    y_arr = events.get("y", [])
    polarity_arr = events.get("polarity", [])
    delta_arr = events.get("delta", None)

    if len(frame_offsets) <= 1:
        return np.zeros((num_slices, 2, target_h, target_w), dtype=np.float32)

    total_frames = len(frame_offsets) - 1
    if end_fid > total_frames:
        end_fid = total_frames
    if start_fid >= end_fid:
        return np.zeros((num_slices, 2, target_h, target_w), dtype=np.float32)

    start_offset = int(frame_offsets[start_fid])
    end_offset = int(frame_offsets[end_fid])

    if end_offset <= start_offset:
        return np.zeros((num_slices, 2, target_h, target_w), dtype=np.float32)

    sub_x = x_arr[start_offset:end_offset]
    sub_y = y_arr[start_offset:end_offset]
    sub_p = polarity_arr[start_offset:end_offset]
    if delta_arr is not None:
        sub_d_all = delta_arr[start_offset:end_offset]
    else:
        sub_d_all = np.ones(len(sub_x), dtype=np.float32)

    total_frames_in_range = end_fid - start_fid
    total_events = end_offset - start_offset

    slices = np.zeros((num_slices, 2, target_h, target_w), dtype=np.float32)

    if total_frames_in_range < num_slices:
        events_per_bin = max(1, total_events // num_slices)
        for s in range(num_slices):
            s_off = s * events_per_bin
            e_off = min((s + 1) * events_per_bin, total_events)
            if s_off >= e_off:
                continue
            xs = sub_x[s_off:e_off].astype(np.int32)
            ys = sub_y[s_off:e_off].astype(np.int32)
            ps = sub_p[s_off:e_off]
            ds = np.abs(sub_d_all[s_off:e_off].astype(np.float32))

            xs_full = (xs * event_downsample + event_downsample // 2).astype(np.int32)
            ys_full = (ys * event_downsample + event_downsample // 2).astype(np.int32)

            valid = (xs_full >= 0) & (xs_full < target_w) & (ys_full >= 0) & (ys_full < target_h)
            xs_full, ys_full, ps, ds = xs_full[valid], ys_full[valid], ps[valid], ds[valid]

            for i in range(len(xs_full)):
                delta_val = max(ds[i], 1.0)
                if ps[i] >= 0:
                    slices[s, 0, ys_full[i], xs_full[i]] += delta_val
                else:
                    slices[s, 1, ys_full[i], xs_full[i]] += delta_val
    else:
        boundaries = np.linspace(start_fid, end_fid, num_slices + 1, dtype=int)
        for s in range(num_slices):
            bin_start = boundaries[s]
            bin_end = boundaries[s + 1]
            if bin_end <= bin_start:
                continue
            s_off = int(frame_offsets[bin_start]) - start_offset
            e_off = int(frame_offsets[bin_end]) - start_offset
            if s_off >= e_off:
                continue
            xs = sub_x[s_off:e_off].astype(np.int32)
            ys = sub_y[s_off:e_off].astype(np.int32)
            ps = sub_p[s_off:e_off]
            ds = np.abs(sub_d_all[s_off:e_off].astype(np.float32))

            xs_full = (xs * event_downsample + event_downsample // 2).astype(np.int32)
            ys_full = (ys * event_downsample + event_downsample // 2).astype(np.int32)

            valid = (xs_full >= 0) & (xs_full < target_w) & (ys_full >= 0) & (ys_full < target_h)
            xs_full, ys_full, ps, ds = xs_full[valid], ys_full[valid], ps[valid], ds[valid]

            for i in range(len(xs_full)):
                delta_val = max(ds[i], 1.0)
                if ps[i] >= 0:
                    slices[s, 0, ys_full[i], xs_full[i]] += delta_val
                else:
                    slices[s, 1, ys_full[i], xs_full[i]] += delta_val

    return slices


def compute_static_mask(event_slices):
    total = event_slices.sum(axis=(0, 1))
    return ((total == 0) * 255).astype(np.uint8)


def make_panel_grid(images, labels, cols, cell_h, cell_w):
    n = len(images)
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = i // cols, i % cols
        y0, x0 = r * cell_h, c * cell_w

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[:2] != (cell_h, cell_w):
            img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)

        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = img

        cv2.putText(canvas, label, (x0 + 4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    return canvas


def compute_mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def diff_heatmap(diff: np.ndarray, max_val: float = 60.0) -> np.ndarray:
    diff_clipped = np.clip(diff / max_val, 0, 1)
    heat = (diff_clipped * 255).astype(np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_HOT)


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    keyframes = load_keyframes(args.packet_dir / "keyframes")
    events = load_events(args.packet_dir / "events.npz")
    gt_frames = load_gt_frames(args.gt_dir)

    if not keyframes:
        print("[error] no keyframes found")
        return

    first_kf = next(iter(keyframes.values()))
    kf_h, kf_w = first_kf.shape[:2]
    if args.target_size is not None:
        target_h, target_w = args.target_size
    else:
        target_h, target_w = kf_h, kf_w

    print(f"[preview] keyframes={len(keyframes)} gt_frames={len(gt_frames)} "
          f"events={len(events.get('x', []))} resolution={target_w}x{target_h}")

    event_downsample = detect_event_downsample(events, target_w, target_h)
    print(f"[preview] auto-detected event_downsample={event_downsample}")

    model = SNNFromEvents(T=args.num_slices, hidden=128)
    model = model.to(device)
    model.eval()

    print(f"[preview] model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"output/snn_preview_{timestamp}/panel.avi")
    else:
        args.output = args.output.with_suffix(".avi")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Output: two 1×2 videos (SNN | GT), full cell resolution
    out_dir = args.output.parent
    frames_acc = out_dir / ".frames_acc"
    frames_frame = out_dir / ".frames_frame"
    frames_acc.mkdir(parents=True, exist_ok=True)
    frames_frame.mkdir(parents=True, exist_ok=True)

    cell_w, cell_h = target_w, target_h
    print(f"[preview] output dir -> {out_dir}/ (acc_vs_gt.mp4 + frame_vs_gt.mp4)")

    kf_ids = sorted(keyframes.keys())
    max_pairs = min(args.num_pairs, len(kf_ids) - 1)

    frame_idx = 0

    for pair_idx in range(max_pairs):
        prev_kf = kf_ids[pair_idx]
        next_kf = kf_ids[pair_idx + 1]

        span = next_kf - prev_kf
        if span <= 1:
            continue

        membrane_acc = None

        for target_fid in range(prev_kf + 1, next_kf):
            if target_fid not in gt_frames:
                continue

            gt_rgb = cv2.resize(gt_frames[target_fid], (target_w, target_h), interpolation=cv2.INTER_AREA)
            gt_gray = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2GRAY)

            event_slices = extract_temporal_slices(
                events, target_fid - 1, target_fid, target_h, target_w, args.num_slices, event_downsample,
            )
            events_t = torch.from_numpy(event_slices).float().unsqueeze(0).to(device)

            # Accumulated SNN: keep membrane across frames
            with torch.no_grad():
                per_t_acc, _, membrane_acc = model.forward_serial(events_t, membrane_acc)

            # Per-frame SNN: fresh membrane, only this frame's events
            with torch.no_grad():
                per_t_frame, _, _ = model.forward_serial(events_t, None)

            acc_frame = (per_t_acc[0, -1, 0].cpu().numpy() * 255).astype(np.uint8)
            frame_frame = (per_t_frame[0, -1, 0].cpu().numpy() * 255).astype(np.uint8)

            mse_acc = compute_mse(acc_frame, gt_gray)
            mse_frame = compute_mse(frame_frame, gt_gray)
            alpha_f = (target_fid - prev_kf) / span

            # Build 1×2 panels: [SNN | GT]
            acc_panel = make_panel_grid(
                [acc_frame, gt_gray],
                [f"acc a={alpha_f:.2f} MSE={mse_acc:.1f}", f"GT={target_fid}"],
                cols=2, cell_h=cell_h, cell_w=cell_w,
            )
            frame_panel = make_panel_grid(
                [frame_frame, gt_gray],
                [f"frame a={alpha_f:.2f} MSE={mse_frame:.1f}", f"GT={target_fid}"],
                cols=2, cell_h=cell_h, cell_w=cell_w,
            )

            cv2.imwrite(str(frames_acc / f"{frame_idx:06d}.png"), acc_panel)
            cv2.imwrite(str(frames_frame / f"{frame_idx:06d}.png"), frame_panel)
            frame_idx += 1

            print(f"[preview] pair {pair_idx+1}/{max_pairs}: "
                  f"prev={prev_kf} target={target_fid} alpha={alpha_f:.3f} "
                  f"MSE(acc)={mse_acc:.1f} MSE(frame)={mse_frame:.1f}")

    print(f"[preview] encoding {frame_idx} frames with ffmpeg...")
    for name, frames_dir in [("acc_vs_gt", frames_acc), ("frame_vs_gt", frames_frame)]:
        out_path = out_dir / f"{name}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "5",
            "-i", str(frames_dir / "%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        shutil.rmtree(str(frames_dir))
        print(f"[preview]   {out_path}")

    print(f"[preview] done")


if __name__ == "__main__":
    main()