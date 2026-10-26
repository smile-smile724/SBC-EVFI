"""Color Shift Diagnostic: measure per-step RGB means in cascade inference.

Usage (from project root):
    # Phase 1 checkpoint (loaded into Phase 2 arch, deblur_head replicated)
    python -m main.train.color_diag --ckpt experiments/test_v21/phase1/ckpt/ckpt_phase1_best.pth --device cuda
    # Phase 2 checkpoint
    python -m main.train.color_diag --ckpt experiments/test_v21/phase2/ckpt/ckpt_phase2_best.pth --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import SNNReconstruct
from .data import extract_temporal_slices, load_gt_frame, discover_sequences, preload_events, build_keyframe_pairs
from .cascade import run_cascade_inference, StepFeatureCache
from .event_mask import _make_grid_and_scale


def _load_ckpt(model, ckpt_path, device):
    """Load checkpoint, handling Phase1→Phase2 deblur_head replication."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_sd = ckpt.get('model', ckpt)
    # Strip _orig_mod prefix
    fixed = {}
    for k, v in ckpt_sd.items():
        fixed[k.replace('_orig_mod.', '')] = v

    # Phase1→Phase2: replicate deblur_head
    for dst in ['deblur_head_bwd', 'deblur_head_fwd', 'deblur_head_fusion']:
        if dst not in fixed and 'deblur_head.net.0.weight' in fixed:
            for src_k, src_v in list(fixed.items()):
                if src_k.startswith('deblur_head.'):
                    dst_k = src_k.replace('deblur_head.', dst + '.')
                    if dst_k not in fixed:
                        fixed[dst_k] = src_v.clone()

    # Phase1→Phase2: replicate offset_head -> offset_head_fwd/bwd
    for dst in ['offset_head_fwd', 'offset_head_bwd']:
        if dst not in fixed and 'offset_head.conv1.weight' in fixed:
            for src_k, src_v in list(fixed.items()):
                if src_k.startswith('offset_head.'):
                    dst_k = src_k.replace('offset_head.', dst + '.')
                    if dst_k not in fixed:
                        fixed[dst_k] = src_v.clone()

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    return missing, unexpected


def main():
    parser = argparse.ArgumentParser(description="Color shift cascade diagnostic")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/data"))
    parser.add_argument("--gt-root", type=Path, default=Path("datasets/GT"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--key-interval", type=int, default=9)
    args = parser.parse_args()

    device = torch.device(args.device)
    is_phase1_ckpt = 'phase1' in str(args.ckpt)
    label = f"Phase {'1' if is_phase1_ckpt else '2'}"
    print(f"[{label}] checkpoint: {args.ckpt}")
    print(f"  device={device}")

    # Always use Phase 2 architecture (has all cascade modules)
    model = SNNReconstruct(T=args.key_interval, hidden=args.hidden, phase=2).to(device)
    missing, unexpected = _load_ckpt(model, args.ckpt, device)
    new_keys = [k for k in missing if not k.startswith('blurry_predictor_') and not k.startswith('fusion_head.')]
    print(f"  missing: {len(missing)} ({len(new_keys)} non-blurry/fusion)")
    print(f"  unexpected: {len(unexpected)}")
    model.eval()

    # Load validation data
    val_seqs = discover_sequences(args.data_root, "val")
    if not val_seqs:
        print("ERROR: no validation sequences")
        return
    seq = val_seqs[0]
    gt_dir = args.gt_root / "val" / seq['gt_dir_name']
    gt_files = sorted(gt_dir.glob("*.png"))
    n_frames = len(gt_files)
    print(f"  seq: {seq['gt_dir_name']}  ({n_frames} frames)")

    events_dict = preload_events([seq])
    ev = events_dict[seq['seq_id']]
    event_h = int(np.max(ev['y'])) + 1 if len(ev.get('y', [])) > 0 else 360
    event_w = int(np.max(ev['x'])) + 1 if len(ev.get('x', [])) > 0 else 640
    event_ds = int(ev.get("event_support", 1) or 1)
    print(f"  event: {event_w}x{event_h}  ds={event_ds}")

    # Get first keyframe pair
    kf_positions = build_keyframe_pairs(n_frames, args.key_interval)
    if len(kf_positions) < 2:
        print("ERROR: need at least 2 keyframes")
        return
    prev_idx = kf_positions[0]
    next_idx = kf_positions[1] if len(kf_positions) > 1 else kf_positions[0] + args.key_interval
    KI = next_idx - prev_idx
    print(f"  keyframe positions: {kf_positions[:5]}...  pair: [{prev_idx}, {next_idx}]  KI={KI}")

    # Load frames (use the trainer's call pattern)
    gt_frames_cpu = []
    for fid in range(prev_idx, next_idx + 1):
        gt_np = load_gt_frame(gt_dir, fid, event_h * event_ds, event_w * event_ds)
        gt_frames_cpu.append(torch.from_numpy(gt_np).float().permute(2, 0, 1).unsqueeze(0))
    gt_frames = [f.to(device) for f in gt_frames_cpu]
    prev_kf = gt_frames[0]
    next_kf = gt_frames[-1]

    # Event slice (use trainer's call pattern)
    event_slices = extract_temporal_slices(
        ev, prev_idx, next_idx, event_h, event_w,
        num_slices=KI, event_downsample=event_ds)
    events_t = torch.from_numpy(event_slices).float().unsqueeze(0).to(device)
    # Reshape from [B, T, 2, H, W] to [B, 2*T, H, W] for cache
    B, T_slices, _, H_ev, W_ev = events_t.shape
    events_up = events_t.view(B, T_slices * 2, H_ev, W_ev)

    B, _, H_img, W_img = prev_kf.shape

    # Forward pass
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            all_feats, all_membranes, all_offset_feats = model(events_t)

    # Interpolate events_up to GT frame resolution (matching evaluator)
    events_up = torch.nn.functional.interpolate(
        events_up, size=(H_img, W_img), mode="bilinear", align_corners=True)

    H, W = H_img, W_img
    grid, scale_vec = _make_grid_and_scale(B, H, W, device)

    cache = StepFeatureCache(
        all_feats, all_membranes, all_offset_feats, events_up,
        H, W, all_feats[0].shape[1], model, device,
        mask_kernel_size=5, mask_sigma=0.5)

    # Per-step color hook
    diag = []

    def _diag_hook(t, idx, output, event_mask, sample_grid, warp_est):
        gt = gt_frames[t]
        diag.append({
            'step': t,
            'warp': tuple(warp_est.float().mean(dim=[0, 2, 3]).tolist()),
            'output': tuple(output.float().mean(dim=[0, 2, 3]).tolist()),
            'gt': tuple(gt.float().mean(dim=[0, 2, 3]).tolist()),
        })

    # ===== BACKWARD =====
    print(f"\n{'='*90}")
    print(f"BACKWARD cascade (t={KI}→1) {label}")
    print(f"{'='*90}")
    diag.clear()

    cascade_bwd, masks_bwd = run_cascade_inference(
        model, cache, KI, next_kf, grid, scale_vec,
        direction='bwd', use_deblur=True, store_inference=False,
        per_step_fn=_diag_hook)

    # Print step-by-step
    hdr = f"{'Step':>4s} {'t/KI':>6s}  " \
          f"{'GT_R':>7s} {'GT_G':>7s} {'GT_B':>7s} | " \
          f"{'warp_R':>7s} {'warp_G':>7s} {'warp_B':>7s} | " \
          f"{'out_R':>7s} {'out_G':>7s} {'out_B':>7s} | " \
          f"{'Δwarp_R':>8s} {'Δwarp_G':>8s} {'Δwarp_B':>8s} | " \
          f"{'Δout_R':>8s} {'Δout_G':>8s} {'Δout_B':>8s}"
    print(hdr)
    print("-" * 90)

    warp_errors = [[] for _ in range(3)]
    out_errors = [[] for _ in range(3)]
    for d in sorted(diag, key=lambda x: x['step'], reverse=True):
        t = d['step']
        t_ratio = (KI - t) / KI
        w_e = [d['warp'][c] - d['gt'][c] for c in range(3)]
        o_e = [d['output'][c] - d['gt'][c] for c in range(3)]
        for c in range(3):
            warp_errors[c].append(w_e[c])
            out_errors[c].append(o_e[c])
        print(f"{t:4d} {t_ratio:6.3f}  "
              f"{d['gt'][0]:7.4f} {d['gt'][1]:7.4f} {d['gt'][2]:7.4f} | "
              f"{d['warp'][0]:7.4f} {d['warp'][1]:7.4f} {d['warp'][2]:7.4f} | "
              f"{d['output'][0]:7.4f} {d['output'][1]:7.4f} {d['output'][2]:7.4f} | "
              f"{w_e[0]:+8.5f} {w_e[1]:+8.5f} {w_e[2]:+8.5f} | "
              f"{o_e[0]:+8.5f} {o_e[1]:+8.5f} {o_e[2]:+8.5f}")

    print(f"\n  mean warp-GT  error: R={np.mean(warp_errors[0]):+.5f} G={np.mean(warp_errors[1]):+.5f} B={np.mean(warp_errors[2]):+.5f}")
    print(f"  mean output-GT error: R={np.mean(out_errors[0]):+.5f} G={np.mean(out_errors[1]):+.5f} B={np.mean(out_errors[2]):+.5f}")
    deblur_correction = [np.mean(out_errors[c]) - np.mean(warp_errors[c]) for c in range(3)]
    print(f"  deblur correction:    R={deblur_correction[0]:+.5f} G={deblur_correction[1]:+.5f} B={deblur_correction[2]:+.5f}")

    # ===== FORWARD =====
    diag.clear()
    print(f"\n{'='*90}")
    print(f"FORWARD cascade (t=0→{KI}) {label}")
    print(f"{'='*90}")
    cascade_fwd, masks_fwd = run_cascade_inference(
        model, cache, KI, prev_kf, grid, scale_vec,
        direction='fwd', use_deblur=True, store_inference=False,
        per_step_fn=_diag_hook)

    print(hdr)
    print("-" * 90)
    warp_errors = [[] for _ in range(3)]
    out_errors = [[] for _ in range(3)]
    for d in sorted(diag, key=lambda x: x['step']):
        t = d['step']
        t_ratio = t / KI
        w_e = [d['warp'][c] - d['gt'][c] for c in range(3)]
        o_e = [d['output'][c] - d['gt'][c] for c in range(3)]
        for c in range(3):
            warp_errors[c].append(w_e[c])
            out_errors[c].append(o_e[c])
        print(f"{t:4d} {t_ratio:6.3f}  "
              f"{d['gt'][0]:7.4f} {d['gt'][1]:7.4f} {d['gt'][2]:7.4f} | "
              f"{d['warp'][0]:7.4f} {d['warp'][1]:7.4f} {d['warp'][2]:7.4f} | "
              f"{d['output'][0]:7.4f} {d['output'][1]:7.4f} {d['output'][2]:7.4f} | "
              f"{w_e[0]:+8.5f} {w_e[1]:+8.5f} {w_e[2]:+8.5f} | "
              f"{o_e[0]:+8.5f} {o_e[1]:+8.5f} {o_e[2]:+8.5f}")

    print(f"\n  mean warp-GT  error: R={np.mean(warp_errors[0]):+.5f} G={np.mean(warp_errors[1]):+.5f} B={np.mean(warp_errors[2]):+.5f}")
    print(f"  mean output-GT error: R={np.mean(out_errors[0]):+.5f} G={np.mean(out_errors[1]):+.5f} B={np.mean(out_errors[2]):+.5f}")
    deblur_correction = [np.mean(out_errors[c]) - np.mean(warp_errors[c]) for c in range(3)]
    print(f"  deblur correction:    R={deblur_correction[0]:+.5f} G={deblur_correction[1]:+.5f} B={deblur_correction[2]:+.5f}")


if __name__ == "__main__":
    main()