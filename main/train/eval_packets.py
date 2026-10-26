"""Standalone inference + evaluation script for the SNN model.

Reads decoded packets (decompress-ing structure) and GT frames,
runs full cascade + fusion inference, computes PSNR / SSIM / LPIPS.

Usage:
    # With metrics:
    python -m main.train.eval_packets \
        --checkpoint experiments/test_v21/phase2/ckpt/ckpt_phase2_best.pth \
        --data-root datasets/data \
        --gt-root datasets/GT \
        --device cuda

    # Single sequence:
    python -m main.train.eval_packets \
        --checkpoint experiments/test_v21/phase2/ckpt/ckpt_phase2_best.pth \
        --data-root datasets/data \
        --gt-root datasets/GT \
        --seq 029 \
        --device cuda

    # Pure inference (output video, no metrics):
    python -m main.train.eval_packets \
        --checkpoint experiments/test_v21/phase2/ckpt/ckpt_phase2_best.pth \
        --data-root datasets/data \
        --keyframe-root datasets/GT \
        --seq 029 \
        --device cuda
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F

from .model import SNNReconstruct
from .data import (
    build_keyframe_pairs,
    count_gt_frames,
    discover_sequences,
    extract_temporal_slices,
    load_gt_frame,
    preload_events,
)
from .cascade import (
    disable_color_diag,
    run_cascade_inference,
    run_fusion_inference,
    StepFeatureCache,
)
from .event_mask import _make_grid_and_scale
from .metrics import compute_psnr, compute_ssim


def _wrap_offset_head(original):
    """Wrap OffsetHead to zero out struct_src channels (last 32 of 297ch input)."""
    class AblatedOffset(torch.nn.Module):
        def forward(self, x, mask, t_ratio=None):
            y = x.clone()
            y[:, -32:, :, :] = 0.0
            return original(y, mask, t_ratio=t_ratio)
    return AblatedOffset().to(next(original.parameters()).device)


def _wrap_deblur_head(original):
    """Wrap DeblurHead to force blurry mask = 1.0 (disable gating)."""
    class AblatedDeblur(torch.nn.Module):
        def forward(self, warped, blurry_mask):
            return original(warped, torch.ones_like(blurry_mask))
    return AblatedDeblur().to(next(original.parameters()).device)


def _wrap_fusion_head(original, fusion_mode, no_delta):
    """Wrap FusionHead for fixed-weight and/or no-delta ablation.

    Input channels: warp_fwd(3)|warp_bwd(3)|prev_kf(3)|next_kf(3)|t_ch(1)|bm_fwd(1)|bm_bwd(1) = 15
    """
    class AblatedFusion(torch.nn.Module):
        def forward(self, x):
            out = original(x)
            if fusion_mode == "uniform":
                out[:, :4] = 0.0
            elif fusion_mode == "t-linear":
                t_map = x[:, 12:13, :, :]
                w_fwd = 1.0 - t_map
                w_bwd = t_map
                neg_inf = -1e9
                eps = 1e-9
                out[:, 0:1] = (w_fwd + eps).log()
                out[:, 1:2] = (w_bwd + eps).log()
                out[:, 2:3] = neg_inf
                out[:, 3:4] = neg_inf
            if fusion_mode is not None:
                out[:, 4:7] = 0.0
            if no_delta:
                out[:, 4:7] = 0.0
            return out
    return AblatedFusion().to(next(original.parameters()).device)


def _ensure_model_compiled(model):
    if getattr(model, "_compiled", False):
        return
    for name in [
        "structure_encoder",
        "deblur_head_bwd",
        "deblur_head_fwd",
        "blurry_predictor_fwd",
        "blurry_predictor_bwd",
        "offset_head_fwd",
        "offset_head_bwd",
    ]:
        if hasattr(model, name):
            setattr(model, name, torch.compile(getattr(model, name)))
    if hasattr(model, "deblur_head_fusion"):
        model.deblur_head_fusion = torch.compile(model.deblur_head_fusion)
    if hasattr(model, "fusion_head"):
        model.fusion_head = torch.compile(model.fusion_head)
    model._compiled = True


def _setup_lpips(device):
    try:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_fn.eval()
        return lpips_fn
    except ImportError:
        print("[WARN] lpips not installed, skipping LPIPS")
        return None


def _load_packet_keyframe(kf_dir: Path, fid: int, target_h: int, target_w: int) -> np.ndarray:
    for pattern in [f"frame_{fid:04d}.png", f"keyframe_{fid:04d}.png", f"{fid:08d}.png"]:
        p = kf_dir / pattern
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                if img_rgb.shape[:2] != (target_h, target_w):
                    img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
                return img_rgb
    raise FileNotFoundError(f"missing keyframe for fid={fid} in {kf_dir}")


def run_snn_forward(vevents, prev_fid, next_fid, event_h, event_w, target_h, target_w, event_downsample, num_slices, model, device):
    event_slices = extract_temporal_slices(vevents, prev_fid, next_fid, event_h, event_w, num_slices, event_downsample)
    events_t = torch.from_numpy(event_slices).float().unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        all_feats, all_membranes, all_offset_feats = model(events_t)
    C = all_feats[0].shape[1]
    H, W = target_h, target_w
    events_up = F.interpolate(events_t.reshape(1, -1, event_h, event_w), size=(H, W), mode="bilinear", align_corners=True)
    grid, scale = _make_grid_and_scale(1, H, W, device)
    return all_feats, all_membranes, all_offset_feats, C, events_up, grid, scale


def evaluate_sequence(
    model, seq_id, vevents, vgt_dir, vnf, device,
    event_h, event_w, target_h, target_w, event_downsample,
    num_slices, key_interval, lpips_fn,
    mask_kernel_size=5, mask_sigma=0.5,
    delta_range=0.15,
    output_video=None,
    compute_metrics=None,
    kf_dir=None,
    fusion_mode=None,
    no_delta=False,
    ablate_struct=False,
    ablate_gate=False,
):
    model.eval()
    disable_color_diag()

    if compute_metrics is None:
        compute_metrics = vgt_dir is not None

    # --- Apply ablation wraps ---
    _saved_modules = {}
    if ablate_struct:
        for key in ["offset_head_fwd", "offset_head_bwd"]:
            if hasattr(model, key):
                _saved_modules[key] = getattr(model, key)
                setattr(model, key, _wrap_offset_head(_saved_modules[key]))
    if ablate_gate:
        for key in ["deblur_head_fwd", "deblur_head_bwd", "deblur_head_fusion"]:
            if hasattr(model, key):
                _saved_modules[key] = getattr(model, key)
                setattr(model, key, _wrap_deblur_head(_saved_modules[key]))
    if fusion_mode is not None or no_delta:
        if hasattr(model, "fusion_head"):
            _saved_modules["fusion_head"] = model.fusion_head
            model.fusion_head = _wrap_fusion_head(_saved_modules["fusion_head"], fusion_mode, no_delta)
    # ---------------------------------

    all_psnr_fusion = []
    all_ssim_fusion = []
    all_lpips_fusion = []
    all_psnr_fwd = []
    all_ssim_fwd = []
    all_lpips_fwd = []
    all_psnr_bwd = []
    all_ssim_bwd = []
    all_lpips_bwd = []
    total_time_ms = 0.0
    total_frames = 0
    last_next_t = None

    video_writer = None
    if output_video is not None:
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video), fourcc, 30, (target_w, target_h))

    max_kf = vnf - key_interval
    if max_kf <= 0:
        if video_writer is not None:
            video_writer.release()
        return {}

    with torch.no_grad():
        for prev_fid in build_keyframe_pairs(max_kf, key_interval):
            next_fid = prev_fid + key_interval

            all_feats, all_membranes, all_offset_feats, C, events_up, grid, scale = run_snn_forward(
                vevents, prev_fid, next_fid, event_h, event_w, target_h, target_w, event_downsample, num_slices, model, device)
            H, W = target_h, target_w
            KI = key_interval

            load_kf = _load_packet_keyframe if kf_dir is not None else load_gt_frame
            kf_src = kf_dir if kf_dir is not None else vgt_dir

            prev_t = torch.from_numpy(load_kf(kf_src, prev_fid, H, W)).float().permute(2, 0, 1).to(device)
            next_t = torch.from_numpy(load_kf(kf_src, next_fid, H, W)).float().permute(2, 0, 1).to(device)
            last_next_t = next_t

            cache = StepFeatureCache(all_feats, all_membranes, all_offset_feats, events_up, H, W, C, model, device, mask_kernel_size=mask_kernel_size, mask_sigma=mask_sigma)

            all_gts = []
            if compute_metrics:
                for target_fid in range(prev_fid + 1, next_fid):
                    target_np = load_gt_frame(vgt_dir, target_fid, H, W)
                    all_gts.append(torch.from_numpy(target_np).float().permute(2, 0, 1).to(device))

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            s_bwd = torch.cuda.Stream()
            s_fwd = torch.cuda.Stream()

            with torch.cuda.stream(s_bwd):
                cascade_bwd, masks_bwd = run_cascade_inference(
                    model, cache, KI, next_t, grid, scale,
                    direction="bwd", use_deblur=True, store_inference=False)

            with torch.cuda.stream(s_fwd):
                cascade_fwd, masks_fwd = run_cascade_inference(
                    model, cache, KI, prev_t, grid, scale,
                    direction="fwd", use_deblur=True, store_inference=False)

            s_bwd.synchronize()
            s_fwd.synchronize()

            fusion_outputs, _, _ = run_fusion_inference(model, cascade_fwd, cascade_bwd, prev_t, next_t, KI, delta_range,
                                                        blurry_masks_fwd=masks_fwd, blurry_masks_bwd=masks_bwd)

            torch.cuda.synchronize()
            total_time_ms += (time.perf_counter() - t_start) * 1000
            total_frames += KI - 1

            if video_writer is not None:
                kf_np = (prev_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                kf_bgr = cv2.cvtColor(kf_np, cv2.COLOR_RGB2BGR)
                video_writer.write(kf_bgr)

            for t in range(1, KI):
                fusion_out = fusion_outputs[t - 1].squeeze(0)

                if video_writer is not None:
                    fusion_np = (fusion_out.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    fusion_bgr = cv2.cvtColor(fusion_np, cv2.COLOR_RGB2BGR)
                    video_writer.write(fusion_bgr)

                if compute_metrics:
                    target_t = all_gts[t - 1]
                    psnr_val = compute_psnr(fusion_out, target_t)
                    _, ssim_val = compute_ssim(fusion_out, target_t)
                    all_psnr_fusion.append(psnr_val)
                    all_ssim_fusion.append(ssim_val.item())
                    if lpips_fn is not None:
                        img_in = fusion_out * 2.0 - 1.0
                        tgt_in = target_t * 2.0 - 1.0
                        all_lpips_fusion.append(lpips_fn(img_in.unsqueeze(0), tgt_in.unsqueeze(0)).item())

                    fwd_out = cascade_fwd[t].squeeze(0)
                    psnr_val = compute_psnr(fwd_out, target_t)
                    _, ssim_val = compute_ssim(fwd_out, target_t)
                    all_psnr_fwd.append(psnr_val)
                    all_ssim_fwd.append(ssim_val.item())
                    if lpips_fn is not None:
                        img_in = fwd_out * 2.0 - 1.0
                        tgt_in = target_t * 2.0 - 1.0
                        all_lpips_fwd.append(lpips_fn(img_in.unsqueeze(0), tgt_in.unsqueeze(0)).item())

                    bwd_out = cascade_bwd[t].squeeze(0)
                    psnr_val = compute_psnr(bwd_out, target_t)
                    _, ssim_val = compute_ssim(bwd_out, target_t)
                    all_psnr_bwd.append(psnr_val)
                    all_ssim_bwd.append(ssim_val.item())
                    if lpips_fn is not None:
                        img_in = bwd_out * 2.0 - 1.0
                        tgt_in = target_t * 2.0 - 1.0
                        all_lpips_bwd.append(lpips_fn(img_in.unsqueeze(0), tgt_in.unsqueeze(0)).item())

    if video_writer is not None:
        if last_next_t is not None:
            kf_np = (last_next_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            kf_bgr = cv2.cvtColor(kf_np, cv2.COLOR_RGB2BGR)
            video_writer.write(kf_bgr)
        video_writer.release()
        print(f"  Video saved to {output_video}")

    # --- Restore original modules ---
    for key, mod in _saved_modules.items():
        setattr(model, key, mod)
    # ---------------------------------

    per_frame_ms = total_time_ms / max(total_frames, 1)

    return {
        "seq_id": seq_id,
        "psnr_fusion": float(np.mean(all_psnr_fusion)) if all_psnr_fusion else 0.0,
        "ssim_fusion": float(np.mean(all_ssim_fusion)) if all_ssim_fusion else 0.0,
        "lpips_fusion": float(np.mean(all_lpips_fusion)) if all_lpips_fusion else 0.0,
        "psnr_fwd": float(np.mean(all_psnr_fwd)) if all_psnr_fwd else 0.0,
        "ssim_fwd": float(np.mean(all_ssim_fwd)) if all_ssim_fwd else 0.0,
        "lpips_fwd": float(np.mean(all_lpips_fwd)) if all_lpips_fwd else 0.0,
        "psnr_bwd": float(np.mean(all_psnr_bwd)) if all_psnr_bwd else 0.0,
        "ssim_bwd": float(np.mean(all_ssim_bwd)) if all_ssim_bwd else 0.0,
        "lpips_bwd": float(np.mean(all_lpips_bwd)) if all_lpips_bwd else 0.0,
        "per_frame_ms": per_frame_ms,
        "total_frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="SNN Standalone Packet Evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory of decoded packets (e.g. datasets/data)")
    parser.add_argument("--gt-root", type=Path, default=None, help="Root directory of GT frames (e.g. datasets/GT). If omitted, skip metrics and output video.")
    parser.add_argument("--keyframe-root", type=Path, default=None, help="Root directory of keyframes (defaults to --gt-root). Required for cascade inference.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save reconstructed videos (default: output/<checkpoint_name>/)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seq", type=str, default=None, help="Evaluate a single sequence (e.g. 029); omit for all val sequences")
    parser.add_argument("--split", type=str, default="val", help="Data split: train or val")
    parser.add_argument("--key-interval", type=int, default=9)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mask-kernel-size", type=int, default=5)
    parser.add_argument("--mask-sigma", type=float, default=0.5)
    parser.add_argument("--delta-range", type=float, default=0.15)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--fusion-mode", type=str, default=None, choices=["uniform", "t-linear"],
                        help="Ablation: override fusion weights (uniform=0.25 each, t-linear=based on time ratio)")
    parser.add_argument("--no-delta", action="store_true",
                        help="Ablation: disable delta correction in fusion")
    parser.add_argument("--ablate-struct", action="store_true",
                        help="Ablation: zero out StructureEncoder features (last 32ch of offset_head input)")
    parser.add_argument("--ablate-gate", action="store_true",
                        help="Ablation: force blurry masks to 1.0 (disable multiplicative gating)")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if not use_amp:
        print("[WARN] BF16 not supported, running in FP32")

    seqs = discover_sequences(args.data_root, args.split)
    if not seqs:
        print(f"[ERROR] no sequences found in {args.data_root / args.split}")
        return
    print(f"Found {len(seqs)} sequences in {args.split}")

    if args.seq is not None:
        seqs = [s for s in seqs if s["seq_id"] == args.seq]
        if not seqs:
            print(f"[ERROR] sequence {args.seq} not found")
            return

    all_events = preload_events(seqs)

    compute_metrics = args.gt_root is not None

    if compute_metrics:
        vnf = count_gt_frames(args.gt_root / args.split, seqs)
    else:
        vnf = {}
        for seq in seqs:
            events = all_events[seq["seq_id"]]
            frame_offsets = events.get("frame_offsets", [])
            vnf[seq["seq_id"]] = len(frame_offsets) - 1 if len(frame_offsets) > 1 else 0
        print("[INFO] --gt-root not provided, skipping metrics, outputting video")

    first_seq = seqs[0]
    first_events = all_events[first_seq["seq_id"]]
    x_arr = first_events.get("x", [])
    y_arr = first_events.get("y", [])
    event_h = int(np.max(y_arr)) + 1 if len(y_arr) > 0 else 360
    event_w = int(np.max(x_arr)) + 1 if len(x_arr) > 0 else 640
    event_downsample = int(first_events.get("event_support", 1) or 1)

    if compute_metrics:
        sample_gt_dir = args.gt_root / args.split / first_seq["gt_dir_name"]
        sample_p = next(sample_gt_dir.glob("*.png"))
        sample_img = cv2.imread(str(sample_p))
        target_h, target_w = sample_img.shape[:2]
    else:
        # Get resolution from packet directory's keyframes/ subdirectory
        sample_kf = first_seq["packet_dir"] / "keyframes"
        if not sample_kf.is_dir():
            print(f"[ERROR] keyframes directory not found: {sample_kf}")
            print("[ERROR] Keyframes are required for cascade inference.")
            print("[ERROR] Provide --gt-root or --keyframe-root pointing to GT frames.")
            return
        # Try multiple naming patterns
        pngs = list(sample_kf.glob("frame_*.png")) or list(sample_kf.glob("keyframe_*.png")) or list(sample_kf.glob("*.png"))
        if not pngs:
            print(f"[ERROR] No keyframe PNGs found in {sample_kf}")
            print("[ERROR] Provide --gt-root or --keyframe-root pointing to GT frames.")
            return
        sample_img = cv2.imread(str(pngs[0]))
        target_h, target_w = sample_img.shape[:2]

    num_slices = args.key_interval
    print(f"event={event_w}x{event_h}, target={target_w}x{target_h}, event_downsample={event_downsample}, num_slices={num_slices}")

    model = SNNReconstruct(T=num_slices, hidden=args.hidden, phase=2)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = model.state_dict()

    new_ckpt = {}
    for key, value in ckpt["model"].items():
        clean_key = key.replace("_orig_mod.", "")
        if clean_key in model_state and value.shape != model_state[clean_key].shape:
            if value.dim() >= 2 and value.shape[0] > model_state[clean_key].shape[0]:
                value = value[: model_state[clean_key].shape[0], ...]
            elif value.dim() >= 2 and value.shape[1] > model_state[clean_key].shape[1]:
                value = value[:, : model_state[clean_key].shape[1], ...]
        new_ckpt[clean_key] = value
    ckpt["model"] = new_ckpt

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}):")
        for k in missing:
            print(f"  - {k}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print(f"  - {k}")
    model.to(device)
    model.eval()
    print(f"Model loaded from {args.checkpoint}, params={sum(p.numel() for p in model.parameters()):,}")

    if args.no_compile:
        print("[INFO] torch.compile disabled")
        torch._dynamo.reset()
    else:
        _ensure_model_compiled(model)

    lpips_fn = _setup_lpips(device) if compute_metrics else None

    output_base = None
    if not compute_metrics:
        if args.output_dir is not None:
            output_base = args.output_dir
        else:
            ckpt_stem = args.checkpoint.stem
            output_base = Path("output") / ckpt_stem
        output_base.mkdir(parents=True, exist_ok=True)

    if compute_metrics:
        print(f"\n{'='*60}")
        ablate_info = []
        if args.fusion_mode:
            ablate_info.append(f"fusion={args.fusion_mode}")
        if args.no_delta:
            ablate_info.append("no-delta")
        if args.ablate_struct:
            ablate_info.append("no-struct")
        if args.ablate_gate:
            ablate_info.append("no-gate")
        tag = f"  [ABLATION: {', '.join(ablate_info)}]" if ablate_info else ""
        print(f"{'Seq':>6s}  {'Frames':>6s}  {'Fusion PSNR':>12s}  {'Fusion SSIM':>11s}  {'Fusion LPIPS':>13s}  {'FWD PSNR':>10s}  {'BWD PSNR':>10s}  {'ms/frame':>9s}{tag}")
        print(f"{'='*60}")

    all_results = []
    for seq in seqs:
        seq_id = seq["seq_id"]
        vevents = all_events[seq_id]
        vgt_dir = args.gt_root / args.split / seq["gt_dir_name"] if compute_metrics else None
        kf_dir = seq["packet_dir"] / "keyframes" if not compute_metrics else None

        output_video = None
        if not compute_metrics:
            output_video = output_base / f"{seq_id}.mp4"

        result = evaluate_sequence(
            model, seq_id, vevents, vgt_dir, vnf[seq_id], device,
            event_h, event_w, target_h, target_w, event_downsample,
            num_slices, args.key_interval, lpips_fn,
            mask_kernel_size=args.mask_kernel_size,
            mask_sigma=args.mask_sigma,
            delta_range=args.delta_range,
            output_video=output_video,
            compute_metrics=compute_metrics,
            kf_dir=kf_dir,
            fusion_mode=args.fusion_mode,
            no_delta=args.no_delta,
            ablate_struct=args.ablate_struct,
            ablate_gate=args.ablate_gate,
        )
        if not result:
            continue
        all_results.append(result)

        if compute_metrics:
            print(
                f"{seq_id:>6s}  {result['total_frames']:>6d}  "
                f"{result['psnr_fusion']:>12.4f}  {result['ssim_fusion']:>11.4f}  "
                f"{result['lpips_fusion']:>13.4f}  {result['psnr_fwd']:>10.4f}  "
                f"{result['psnr_bwd']:>10.4f}  {result['per_frame_ms']:>9.1f}"
            )

    if len(all_results) > 1 and compute_metrics:
        avg_psnr_fusion = float(np.mean([r["psnr_fusion"] for r in all_results]))
        avg_ssim_fusion = float(np.mean([r["ssim_fusion"] for r in all_results]))
        avg_lpips_fusion = float(np.mean([r["lpips_fusion"] for r in all_results]))
        avg_psnr_fwd = float(np.mean([r["psnr_fwd"] for r in all_results]))
        avg_psnr_bwd = float(np.mean([r["psnr_bwd"] for r in all_results]))
        avg_ssim_fwd = float(np.mean([r["ssim_fwd"] for r in all_results if r["ssim_fwd"] != 0])) if any(r["ssim_fwd"] != 0 for r in all_results) else 0.0
        avg_lpips_fwd = float(np.mean([r["lpips_fwd"] for r in all_results if r["lpips_fwd"] != 0])) if any(r["lpips_fwd"] != 0 for r in all_results) else 0.0
        avg_ssim_bwd = float(np.mean([r["ssim_bwd"] for r in all_results if r["ssim_bwd"] != 0])) if any(r["ssim_bwd"] != 0 for r in all_results) else 0.0
        avg_lpips_bwd = float(np.mean([r["lpips_bwd"] for r in all_results if r["lpips_bwd"] != 0])) if any(r["lpips_bwd"] != 0 for r in all_results) else 0.0
        avg_ms = float(np.mean([r["per_frame_ms"] for r in all_results]))
        total_f = sum(r["total_frames"] for r in all_results)
        print(f"{'='*60}")
        print(
            f"{'AVG':>6s}  {total_f:>6d}  "
            f"{avg_psnr_fusion:>12.4f}  {avg_ssim_fusion:>11.4f}  "
            f"{avg_lpips_fusion:>13.4f}  {avg_psnr_fwd:>10.4f}  "
            f"{avg_psnr_bwd:>10.4f}  {avg_ms:>9.1f}"
        )

    # --- Write CSV ---
    if compute_metrics and all_results:
        suffix_parts = []
        if args.fusion_mode:
            suffix_parts.append(args.fusion_mode)
        if args.no_delta:
            suffix_parts.append("nod")
        if args.ablate_struct:
            suffix_parts.append("ns")
        if args.ablate_gate:
            suffix_parts.append("ng")
        csv_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
        ckpt_stem = args.checkpoint.stem
        log_dir = Path("compare_DATA")
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / f"{ckpt_stem}{csv_suffix}.csv"

        import csv as _csv
        headers = ["seq", "frames", "psnr_fusion", "ssim_fusion", "lpips_fusion",
                    "psnr_fwd", "ssim_fwd", "lpips_fwd", "psnr_bwd", "ssim_bwd",
                    "lpips_bwd", "ms_per_frame"]
        rows = []
        for r in all_results:
            rows.append([
                r.get("seq_id", ""), r["total_frames"],
                f"{r['psnr_fusion']:.4f}", f"{r['ssim_fusion']:.4f}", f"{r['lpips_fusion']:.4f}",
                f"{r['psnr_fwd']:.4f}", f"{r['ssim_fwd']:.4f}", f"{r['lpips_fwd']:.4f}",
                f"{r['psnr_bwd']:.4f}", f"{r['ssim_bwd']:.4f}", f"{r['lpips_bwd']:.4f}",
                f"{r['per_frame_ms']:.1f}",
            ])
        if len(all_results) > 1:
            rows.append([
                "AVG", total_f,
                f"{avg_psnr_fusion:.4f}", f"{avg_ssim_fusion:.4f}", f"{avg_lpips_fusion:.4f}",
                f"{avg_psnr_fwd:.4f}", f"{avg_ssim_fwd:.4f}", f"{avg_lpips_fwd:.4f}",
                f"{avg_psnr_bwd:.4f}", f"{avg_ssim_bwd:.4f}", f"{avg_lpips_bwd:.4f}",
                f"{avg_ms:.1f}",
            ])
        rows.append([])
        rows.append(["config", f"checkpoint={args.checkpoint}",
                     f"fusion_mode={args.fusion_mode or 'learned'}",
                     f"no_delta={args.no_delta}",
                     f"ablate_struct={args.ablate_struct}",
                     f"ablate_gate={args.ablate_gate}",
                     f"split={args.split}"])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        print(f"\n[CSV] saved to {csv_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()