from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import SNNReconstruct
from .data import extract_temporal_slices, load_gt_frame, build_keyframe_pairs
from .cascade import _prepare_step_features, _prepare_offset_features, \
    run_phase1_warp, run_cascade_inference, run_fusion_inference, run_fusion_step, \
    StepFeatureCache, disable_color_diag
from .metrics import compute_psnr, compute_ssim
from .event_mask import make_event_features_fwd, make_event_features_bwd, _make_grid_and_scale


def _ensure_model_compiled(model):
    if getattr(model, '_compiled', False):
        return
    for name in ['structure_encoder', 'deblur_head_bwd', 'deblur_head_fwd',
                 'blurry_predictor_fwd', 'blurry_predictor_bwd',
                 'offset_head_fwd', 'offset_head_bwd']:
        if hasattr(model, name):
            setattr(model, name, torch.compile(getattr(model, name)))
    if hasattr(model, 'deblur_head_fusion'):
        model.deblur_head_fusion = torch.compile(model.deblur_head_fusion)
    if hasattr(model, 'fusion_head'):
        model.fusion_head = torch.compile(model.fusion_head)
    model._compiled = True


def _compute_eval_metrics(pred, target, mask, lpips_fn=None):
    """Compute PSNR (total/static/dynamic), SSIM (total/static/dynamic), and LPIPS for a prediction vs target."""
    with torch.amp.autocast('cuda', enabled=False):
        ssim_map, ssim_val = compute_ssim(pred.float(), target.float())
    C = pred.shape[0]
    metrics = {'psnr': compute_psnr(pred, target),
               'ssim': ssim_val.item()}

    if mask is None:
        metrics['psnr_static'] = 0.0
        metrics['psnr_dynamic'] = 0.0
        metrics['ssim_static'] = 0.0
        metrics['ssim_dynamic'] = 0.0
    else:
        mask_np = mask.squeeze(0).squeeze(0).float()
        n_static = (1.0 - mask_np).sum().item()
        n_dynamic = mask_np.sum().item()
        mask_3d = mask_np.unsqueeze(0)
        mask_4d = mask_np.unsqueeze(0).unsqueeze(0)
        if n_static > 0:
            mse_static = ((pred - target) ** 2 * (1.0 - mask_3d)).sum() / (n_static * 3 + 1e-8)
            metrics['psnr_static'] = 10.0 * math.log10(1.0 / max(mse_static.item(), 1e-10))
            metrics['ssim_static'] = (ssim_map * (1.0 - mask_4d)).sum().item() / (n_static * C + 1e-8)
        else:
            metrics['psnr_static'] = 0.0
            metrics['ssim_static'] = 0.0
        if n_dynamic > 0:
            mse_dynamic = ((pred - target) ** 2 * mask_3d).sum() / (n_dynamic * 3 + 1e-8)
            metrics['psnr_dynamic'] = 10.0 * math.log10(1.0 / max(mse_dynamic.item(), 1e-10))
            metrics['ssim_dynamic'] = (ssim_map * mask_4d).sum().item() / (n_dynamic * C + 1e-8)
        else:
            metrics['psnr_dynamic'] = 0.0
            metrics['ssim_dynamic'] = 0.0

    if lpips_fn is not None:
        img_in = pred * 2.0 - 1.0
        target_in = target * 2.0 - 1.0
        metrics['lpips'] = lpips_fn(img_in.unsqueeze(0), target_in.unsqueeze(0)).item()

    return metrics


def _compute_step_metrics(pred, target_t, event_mask, gt_warped, warp_est, lpips_fn):
    """Unified per-step metrics for forward and backward directions.

    Args:
        pred: prediction to compare with target (Phase 1: gt_warped, Phase 2+: cascade output)
        target_t: ground truth frame at time t
        event_mask: event mask for static/dynamic PSNR split
        gt_warped: GT frame warped by predicted offset
        warp_est: raw warp estimation (before clamp)
        lpips_fn: LPIPS function or None
    Returns:
        dict: psnr, psnr_static, psnr_dynamic, lpips(optional), warp_l1, cascade_gap
    """
    result = {}
    m = _compute_eval_metrics(pred, target_t, event_mask, lpips_fn)
    result['psnr'] = m['psnr']
    result['psnr_static'] = m['psnr_static']
    result['psnr_dynamic'] = m['psnr_dynamic']
    result['ssim'] = m['ssim']
    result['ssim_static'] = m['ssim_static']
    result['ssim_dynamic'] = m['ssim_dynamic']
    if lpips_fn is not None:
        result['lpips'] = m['lpips']
    result['warp_l1'] = F.l1_loss(gt_warped, target_t).item()
    result['cascade_gap'] = (warp_est - gt_warped).abs().mean().item()
    return result


def _run_snn_forward(vevents, prev_fid, next_fid, event_h, event_w, target_h, target_w,
                     event_downsample, num_slices, model, device):
    event_slices = extract_temporal_slices(
        vevents, prev_fid, next_fid, event_h, event_w, num_slices, event_downsample)
    events_t = torch.from_numpy(event_slices).float().unsqueeze(0).to(device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        all_feats, all_membranes, all_offset_feats = model(events_t)
    C = all_feats[0].shape[1]
    H, W = target_h, target_w
    events_up = F.interpolate(events_t.reshape(1, -1, event_h, event_w),
                              size=(H, W), mode="bilinear", align_corners=True)
    grid, scale = _make_grid_and_scale(1, H, W, device)
    return all_feats, all_membranes, all_offset_feats, C, events_up, grid, scale


def run_validation(model: SNNReconstruct, val_seqs, val_events, val_frames, val_ranges,
                   gt_root, device, event_h, event_w, target_h, target_w,
                   event_downsample, num_slices, key_interval,
                   lpips_fn, phase=1, mask_kernel_size=5, mask_sigma=0.5,
                   delta_scale=10.0, delta_range=0.15, use_deblur=True):
    model.eval()
    disable_color_diag()

    val_psnr_total = []
    val_psnr_static = []
    val_psnr_dynamic = []
    val_ssim_fwd = []
    val_ssim_fwd_static = []
    val_ssim_fwd_dynamic = []
    val_psnr_bwd = []
    val_psnr_bwd_static = []
    val_psnr_bwd_dynamic = []
    val_ssim_bwd = []
    val_ssim_bwd_static = []
    val_ssim_bwd_dynamic = []
    val_psnr_fusion = []
    val_psnr_fusion_static = []
    val_psnr_fusion_dynamic = []
    val_ssim_fusion = []
    val_ssim_fusion_static = []
    val_ssim_fusion_dynamic = []
    val_fusion_l1 = []
    val_lpips = []
    val_lpips_bwd = []
    val_lpips_fusion = []
    val_w0 = []
    val_w1 = []
    val_w2 = []
    val_w3 = []
    val_delta_mean = []
    val_gt_warp_l1 = []
    val_gt_warp_l1_bwd = []
    val_cascade_gap = []
    val_cascade_gap_bwd = []

    total_time_ms = 0.0
    total_frames = 0
    total_enc_ms = 0.0
    total_bwd_ms = 0.0
    total_fwd_ms = 0.0

    with torch.no_grad():
        _ensure_model_compiled(model)

        if not getattr(model, '_warmed_up', False):
            _dummy = torch.randn(1, num_slices * 2, event_h, event_w, device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(_dummy)
            model._warmed_up = True
            torch.cuda.synchronize()

        for vs in val_seqs:
            vsid = vs["seq_id"]
            if vsid not in val_ranges:
                continue
            vevents = val_events[vsid]
            vgt_dir = gt_root / "val" / vsid
            vnf = val_frames[vsid]

            for prev_fid in val_ranges[vsid]:
                next_fid = prev_fid + key_interval

                prev_kf_np = load_gt_frame(vgt_dir, prev_fid, target_h, target_w)
                next_kf_np = load_gt_frame(vgt_dir, next_fid, target_h, target_w)
                prev_t = torch.from_numpy(prev_kf_np).float().permute(2, 0, 1).to(device)
                next_t = torch.from_numpy(next_kf_np).float().permute(2, 0, 1).to(device)

                event_slices = extract_temporal_slices(
                    vevents, prev_fid, next_fid, event_h, event_w,
                    num_slices, event_downsample,
                )
                events_t = torch.from_numpy(event_slices).float().unsqueeze(0).to(device)

                torch.cuda.synchronize()
                t_enc = time.perf_counter()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    all_feats, all_membranes, all_offset_feats = model(events_t)
                torch.cuda.synchronize()
                enc_ms = (time.perf_counter() - t_enc) * 1000
                total_enc_ms += enc_ms
                C = all_feats[0].shape[1]

                prev_frame = prev_t.clone()
                prev_gt = prev_t.clone()

                _, H, W = prev_frame.shape
                events_up = F.interpolate(events_t.reshape(1, -1, event_h, event_w),
                                          size=(H, W), mode="bilinear", align_corners=True)

                struct_prev = model.structure_encoder(prev_t.unsqueeze(0))
                struct_next = model.structure_encoder(next_t.unsqueeze(0))
                struct_cascade = struct_prev

                B = 1
                grid, scale = _make_grid_and_scale(B, H, W, device)

                cache = StepFeatureCache(all_feats, all_membranes, all_offset_feats, events_up, H, W, C, model, device,
                          mask_kernel_size=mask_kernel_size, mask_sigma=mask_sigma)

                KI = key_interval

                all_gts = []
                for target_fid in range(prev_fid + 1, next_fid):
                    target_np = load_gt_frame(vgt_dir, target_fid, target_h, target_w)
                    all_gts.append(torch.from_numpy(target_np).float().permute(2, 0, 1).to(device))

                def _record_bwd_metrics(t, output, event_mask, sample_grid, warp_est):
                    target_t = all_gts[t - 1]
                    src_gt = all_gts[t] if t < KI - 1 else next_t
                    gt_warped = F.grid_sample(
                        src_gt.unsqueeze(0), sample_grid,
                        align_corners=True, padding_mode='border').squeeze(0)
                    m = _compute_step_metrics(
                        output.squeeze(0), target_t, event_mask, gt_warped, warp_est.squeeze(0), lpips_fn)
                    val_psnr_bwd.append(m['psnr'])
                    val_psnr_bwd_static.append(m['psnr_static'])
                    val_psnr_bwd_dynamic.append(m['psnr_dynamic'])
                    val_ssim_bwd.append(m['ssim'])
                    val_ssim_bwd_static.append(m['ssim_static'])
                    val_ssim_bwd_dynamic.append(m['ssim_dynamic'])
                    if lpips_fn is not None:
                        val_lpips_bwd.append(m['lpips'])
                    val_gt_warp_l1_bwd.append(m['warp_l1'])
                    val_cascade_gap_bwd.append(m['cascade_gap'])

                def _record_fwd_metrics(t, output, event_mask, sample_grid, warp_est):
                    target_t = all_gts[t - 1]
                    src_gt = all_gts[t - 2] if t > 1 else prev_t
                    gt_warped = F.grid_sample(
                        src_gt.unsqueeze(0), sample_grid,
                        align_corners=True, padding_mode='border').squeeze(0)
                    m = _compute_step_metrics(
                        output.squeeze(0), target_t, event_mask, gt_warped, warp_est.squeeze(0), lpips_fn)
                    val_psnr_total.append(m['psnr'])
                    val_psnr_static.append(m['psnr_static'])
                    val_psnr_dynamic.append(m['psnr_dynamic'])
                    val_ssim_fwd.append(m['ssim'])
                    val_ssim_fwd_static.append(m['ssim_static'])
                    val_ssim_fwd_dynamic.append(m['ssim_dynamic'])
                    if lpips_fn is not None:
                        val_lpips.append(m['lpips'])
                    val_gt_warp_l1.append(m['warp_l1'])
                    val_cascade_gap.append(m['cascade_gap'])

                if phase == 1:
                    torch.cuda.synchronize()
                    t_bwd = time.perf_counter()

                    _bwd_refs = [None] + [all_gts[i] for i in range(1, KI - 1)] + [next_t]
                    for bt in range(KI - 1, 0, -1):
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            idx = bt - 1
                            features = cache.get(idx)
                            _, _, _, _, _, _, pol_feat_bwd, _, event_mask_bwd = features
                            off_t_bwd, _, off_tp1_bwd = cache.get_offset(bt)
                            event_pos = events_up[:, idx * 2 + 2:idx * 2 + 3]
                            event_neg = events_up[:, idx * 2 + 3:idx * 2 + 4]
                            feat_t, _ = _prepare_step_features(
                                cache._feats, cache._membs, bt, H, W, C)

                            src_gt = _bwd_refs[bt].unsqueeze(0)
                            struct_src = model.structure_encoder(src_gt)

                            result = run_phase1_warp(
                                model, src_gt, struct_src,
                                off_t_bwd, off_tp1_bwd if bt < KI - 1 else off_t_bwd, pol_feat_bwd,
                                grid, scale, 'bwd',
                                event_mask_bwd, event_pos, event_neg, feat_t,
                                use_deblur=use_deblur)

                        _record_bwd_metrics(bt, result['output'], event_mask_bwd,
                                            result['sample_grid'], result['warp_raw'])

                    torch.cuda.synchronize()
                    bwd_ms = (time.perf_counter() - t_bwd) * 1000
                    total_bwd_ms += bwd_ms
                    total_time_ms += bwd_ms

                    torch.cuda.synchronize()
                    t_fwd = time.perf_counter()

                    _fwd_refs = [None, prev_t] + [all_gts[i] for i in range(0, KI - 2)]
                    for t in range(1, KI):
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            idx = t - 1
                            features = cache.get(idx)
                            _, _, _, _, _, pol_feat_fwd, _, event_mask_fwd, _ = features
                            off_t, off_tm1, off_tp1 = cache.get_offset(idx)
                            event_pos = events_up[:, idx * 2:idx * 2 + 1]
                            event_neg = events_up[:, idx * 2 + 1:idx * 2 + 2]
                            feat_t, _ = _prepare_step_features(
                                cache._feats, cache._membs, idx, H, W, C)

                            src_gt = _fwd_refs[t].unsqueeze(0)
                            struct_src = model.structure_encoder(src_gt)

                            result = run_phase1_warp(
                                model, src_gt, struct_src,
                                off_tm1 if t > 1 else off_t, off_t, pol_feat_fwd,
                                grid, scale, 'fwd',
                                event_mask_fwd, event_pos, event_neg, feat_t,
                                use_deblur=use_deblur)

                        _record_fwd_metrics(t, result['output'], event_mask_fwd,
                                            result['sample_grid'], result['warp_raw'])

                        val_w0.append(0.0)
                        val_w1.append(0.0)
                        val_w2.append(0.0)
                        val_w3.append(0.0)
                        val_delta_mean.append(0.0)

                    torch.cuda.synchronize()
                    fwd_ms = (time.perf_counter() - t_fwd) * 1000
                    total_fwd_ms += fwd_ms
                    total_time_ms += fwd_ms
                    total_frames += (KI - 1)
                else:
                    torch.cuda.synchronize()
                    t_start = time.perf_counter()

                    def _bwd_cb(bt, _idx, output, event_mask, sample_grid, warp_est):
                        _record_bwd_metrics(bt, output, event_mask, sample_grid, warp_est)

                    def _fwd_cb(ft, _idx, output, event_mask, sample_grid, warp_est):
                        _record_fwd_metrics(ft, output, event_mask, sample_grid, warp_est)

                    s_bwd = torch.cuda.Stream()
                    s_fwd = torch.cuda.Stream()

                    with torch.cuda.stream(s_bwd):
                        cascade_bwd, masks_bwd = run_cascade_inference(
                            model, cache, KI, next_t, grid, scale,
                            direction='bwd', per_step_fn=_bwd_cb, use_deblur=True,
                            store_inference=False)

                    with torch.cuda.stream(s_fwd):
                        cascade_fwd, masks_fwd = run_cascade_inference(
                            model, cache, KI, prev_t, grid, scale,
                            direction='fwd', per_step_fn=_fwd_cb, use_deblur=True,
                            store_inference=False)

                    s_bwd.synchronize()
                    s_fwd.synchronize()
                    cascade_ms = (time.perf_counter() - t_start) * 1000
                    total_bwd_ms += cascade_ms * 0.5
                    total_fwd_ms += cascade_ms * 0.5
                    total_time_ms += cascade_ms

                    torch.cuda.synchronize()
                    t_fusion = time.perf_counter()

                    fusion_outputs, fusion_weights, fusion_deltas = run_fusion_inference(
                        model, cascade_fwd, cascade_bwd, prev_t, next_t, KI, delta_range,
                       blurry_masks_fwd=masks_fwd, blurry_masks_bwd=masks_bwd)

                    for i, fusion_out in enumerate(fusion_outputs):
                        t = i + 1
                        idx = t - 1
                        _, _, _, _, _, _, _, event_mask_fwd, event_mask_bwd = cache.get(idx)
                        event_mask = torch.max(event_mask_fwd, event_mask_bwd)
                        target_t = all_gts[t - 1]
                        fusion_m = _compute_eval_metrics(
                            fusion_out.squeeze(0), target_t, event_mask, lpips_fn)
                        val_psnr_fusion.append(fusion_m['psnr'])
                        val_psnr_fusion_static.append(fusion_m['psnr_static'])
                        val_psnr_fusion_dynamic.append(fusion_m['psnr_dynamic'])
                        val_ssim_fusion.append(fusion_m['ssim'])
                        val_ssim_fusion_static.append(fusion_m['ssim_static'])
                        val_ssim_fusion_dynamic.append(fusion_m['ssim_dynamic'])
                        if lpips_fn is not None:
                            val_lpips_fusion.append(fusion_m['lpips'])
                        val_fusion_l1.append(F.l1_loss(fusion_out, target_t.unsqueeze(0)).item())

                        w = fusion_weights[i]
                        val_w0.append(w[:, 0].mean().item())
                        val_w1.append(w[:, 1].mean().item())
                        val_w2.append(w[:, 2].mean().item())
                        val_w3.append(w[:, 3].mean().item())
                        val_delta_mean.append(fusion_deltas[i].abs().mean().item())

                    torch.cuda.synchronize()
                    fusion_ms = (time.perf_counter() - t_fusion) * 1000
                    total_time_ms += fusion_ms
                    total_frames += (KI - 1)

                total_time_ms += enc_ms

    psnr_total = np.mean(val_psnr_total) if val_psnr_total else 0
    psnr_static = np.mean(val_psnr_static) if val_psnr_static else 0
    psnr_dynamic = np.mean(val_psnr_dynamic) if val_psnr_dynamic else 0
    ssim_fwd = np.mean(val_ssim_fwd) if val_ssim_fwd else 0
    ssim_fwd_static = np.mean(val_ssim_fwd_static) if val_ssim_fwd_static else 0
    ssim_fwd_dynamic = np.mean(val_ssim_fwd_dynamic) if val_ssim_fwd_dynamic else 0
    psnr_bwd = np.mean(val_psnr_bwd) if val_psnr_bwd else 0
    psnr_bwd_static = np.mean(val_psnr_bwd_static) if val_psnr_bwd_static else 0
    psnr_bwd_dynamic = np.mean(val_psnr_bwd_dynamic) if val_psnr_bwd_dynamic else 0
    ssim_bwd = np.mean(val_ssim_bwd) if val_ssim_bwd else 0
    ssim_bwd_static = np.mean(val_ssim_bwd_static) if val_ssim_bwd_static else 0
    ssim_bwd_dynamic = np.mean(val_ssim_bwd_dynamic) if val_ssim_bwd_dynamic else 0
    psnr_fusion = np.mean(val_psnr_fusion) if val_psnr_fusion else 0
    psnr_fusion_static = np.mean(val_psnr_fusion_static) if val_psnr_fusion_static else 0
    psnr_fusion_dynamic = np.mean(val_psnr_fusion_dynamic) if val_psnr_fusion_dynamic else 0
    ssim_fusion = np.mean(val_ssim_fusion) if val_ssim_fusion else 0
    ssim_fusion_static = np.mean(val_ssim_fusion_static) if val_ssim_fusion_static else 0
    ssim_fusion_dynamic = np.mean(val_ssim_fusion_dynamic) if val_ssim_fusion_dynamic else 0
    fusion_l1 = np.mean(val_fusion_l1) if val_fusion_l1 else 0
    lpips_avg = np.mean(val_lpips) if val_lpips else 0
    lpips_bwd_avg = np.mean(val_lpips_bwd) if val_lpips_bwd else 0
    lpips_fusion_avg = np.mean(val_lpips_fusion) if val_lpips_fusion else 0

    w0 = np.mean(val_w0) if val_w0 else 0
    w1 = np.mean(val_w1) if val_w1 else 0
    w2 = np.mean(val_w2) if val_w2 else 0
    w3 = np.mean(val_w3) if val_w3 else 0
    delta_m = np.mean(val_delta_mean) if val_delta_mean else 0
    gt_warp_l1 = np.mean(val_gt_warp_l1) if val_gt_warp_l1 else 0
    gt_warp_l1_bwd = np.mean(val_gt_warp_l1_bwd) if val_gt_warp_l1_bwd else 0
    cascade_gap = np.mean(val_cascade_gap) if val_cascade_gap else 0
    cascade_gap_bwd = np.mean(val_cascade_gap_bwd) if val_cascade_gap_bwd else 0
    per_frame_ms = total_time_ms / max(total_frames, 1)

    model.train()

    return {
        'psnr_total': psnr_total,
        'psnr_static': psnr_static,
        'psnr_dynamic': psnr_dynamic,
        'ssim_fwd': ssim_fwd,
        'ssim_fwd_static': ssim_fwd_static,
        'ssim_fwd_dynamic': ssim_fwd_dynamic,
        'psnr_bwd': psnr_bwd,
        'psnr_bwd_static': psnr_bwd_static,
        'psnr_bwd_dynamic': psnr_bwd_dynamic,
        'ssim_bwd': ssim_bwd,
        'ssim_bwd_static': ssim_bwd_static,
        'ssim_bwd_dynamic': ssim_bwd_dynamic,
        'psnr_fusion': psnr_fusion,
        'psnr_fusion_static': psnr_fusion_static,
        'psnr_fusion_dynamic': psnr_fusion_dynamic,
        'ssim_fusion': ssim_fusion,
        'ssim_fusion_static': ssim_fusion_static,
        'ssim_fusion_dynamic': ssim_fusion_dynamic,
        'fusion_l1': fusion_l1,
        'lpips': lpips_avg,
        'lpips_bwd': lpips_bwd_avg,
        'lpips_fusion': lpips_fusion_avg,
        'w0': w0, 'w1': w1, 'w2': w2, 'w3': w3,
        'delta_mean': delta_m,
        'gt_warp_l1': gt_warp_l1,
        'gt_warp_l1_bwd': gt_warp_l1_bwd,
        'cascade_gap': cascade_gap,
        'cascade_gap_bwd': cascade_gap_bwd,
        'per_frame_ms': per_frame_ms,
        'total_enc_ms': total_enc_ms,
        'total_bwd_ms': total_bwd_ms,
        'total_fwd_ms': total_fwd_ms,
        'total_time_ms': total_time_ms,
        'total_frames': total_frames,
    }


def generate_all_previews(model, val_seq, val_events, val_nf, gt_root, device,
                          event_h, event_w, target_h, target_w, event_downsample,
                          num_slices, key_interval, output_dir, epoch=1,
                          mask_kernel_size=5, mask_sigma=0.5, phase=1,
                          delta_scale=10.0, delta_range=0.15):
    model.eval()

    vgt_dir = Path(gt_root) / "val" / val_seq["seq_id"]
    max_kf = val_nf - key_interval
    if max_kf <= 0:
        return

    panel_w = target_w // 2
    panel_h = target_h // 2
    total_w = panel_w * 2
    total_h = panel_h * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thick = 1

    writers = {}
    writers['recon'] = cv2.VideoWriter(str(output_dir / "recon.mp4"), fourcc, 10, (total_w, total_h))
    writers['mask'] = cv2.VideoWriter(str(output_dir / "mask.mp4"), fourcc, 10, (total_w, total_h))
    writers['offset'] = cv2.VideoWriter(str(output_dir / "offset.mp4"), fourcc, 10, (total_w, total_h))
    writers['struct'] = cv2.VideoWriter(str(output_dir / "struct.mp4"), fourcc, 10, (total_w, total_h))
    if phase == 2:
        writers['fusion_comp'] = cv2.VideoWriter(str(output_dir / "fusion_comp.mp4"), fourcc, 10, (total_w, total_h))

    with torch.no_grad():
        _ensure_model_compiled(model)
        for prev_fid in build_keyframe_pairs(max_kf, key_interval):
            next_fid = prev_fid + key_interval

            all_feats, all_membranes, all_offset_feats, C, events_up, grid, scale = _run_snn_forward(
                val_events, prev_fid, next_fid, event_h, event_w, target_h, target_w,
                event_downsample, num_slices, model, device)
            H, W = target_h, target_w
            KI = key_interval

            prev_t = torch.from_numpy(load_gt_frame(vgt_dir, prev_fid, target_h, target_w)).float().permute(2, 0, 1).to(device)
            next_t = torch.from_numpy(load_gt_frame(vgt_dir, next_fid, target_h, target_w)).float().permute(2, 0, 1).to(device)
            struct_prev = model.structure_encoder(prev_t.unsqueeze(0))
            struct_next = model.structure_encoder(next_t.unsqueeze(0))

            cache = StepFeatureCache(all_feats, all_membranes, all_offset_feats, events_up, H, W, C, model, device,
                                      mask_kernel_size=mask_kernel_size, mask_sigma=mask_sigma)

            if phase == 2:
                cascade_bwd, masks_bwd = run_cascade_inference(model, cache, KI, next_t, grid, scale, direction='bwd', use_deblur=True)

            if phase == 1:
                _preview_phase1_body(model, cache, all_feats, all_membranes, all_offset_feats,
                                     events_up, C, H, W, KI, grid, scale, vgt_dir, prev_fid,
                                     device, mask_kernel_size, mask_sigma,
                                     writers, panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)
            else:
                _preview_phase2_body(model, cache, all_feats, all_membranes, all_offset_feats,
                                     events_up, C, H, W, KI, grid, scale, vgt_dir, prev_fid,
                                     cascade_bwd, masks_bwd, struct_prev, struct_next, prev_t, next_t,
                                     device, mask_kernel_size, mask_sigma, target_h, target_w,
                                     delta_scale, delta_range,
                                     writers, panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)

    for w in writers.values():
        w.release()
    print(f"  [preview] saved to {output_dir}")


def _preview_phase1_body(model, cache, all_feats, all_membranes, all_offset_feats,
                         events_up, C, H, W, KI, grid, scale, vgt_dir, prev_fid,
                         device, mask_kernel_size, mask_sigma,
                         writers, panel_w, panel_h, total_w, total_h, font, font_scale, font_thick):

    fwd_results = []
    for t in range(1, KI):
        idx = t - 1
        off_t, off_tm1, off_tp1 = _prepare_offset_features(all_offset_feats, idx, H, W)

        pol_feat_fwd, event_mask_fwd, event_pos_fwd, event_neg_fwd = make_event_features_fwd(
            events_up, t, model, mask_kernel_size, mask_sigma)

        feat_t_fwd, _ = _prepare_step_features(all_feats, all_membranes, idx, H, W, C)

        target_np = load_gt_frame(vgt_dir, prev_fid + t, target_h=H, target_w=W)
        gt_fwd = load_gt_frame(vgt_dir, prev_fid + t - 1, target_h=H, target_w=W)
        gt_fwd_t = torch.from_numpy(gt_fwd).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            struct_fwd = model.structure_encoder(gt_fwd_t)
            result = run_phase1_warp(
                model, gt_fwd_t, struct_fwd,
                off_tm1 if t > 1 else off_t, off_t, pol_feat_fwd,
                grid, scale, 'fwd',
                event_mask_fwd, event_pos_fwd, event_neg_fwd, feat_t_fwd)

        offset_fwd = result['offset_raw'] * event_mask_fwd

        fwd_results.append({
            't': t, 'target_np': target_np,
            'warped_fwd': result['output'],
            'blurry_mask_fwd': result['blurry_mask'],
            'event_mask_fwd': event_mask_fwd,
            'offset_fwd': offset_fwd,
            'struct_t': result['struct_t'],
        })

    bwd_results = {}
    for rb in range(KI - 1, 0, -1):
        idx = rb - 1
        off_t_bwd, _, off_tp1_bwd = _prepare_offset_features(all_offset_feats, rb, H, W)

        pol_feat_bwd, event_mask_bwd, event_pos_bwd, event_neg_bwd = make_event_features_bwd(
            events_up, rb, model, mask_kernel_size, mask_sigma)

        feat_t_bwd, _ = _prepare_step_features(all_feats, all_membranes, rb, H, W, C)

        target_np = load_gt_frame(vgt_dir, prev_fid + rb, target_h=H, target_w=W)
        gt_bwd = load_gt_frame(vgt_dir, prev_fid + rb + 1, target_h=H, target_w=W)
        gt_bwd_t = torch.from_numpy(gt_bwd).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            struct_bwd = model.structure_encoder(gt_bwd_t)
            result = run_phase1_warp(
                model, gt_bwd_t, struct_bwd,
                off_t_bwd, off_tp1_bwd if rb < KI - 1 else off_t_bwd, pol_feat_bwd,
                grid, scale, 'bwd',
                event_mask_bwd, event_pos_bwd, event_neg_bwd, feat_t_bwd)

        bwd_results[rb] = result['output']

    for t in range(1, KI):
        fwd_entry = fwd_results[t - 1]
        warped_bwd = bwd_results[t]

        fwd_np = fwd_entry['warped_fwd'].squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        bwd_np = warped_bwd.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        blurry_np = fwd_entry['blurry_mask_fwd'].squeeze(0).squeeze(0).float().cpu().numpy()
        coarse_np = fwd_entry['event_mask_fwd'].squeeze(0).squeeze(0).float().cpu().numpy()
        offset_np = fwd_entry['offset_fwd'].squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        offset_mag_np = fwd_entry['offset_fwd'].abs().mean(dim=1, keepdim=True).squeeze(0).squeeze(0).float().cpu().numpy()

        _render_recon_phase1(fwd_entry['target_np'], fwd_np, bwd_np, writers['recon'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)
        _render_mask(fwd_entry['target_np'], blurry_np, coarse_np, offset_mag_np, writers['mask'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)
        _render_offset(offset_np, offset_mag_np, coarse_np, writers['offset'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)
        _render_struct(fwd_entry['target_np'], fwd_entry['struct_t'], blurry_np, writers['struct'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)


def _preview_phase2_body(model, cache, all_feats, all_membranes, all_offset_feats,
                         events_up, C, H, W, KI, grid, scale, vgt_dir, prev_fid,
                         cascade_bwd, masks_bwd, struct_prev, struct_next, prev_t, next_t,
                         device, mask_kernel_size, mask_sigma, target_h, target_w,
                         delta_scale, delta_range,
                         writers, panel_w, panel_h, total_w, total_h, font, font_scale, font_thick):
    cascade_fwd, masks_fwd = run_cascade_inference(
        model, cache, KI, prev_t, grid, scale, direction='fwd', use_deblur=True)

    fusion_outputs, fusion_weights, fusion_deltas = run_fusion_inference(
        model, cascade_fwd, cascade_bwd, prev_t, next_t, KI, delta_range,
       blurry_masks_fwd=masks_fwd, blurry_masks_bwd=masks_bwd)

    for i, fusion_out in enumerate(fusion_outputs):
        t = i + 1
        idx = t - 1
        features = cache.get(idx)
        _, _, _, _, _, _, _, event_mask_fwd, _ = features

        inf_fwd = cache.get_inference(idx, 'fwd')

        target_np = load_gt_frame(vgt_dir, prev_fid + t, target_h, target_w)

        fusion_np = fusion_out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        cascade_fwd_np = cascade_fwd[t].squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        cascade_bwd_np = cascade_bwd[t].squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        w = fusion_weights[i]
        delta_np = fusion_deltas[i].squeeze(0).permute(1, 2, 0).float().cpu().numpy()

        coarse_np = event_mask_fwd.squeeze(0).squeeze(0).float().cpu().numpy()

        _render_recon_phase2(target_np, fusion_np, delta_np, writers['recon'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)
        _render_fusion_comp(target_np, cascade_fwd_np, cascade_bwd_np, fusion_np, writers['fusion_comp'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)

        if inf_fwd:
            bm_fwd = masks_fwd[t] if masks_fwd[t] is not None else torch.zeros_like(event_mask_fwd)
            bm_bwd = masks_bwd[t] if masks_bwd[t] is not None else torch.zeros_like(event_mask_fwd)
            combined_blurry = torch.max(bm_fwd, bm_bwd)
            blurry_np = combined_blurry.squeeze(0).squeeze(0).float().cpu().numpy()
            offset_fwd = inf_fwd['offset_raw'] * event_mask_fwd
            offset_np = offset_fwd.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            offset_mag_np = offset_fwd.abs().mean(dim=1, keepdim=True).squeeze(0).squeeze(0).float().cpu().numpy()
            struct_t = inf_fwd['struct_t']

            _render_mask(target_np, blurry_np, coarse_np, offset_mag_np, writers['mask'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)
            _render_offset(offset_np, offset_mag_np, coarse_np, writers['offset'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)
            _render_struct(target_np, struct_t, blurry_np, writers['struct'], panel_w, panel_h, total_w, total_h, font, font_scale, font_thick)


def _render_recon_phase1(target_np, fwd_np, bwd_np, out, pw, ph, tw, th, font, fs, ft):
    """Phase 1: GT | warp_fwd | |warp_fwd-GT|(x10) | warp_bwd"""
    diff_np = np.abs(fwd_np - target_np)
    panels = [
        _rgb_panel(target_np, pw, ph, "GT"),
        _rgb_panel(fwd_np, pw, ph, "warp_fwd"),
        _error_panel(diff_np, pw, ph, "|warp_fwd-GT|"),
        _rgb_panel(bwd_np, pw, ph, "warp_bwd"),
    ]
    out.write(_compose_frame(panels, pw, ph, tw, th))


def _render_recon_phase2(target_np, fusion_np, delta_np, out, pw, ph, tw, th, font, fs, ft):
    """GT | fusion_output | |fusion-GT|(x10) | delta_correction"""
    diff_np = np.abs(fusion_np - target_np)
    panels = [
        _rgb_panel(target_np, pw, ph, "GT"),
        _rgb_panel(fusion_np, pw, ph, "fusion_output"),
        _error_panel(diff_np, pw, ph, "|fusion-GT|"),
        _delta_rgb_panel(delta_np, pw, ph, "delta_correction"),
    ]
    out.write(_compose_frame(panels, pw, ph, tw, th))


def _render_mask(target_np, blurry_np, coarse_np, offset_mag_np, out, pw, ph, tw, th, font, fs, ft):
    """GT+blurry_mask | blurry_mask(heat) | coarse_mask(heat) | |offset_mag|(heat)"""
    panels = [
        _mask_overlay_panel(target_np, blurry_np, pw, ph, "GT+blurry_mask"),
        _heatmap_panel(blurry_np, pw, ph, "blurry_mask"),
        _heatmap_panel(coarse_np, pw, ph, "coarse_mask"),
        _heatmap_panel(offset_mag_np, pw, ph, "|offset_mag|"),
    ]
    out.write(_compose_frame(panels, pw, ph, tw, th))


def _render_offset(offset_np, offset_mag_np, coarse_np, out, pw, ph, tw, th, font, fs, ft):
    """offset_flow(dir) | |offset|(mag) | offset_grad | coarse_mask(overlay)"""
    grad_np = _compute_offset_grad(offset_np)
    panels = [
        _flow_panel(offset_np, pw, ph, "offset_flow"),
        _heatmap_panel(offset_mag_np, pw, ph, "|offset|"),
        _heatmap_panel(grad_np, pw, ph, "offset_grad"),
        _heatmap_panel(coarse_np, pw, ph, "coarse_mask"),
    ]
    out.write(_compose_frame(panels, pw, ph, tw, th))


def _render_struct(target_np, struct_t, blurry_np, out, pw, ph, tw, th, font, fs, ft):
    """blurry_mask(heat) | struct(ch0-2,RGB) | GT+blurry_mask | struct(ch3-5,RGB)"""
    s = struct_t.squeeze(0).float().cpu().numpy()
    ch02 = _normalize_struct_channels(s[:3])
    ch35 = _normalize_struct_channels(s[3:6])
    panels = [
        _heatmap_panel(blurry_np, pw, ph, "blurry_mask"),
        _rgb_panel(ch02.transpose(1, 2, 0), pw, ph, "struct(ch0-2)"),
        _mask_overlay_panel(target_np, blurry_np, pw, ph, "GT+blurry_mask"),
        _rgb_panel(ch35.transpose(1, 2, 0), pw, ph, "struct(ch3-5)"),
    ]
    out.write(_compose_frame(panels, pw, ph, tw, th))


def _render_fusion_comp(target_np, cascade_fwd_np, cascade_bwd_np, fusion_np, out, pw, ph, tw, th, font, fs, ft):
    """GT | cascade_fwd | cascade_bwd | fusion_output"""
    panels = [
        _rgb_panel(target_np, pw, ph, "GT"),
        _rgb_panel(cascade_fwd_np, pw, ph, "cascade_fwd"),
        _rgb_panel(cascade_bwd_np, pw, ph, "cascade_bwd"),
        _rgb_panel(fusion_np, pw, ph, "fusion_output"),
    ]
    out.write(_compose_frame(panels, pw, ph, tw, th))


def _rgb_panel(img_np, pw, ph, label, color=(0, 255, 0)):
    p = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
    p = cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA)
    cv2.putText(p, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return p


def _error_panel(diff_np, pw, ph, label):
    p = (np.clip(diff_np * 10, 0, 1) * 255).astype(np.uint8)
    p = cv2.applyColorMap(p, cv2.COLORMAP_HOT)
    p = cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA)
    cv2.putText(p, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def _heatmap_panel(data_np, pw, ph, label):
    img = _percentile_norm(data_np)
    img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_LINEAR)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _mask_overlay_panel(gt_np, mask_np, pw, ph, label):
    p = (np.clip(gt_np, 0, 1) * 255).astype(np.uint8)
    p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
    p = cv2.resize(p, (pw, ph), interpolation=cv2.INTER_AREA)
    m = mask_np.astype(np.float32)
    m = cv2.resize(m, (pw, ph), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(m * 1, 0, 1)[:, :, np.newaxis]
    red = np.full_like(p, np.array([0, 0, 255], dtype=np.uint8))
    p = (p * (1 - alpha) + red * alpha).astype(np.uint8)
    cv2.putText(p, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return p


def _flow_panel(offset_np, pw, ph, label):
    o = cv2.resize(offset_np, (pw, ph), interpolation=cv2.INTER_LINEAR)
    dx = o[..., 0]
    dy = o[..., 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    max_mag = np.percentile(mag, 95) or 1e-6
    mag_clip = np.clip(mag / max_mag, 0, 1)
    ang = np.arctan2(dy, dx) + np.pi
    hsv = np.zeros((ph, pw, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (mag_clip * 255).astype(np.uint8)
    p = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.putText(p, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return p


def _delta_rgb_panel(delta_np, pw, ph, label):
    d = delta_np.astype(np.float32)
    d_abs = np.abs(d)
    d_max = d_abs.max() if d_abs.max() > 0 else 1.0
    d_norm = np.clip(d / d_max, -1, 1)
    d_vis = ((d_norm * 0.5 + 0.5) * 255).astype(np.uint8)
    d_vis = cv2.cvtColor(d_vis, cv2.COLOR_RGB2BGR)
    d_vis = cv2.resize(d_vis, (pw, ph), interpolation=cv2.INTER_AREA)
    cv2.putText(d_vis, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(d_vis, f"max={d_max:.3f}", (4, ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    return d_vis


def _compose_frame(panels, pw, ph, tw, th):
    frame = np.zeros((th, tw, 3), dtype=np.uint8)
    for i, p in enumerate(panels):
        row = i // 2
        col = i % 2
        frame[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw] = p
    return frame


def _normalize_struct_channels(ch):
    vmin, vmax = ch.min(), ch.max()
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    return (ch - vmin) / (vmax - vmin)


def _compute_offset_grad(offset_np):
    dy = np.abs(np.diff(offset_np, axis=0))
    dx = np.abs(np.diff(offset_np, axis=1))
    dy = np.pad(dy, ((0, 1), (0, 0), (0, 0)), mode='edge')
    dx = np.pad(dx, ((0, 0), (0, 1), (0, 0)), mode='edge')
    return (dy.mean(axis=2) + dx.mean(axis=2)) / 2.0


def _percentile_norm(tensor_1ch, low=2, high=98):
    if isinstance(tensor_1ch, torch.Tensor):
        arr = tensor_1ch.float().detach().cpu().numpy()
    else:
        arr = np.asarray(tensor_1ch, dtype=np.float32)
    vmin, vmax = np.percentile(arr, [low, high])
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    return (arr * 255).astype(np.uint8)