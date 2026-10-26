from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from .event_mask import make_event_features, make_membrane_conf


def _prepare_step_features(all_feats, all_membranes, idx, H, W, C):
    core = torch.cat([all_feats[idx], all_membranes[idx]], dim=1)
    core_up = F.interpolate(core, size=(H, W), mode="bilinear", align_corners=True)
    feat_t, membrane_t = core_up.split([C, C], dim=1)

    return feat_t, membrane_t


def _prepare_offset_features(all_offset_feats, idx, H, W):
    def _get_offset_adj(adj_idx):
        if adj_idx < 0:
            adj_idx = 0
        elif adj_idx >= len(all_offset_feats):
            adj_idx = len(all_offset_feats) - 1
        return F.interpolate(all_offset_feats[adj_idx], size=(H, W),
                             mode="bilinear", align_corners=True)

    offset_feat_t = F.interpolate(all_offset_feats[idx], size=(H, W),
                                  mode="bilinear", align_corners=True)
    offset_feat_tm1 = _get_offset_adj(idx - 1)
    offset_feat_tp1 = _get_offset_adj(idx + 1)

    return offset_feat_t, offset_feat_tm1, offset_feat_tp1


def predict_offset_and_grid(model, feat_earlier, feat_later,
                            pol_feat, grid, scale,
                            direction='fwd', use_checkpoint=False, mask=None,
                            struct_feat=None,
                            offset_head=None, t_ratio=None):
    if struct_feat is None:
        struct_feat = torch.zeros_like(feat_earlier[:, :32])
    if offset_head is None:
        offset_head = model.offset_head_fwd if direction == 'fwd' else model.offset_head_bwd

    offset = offset_head(
        torch.cat([feat_earlier, feat_later, pol_feat, mask, struct_feat], dim=1),
        mask, t_ratio=t_ratio)

    sign = -1 if direction == 'fwd' else 1
    sample_grid = grid + sign * offset.permute(0, 2, 3, 1)

    return offset, sample_grid


def run_phase1_warp(model, src_frame, struct_src,
                    off_earlier, off_later, pol_feat,
                    grid, scale, direction,
                    coarse_mask, event_pos, event_neg, feat_t,
                    use_deblur=True, use_checkpoint=False, t_ratio=None,
                    detach_warp_for_deblur=False):
    """Single-direction warp inference (no loss, shared by train/val/preview).

    Flow: offset_head -> warp -> blurry_predictor -> deblur_head

    Args:
        direction: 'fwd' or 'bwd'
        src_frame:  source frame (GT or cascade output)
        struct_src: StructureEncoder(src_frame)
        use_deblur: if True, apply deblur_head; if False, return raw warped frame
        t_ratio: cascade temporal position (t/KI for fwd, (KI-t)/KI for bwd), None in phase1
        detach_warp_for_deblur: if True, detach warp_raw before feeding to
            blurry_predictor and deblur_head so deblur loss does not pollute
            offset_head gradients (used in Phase 2).

    Returns dict:
        output         - final output (deblurred + clamped)
        blurry_mask    - blurry_predictor output
        offset_raw     - raw offset (before mask multiplication)
        sample_grid    - final grid (offset * mask applied)
        sample_grid_raw- raw grid (no mask, for cache)
        warp_raw       - raw warped frame (before deblur)
        struct_t       - target-time structure feature
    """
    if direction == 'bwd':
        offset_head = model.offset_head_bwd
        blurry_predictor = model.blurry_predictor_bwd
        sign = 1
    else:
        offset_head = model.offset_head_fwd
        blurry_predictor = model.blurry_predictor_fwd
        sign = -1

    offset_raw, _ = predict_offset_and_grid(
        model, off_earlier, off_later, pol_feat,
        grid, scale, direction, use_checkpoint=use_checkpoint, mask=coarse_mask,
        struct_feat=struct_src,
        offset_head=offset_head, t_ratio=t_ratio)

    offset = offset_raw * coarse_mask
    sample_grid = grid + sign * offset.permute(0, 2, 3, 1)
    sample_grid_raw = grid + sign * offset_raw.permute(0, 2, 3, 1)
    struct_t = F.grid_sample(struct_src, sample_grid,
                             align_corners=True, padding_mode='border')
    warp_raw = F.grid_sample(src_frame, sample_grid,
                             align_corners=True, padding_mode='border')

    warp_for_blurry = warp_raw.detach() if detach_warp_for_deblur else warp_raw

    blurry_mask = blurry_predictor(feat_t, event_pos, event_neg, coarse_mask,
                                    warp_for_blurry, t_ratio=t_ratio)

    if use_deblur:
        deblur_head = model.deblur_head_bwd if direction == 'bwd' else model.deblur_head_fwd
        output = deblur_head(warp_for_blurry, blurry_mask)
        _print_color_diag(f"cascade_{direction}", src=src_frame,
                          warp=warp_for_blurry, output=None,
                          blurry_mask=blurry_mask, deblur_raw=output)
        output = torch.clamp(output, 0.0, 1.0)
    else:
        output = warp_raw

    return {
        'output': output,
        'blurry_mask': blurry_mask,
        'offset_raw': offset_raw,
        'sample_grid': sample_grid,
        'sample_grid_raw': sample_grid_raw,
        'warp_raw': warp_raw,
        'struct_t': struct_t,
    }


def run_fusion_step(model, warp_fwd, warp_bwd, prev_kf, next_kf, t_ratio=0.5, delta_range=0.15,
                    blurry_mask_fwd=None, blurry_mask_bwd=None):
    B, _, H, W = warp_fwd.shape
    t_channel = torch.full((B, 1, H, W), t_ratio, device=warp_fwd.device, dtype=warp_fwd.dtype)
    bm_fwd = blurry_mask_fwd if blurry_mask_fwd is not None else torch.zeros_like(warp_fwd[:, :1])
    bm_bwd = blurry_mask_bwd if blurry_mask_bwd is not None else torch.zeros_like(warp_bwd[:, :1])
    fusion_out = model.fusion_head(
        torch.cat([warp_fwd, warp_bwd, prev_kf, next_kf, t_channel, bm_fwd, bm_bwd], dim=1))
    weights = F.softmax(fusion_out[:, :4], dim=1)
    delta = torch.tanh(fusion_out[:, 4:7]) * delta_range

    fused = (weights[:, 0:1] * warp_fwd + weights[:, 1:2] * warp_bwd +
             weights[:, 2:3] * prev_kf + weights[:, 3:4] * next_kf)

    if blurry_mask_fwd is not None and blurry_mask_bwd is not None:
        combined_mask = torch.max(blurry_mask_fwd, blurry_mask_bwd)
    elif blurry_mask_fwd is not None:
        combined_mask = blurry_mask_fwd
    elif blurry_mask_bwd is not None:
        combined_mask = blurry_mask_bwd
    else:
        combined_mask = torch.zeros_like(fused[:, :1])

    fused_with_delta = fused + delta
    result = model.deblur_head_fusion(fused_with_delta, combined_mask)
    _print_color_diag("fusion", src=prev_kf, warp=fused,
                      output=None, blurry_mask=combined_mask, deblur_raw=result)
    output = torch.clamp(result, 0.0, 1.0)
    return output, weights, delta, fused, result


def run_cascade_inference(model, cache, KI, keyframe, grid, scale,
                          direction='bwd', per_step_fn=None, use_deblur=False,
                          store_inference=True):
    cascade = [None] * (KI + 1)
    blurry_masks = [None] * (KI + 1)
    if keyframe.dim() == 3:
        keyframe = keyframe.unsqueeze(0)
    if direction == 'bwd':
        cascade[KI] = keyframe
        rng = list(range(KI - 1, 0, -1))
    else:
        cascade[0] = keyframe
        rng = list(range(1, KI))

    for t in rng:
        idx = t - 1
        features = cache.get(idx)
        feat_t, _, off_t, off_tm1, off_tp1, pol_feat_fwd, pol_feat_bwd, event_mask_fwd, event_mask_bwd = features

        if direction == 'bwd':
            t_ratio = (KI - t) / KI
            off_t_val, _, off_tp1_val = cache.get_offset(t)
            off_earlier = off_t_val
            off_later = off_tp1_val
            pol_feat = pol_feat_bwd
            event_mask = event_mask_bwd
            cascade_ref = cascade[t + 1]
            event_pos = cache._events_up[:, idx * 2 + 2:idx * 2 + 3]
            event_neg = cache._events_up[:, idx * 2 + 3:idx * 2 + 4]
            feat_t_use, _ = _prepare_step_features(
                cache._feats, cache._membs, t, cache._H, cache._W, cache._C)
            blurry_predictor = model.blurry_predictor_bwd
            offset_head = model.offset_head_bwd
            sign = 1
        else:
            t_ratio = t / KI
            off_earlier = off_tm1 if t > 1 else off_t
            off_later = off_t
            pol_feat = pol_feat_fwd
            event_mask = event_mask_fwd
            cascade_ref = cascade[t - 1]
            event_pos = cache._events_up[:, idx * 2:idx * 2 + 1]
            event_neg = cache._events_up[:, idx * 2 + 1:idx * 2 + 2]
            feat_t_use = feat_t
            blurry_predictor = model.blurry_predictor_fwd
            offset_head = model.offset_head_fwd
            sign = -1

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            struct_ref = model.structure_encoder(cascade_ref)

            offset_raw, _ = predict_offset_and_grid(
                model, off_earlier, off_later, pol_feat,
                grid, scale, direction, mask=event_mask,
                struct_feat=struct_ref,
                offset_head=offset_head, t_ratio=t_ratio)

            offset = offset_raw * event_mask
            sample_grid = grid + sign * offset.permute(0, 2, 3, 1)
            sample_grid_raw = grid + sign * offset_raw.permute(0, 2, 3, 1)
            struct_t = F.grid_sample(struct_ref, sample_grid,
                                     align_corners=True, padding_mode='border')
            warp_est = F.grid_sample(cascade_ref, sample_grid,
                                     align_corners=True, padding_mode='border')

            blurry_mask = blurry_predictor(feat_t_use, event_pos, event_neg, event_mask, warp_est, t_ratio=t_ratio)

            if use_deblur:
                deblur_head = model.deblur_head_bwd if direction == 'bwd' else model.deblur_head_fwd
                output = deblur_head(warp_est, blurry_mask)
                output = torch.clamp(output, 0.0, 1.0)
            else:
                output = warp_est

        cascade[t] = output.detach()
        blurry_masks[t] = blurry_mask.detach()

        if store_inference:
            cache.set_inference(idx, direction, offset_raw, sample_grid_raw,
                                blurry_mask, struct_t)

        if per_step_fn is not None:
            per_step_fn(t, idx, output, event_mask, sample_grid, warp_est)

    return cascade, blurry_masks


_color_diag_step = 0
_color_diag_enabled = True


def _print_color_diag(tag, src, warp, output, blurry_mask, deblur_raw=None, **_kw):
    global _color_diag_step
    if not _color_diag_enabled:
        return
    _color_diag_step += 1
    if _color_diag_step > 50:
        return

    def chmean(t, name):
        if t is None:
            return f"  {name}: None"
        t_f = t.float()
        r = t_f[:, 0].mean().item()
        g = t_f[:, 1].mean().item() if t_f.shape[1] >= 2 else -1
        b = t_f[:, 2].mean().item() if t_f.shape[1] >= 3 else -1
        return f"  {name}: R={r:.4f} G={g:.4f} B={b:.4f}"

    lines = [f"[COLOR {tag} step={_color_diag_step}]"]
    lines.append(chmean(src, "src"))
    lines.append(chmean(warp, "warp"))
    if blurry_mask is not None:
        bm = blurry_mask.float()
        lines.append(f"  blurry_mask: mean={bm.mean().item():.4f} active_frac={(bm > 0.1).float().mean().item():.4f}")
    if deblur_raw is not None:
        lines.append(chmean(deblur_raw, "deblur_raw(pre-clamp)"))
    if output is not None:
        lines.append(chmean(output, "output(post-clamp)"))
        if warp is not None and deblur_raw is not None:
            diff = (deblur_raw.float() - warp.float())
            r_d = diff[:, 0].mean().item()
            g_d = diff[:, 1].mean().item() if diff.shape[1] >= 2 else -1
            b_d = diff[:, 2].mean().item() if diff.shape[1] >= 3 else -1
            lines.append(f"  deblur-warp diff: R={r_d:.5f} G={g_d:.5f} B={b_d:.5f}")
    print("\n".join(lines))


def disable_color_diag():
    global _color_diag_enabled
    _color_diag_enabled = False


def reset_color_diag():
    global _color_diag_step
    _color_diag_step = 0


def run_fusion_inference(model, cascade_fwd, cascade_bwd, prev_kf, next_kf, KI, delta_range=0.15,
                         blurry_masks_fwd=None, blurry_masks_bwd=None):
    if prev_kf.dim() == 3:
        prev_kf = prev_kf.unsqueeze(0)
    if next_kf.dim() == 3:
        next_kf = next_kf.unsqueeze(0)
    outputs, weights_list, deltas = [], [], []
    for t in range(1, KI):
        mask_fwd = blurry_masks_fwd[t] if blurry_masks_fwd is not None else None
        mask_bwd = blurry_masks_bwd[t] if blurry_masks_bwd is not None else None
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output, weights, delta, _fused, _deblurred = run_fusion_step(
                model, cascade_fwd[t], cascade_bwd[t], prev_kf, next_kf,
                t_ratio=t / KI, delta_range=delta_range,
                blurry_mask_fwd=mask_fwd, blurry_mask_bwd=mask_bwd)
        outputs.append(output)
        weights_list.append(weights)
        deltas.append(delta)
    return outputs, weights_list, deltas


_tsc_ref_nan_diag_done = False
_tsc_out_nan_diag_done = False


def _train_step_core(model,
                     feat_t, membrane_t,
                     off_t_prim, off_tp1_prim,
                     events_up, t,
                     target,
                     grid, scale,
                     cascade_ref, struct_ref,
                     direction='bwd',
                     mask_kernel_size=5, mask_sigma=0.5,
                     use_checkpoint=True,
                     use_deblur=False, t_ratio=None,
                    detach_warp_for_deblur=False,
                    offset_reg_weight=0.0):
    pol_feat_fwd, pol_feat_bwd, mask_fwd, mask_bwd, event_pos_fwd, event_neg_fwd, event_pos_bwd, event_neg_bwd = make_event_features(
        events_up, t, model, mask_kernel_size, mask_sigma)

    membrane_conf = make_membrane_conf(membrane_t)

    if direction == 'bwd':
        prim_pol = pol_feat_bwd
        prim_mask = mask_bwd
        prim_event_pos = event_pos_bwd
        prim_event_neg = event_neg_bwd
        prim_dir = 'bwd'
        prim_off_earlier = off_t_prim
        prim_off_later = off_tp1_prim
    else:
        prim_pol = pol_feat_fwd
        prim_mask = mask_fwd
        prim_event_pos = event_pos_fwd
        prim_event_neg = event_neg_fwd
        prim_dir = 'fwd'
        prim_off_earlier = off_t_prim
        prim_off_later = off_tp1_prim

    cascade_ref_check_safe = torch.isfinite(cascade_ref).all()
    struct_ref_check_safe = torch.isfinite(struct_ref).all()
    if not (cascade_ref_check_safe and struct_ref_check_safe):
        global _tsc_ref_nan_diag_done
        if not _tsc_ref_nan_diag_done:
            _tsc_ref_nan_diag_done = True
            print(f"[CASCADE NaN DIAG REF t_ratio={t_ratio} dir={direction}] "
                  f"cascade_ref finite={cascade_ref_check_safe.item()} "
                  f"mean={cascade_ref.float().mean().item():.6f} "
                  f"min={cascade_ref.float().min().item():.6f} "
                  f"max={cascade_ref.float().max().item():.6f}")
            print(f"[CASCADE NaN DIAG REF t_ratio={t_ratio} dir={direction}] "
                  f"struct_ref finite={struct_ref_check_safe.item()} "
                  f"mean={struct_ref.float().mean().item():.6f} "
                  f"min={struct_ref.float().min().item():.6f} "
                  f"max={struct_ref.float().max().item():.6f}")

    result = run_phase1_warp(
        model, cascade_ref, struct_ref,
        prim_off_earlier, prim_off_later, prim_pol,
        grid, scale, prim_dir,
        prim_mask, prim_event_pos, prim_event_neg, feat_t,
        use_deblur=use_deblur, use_checkpoint=use_checkpoint, t_ratio=t_ratio,
        detach_warp_for_deblur=detach_warp_for_deblur)

    blurry_mask = result['blurry_mask']
    offset_prim_raw = result['offset_raw']
    sample_grid_prim = result['sample_grid']
    struct_t = result['struct_t']
    output = result['output']
    warp_raw = result['warp_raw']

    if not torch.isfinite(output).all():
        global _tsc_out_nan_diag_done
        if not _tsc_out_nan_diag_done:
            _tsc_out_nan_diag_done = True
            print(f"[CASCADE NaN DIAG OUT t_ratio={t_ratio} dir={direction}] AFTER run_phase1_warp:")
            print(f"  warp_raw: finite={torch.isfinite(warp_raw).all().item()} "
                  f"mean={warp_raw.float().mean().item():.6f} min={warp_raw.float().min().item():.6f} "
                  f"max={warp_raw.float().max().item():.6f}")
            print(f"  blurry_mask: finite={torch.isfinite(blurry_mask).all().item()} "
                  f"mean={blurry_mask.float().mean().item():.6f} min={blurry_mask.float().min().item():.6f} "
                  f"max={blurry_mask.float().max().item():.6f}")
            print(f"  offset_raw: finite={torch.isfinite(offset_prim_raw).all().item()} "
                  f"mean={offset_prim_raw.float().mean().item():.6f} min={offset_prim_raw.float().min().item():.6f} "
                  f"max={offset_prim_raw.float().max().item():.6f}")
            print(f"  output: mean={output.float().mean().item():.6f} min={output.float().min().item():.6f} "
                  f"max={output.float().max().item():.6f}")

    l_recon_warp = (F.l1_loss(warp_raw, target, reduction='none')
               .mean(dim=1, keepdim=True) * prim_mask * membrane_conf.detach()).mean()
    if use_deblur:
        l_recon_deblur = (F.l1_loss(output, target, reduction='none')
                          .mean(dim=1, keepdim=True) * prim_mask * membrane_conf.detach()).mean()
        l_recon = 0.7 * l_recon_deblur
    else:
        l_recon = l_recon_warp

    offset_mag = (offset_prim_raw * scale.view(1, 2, 1, 1)).abs().mean()

    if offset_reg_weight > 0:
        offset_pixel = offset_prim_raw * scale.view(1, 2, 1, 1)
        l_offset_reg = (offset_pixel.square().sum(dim=1, keepdim=True) * prim_mask).mean()
        l_recon = l_recon + offset_reg_weight * l_offset_reg

    return output, struct_t, blurry_mask, offset_prim_raw, warp_raw, {
        'recon': l_recon,
        'offset_mag': offset_mag,
    }


def run_fusion_train(model, warp_fwd, warp_bwd, prev_kf, next_kf,
                     gt_frames, KI, n_frames, device,
                     events_up=None, all_feats=None, all_membranes=None,
                     H=None, W=None, C=None,
                     mask_kernel_size=5, mask_sigma=0.5,
                     wsm_w=0.01, delta_range=0.15,
                     lpips_fn=None, lpips_weight=0.02,
                     fusion_w=1.0,
                     blurry_masks_fwd=None, blurry_masks_bwd=None,
                     blurry_sparsity_weight=0.0):
    total_loss_val = 0.0
    acc = {'recon_fusion': 0.0, 'weight_smooth': 0.0, 'lpips': 0.0,
           'blurry_sparsity': 0.0}

    compute_blurry = events_up is not None and all_feats is not None

    for t in range(1, KI):
        target = gt_frames[t].to(device, non_blocking=True)

        if compute_blurry:
            idx_fwd = t - 1
            pol_fwd, pol_bwd, mask_evt_fwd, mask_evt_bwd, ep_fwd, en_fwd, ep_bwd, en_bwd = \
                make_event_features(events_up, t, model, mask_kernel_size, mask_sigma)
            feat_fwd, _ = _prepare_step_features(all_feats, all_membranes, idx_fwd, H, W, C)
            mask_fwd = model.blurry_predictor_fwd(feat_fwd, ep_fwd, en_fwd, mask_evt_fwd, warp_fwd[t], t_ratio=t / KI)
            feat_bwd, _ = _prepare_step_features(all_feats, all_membranes, t, H, W, C)
            mask_bwd = model.blurry_predictor_bwd(feat_bwd, ep_bwd, en_bwd, mask_evt_bwd, warp_bwd[t], t_ratio=(KI - t) / KI)
        else:
            mask_fwd = blurry_masks_fwd[t] if blurry_masks_fwd is not None else None
            mask_bwd = blurry_masks_bwd[t] if blurry_masks_bwd is not None else None

        output, weights, delta, fused, deblurred = run_fusion_step(
            model, warp_fwd[t], warp_bwd[t], prev_kf, next_kf,
            t_ratio=t / KI, delta_range=delta_range,
            blurry_mask_fwd=mask_fwd, blurry_mask_bwd=mask_bwd)

        if t == 1 and mask_fwd is not None and mask_bwd is not None:
            _cm = torch.max(mask_fwd, mask_bwd)
            print(f"[DIAG Fusion step] combined_mask(max): mean={_cm.mean().item():.4f} "
                  f"std={_cm.std().item():.4f} min={_cm.min().item():.4f} max={_cm.max().item():.4f}")

        l_recon = F.l1_loss(output, target)
        l_smooth = ((weights[:, :, :, 1:] - weights[:, :, :, :-1]).abs().mean() +
                    (weights[:, :, 1:, :] - weights[:, :, :-1, :]).abs().mean())

        l_lpips = torch.tensor(0.0, device=device)
        if lpips_fn is not None and lpips_weight > 0:
            l_lpips = lpips_fn(output * 2.0 - 1.0, target * 2.0 - 1.0).mean()

        loss_t = (fusion_w * l_recon + wsm_w * l_smooth + lpips_weight * l_lpips) / n_frames

        l_blurry_sparsity = torch.tensor(0.0, device=device)

        if blurry_sparsity_weight > 0 and mask_fwd is not None and mask_bwd is not None:
            combined_mask = torch.max(mask_fwd, mask_bwd)
            l_blurry_sparsity = blurry_sparsity_weight * combined_mask.mean() / n_frames
            loss_t = loss_t + l_blurry_sparsity

        loss_t.backward()
        total_loss_val += loss_t.item()
        acc['recon_fusion'] += l_recon.item()
        acc['weight_smooth'] += l_smooth.item()
        acc['lpips'] += l_lpips.item()
        acc['blurry_sparsity'] += l_blurry_sparsity.item()

    if compute_blurry:
        _diag = getattr(run_fusion_train, '_diag_cnt', 0)
        if _diag < 5:
            for name, param in model.named_parameters():
                if 'blurry_predictor' in name and param.grad is not None:
                    print(f"[DIAG Fusion blurry grad {name}] mean={param.grad.mean().item():.8f} "
                          f"std={param.grad.std().item():.8f}")
            run_fusion_train._diag_cnt = _diag + 1

    return total_loss_val, acc


_ckpt_raw_nan_diag_done = False

def _make_train_step_ckpt_fn(model, direction, mask_kernel_size, mask_sigma, t_val, H, W, C, use_deblur=False, t_ratio=None, detach_warp_for_deblur=False, offset_reg_weight=0.0):
    def fn(feat_raw, membrane_raw,
           off_t_raw, off_tp1_raw,
           events_up, target,
           grid, scale, cascade_ref, struct_ref):
        global _ckpt_raw_nan_diag_done
        if not _ckpt_raw_nan_diag_done:
            fr_ok = torch.isfinite(feat_raw).all().item()
            mr_ok = torch.isfinite(membrane_raw).all().item()
            if not (fr_ok and mr_ok):
                _ckpt_raw_nan_diag_done = True
                print(f'[CKPT RAW NaN DIAG dir={direction} t={t_val} t_ratio={t_ratio}]')
                print(f'  feat_raw: finite={fr_ok} mean={feat_raw.float().mean().item():.6f} std={feat_raw.float().std().item():.6f}')
                print(f'  membrane_raw: finite={mr_ok} mean={membrane_raw.float().mean().item():.6f} std={membrane_raw.float().std().item():.6f}')
                print(f'  cascade_ref: finite={torch.isfinite(cascade_ref).all().item()} mean={cascade_ref.float().mean().item():.6f}')
        core = torch.cat([feat_raw, membrane_raw], dim=1)
        core_up = F.interpolate(core, size=(H, W), mode='bilinear', align_corners=True)
        feat_t, membrane_t = core_up.split([C, C], dim=1)

        off_t_prim = F.interpolate(off_t_raw, size=(H, W), mode='bilinear', align_corners=True)
        off_tp1_prim = F.interpolate(off_tp1_raw, size=(H, W), mode='bilinear', align_corners=True)

        amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
                   if feat_raw.dtype == torch.bfloat16 else contextlib.nullcontext())
        with amp_ctx:
            output, struct_t, blurry_mask, offset_prim_raw, warp_raw, losses = _train_step_core(
                model,
                feat_t, membrane_t,
                off_t_prim, off_tp1_prim,
                events_up, t_val,
                target,
                grid, scale,
                cascade_ref, struct_ref,
                direction=direction,
                mask_kernel_size=mask_kernel_size, mask_sigma=mask_sigma,
                use_checkpoint=False,
                use_deblur=use_deblur, t_ratio=t_ratio,
                detach_warp_for_deblur=detach_warp_for_deblur,
                offset_reg_weight=offset_reg_weight)
        return output, struct_t, blurry_mask, offset_prim_raw, warp_raw, \
            losses['recon'], losses['offset_mag']
    return fn


def run_cascade_train(model, all_feats, all_membranes, all_offset_feats,
                     events_up, gt_frames, prev_kf, next_kf, KI, H, W, C, n_frames,
                     grid, scale, device,
                     struct_prev, struct_next,
                     direction='bwd',
                     cascade_mode=True,
                     warp_w=0.5,
                     mask_kernel_size=5, mask_sigma=0.5,
                     use_deblur=False,
                     log_interval=100,
                     diag_prefix='cascade',
                    detach_warp_for_deblur=False,
                    offset_reg_weight=0.0,
                    blurry_sparsity_weight=0.0):
    cascade = [None] * (KI + 1)
    struct_list = [None] * (KI + 1)
    blurry_masks = [None] * (KI + 1)

    if direction == 'bwd':
        cascade[KI] = next_kf
        struct_list[KI] = struct_next
        rng = list(range(KI - 1, 0, -1))
    else:
        cascade[0] = prev_kf
        struct_list[0] = struct_prev
        rng = list(range(1, KI))

    total_loss_val = 0.0
    acc = {k: 0.0 for k in ['recon', 'offset_mag', 'blurry_sparsity']}

    for t in rng:
        idx = t - 1

        if direction == 'bwd':
            feat_raw = all_feats[t]
            membrane_raw = all_membranes[t]
            N = len(all_offset_feats)
            off_t_raw = all_offset_feats[t]
            off_tp1_raw = all_offset_feats[min(t + 1, N - 1)]
            cascade_ref = cascade[t + 1] if cascade_mode else next_kf
            struct_ref = struct_list[t + 1] if cascade_mode else struct_next
        else:
            feat_raw = all_feats[idx]
            membrane_raw = all_membranes[idx]
            N = len(all_offset_feats)
            off_t_raw = all_offset_feats[max(idx - 1, 0)]
            off_tp1_raw = all_offset_feats[idx]
            cascade_ref = cascade[t - 1] if cascade_mode else prev_kf
            struct_ref = struct_list[t - 1] if cascade_mode else struct_prev

        target = gt_frames[t].to(device, non_blocking=True)

        if direction == 'bwd':
            t_ratio = (KI - t) / KI
        else:
            t_ratio = t / KI

        ckpt_fn = _make_train_step_ckpt_fn(model, direction, mask_kernel_size, mask_sigma, t, H, W, C, use_deblur=use_deblur, t_ratio=t_ratio, detach_warp_for_deblur=detach_warp_for_deblur, offset_reg_weight=offset_reg_weight)
        output, struct_t, blurry_mask, off_raw, warp_raw, l_recon, offset_mag = \
            torch.utils.checkpoint.checkpoint(
                ckpt_fn,
                feat_raw, membrane_raw,
                off_t_raw, off_tp1_raw,
                events_up, target,
                grid, scale, cascade_ref, struct_ref,
                use_reentrant=False)
        losses = {
            'recon': l_recon,
            'offset_mag': offset_mag,
        }

        diag_key = f'_diag_step_{diag_prefix}'
        if not hasattr(run_cascade_train, diag_key):
            setattr(run_cascade_train, diag_key, 0)
        current = getattr(run_cascade_train, diag_key) + 1
        setattr(run_cascade_train, diag_key, current)
        if current <= 5 or current % log_interval == 0:
            if blurry_mask is not None:
                print(f"[DIAG Mask {diag_prefix} step={current}] "
                      f"blurry_mask: mean={blurry_mask.mean().item():.4f} "
                      f"std={blurry_mask.std().item():.4f} "
                      f"min={blurry_mask.min().item():.4f} "
                      f"max={blurry_mask.max().item():.4f}")

        l_blurry_sparsity = torch.tensor(0.0, device=device)

        for k in losses:
            acc[k] += losses[k].item()

        if current <= 5 or current % log_interval == 0:
            warp_l1 = F.l1_loss(warp_raw, target).item()
            deblur_l1 = losses['recon'].item()
            if use_deblur and blurry_mask is not None:
                print(f"[DIAG Cascade {diag_prefix} step={current}] "
                      f"raw_warp_L1={warp_l1:.6f}  deblur_L1={deblur_l1:.6f}  "
                      f"gain={warp_l1 - deblur_l1:.6f}  t_ratio={t_ratio:.3f}")
            else:
                print(f"[DIAG Cascade {diag_prefix} step={current}] "
                      f"warp_L1={warp_l1:.6f}  t_ratio={t_ratio:.3f}")

        loss_t = losses['recon'] / n_frames

        if blurry_sparsity_weight > 0 and blurry_mask is not None:
            l_blurry_sparsity = blurry_sparsity_weight * blurry_mask.mean() / n_frames
            loss_t = loss_t + l_blurry_sparsity

        acc['blurry_sparsity'] += l_blurry_sparsity.item()

        total_loss_val += loss_t.item()

        if t != rng[-1]:
            loss_t.backward(retain_graph=True)
        else:
            loss_t.backward(retain_graph=False)

        if current <= 5 or current % log_interval == 0:
            for name, param in model.named_parameters():
                if 'deblur_head' in name and param.grad is not None:
                    g_mean = param.grad.mean().item()
                    g_std = param.grad.std(unbiased=False).item()
                    print(f"[DIAG Grad {diag_prefix} {name}] mean={g_mean:.8f} std={g_std:.8f}")
            for name, param in model.named_parameters():
                if 'blurry_predictor' in name and param.grad is not None:
                    g_mean = param.grad.mean().item()
                    g_std = param.grad.std(unbiased=False).item()
                    print(f"[DIAG Grad {diag_prefix} {name}] mean={g_mean:.8f} std={g_std:.8f}")
            for name, param in model.named_parameters():
                if 'offset_head' in name and param.grad is not None:
                    g_mean = param.grad.mean().item()
                    g_std = param.grad.std(unbiased=False).item()
                    print(f"[DIAG Grad {diag_prefix} {name}] mean={g_mean:.8f} std={g_std:.8f}")

        cascade[t] = output.detach()
        struct_list[t] = struct_t.detach()
        blurry_masks[t] = blurry_mask.detach() if blurry_mask is not None else None

        if direction == 'bwd':
            kf = next_kf
        else:
            kf = prev_kf
        _print_color_diag(f"cascade_{direction}_acc_t{t}", src=kf, warp=output.detach(),
                          output=None, blurry_mask=None, deblur_raw=None)

    return total_loss_val, acc, cascade, blurry_masks


def reset_cascade_diag():
    for prefix in ['cascade', 'bwd', 'fwd']:
        key = f'_diag_step_{prefix}'
        if hasattr(run_cascade_train, key):
            setattr(run_cascade_train, key, 0)


def reset_fusion_diag():
    run_fusion_train._diag_cnt = 0


class StepFeatureCache:
    def __init__(self, all_feats, all_membranes, all_offset_feats, events_up, H, W, C, model, device,
                 mask_kernel_size=5, mask_sigma=0.5):
        self._feats = all_feats
        self._membs = all_membranes
        self._offsets = all_offset_feats
        self._events_up = events_up
        self._H = H
        self._W = W
        self._C = C
        self._model = model
        self._device = device
        self._mask_kernel_size = mask_kernel_size
        self._mask_sigma = mask_sigma
        self._events_cache = {}
        self._inf = {}

    def get(self, idx):
        feat_t, membrane_t = _prepare_step_features(
            self._feats, self._membs, idx, self._H, self._W, self._C)
        off_t, off_tm1, off_tp1 = _prepare_offset_features(
            self._offsets, idx, self._H, self._W)

        if idx in self._events_cache:
            pol_feat_fwd, pol_feat_bwd, event_mask_fwd, event_mask_bwd = self._events_cache[idx]
        else:
            t = idx + 1
            pol_feat_fwd, pol_feat_bwd, event_mask_fwd, event_mask_bwd, _, _, _, _ = make_event_features(
                self._events_up, t, self._model, self._mask_kernel_size, self._mask_sigma)
            self._events_cache[idx] = (pol_feat_fwd, pol_feat_bwd, event_mask_fwd, event_mask_bwd)

        return (feat_t, membrane_t,
                off_t, off_tm1, off_tp1,
                pol_feat_fwd, pol_feat_bwd, event_mask_fwd, event_mask_bwd)

    def get_offset(self, idx):
        off_t, off_tm1, off_tp1 = _prepare_offset_features(
            self._offsets, idx, self._H, self._W)
        return off_t, off_tm1, off_tp1

    def has_inference(self, idx, direction='bwd'):
        entry = self._inf.get(idx, {})
        return direction in entry

    def get_inference(self, idx, direction='bwd'):
        entry = self._inf.get(idx, {})
        if direction not in entry:
            return {}
        c = entry[direction]
        return dict(
            offset_raw=c['offset_raw'].to(self._device),
            sample_grid_raw=c['sample_grid_raw'].to(self._device),
            blurry_mask=c['blurry_mask'].to(self._device),
            struct_t=c['struct_t'].to(self._device),
        )

    def set_inference(self, idx, direction, offset_raw, sample_grid_raw,
                      blurry_mask, struct_t):
        if idx not in self._inf:
            self._inf[idx] = {}
        self._inf[idx][direction] = dict(
            offset_raw=offset_raw.detach().cpu(),
            sample_grid_raw=sample_grid_raw.detach().cpu(),
            blurry_mask=blurry_mask.detach().cpu(),
            struct_t=struct_t.detach().cpu(),
        )