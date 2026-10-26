from __future__ import annotations

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from .cascade import (
    run_phase1_warp,
    run_cascade_train,
    run_fusion_train,
    _prepare_step_features,
    _prepare_offset_features,
    reset_color_diag,
)
from .event_mask import make_event_mask, make_membrane_conf, _make_grid_and_scale


class Phase1Trainer:
    def __init__(self, model, optimizer, device, warp_w=1.0,
                 mask_kernel_size=5, mask_sigma=0.5, use_amp=True, log_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.warp_w = warp_w
        self.mask_kernel_size = mask_kernel_size
        self.mask_sigma = mask_sigma
        self.use_amp = use_amp
        self.log_interval = log_interval

    def train_step(self, events_t, gt_frames, prev_kf, next_kf, H, W, n_frames, epoch=1, step_id=0):
        self.model.train()
        self.optimizer.zero_grad()

        B, _, H_orig, W_orig = prev_kf.shape
        grid, scale = _make_grid_and_scale(B, H, W, self.device)
        KI = gt_frames.__len__() - 1
        n_steps = KI - 1

        amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext())

        with amp_ctx:
            all_feats, all_membranes, all_offset_feats = self.model(events_t)

            events_up = F.interpolate(
                events_t.reshape(events_t.shape[0], -1, events_t.shape[-2], events_t.shape[-1]),
                size=(H, W), mode="bilinear", align_corners=True)

            C = all_feats[0].shape[1]

            _do_diag = (step_id <= 5 or step_id % self.log_interval == 0)
            if _do_diag:
                _feat_vals = torch.cat([f.detach().flatten() for f in all_feats])
                _off_vals = torch.cat([f.detach().flatten() for f in all_offset_feats])
                _memb_vals = torch.cat([m.detach().flatten() for m in all_membranes])
                print(f"[DIAG step={step_id}] encoder feat: mean={_feat_vals.mean().item():.4f} "
                      f"std={_feat_vals.std().item():.4f} "
                      f"min={_feat_vals.min().item():.4f} max={_feat_vals.max().item():.4f}")
                print(f"[DIAG step={step_id}] encoder feat (offset): mean={_off_vals.mean().item():.4f} "
                      f"std={_off_vals.std().item():.4f} "
                      f"min={_off_vals.min().item():.4f} max={_off_vals.max().item():.4f}")
                print(f"[DIAG step={step_id}] membrane: mean={_memb_vals.mean().item():.4f} "
                      f"std={_memb_vals.std().item():.4f} "
                      f"min={_memb_vals.min().item():.4f} max={_memb_vals.max().item():.4f}")

            total_loss_val = 0.0
            acc = {'warp': 0.0, 'warp_fwd': 0.0, 'warp_bwd': 0.0,
                   'offset_mag': 0.0, 'mem': 0.0}

            # FWD: warp gt[t-1] -> gt[t]
            for t in range(1, KI):
                idx = t - 1

                feat_t_fwd, membrane_t_fwd = _prepare_step_features(
                    all_feats, all_membranes, idx, H, W, C)
                off_t_fwd, off_tm1_fwd, off_tp1_fwd = _prepare_offset_features(
                    all_offset_feats, idx, H, W)

                membrane_conf_fwd = make_membrane_conf(membrane_t_fwd).detach()
                event_pos_fwd = events_up[:, idx * 2:idx * 2 + 1]
                event_neg_fwd = events_up[:, idx * 2 + 1:idx * 2 + 2]
                pol_feat_fwd = self.model.polarity_proj(torch.cat([event_pos_fwd, event_neg_fwd], dim=1))
                coarse_mask_fwd = make_event_mask(event_pos_fwd, event_neg_fwd,
                                                  self.mask_kernel_size, self.mask_sigma)
                coarse_mask_fwd = coarse_mask_fwd.to(pol_feat_fwd.dtype)

                off_earlier_fwd = off_tm1_fwd if t > 1 else off_t_fwd
                off_later_fwd = off_t_fwd

                src_fwd = gt_frames[t - 1].to(self.device, non_blocking=True)
                target_t = gt_frames[t].to(self.device, non_blocking=True)

                struct_src_fwd = self.model.structure_encoder(src_fwd)

                result_fwd = run_phase1_warp(
                    self.model, src_fwd, struct_src_fwd,
                    off_earlier_fwd, off_later_fwd, pol_feat_fwd,
                    grid, scale, 'fwd',
                    coarse_mask_fwd, event_pos_fwd, event_neg_fwd, feat_t_fwd,
                    use_deblur=True, use_checkpoint=True)

                blurry_mask_fwd = result_fwd['blurry_mask']
                offset_raw_fwd = result_fwd['offset_raw']
                output_fwd = result_fwd['output']

                l_warp_fwd = (F.l1_loss(output_fwd, target_t, reduction='none')
                              .mean(dim=1, keepdim=True) * coarse_mask_fwd * membrane_conf_fwd).mean()

                if _do_diag and t == 1:
                    _offset_raw_fwd = offset_raw_fwd.detach()
                    _offset_pix_fwd = (_offset_raw_fwd * scale.view(1, 2, 1, 1)).abs()
                    _blurry_fwd = blurry_mask_fwd.detach()
                    print(f"[DIAG step={step_id}] t={t} FWD mask_d={coarse_mask_fwd.detach().mean().item():.4f} "
                          f"blurry_d={_blurry_fwd.mean().item():.4f} "
                          f"blurry_std={_blurry_fwd.std().item():.4f} "
                          f"blurry_max={_blurry_fwd.max().item():.4f} "
                          f"memb_c={membrane_conf_fwd.detach().mean().item():.4f}")
                    print(f"[DIAG step={step_id}] t={t} FWD l_warp={l_warp_fwd.item():.6f} "
                          f"L1(src,gt)={F.l1_loss(src_fwd, target_t).item():.4f}")
                    print(f"[DIAG step={step_id}] t={t} FWD offset_raw: "
                          f"mean=({_offset_raw_fwd[:,0].mean().item():.4f},{_offset_raw_fwd[:,1].mean().item():.4f}) "
                          f"std=({_offset_raw_fwd[:,0].std().item():.4f},{_offset_raw_fwd[:,1].std().item():.4f})")
                    print(f"[DIAG step={step_id}] t={t} FWD offset_pix: "
                          f"mean=({_offset_pix_fwd[:,0].mean().item():.2f},{_offset_pix_fwd[:,1].mean().item():.2f}) "
                          f"max=({_offset_pix_fwd[:,0].max().item():.2f},{_offset_pix_fwd[:,1].max().item():.2f})")

                loss_t_fwd = self.warp_w * l_warp_fwd
                total_loss_val += loss_t_fwd.item()

                acc['warp'] += l_warp_fwd.item() / 2
                acc['warp_fwd'] += l_warp_fwd.item()
                acc['offset_mag'] += (
                    (offset_raw_fwd * scale.view(1, 2, 1, 1)).abs().mean().item()
                ) / 2
                acc['mem'] += membrane_t_fwd.abs().mean().item() / 2

                if t < KI - 1:
                    loss_t_fwd.backward(retain_graph=True)
                else:
                    loss_t_fwd.backward(retain_graph=True)

            # BWD: warp gt[rb+1] -> gt[rb]
            for rb in range(KI - 1, 0, -1):
                idx = rb - 1

                feat_t_bwd, membrane_t_bwd = _prepare_step_features(
                    all_feats, all_membranes, rb, H, W, C)
                off_t_bwd, off_tm1_bwd, off_tp1_bwd = _prepare_offset_features(
                    all_offset_feats, rb, H, W)

                membrane_conf_bwd = make_membrane_conf(membrane_t_bwd).detach()
                off_earlier_bwd = off_t_bwd
                off_later_bwd = off_tp1_bwd if rb < KI - 1 else off_t_bwd

                src_bwd = gt_frames[rb + 1].to(self.device, non_blocking=True)
                target_t = gt_frames[rb].to(self.device, non_blocking=True)

                event_pos_bwd = events_up[:, rb * 2:rb * 2 + 1]
                event_neg_bwd = events_up[:, rb * 2 + 1:rb * 2 + 2]
                pol_feat_bwd = self.model.polarity_proj(torch.cat([event_pos_bwd, event_neg_bwd], dim=1))
                coarse_mask_bwd = make_event_mask(event_pos_bwd, event_neg_bwd,
                                                  self.mask_kernel_size, self.mask_sigma)
                coarse_mask_bwd = coarse_mask_bwd.to(pol_feat_bwd.dtype)

                struct_src_bwd = self.model.structure_encoder(src_bwd)

                result_bwd = run_phase1_warp(
                    self.model, src_bwd, struct_src_bwd,
                    off_earlier_bwd, off_later_bwd, pol_feat_bwd,
                    grid, scale, 'bwd',
                    coarse_mask_bwd, event_pos_bwd, event_neg_bwd, feat_t_bwd,
                    use_deblur=True, use_checkpoint=True)

                blurry_mask_bwd = result_bwd['blurry_mask']
                offset_raw_bwd = result_bwd['offset_raw']
                output_bwd = result_bwd['output']

                l_warp_bwd = (F.l1_loss(output_bwd, target_t, reduction='none')
                              .mean(dim=1, keepdim=True) * coarse_mask_bwd * membrane_conf_bwd).mean()

                if _do_diag and rb == KI - 1:
                    _offset_raw_bwd = offset_raw_bwd.detach()
                    _offset_pix_bwd = (_offset_raw_bwd * scale.view(1, 2, 1, 1)).abs()
                    _blurry_bwd = blurry_mask_bwd.detach()
                    print(f"[DIAG step={step_id}] rb={rb} BWD mask_d={coarse_mask_bwd.detach().mean().item():.4f} "
                          f"blurry_d={_blurry_bwd.mean().item():.4f} "
                          f"blurry_std={_blurry_bwd.std().item():.4f} "
                          f"blurry_max={_blurry_bwd.max().item():.4f} "
                          f"memb_c={membrane_conf_bwd.detach().mean().item():.4f}")
                    print(f"[DIAG step={step_id}] rb={rb} BWD l_warp={l_warp_bwd.item():.6f} "
                          f"L1(src,gt)={F.l1_loss(src_bwd, target_t).item():.4f}")
                    print(f"[DIAG step={step_id}] rb={rb} BWD offset_raw: "
                          f"mean=({_offset_raw_bwd[:,0].mean().item():.4f},{_offset_raw_bwd[:,1].mean().item():.4f}) "
                          f"std=({_offset_raw_bwd[:,0].std().item():.4f},{_offset_raw_bwd[:,1].std().item():.4f})")
                    print(f"[DIAG step={step_id}] rb={rb} BWD offset_pix: "
                          f"mean=({_offset_pix_bwd[:,0].mean().item():.2f},{_offset_pix_bwd[:,1].mean().item():.2f}) "
                          f"max=({_offset_pix_bwd[:,0].max().item():.2f},{_offset_pix_bwd[:,1].max().item():.2f})")

                loss_t_bwd = self.warp_w * l_warp_bwd
                total_loss_val += loss_t_bwd.item()

                acc['warp'] += l_warp_bwd.item() / 2
                acc['warp_bwd'] += l_warp_bwd.item()
                acc['offset_mag'] += (
                    (offset_raw_bwd * scale.view(1, 2, 1, 1)).abs().mean().item()
                ) / 2
                acc['mem'] += membrane_t_bwd.abs().mean().item() / 2

                if rb > 1:
                    loss_t_bwd.backward(retain_graph=True)
                else:
                    loss_t_bwd.backward(retain_graph=False)

            if _do_diag:
                _total_grad_norm = 0.0
                for _p in self.model.parameters():
                    if _p.grad is not None:
                        _total_grad_norm += _p.grad.norm().item() ** 2
                _total_grad_norm = _total_grad_norm ** 0.5

                _se_grads = {}
                for _name, _p in self.model.structure_encoder.named_parameters():
                    if _p.grad is not None:
                        _se_grads[_name] = _p.grad.norm().item()
                if _se_grads:
                    _lines = "  ".join(f"{k}={v:.2e}" for k, v in _se_grads.items())
                    _se_total = sum(v for v in _se_grads.values())
                    print(f"[DIAG step={step_id}] struct_encoder grads (sum={_se_total:.2e}): {_lines}")
                else:
                    print(f"[DIAG step={step_id}] struct_encoder grads: ALL ZERO!")

                _oh_grads = {}
                for _name, _p in self.model.offset_head_fwd.named_parameters():
                    if _p.grad is not None:
                        _oh_grads[_name] = _p.grad.norm().item()
                if _oh_grads:
                    _oh_total = sum(v for v in _oh_grads.values())
                    _lines = "  ".join(f"{k}={v:.2e}" for k, v in list(_oh_grads.items())[:4])
                    print(f"[DIAG step={step_id}] offset_head_fwd grads (sum={_oh_total:.2e} first4): {_lines}")

                _oh_bwd_grads = {}
                for _name, _p in self.model.offset_head_bwd.named_parameters():
                    if _p.grad is not None:
                        _oh_bwd_grads[_name] = _p.grad.norm().item()
                if _oh_bwd_grads:
                    _oh_bwd_total = sum(v for v in _oh_bwd_grads.values())
                    _lines = "  ".join(f"{k}={v:.2e}" for k, v in list(_oh_bwd_grads.items())[:4])
                    print(f"[DIAG step={step_id}] offset_head_bwd grads (sum={_oh_bwd_total:.2e} first4): {_lines}")

                _bp_grads = {}
                for _name, _p in self.model.blurry_predictor_fwd.named_parameters():
                    if _p.grad is not None:
                        _bp_grads[_name] = _p.grad.norm().item()
                if _bp_grads:
                    _bp_total = sum(v for v in _bp_grads.values())
                    print(f"[DIAG step={step_id}] blurry_predictor_fwd grads (sum={_bp_total:.2e}): "
                          f"bp_fwd={_bp_total/max(_total_grad_norm,1e-12):.4f}")

                _bp_bwd_grads = {}
                for _name, _p in self.model.blurry_predictor_bwd.named_parameters():
                    if _p.grad is not None:
                        _bp_bwd_grads[_name] = _p.grad.norm().item()
                if _bp_bwd_grads:
                    _bp_bwd_total = sum(v for v in _bp_bwd_grads.values())
                    print(f"[DIAG step={step_id}] blurry_predictor_bwd grads (sum={_bp_bwd_total:.2e}): "
                          f"bp_bwd={_bp_bwd_total/max(_total_grad_norm,1e-12):.4f}")

                _eb_grads = {}
                for _name, _p in self.model.encoder_body.named_parameters():
                    if _p.grad is not None:
                        _eb_grads[_name] = _p.grad.norm().item()
                if _eb_grads:
                    _eb_total = sum(v for v in _eb_grads.values())
                    _lines = "  ".join(f"{k}={v:.2e}" for k, v in list(_eb_grads.items())[:4])
                    print(f"[DIAG step={step_id}] encoder_body grads (sum={_eb_total:.2e} first4): {_lines}")
                else:
                    print(f"[DIAG step={step_id}] encoder_body grads: ALL ZERO!")

                _dh_grads = {}
                for _name, _p in self.model.named_parameters():
                    if 'deblur_head' in _name and _p.grad is not None:
                        _dh_grads[_name] = _p.grad.norm().item()
                if _dh_grads:
                    _dh_total = sum(v for v in _dh_grads.values())
                    print(f"[DIAG step={step_id}] deblur_head grads (sum={_dh_total:.2e}): "
                          f"dh={_dh_total/max(_total_grad_norm,1e-12):.4f}")

                print(f"[DIAG step={step_id}] grad_ratio se/total={_se_total/max(_total_grad_norm,1e-12):.6f} "
                      f"oh_fwd/total={_oh_total/max(_total_grad_norm,1e-12):.6f}  "
                      f"oh_bwd/total={_oh_bwd_total/max(_total_grad_norm,1e-12):.6f}  "
                      f"eb/total={_eb_total/max(_total_grad_norm,1e-12):.6f}")
            else:
                _se_total = _oh_total = _oh_bwd_total = _eb_total = _total_grad_norm = 0.0

            _grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if _do_diag:
                print(f"[DIAG step={step_id}] grad_norm(before_clip)={_grad_norm_before.item():.6f}")

                for _dh_tag, _dh in [('fwd', self.model.deblur_head_fwd),
                                     ('bwd', self.model.deblur_head_bwd)]:
                    if hasattr(_dh, 'last_color_bias') and _dh.last_color_bias is not None:
                        cb = _dh.last_color_bias.float()
                        print(f"[DIAG color_bias p1_{_dh_tag}] R={cb[:, 0].mean().item():+.5f}[{cb[:, 0].std().item():.4f}] "
                              f"G={cb[:, 1].mean().item():+.5f}[{cb[:, 1].std().item():.4f}] "
                              f"B={cb[:, 2].mean().item():+.5f}[{cb[:, 2].std().item():.4f}] "
                              f"absmax={cb.abs().max().item():.4f}")
                    else:
                        print(f"[DIAG color_bias p1_{_dh_tag}] None")
                _cn_grad_norm = 0.0
                for name, param in self.model.named_parameters():
                    if 'color_net' in name and param.grad is not None:
                        _cn_grad_norm += param.grad.norm().item() ** 2
                _cn_grad_norm = _cn_grad_norm ** 0.5
                print(f"[DIAG color_net p1] total_grad_norm={_cn_grad_norm:.6e}")

        self.optimizer.step()

        for k in acc:
            acc[k] /= n_steps

        return total_loss_val, acc


class Phase2Trainer:
    def __init__(self, model, optimizer, device, warp_w=0.5,
                 mask_kernel_size=5, mask_sigma=0.5, use_amp=True,
                 fusion_w=1.0, wsm_w=None, delta_scale=None, delta_range=0.15,
                 lpips_fn=None, lpips_weight=0.02,
                 use_deblur=True,
                 blurry_lr=None,
                 log_interval=100,
                 use_parallel=True,
                 blurry_sparsity_weight=0.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.warp_w = warp_w
        self.mask_kernel_size = mask_kernel_size
        self.mask_sigma = mask_sigma
        self.use_amp = use_amp

        self.fusion_w = fusion_w
        self.wsm_w = wsm_w
        self.delta_scale = delta_scale
        self.delta_range = delta_range
        self.lpips_fn = lpips_fn
        self.lpips_weight = lpips_weight
        self.use_deblur = use_deblur
        self.blurry_lr = blurry_lr
        self.log_interval = log_interval
        self.use_parallel = use_parallel
        self.blurry_sparsity_weight = blurry_sparsity_weight

        self.model.encoder_body.requires_grad_(False)

    def train_step(self, events_t, gt_frames, prev_kf, next_kf, H, W, n_frames, epoch=1, step_id=0, **kwargs):
        reset_color_diag()
        self.model.train()
        self.optimizer.zero_grad()

        B, _, H_orig, W_orig = prev_kf.shape
        grid, scale = _make_grid_and_scale(B, H, W, self.device)
        C = self.model.encoder_body[0].out_channels
        KI = gt_frames.__len__() - 1
        n_frames_inner = KI - 1

        struct_prev = self.model.structure_encoder(prev_kf)
        struct_next = self.model.structure_encoder(next_kf)

        amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext())

        _do_diag = (step_id <= 5 or step_id % self.log_interval == 0)

        with amp_ctx:
            events_up = F.interpolate(
                events_t.reshape(events_t.shape[0], -1, events_t.shape[-2], events_t.shape[-1]),
                size=(H, W), mode="bilinear", align_corners=True)

            all_feats, all_membranes, all_offset_feats = self.model(events_t)

        if not getattr(Phase2Trainer, '_enc_diag_done', False):
            events_ok = torch.isfinite(events_t).all().item()
            feats_ok = [torch.isfinite(f).all().item() for f in all_feats]
            membs_ok = [torch.isfinite(m).all().item() for m in all_membranes]
            offs_ok = [torch.isfinite(o).all().item() for o in all_offset_feats]
            any_bad = not (events_ok and all(feats_ok) and all(membs_ok) and all(offs_ok))
            if any_bad:
                Phase2Trainer._enc_diag_done = True
                print(f'[ENC NaN DIAG step={step_id}] events_t finite={events_ok}')
                for i, ok in enumerate(feats_ok):
                    if not ok:
                        print(f'  all_feats[{i}]: finite={ok} mean={all_feats[i].float().mean().item():.4f}')
                for i, ok in enumerate(membs_ok):
                    if not ok:
                        print(f'  all_membranes[{i}]: finite={ok} mean={all_membranes[i].float().mean().item():.4f}')
                for i, ok in enumerate(offs_ok):
                    if not ok:
                        print(f'  all_offset_feats[{i}]: finite={ok} mean={all_offset_feats[i].float().mean().item():.4f}')

        # ============================================================
        # STAGE 1: Cascade Deblur with Cascade Source
        #   deblur→offset gradient path preserved (detach_warp_for_deblur=False)
        #   offset_head receives aligned weak gradients via cascade deblur loss
        #   blurry_predictor + deblur_head receive the main cascade gradients
        # ============================================================
        if self.use_parallel:
            stream_bwd = torch.cuda.Stream()
            stream_fwd = torch.cuda.Stream()

            with torch.cuda.stream(stream_bwd):
                with amp_ctx:
                    bwd_loss, acc_bwd, cascade_bwd, masks_bwd = run_cascade_train(
                        self.model, all_feats, all_membranes, all_offset_feats, events_up,
                        gt_frames, prev_kf, next_kf, KI, H, W, C, n_frames_inner,
                        grid, scale, self.device,
                        struct_prev, struct_next,
                        direction='bwd',
                        warp_w=self.warp_w,
                        mask_kernel_size=self.mask_kernel_size, mask_sigma=self.mask_sigma,
                        use_deblur=self.use_deblur,
                        detach_warp_for_deblur=False,
                        offset_reg_weight=0.0,
                        log_interval=self.log_interval,
                        blurry_sparsity_weight=self.blurry_sparsity_weight,
                        diag_prefix='bwd')

            with torch.cuda.stream(stream_fwd):
                with amp_ctx:
                    fwd_loss, acc_fwd, cascade_fwd, masks_fwd = run_cascade_train(
                        self.model, all_feats, all_membranes, all_offset_feats, events_up,
                        gt_frames, prev_kf, next_kf, KI, H, W, C, n_frames_inner,
                        grid, scale, self.device,
                        struct_prev, struct_next,
                        direction='fwd',
                        warp_w=self.warp_w,
                        mask_kernel_size=self.mask_kernel_size, mask_sigma=self.mask_sigma,
                        use_deblur=self.use_deblur,
                        detach_warp_for_deblur=False,
                        offset_reg_weight=0.0,
                        log_interval=self.log_interval,
                        blurry_sparsity_weight=self.blurry_sparsity_weight,
                        diag_prefix='fwd')

            torch.cuda.synchronize()
        else:
            with amp_ctx:
                bwd_loss, acc_bwd, cascade_bwd, masks_bwd = run_cascade_train(
                    self.model, all_feats, all_membranes, all_offset_feats, events_up,
                    gt_frames, prev_kf, next_kf, KI, H, W, C, n_frames_inner,
                    grid, scale, self.device,
                    struct_prev, struct_next,
                    direction='bwd',
                    warp_w=self.warp_w,
                    mask_kernel_size=self.mask_kernel_size, mask_sigma=self.mask_sigma,
                    use_deblur=self.use_deblur,
                    detach_warp_for_deblur=False,
                    offset_reg_weight=0.0,
                    log_interval=self.log_interval,
                    blurry_sparsity_weight=self.blurry_sparsity_weight,
                    diag_prefix='bwd')
            with amp_ctx:
                fwd_loss, acc_fwd, cascade_fwd, masks_fwd = run_cascade_train(
                    self.model, all_feats, all_membranes, all_offset_feats, events_up,
                    gt_frames, prev_kf, next_kf, KI, H, W, C, n_frames_inner,
                    grid, scale, self.device,
                    struct_prev, struct_next,
                    direction='fwd',
                    warp_w=self.warp_w,
                    mask_kernel_size=self.mask_kernel_size, mask_sigma=self.mask_sigma,
                    use_deblur=self.use_deblur,
                    detach_warp_for_deblur=False,
                    offset_reg_weight=0.0,
                    log_interval=self.log_interval,
                    blurry_sparsity_weight=self.blurry_sparsity_weight,
                    diag_prefix='fwd')

        cascade_bwd = [c.detach() if c is not None else None for c in cascade_bwd]
        cascade_fwd = [c.detach() if c is not None else None for c in cascade_fwd]

        # ============================================================
        # STAGE 2: Fusion (detach cascade output, blurry_predictor re-forward)
        #   gradients injected into fusion_head + deblur_head_fusion + blurry_predictor
        # ============================================================
        with amp_ctx:
            fusion_loss, acc_fusion = run_fusion_train(
                self.model, cascade_fwd, cascade_bwd, prev_kf, next_kf,
                gt_frames, KI, n_frames_inner, self.device,
                events_up=events_up,
                all_feats=all_feats, all_membranes=all_membranes,
                H=H, W=W, C=C,
                blurry_masks_fwd=masks_fwd, blurry_masks_bwd=masks_bwd,
                mask_kernel_size=self.mask_kernel_size, mask_sigma=self.mask_sigma,
                wsm_w=self.wsm_w, delta_range=self.delta_range,
                lpips_fn=self.lpips_fn, lpips_weight=self.lpips_weight,
                fusion_w=self.fusion_w,
                blurry_sparsity_weight=self.blurry_sparsity_weight)

        del all_feats, all_membranes, all_offset_feats

        # ============================================================
        # Single optimizer.step() — cascade + fusion gradients already accumulated in each backward()
        # ============================================================
        _grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if _do_diag:
            print(f"[DIAG step={step_id}] grad_norm={_grad_norm.item():.6f}")

            _s_vals = torch.cat([struct_prev.detach().flatten(), struct_next.detach().flatten()])
            print(f"[DIAG step={step_id}] struct_feat: mean={_s_vals.mean().item():.4f} "
                  f"std={_s_vals.std().item():.4f} "
                  f"min={_s_vals.min().item():.4f} max={_s_vals.max().item():.4f}")

            _fusion_keys = ['recon_fusion', 'weight_smooth', 'lpips']
            _fusion_vals = "  ".join(f"{k}={acc_fusion.get(k, 0) / n_frames_inner:.5f}" for k in _fusion_keys)
            print(f"[DIAG step={step_id}] fusion: {_fusion_vals}")

            for name, param in self.model.named_parameters():
                if ('fusion_head' in name or 'deblur_head' in name or 'blurry_predictor' in name
                        or 'offset_head' in name) and param.grad is not None:
                    g_mean = param.grad.mean().item()
                    g_std = param.grad.std(unbiased=False).item()
                    print(f"[DIAG Grad {name}] mean={g_mean:.8f} std={g_std:.8f}")

            for tag, dh in [('deblur_fwd', self.model.deblur_head_fwd),
                            ('deblur_bwd', self.model.deblur_head_bwd),
                            ('deblur_fusion', self.model.deblur_head_fusion)]:
                if hasattr(dh, 'last_color_bias') and dh.last_color_bias is not None:
                    cb = dh.last_color_bias.float()
                    print(f"[DIAG color_bias {tag}] R={cb[:, 0].mean().item():+.5f}[{cb[:, 0].std().item():.4f}] "
                          f"G={cb[:, 1].mean().item():+.5f}[{cb[:, 1].std().item():.4f}] "
                          f"B={cb[:, 2].mean().item():+.5f}[{cb[:, 2].std().item():.4f}] "
                          f"absmax={cb.abs().max().item():.4f}")
                else:
                    print(f"[DIAG color_bias {tag}] None (not yet computed)")
            _cn_grad_norm = 0.0
            for name, param in self.model.named_parameters():
                if 'color_net' in name and param.grad is not None:
                    _cn_grad_norm += param.grad.norm().item() ** 2
            _cn_grad_norm = _cn_grad_norm ** 0.5
            print(f"[DIAG color_net] total_grad_norm={_cn_grad_norm:.6e}")

        acc = {}
        for k in acc_bwd:
            acc[k] = (acc_bwd[k] + acc_fwd.get(k, 0)) / (2 * n_frames_inner)
        for k in ['recon_fusion', 'weight_smooth', 'lpips']:
            if k in acc_fusion:
                acc[k] = acc_fusion[k] / n_frames_inner
        acc['loss'] = (bwd_loss + fwd_loss + fusion_loss) / 3
        acc['bwd_loss'] = bwd_loss
        acc['fwd_loss'] = fwd_loss
        acc['fusion_loss'] = fusion_loss

        return bwd_loss + fwd_loss + fusion_loss, acc