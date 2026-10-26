from __future__ import annotations

import random
import signal
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .model import SNNReconstruct
from .data import extract_temporal_slices, load_gt_frame, discover_sequences, preload_events, count_gt_frames, build_keyframe_pairs
from .phases import Phase1Trainer, Phase2Trainer
from .cascade import reset_cascade_diag, reset_fusion_diag
from .evaluator import run_validation, generate_all_previews


class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_file, "a", encoding="utf-8")

    def _ts(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, level, msg):
        ts = self._ts()
        line = f"[{ts}] [{level}] {msg}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def info(self, msg):
        self._write("INFO", msg)

    def warning(self, msg):
        self._write("WARN", msg)

    def error(self, msg):
        self._write("ERROR", msg)

    def close(self):
        self._file.close()


import re
import hashlib

def _extract_epoch_num(path):
    """Extract numeric epoch from checkpoint filename for natural sorting."""
    m = re.search(r'epoch(\d+)', path.name)
    return int(m.group(1)) if m else 0


def _matched_checksums(model, ckpt_sd, module_path):
    sd_keys = sorted(k for k in ckpt_sd if k.startswith('_orig_mod.' + module_path + '.') or k.startswith(module_path + '.'))
    if not sd_keys:
        return 'NO_KEYS', 'NO_KEYS'

    named = dict(model.named_parameters())
    ckpt_parts = [ckpt_sd[k].flatten().cpu() for k in sd_keys]
    model_parts = [named[k].data.flatten() for k in sd_keys]

    ckpt_sum = hashlib.sha256(torch.cat(ckpt_parts).float().numpy().tobytes()).hexdigest()[:8]
    model_sum = hashlib.sha256(torch.cat(model_parts).cpu().float().numpy().tobytes()).hexdigest()[:8]
    return model_sum, ckpt_sum


class Trainer:
    def __init__(self, args):
        self.args = args
        self.exp_root = Path("experiments") / args.exp_name
        self.phase = args.phase

        self.phase_dir = self.exp_root / f"phase{self.phase}"
        self.ckpt_dir = self.phase_dir / "ckpt"
        self.log = None

    def setup(self):
        args = self.args
        self.phase_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log = Logger(self.phase_dir / "logs" / "train.log")

        self.device = torch.device(args.device)
        self.log.info(f"device={self.device}  exp={args.exp_name}  phase={self.phase}")

        self.use_amp = self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        if self.device.type == 'cuda' and not self.use_amp:
            self.log.warning("bf16 not supported, running in fp32")
        elif self.use_amp:
            self.log.info("bf16 AMP enabled")

        train_seqs = discover_sequences(args.data_root, "train")
        val_seqs = discover_sequences(args.data_root, "val")
        if not train_seqs:
            self.log.error("no training sequences found")
            return False

        self.log.info(f"found {len(train_seqs)} train sequences, {len(val_seqs)} val sequences")

        self.train_events = preload_events(train_seqs)
        self.val_events = preload_events(val_seqs) if val_seqs else {}
        self.train_frames = count_gt_frames(args.gt_root / "train", train_seqs)
        self.val_frames = count_gt_frames(args.gt_root / "val", val_seqs) if val_seqs else {}

        first_seq = train_seqs[0]
        first_events = self.train_events[first_seq["seq_id"]]
        x_arr = first_events.get("x", [])
        y_arr = first_events.get("y", [])
        self.event_h = int(np.max(y_arr)) + 1 if len(y_arr) > 0 else 360
        self.event_w = int(np.max(x_arr)) + 1 if len(x_arr) > 0 else 640

        sample_gt_dir = args.gt_root / "train" / first_seq["gt_dir_name"]
        sample_p = next(sample_gt_dir.glob("*.png"))
        import cv2
        sample_img = cv2.imread(str(sample_p))
        self.target_h, self.target_w = sample_img.shape[:2]

        self.event_downsample = int(first_events.get("event_support", 1) or 1)
        self.log.info(f"event={self.event_w}x{self.event_h}, target={self.target_w}x{self.target_h}  "
                      f"event_downsample={self.event_downsample}")

        self.model = SNNReconstruct(T=args.key_interval, hidden=args.hidden, phase=args.phase).to(self.device)
        self.log.info(f"model params: {sum(p.numel() for p in self.model.parameters()):,}  "
                      f"hidden={args.hidden}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.start_epoch = 1
        self.resume_step = 0
        self._se_opt_state = None
        self.best_metric = -1.0
        self.best_epoch = 0
        self._best_ckpt_path = None

        lpips_fn = None
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex', spatial=False).to(self.device)
            lpips_fn.eval()
            self.log.info("LPIPS metric enabled (alex)")
        except ImportError:
            self.log.info("LPIPS not available")

        self.lpips_fn = lpips_fn

        train_ranges = {}
        for seq in train_seqs:
            sid = seq["seq_id"]
            nf = self.train_frames.get(sid, 0)
            max_kf = nf - args.key_interval
            if max_kf > 0:
                train_ranges[sid] = build_keyframe_pairs(max_kf, args.key_interval)

        self.all_pairs = []
        for sid in train_ranges:
            for prev_fid in train_ranges[sid]:
                self.all_pairs.append((sid, prev_fid))
        self.total_pairs = len(self.all_pairs)
        self.log.info(f"train pairs: {self.total_pairs}")

        val_ranges = {}
        if val_seqs:
            for seq in val_seqs:
                sid = seq["seq_id"]
                nf = self.val_frames.get(sid, 0)
                max_kf = nf - args.key_interval
                if max_kf > 0:
                    val_ranges[sid] = build_keyframe_pairs(max_kf, args.key_interval)
        self.val_ranges = val_ranges
        self.val_seqs = val_seqs

        self.preview_seq = None
        if args.preview_seq:
            for vs in val_seqs:
                if vs["seq_id"] == args.preview_seq:
                    self.preview_seq = vs
                    break
        self.preview_interval = getattr(args, 'preview_interval', 1)

        self.num_slices = args.key_interval
        self.n_frames = self.num_slices

        self.train_seqs = train_seqs

        self._setup_phase_trainer()
        self._load_checkpoint()

        if self._se_opt_state is not None and hasattr(self.phase_trainer, 'struct_encoder_optimizer'):
            se_opt = self.phase_trainer.struct_encoder_optimizer
            if se_opt is not None:
                se_opt.load_state_dict(self._se_opt_state)
                self.log.info("resume struct_encoder_optimizer")

        import csv
        self.metrics_path = self.phase_dir / "metrics.csv"
        csv_exists = self.metrics_path.exists()
        self.csv_file = open(self.metrics_path, "a" if csv_exists else "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not csv_exists:
            if self.phase == 1:
                header = [
                    "epoch", "train_loss",
                    "train_warp", "train_offset_mag",
                    "val_psnr_fwd", "val_psnr_fwd_static", "val_psnr_fwd_dynamic",
                    "val_psnr_bwd", "val_psnr_bwd_static", "val_psnr_bwd_dynamic",
                    "val_ssim_fwd", "val_ssim_fwd_static", "val_ssim_fwd_dynamic",
                    "val_ssim_bwd", "val_ssim_bwd_static", "val_ssim_bwd_dynamic",
                    "val_lpips_fwd", "val_lpips_bwd",
                    "val_gt_warp_l1_fwd", "val_gt_warp_l1_bwd",
                    "per_frame_ms",
                ]
            else:
                header = [
                    "epoch", "train_loss",
                    "train_recon", "train_wsm",
                    "train_warp", "train_offset_mag",
                    "val_psnr_fwd", "val_psnr_fwd_static", "val_psnr_fwd_dynamic",
                    "val_psnr_bwd", "val_psnr_bwd_static", "val_psnr_bwd_dynamic",
                    "val_ssim_fwd", "val_ssim_fwd_static", "val_ssim_fwd_dynamic",
                    "val_ssim_bwd", "val_ssim_bwd_static", "val_ssim_bwd_dynamic",
                ]
                if self.lpips_fn is not None:
                    header += ["val_lpips_fwd", "val_lpips_bwd"]
                header += [
                    "val_w0", "val_w1", "val_w2", "val_w3", "val_delta_mean",
                    "val_gt_warp_l1_fwd", "val_gt_warp_l1_bwd", "val_cascade_gap_fwd", "val_cascade_gap_bwd",
                    "val_psnr_fusion", "val_psnr_fusion_static", "val_psnr_fusion_dynamic",
                    "val_ssim_fusion", "val_ssim_fusion_static", "val_ssim_fusion_dynamic",
                    "val_fusion_l1",
                ]
                if self.lpips_fn is not None:
                    header.append("val_lpips_fusion")
                header.append("per_frame_ms")
            self.csv_writer.writerow(header)

        return True

    def _setup_phase_trainer(self):
        args = self.args
        p = self.phase

        if p == 1:
            se_params = list(self.model.structure_encoder.parameters())
            se_ids = {id(p) for p in se_params}
            for group in self.optimizer.param_groups:
                group['params'] = [p for p in group['params'] if id(p) not in se_ids]
            self.optimizer.add_param_group({'params': se_params, 'lr': args.lr * 10})
            self.log.info(f"structure_encoder lr={args.lr*10:.0e} (10x base)")

            warp_w = args.warp_w if args.warp_w is not None else 1.0
            self.phase_trainer = Phase1Trainer(
                self.model, self.optimizer, self.device,
                warp_w=warp_w,
                mask_kernel_size=args.mask_kernel_size, mask_sigma=args.mask_sigma,
                use_amp=self.use_amp,
                log_interval=args.log_interval)
            self.phase_epochs = args.phase1_epochs
        elif p == 2:
            struct_pids = {id(p) for p in self.model.structure_encoder.parameters()}
            for group in self.optimizer.param_groups:
                group['params'] = [p for p in group['params'] if id(p) not in struct_pids]

            t_anchor_names = {'t_scale', 't_bias_proj', 't_conv'}
            t_anchor_pids = {id(p) for n, p in self.model.named_parameters()
                             if any(tn in n for tn in t_anchor_names)}
            for group in self.optimizer.param_groups:
                group['params'] = [p for p in group['params'] if id(p) not in t_anchor_pids]
            t_anchor_params = [p for n, p in self.model.named_parameters()
                               if any(tn in n for tn in t_anchor_names)]
            _a_lr = getattr(args, 't_anchor_lr', None)
            t_anchor_lr = _a_lr if _a_lr is not None else args.lr * 10
            if t_anchor_params:
                self.optimizer.add_param_group({
                    'params': t_anchor_params,
                    'lr': t_anchor_lr,
                    'betas': (0.9, 0.999),
                })
                self.log.info(f"t_anchor params: {len(t_anchor_params)}  lr={t_anchor_lr:.0e} (base={args.lr:.0e})")

            blurry_lr = getattr(args, 'blurry_lr', None)
            if blurry_lr is not None and blurry_lr > 0:
                blurry_pids = set()
                for p in list(self.model.blurry_predictor_fwd.parameters()) + list(self.model.blurry_predictor_bwd.parameters()):
                    if id(p) in t_anchor_pids:
                        continue
                    blurry_pids.add(id(p))
                for group in self.optimizer.param_groups:
                    group['params'] = [p for p in group['params'] if id(p) not in blurry_pids]
                blurry_params = [p for p in list(self.model.blurry_predictor_fwd.parameters()) + list(self.model.blurry_predictor_bwd.parameters())
                                 if id(p) not in t_anchor_pids]
                if blurry_params:
                    self.optimizer.add_param_group({
                        'params': blurry_params,
                        'lr': blurry_lr,
                        'betas': (0.9, 0.999),
                    })

            warp_w = args.warp_w if args.warp_w is not None else 0.5

            fusion_w = getattr(args, 'fusion_w', 1.0)
            wsm_w = getattr(args, 'wsm_w', None)
            delta_scale = getattr(args, 'delta_scale', 10.0)
            delta_range = getattr(args, 'delta_range', 0.15)

            self.phase_trainer = Phase2Trainer(
                self.model, self.optimizer, self.device,
                warp_w=warp_w,
                mask_kernel_size=args.mask_kernel_size, mask_sigma=args.mask_sigma,
                use_amp=self.use_amp,
                fusion_w=fusion_w, wsm_w=wsm_w,
                delta_scale=delta_scale, delta_range=delta_range,
                lpips_fn=self.lpips_fn, lpips_weight=getattr(args, 'lpips_w', 0.02),
                use_deblur=True,
                blurry_lr=blurry_lr,
                log_interval=args.log_interval,
                use_parallel=True,
                blurry_sparsity_weight=getattr(args, 'blurry_sparsity_w', 0.0))

            self.phase_epochs = args.phase2_epochs

    def _load_checkpoint(self):
        args = self.args
        if not args.resume:
            return

        ckpt_dir = self.ckpt_dir
        if args.resume_path:
            ckpt_path = Path(args.resume_path)
            self.log.info(f"resuming from {ckpt_path}")
        else:
            ckpt_files = sorted(ckpt_dir.glob("ckpt_epoch*.pth"), key=_extract_epoch_num)
            if not ckpt_files:
                ckpt_files = sorted(ckpt_dir.glob("ckpt_phase*.pth"))
            if not ckpt_files:
                self.log.info("resume: no checkpoint found, starting from scratch")
                return
            ckpt_path = ckpt_files[-1]
            self.log.info(f"resuming from {ckpt_path.name}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        model_state = self.model.state_dict()
        for key in list(ckpt["model"].keys()):
            if key in model_state and ckpt["model"][key].shape != model_state[key].shape:
                old_w = ckpt["model"][key]
                new_w = torch.zeros_like(model_state[key])
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_w.shape, new_w.shape))
                new_w[slices] = old_w[slices]
                ckpt["model"][key] = new_w
                self.log.info(f"  weight migrated: {key} {list(old_w.shape)} -> {list(new_w.shape)}")

        ckpt_model = ckpt["model"]
        remapped = {}
        for k, v in ckpt_model.items():
            rk = k.replace('_orig_mod.', '')
            remapped[rk] = v
        stripped_count = len(remapped) - len([k for k in remapped if k in ckpt_model and ckpt_model[k] is remapped[k]])
        if stripped_count > 0:
            self.log.info(f"  stripped _orig_mod. from {stripped_count} ckpt keys")

        dh_keys = [k for k in remapped if k.startswith('deblur_head.')]
        if dh_keys:
            dh_targets = ['deblur_head_bwd', 'deblur_head_fwd']
            if self.phase >= 2:
                dh_targets.append('deblur_head_fusion')
            self.log.info(f"  copying deblur_head -> {dh_targets}")
            for dh_key in dh_keys:
                suffix = dh_key.split('.', 1)[1]
                for target in dh_targets:
                    if f'{target}.{suffix}' not in remapped:
                        remapped[f'{target}.{suffix}'] = remapped[dh_key].clone()
            for dh_key in dh_keys:
                del remapped[dh_key]

        self.model.load_state_dict(remapped, strict=False)

        model_keys = set(model_state.keys())
        ckpt_keys = set(remapped.keys())
        matched = model_keys & ckpt_keys
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        self.log.info(f"phase{self.phase} load: {len(matched)} matched, {len(missing)} missing, {len(unexpected)} unexpected")
        self.log.info(f"  ckpt epoch={ckpt.get('epoch','?')} phase={ckpt.get('phase','?')} best_metric={ckpt.get('best_metric','?')}")
        if missing:
            self.log.info(f"  missing keys: {sorted(missing)}")
        if unexpected:
            self.log.info(f"  unexpected keys: {sorted(unexpected)}")

        self.log.info(f"weight verification (model vs ckpt, sha256[:8]):")
        for mpath, label in [('structure_encoder', 'encoder'),
                            ('offset_head_fwd', 'offset_fwd'),
                            ('offset_head_bwd', 'offset_bwd'),
                            ('deblur_head_bwd', 'deblur_bwd'),
                            ('deblur_head_fwd', 'deblur_fwd')]:
            ms, cs = _matched_checksums(self.model, remapped, mpath)
            ok = 'OK' if ms == cs else 'MISMATCH'
            self.log.info(f"  {label}: model={ms} ckpt={cs} [{ok}]")

        ckpt_phase = ckpt.get("phase", self.phase)
        if ckpt_phase == self.phase:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                self.log.info("optimizer param groups mismatch, resetting optimizer state")
                self.log.info("resume optimizer skipped (param group mismatch)")
            else:
                _reset_count = 0
                for _pg in self.optimizer.param_groups:
                    for _p in _pg['params']:
                        _st = self.optimizer.state.get(_p)
                        if _st is None:
                            continue
                        _mismatch = False
                        for _sk in ['exp_avg', 'exp_avg_sq']:
                            if _sk in _st and _st[_sk].shape != _p.shape:
                                _mismatch = True
                                break
                        if _mismatch:
                            del self.optimizer.state[_p]
                            _reset_count += 1
                if _reset_count > 0:
                    self.log.info(f"reset optimizer state for {_reset_count} shape-mismatched params")

                self.log.info("resume optimizer (same phase)")
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_metric = ckpt.get("best_metric", -1.0)
            self.best_epoch = ckpt.get("best_epoch", 0) if self.best_metric > 0 else 0
            if self.best_metric <= 0:
                best_file = self.ckpt_dir / f"ckpt_phase{self.phase}_best.pth"
                if best_file.exists():
                    best_ckpt = torch.load(best_file, map_location=self.device, weights_only=False)
                    self.best_metric = best_ckpt.get("best_metric", -1.0)
                    self.best_epoch = best_ckpt.get("best_epoch", 0)
                    self.log.info(f"restored best: epoch={self.best_epoch}  metric={self.best_metric:.4f}")
        else:
            self.log.info("cross-phase resume: optimizer reset")
            self.start_epoch = 1

            if self.phase == 2:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
                self._setup_phase_trainer()
                self.log.info("Phase 2: optimizer rebuilt with new components")

        self.resume_step = ckpt.get("step", 0)
        self._se_opt_state = ckpt.get("struct_encoder_optimizer", None)

    def _save_checkpoint(self, epoch, step, is_phase_end=False):
        ckpt = {
            "phase": self.phase,
            "epoch": epoch,
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if hasattr(self, 'phase_trainer') and hasattr(self.phase_trainer, 'struct_encoder_optimizer'):
            se_opt = self.phase_trainer.struct_encoder_optimizer
            if se_opt is not None:
                ckpt["struct_encoder_optimizer"] = se_opt.state_dict()
        if is_phase_end:
            ckpt_path = self.ckpt_dir / f"ckpt_phase{self.phase}.pth"
        else:
            ckpt_path = self.ckpt_dir / f"ckpt_epoch{epoch}.pth"
        torch.save(ckpt, ckpt_path)
        return ckpt_path

    def run(self):
        if not self.setup():
            return

        self._print_phase_header()
        step = self.resume_step

        _interrupted = [False]
        def _on_interrupt(_signum, _frame):
            if _interrupted[0]:
                return
            _interrupted[0] = True
            self.log.info("Interrupted, cleaning up...")
            self.csv_file.close()
            torch.cuda.empty_cache()
            self.log.close()
            sys.exit(1)
        signal.signal(signal.SIGINT, _on_interrupt)
        signal.signal(signal.SIGTERM, _on_interrupt)

        for epoch in range(self.start_epoch, self.phase_epochs + 1):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_warp = 0.0
            epoch_wsm = 0.0
            epoch_offset_mag = 0.0
            epoch_mem = 0.0
            epoch_recon_fusion = 0.0
            epoch_lpips = 0.0
            epoch_bwd_loss = 0.0
            epoch_fwd_loss = 0.0
            epoch_fusion_loss = 0.0
            epoch_count = 0

            reset_cascade_diag()
            reset_fusion_diag()

            if epoch <= 2 or epoch % self.args.log_interval == 0:
                _w_last = self.model.offset_head_fwd.conv4.weight
                _b_last = self.model.offset_head_fwd.conv4.bias
                _w_enc = self.model.encoder_body[0].weight
                _w_pol = self.model.polarity_proj[0].weight
                print(f"[DIAG epoch={epoch} pre] offset_head conv4: "
                      f"w_mean={_w_last.mean().item():.6f} w_std={_w_last.std().item():.6f} "
                      f"b_mean={_b_last.mean().item():.6f}")
                print(f"[DIAG epoch={epoch} pre] encoder_body[0]: "
                      f"w_mean={_w_enc.mean().item():.6f} w_std={_w_enc.std().item():.6f}")
                print(f"[DIAG epoch={epoch} pre] polarity_proj[0]: "
                      f"w_mean={_w_pol.mean().item():.6f} w_std={_w_pol.std().item():.6f}")
                if self.phase >= 1:
                    _w_se = self.model.structure_encoder.edge[0].weight
                    _w_sm = self.model.structure_encoder.semantic[0].weight
                    print(f"[DIAG epoch={epoch} pre] struct_encoder.edge[0]: "
                          f"w_mean={_w_se.mean().item():.6f} w_std={_w_se.std().item():.6f}")
                    print(f"[DIAG epoch={epoch} pre] struct_encoder.semantic[0]: "
                          f"w_mean={_w_sm.mean().item():.6f} w_std={_w_sm.std().item():.6f}")
                    if self.phase >= 2:
                        _w_deblur_bwd = self.model.deblur_head_bwd.net[0].weight
                        _w_deblur_fwd = self.model.deblur_head_fwd.net[0].weight
                        print(f"[DIAG epoch={epoch} pre] deblur_head_bwd.net[0]: "
                              f"w_mean={_w_deblur_bwd.mean().item():.6f} w_std={_w_deblur_bwd.std().item():.6f}")
                        print(f"[DIAG epoch={epoch} pre] deblur_head_fwd.net[0]: "
                              f"w_mean={_w_deblur_fwd.mean().item():.6f} w_std={_w_deblur_fwd.std().item():.6f}")
                        _w_bp_fwd = self.model.blurry_predictor_fwd.net[0].weight
                        print(f"[DIAG epoch={epoch} pre] blurry_predictor_fwd.net[0]: "
                              f"w_mean={_w_bp_fwd.mean().item():.6f} w_std={_w_bp_fwd.std().item():.6f}")
                        _w_bp_bwd = self.model.blurry_predictor_bwd.net[0].weight
                        print(f"[DIAG epoch={epoch} pre] blurry_predictor_bwd.net[0]: "
                              f"w_mean={_w_bp_bwd.mean().item():.6f} w_std={_w_bp_bwd.std().item():.6f}")

            random.shuffle(self.all_pairs)

            for pair_idx, (sid, prev_fid) in enumerate(self.all_pairs):
                step += 1
                next_fid = prev_fid + self.args.key_interval
                events = self.train_events[sid]
                gt_dir = self.args.gt_root / "train" / sid

                event_slices = extract_temporal_slices(
                    events, prev_fid, next_fid, self.event_h, self.event_w,
                    self.num_slices, self.event_downsample,
                )
                events_t = torch.from_numpy(event_slices).float().unsqueeze(0).to(self.device)

                gt_frames_cpu = []
                for fid in range(prev_fid, next_fid + 1):
                    gt_np = load_gt_frame(gt_dir, fid, self.target_h, self.target_w)
                    gt_t = torch.from_numpy(gt_np).float().permute(2, 0, 1).unsqueeze(0)
                    gt_frames_cpu.append(gt_t)

                gt_frames_gpu = [f.to(self.device) for f in gt_frames_cpu]
                prev_kf = gt_frames_gpu[0]
                next_kf = gt_frames_gpu[-1]

                B, _, H, W = prev_kf.shape

                loss_val, acc = self.phase_trainer.train_step(
                    events_t, gt_frames_gpu, prev_kf, next_kf,
                    H, W, self.n_frames,
                    epoch=epoch,
                    step_id=step,
                )

                recon = acc.get('recon', 0)
                if self.phase == 1:
                    warp = acc.get('warp', 0)
                else:
                    warp = acc.get('recon', 0)
                wsm = acc.get('weight_smooth', 0)
                offset_mag = acc.get('offset_mag', 0)
                mem = acc.get('mem', 0)
                recon_fusion = acc.get('recon_fusion', 0)
                lpips = acc.get('lpips', 0)
                bwd_loss = acc.get('bwd_loss', 0)
                fwd_loss = acc.get('fwd_loss', 0)
                fusion_loss = acc.get('fusion_loss', 0)

                epoch_loss += loss_val
                epoch_recon += recon
                epoch_warp += warp
                epoch_wsm += wsm
                epoch_offset_mag += offset_mag
                epoch_mem += mem
                epoch_recon_fusion += recon_fusion
                epoch_lpips += lpips
                epoch_bwd_loss += bwd_loss
                epoch_fwd_loss += fwd_loss
                epoch_fusion_loss += fusion_loss
                epoch_count += 1

                if (pair_idx + 1) % self.args.log_interval == 0:
                    avg_loss = epoch_loss / epoch_count
                    avg_recon = epoch_recon / epoch_count
                    avg_warp = epoch_warp / epoch_count
                    avg_wsm = epoch_wsm / epoch_count
                    avg_offset_mag = epoch_offset_mag / epoch_count
                    avg_mem = epoch_mem / epoch_count
                    pct = 100.0 * (pair_idx + 1) / self.total_pairs
                    if self.phase == 1:
                        base = f"epoch={epoch:2d}/{self.phase_epochs} step={pair_idx + 1:5d}/{self.total_pairs} ({pct:3.0f}%) | loss={avg_loss:.4f}  warp={avg_warp:.4f}  offset={avg_offset_mag:.4f}  mem={avg_mem:.3f}"
                    else:
                        base = f"epoch={epoch:2d}/{self.phase_epochs} step={pair_idx + 1:5d}/{self.total_pairs} ({pct:3.0f}%) | loss={avg_loss:.4f}  bwd={epoch_bwd_loss/epoch_count:.4f}  fwd={epoch_fwd_loss/epoch_count:.4f}  fus={epoch_fusion_loss/epoch_count:.4f}  recon={avg_recon:.4f}  offset={avg_offset_mag:.4f}  lpips={epoch_lpips/epoch_count:.5f}  fus_recon={epoch_recon_fusion/epoch_count:.4f}  wsm={avg_wsm:.4f}"
                    self.log.info(base)

            # end of epoch
            ckpt_path = self._save_checkpoint(epoch, step)
            self.log.info(f"ckpt saved: {ckpt_path.name}")

            if epoch <= 2 or epoch % self.args.log_interval == 0:
                _w_last = self.model.offset_head_fwd.conv4.weight
                _b_last = self.model.offset_head_fwd.conv4.bias
                _w_enc = self.model.encoder_body[0].weight
                _w_pol = self.model.polarity_proj[0].weight
                print(f"[DIAG epoch={epoch} post] offset_head conv4: "
                      f"w_mean={_w_last.mean().item():.6f} w_std={_w_last.std().item():.6f} "
                      f"b_mean={_b_last.mean().item():.6f}")
                print(f"[DIAG epoch={epoch} post] encoder_body[0]: "
                      f"w_mean={_w_enc.mean().item():.6f} w_std={_w_enc.std().item():.6f}")
                print(f"[DIAG epoch={epoch} post] polarity_proj[0]: "
                      f"w_mean={_w_pol.mean().item():.6f} w_std={_w_pol.std().item():.6f}")
                if self.phase >= 2:
                    _w_deblur_bwd = self.model.deblur_head_bwd.net[0].weight
                    _w_deblur_fwd = self.model.deblur_head_fwd.net[0].weight
                    print(f"[DIAG epoch={epoch} post] deblur_head_bwd.net[0]: "
                          f"w_mean={_w_deblur_bwd.mean().item():.6f} w_std={_w_deblur_bwd.std().item():.6f}")
                    print(f"[DIAG epoch={epoch} post] deblur_head_fwd.net[0]: "
                          f"w_mean={_w_deblur_fwd.mean().item():.6f} w_std={_w_deblur_fwd.std().item():.6f}")
                    _w_bp_fwd = self.model.blurry_predictor_fwd.net[0].weight
                    print(f"[DIAG epoch={epoch} post] blurry_predictor_fwd.net[0]: "
                          f"w_mean={_w_bp_fwd.mean().item():.6f} w_std={_w_bp_fwd.std().item():.6f}")
                    _w_bp_bwd = self.model.blurry_predictor_bwd.net[0].weight
                    print(f"[DIAG epoch={epoch} post] blurry_predictor_bwd.net[0]: "
                          f"w_mean={_w_bp_bwd.mean().item():.6f} w_std={_w_bp_bwd.std().item():.6f}")

            if self.val_seqs:
                avg_loss = epoch_loss / max(epoch_count, 1)
                avg_recon = epoch_recon / max(epoch_count, 1)
                avg_warp = epoch_warp / max(epoch_count, 1)
                avg_wsm = epoch_wsm / max(epoch_count, 1)
                avg_offset_mag = epoch_offset_mag / max(epoch_count, 1)
                avg_mem = epoch_mem / max(epoch_count, 1)
                avg_recon_fusion = epoch_recon_fusion / max(epoch_count, 1)
                avg_lpips = epoch_lpips / max(epoch_count, 1)

                val_metrics = run_validation(
                    self.model, self.val_seqs, self.val_events, self.val_frames, self.val_ranges,
                    self.args.gt_root, self.device, self.event_h, self.event_w, self.target_h, self.target_w,
                    self.event_downsample, self.num_slices, self.args.key_interval,
                    self.lpips_fn, phase=self.phase,
                    mask_kernel_size=self.args.mask_kernel_size, mask_sigma=self.args.mask_sigma,
                    delta_scale=self.args.delta_scale,
                    delta_range=getattr(self.args, 'delta_range', None) or 0.15,
                use_deblur=True,
                )

                self._log_epoch_summary(epoch, avg_loss, avg_recon, avg_warp, avg_wsm, avg_offset_mag, avg_mem, avg_recon_fusion, avg_lpips, val_metrics)
                self._write_metrics_csv(epoch, avg_loss, avg_recon, avg_warp, avg_wsm, avg_offset_mag, val_metrics)

                if self.phase == 2:
                    cur_metric = val_metrics['psnr_fusion']
                else:
                    cur_metric = (val_metrics['psnr_dynamic'] + val_metrics['psnr_bwd_dynamic']) / 2
                if cur_metric > self.best_metric:
                    self.best_metric = cur_metric
                    self.best_epoch = epoch
                    best_path = self.ckpt_dir / f"ckpt_phase{self.phase}_best.pth"
                    ckpt = {
                        "phase": self.phase,
                        "epoch": epoch,
                        "step": step,
                        "best_metric": self.best_metric,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }
                    if hasattr(self, 'phase_trainer') and hasattr(self.phase_trainer, 'struct_encoder_optimizer'):
                        se_opt = self.phase_trainer.struct_encoder_optimizer
                        if se_opt is not None:
                            ckpt["struct_encoder_optimizer"] = se_opt.state_dict()
                    torch.save(ckpt, best_path)
                    self.log.info(f"best ckpt: epoch={epoch}  metric={self.best_metric:.4f}")

            if (self.preview_seq or self.val_seqs) and epoch % self.preview_interval == 0:
                self._generate_preview(epoch)

        # Phase end
        if self.best_epoch > 0:
            phase_path = self.ckpt_dir / f"ckpt_phase{self.phase}.pth"
            best_path = self.ckpt_dir / f"ckpt_phase{self.phase}_best.pth"
            if best_path.exists():
                import shutil
                shutil.copy(best_path, phase_path)
                self.log.info(f"phase end: best epoch={self.best_epoch} (metric={self.best_metric:.4f}) saved as ckpt_phase{self.phase}.pth")
            else:
                self._save_checkpoint(self.phase_epochs, step, is_phase_end=True)
        else:
            self._save_checkpoint(self.phase_epochs, step, is_phase_end=True)
        self._print_phase_end()
        self.csv_file.close()
        torch.cuda.empty_cache()
        self.log.close()

    def _generate_preview(self, epoch):
        diag_seq = self.preview_seq if self.preview_seq else self.val_seqs[0]
        preview_base = self.phase_dir / "preview"
        preview_base.mkdir(parents=True, exist_ok=True)
        preview_dir = preview_base / f"epoch_{epoch:03d}"
        preview_dir.mkdir(parents=True, exist_ok=True)

        generate_all_previews(
            self.model, diag_seq, self.val_events[diag_seq["seq_id"]],
            self.val_frames[diag_seq["seq_id"]], self.args.gt_root, self.device,
            self.event_h, self.event_w, self.target_h, self.target_w, self.event_downsample,
            self.num_slices, self.args.key_interval, preview_dir, epoch,
            self.args.mask_kernel_size, self.args.mask_sigma,
            phase=self.phase,
            delta_scale=getattr(self.args, 'delta_scale', None) or 10.0,
            delta_range=getattr(self.args, 'delta_range', None) or 0.15)

        self.log.info(f"preview saved: {preview_dir}")

    def _log_epoch_summary(self, epoch, avg_loss, avg_recon, avg_warp, avg_wsm, avg_offset_mag, avg_mem, avg_recon_fusion, avg_lpips, val_metrics):
        self.log.info(f"===== epoch {epoch}/{self.phase_epochs} done =====")
        if self.phase == 1:
            self.log.info(
                f"  train:   loss={avg_loss:.4f}  warp={avg_warp:.4f}  offset={avg_offset_mag:.4f}  "
                f"mem={avg_mem:.3f}"
            )
        else:
            self.log.info(
                f"  train:   loss={avg_loss:.4f}  recon={avg_recon:.4f}  "
                f"offset={avg_offset_mag:.4f}  "
                f"lpips={avg_lpips:.5f}")

        if self.phase == 2:
            self.log.info("  ===== Cascade =====")
            self.log.info(
                f"  PSNR fwd: total={val_metrics['psnr_total']:.2f}  static={val_metrics['psnr_static']:.2f}  dynamic={val_metrics['psnr_dynamic']:.2f}"
            )
            self.log.info(
                f"  PSNR bwd: total={val_metrics['psnr_bwd']:.2f}  static={val_metrics['psnr_bwd_static']:.2f}  dynamic={val_metrics['psnr_bwd_dynamic']:.2f}"
            )
            self.log.info(
                f"  SSIM fwd: total={val_metrics['ssim_fwd']:.4f}  static={val_metrics['ssim_fwd_static']:.4f}  dynamic={val_metrics['ssim_fwd_dynamic']:.4f}"
            )
            self.log.info(
                f"  SSIM bwd: total={val_metrics['ssim_bwd']:.4f}  static={val_metrics['ssim_bwd_static']:.4f}  dynamic={val_metrics['ssim_bwd_dynamic']:.4f}"
            )
            if self.lpips_fn is not None:
                self.log.info(f"  LPIPS Cascade: fwd={val_metrics['lpips']:.4f}  bwd={val_metrics['lpips_bwd']:.4f}")

            self.log.info("  ===== Fusion =====")
            self.log.info(
                f"  PSNR:  total={val_metrics['psnr_fusion']:.2f}  "
                f"static={val_metrics['psnr_fusion_static']:.2f}  dynamic={val_metrics['psnr_fusion_dynamic']:.2f}"
            )
            self.log.info(
                f"  SSIM:  total={val_metrics['ssim_fusion']:.4f}  static={val_metrics['ssim_fusion_static']:.4f}  dynamic={val_metrics['ssim_fusion_dynamic']:.4f}"
            )
            if self.lpips_fn is not None:
                self.log.info(f"  LPIPS Fusion: {val_metrics['lpips_fusion']:.4f}")

            self.log.info(
                f"  weights: w0={val_metrics['w0']:.2f}  w1={val_metrics['w1']:.2f}  "
                f"w2={val_metrics['w2']:.2f}  w3={val_metrics['w3']:.2f}  delta={val_metrics['delta_mean']:.4f}"
            )
            self.log.info(f"  loss: recon={avg_recon_fusion:.4f}  wsm={avg_wsm:.4f}  lpips={avg_lpips:.5f}")
            self.log.info(
                f"  gap: cascade_fwd={val_metrics['cascade_gap']:.4f}  "
                f"cascade_bwd={val_metrics['cascade_gap_bwd']:.4f}  fusion={val_metrics['fusion_l1']:.4f}  "
                f"gt_warp_fwd={val_metrics['gt_warp_l1']:.4f}  gt_warp_bwd={val_metrics['gt_warp_l1_bwd']:.4f}"
            )
            self.log.info(f"  speed: {val_metrics['per_frame_ms']:.1f}ms/frame")
        else:
            self.log.info(
                f"  PSNR fwd: total={val_metrics['psnr_total']:.2f}  "
                f"static={val_metrics['psnr_static']:.2f}  dynamic={val_metrics['psnr_dynamic']:.2f}"
            )
            self.log.info(
                f"  PSNR bwd: total={val_metrics['psnr_bwd']:.2f}  "
                f"static={val_metrics['psnr_bwd_static']:.2f}  dynamic={val_metrics['psnr_bwd_dynamic']:.2f}"
            )
            ssim_str = (f"  SSIM fwd: total={val_metrics['ssim_fwd']:.4f}  "
                        f"static={val_metrics['ssim_fwd_static']:.4f}  dynamic={val_metrics['ssim_fwd_dynamic']:.4f}")
            self.log.info(ssim_str)
            ssim_bwd_str = (f"  SSIM bwd: total={val_metrics['ssim_bwd']:.4f}  "
                            f"static={val_metrics['ssim_bwd_static']:.4f}  dynamic={val_metrics['ssim_bwd_dynamic']:.4f}")
            self.log.info(ssim_bwd_str)
            if self.lpips_fn is not None:
                self.log.info(f"  LPIPS: fwd={val_metrics['lpips']:.4f}  bwd={val_metrics['lpips_bwd']:.4f}")
            self.log.info(
                f"  warp L1: fwd={val_metrics['gt_warp_l1']:.4f}  bwd={val_metrics['gt_warp_l1_bwd']:.4f}  |  "
                f"speed: {val_metrics['per_frame_ms']:.1f}ms/frame"
            )

    def _write_metrics_csv(self, epoch, avg_loss, avg_recon, avg_warp, avg_wsm, avg_offset_mag, val_metrics):
        if self.phase == 1:
            row = [
                epoch, avg_loss,
                avg_warp, avg_offset_mag,
                val_metrics['psnr_total'], val_metrics['psnr_static'], val_metrics['psnr_dynamic'],
                val_metrics['psnr_bwd'], val_metrics['psnr_bwd_static'], val_metrics['psnr_bwd_dynamic'],
                val_metrics['ssim_fwd'], val_metrics['ssim_fwd_static'], val_metrics['ssim_fwd_dynamic'],
                val_metrics['ssim_bwd'], val_metrics['ssim_bwd_static'], val_metrics['ssim_bwd_dynamic'],
                val_metrics['lpips'], val_metrics['lpips_bwd'],
                val_metrics['gt_warp_l1'], val_metrics['gt_warp_l1_bwd'],
                val_metrics['per_frame_ms'],
            ]
        else:
            row = [
                epoch, avg_loss,
                avg_recon, avg_wsm,
                avg_warp, avg_offset_mag,
                val_metrics['psnr_total'], val_metrics['psnr_static'], val_metrics['psnr_dynamic'],
                val_metrics['psnr_bwd'], val_metrics['psnr_bwd_static'], val_metrics['psnr_bwd_dynamic'],
                val_metrics['ssim_fwd'], val_metrics['ssim_fwd_static'], val_metrics['ssim_fwd_dynamic'],
                val_metrics['ssim_bwd'], val_metrics['ssim_bwd_static'], val_metrics['ssim_bwd_dynamic'],
            ]
            if self.lpips_fn is not None:
                row += [val_metrics['lpips'], val_metrics['lpips_bwd']]
            row += [
                val_metrics['w0'], val_metrics['w1'], val_metrics['w2'], val_metrics['w3'], val_metrics['delta_mean'],
                val_metrics['gt_warp_l1'], val_metrics['gt_warp_l1_bwd'], val_metrics['cascade_gap'], val_metrics['cascade_gap_bwd'],
                val_metrics['psnr_fusion'], val_metrics['psnr_fusion_static'], val_metrics['psnr_fusion_dynamic'],
                val_metrics['ssim_fusion'], val_metrics['ssim_fusion_static'], val_metrics['ssim_fusion_dynamic'],
                val_metrics['fusion_l1'],
            ]
            if self.lpips_fn is not None:
                row.append(val_metrics['lpips_fusion'])
            row.append(val_metrics['per_frame_ms'])
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def _print_phase_header(self):
        phase_names = {1: "Phase 1: Offset Warmup", 2: "Phase 2: Cascade + Fusion Joint Training"}
        name = phase_names.get(self.phase, f"Phase {self.phase}")
        self.log.info(f"==================== {name} ====================")
        if self.phase == 1:
            warp_w = self.args.warp_w if self.args.warp_w is not None else 1.0
            self.log.info(f" epochs={self.phase_epochs}  lr={self.args.lr}  warp_w={warp_w}  direction=alternate")
        else:
            warp_w = self.args.warp_w if self.args.warp_w is not None else 0.5
            base = f" epochs={self.phase_epochs}  lr={self.args.lr}"
            base += f"  fusion_w={self.phase_trainer.fusion_w}  wsm_w={self.phase_trainer.wsm_w}  delta_scale={self.phase_trainer.delta_scale}"
            base += f"  lpips_w={self.phase_trainer.lpips_weight}"
            if getattr(self.phase_trainer, 'blurry_lr', None):
                base += f"  blurry_lr={self.phase_trainer.blurry_lr}"
            self.log.info(base)
        self.log.info("=" * 60)

    def _print_phase_end(self):
        phase_names = {1: "PHASE 1 DONE", 2: "PHASE 2 DONE"}
        name = phase_names.get(self.phase, f"PHASE {self.phase} DONE")
        self.log.info(f"==================== {name} ====================")
        ckpt_name = f"ckpt_phase{self.phase}.pth"
        self.log.info(f" ckpt saved: {self.ckpt_dir / ckpt_name}")
        self.log.info("=" * 60)